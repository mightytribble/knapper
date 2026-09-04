//! Custom properties (#66): a named value on a note, read from frontmatter
//! under Obsidian's rules and from Dataview inline fields under Dataview's.
//!
//! This module owns the kind vocabulary and the two extractors. Nothing else
//! in the crate reads a property out of a note.
//!
//! # Reading rows back
//!
//! A property read can fail the way any store read can. One policy across
//! the crate: propagate it where the signature carries an error, and log it
//! where the signature cannot. A caller that filtered on a property asked
//! for those rows, so answering an empty list in place of a failure would
//! report the vault holds nothing when the read is what went wrong.
//! `context_read` and `context_list` return `Result` and propagate;
//! `search::finalize_search_output` returns a `SearchOutput` and cannot, so
//! it warns and leaves the field empty.

/// The kind of a property value, read from the value's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Text,
    Number,
    Checkbox,
    Link,
    Empty,
}

impl Kind {
    /// The name the store writes into `properties.kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Number => "number",
            Kind::Checkbox => "checkbox",
            Kind::Link => "link",
            Kind::Empty => "empty",
        }
    }

    /// The inverse of [`Kind::as_str`]. `None` for a name this build does
    /// not know.
    pub fn parse(name: &str) -> Option<Kind> {
        match name {
            "text" => Some(Kind::Text),
            "number" => Some(Kind::Number),
            "checkbox" => Some(Kind::Checkbox),
            "link" => Some(Kind::Link),
            "empty" => Some(Kind::Empty),
            _ => None,
        }
    }
}

use crate::store::DOC_LEVEL;

/// The three properties Obsidian provides. They keep their own handling
/// and write no property row.
pub const BUILT_IN: [&str; 3] = ["tags", "aliases", "cssclasses"];

/// One property value as an extractor found it, before its link target is
/// resolved against the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    /// [`DOC_LEVEL`] for a frontmatter row, else the chunk's `seq`.
    pub chunk_seq: i64,
    pub name: String,
    /// The value as text. For a link, the target as written with its
    /// `#Heading` and `|Display` dropped.
    pub value: String,
    pub kind: Kind,
    /// The wikilink target of a `link` row, for the indexer to resolve.
    pub link_target: Option<String>,
}

/// The shape rule: what kind a value is, from its text alone.
///
/// A value that is one wikilink and nothing else is a link. `true` and
/// `false` are a checkbox. Text that parses as a finite number and opens
/// with a digit, a sign or a point is a number, so `inf` and `nan` stay
/// text. A blank value is empty. Everything else is text. Dates are text:
/// Obsidian's Date type is a declared type, not a value shape.
pub fn kind_of(value: &str) -> (Kind, Option<String>) {
    let v = value.trim();
    if v.is_empty() {
        return (Kind::Empty, None);
    }
    if let Some(target) = sole_wikilink(v) {
        return (Kind::Link, Some(target));
    }
    if v == "true" || v == "false" {
        return (Kind::Checkbox, None);
    }
    if is_number(v) {
        return (Kind::Number, None);
    }
    (Kind::Text, None)
}

fn is_number(v: &str) -> bool {
    let opens_like_one = v
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit() || c == '-' || c == '+' || c == '.');
    opens_like_one && v.parse::<f64>().is_ok_and(f64::is_finite)
}

/// The target of `v` when `v` is one wikilink and nothing else.
fn sole_wikilink(v: &str) -> Option<String> {
    let inner = v.strip_prefix("[[")?.strip_suffix("]]")?;
    if inner.contains("]]") || inner.contains("[[") {
        return None;
    }
    let links = crate::graph::extract_wikilinks(v);
    match links.as_slice() {
        [link] => Some(link.target.clone()),
        _ => None,
    }
}

/// Every custom property in a frontmatter block, under Obsidian's rules.
///
/// `block` is the text between the `---` fences, as
/// `markdown::split_frontmatter` returns it. A block that does not parse
/// writes no rows; `validate` reports it as `MalformedFrontmatter`.
///
/// A scalar is one value. A sequence is one value per scalar element. A
/// nested mapping, a tagged value and a key that is not a string are
/// skipped. A string that is one wikilink is a link; an unquoted
/// `[[Target]]` reads as a sequence inside a sequence and writes nothing,
/// because Obsidian reads no link from it either. A YAML string keeps its
/// type: `"5"` is text where bare `5` is a number.
pub fn from_frontmatter(block: &str) -> Vec<Extracted> {
    let Ok(serde_yaml::Value::Mapping(map)) = serde_yaml::from_str::<serde_yaml::Value>(block)
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (key, value) in map {
        let Some(name) = key.as_str() else { continue };
        let name = name.trim();
        if name.is_empty() || BUILT_IN.contains(&name) {
            continue;
        }
        match value {
            serde_yaml::Value::Sequence(items) => {
                for item in &items {
                    push_yaml_scalar(&mut out, name, item);
                }
            }
            other => push_yaml_scalar(&mut out, name, &other),
        }
    }
    out
}

fn push_yaml_scalar(out: &mut Vec<Extracted>, name: &str, v: &serde_yaml::Value) {
    let (value, kind, link_target) = match v {
        serde_yaml::Value::Null => (String::new(), Kind::Empty, None),
        serde_yaml::Value::Bool(b) => (b.to_string(), Kind::Checkbox, None),
        serde_yaml::Value::Number(n) => (n.to_string(), Kind::Number, None),
        serde_yaml::Value::String(s) => match sole_wikilink(s.trim()) {
            Some(target) => (target.clone(), Kind::Link, Some(target)),
            None if s.trim().is_empty() => (String::new(), Kind::Empty, None),
            None => (s.trim().to_string(), Kind::Text, None),
        },
        // A nested mapping, a sequence inside a sequence, a tagged value.
        _ => return,
    };
    out.push(Extracted {
        chunk_seq: DOC_LEVEL,
        name: name.to_string(),
        value,
        kind,
        link_target,
    });
}

/// Every Dataview inline field in one chunk's text, under Dataview's rules.
///
/// Three forms: the full-line `Key:: value`, with an optional list-item or
/// task prefix; and the embedded `[key:: value]` and `(key:: value)`,
/// any number per line. The key is everything before the `::`, trimmed,
/// and may contain spaces. A line inside a fenced code block is skipped.
///
/// A value with a top-level comma is a list, one row per element. An
/// element that holds wikilinks writes one link row per link and the text
/// around them writes nothing. An element with no wikilink is one row
/// under [`kind_of`].
pub fn from_chunk(seq: i64, text: &str) -> Vec<Extracted> {
    let mut out = Vec::new();
    let mut fenced = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        for (key, value) in inline_fields(line) {
            push_inline(&mut out, seq, &key, &value);
        }
    }
    out
}

/// The `(key, value)` pairs one line holds.
fn inline_fields(line: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let body = strip_line_prefix(line.trim_start());
    if !body.starts_with('[')
        && !body.starts_with('(')
        && let Some(pair) = split_field(body)
    {
        out.push(pair);
        return out;
    }
    for (open, close) in [('[', ']'), ('(', ')')] {
        let mut rest = line;
        while let Some(start) = rest.find(open) {
            let after = &rest[start + 1..];
            let Some(end) = find_close(after, open, close) else {
                break;
            };
            if let Some(pair) = split_field(&after[..end]) {
                out.push(pair);
            }
            rest = &after[end + 1..];
        }
    }
    out
}

/// The index in `text` of the `close` that matches an `open` already
/// consumed, honouring nesting: a wikilink's own `[[Target]]` brackets
/// nest inside the bracket field form, so the bracket form's own close is
/// the one that brings the count back to zero rather than the first one
/// found. `None` on an unclosed `open`.
fn find_close(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1i32;
    for (i, c) in text.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// `Key:: value` split on the first `::`. A key that is empty or holds a
/// bracket, a parenthesis or a backtick is not one: a wikilink or an
/// embedded field is not a full-line key.
fn split_field(text: &str) -> Option<(String, String)> {
    let (key, value) = text.split_once("::")?;
    let key = key.trim();
    if key.is_empty() || key.contains(['[', ']', '(', ')', '`']) {
        return None;
    }
    Some((key.to_string(), value.trim().to_string()))
}

/// Drop a blockquote marker, a list-item marker and a task box from the
/// head of a line, in that order.
///
/// A callout body is `> Key:: value`, and `>` is not a character
/// `split_field` rejects, so the marker would otherwise be filed as part of
/// the name. The space after `>` is optional and the quote may nest (#66).
fn strip_line_prefix(line: &str) -> &str {
    let mut quoted = line;
    while let Some(rest) = quoted.strip_prefix('>') {
        quoted = rest.trim_start();
    }
    let after_marker = ["- ", "* ", "+ "]
        .iter()
        .find_map(|m| quoted.strip_prefix(m))
        .unwrap_or(quoted)
        .trim_start();
    ["[ ] ", "[x] ", "[X] "]
        .iter()
        .find_map(|m| after_marker.strip_prefix(m))
        .unwrap_or(after_marker)
        .trim_start()
}

/// Split on commas outside `[...]`, so a note name with a comma stays whole.
fn split_list(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in value.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth <= 0 => {
                out.push(current.trim().to_string());
                current.clear();
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    out.push(current.trim().to_string());
    out
}

fn push_inline(out: &mut Vec<Extracted>, seq: i64, name: &str, value: &str) {
    if BUILT_IN.contains(&name) {
        return;
    }
    for element in split_list(value) {
        let links = crate::graph::extract_wikilinks(&element);
        if links.is_empty() {
            let (kind, _) = kind_of(&element);
            out.push(Extracted {
                chunk_seq: seq,
                name: name.to_string(),
                value: element,
                kind,
                link_target: None,
            });
            continue;
        }
        for link in links {
            out.push(Extracted {
                chunk_seq: seq,
                name: name.to_string(),
                value: link.target.clone(),
                kind: Kind::Link,
                link_target: Some(link.target),
            });
        }
    }
}

use std::collections::BTreeMap;
use std::path::Path;

/// Obsidian's declared property types, from `<vault>/.obsidian/types.json`,
/// read when the call runs. Obsidian does not document the file, so it is a
/// hint and never a source of rows. Absent or unparseable answers empty.
pub fn declared_types(vault_path: &Path) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(vault_path.join(".obsidian/types.json")) else {
        return BTreeMap::new();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return BTreeMap::new();
    };
    json.get("types")
        .and_then(serde_json::Value::as_object)
        .map(|types| {
            types
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// The vault's property registry: one row per name, by note count and then
/// name, each with the kinds seen and Obsidian's declared type. A name
/// `types.json` declares that no note carries appears with a count of zero.
/// This mirrors Obsidian's "All properties" view.
pub fn registry(
    store: &crate::store::Store,
    vault_path: &Path,
) -> anyhow::Result<Vec<crate::store::PropertyCount>> {
    let declared = declared_types(vault_path);
    let mut rows = store.property_registry()?;
    for row in &mut rows {
        row.declared_type = declared.get(&row.name).cloned();
    }
    for (name, ty) in &declared {
        if BUILT_IN.contains(&name.as_str()) || rows.iter().any(|r| &r.name == name) {
            continue;
        }
        rows.push(crate::store::PropertyCount {
            name: name.clone(),
            note_count: 0,
            kinds: Vec::new(),
            declared_type: Some(ty.clone()),
        });
    }
    rows.sort_by(|a, b| {
        b.note_count
            .cmp(&a.note_count)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(rows)
}

/// What `properties` answers: the registry, or one name's values.
#[derive(Debug, serde::Serialize)]
#[serde(untagged)]
pub enum PropertiesReport {
    Registry(Vec<crate::store::PropertyCount>),
    Values(Vec<crate::store::ValueCount>),
}

/// The one path the three surfaces take (#66).
pub fn run(
    store: &crate::store::Store,
    vault_path: &Path,
    params: &crate::params::Properties,
) -> anyhow::Result<PropertiesReport> {
    Ok(match &params.name {
        Some(name) => PropertiesReport::Values(store.property_values(name)?),
        None => PropertiesReport::Registry(registry(store, vault_path)?),
    })
}

/// The CLI's text form: `name (count) kinds [declared]` per registry row,
/// `value (count) kind` per value row.
pub fn render_text(report: &PropertiesReport) -> String {
    let mut out = String::new();
    match report {
        PropertiesReport::Registry(rows) => {
            for r in rows {
                let kinds: Vec<&str> = r.kinds.iter().map(|k| k.as_str()).collect();
                out.push_str(&format!(
                    "{} ({}) {}",
                    r.name,
                    r.note_count,
                    kinds.join(",")
                ));
                if let Some(ty) = &r.declared_type {
                    out.push_str(&format!(" [{ty}]"));
                }
                out.push('\n');
            }
        }
        PropertiesReport::Values(rows) => {
            for r in rows {
                out.push_str(&format!(
                    "{} ({}) {}\n",
                    r.value,
                    r.note_count,
                    r.kind.as_str()
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_round_trips_through_its_name() {
        for kind in [
            Kind::Text,
            Kind::Number,
            Kind::Checkbox,
            Kind::Link,
            Kind::Empty,
        ] {
            assert_eq!(Kind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(Kind::parse("date"), None);
    }

    fn names_values(rows: &[Extracted]) -> Vec<(String, String, Kind)> {
        rows.iter()
            .map(|r| (r.name.clone(), r.value.clone(), r.kind))
            .collect()
    }

    #[test]
    fn the_shape_rule_reads_each_kind() {
        assert_eq!(kind_of("[[Acme]]"), (Kind::Link, Some("Acme".into())));
        assert_eq!(
            kind_of("[[Acme#History|the firm]]"),
            (Kind::Link, Some("Acme".into()))
        );
        assert_eq!(kind_of("true"), (Kind::Checkbox, None));
        assert_eq!(kind_of("false"), (Kind::Checkbox, None));
        assert_eq!(kind_of("5"), (Kind::Number, None));
        assert_eq!(kind_of("-3.25"), (Kind::Number, None));
        assert_eq!(kind_of(""), (Kind::Empty, None));
        assert_eq!(kind_of("   "), (Kind::Empty, None));
        assert_eq!(kind_of("2026-09-03"), (Kind::Text, None));
        assert_eq!(kind_of("inf"), (Kind::Text, None));
        assert_eq!(kind_of("see [[Acme]]"), (Kind::Text, None));
        assert_eq!(kind_of("[[Acme]] [[Bolt]]"), (Kind::Text, None));
    }

    #[test]
    fn frontmatter_reads_every_top_level_key_but_the_built_ins() {
        let block = "tags: [a]\naliases: [b]\ncssclasses: [c]\nstatus: draft\nrating: 5\ndone: true\nnothing:\n";
        let rows = from_frontmatter(block);
        assert_eq!(
            names_values(&rows),
            vec![
                ("status".into(), "draft".into(), Kind::Text),
                ("rating".into(), "5".into(), Kind::Number),
                ("done".into(), "true".into(), Kind::Checkbox),
                ("nothing".into(), "".into(), Kind::Empty),
            ]
        );
        assert!(rows.iter().all(|r| r.chunk_seq == crate::store::DOC_LEVEL));
    }

    #[test]
    fn a_quoted_yaml_scalar_is_text_and_a_bare_one_takes_its_yaml_type() {
        let rows = from_frontmatter("a: \"5\"\nb: 5\nc: \"true\"\n");
        assert_eq!(
            names_values(&rows),
            vec![
                ("a".into(), "5".into(), Kind::Text),
                ("b".into(), "5".into(), Kind::Number),
                ("c".into(), "true".into(), Kind::Text),
            ]
        );
    }

    #[test]
    fn a_sequence_is_one_row_per_scalar_element() {
        let rows = from_frontmatter("people:\n  - \"[[Ada]]\"\n  - Bob\n  - 7\n  - {x: 1}\n");
        assert_eq!(
            names_values(&rows),
            vec![
                ("people".into(), "Ada".into(), Kind::Link),
                ("people".into(), "Bob".into(), Kind::Text),
                ("people".into(), "7".into(), Kind::Number),
            ]
        );
        assert_eq!(rows[0].link_target.as_deref(), Some("Ada"));
    }

    #[test]
    fn a_quoted_wikilink_is_a_link_and_an_unquoted_one_writes_nothing() {
        let rows = from_frontmatter("employer: \"[[Acme|the firm]]\"\nmanager: [[Acme]]\n");
        assert_eq!(
            names_values(&rows),
            vec![("employer".into(), "Acme".into(), Kind::Link)]
        );
        assert_eq!(rows[0].link_target.as_deref(), Some("Acme"));
    }

    #[test]
    fn a_nested_mapping_a_non_string_key_and_a_bad_block_write_nothing() {
        assert!(from_frontmatter("meta:\n  a: 1\n").is_empty());
        assert!(from_frontmatter("1: one\n").is_empty());
        assert!(from_frontmatter("title: : :\n").is_empty());
        assert!(from_frontmatter("").is_empty());
    }

    #[test]
    fn a_name_is_trimmed_and_kept_as_written() {
        let rows = from_frontmatter("\"employee of\": Acme\nDue Date: soon\n");
        assert_eq!(rows[0].name, "employee of");
        assert_eq!(rows[1].name, "Due Date");
    }

    #[test]
    fn the_three_inline_forms_are_read() {
        let rows = from_chunk(
            3,
            "Employer:: [[Acme]]\nShe joined [rating:: 5] and (status:: active) in one line.\n",
        );
        assert_eq!(
            names_values(&rows),
            vec![
                ("Employer".into(), "Acme".into(), Kind::Link),
                ("rating".into(), "5".into(), Kind::Number),
                ("status".into(), "active".into(), Kind::Text),
            ]
        );
        assert!(rows.iter().all(|r| r.chunk_seq == 3));
        assert_eq!(rows[0].link_target.as_deref(), Some("Acme"));
    }

    #[test]
    fn a_list_item_or_task_prefix_is_stripped_from_the_full_line_form() {
        let rows = from_chunk(
            0,
            "- Mentor:: [[Bob]]\n* Due:: 2026-09-03\n- [ ] Owner:: Ada\n- [x] Done:: true\n",
        );
        assert_eq!(
            names_values(&rows),
            vec![
                ("Mentor".into(), "Bob".into(), Kind::Link),
                ("Due".into(), "2026-09-03".into(), Kind::Text),
                ("Owner".into(), "Ada".into(), Kind::Text),
                ("Done".into(), "true".into(), Kind::Checkbox),
            ]
        );
    }

    /// A callout body is `> Key:: value`, and `>` is not a character
    /// `split_field` rejects, so the marker would otherwise be filed as
    /// part of the name (#66).
    #[test]
    fn a_blockquote_or_callout_marker_is_stripped_from_the_full_line_form() {
        let rows = from_chunk(
            0,
            "> [!note] Client\n> Owner:: Ada\n>> Deputy:: Bob\n>Rank:: 3\n> - Mentor:: [[Bob]]\n",
        );
        assert_eq!(
            names_values(&rows),
            vec![
                ("Owner".into(), "Ada".into(), Kind::Text),
                ("Deputy".into(), "Bob".into(), Kind::Text),
                ("Rank".into(), "3".into(), Kind::Number),
                ("Mentor".into(), "Bob".into(), Kind::Link),
            ]
        );
    }

    #[test]
    fn a_comma_list_is_one_row_per_element() {
        let rows = from_chunk(0, "Tags Seen:: alpha, 7, [[Beta, the note]]\n");
        assert_eq!(
            names_values(&rows),
            vec![
                ("Tags Seen".into(), "alpha".into(), Kind::Text),
                ("Tags Seen".into(), "7".into(), Kind::Number),
                ("Tags Seen".into(), "Beta, the note".into(), Kind::Link),
            ]
        );
    }

    #[test]
    fn an_element_with_links_writes_one_link_row_per_link_and_no_text() {
        let rows = from_chunk(0, "Example:: see [[Acme]] and [[Bolt|the other]]\n");
        assert_eq!(
            names_values(&rows),
            vec![
                ("Example".into(), "Acme".into(), Kind::Link),
                ("Example".into(), "Bolt".into(), Kind::Link),
            ]
        );
    }

    #[test]
    fn a_fenced_code_line_and_a_bracketed_key_are_not_fields() {
        let rows = from_chunk(
            0,
            "```\nKey:: value\n```\nSee [[Note]]:: not a field\n:::note\nurl:: https://x\n",
        );
        assert_eq!(
            names_values(&rows),
            vec![("url".into(), "https://x".into(), Kind::Text)]
        );
    }

    #[test]
    fn an_empty_inline_value_is_empty() {
        let rows = from_chunk(0, "Owner::\n");
        assert_eq!(
            names_values(&rows),
            vec![("Owner".into(), "".into(), Kind::Empty)]
        );
    }

    #[test]
    fn the_bracket_form_finds_its_own_matching_close_not_the_first_one() {
        let rows = from_chunk(0, "[Owner:: [[Alice]]]\n");
        assert_eq!(
            names_values(&rows),
            vec![("Owner".into(), "Alice".into(), Kind::Link)]
        );
        assert_eq!(rows[0].link_target.as_deref(), Some("Alice"));

        let rows = from_chunk(0, "[Both:: [[Alice]] and [[Bob]]]\n");
        assert_eq!(
            names_values(&rows),
            vec![
                ("Both".into(), "Alice".into(), Kind::Link),
                ("Both".into(), "Bob".into(), Kind::Link),
            ]
        );
    }

    #[test]
    fn a_body_field_named_for_a_built_in_writes_no_row() {
        let rows = from_chunk(
            0,
            "tags:: work, urgent\naliases:: Al\ncssclasses:: x\ntagsmith:: y\n",
        );
        assert_eq!(
            names_values(&rows),
            vec![("tagsmith".into(), "y".into(), Kind::Text)]
        );
    }

    #[test]
    fn declared_types_reads_the_obsidian_file_and_answers_empty_otherwise() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        assert!(declared_types(root).is_empty(), "no .obsidian at all");
        std::fs::create_dir_all(root.join(".obsidian")).unwrap();
        std::fs::write(root.join(".obsidian/types.json"), "{not json").unwrap();
        assert!(
            declared_types(root).is_empty(),
            "a broken file raises nothing"
        );
        std::fs::write(
            root.join(".obsidian/types.json"),
            r#"{"types":{"status":"text","rating":"number","tags":"tags"}}"#,
        )
        .unwrap();
        let got = declared_types(root);
        assert_eq!(got.get("status").map(String::as_str), Some("text"));
        assert_eq!(got.get("rating").map(String::as_str), Some("number"));
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn the_registry_carries_declared_types_and_declared_only_names() {
        use crate::store::{DOC_LEVEL, NewProperty, Store};
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join(".obsidian")).unwrap();
        std::fs::write(
            root.join(".obsidian/types.json"),
            r#"{"types":{"status":"text","phantom":"checkbox","tags":"tags"}}"#,
        )
        .unwrap();
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store
            .replace_file_properties(
                a,
                &[NewProperty {
                    chunk_seq: DOC_LEVEL,
                    name: "status",
                    value: "draft",
                    kind: Kind::Text,
                    target_file: None,
                }],
            )
            .unwrap();
        let rows = registry(&store, root).unwrap();
        let got: Vec<(&str, usize, Option<&str>)> = rows
            .iter()
            .map(|r| (r.name.as_str(), r.note_count, r.declared_type.as_deref()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("status", 1, Some("text")),
                ("phantom", 0, Some("checkbox"))
            ],
            "tags is a built-in and is not a declared-only row"
        );
    }

    #[test]
    fn run_answers_the_registry_or_one_names_values() {
        use crate::store::{DOC_LEVEL, NewProperty, Store};
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store
            .replace_file_properties(
                a,
                &[NewProperty {
                    chunk_seq: DOC_LEVEL,
                    name: "status",
                    value: "draft",
                    kind: Kind::Text,
                    target_file: None,
                }],
            )
            .unwrap();
        let whole = run(
            &store,
            tmp.path(),
            &crate::params::Properties { name: None },
        )
        .unwrap();
        assert_eq!(render_text(&whole), "status (1) text\n");
        let one = run(
            &store,
            tmp.path(),
            &crate::params::Properties {
                name: Some("status".into()),
            },
        )
        .unwrap();
        assert_eq!(render_text(&one), "draft (1) text\n");
        let json = serde_json::to_value(&one).unwrap();
        assert_eq!(json[0]["value"], "draft");
    }
}
