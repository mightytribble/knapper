//! Custom properties (#66): a named value on a note, read from frontmatter
//! under Obsidian's rules and from Dataview inline fields under Dataview's.
//!
//! This module owns the kind vocabulary and the two extractors. Nothing else
//! in the crate reads a property out of a note.

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
}
