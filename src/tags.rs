use anyhow::Result;
use rusqlite::{Connection, params};
use std::collections::HashSet;
use strsim::levenshtein;

/// A tag, as the store keys it and as the vault wrote it.
///
/// Obsidian matches a tag without regard to case and displays the capitalisation
/// the vault used, so both strings are kept: `path` is the identity and
/// `display` is what a reader sees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    /// Folded: `type/undead`. Unique in the `tags` table.
    pub path: String,
    /// As written: `Type/Undead`.
    pub display: String,
}

impl Tag {
    fn new(written: &str) -> Self {
        Tag {
            path: written.to_lowercase(),
            display: written.to_string(),
        }
    }

    /// The first segment of the path. `type/undead` has the axis `type`.
    pub fn axis(&self) -> &str {
        match self.path.split_once('/') {
            Some((head, _)) => head,
            None => &self.path,
        }
    }
}

/// Every tag a note carries, property and body, in that order.
///
/// The two sources are peers: neither requires the other, and a note tagged in
/// both holds the tag once. Only the path a note carries is returned —
/// `type/undead` does not imply a `type` tag, and a query reads the ancestors
/// out of the path text.
pub fn extract(content: &str) -> Vec<Tag> {
    let (frontmatter, body) = crate::markdown::split_frontmatter(content);
    let mut out: Vec<Tag> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    if let Some(fm) = frontmatter {
        for written in property_tags(&fm) {
            push_tag(&mut out, &mut seen, &written);
        }
    }
    for written in body_tags(&body) {
        push_tag(&mut out, &mut seen, &written);
    }
    out
}

/// Keep the first spelling of a path and drop the rest.
fn push_tag(out: &mut Vec<Tag>, seen: &mut HashSet<String>, written: &str) {
    let tag = Tag::new(written);
    if tag.path.is_empty() {
        return;
    }
    if seen.insert(tag.path.clone()) {
        out.push(tag);
    }
}

/// The `tags` property, in the three forms Obsidian accepts.
///
/// `tags: [a, b]`, a block sequence of `- a` lines, and a single scalar
/// `tags: a`. A value may carry one leading `#`, which is stripped. A value
/// runs through the same token test as a body tag: the character set of
/// `is_tag_char` and at least one non-numeric character. A value that fails
/// the test is dropped, and dropping it does not drop the other values on the
/// line.
///
/// A trailing YAML comment is cut before the value split, and the cut is
/// shaped to where a comment can be. The flow-sequence form reads only the
/// text between `[` and the matching `]`, so a `#` anywhere inside the
/// brackets — first item or later — is always a value's own hash, never a
/// comment mark: `tags: [alpha, #beta]` keeps `beta`, and anything past the
/// `]` is a comment or nothing, either way not a value. The scalar and
/// block-sequence forms carry no brackets, so `strip_trailing_comment` runs
/// on them instead: a `#` after whitespace opens a comment, and a `#` at the
/// front of the value is that value's own hash and stays. The key is read at
/// column 0 only, because Obsidian's properties are top level and an
/// indented `tags:` belongs to some other mapping. The singular `tag`
/// property is not read: Obsidian dropped support for it at 1.9.
fn property_tags(frontmatter: &str) -> Vec<String> {
    let lines: Vec<&str> = frontmatter.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let Some(after) = line.strip_prefix("tags:") else {
            continue;
        };
        let after = after.trim();
        if let Some(inner) = after.strip_prefix('[') {
            // A comment can only follow the closing `]`. Read up to the
            // first `]` only, so nothing past it, comment or not, reaches
            // the split, and a `#` inside the brackets is never mistaken
            // for one.
            let items = match inner.find(']') {
                Some(close) => &inner[..close],
                None => inner,
            };
            return items
                .split(',')
                .map(clean_property_value)
                .filter(|s| is_valid_tag_token(s))
                .collect();
        }
        let after = strip_trailing_comment(after);
        if !after.is_empty() {
            let value = clean_property_value(after);
            return if is_valid_tag_token(&value) {
                vec![value]
            } else {
                Vec::new()
            };
        }
        let mut out = Vec::new();
        for subsequent in &lines[i + 1..] {
            let trimmed = subsequent.trim();
            if let Some(item) = trimmed.strip_prefix("- ") {
                let value = clean_property_value(strip_trailing_comment(item));
                if is_valid_tag_token(&value) {
                    out.push(value);
                }
            } else if trimmed.is_empty() {
                continue;
            } else {
                break;
            }
        }
        return out;
    }
    Vec::new()
}

/// Cut a trailing YAML comment from a property value that carries no
/// brackets: the scalar form and one block-sequence item.
///
/// A `#` is a comment mark when it follows a space. A `#` at the very start
/// of the text opens a value instead and is kept: this is what lets a bare
/// `tags: #undead` keep its tag, while still cutting `tags: alpha  # todo`
/// down to `alpha`. The flow-sequence form does not call this function: it
/// reads only the text inside its brackets, where every `#` is a value's own
/// hash by construction, so a comment cut there would misread the space
/// before a later item's `#` as a comment mark.
fn strip_trailing_comment(s: &str) -> &str {
    let mut prev_is_space = false;
    for (idx, c) in s.char_indices() {
        if c == '#' && prev_is_space {
            return s[..idx].trim_end();
        }
        prev_is_space = c.is_whitespace();
    }
    s
}

/// Trim quotes and one leading `#` from a raw property value.
///
/// This only cleans the text. It does not check whether the result is a
/// valid tag; call `is_valid_tag_token` on the result for that.
fn clean_property_value(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('"').trim_matches('\'').trim();
    trimmed.strip_prefix('#').unwrap_or(trimmed).to_string()
}

/// The token test a tag must pass, from either source.
///
/// A tag holds only `is_tag_char` characters and at least one non-numeric
/// character. A body token already meets the character-set half by
/// construction, because `line_tags` stops reading at the first character
/// outside the set; a property value is free text and must be checked in
/// full, so a value such as `"my tag"` is dropped rather than cut at the
/// space.
fn is_valid_tag_token(token: &str) -> bool {
    !token.is_empty()
        && token.chars().all(is_tag_char)
        && token.chars().any(|c| !c.is_ascii_digit())
}

/// Every `#tag` token the body holds, with the five rejections applied.
fn body_tags(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut fenced = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        out.extend(line_tags(line));
    }
    out
}

/// The token rule, over one line of body text.
fn line_tags(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut in_code_span = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '`' {
            in_code_span = !in_code_span;
            i += 1;
            continue;
        }
        if chars[i] != '#' || in_code_span {
            i += 1;
            continue;
        }
        // A tag opens a line or follows whitespace. That one test rejects three
        // of the five constructions: a URL fragment (`…/#section`) is preceded
        // by `/`, a wikilink heading (`[[note#Heading]]`) by a letter or `[`,
        // and an escaped `\#` by the backslash.
        if i > 0 && !chars[i - 1].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut end = start;
        while end < chars.len() && is_tag_char(chars[end]) {
            end += 1;
        }
        let token: String = chars[start..end].iter().collect();
        // An ATX heading is `#` followed by a space, which leaves the token
        // empty. `#1984` holds no non-numeric character and is not a tag.
        // This is the same token test a property value must pass.
        if is_valid_tag_token(&token) {
            out.push(token);
        }
        i = end.max(start);
    }
    out
}

/// Letters, digits, `_`, `-`, `/` and Unicode characters. A tag holds no space.
fn is_tag_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-' || c == '/'
}

/// Result of resolving a proposed tag against the registry.
#[derive(Debug, Clone, PartialEq)]
pub enum TagResolution {
    /// Exact case-insensitive match found.
    Exact(String),
    /// Fuzzy match: Levenshtein distance ≤ 2.
    Fuzzy {
        proposed: String,
        resolved: String,
        distance: usize,
    },
    /// Proposed tag extends an existing tag via `/` hierarchy.
    Extension(String),
    /// No match — this is a brand-new tag.
    New(String),
}

/// Resolve a single proposed tag against the vocabulary the vault holds.
///
/// Resolution tiers (priority order):
/// 1. Exact match (without regard to case), returning the vault's own spelling
/// 2. Fuzzy match (Levenshtein distance ≤ 2) against the tags that share a
///    parent, comparing the last segment alone
/// 3. Prefix extension (proposed starts with `existing_tag/`)
/// 4. New tag
///
/// Tier 2 is scoped because a tag is a path and its segments are not
/// interchangeable: `type/undead` and `type/undeed` are one misspelling apart,
/// and `type/undead` and `habitat/undead` are two different attributes.
pub fn resolve_tag(conn: &Connection, proposed: &str) -> Result<TagResolution> {
    let lower = proposed.to_lowercase();

    let exact: Option<String> = conn
        .prepare("SELECT display FROM tags WHERE path = ?1")?
        .query_map(params![lower], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .next();

    if let Some(display) = exact {
        return Ok(TagResolution::Exact(display));
    }

    let all_tags: Vec<String> = conn
        .prepare("SELECT display FROM tags")?
        .query_map([], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    // Tier 2: the last segment, against the last segments under the same parent.
    let (parent, leaf) = split_leaf(&lower);
    let mut best: Option<(String, usize)> = None;
    for tag in &all_tags {
        let candidate = tag.to_lowercase();
        let (candidate_parent, candidate_leaf) = split_leaf(&candidate);
        if candidate_parent != parent {
            continue;
        }
        let dist = levenshtein(leaf, candidate_leaf);
        if dist > 0 && dist <= 2 && (best.is_none() || dist < best.as_ref().unwrap().1) {
            best = Some((tag.clone(), dist));
        }
    }
    if let Some((resolved, distance)) = best {
        return Ok(TagResolution::Fuzzy {
            proposed: proposed.to_string(),
            resolved,
            distance,
        });
    }

    // Tier 3: prefix extension — proposed starts with `existing_tag/`.
    for tag in &all_tags {
        if lower.starts_with(&format!("{}/", tag.to_lowercase())) {
            return Ok(TagResolution::Extension(proposed.to_string()));
        }
    }

    Ok(TagResolution::New(proposed.to_string()))
}

/// `type/undead` → (`type`, `undead`). A top-level tag has the empty parent.
fn split_leaf(path: &str) -> (&str, &str) {
    match path.rsplit_once('/') {
        Some((parent, leaf)) => (parent, leaf),
        None => ("", path),
    }
}

/// Resolve a list of proposed tags, returning the final tag names.
///
/// - Exact / Fuzzy matches map to the resolved name.
/// - Extension / New tags pass through as-is.
pub fn resolve_tags(conn: &Connection, proposed: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(proposed.len());
    for tag in proposed {
        let resolved = resolve_tag(conn, tag)?;
        let name = match resolved {
            TagResolution::Exact(name) => name,
            TagResolution::Fuzzy { resolved, .. } => resolved,
            TagResolution::Extension(name) => name,
            TagResolution::New(name) => name,
        };
        out.push(name);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    // ── The extraction rule ──────────────────────────────────────

    fn paths(content: &str) -> Vec<String> {
        extract(content).into_iter().map(|t| t.path).collect()
    }

    #[test]
    fn the_property_is_read_in_its_three_forms() {
        assert_eq!(
            paths("---\ntags: [alpha, beta]\n---\nbody\n"),
            ["alpha", "beta"]
        );
        assert_eq!(
            paths("---\ntags:\n  - alpha\n  - beta\n---\nbody\n"),
            ["alpha", "beta"]
        );
        assert_eq!(paths("---\ntags: alpha\n---\nbody\n"), ["alpha"]);
    }

    #[test]
    fn a_property_value_may_carry_one_hash() {
        assert_eq!(paths("---\ntags: [#undead]\n---\nbody\n"), ["undead"]);
    }

    #[test]
    fn the_singular_tag_property_is_not_read() {
        // Obsidian dropped support for it at 1.9.
        assert!(paths("---\ntag: alpha\n---\nbody\n").is_empty());
    }

    #[test]
    fn the_body_supplies_tags_and_the_two_sources_are_a_union() {
        assert_eq!(paths("body with #alpha in it\n"), ["alpha"]);
        assert_eq!(
            paths("---\ntags: [alpha]\n---\nbody with #beta\n"),
            ["alpha", "beta"]
        );
    }

    #[test]
    fn one_tag_from_both_sources_is_one_tag() {
        assert_eq!(
            paths("---\ntags: [alpha]\n---\nbody with #alpha\n"),
            ["alpha"]
        );
    }

    #[test]
    fn an_atx_heading_is_not_a_tag() {
        assert!(paths("# Heading\n\n## Deeper\n").is_empty());
    }

    #[test]
    fn code_is_not_a_tag() {
        assert!(paths("```\n#alpha\n```\n").is_empty());
        assert!(paths("~~~\n#alpha\n~~~\n").is_empty());
        assert!(paths("a `#alpha` span\n").is_empty());
    }

    #[test]
    fn a_url_fragment_is_not_a_tag() {
        assert!(paths("see https://example.com/#section for more\n").is_empty());
    }

    #[test]
    fn a_wikilink_heading_is_not_a_tag() {
        assert!(paths("see [[note#Heading]]\n").is_empty());
    }

    #[test]
    fn an_escaped_hash_is_not_a_tag() {
        assert!(paths("a literal \\#alpha\n").is_empty());
    }

    #[test]
    fn a_tag_holds_at_least_one_non_numeric_character() {
        assert!(paths("in #1984 nothing happened\n").is_empty());
        assert_eq!(paths("in #y1984 something did\n"), ["y1984"]);
    }

    #[test]
    fn a_tag_ends_at_the_first_character_outside_the_set() {
        assert_eq!(paths("tagged #type/undead, and armed.\n"), ["type/undead"]);
        assert_eq!(paths("#alpha_beta-1/two.\n"), ["alpha_beta-1/two"]);
    }

    #[test]
    fn the_path_is_folded_and_the_display_form_is_not() {
        let tags = extract("body #Type/Undead here\n");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].path, "type/undead");
        assert_eq!(tags[0].display, "Type/Undead");
        assert_eq!(tags[0].axis(), "type");
    }

    #[test]
    fn one_tag_in_two_spellings_keeps_the_first() {
        let tags = extract("---\ntags: [Type/Undead]\n---\nbody #type/undead\n");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].display, "Type/Undead");
    }

    #[test]
    fn a_property_value_holds_the_same_token_test_as_the_body() {
        assert!(paths("---\ntags: [1984]\n---\nbody\n").is_empty());
        assert!(paths("---\ntags: [my tag]\n---\nbody\n").is_empty());
    }

    #[test]
    fn a_trailing_comment_on_the_tags_line_is_not_a_tag() {
        assert_eq!(
            paths("---\ntags: [alpha, beta]  # todo\n---\nbody\n"),
            ["alpha", "beta"]
        );
        assert_eq!(paths("---\ntags: alpha  # todo\n---\nbody\n"), ["alpha"]);
    }

    #[test]
    fn a_hash_prefixed_value_after_the_first_is_not_a_comment() {
        assert_eq!(
            paths("---\ntags: [alpha, #beta]\n---\nbody\n"),
            ["alpha", "beta"]
        );
    }

    #[test]
    fn a_flow_sequence_may_carry_hash_values_and_a_trailing_comment() {
        assert_eq!(
            paths("---\ntags: [#alpha, #beta]  # todo\n---\nbody\n"),
            ["alpha", "beta"]
        );
    }

    fn setup_store() -> Store {
        let store = Store::open_memory().unwrap();
        let conn = store.conn();
        for display in [
            "domaine",
            "scentbird",
            "engraph",
            "work",
            "work/domaine",
            "Type/Undead",
            "habitat/undead",
        ] {
            conn.execute(
                "INSERT INTO tags (path, display) VALUES (?1, ?2)",
                params![display.to_lowercase(), display],
            )
            .unwrap();
        }
        store
    }

    #[test]
    fn fuzzy_matching_is_scoped_to_the_segments_that_share_a_parent() {
        let store = setup_store();
        match resolve_tag(store.conn(), "type/undeed").unwrap() {
            TagResolution::Fuzzy { resolved, .. } => assert_eq!(resolved, "Type/Undead"),
            other => panic!("expected Fuzzy, got {other:?}"),
        }
        // `habitat/undead` is one edit from `habitat/undeed` and nothing else:
        // a different parent is a different tag, not a misspelling of this one.
        match resolve_tag(store.conn(), "quality/undead").unwrap() {
            TagResolution::New(name) => assert_eq!(name, "quality/undead"),
            other => panic!("expected New, got {other:?}"),
        }
    }

    #[test]
    fn an_exact_match_returns_the_vaults_own_spelling() {
        let store = setup_store();
        assert_eq!(
            resolve_tag(store.conn(), "type/undead").unwrap(),
            TagResolution::Exact("Type/Undead".to_string())
        );
    }

    #[test]
    fn test_exact_match() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "domaine").unwrap();
        assert_eq!(res, TagResolution::Exact("domaine".to_string()));
    }

    #[test]
    fn test_exact_match_case_insensitive() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "Domaine").unwrap();
        assert_eq!(res, TagResolution::Exact("domaine".to_string()));
    }

    #[test]
    fn test_fuzzy_match() {
        let store = setup_store();
        // "doamine" is Levenshtein distance 2 from "domaine" (transposition).
        let res = resolve_tag(store.conn(), "doamine").unwrap();
        match res {
            TagResolution::Fuzzy {
                proposed,
                resolved,
                distance,
            } => {
                assert_eq!(proposed, "doamine");
                assert_eq!(resolved, "domaine");
                assert!(distance <= 2);
            }
            other => panic!("expected Fuzzy, got {other:?}"),
        }
    }

    #[test]
    fn test_hierarchy_extension() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "work/domaine/bre").unwrap();
        assert_eq!(
            res,
            TagResolution::Extension("work/domaine/bre".to_string())
        );
    }

    #[test]
    fn test_new_tag() {
        let store = setup_store();
        let res = resolve_tag(store.conn(), "completely-new").unwrap();
        assert_eq!(res, TagResolution::New("completely-new".to_string()));
    }

    #[test]
    fn test_resolve_tags() {
        let store = setup_store();
        let input = vec![
            "domaine".to_string(),
            "doamine".to_string(),
            "completely-new".to_string(),
        ];
        let resolved = resolve_tags(store.conn(), &input).unwrap();
        assert_eq!(resolved, vec!["domaine", "domaine", "completely-new"]);
    }
}
