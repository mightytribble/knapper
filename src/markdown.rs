#[derive(Debug, Clone)]
pub struct HeadingInfo {
    pub line: usize,
    pub level: u8,
    pub text: String,
    /// A bold-only line promoted to a heading, rather than an ATX heading
    /// (issue #44). The flag is what a caller reads, so nothing outside this
    /// module tests a level against [`PROMOTED_LEVEL`] (issue #69).
    pub promoted: bool,
}

pub fn parse_headings(content: &str) -> Vec<HeadingInfo> {
    let mut headings = Vec::new();
    let mut in_code_block = false;
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            let hashes = rest.chars().take_while(|&c| c == '#').count();
            let level = 1 + hashes as u8;
            let after_hashes = &rest[hashes..];
            if level <= 6 && (after_hashes.is_empty() || after_hashes.starts_with(' ')) {
                let text = after_hashes.trim().trim_end_matches('#').trim();
                headings.push(HeadingInfo {
                    line: i,
                    level,
                    text: text.to_string(),
                    promoted: false,
                });
            }
        }
    }
    headings
}

/// The level a promoted line takes (issue #44).
///
/// It is deeper than every `#` level, so an ancestor walk pops it for the
/// next heading of any depth and for the next promoted line, and it is an
/// ancestor of nothing. The value decides where a section ends and what a
/// breadcrumb holds; it is written to no row and rendered as no markdown.
pub(crate) const PROMOTED_LEVEL: u8 = u8::MAX;

/// The text of a bold-only line, or `None` when the line is not one.
///
/// A promoted heading is one bold span and nothing else: `**Text**`,
/// `__Text__`, or either with a single colon directly after the closing
/// marker (issue #44). A table row, a list item and the bestiary's
/// `**Rank**: S • **Levels**: …` preamble all carry text outside the span, so
/// the content test rejects them. So does a bold span wrapping an italic one,
/// which is emphasis rather than a label.
fn bold_heading_text(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let body = trimmed.strip_suffix(':').unwrap_or(trimmed).trim_end();
    let inner = body
        .strip_prefix("**")
        .and_then(|rest| rest.strip_suffix("**"))
        .or_else(|| {
            body.strip_prefix("__")
                .and_then(|rest| rest.strip_suffix("__"))
        })?;
    let inner = inner.trim();
    if inner.is_empty()
        || inner.contains("**")
        || inner.contains("__")
        || inner.starts_with('*')
        || inner.ends_with('*')
        || inner.starts_with('_')
        || inner.ends_with('_')
    {
        return None;
    }
    Some(inner)
}

/// Every heading a file holds: the ATX headings, and every bold-only line
/// promoted to one (issue #44).
///
/// This is the set `find_section` addresses and `list --detailed`
/// enumerates, so a caller can name the section the outline printed (issue
/// #69). The chunker's set is this one with `drop_bodyless_promotions`
/// applied: a promoted line with no body of its own starts no chunk, and it
/// is still a section a caller may read or fill.
pub fn headings_with_promotions(content: &str) -> Vec<HeadingInfo> {
    let headings = parse_headings(content);
    let mut merged: Vec<HeadingInfo> = Vec::with_capacity(headings.len());
    let mut next = 0usize;
    let mut in_fence = false;

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if next < headings.len() && headings[next].line == i {
            merged.push(headings[next].clone());
            next += 1;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(text) = bold_heading_text(line) {
            merged.push(HeadingInfo {
                line: i,
                level: PROMOTED_LEVEL,
                text: text.to_string(),
                promoted: true,
            });
        }
    }
    merged
}

#[derive(Debug, Clone)]
pub struct Section {
    pub heading: HeadingInfo,
    pub body_start: usize,
    pub body_end: usize,
    pub content: String,
}

/// The section a caller named, or `None`.
///
/// `heading_text` is one heading's own text, or the heading's full path from
/// its own root with the segments joined by `>` (issue #69). Both fold case.
/// A single segment resolves the first heading with that text at any depth,
/// which is what this function has always answered; a path resolves the one
/// heading whose ancestry is exactly it, so a note that repeats a heading
/// text is addressable at every occurrence whose ancestry differs.
///
/// A partial path resolves nothing. A path that must be complete matches one
/// heading or none, so a wrong guess is an error rather than an edit to the
/// wrong section.
pub fn find_section(content: &str, heading_text: &str) -> Option<Section> {
    // ATX first, then the same two lookups over the merged set, so an ATX
    // `### History` keeps precedence over a `**History**` under the same
    // parent: the second pass runs only when the first answered nothing
    // (issue #69).
    [parse_headings(content), headings_with_promotions(content)]
        .into_iter()
        .find_map(|headings| {
            let idx =
                by_text(&headings, heading_text).or_else(|| by_path(&headings, heading_text))?;
            Some(section_at(content, &headings, idx))
        })
}

/// The first heading whose own text is `query`.
fn by_text(headings: &[HeadingInfo], query: &str) -> Option<usize> {
    let target = normalise(query);
    headings.iter().position(|h| normalise(&h.text) == target)
}

/// The heading whose path is `query`.
///
/// One segment is not a path: [`by_text`] answers that form, and it runs
/// first, so a heading whose own text holds a `>` resolves by name.
fn by_path(headings: &[HeadingInfo], query: &str) -> Option<usize> {
    let segments: Vec<String> = query.split('>').map(normalise).collect();
    if segments.len() < 2 || segments.iter().any(String::is_empty) {
        return None;
    }
    (0..headings.len()).find(|&i| path_of(headings, i) == segments)
}

/// A heading's ancestors from its own root, its own text last.
///
/// The walk pops an open heading of equal or greater level, which is the rule
/// the chunker's ancestor stack follows, so a skipped level does not break an
/// ancestry and a promoted line is an ancestor of nothing (issue #44).
fn path_of(headings: &[HeadingInfo], idx: usize) -> Vec<String> {
    let mut stack: Vec<(u8, String)> = Vec::new();
    for h in &headings[..=idx] {
        while stack.last().is_some_and(|(level, _)| *level >= h.level) {
            stack.pop();
        }
        stack.push((h.level, normalise(&h.text)));
    }
    stack.into_iter().map(|(_, text)| text).collect()
}

/// One segment as it is compared: the bold markers of a promoted heading
/// removed, then trimmed and folded.
///
/// `chunks.heading` holds the raw `**Spells**` and `chunks.heading_path` holds
/// the stripped `Spells`, so both spellings are in circulation and a caller
/// may paste either. `bold_heading_text` is what defines the marker on both
/// sides, so the comparison and the merge cannot disagree (issue #69).
fn normalise(segment: &str) -> String {
    let trimmed = segment.trim();
    bold_heading_text(trimmed)
        .unwrap_or(trimmed)
        .trim()
        .to_lowercase()
}

/// The span a heading owns: from the line after it to the next heading at or
/// above its own level, or the end of the file.
fn section_at(content: &str, headings: &[HeadingInfo], idx: usize) -> Section {
    let lines: Vec<&str> = content.lines().collect();
    let h = &headings[idx];
    let body_start = h.line + 1;
    let body_end = headings[idx + 1..]
        .iter()
        .find(|next| next.level <= h.level)
        .map(|next| next.line)
        .unwrap_or(lines.len());

    Section {
        heading: h.clone(),
        body_start,
        body_end,
        content: lines[body_start..body_end].join("\n"),
    }
}

pub fn split_frontmatter(content: &str) -> (Option<String>, String) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.first().map(|l| l.trim()) != Some("---") {
        return (None, content.to_string());
    }
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim() == "---" {
            let fm = lines[1..i].join("\n");
            let body = lines[i + 1..].join("\n");
            return (Some(fm), body);
        }
    }
    (None, content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_headings_basic() {
        let content = "# Title\n\nSome text\n\n## Section A\n\nContent\n\n## Section B\n";
        let headings = parse_headings(content);
        assert_eq!(headings.len(), 3);
        assert_eq!(headings[0].level, 1);
        assert_eq!(headings[0].text, "Title");
        assert_eq!(headings[1].level, 2);
        assert_eq!(headings[1].text, "Section A");
    }

    #[test]
    fn test_parse_headings_ignores_code_blocks() {
        let content = "# Real\n\n```\n# Not a heading\n```\n\n## Also Real\n";
        let headings = parse_headings(content);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].text, "Real");
        assert_eq!(headings[1].text, "Also Real");
    }

    #[test]
    fn test_parse_headings_strips_trailing_hashes() {
        let content = "## Heading ##\n";
        let headings = parse_headings(content);
        assert_eq!(headings[0].text, "Heading");
    }

    #[test]
    fn test_find_section_basic() {
        let content = "# Title\n\n## Interactions\n\nEntry 1\nEntry 2\n\n## Links\n\nSome links\n";
        let section = find_section(content, "Interactions").unwrap();
        assert_eq!(section.heading.text, "Interactions");
        assert!(section.content.contains("Entry 1"));
        assert!(!section.content.contains("Some links"));
    }

    #[test]
    fn test_find_section_case_insensitive() {
        let content = "## My Section\n\nContent\n";
        assert!(find_section(content, "my section").is_some());
    }

    #[test]
    fn test_find_section_with_subsections() {
        let content = "# Title\n\n## Interactions\n\nEntry\n\n### Sub-detail\n\nMore\n\n## Links\n\nSome links\n";
        let section = find_section(content, "Interactions").unwrap();
        assert!(section.content.contains("Entry"));
        assert!(section.content.contains("Sub-detail"));
        assert!(!section.content.contains("Some links"));
    }

    #[test]
    fn test_find_section_not_found() {
        let content = "## Existing\n\nContent\n";
        assert!(find_section(content, "Missing").is_none());
    }

    #[test]
    fn test_split_frontmatter_valid() {
        let content = "---\ntitle: Test\ntags:\n  - foo\n---\n\n# Body\n";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_some());
        assert!(fm.unwrap().contains("title: Test"));
        assert!(body.contains("# Body"));
    }

    #[test]
    fn test_split_frontmatter_none() {
        let content = "# No frontmatter\n\nJust content\n";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_none());
        assert!(body.contains("No frontmatter"));
    }

    #[test]
    fn test_parse_headings_ignores_inline_tags() {
        let content = "# Title\n\nSome text with #tag and #another-tag\n\n## Real Section\n";
        let headings = parse_headings(content);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].text, "Title");
        assert_eq!(headings[1].text, "Real Section");
    }

    /// The merged set is every heading the file holds: the ATX headings, and
    /// every bold-only line beside them (#44, #69).
    #[test]
    fn the_merged_set_holds_atx_headings_and_promoted_lines() {
        let content = "## Stat Block\n\nAC 20\n\n**Spells**\n\nFireball\n";
        let headings = headings_with_promotions(content);
        let got: Vec<(&str, bool)> = headings
            .iter()
            .map(|h| (h.text.as_str(), h.promoted))
            .collect();
        assert_eq!(got, vec![("Stat Block", false), ("Spells", true)]);
    }

    /// A promoted line with no body of its own is in the merged set, because
    /// addressing an empty section is how a caller fills it. The chunker is
    /// what drops it, and for its own reason (#69).
    #[test]
    fn the_merged_set_keeps_a_bodyless_promoted_line() {
        let content = "## Stat Block\n\n**Spells**\n**Notes**\n\nSee below\n";
        let headings = headings_with_promotions(content);
        let got: Vec<&str> = headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(got, vec!["Stat Block", "Spells", "Notes"]);
    }

    /// A bold-only line inside a fence is a code sample and not a heading,
    /// as a `#` line inside one is not (#44).
    #[test]
    fn a_bold_line_inside_a_fence_is_not_promoted() {
        let content = "## Stat Block\n\n```md\n**Spells**\n```\n";
        let headings = headings_with_promotions(content);
        let got: Vec<&str> = headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(got, vec!["Stat Block"]);
    }

    /// `parse_headings` answers ATX headings, so nothing it returns is
    /// promoted (#69).
    #[test]
    fn an_atx_heading_is_not_promoted() {
        let headings = parse_headings("# Title\n\n## Section\n");
        assert!(headings.iter().all(|h| !h.promoted));
    }

    #[test]
    fn bold_only_lines_are_recognised() {
        assert_eq!(bold_heading_text("**Spells**"), Some("Spells"));
        assert_eq!(bold_heading_text("__Spells__"), Some("Spells"));
        assert_eq!(bold_heading_text("**Spells**:"), Some("Spells"));
        assert_eq!(bold_heading_text("  **Spells**  "), Some("Spells"));
        assert_eq!(bold_heading_text("**Human Forms**"), Some("Human Forms"));
    }

    #[test]
    fn a_line_with_anything_outside_the_bold_span_is_not_a_heading() {
        // The bestiary preamble: one per file, and it is data, not structure.
        assert_eq!(
            bold_heading_text(
                "**Rank**: S • **Levels**: 110-255 • **Threat**: peer of a Demon Lord"
            ),
            None
        );
        assert_eq!(bold_heading_text("- **Spells**"), None);
        assert_eq!(bold_heading_text("| **Spells** |"), None);
        assert_eq!(bold_heading_text("**Spells** and more"), None);
        assert_eq!(bold_heading_text("Spells"), None);
        assert_eq!(bold_heading_text(""), None);
        assert_eq!(bold_heading_text("**"), None);
        assert_eq!(bold_heading_text("****"), None);
        // Bold wrapping an italic span is emphasis, not a label.
        assert_eq!(bold_heading_text("***Spells***"), None);
    }

    /// A note that repeats a heading text: the bare form takes the first,
    /// and a path takes the one whose ancestry it names (#69).
    #[test]
    fn a_path_addresses_the_repeat_a_bare_heading_cannot() {
        let content = "# About the Empire\n\n## History\n\nFounding\n\n## Current Events\n\n### History\n\nRecent\n";
        assert!(
            find_section(content, "History")
                .unwrap()
                .content
                .contains("Founding")
        );
        assert!(
            find_section(content, "About the Empire > Current Events > History")
                .unwrap()
                .content
                .contains("Recent")
        );
        assert!(
            find_section(content, "About the Empire > History")
                .unwrap()
                .content
                .contains("Founding")
        );
    }

    /// A path is the whole ancestry or it is nothing. A partial path names no
    /// heading, so a caller that guesses gets an error rather than another
    /// note's section (#69).
    #[test]
    fn a_partial_path_resolves_nothing() {
        let content = "# About the Empire\n\n## Current Events\n\n### History\n\nRecent\n";
        assert!(find_section(content, "Current Events > History").is_none());
    }

    /// An empty segment is not a heading text, so a path holding one names
    /// nothing (#69).
    #[test]
    fn an_empty_segment_resolves_nothing() {
        let content = "# A\n\n## B\n\nBody\n";
        assert!(find_section(content, "A >  > B").is_none());
        assert!(find_section(content, "A > B > ").is_none());
    }

    /// The text form runs before the path form, so a heading whose own text
    /// holds a `>` is still addressable by name (#69).
    #[test]
    fn a_heading_whose_text_holds_an_angle_bracket_resolves_by_name() {
        let content = "## A > B\n\nBody\n";
        assert!(
            find_section(content, "A > B")
                .unwrap()
                .content
                .contains("Body")
        );
    }

    /// A skipped level does not break the ancestry: a `###` under a `#` is
    /// that heading's child (#69).
    #[test]
    fn a_skipped_level_is_still_the_parent() {
        let content = "# A\n\n### B\n\nBody\n";
        assert!(
            find_section(content, "A > B")
                .unwrap()
                .content
                .contains("Body")
        );
    }

    /// Two same-named siblings under one parent share a path, so the first in
    /// document order is the one that resolves — the rule the bare form has
    /// always followed (#69).
    #[test]
    fn twin_siblings_resolve_the_first() {
        let content = "# A\n\n## Notes\n\nFirst\n\n## Notes\n\nSecond\n";
        assert!(
            find_section(content, "A > Notes")
                .unwrap()
                .content
                .contains("First")
        );
    }

    /// Every segment folds case, as the bare form has always folded it (#69).
    #[test]
    fn a_path_folds_case_segment_by_segment() {
        let content = "# About the Empire\n\n## History\n\nFounding\n";
        assert!(find_section(content, "about the empire > HISTORY").is_some());
    }

    /// A promoted bold line is a section the resolver reaches, so the section
    /// a search result names is a section a caller can read (#53, #69).
    #[test]
    fn a_promoted_line_is_addressable_bare_and_by_path() {
        let content = "## Stat Block\n\nAC 20\n\n**Spells**\n\nFireball\n\n## Lore\n\nOld\n";
        let bare = find_section(content, "Spells").unwrap();
        assert!(bare.content.contains("Fireball"));
        assert!(!bare.content.contains("Old"));
        let path = find_section(content, "Stat Block > Spells").unwrap();
        assert_eq!(path.heading.line, bare.heading.line);
        assert!(path.heading.promoted);
    }

    /// `chunks.heading` holds the raw line and `chunks.heading_path` holds
    /// the stripped text, so a caller may paste either spelling (#69).
    #[test]
    fn every_spelling_of_a_promoted_heading_resolves() {
        let content = "## Stat Block\n\nAC 20\n\n**Spells**\n\nFireball\n";
        for named in ["Spells", "**Spells**", "__Spells__", "**Spells**:"] {
            assert!(
                find_section(content, named).is_some(),
                "{named} resolved nothing"
            );
        }
        assert!(
            find_section(content, "Stat Block > **Spells**")
                .unwrap()
                .content
                .contains("Fireball")
        );
    }

    /// The ATX pass runs first, so a `###` wins over a `**bold**` of the same
    /// name under the same parent (#69).
    #[test]
    fn an_atx_heading_wins_over_a_promoted_one_of_the_same_name() {
        let content = "## Stat Block\n\n**History**\n\nBold body\n\n### History\n\nAtx body\n";
        assert!(
            find_section(content, "History")
                .unwrap()
                .content
                .contains("Atx body")
        );
    }

    /// A promoted section ends where #44 says it does: at the next promoted
    /// line, or at the next `#` heading of any depth (#69).
    #[test]
    fn a_promoted_section_ends_at_the_next_promotion_or_heading() {
        let content =
            "## Stat Block\n\n**Abilities**\n\nFlight\n\n**Spells**\n\nFireball\n\n# Lore\n\nOld\n";
        let abilities = find_section(content, "Abilities").unwrap();
        assert!(abilities.content.contains("Flight"));
        assert!(!abilities.content.contains("Fireball"));
        let spells = find_section(content, "Spells").unwrap();
        assert!(spells.content.contains("Fireball"));
        assert!(!spells.content.contains("Old"));
    }

    /// An empty section is addressable, because addressing it is how a caller
    /// fills it. The chunker drops a bodyless promoted line from its own set;
    /// the resolver reads the set before that drop (#69).
    #[test]
    fn a_bodyless_promoted_line_is_addressable() {
        let content = "## Stat Block\n\n**Spells**\n**Notes**\n\nSee below\n";
        let spells = find_section(content, "Spells").unwrap();
        assert_eq!(spells.content, "");
        assert_eq!(spells.body_start, spells.body_end);
    }

    /// A bold-only line inside a fence is not a section on either pass (#69).
    #[test]
    fn a_bold_line_inside_a_fence_is_not_addressable() {
        let content = "## Stat Block\n\n```md\n**Spells**\n\nFireball\n```\n";
        assert!(find_section(content, "Spells").is_none());
    }
}
