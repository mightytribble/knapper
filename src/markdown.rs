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

pub fn find_section(content: &str, heading_text: &str) -> Option<Section> {
    let headings = parse_headings(content);
    let target = heading_text.trim().to_lowercase();
    let lines: Vec<&str> = content.lines().collect();

    let idx = headings
        .iter()
        .position(|h| h.text.to_lowercase() == target)?;
    let h = &headings[idx];
    let body_start = h.line + 1;
    let body_end = headings[idx + 1..]
        .iter()
        .find(|next| next.level <= h.level)
        .map(|next| next.line)
        .unwrap_or(lines.len());

    let content_str = lines[body_start..body_end].join("\n");
    Some(Section {
        heading: HeadingInfo {
            line: h.line,
            level: h.level,
            text: h.text.clone(),
            promoted: h.promoted,
        },
        body_start,
        body_end,
        content: content_str,
    })
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
}
