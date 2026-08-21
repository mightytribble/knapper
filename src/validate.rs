//! Markdown validation (#70): structural and indexing-quality checks over a
//! vault's `.md` files, read from disk. Read-only, no model, no store.

use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

/// One kind of finding. Serializes to its kebab-case name (the stable `rule`
/// code a JSON consumer and the strict gate key on), and owns its severity so
/// a check never assigns one by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Rule {
    UnterminatedCodeFence,
    MalformedFrontmatter,
    MalformedWikilink,
    UnresolvableWikilink,
    DuplicateSiblingHeadings,
    EmptySection,
    MissingTitle,
    MultipleTitles,
    ShortSection,
    LongParagraph,
    MalformedTags,
    FileUnreadable,
}

impl Rule {
    pub fn severity(self) -> Severity {
        use Rule::*;
        match self {
            UnterminatedCodeFence | MalformedFrontmatter | MalformedWikilink | FileUnreadable => {
                Severity::Error
            }
            _ => Severity::Warning,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub file: String,
    pub line: Option<usize>,
    pub severity: Severity,
    pub rule: Rule,
    pub message: String,
}

impl Finding {
    pub fn new(file: &str, line: Option<usize>, rule: Rule, message: String) -> Self {
        Finding {
            file: file.to_string(),
            line,
            severity: rule.severity(),
            rule,
            message,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidateReport {
    pub findings: Vec<Finding>,
    pub files_checked: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub ok: bool,
}

impl ValidateReport {
    pub fn build(findings: Vec<Finding>, files_checked: usize, strict: bool) -> Self {
        let error_count = findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count();
        let warning_count = findings.len() - error_count;
        let ok = error_count == 0 && (!strict || warning_count == 0);
        ValidateReport {
            findings,
            files_checked,
            error_count,
            warning_count,
            ok,
        }
    }
}

/// The vault's file names, for resolving a wikilink the way the indexer does:
/// by exact relative path, then by a `/`-suffix basename match, shortest path
/// winning (`indexer::resolve_link_target`). No aliases, no fuzzy matching.
/// The map is the cache — a lookup is O(1) in the basename bucket.
pub struct NameSet {
    /// last path segment (lowercased, with `.md`) -> the full relative paths
    /// (lowercased) that end in it.
    by_basename: HashMap<String, Vec<String>>,
}

impl NameSet {
    pub fn empty() -> Self {
        NameSet {
            by_basename: HashMap::new(),
        }
    }

    pub fn from_paths(paths: impl IntoIterator<Item = String>) -> Self {
        let mut by_basename: HashMap<String, Vec<String>> = HashMap::new();
        for p in paths {
            let lower = p.to_lowercase();
            let base = last_segment(&lower).to_string();
            by_basename.entry(base).or_default().push(lower);
        }
        NameSet { by_basename }
    }

    pub fn resolve(&self, target: &str) -> bool {
        let note = note_part(target);
        if note.is_empty() {
            return false;
        }
        let with_ext = if note.ends_with(".md") {
            note.to_lowercase()
        } else {
            format!("{}.md", note.to_lowercase())
        };
        let base = last_segment(&with_ext).to_string();
        match self.by_basename.get(&base) {
            Some(candidates) => candidates
                .iter()
                .any(|p| *p == with_ext || p.ends_with(&format!("/{with_ext}"))),
            None => false,
        }
    }
}

/// The note part of a wikilink target: the text before the first `#` or `|`.
pub fn note_part(target: &str) -> &str {
    let end = target.find(['#', '|']).unwrap_or(target.len());
    target[..end].trim()
}

fn last_segment(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Build the whole-vault name set from disk, respecting `.gitignore`.
pub fn build_name_set(root: &Path) -> anyhow::Result<NameSet> {
    let files = crate::indexer::walk_vault(root, &[], true)?;
    Ok(NameSet::from_paths(
        files.iter().filter_map(|p| rel_path(root, p)),
    ))
}

/// A file's vault-relative path with forward slashes, or `None` if it is not
/// under `root`.
fn rel_path(root: &Path, p: &Path) -> Option<String> {
    p.strip_prefix(root)
        .ok()
        .map(|rel| rel.to_string_lossy().replace('\\', "/"))
}

fn is_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

fn check_code_fences(file: &str, content: &str) -> Vec<Finding> {
    let mut open_line: Option<usize> = None;
    for (i, line) in content.lines().enumerate() {
        if is_fence(line) {
            open_line = match open_line {
                Some(_) => None,
                None => Some(i),
            };
        }
    }
    match open_line {
        Some(i) => vec![Finding::new(
            file,
            Some(i + 1),
            Rule::UnterminatedCodeFence,
            "code fence opened here is never closed".into(),
        )],
        None => vec![],
    }
}

fn check_wikilink_brackets(file: &str, content: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, line) in content.lines().enumerate() {
        if is_fence(line) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let mut rest = line;
        while let Some(open) = rest.find("[[") {
            let after = &rest[open + 2..];
            match after.find("]]") {
                Some(close) => rest = &after[close + 2..],
                None => {
                    out.push(Finding::new(
                        file,
                        Some(i + 1),
                        Rule::MalformedWikilink,
                        "wikilink opened with `[[` is not closed with `]]`".into(),
                    ));
                    break;
                }
            }
        }
    }
    out
}

fn check_frontmatter(file: &str, content: &str) -> Vec<Finding> {
    if content.lines().next().map(str::trim) != Some("---") {
        return vec![]; // no frontmatter block
    }
    let (fm, _body) = crate::markdown::split_frontmatter(content);
    match fm {
        None => vec![Finding::new(
            file,
            Some(1),
            Rule::MalformedFrontmatter,
            "frontmatter opened with `---` is never closed".into(),
        )],
        Some(fm) => match serde_yaml::from_str::<serde_yaml::Value>(&fm) {
            Ok(_) => vec![],
            Err(e) => vec![Finding::new(
                file,
                Some(1),
                Rule::MalformedFrontmatter,
                format!("frontmatter is not valid YAML: {e}"),
            )],
        },
    }
}

fn check_titles(file: &str, content: &str) -> Vec<Finding> {
    let headings = crate::markdown::parse_headings(content);
    let mut out = Vec::new();
    match headings.first() {
        None => out.push(Finding::new(
            file,
            None,
            Rule::MissingTitle,
            "file has no heading; expected a level-1 `# Title`".into(),
        )),
        Some(h) if h.level != 1 => out.push(Finding::new(
            file,
            Some(h.line + 1),
            Rule::MissingTitle,
            "file does not open with a level-1 `# Title`".into(),
        )),
        _ => {}
    }
    let mut seen_first = false;
    for h in headings.iter().filter(|h| h.level == 1) {
        if seen_first {
            out.push(Finding::new(
                file,
                Some(h.line + 1),
                Rule::MultipleTitles,
                "a second level-1 `#` heading; a file should have one title".into(),
            ));
        }
        seen_first = true;
    }
    out
}

fn check_duplicate_siblings(file: &str, content: &str) -> Vec<Finding> {
    let headings = crate::markdown::parse_headings(content);
    let mut out = Vec::new();
    let mut stack: Vec<(u8, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for h in &headings {
        while stack.last().is_some_and(|(l, _)| *l >= h.level) {
            stack.pop();
        }
        if h.level >= 2 {
            let parent: Vec<&str> = stack.iter().map(|(_, t)| t.as_str()).collect();
            let key = format!("{}\u{0}{}", parent.join(">"), h.text.to_lowercase());
            if !seen.insert(key) {
                out.push(Finding::new(
                    file,
                    Some(h.line + 1),
                    Rule::DuplicateSiblingHeadings,
                    format!(
                        "duplicate sibling heading `{}` under the same parent",
                        h.text
                    ),
                ));
            }
        }
        stack.push((h.level, h.text.to_lowercase()));
    }
    out
}

fn check_empty_sections(file: &str, content: &str) -> Vec<Finding> {
    let headings = crate::markdown::parse_headings(content);
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    for (idx, h) in headings.iter().enumerate() {
        let body_start = h.line + 1;
        let body_end = headings.get(idx + 1).map(|n| n.line).unwrap_or(lines.len());
        let has_body = lines[body_start..body_end]
            .iter()
            .any(|l| !l.trim().is_empty());
        if !has_body {
            out.push(Finding::new(
                file,
                Some(h.line + 1),
                Rule::EmptySection,
                format!(
                    "heading `{}` has no body text before the next heading",
                    h.text
                ),
            ));
        }
    }
    out
}

fn check_short_sections(file: &str, content: &str, min_chars: usize) -> Vec<Finding> {
    if min_chars == 0 {
        return vec![];
    }
    let headings = crate::markdown::parse_headings(content);
    let lines: Vec<&str> = content.lines().collect();
    let mut out = Vec::new();
    for (idx, h) in headings.iter().enumerate() {
        let body_start = h.line + 1;
        let body_end = headings.get(idx + 1).map(|n| n.line).unwrap_or(lines.len());
        let body = lines[body_start..body_end].join("\n");
        let trimmed = body.trim();
        let len = trimmed.chars().count();
        if !trimmed.is_empty() && len < min_chars {
            out.push(Finding::new(
                file,
                Some(h.line + 1),
                Rule::ShortSection,
                format!(
                    "section `{}` body is {len} chars, under the {min_chars}-char minimum; it will merge into the previous chunk",
                    h.text
                ),
            ));
        }
    }
    out
}

/// True when a block's first line opens a list, table, quote, heading, or is
/// a fence — none of which is a paragraph.
fn is_non_paragraph(first_line: &str) -> bool {
    let t = first_line.trim_start();
    t.starts_with('#')
        || t.starts_with('|')
        || t.starts_with('-')
        || t.starts_with('*')
        || t.starts_with('+')
        || t.starts_with('>')
        || is_fence(t)
        || is_ordered_list_item(t)
}

fn is_ordered_list_item(s: &str) -> bool {
    let digits: usize = s.chars().take_while(|c| c.is_ascii_digit()).count();
    digits > 0 && s[digits..].starts_with(['.', ')'])
}

fn check_long_paragraphs(file: &str, content: &str, max_chars: usize) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut block: Vec<&str> = Vec::new();
    let mut block_start = 0usize; // 1-based start line of the current block

    let flush = |block: &mut Vec<&str>, start: usize, out: &mut Vec<Finding>| {
        if !block.is_empty() {
            if !is_non_paragraph(block[0]) {
                let text = block.join("\n");
                if text.chars().count() > max_chars {
                    out.push(Finding::new(
                        file,
                        Some(start),
                        Rule::LongParagraph,
                        format!(
                            "paragraph is {} chars, over the {max_chars}-char ceiling; it fills a whole chunk and may be a semantic run-on",
                            text.chars().count()
                        ),
                    ));
                }
            }
            block.clear();
        }
    };

    for (i, line) in content.lines().enumerate() {
        if is_fence(line) {
            flush(&mut block, block_start, &mut out);
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if line.trim().is_empty() {
            flush(&mut block, block_start, &mut out);
        } else {
            if block.is_empty() {
                block_start = i + 1;
            }
            block.push(line);
        }
    }
    flush(&mut block, block_start, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_counts_and_gate() {
        let with_error = vec![
            Finding::new("a.md", Some(1), Rule::UnterminatedCodeFence, "x".into()),
            Finding::new("a.md", Some(2), Rule::MissingTitle, "y".into()),
        ];
        let r = ValidateReport::build(with_error, 1, false);
        assert_eq!(r.error_count, 1);
        assert_eq!(r.warning_count, 1);
        assert!(!r.ok, "an error gates even without strict");

        let warn = || vec![Finding::new("a.md", None, Rule::MissingTitle, "y".into())];
        assert!(
            ValidateReport::build(warn(), 1, false).ok,
            "a warning does not gate"
        );
        assert!(
            !ValidateReport::build(warn(), 1, true).ok,
            "strict gates on a warning"
        );
    }

    #[test]
    fn nameset_resolves_like_the_indexer() {
        let names = NameSet::from_paths([
            "Lore/Archdragon.md".to_string(),
            "People/Ada.md".to_string(),
            "sub/foo.md".to_string(),
            "foo.md".to_string(),
        ]);
        // bare basename, any case
        assert!(names.resolve("archdragon"));
        assert!(names.resolve("Archdragon"));
        // note part only: heading and alias fragments are ignored
        assert!(names.resolve("Ada#Bio"));
        assert!(names.resolve("Ada|Ada Lovelace"));
        // a path target requires that path suffix
        assert!(names.resolve("sub/foo"));
        // an unknown target does not resolve
        assert!(!names.resolve("missing"));
        assert!(!names.resolve(""));
    }

    #[test]
    fn build_name_set_reads_disk_not_a_store() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Lore")).unwrap();
        std::fs::write(dir.path().join("Lore/Archdragon.md"), "# Archdragon\n").unwrap();
        // A file never indexed still resolves, because resolution is disk-truth.
        let names = build_name_set(dir.path()).unwrap();
        assert!(names.resolve("Archdragon"));
        assert!(!names.resolve("Nonexistent"));
    }

    #[test]
    fn error_checks_flag_malformed_structure() {
        let f = "note.md";
        // unterminated fence
        let r = check_code_fences(f, "# T\n\n```rust\nlet x = 1;\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::UnterminatedCodeFence);
        assert_eq!(r[0].line, Some(3));
        // balanced fence: nothing
        assert!(check_code_fences(f, "```\ncode\n```\n").is_empty());

        // unclosed wikilink
        let r = check_wikilink_brackets(f, "see [[Note\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::MalformedWikilink);
        // closed wikilink: nothing
        assert!(check_wikilink_brackets(f, "see [[Note]] and [[Other]]\n").is_empty());

        // unclosed frontmatter
        let r = check_frontmatter(f, "---\ntitle: T\n# no close\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::MalformedFrontmatter);
        // invalid YAML
        let r = check_frontmatter(f, "---\ntitle: : :\n---\n# T\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::MalformedFrontmatter);
        // valid frontmatter, and no frontmatter: nothing
        assert!(check_frontmatter(f, "---\ntitle: T\n---\n# T\n").is_empty());
        assert!(check_frontmatter(f, "# T\n\nbody\n").is_empty());
    }

    #[test]
    fn heading_checks() {
        let f = "n.md";

        // missing title: no heading, then a non-level-1 first heading
        assert_eq!(check_titles(f, "just body\n")[0].rule, Rule::MissingTitle);
        assert_eq!(check_titles(f, "## Sub\n\nx\n")[0].rule, Rule::MissingTitle);
        // multiple titles: the second `#` is flagged
        let r = check_titles(f, "# One\n\nx\n\n# Two\n\ny\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::MultipleTitles);
        assert_eq!(r[0].line, Some(5));
        // a single title, first: nothing
        assert!(check_titles(f, "# Title\n\nbody\n").is_empty());

        // duplicate siblings under one parent (level 2+)
        let r = check_duplicate_siblings(f, "# A\n\n## Notes\n\nx\n\n## Notes\n\ny\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::DuplicateSiblingHeadings);
        assert_eq!(r[0].line, Some(7));
        // same text under different parents: nothing
        assert!(
            check_duplicate_siblings(
                f,
                "# A\n\n## H\n\n### Notes\n\nx\n\n## B\n\n### Notes\n\ny\n"
            )
            .is_empty()
        );
        // level-1 repeats are multiple-titles' job, not this check's
        assert!(check_duplicate_siblings(f, "# A\n\nx\n\n# A\n\ny\n").is_empty());

        // empty sections (the three issue #70 shapes)
        let empties = check_empty_sections(f, "# H1\n## H2\n\nbody\n");
        assert!(
            empties
                .iter()
                .any(|x| x.line == Some(1) && x.rule == Rule::EmptySection)
        );
        let empties = check_empty_sections(f, "### H3\n## H2\n\nbody\n");
        assert!(empties.iter().any(|x| x.line == Some(1)));
        let empties = check_empty_sections(f, "## Nobody\n## Body\n\nstuff\n");
        assert!(empties.iter().any(|x| x.line == Some(1)));
        // a heading with body: not empty
        assert!(check_empty_sections(f, "# T\n\nbody\n").is_empty());
    }

    #[test]
    fn length_checks() {
        let f = "n.md";
        // a section body under the minimum is flagged; an empty body is not (that
        // is empty-section's job)
        let short = check_short_sections(f, "# T\n\ntiny\n", 120);
        assert_eq!(short.len(), 1);
        assert_eq!(short[0].rule, Rule::ShortSection);
        assert!(check_short_sections(f, "# T\n\n", 120).is_empty());
        // a body over the minimum is not flagged
        let big = "x".repeat(200);
        assert!(check_short_sections(f, &format!("# T\n\n{big}\n"), 120).is_empty());
        // min_chars 0 disables the check
        assert!(check_short_sections(f, "# T\n\ntiny\n", 0).is_empty());

        // a paragraph over the ceiling is flagged
        let huge = "word ".repeat(500); // ~2500 chars, one block
        let r = check_long_paragraphs(f, &format!("# T\n\n{huge}\n"), 2048);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rule, Rule::LongParagraph);
        // a long list or table is not one paragraph
        let list = (0..300)
            .map(|i| format!("- item {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(check_long_paragraphs(f, &format!("# T\n\n{list}\n"), 2048).is_empty());
        let table = (0..300)
            .map(|i| format!("| {i} | x |"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(check_long_paragraphs(f, &format!("# T\n\n{table}\n"), 2048).is_empty());
    }
}
