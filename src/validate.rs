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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
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

fn check_tags(file: &str, content: &str) -> Vec<Finding> {
    let (fm, _) = crate::markdown::split_frontmatter(content);
    let Some(fm) = fm else {
        return vec![];
    };
    let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(&fm) else {
        return vec![]; // malformed YAML is check_frontmatter's finding, not this one
    };
    let Some(tags) = value.get("tags") else {
        return vec![];
    };
    let ok = match tags {
        serde_yaml::Value::String(_) | serde_yaml::Value::Null => true,
        serde_yaml::Value::Sequence(seq) => seq.iter().all(serde_yaml::Value::is_string),
        _ => false,
    };
    if ok {
        vec![]
    } else {
        vec![Finding::new(
            file,
            Some(1),
            Rule::MalformedTags,
            "frontmatter `tags:` must be a string or a list of strings".into(),
        )]
    }
}

fn check_wikilink_targets(file: &str, content: &str, names: &NameSet) -> Vec<Finding> {
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
        for target in wikilink_targets(line) {
            if !names.resolve(&target) {
                out.push(Finding::new(
                    file,
                    Some(i + 1),
                    Rule::UnresolvableWikilink,
                    format!("wikilink `[[{target}]]` resolves to no file"),
                ));
            }
        }
    }
    out
}

/// Every closed `[[...]]` target on a line, inner text verbatim.
pub fn wikilink_targets(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find("[[") {
        let after = &rest[open + 2..];
        match after.find("]]") {
            Some(close) => {
                out.push(after[..close].to_string());
                rest = &after[close + 2..];
            }
            None => break,
        }
    }
    out
}

/// The chunker limits the length checks read: the section-body minimum and the
/// pack target, both in characters for the long-paragraph ceiling.
pub struct ChunkLimits {
    pub min_chars: usize,
    pub target_tokens: usize,
}

impl ChunkLimits {
    /// The long-paragraph ceiling: a block this long on its own fills a whole
    /// chunk's token budget (`TARGET_TOKENS * 4` chars, the chunker's own
    /// chars-per-token estimate).
    pub fn long_paragraph_chars(&self) -> usize {
        self.target_tokens * 4
    }
}

/// Run every check over one file's text.
pub fn validate_file(
    file: &str,
    content: &str,
    names: &NameSet,
    limits: &ChunkLimits,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    findings.extend(check_code_fences(file, content));
    findings.extend(check_frontmatter(file, content));
    findings.extend(check_wikilink_brackets(file, content));
    findings.extend(check_wikilink_targets(file, content, names));
    findings.extend(check_titles(file, content));
    findings.extend(check_duplicate_siblings(file, content));
    findings.extend(check_empty_sections(file, content));
    findings.extend(check_short_sections(file, content, limits.min_chars));
    findings.extend(check_long_paragraphs(
        file,
        content,
        limits.long_paragraph_chars(),
    ));
    findings.extend(check_tags(file, content));
    findings.sort_by_key(|f| (f.line.unwrap_or(0), f.rule as u8));
    findings
}

use std::path::PathBuf;

/// What to validate, all resolved under the vault root.
pub enum Target {
    Vault,
    Note(String),
    Scope(crate::tags::Scope),
}

/// Validate a target under `root`, disk-truth throughout.
pub fn validate_target(
    root: &Path,
    target: &Target,
    limits: &ChunkLimits,
    strict: bool,
) -> anyhow::Result<ValidateReport> {
    // A single note walks names lazily: only if it holds a wikilink.
    if let Target::Note(rel) = target {
        let path = resolve_note(root, rel)?;
        let content = std::fs::read_to_string(&path);
        let file = rel_path(root, &path).unwrap_or_else(|| rel.clone());
        return Ok(match content {
            Ok(text) => {
                let names = if text.contains("[[") {
                    build_name_set(root)?
                } else {
                    NameSet::empty()
                };
                ValidateReport::build(validate_file(&file, &text, &names, limits), 1, strict)
            }
            Err(e) => ValidateReport::build(
                vec![Finding::new(
                    &file,
                    None,
                    Rule::FileUnreadable,
                    format!("cannot read file: {e}"),
                )],
                0,
                strict,
            ),
        });
    }

    // Vault and scope share one walk, which also builds the name set.
    let all = crate::indexer::walk_vault(root, &[], true)?;
    let names = NameSet::from_paths(all.iter().filter_map(|p| rel_path(root, p)));
    let needs_tags = matches!(target, Target::Scope(s)
        if s.all.iter().chain(&s.any).chain(&s.none).any(|t| matches!(t, crate::tags::ScopeTerm::Tag(_))));

    let mut findings = Vec::new();
    let mut checked = 0usize;
    for path in &all {
        let rel = rel_path(root, path).unwrap_or_else(|| path.to_string_lossy().into_owned());
        if let Target::Scope(scope) = target {
            let content_tags = if needs_tags {
                match std::fs::read_to_string(path) {
                    Ok(text) => Some(crate::tags::extract(&text)),
                    Err(e) => {
                        findings.push(Finding::new(
                            &rel,
                            None,
                            Rule::FileUnreadable,
                            format!("cannot read file: {e}"),
                        ));
                        continue;
                    }
                }
            } else {
                None
            };
            if !scope_admits(&rel, content_tags.as_deref(), scope) {
                continue;
            }
        }
        findings.extend(check_path(path, &rel, &names, limits));
        checked += 1;
    }
    Ok(ValidateReport::build(findings, checked, strict))
}

/// Read one file and validate it, or report it unreadable.
pub fn check_path(path: &Path, rel: &str, names: &NameSet, limits: &ChunkLimits) -> Vec<Finding> {
    match std::fs::read_to_string(path) {
        Ok(text) => validate_file(rel, &text, names, limits),
        Err(e) => vec![Finding::new(
            rel,
            None,
            Rule::FileUnreadable,
            format!("cannot read file: {e}"),
        )],
    }
}

/// Resolve a vault-relative note reference to a path: exact relative path
/// first, then a case-insensitive basename over the walk, shortest path.
fn resolve_note(root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    let with_ext = if rel.ends_with(".md") {
        rel.to_string()
    } else {
        format!("{rel}.md")
    };
    let direct = root.join(&with_ext);
    if direct.is_file() {
        return Ok(direct);
    }
    let base = last_segment(&with_ext).to_lowercase();
    let mut matches: Vec<PathBuf> = crate::indexer::walk_vault(root, &[], true)?
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.to_lowercase() == base)
                .unwrap_or(false)
        })
        .collect();
    matches.sort_by_key(|p| p.as_os_str().len());
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no note matches '{rel}'"))
}

/// Whether a file, by its relative path and (when read) its tags, satisfies a
/// scope: every `all` term, at least one `any` term when `any` is non-empty,
/// and no `none` term. Mirrors `tags::Scope` semantics on disk.
fn scope_admits(rel: &str, tags: Option<&[crate::tags::Tag]>, scope: &crate::tags::Scope) -> bool {
    let matches = |term: &crate::tags::ScopeTerm| term_matches(rel, tags, term);
    scope.all.iter().all(&matches)
        && (scope.any.is_empty() || scope.any.iter().any(&matches))
        && !scope.none.iter().any(&matches)
}

fn term_matches(
    rel: &str,
    tags: Option<&[crate::tags::Tag]>,
    term: &crate::tags::ScopeTerm,
) -> bool {
    use crate::tags::{FolderTerm, ScopeTerm, TagTerm};
    match term {
        ScopeTerm::Folder(FolderTerm::Exact(d)) => parent_dir(rel) == d.as_str(),
        ScopeTerm::Folder(FolderTerm::Subtree(d)) => rel.starts_with(&format!("{d}/")),
        ScopeTerm::Tag(TagTerm::Exact(t)) => {
            tags.is_some_and(|ts| ts.iter().any(|tag| &tag.path == t))
        }
        ScopeTerm::Tag(TagTerm::Subtree(t)) => tags.is_some_and(|ts| {
            ts.iter()
                .any(|tag| &tag.path == t || tag.path.starts_with(&format!("{t}/")))
        }),
    }
}

fn parent_dir(rel: &str) -> &str {
    rel.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
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

    #[test]
    fn semantic_checks() {
        let f = "n.md";
        // tags as a string or a list: fine
        assert!(check_tags(f, "---\ntags: work\n---\n# T\n").is_empty());
        assert!(check_tags(f, "---\ntags:\n  - work\n  - x\n---\n# T\n").is_empty());
        // tags as a map or a number: warning
        assert_eq!(
            check_tags(f, "---\ntags:\n  a: b\n---\n# T\n")[0].rule,
            Rule::MalformedTags
        );
        assert_eq!(
            check_tags(f, "---\ntags: 3\n---\n# T\n")[0].rule,
            Rule::MalformedTags
        );

        // link resolution against a name set
        let names = NameSet::from_paths(["Real.md".to_string()]);
        assert!(check_wikilink_targets(f, "see [[Real]]\n", &names).is_empty());
        assert!(check_wikilink_targets(f, "see [[Real#Heading]]\n", &names).is_empty());
        let miss = check_wikilink_targets(f, "see [[Ghost]]\n", &names);
        assert_eq!(miss.len(), 1);
        assert_eq!(miss[0].rule, Rule::UnresolvableWikilink);
        // a link inside a fence is not checked
        assert!(check_wikilink_targets(f, "```\n[[Ghost]]\n```\n", &names).is_empty());
    }

    #[test]
    fn validate_file_runs_every_check_and_orders_by_line() {
        let names = NameSet::from_paths(["Real.md".to_string()]);
        let limits = ChunkLimits {
            min_chars: 120,
            target_tokens: 512,
        };
        let content = "## Sub only\n\nshort body\n\nsee [[Ghost]]\n";
        let findings = validate_file("n.md", content, &names, &limits);
        let rules: std::collections::HashSet<Rule> = findings.iter().map(|f| f.rule).collect();
        assert!(rules.contains(&Rule::MissingTitle));
        assert!(rules.contains(&Rule::UnresolvableWikilink));
        // findings are ordered by line
        let lines: Vec<usize> = findings.iter().map(|f| f.line.unwrap_or(0)).collect();
        let mut sorted = lines.clone();
        sorted.sort_unstable();
        assert_eq!(lines, sorted);

        // a clean file has no findings
        let clean = "# Title\n\nThis is a body long enough to clear the minimum chunk size, so no short-section warning fires and the note reads as one healthy block of prose that comfortably exceeds one hundred and twenty characters.\n";
        assert!(validate_file("c.md", clean, &names, &limits).is_empty());
    }

    #[test]
    fn validate_target_resolves_note_scope_and_vault() {
        use crate::tags::Scope;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("Work")).unwrap();
        std::fs::write(
            root.join("Work/a.md"),
            "## Sub only\n\nsee [[home]] which is out of scope\n",
        )
        .unwrap();
        std::fs::write(root.join("home.md"), "# Home\n\nbody\n").unwrap();
        let limits = ChunkLimits {
            min_chars: 120,
            target_tokens: 512,
        };

        // whole vault: both files checked
        let r = validate_target(root, &Target::Vault, &limits, false).unwrap();
        assert_eq!(r.files_checked, 2);

        // one note by reference
        let r = validate_target(root, &Target::Note("Work/a.md".into()), &limits, false).unwrap();
        assert_eq!(r.files_checked, 1);

        // a directory scope checks only its subtree, but a link out of it resolves
        let scope = Scope::parse(&["/Work/".to_string()], &[], &[]).unwrap();
        let r = validate_target(root, &Target::Scope(scope), &limits, false).unwrap();
        assert_eq!(r.files_checked, 1);
        // Work/a.md's [[home]] link points out of the /Work/ scope, but it still
        // resolves: the name set is built from the whole vault, not the scoped
        // subtree. If the name set were scoped to /Work/, home.md would be absent
        // and [[home]] would flag UnresolvableWikilink, failing this assertion.
        assert!(
            r.findings
                .iter()
                .all(|f| f.rule != Rule::UnresolvableWikilink)
        );

        // a note reference that resolves to nothing is a hard error
        assert!(validate_target(root, &Target::Note("nope".into()), &limits, false).is_err());
    }

    #[test]
    fn check_path_reports_an_unreadable_file() {
        let names = NameSet::empty();
        let limits = ChunkLimits {
            min_chars: 120,
            target_tokens: 512,
        };
        let findings = check_path(
            std::path::Path::new("does/not/exist.md"),
            "exist.md",
            &names,
            &limits,
        );
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].rule, Rule::FileUnreadable);
    }
}
