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
}
