//! Markdown validation (#70): structural and indexing-quality checks over a
//! vault's `.md` files, read from disk. Read-only, no model, no store.

use serde::Serialize;

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
}
