//! Literal string matching over the indexed chunk text (#106).
//!
//! `search` is ranked, budgeted and cut to `top_n`, so it always answers
//! something and cannot report that a string is **absent**. This module is the
//! other contract: one literal pattern, every note the scope admits, no
//! ranking, and a count that comes back whole even when the reported lines are
//! capped. It is for verification and maintenance — "prove nothing still says
//! X" — and not for discovery, which is what `search` is for.

use std::collections::HashSet;

/// One chunk's text, with what addresses it.
///
/// The scan reads `chunks.text`, so it sees a note's body and not its
/// frontmatter: the chunker strips the YAML block before it cuts a file into
/// sections. That is the declared limit of the capability.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub file: String,
    pub heading_path: Option<String>,
    pub text: String,
}

/// One literal pattern and how it compares.
#[derive(Debug, Clone)]
pub struct Query {
    pattern: String,
    case_sensitive: bool,
    /// The pattern folded once, for the insensitive comparison.
    folded: String,
}

impl Query {
    pub fn new(pattern: &str, case_sensitive: bool) -> Self {
        Query {
            pattern: pattern.to_string(),
            case_sensitive,
            folded: pattern.to_lowercase(),
        }
    }

    /// Whether the haystack holds the pattern. The pattern is a literal, so
    /// `.` and `*` are themselves and nothing else.
    fn matches(&self, haystack: &str) -> bool {
        if self.case_sensitive {
            haystack.contains(&self.pattern)
        } else {
            haystack.to_lowercase().contains(&self.folded)
        }
    }
}

/// One matched line.
///
/// The line is the text as the note wrote it, not a file line number: a chunk
/// row carries no offset into its file, and reading one would mean a vault
/// read this capability does not do. `heading_path` is what `read --section`
/// and `update --section` address, so a hit is actionable as it stands.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Hit {
    pub file: String,
    pub heading_path: Option<String>,
    pub line: String,
}

/// What a `match` call answers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MatchReport {
    /// The pattern as the caller wrote it.
    pub pattern: String,
    /// Notes holding at least one match. This is the answer to the absence
    /// question, and `limit` never truncates it.
    pub notes: usize,
    /// Distinct matched lines across every note, likewise never truncated.
    pub lines: usize,
    /// The matched lines, capped by `limit`.
    pub hits: Vec<Hit>,
}

/// Accumulates matches one chunk row at a time.
///
/// A visitor rather than a returned list, so the store can stream the chunk
/// table past it: the scan is exhaustive by contract, so it reads every row in
/// scope whatever the limit, and materializing the vault's text to do that
/// would be waste.
pub struct Scanner {
    query: Query,
    limit: Option<usize>,
    notes: HashSet<String>,
    /// Distinct `(file, heading path, line)` triples seen. The dedup is what
    /// keeps an oversized-chunk split from reporting one line twice:
    /// `split_oversized_chunks` repeats `OVERLAP_TOKENS` of the previous piece
    /// at the head of the next and gives both the same `heading_path`, so the
    /// repeated lines arrive as two rows of one section.
    seen: HashSet<(String, Option<String>, String)>,
    hits: Vec<Hit>,
}

impl Scanner {
    pub fn new(query: Query, limit: Option<usize>) -> Self {
        Scanner {
            query,
            limit,
            notes: HashSet::new(),
            seen: HashSet::new(),
            hits: Vec::new(),
        }
    }

    /// Read one chunk row, recording every line of it that holds the pattern.
    pub fn push(&mut self, row: ChunkRow) {
        for line in row.text.lines() {
            if !self.query.matches(line) {
                continue;
            }
            let key = (row.file.clone(), row.heading_path.clone(), line.to_string());
            if !self.seen.insert(key) {
                continue;
            }
            self.notes.insert(row.file.clone());
            if self.limit.is_none_or(|n| self.hits.len() < n) {
                self.hits.push(Hit {
                    file: row.file.clone(),
                    heading_path: row.heading_path.clone(),
                    line: line.to_string(),
                });
            }
        }
    }

    pub fn finish(self) -> MatchReport {
        MatchReport {
            pattern: self.query.pattern,
            notes: self.notes.len(),
            lines: self.seen.len(),
            hits: self.hits,
        }
    }
}

/// Run one `match` call against the index.
///
/// The one path all three surfaces take, so the contract cannot differ
/// between them.
pub fn run(
    store: &crate::store::Store,
    params: &crate::params::Match,
) -> anyhow::Result<MatchReport> {
    if params.pattern.is_empty() {
        anyhow::bail!("pattern is empty: every line holds the empty string");
    }
    let scope = params.scope()?;
    let mut scanner = Scanner::new(
        Query::new(&params.pattern, params.case_sensitive),
        params.limit,
    );
    store.for_each_chunk_in_scope(&scope, |row| scanner.push(row))?;
    Ok(scanner.finish())
}

/// Render a report as text.
///
/// The counts lead, because the question is how many — and most often
/// whether the answer is none. `lines` is what the vault holds and `hits` is
/// what the limit let through, so the difference is stated rather than left
/// to the reader's arithmetic.
pub fn render_text(report: &MatchReport) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    if report.notes == 0 {
        let _ = writeln!(out, "No note holds \"{}\".", report.pattern);
        return out;
    }
    let notes = if report.notes == 1 { "note" } else { "notes" };
    let lines = if report.lines == 1 { "line" } else { "lines" };
    let _ = writeln!(
        out,
        "{} {notes}, {} {lines} hold \"{}\".",
        report.notes, report.lines, report.pattern
    );
    for hit in &report.hits {
        // A chunk under no heading takes the note's own name as its whole
        // breadcrumb — `[breadcrumb_root] = path` (#46) — so printing both
        // would say the same thing twice.
        match &hit.heading_path {
            Some(path) if path != &hit.file => {
                let _ = writeln!(out, "  {} [{path}]", hit.file);
            }
            _ => {
                let _ = writeln!(out, "  {}", hit.file);
            }
        }
        let _ = writeln!(out, "    {}", hit.line.trim());
    }
    if report.hits.len() < report.lines {
        let _ = writeln!(
            out,
            "  ({} more not shown; raise --limit to see them)",
            report.lines - report.hits.len()
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(file: &str, heading: Option<&str>, text: &str) -> ChunkRow {
        ChunkRow {
            file: file.to_string(),
            heading_path: heading.map(str::to_string),
            text: text.to_string(),
        }
    }

    fn scan(rows: Vec<ChunkRow>, pattern: &str, case_sensitive: bool) -> MatchReport {
        let mut scanner = Scanner::new(Query::new(pattern, case_sensitive), None);
        for r in rows {
            scanner.push(r);
        }
        scanner.finish()
    }

    /// A store holding two notes, one chunk each.
    fn seeded_store() -> crate::store::Store {
        let store = crate::store::Store::open_memory().unwrap();
        for (i, (path, text)) in [
            (
                "People/ilse.md",
                "She is sixteen years old at the start of the story.",
            ),
            ("Places/academy.md", "The Academy is in Kessel."),
        ]
        .iter()
        .enumerate()
        {
            let file_id = store
                .insert_file(path, "h", i as i64, &format!("d00000{i}"), None, None)
                .unwrap();
            store
                .insert_chunk(&crate::store::NewChunk {
                    file_id,
                    seq: 0,
                    heading: "Biography",
                    heading_path: "Biography",
                    tags_text: "",
                    text,
                    vector_id: i as u64,
                    token_count: 10,
                })
                .unwrap();
        }
        store
    }

    fn params(pattern: &str) -> crate::params::Match {
        crate::params::Match {
            pattern: pattern.to_string(),
            case_sensitive: false,
            scope: Vec::new(),
            all: Vec::new(),
            any: Vec::new(),
            none: Vec::new(),
            limit: None,
        }
    }

    #[test]
    fn a_breadcrumb_that_repeats_the_note_name_is_not_printed_beside_it() {
        // A chunk under no heading takes the note's own name as its whole
        // breadcrumb — `[breadcrumb_root] = path` (#46) — so printing both
        // says the same thing twice.
        let report = scan(
            vec![row(
                "characters/Barnaby Finch.md",
                Some("characters/Barnaby Finch.md"),
                "A noble student at the Royal Academy.",
            )],
            "Royal Academy",
            false,
        );

        let text = render_text(&report);
        assert!(
            text.contains("  characters/Barnaby Finch.md\n"),
            "the note should be named once: {text}"
        );
        assert!(
            !text.contains('['),
            "no breadcrumb should be printed: {text}"
        );
    }

    #[test]
    fn a_breadcrumb_that_names_a_section_is_printed() {
        let report = scan(
            vec![row(
                "characters/Barnaby Finch.md",
                Some("characters/Barnaby Finch.md > Biography"),
                "A noble student at the Royal Academy.",
            )],
            "Royal Academy",
            false,
        );

        assert!(render_text(&report).contains("[characters/Barnaby Finch.md > Biography]"));
    }

    #[test]
    fn a_report_with_no_notes_renders_as_absence() {
        let report = scan(vec![row("a.md", None, "nothing")], "gone", false);
        assert_eq!(render_text(&report), "No note holds \"gone\".\n");
    }

    #[test]
    fn a_capped_report_says_how_many_it_did_not_print() {
        let mut scanner = Scanner::new(Query::new("Kessel", false), Some(1));
        scanner.push(row("a.md", None, "Kessel"));
        scanner.push(row("b.md", None, "Kessel"));
        scanner.push(row("c.md", None, "Kessel"));

        assert!(render_text(&scanner.finish()).contains("(2 more not shown"));
    }

    #[test]
    fn a_run_answers_the_notes_the_index_holds_the_pattern_in() {
        let report = run(&seeded_store(), &params("years old at the start")).unwrap();

        assert_eq!(report.notes, 1);
        assert_eq!(report.hits[0].file, "People/ilse.md");
    }

    #[test]
    fn a_run_answers_zero_notes_for_a_pattern_the_index_does_not_hold() {
        let report = run(&seeded_store(), &params("years old at the start")).unwrap();
        assert_eq!(report.notes, 1, "fixture check");

        let gone = run(&seeded_store(), &params("no note says this")).unwrap();
        assert_eq!(gone.notes, 0);
        assert!(gone.hits.is_empty());
    }

    #[test]
    fn a_run_looks_only_where_its_scope_admits() {
        assert_eq!(
            run(&seeded_store(), &params("is")).unwrap().notes,
            2,
            "fixture check: both notes hold the pattern"
        );

        let mut p = params("is");
        p.scope = vec!["/Places/".to_string()];
        let report = run(&seeded_store(), &p).unwrap();

        assert_eq!(report.notes, 1);
        assert_eq!(report.hits[0].file, "Places/academy.md");
    }

    #[test]
    fn an_empty_pattern_is_refused() {
        // Every line holds the empty string, so an empty pattern answers the
        // whole vault and says nothing. It is the caller's own mistake.
        let err = run(&seeded_store(), &params("")).unwrap_err();
        assert!(
            err.to_string().contains("pattern"),
            "the error should name the pattern: {err}"
        );
    }

    #[test]
    fn a_pattern_the_vault_holds_is_found_in_the_note_that_holds_it() {
        let report = scan(
            vec![row(
                "People/Ilse.md",
                Some("Ilse > Biography"),
                "She is sixteen years old at the start of the story.",
            )],
            "years old at the start of the story",
            false,
        );

        assert_eq!(report.notes, 1);
        assert_eq!(report.lines, 1);
        assert_eq!(
            report.hits,
            vec![Hit {
                file: "People/Ilse.md".to_string(),
                heading_path: Some("Ilse > Biography".to_string()),
                line: "She is sixteen years old at the start of the story.".to_string(),
            }]
        );
    }

    #[test]
    fn the_default_comparison_folds_case() {
        let report = scan(
            vec![row("Lore/Reckoning.md", None, "The year is 197 AR.")],
            "197 ar",
            false,
        );

        assert_eq!(report.notes, 1);
    }

    #[test]
    fn a_case_sensitive_pattern_refuses_a_different_case() {
        let report = scan(
            vec![row("Lore/Reckoning.md", None, "The year is 197 AR.")],
            "197 ar",
            true,
        );

        assert_eq!(report.notes, 0);
    }

    #[test]
    fn a_pattern_is_a_literal_and_not_a_regex() {
        let report = scan(
            vec![row("Lore/Reckoning.md", None, "The year is 197 AR.")],
            "197.AR",
            false,
        );

        assert_eq!(report.notes, 0, "`.` matched a character it should not");
    }

    #[test]
    fn two_matched_lines_in_one_note_count_one_note() {
        let report = scan(
            vec![
                row(
                    "People/Ilse.md",
                    Some("Ilse > Biography"),
                    "Sixteen years old at the start of the story.",
                ),
                row(
                    "People/Ilse.md",
                    Some("Ilse > Schooling"),
                    "She was fifteen years old at the start of the story.",
                ),
            ],
            "years old at the start of the story",
            false,
        );

        assert_eq!(report.notes, 1);
        assert_eq!(report.lines, 2);
    }

    #[test]
    fn a_line_an_oversized_chunk_split_repeated_is_reported_once() {
        // `split_oversized_chunks` repeats `OVERLAP_TOKENS` of the previous
        // piece at the head of the next, and gives both pieces the same
        // heading path. The repeated line is one line of the note.
        let overlapped = "She is sixteen years old at the start of the story.";
        let report = scan(
            vec![
                row(
                    "People/Ilse.md",
                    Some("Ilse > Biography"),
                    &format!("Born in Kessel.\n{overlapped}"),
                ),
                row(
                    "People/Ilse.md",
                    Some("Ilse > Biography"),
                    &format!("{overlapped}\nShe reads widely."),
                ),
            ],
            "years old at the start of the story",
            false,
        );

        assert_eq!(report.lines, 1);
        assert_eq!(report.hits.len(), 1);
    }

    #[test]
    fn the_counts_are_whole_when_the_limit_caps_the_hits() {
        let mut scanner = Scanner::new(Query::new("Kessel", false), Some(1));
        scanner.push(row("People/Ilse.md", None, "Born in Kessel."));
        scanner.push(row("People/Rolf.md", None, "Also of Kessel."));
        scanner.push(row("Places/Kessel.md", None, "Kessel is a river town."));
        let report = scanner.finish();

        assert_eq!(report.notes, 3);
        assert_eq!(report.lines, 3);
        assert_eq!(report.hits.len(), 1);
    }

    #[test]
    fn a_limit_of_zero_answers_the_counts_and_no_hits() {
        let mut scanner = Scanner::new(Query::new("Kessel", false), Some(0));
        scanner.push(row("People/Ilse.md", None, "Born in Kessel."));
        let report = scanner.finish();

        assert_eq!(report.notes, 1);
        assert_eq!(report.lines, 1);
        assert!(report.hits.is_empty());
    }

    #[test]
    fn a_pattern_no_note_holds_answers_zero_notes_and_no_hits() {
        let report = scan(
            vec![row(
                "People/Ilse.md",
                None,
                "She is sixteen at the Academy.",
            )],
            "years old at the start of the story",
            false,
        );

        assert_eq!(report.notes, 0);
        assert_eq!(report.lines, 0);
        assert!(report.hits.is_empty());
    }
}
