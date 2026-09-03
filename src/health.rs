use anyhow::Result;

use crate::store::Store;

/// Full vault health report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthReport {
    pub orphans: Vec<String>,
    pub broken_links: Vec<BrokenLink>,
    pub stale_headings: Vec<StaleHeading>,
    pub stale_notes: Vec<String>,
    pub inbox_pending: Vec<String>,
    pub tag_issues: Vec<TagIssue>,
    pub index_age_seconds: u64,
    pub total_files: usize,
}

/// A wikilink that could not be resolved to any indexed file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrokenLink {
    pub source: String,
    pub target: String,
}

/// A wikilink whose note resolves but whose `#Heading` the note no longer
/// holds (#99).
///
/// Not a [`BrokenLink`]: the file is there, so nothing is unresolved and the
/// `unresolved_links` table never sees it. What breaks is quieter — the edge
/// degrades from the passage to the document on the linking note's next
/// re-index, so the graph lane expands a whole note where it used to expand
/// one section, and the link in the vault names a heading that is not there.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StaleHeading {
    /// The linking note's path.
    pub source: String,
    /// The linked note's path. It exists; only the heading does not.
    pub target: String,
    /// The `#Heading` the link named, as written.
    pub heading: String,
}

/// A tag-related problem in a file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagIssue {
    pub file: String,
    pub issue: String,
}

/// Configuration controlling which folders are excluded from health checks.
pub struct HealthConfig {
    pub daily_folder: Option<String>,
    pub inbox_folder: Option<String>,
}

/// Find files with no edges (neither incoming nor outgoing).
///
/// Excludes files whose path starts with the configured daily or inbox folder
/// prefixes — those are expected to be unlinked.
pub fn find_orphans(store: &Store, config: &HealthConfig) -> Result<Vec<String>> {
    let mut exclude = Vec::new();
    if let Some(ref daily) = config.daily_folder {
        exclude.push(daily.as_str());
    }
    if let Some(ref inbox) = config.inbox_folder {
        exclude.push(inbox.as_str());
    }
    let isolated = store.find_isolated_files(&exclude)?;
    Ok(isolated.into_iter().map(|f| f.path).collect())
}

/// Find wikilink references that could not be resolved to any indexed file.
///
/// These are recorded in the `unresolved_links` table during indexing.
pub fn find_broken_links(store: &Store) -> Result<Vec<BrokenLink>> {
    let unresolved = store.get_unresolved_links()?;
    Ok(unresolved
        .into_iter()
        .map(|(source, target)| BrokenLink { source, target })
        .collect())
}

/// Find wikilinks whose note resolves but whose `#Heading` it no longer holds.
///
/// The other end of #99's fact. A rename tells its caller what it broke, but
/// only for the rename knapper itself made; this finds the same thing from the
/// vault side, so a heading renamed in Obsidian is reported too, and no rename
/// has to have happened for it to be true.
///
/// Reads `chunks.text` for every note, which is a pass the vault-wide edge
/// backfill already makes in well under a second on a few hundred notes. A
/// link inside a code fence is counted — see [`crate::graph::deep_links_from`]
/// — so a note that documents the wikilink syntax reports one finding that is
/// not real.
pub fn find_stale_headings(store: &Store) -> Result<Vec<StaleHeading>> {
    let sources: Vec<i64> = store.get_all_files()?.into_iter().map(|f| f.id).collect();
    let mut out = Vec::new();
    for link in crate::graph::deep_links_from(store, &sources)? {
        if !store
            .chunk_seqs_with_heading(link.target_id, &link.heading)?
            .is_empty()
        {
            continue;
        }
        let Some(target) = store.get_file_by_id(link.target_id)? else {
            continue; // the note went away mid-read
        };
        out.push(StaleHeading {
            source: link.source,
            target: target.path,
            heading: link.heading,
        });
    }
    Ok(out)
}

/// Find notes that haven't been updated in the given number of days.
///
/// Stub — returns an empty vec for now. A full implementation would check
/// `mtime` or a `reviewed_at` frontmatter field.
pub fn find_stale_notes(_store: &Store, _days: u32) -> Result<Vec<String>> {
    Ok(Vec::new())
}

/// Generate a combined health report for the vault.
pub fn generate_health_report(store: &Store, config: &HealthConfig) -> Result<HealthReport> {
    let orphans = find_orphans(store, config)?;
    let broken_links = find_broken_links(store)?;
    let stale_headings = find_stale_headings(store)?;
    let stale_notes = find_stale_notes(store, 90)?;

    // Inbox pending: files in the inbox folder.
    let inbox_pending = if let Some(ref inbox) = config.inbox_folder {
        store
            .find_files_by_prefix(&format!("{}%", inbox))?
            .into_iter()
            .map(|f| f.path)
            .collect()
    } else {
        Vec::new()
    };

    let all_files = store.get_all_files()?;
    let total_files = all_files.len();

    // Tag issues: find work notes missing required tags.
    let tag_issues = all_files
        .iter()
        .filter(|f| f.path.contains("Work/") || f.path.contains("01-Projects/Work/"))
        .filter(|f| !f.tags.iter().any(|t| t.eq_ignore_ascii_case("work")))
        .map(|f| TagIssue {
            file: f.path.clone(),
            issue: "work note missing 'work' tag".to_string(),
        })
        .collect();

    // Index age: seconds since the most recent indexed_at timestamp.
    let index_age_seconds = {
        let last = all_files
            .iter()
            .filter_map(|f| f.indexed_at.parse::<u64>().ok())
            .max()
            .unwrap_or(0);
        if last == 0 {
            0
        } else {
            use std::time::SystemTime;
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now.saturating_sub(last)
        }
    };

    Ok(HealthReport {
        orphans,
        broken_links,
        stale_headings,
        stale_notes,
        inbox_pending,
        tag_issues,
        index_age_seconds,
        total_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{DOC_LEVEL, Store};

    /// A note whose passages are `(heading, text)`, so a link one note wrote
    /// can be resolved against the headings another note holds.
    fn note(store: &Store, path: &str, passages: &[(&str, &str)]) -> i64 {
        let id = store
            .insert_file(
                path,
                "h",
                100,
                &crate::docid::generate_docid(path),
                None,
                None,
            )
            .unwrap();
        for (seq, (heading, text)) in passages.iter().enumerate() {
            store
                .insert_chunk(&crate::store::NewChunk {
                    file_id: id,
                    seq: seq as i64,
                    heading,
                    text,
                    vector_id: (id * 100 + seq as i64) as u64,
                    token_count: 20,
                    ..Default::default()
                })
                .unwrap();
        }
        id
    }

    /// The fact #99 names: the note resolves, the heading does not, and
    /// nothing reported it. `build_edges_for_file` degrades the edge to
    /// `DOC_LEVEL` instead, which is indistinguishable from a plain link.
    #[test]
    fn a_link_to_a_heading_the_note_no_longer_holds_is_stale() {
        let store = Store::open_memory().unwrap();
        note(&store, "Roads.md", &[("## Norlund to Westport", "body")]);
        note(
            &store,
            "Trade.md",
            &[("## Legs", "See [[Roads#Norlund to Westport via Bend]].")],
        );
        assert_eq!(
            find_stale_headings(&store).unwrap(),
            vec![StaleHeading {
                source: "Trade.md".into(),
                target: "Roads.md".into(),
                heading: "Norlund to Westport via Bend".into(),
            }]
        );
    }

    /// A link that still names a heading the note holds is not a finding (#99).
    #[test]
    fn a_link_to_a_heading_the_note_still_holds_is_not_stale() {
        let store = Store::open_memory().unwrap();
        note(&store, "Roads.md", &[("## Norlund to Westport", "body")]);
        note(
            &store,
            "Trade.md",
            &[("## Legs", "See [[Roads#Norlund to Westport]].")],
        );
        assert!(find_stale_headings(&store).unwrap().is_empty());
    }

    /// An oversized section becomes `## Events` and `## Events (cont.)`, and a
    /// link to `#Events` means both. `normalise_heading` is what says so, and
    /// this check reads headings through it (#99).
    #[test]
    fn a_heading_split_across_two_passages_still_resolves() {
        let store = Store::open_memory().unwrap();
        note(
            &store,
            "Session.md",
            &[("## Events", "first"), ("## Events (cont.)", "second")],
        );
        note(
            &store,
            "Trade.md",
            &[("## Legs", "See [[Session#Events]].")],
        );
        assert!(find_stale_headings(&store).unwrap().is_empty());
    }

    /// Heading matching folds case, so a link that spells the heading
    /// differently is not a finding (#99).
    #[test]
    fn a_heading_named_in_another_case_is_not_stale() {
        let store = Store::open_memory().unwrap();
        note(&store, "Roads.md", &[("## Norlund to Westport", "body")]);
        note(
            &store,
            "Trade.md",
            &[("## Legs", "See [[roads#norlund TO westport]].")],
        );
        assert!(find_stale_headings(&store).unwrap().is_empty());
    }

    /// A link to a note that does not exist is `broken_links`' finding. This
    /// check would name the same link a second time under another heading, so
    /// it leaves it alone (#99).
    #[test]
    fn a_link_whose_note_does_not_resolve_is_left_to_broken_links() {
        let store = Store::open_memory().unwrap();
        note(&store, "Trade.md", &[("## Legs", "See [[Nowhere#Bend]].")]);
        assert!(find_stale_headings(&store).unwrap().is_empty());
    }

    fn setup_health_store() -> Store {
        let store = Store::open_memory().unwrap();
        // Insert files with edges to test orphan detection.
        let linked_id = store
            .insert_file("linked.md", "aaa111", 100, "aaa111", None, None)
            .unwrap();
        let orphan_id = store
            .insert_file("orphan.md", "bbb222", 100, "bbb222", None, None)
            .unwrap();
        let _daily_id = store
            .insert_file("daily/2026-03-26.md", "ccc333", 100, "ccc333", None, None)
            .unwrap();
        // Add edge: linked.md → orphan.md (both files are "connected")
        store
            .insert_edge(linked_id, DOC_LEVEL, orphan_id, DOC_LEVEL, "wikilink")
            .unwrap();
        store
    }

    #[test]
    fn test_find_orphans_excludes_daily() {
        let store = setup_health_store();
        let config = HealthConfig {
            daily_folder: Some("daily/".to_string()),
            inbox_folder: None,
        };
        let orphans = find_orphans(&store, &config).unwrap();
        // linked.md has outgoing edge, orphan.md has incoming edge — both connected.
        // daily note is excluded by prefix. Result should be empty.
        assert!(orphans.is_empty());
    }

    #[test]
    fn test_find_orphans_detects_isolated() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("connected.md", "h1", 100, "d1", None, None)
            .unwrap();
        let iso_id = store
            .insert_file("island.md", "h2", 100, "d2", None, None)
            .unwrap();
        let other_id = store
            .insert_file("other.md", "h3", 100, "d3", None, None)
            .unwrap();
        store
            .insert_edge(iso_id, DOC_LEVEL, other_id, DOC_LEVEL, "wikilink")
            .unwrap();

        let config = HealthConfig {
            daily_folder: None,
            inbox_folder: None,
        };
        let orphans = find_orphans(&store, &config).unwrap();
        // connected.md has no edges at all — it's the orphan.
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0], "connected.md");
    }

    #[test]
    fn test_find_broken_links() {
        let store = setup_health_store();
        // Record an unresolved link (wikilink target that doesn't exist).
        let source = store.get_file("linked.md").unwrap().unwrap().id;
        store
            .insert_unresolved_link(source, "nonexistent.md")
            .unwrap();
        let broken = find_broken_links(&store).unwrap();
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0].source, "linked.md");
        assert_eq!(broken[0].target, "nonexistent.md");
    }

    #[test]
    fn test_find_broken_links_empty_when_none() {
        let store = setup_health_store();
        let broken = find_broken_links(&store).unwrap();
        assert!(broken.is_empty());
    }

    #[test]
    fn test_generate_health_report() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("note.md", "h1", 100, "d1", None, None)
            .unwrap();
        store
            .insert_file("00-Inbox/unsorted.md", "h2", 100, "d2", None, None)
            .unwrap();
        let source = store.get_file("note.md").unwrap().unwrap().id;
        store.insert_unresolved_link(source, "missing.md").unwrap();

        let config = HealthConfig {
            daily_folder: Some("daily/".to_string()),
            inbox_folder: Some("00-Inbox/".to_string()),
        };
        let report = generate_health_report(&store, &config).unwrap();
        assert_eq!(report.total_files, 2);
        // note.md has no edges and is not in daily/ or inbox/ — it's an orphan.
        assert_eq!(report.orphans.len(), 1);
        assert_eq!(report.orphans[0], "note.md");
        // One broken link recorded.
        assert_eq!(report.broken_links.len(), 1);
        // One file in inbox.
        assert_eq!(report.inbox_pending.len(), 1);
        assert_eq!(report.inbox_pending[0], "00-Inbox/unsorted.md");
    }
}
