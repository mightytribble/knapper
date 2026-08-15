use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::profile::VaultProfile;
use crate::store::Store;

/// Shared context for all context engine functions.
pub struct ContextParams<'a> {
    pub store: &'a Store,
    pub vault_path: &'a Path,
    pub profile: Option<&'a VaultProfile>,
}

#[derive(Debug, Serialize)]
pub struct NoteContent {
    pub path: String,
    pub docid: Option<String>,
    pub content: String,
    pub tags: Vec<String>,
    pub frontmatter: String,
    pub body: String,
    pub outgoing_links: Vec<LinkRef>,
    pub incoming_links: Vec<LinkRef>,
    pub byte_count: usize,
    pub section: Option<SectionSpan>,
}

/// A link's other end. The docid is here because `graph show` printed it
/// beside every path and `read` did not, and `read` is now the one answer
/// to "what does this note connect to" (#62).
#[derive(Debug, Serialize, PartialEq)]
pub struct LinkRef {
    pub path: String,
    pub docid: Option<String>,
}

/// Where a section sits in its file. `read` reports it when a section was
/// asked for, and nothing when the whole note was (#62).
#[derive(Debug, Serialize)]
pub struct SectionSpan {
    pub heading: String,
    pub line_start: usize,
    pub line_end: usize,
}

/// One heading of a note's outline.
///
/// `line` is 1-based, because a caller reads it to open the file at that
/// heading and every editor counts from one; `markdown::parse_headings`
/// counts from zero, and one conversion in one place keeps the two
/// conventions apart (#68).
#[derive(Debug, Serialize)]
pub struct Heading {
    /// The `#` depth of an ATX heading, and absent for a promoted bold line,
    /// which has no depth of its own. The absence is what the CLI renders as
    /// the bold form, and `markdown::PROMOTED_LEVEL` reaches no surface
    /// (#44, #69).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u8>,
    pub text: String,
    pub line: usize,
}

#[derive(Debug, Serialize)]
pub struct NoteListItem {
    pub path: String,
    pub docid: Option<String>,
    pub tags: Vec<String>,
    pub indexed_at: String,
    pub edge_count: usize,
    /// The note's headings, ATX and promoted bold lines alike, when the
    /// caller asked for them. Absent otherwise, so an undetailed listing
    /// serialises as it did before this field existed (#68, #69).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headings: Option<Vec<Heading>>,
}

#[derive(Debug, Serialize)]
pub struct VaultMap {
    pub vault_path: String,
    pub vault_type: String,
    pub structure: String,
    pub total_files: usize,
    pub total_chunks: usize,
    pub total_edges: usize,
    pub folders: Vec<FolderInfo>,
    pub top_tags: Vec<(String, usize)>,
    pub recent_files: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct FolderInfo {
    pub path: String,
    pub note_count: usize,
}

fn resolve_file(
    params: &ContextParams,
    file_or_docid: &str,
) -> Result<Option<crate::store::FileRecord>> {
    // Docid lookup: #abcdef
    if file_or_docid.starts_with('#') && file_or_docid.len() == 7 {
        return params.store.get_file_by_docid(&file_or_docid[1..]);
    }

    // Exact path lookup
    if let Some(f) = params.store.get_file(file_or_docid)? {
        return Ok(Some(f));
    }

    // Basename fallback via SQL
    params.store.find_file_by_basename(file_or_docid)
}

/// Split content into (frontmatter YAML, body) parts.
fn split_frontmatter(content: &str) -> (String, String) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (String::new(), content.to_string());
    }
    let after = &trimmed[3..];
    let after = after.trim_start_matches('-');
    let after = after.strip_prefix('\n').unwrap_or(after);
    if let Some(end) = after.find("\n---") {
        let fm = after[..end].to_string();
        let body = after[end + 4..]
            .strip_prefix('\n')
            .unwrap_or(&after[end + 4..]);
        (fm, body.to_string())
    } else {
        (String::new(), content.to_string())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read a single note with full content, metadata, and graph edges. A
/// section narrows `content` to one heading's body and reports its span;
/// the file-level fields — tags and links — are the file's either way,
/// because a section's tags and backlinks are its file's (#62).
pub fn context_read(
    params: &ContextParams,
    file_or_docid: &str,
    section: Option<&str>,
) -> Result<NoteContent> {
    let record = resolve_file(params, file_or_docid)?
        .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_or_docid))?;

    let full_path = params.vault_path.join(&record.path);
    let (content, body, frontmatter) = match std::fs::read_to_string(&full_path) {
        Ok(c) => {
            let (fm, b) = split_frontmatter(&c);
            (c, b, fm)
        }
        Err(_) => {
            let msg = "[File not found on disk. Re-run 'engraph index' to update.]".to_string();
            (String::new(), msg, String::new())
        }
    };

    // A section read narrows the content and nothing else: a section's tags
    // and backlinks are its file's, so those fields are the same either way
    // (#62). `find_section` resolves a section by its heading text or its full
    // heading path, and a promoted bold line is one it reaches (#53, #69).
    let (content, body, span) = match section {
        None => (content, body, None),
        Some(heading) => {
            let found = crate::markdown::find_section(&content, heading)
                .ok_or_else(|| anyhow::anyhow!("Section not found: {heading}"))?;
            let span = SectionSpan {
                heading: found.heading.text.clone(),
                line_start: found.body_start,
                line_end: found.body_end,
            };
            (found.content.clone(), found.content, Some(span))
        }
    };

    let outgoing_links: Vec<LinkRef> = params
        .store
        .get_outgoing(record.id, Some("wikilink"))?
        .iter()
        .filter_map(|(fid, _)| params.store.get_file_by_id(*fid).ok().flatten())
        .map(|f| LinkRef {
            path: f.path,
            docid: f.docid,
        })
        .collect();
    let incoming_links: Vec<LinkRef> = params
        .store
        .get_incoming(record.id, Some("wikilink"))?
        .iter()
        .filter_map(|(fid, _)| params.store.get_file_by_id(*fid).ok().flatten())
        .map(|f| LinkRef {
            path: f.path,
            docid: f.docid,
        })
        .collect();
    let byte_count = content.len();
    Ok(NoteContent {
        path: record.path,
        docid: record.docid,
        content,
        tags: record.tags,
        frontmatter,
        body,
        outgoing_links,
        incoming_links,
        byte_count,
        section: span,
    })
}

/// A note's headings, read from disk.
///
/// The index cannot answer this. A section under `chunk_min_chars` merges
/// into the chunk before it and keeps no heading row of its own, a heading
/// whose own body is empty emits no chunk at all, `promote_bold_headings`
/// puts bold-only lines into `chunks.heading` beside real headings, and an
/// oversized section splits across rows that repeat their heading. The file
/// is the only source that holds the outline (#68).
///
/// A file the store holds and the disk does not answers an empty outline
/// and no error: the row is transient, and `writer::verify_index_integrity`
/// drops it at the start of the next index.
///
/// The set is `markdown::headings_with_promotions`, which is what
/// `find_section` addresses, so every entry listed here can be read and
/// written by name (#69).
fn outline(vault_path: &Path, path: &str) -> Vec<Heading> {
    let Ok(content) = std::fs::read_to_string(vault_path.join(path)) else {
        return Vec::new();
    };
    // The frontmatter is stripped before parsing, because a YAML comment
    // line reads as an H1 to a parser that sees it. The lines it removed
    // are added back, so the numbers are the file's own.
    let (_, body) = crate::markdown::split_frontmatter(&content);
    let offset = content.lines().count().saturating_sub(body.lines().count());
    crate::markdown::headings_with_promotions(&body)
        .into_iter()
        .map(|h| Heading {
            level: (!h.promoted).then_some(h.level),
            text: h.text,
            line: h.line + offset + 1,
        })
        .collect()
}

/// The notes a scope admits, in path order (#68). A caller's directory
/// filter is a scope term, which is a case-sensitive range and not a `LIKE`.
///
/// `detailed` costs one file read per listed note, and only then; an
/// undetailed listing touches no file.
pub fn context_list(
    params: &ContextParams,
    tags: &crate::tags::Scope,
    created_by: Option<&str>,
    limit: Option<usize>,
    detailed: bool,
) -> Result<Vec<NoteListItem>> {
    let files = params.store.list_files(tags, created_by, limit)?;
    let file_ids: Vec<i64> = files.iter().map(|f| f.id).collect();
    let edge_counts = params
        .store
        .edge_counts_for_files(&file_ids)
        .unwrap_or_default();
    let mut items = Vec::new();
    for f in files {
        let edge_count = edge_counts.get(&f.id).copied().unwrap_or(0);
        let headings = detailed.then(|| outline(params.vault_path, &f.path));
        items.push(NoteListItem {
            path: f.path,
            docid: f.docid,
            tags: f.tags,
            indexed_at: f.indexed_at,
            edge_count,
            headings,
        });
    }
    Ok(items)
}

/// High-level vault overview: folders, tags, recent files, counts.
pub fn vault_map(params: &ContextParams) -> Result<VaultMap> {
    let stats = params.store.stats()?;
    let edge_stats = params.store.get_edge_stats().ok();

    let (vault_type, structure) = match params.profile {
        Some(p) => (
            format!("{:?}", p.vault_type),
            format!("{:?}", p.structure.method),
        ),
        None => ("Unknown".into(), "Unknown".into()),
    };

    let folder_counts = params.store.folder_note_counts()?;
    let folders: Vec<FolderInfo> = folder_counts
        .into_iter()
        .map(|(path, count)| FolderInfo {
            path,
            note_count: count,
        })
        .collect();

    let top_tags = params.store.top_tags(20)?;

    let recent = params.store.recent_files(10)?;
    let recent_files: Vec<String> = recent.into_iter().map(|f| f.path).collect();

    Ok(VaultMap {
        vault_path: params.vault_path.to_string_lossy().to_string(),
        vault_type,
        structure,
        total_files: stats.file_count,
        total_chunks: stats.chunk_count,
        total_edges: edge_stats.map(|e| e.total_edges).unwrap_or(0),
        folders,
        top_tags,
        recent_files,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::store::{DOC_LEVEL, Store};
    use tempfile::TempDir;

    /// A tag whose display form is its path.
    fn tag(path: &str) -> crate::tags::Tag {
        crate::tags::Tag {
            path: path.into(),
            display: path.into(),
        }
    }

    fn setup_vault() -> (TempDir, Store, std::path::PathBuf) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();

        std::fs::write(
            root.join("note.md"),
            "---\ntags:\n  - rust\n---\n# Note\n\nContent here.\n\nSee [[other]].",
        )
        .unwrap();
        std::fs::write(root.join("other.md"), "# Other\n\nMore content.").unwrap();

        let store = Store::open_memory().unwrap();
        let d1 = generate_docid("note.md");
        let d2 = generate_docid("other.md");
        let note = store
            .insert_file("note.md", "h1", 100, &d1, None, None)
            .unwrap();
        store.reconcile_file_tags(note, &[tag("rust")]).unwrap();
        store
            .insert_file("other.md", "h2", 100, &d2, None, None)
            .unwrap();

        let f1 = store.get_file("note.md").unwrap().unwrap().id;
        let f2 = store.get_file("other.md").unwrap().unwrap().id;
        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f2, DOC_LEVEL, f1, DOC_LEVEL, "wikilink")
            .unwrap();

        (tmp, store, root)
    }

    #[test]
    fn test_read_by_path() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "note.md", None).unwrap();
        assert_eq!(note.path, "note.md");
        assert!(note.content.contains("Content here."));
        assert!(note.body.contains("Content here."));
        assert!(note.frontmatter.contains("tags:"));
        assert!(note.tags.contains(&"rust".to_string()));
        assert_eq!(note.outgoing_links.len(), 1);
        assert_eq!(note.incoming_links.len(), 1);
        assert!(note.byte_count > 0);
    }

    #[test]
    fn test_read_by_docid() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let docid = generate_docid("note.md");
        let note = context_read(&params, &format!("#{}", docid), None).unwrap();
        assert_eq!(note.path, "note.md");
    }

    #[test]
    fn test_read_file_not_on_disk() {
        let (_tmp, store, root) = setup_vault();
        store
            .insert_file("ghost.md", "h3", 100, "ggg333", None, None)
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "ghost.md", None).unwrap();
        assert!(note.body.contains("File not found on disk"));
    }

    #[test]
    fn test_read_by_basename() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "note", None).unwrap();
        assert_eq!(note.path, "note.md");
    }

    #[test]
    fn test_context_list_no_filter() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            Some(20),
            false,
        )
        .unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_context_list_tag_filter() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(
            &params,
            &crate::tags::Scope::parse(&["rust".into()], &[], &[]).unwrap(),
            None,
            Some(20),
            false,
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "note.md");
    }

    /// One scope operator carries a tag term and a directory term at once,
    /// which is why `list` needs no second directory handle (#65, #68).
    #[test]
    fn a_scope_mixing_a_tag_and_a_directory_reaches_the_listing() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let store = Store::open_memory().unwrap();
        let inside = store
            .insert_file("lore/wight.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let outside = store
            .insert_file("bestiary/wolf.md", "h2", 100, "bbb222", None, None)
            .unwrap();
        store
            .reconcile_file_tags(inside, &[tag("type/undead")])
            .unwrap();
        store
            .reconcile_file_tags(outside, &[tag("type/undead")])
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(
            &params,
            &crate::tags::Scope::parse(&["type/undead".into(), "/lore/".into()], &[], &[]).unwrap(),
            None,
            None,
            false,
        )
        .unwrap();
        let paths: Vec<&str> = items.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(paths, vec!["lore/wight.md"]);
    }

    /// One note on disk and one row in the store, listed with `detailed`.
    fn outline_of(content: &str) -> Vec<Heading> {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        std::fs::write(root.join("note.md"), content).unwrap();
        let store = Store::open_memory().unwrap();
        store
            .insert_file("note.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items =
            context_list(&params, &crate::tags::Scope::default(), None, None, true).unwrap();
        items
            .into_iter()
            .next()
            .expect("the one note is listed")
            .headings
            .expect("a detailed listing carries an outline")
    }

    /// The outline is the file's ATX headings in file order, each with its
    /// level and its 1-based line (#68).
    #[test]
    fn an_outline_holds_every_heading_in_file_order() {
        let headings = outline_of(
            "# About the Empire\n\n## History\n\n### The founding\n\nText.\n\n## Current Events\n",
        );
        let got: Vec<(Option<u8>, &str, usize)> = headings
            .iter()
            .map(|h| (h.level, h.text.as_str(), h.line))
            .collect();
        assert_eq!(
            got,
            vec![
                (Some(1), "About the Empire", 1),
                (Some(2), "History", 3),
                (Some(3), "The founding", 5),
                (Some(2), "Current Events", 9),
            ]
        );
    }

    /// `parse_headings` skips fenced blocks, so a `#` line inside one is a
    /// comment in a code sample and not a heading (#68).
    #[test]
    fn a_hash_inside_a_fence_is_not_a_heading() {
        let headings = outline_of("# Real\n\n```bash\n# not a heading\n```\n\n## Also real\n");
        let got: Vec<&str> = headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(got, vec!["Real", "Also real"]);
    }

    /// The outline lists what `--section` can address, promoted bold lines
    /// included, because it is where a caller reads the path it then names
    /// (#69).
    #[test]
    fn an_outline_lists_promoted_headings_beside_atx_ones() {
        let headings =
            outline_of("# Archdragon\n\n## Stat Block\n\nAC 20\n\n**Spells**\n\nFireball\n");
        let got: Vec<(Option<u8>, &str)> = headings
            .iter()
            .map(|h| (h.level, h.text.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                (Some(1), "Archdragon"),
                (Some(2), "Stat Block"),
                (None, "Spells"),
            ]
        );
    }

    /// A promoted line with no body is listed, because addressing an empty
    /// section is how a caller fills it (#69).
    #[test]
    fn an_outline_lists_a_bodyless_promoted_heading() {
        let headings = outline_of("## Stat Block\n\n**Spells**\n**Notes**\n\nSee below\n");
        let got: Vec<&str> = headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(got, vec!["Stat Block", "Spells", "Notes"]);
    }

    /// Every entry the outline lists is a section `find_section` resolves.
    /// The two read one set, and this is what holds them to it (#69).
    #[test]
    fn every_outline_entry_is_addressable() {
        let content = "# Archdragon\n\n## Stat Block\n\nAC 20\n\n**Spells**\n\nFireball\n\n**Notes**\n**Tail**\n\nEnd\n";
        for h in outline_of(content) {
            assert!(
                crate::markdown::find_section(content, &h.text).is_some(),
                "the outline lists {} and find_section resolves nothing",
                h.text
            );
        }
    }

    /// A promoted line carries no `level` key, so an ATX heading serialises
    /// as it did and a consumer reads the absence rather than a sentinel
    /// (#69).
    #[test]
    fn a_promoted_heading_serialises_without_a_level() {
        let headings = outline_of("## Stat Block\n\n**Spells**\n\nFireball\n");
        let json = serde_json::to_string(&headings).unwrap();
        assert!(json.contains(r#"{"level":2,"text":"Stat Block""#));
        assert!(json.contains(r#"{"text":"Spells""#));
    }

    /// The frontmatter is stripped before parsing, because a YAML comment
    /// line reads as an H1 to the parser. The lines it removed are added
    /// back, so the numbers are the file's own (#68).
    #[test]
    fn a_hash_inside_frontmatter_is_not_a_heading_and_the_lines_stay_the_files_own() {
        let headings = outline_of("---\n# a yaml comment\ntags: [a]\n---\n# Real\n");
        let got: Vec<(&str, usize)> = headings.iter().map(|h| (h.text.as_str(), h.line)).collect();
        assert_eq!(got, vec![("Real", 5)]);
    }

    /// A note with no headings has an empty outline, which is a fact about
    /// the note and not a failure (#68).
    #[test]
    fn a_note_with_no_headings_has_an_empty_outline() {
        assert!(outline_of("Just a paragraph.\n").is_empty());
    }

    /// `list` reports the index. A row whose file is gone is transient —
    /// `writer::verify_index_integrity` drops it at the start of the next
    /// index — so it is listed with an empty outline and no error (#68).
    #[test]
    fn a_row_whose_file_is_missing_is_listed_with_an_empty_outline() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let store = Store::open_memory().unwrap();
        store
            .insert_file("ghost.md", "h1", 100, "ggg333", None, None)
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items =
            context_list(&params, &crate::tags::Scope::default(), None, None, true).unwrap();
        assert_eq!(items.len(), 1);
        assert!(
            items[0]
                .headings
                .as_ref()
                .is_some_and(|headings| headings.is_empty()),
            "a missing file is listed with an outline that is empty, not absent"
        );
    }

    /// Without `detailed` the field is absent from the JSON, so an
    /// undetailed listing serialises exactly as it did before, and
    /// `project`, whose child notes are the same type, is untouched (#68).
    #[test]
    fn an_undetailed_listing_carries_no_headings_field() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items =
            context_list(&params, &crate::tags::Scope::default(), None, None, false).unwrap();
        let json = serde_json::to_string(&items).unwrap();
        assert!(!json.contains("headings"), "{json}");
    }

    #[test]
    fn test_vault_map() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let map = vault_map(&params).unwrap();
        assert_eq!(map.total_files, 2);
        assert!(!map.folders.is_empty());
        assert!(map.top_tags.iter().any(|(t, _)| t == "rust"));
    }

    #[test]
    fn test_split_frontmatter() {
        let (fm, body) = split_frontmatter("---\ntags:\n  - rust\n---\n# Hello\nWorld");
        assert!(fm.contains("tags:"));
        assert!(body.contains("# Hello"));
        assert!(!body.contains("---"));
    }

    #[test]
    fn test_split_frontmatter_no_fm() {
        let (fm, body) = split_frontmatter("# Just content\nHere.");
        assert!(fm.is_empty());
        assert!(body.contains("# Just content"));
    }

    /// A vault of one person note, with a `person` tag, a `Role` and an
    /// `Interactions` section, and an outgoing wikilink to `colleague.md` —
    /// Task 9 reuses this fixture and needs that link on `person.md`.
    ///
    /// The frontmatter block is real (not empty) so a span measured against
    /// the frontmatter-stripped body, rather than the whole file
    /// `find_section` reads, would be caught by a test pinning the exact
    /// line numbers.
    fn section_fixture() -> (Store, std::path::PathBuf, TempDir) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let store = Store::open_memory().unwrap();
        let content = "---\ntags:\n  - person\n---\n# Person\n\n## Role\n\nEngineer\n\n## Interactions\n\nMet on 2026-03-26. See [[colleague]].\n";
        std::fs::write(root.join("person.md"), content).unwrap();
        std::fs::write(root.join("colleague.md"), "# Colleague\n").unwrap();
        store
            .insert_file("person.md", "hash", 100, "per123", None, None)
            .unwrap();
        store
            .insert_file("colleague.md", "hash2", 100, "col456", None, None)
            .unwrap();
        let f1 = store.get_file("person.md").unwrap().unwrap().id;
        let f2 = store.get_file("colleague.md").unwrap().unwrap().id;
        store.reconcile_file_tags(f1, &[tag("person")]).unwrap();
        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        (store, root, tmp)
    }

    #[test]
    fn reading_a_section_narrows_the_content_and_keeps_the_file_facts() {
        // Reuse whatever fixture `test_read_section` builds: a store, a vault
        // root and a `person.md` holding an `Interactions` section.
        let (store, root, _tmp) = section_fixture();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };

        let whole = context_read(&params, "person.md", None).unwrap();
        let part = context_read(&params, "person.md", Some("Interactions")).unwrap();

        // The section's content is a part of the note's.
        assert!(whole.content.contains(part.content.trim()));
        assert!(part.content.len() < whole.content.len());

        // A section carries its own span, measured against the whole file
        // `find_section` reads — not the frontmatter-stripped body, four
        // lines shorter in this fixture.
        let span = part.section.expect("a section read reports its span");
        assert_eq!(span.heading, "Interactions");
        assert_eq!(span.line_start, 11);
        assert_eq!(span.line_end, 13);
        assert!(whole.section.is_none());

        // The file-level facts are the file's, whichever way it is read.
        assert_eq!(part.path, whole.path);
        assert_eq!(part.tags, whole.tags);
        assert!(!part.tags.is_empty(), "the fixture's tag must round-trip");
        assert_eq!(part.outgoing_links, whole.outgoing_links);

        // A heading the note does not have is an error, not an empty section.
        assert!(context_read(&params, "person.md", Some("Nope")).is_err());
    }

    #[test]
    fn a_link_carries_the_docid_the_graph_view_used_to_print() {
        let (store, root, _tmp) = section_fixture();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = context_read(&params, "person.md", None).unwrap();
        let first = note.outgoing_links.first().expect("a link");
        assert!(!first.path.is_empty());
        assert!(first.docid.is_some(), "a link names the file's docid");
    }
}
