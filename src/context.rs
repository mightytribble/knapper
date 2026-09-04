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

/// A note's content: the requested text and where it sits, and nothing more.
/// The default read (#80). Metadata — frontmatter, links, size — is a
/// separate read, so a caller pays for the note's prose alone.
#[derive(Debug, Serialize)]
pub struct NoteContent {
    pub path: String,
    pub docid: Option<String>,
    /// The requested text: the whole note's body with the frontmatter
    /// stripped, or one section's markdown with its heading line included
    /// (#80, #81).
    pub content: String,
    /// The section's span, only when a section was asked for (#80).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<SectionSpan>,
}

/// A note's metadata: everything about the note that is not its prose — its
/// frontmatter, its links, and its size. The `--metadata` read (#80). It is
/// always the whole note's, so it takes no section.
#[derive(Debug, Serialize)]
pub struct NoteMetadata {
    pub path: String,
    pub docid: Option<String>,
    pub frontmatter: String,
    pub outgoing_links: Vec<LinkRef>,
    pub incoming_links: Vec<LinkRef>,
    /// Every custom property the note holds, frontmatter and body; a body
    /// row names its section through `heading_path` (#66).
    pub properties: Vec<crate::store::PropertyRow>,
    pub byte_count: usize,
}

/// What a read returns: a note's content, or its metadata. The two are
/// separate reads, so a caller receives exactly one (#80). Serialized
/// untagged, so the JSON is the inner object with no wrapper.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ReadResult {
    Content(NoteContent),
    Metadata(NoteMetadata),
}

/// A link's other end. The docid is here because `graph show` printed it
/// beside every path and `read` did not, and `read` is now the one answer
/// to "what does this note connect to" (#62).
#[derive(Debug, Serialize, PartialEq)]
pub struct LinkRef {
    pub path: String,
    pub docid: Option<String>,
    /// The custom properties this link is filed under. Empty for a plain
    /// wikilink (#66).
    pub properties: Vec<String>,
}

/// Where a section sits in its file. `read` reports it when a section was
/// asked for, and nothing when the whole note was (#62).
///
/// `heading` and `level` are the section's heading, which the content no
/// longer carries: content is the body alone, so that a caller can write it
/// straight back through `update` (#96). The two fields are what a caller
/// reassembles the section's markdown from, and what a rename reads before
/// it writes a new heading through `update`'s `heading` (#97).
///
/// `line_start` and `line_end` are 1-based and inclusive and they bracket
/// the section: `line_start` is the heading's own line, one above the
/// content, and `line_end` is the section's last line.
#[derive(Debug, Serialize)]
pub struct SectionSpan {
    pub heading: String,
    /// The `#` depth of an ATX heading, and absent for a promoted bold line,
    /// which has no depth of its own — the convention the outline follows
    /// (#44, #69).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u8>,
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
    /// The property rows the scope's `property` term matched, narrowed by a
    /// `links_to` term beside it. Absent when the scope carries no property
    /// term, so a listing with no property filter serialises as it did
    /// before, and absent under `linked_from`, where the matched row
    /// belongs to the naming note (#66). `context::matched_properties`
    /// states the rule.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Vec<crate::store::PropertyRow>>,
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

/// The notes at the far end of a set of edges, each with the property
/// names the link is filed under (#66). `names` answers those for one far
/// note, so the caller decides which end of the link is this note.
fn link_refs<T>(
    store: &Store,
    edges: &[(i64, T)],
    names: impl Fn(i64) -> Vec<String>,
) -> Vec<LinkRef> {
    edges
        .iter()
        .filter_map(|(fid, _)| store.get_file_by_id(*fid).ok().flatten())
        .map(|f| LinkRef {
            properties: names(f.id),
            path: f.path,
            docid: f.docid,
        })
        .collect()
}

/// Read a note. Two modes (#80): the default returns the note's content —
/// the whole note's body, frontmatter stripped, or one section's markdown —
/// and `metadata` returns the note's frontmatter, links, and size instead.
/// The two modes are exclusive, and `metadata` describes the whole note, so
/// it cannot be combined with a section.
pub fn context_read(
    params: &ContextParams,
    file_or_docid: &str,
    section: Option<&str>,
    metadata: bool,
) -> Result<ReadResult> {
    if metadata && section.is_some() {
        anyhow::bail!(
            "--section and --metadata cannot be combined: metadata describes the whole note"
        );
    }

    let record = resolve_file(params, file_or_docid)?
        .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_or_docid))?;

    let full_path = params.vault_path.join(&record.path);
    let disk = std::fs::read_to_string(&full_path).ok();

    if metadata {
        let (frontmatter, byte_count) = match &disk {
            Some(c) => (split_frontmatter(c).0, c.len()),
            // A row whose file is gone on disk still has its links and its
            // docid; the frontmatter and the size are the file's, so they are
            // empty rather than invented.
            None => (String::new(), 0),
        };
        let id = record.id;
        return Ok(ReadResult::Metadata(NoteMetadata {
            path: record.path,
            docid: record.docid,
            frontmatter,
            outgoing_links: link_refs(
                params.store,
                &params.store.get_outgoing(id, Some("wikilink"))?,
                |to| {
                    params
                        .store
                        .property_names_for_link(id, to)
                        .unwrap_or_default()
                },
            ),
            incoming_links: link_refs(
                params.store,
                &params.store.get_incoming(id, Some("wikilink"))?,
                |from| {
                    params
                        .store
                        .property_names_for_link(from, id)
                        .unwrap_or_default()
                },
            ),
            properties: params.store.file_properties(id)?,
            byte_count,
        }));
    }

    // Content mode. A file the store holds and the disk does not answers the
    // re-index note in place of content, the way it always has (#62).
    let Some(content_str) = disk else {
        return Ok(ReadResult::Content(NoteContent {
            path: record.path,
            docid: record.docid,
            content: "[File not found on disk. Re-run 'knapper index' to update.]".to_string(),
            section: None,
        }));
    };

    // The whole note's body with the frontmatter stripped, or one section's
    // body with its heading named beside it (#80, #96). `find_section`
    // resolves a section by its heading text or its full heading path, and a
    // promoted bold line is one it reaches (#53, #69).
    let (content, span) = match section {
        None => (note_body(&content_str), None),
        Some(heading) => {
            let found = crate::markdown::find_section(&content_str, heading)
                .ok_or_else(|| anyhow::anyhow!("Section not found: {heading}"))?;
            let span = SectionSpan {
                heading: found.heading.text.clone(),
                level: (!found.heading.promoted).then_some(found.heading.level),
                line_start: found.heading.line + 1,
                line_end: found.body_end,
            };
            (found.body, Some(span))
        }
    };

    Ok(ReadResult::Content(NoteContent {
        path: record.path,
        docid: record.docid,
        content,
        section: span,
    }))
}

/// A note's body, as `update`'s body edit defines it: everything below the
/// frontmatter block and the one line ending that separates the two, which is
/// `frontmatter::split_body`'s own split. `read` used
/// `markdown::split_frontmatter`, which counts that separator as the body's
/// first line instead, so a caller that read a body and wrote it straight
/// back gained a blank line on every round trip. One function decides where
/// a note's body begins and both ends of the round trip read it (#96).
///
/// A block that opens and never closes has no knowable end and `split_body`
/// refuses it. The whole text is the body then, which is what
/// `split_frontmatter` answers for the same note, so a note knapper could
/// read before is a note it can still read.
fn note_body(content: &str) -> String {
    match crate::frontmatter::split_body(content) {
        Ok(Some((_, body))) => body,
        _ => content.to_string(),
    }
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

/// The property rows each listed note shows, keyed by file id, or `None`
/// when the listing shows none (#66).
///
/// `NoteListItem.properties` is the rows the scope's property term matched,
/// so the fill reads the predicate the clause selected on:
///
/// - `property` alone: the note's rows under that name, and its value when
///   the term carries one.
/// - `property` with `links_to`: those rows narrowed to the ones that name
///   the note asked for, because that is what the clause matched.
/// - `property` with `linked_from`: `None`. The matched row belongs to the
///   naming note, so no row of the listed note answers the term, and an
///   empty array would claim the note carries the property.
///
/// The link ids come from `Store::resolve_scope_links`, the resolution
/// `list_files` itself ran, so a clause and a fill cannot read one name two
/// ways and neither is built from an unresolved one.
fn matched_properties(
    params: &ContextParams,
    tags: &crate::tags::Scope,
    file_ids: &[i64],
) -> Result<Option<std::collections::HashMap<i64, Vec<crate::store::PropertyRow>>>> {
    let Some(term) = &tags.property else {
        return Ok(None);
    };
    if tags.linked_from.is_some() {
        return Ok(None);
    }
    let links = params.store.resolve_scope_links(tags)?;
    Ok(Some(params.store.matched_properties_for_files(
        file_ids,
        term,
        links.links_to,
    )?))
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
    let mut matched = matched_properties(params, tags, &file_ids)?;
    let mut items = Vec::new();
    for f in files {
        let edge_count = edge_counts.get(&f.id).copied().unwrap_or(0);
        let headings = detailed.then(|| outline(params.vault_path, &f.path));
        let properties = matched
            .as_mut()
            .map(|m| m.remove(&f.id).unwrap_or_default());
        items.push(NoteListItem {
            path: f.path,
            docid: f.docid,
            tags: f.tags,
            indexed_at: f.indexed_at,
            edge_count,
            headings,
            properties,
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

    /// The one content-mode read in these tests, unwrapped.
    fn content_of(res: ReadResult) -> NoteContent {
        match res {
            ReadResult::Content(note) => note,
            ReadResult::Metadata(_) => panic!("expected content mode"),
        }
    }

    /// The one metadata-mode read in these tests, unwrapped.
    fn metadata_of(res: ReadResult) -> NoteMetadata {
        match res {
            ReadResult::Metadata(meta) => meta,
            ReadResult::Content(_) => panic!("expected metadata mode"),
        }
    }

    #[test]
    fn test_read_by_path() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = content_of(context_read(&params, "note.md", None, false).unwrap());
        assert_eq!(note.path, "note.md");
        assert!(note.content.contains("Content here."));
        assert!(note.section.is_none());
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
        let note = content_of(context_read(&params, &format!("#{}", docid), None, false).unwrap());
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
        let note = content_of(context_read(&params, "ghost.md", None, false).unwrap());
        assert!(note.content.contains("File not found on disk"));
    }

    #[test]
    fn test_read_by_basename() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = content_of(context_read(&params, "note", None, false).unwrap());
        assert_eq!(note.path, "note.md");
    }

    /// The whole-note read strips the frontmatter, so a caller reads the prose
    /// and not the YAML; the frontmatter is a `--metadata` field (#80).
    #[test]
    fn whole_note_content_is_frontmatter_stripped() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = content_of(context_read(&params, "note.md", None, false).unwrap());
        assert!(
            !note.content.contains("tags:"),
            "frontmatter leaked into content: {}",
            note.content
        );
        assert!(note.content.contains("Content here."));
    }

    /// Content mode carries the text and nothing else: no links, no
    /// frontmatter, no parsed tags, no size. Those are the `--metadata`
    /// read, so a default read does not spend the tokens on them (#80).
    #[test]
    fn content_mode_json_carries_no_metadata_fields() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let res = context_read(&params, "note.md", None, false).unwrap();
        let json = serde_json::to_string(&res).unwrap();
        for absent in [
            "outgoing_links",
            "incoming_links",
            "frontmatter",
            "byte_count",
            "\"tags\"",
            "\"body\"",
        ] {
            assert!(
                !json.contains(absent),
                "content mode leaked {absent}: {json}"
            );
        }
    }

    /// `--metadata` returns the note's frontmatter, its links, and its size,
    /// and no content (#80).
    #[test]
    fn metadata_mode_returns_frontmatter_links_and_size() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let meta = metadata_of(context_read(&params, "note.md", None, true).unwrap());
        assert_eq!(meta.path, "note.md");
        assert!(meta.frontmatter.contains("tags:"));
        assert_eq!(meta.outgoing_links.len(), 1);
        assert_eq!(meta.incoming_links.len(), 1);
        assert!(meta.byte_count > 0);
        let json = serde_json::to_string(&meta).unwrap();
        assert!(!json.contains("content"), "metadata leaked content: {json}");
    }

    /// Metadata describes the whole note, so a section makes no sense in that
    /// mode and the two are rejected together on every surface (#80).
    #[test]
    fn section_and_metadata_cannot_be_combined() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        assert!(context_read(&params, "note.md", Some("Note"), true).is_err());
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

    /// A section read returns the section's body and names its heading
    /// beside it. The heading is not in the content, because `update`'s
    /// section `replace` writes the heading already on disk and content
    /// carrying one writes it twice (#96). `heading` and `level` are what a
    /// caller reassembles the section's markdown from (#81).
    #[test]
    fn a_section_read_returns_the_body_and_names_its_heading() {
        let (store, root, _tmp) = section_fixture();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };

        let whole = content_of(context_read(&params, "person.md", None, false).unwrap());
        let part =
            content_of(context_read(&params, "person.md", Some("Interactions"), false).unwrap());

        // The content is the section's body, and a part of the whole note's.
        assert_eq!(part.content, "Met on 2026-03-26. See [[colleague]].");
        assert!(!part.content.contains("## Interactions"));
        assert!(whole.content.contains(&part.content));

        // The span is 1-based and inclusive and it brackets the section:
        // line 11 is `## Interactions`, the heading the content sits under,
        // and line 13 is the section's last line.
        let span = part.section.expect("a section read reports its span");
        assert_eq!(span.heading, "Interactions");
        assert_eq!(span.level, Some(2));
        assert_eq!(span.line_start, 11);
        assert_eq!(span.line_end, 13);
        assert!(whole.section.is_none());

        // A heading the note does not have is an error, not an empty section.
        assert!(context_read(&params, "person.md", Some("Nope"), false).is_err());
    }

    /// A promoted bold line is a section a caller can read, and it has no
    /// depth of its own, so the span carries none — the convention
    /// `list --detailed` already follows (#44, #69).
    #[test]
    fn a_promoted_section_read_carries_no_level() {
        let (store, root, _tmp) = section_fixture();
        std::fs::write(
            root.join("creature.md"),
            "# Wyrm\n\n## Stat Block\n\n**Spells**\n\nFireball\n",
        )
        .unwrap();
        store
            .insert_file("creature.md", "hash", 100, "cre321", None, None)
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };

        let part = content_of(context_read(&params, "creature.md", Some("Spells"), false).unwrap());
        let span = part.section.expect("a section read reports its span");
        assert_eq!(part.content, "Fireball");
        assert_eq!(span.heading, "Spells");
        assert_eq!(span.level, None);
    }

    /// The section half of the same round trip: what a section read returns
    /// is what a section `replace` takes back, and the file is the file it
    /// came from — no second heading, however many times it is repeated
    /// (#96).
    #[test]
    fn a_section_read_can_be_written_straight_back() {
        let (store, root, _tmp) = section_fixture();
        let original = spaced_note(&root, &store);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };

        let body =
            content_of(context_read(&params, "repro.md", Some("Alpha"), false).unwrap()).content;
        let written = crate::writer::apply_note_edits(
            &original,
            &[crate::writer::NoteEdit {
                target: crate::writer::EditTarget::Section("Alpha".into()),
                heading: None,
                mode: crate::writer::EditMode::Replace,
                content: Some(crate::writer::EditContent::Text(body)),
            }],
        )
        .unwrap();

        assert_eq!(written, original);
    }

    #[test]
    fn a_metadata_link_carries_the_docid_the_graph_view_used_to_print() {
        let (store, root, _tmp) = section_fixture();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let meta = metadata_of(context_read(&params, "person.md", None, true).unwrap());
        let first = meta.outgoing_links.first().expect("a link");
        assert!(!first.path.is_empty());
        assert!(first.docid.is_some(), "a link names the file's docid");
    }

    /// A note whose body is separated from its frontmatter by a blank line,
    /// which is the shape the round trip turns on: `markdown::split_frontmatter`
    /// counts that line as the body's first, and `frontmatter::split_body`
    /// counts it as the block's last (#96).
    fn spaced_note(root: &std::path::Path, store: &Store) -> String {
        let content = "---\nname: Repro\ntags: [type/lore]\n---\n\nLead paragraph.\n\n## Alpha\n\nAlpha body.\n";
        std::fs::write(root.join("repro.md"), content).unwrap();
        store
            .insert_file("repro.md", "hash", 100, "rep789", None, None)
            .unwrap();
        content.to_string()
    }

    /// `read`'s output is what `update` takes back. A whole-note read returns
    /// the note's body, and a body `replace` handed that body writes the file
    /// it came from, byte for byte — no blank line gained, however many times
    /// it is repeated (#96).
    #[test]
    fn a_whole_note_read_can_be_written_straight_back() {
        let (store, root, _tmp) = section_fixture();
        let original = spaced_note(&root, &store);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };

        let body = content_of(context_read(&params, "repro.md", None, false).unwrap()).content;
        let written = crate::writer::apply_note_edits(
            &original,
            &[crate::writer::NoteEdit {
                target: crate::writer::EditTarget::Body,
                heading: None,
                mode: crate::writer::EditMode::Replace,
                content: Some(crate::writer::EditContent::Text(body)),
            }],
        )
        .unwrap();

        assert_eq!(written, original);
    }

    // ── Custom properties on read and list (#66) ─────────────────

    fn with_properties() -> (TempDir, Store, std::path::PathBuf) {
        use crate::properties::Kind;
        use crate::store::NewProperty;
        let (tmp, store, root) = setup_vault();
        let note = store.get_file("note.md").unwrap().unwrap().id;
        let other = store.get_file("other.md").unwrap().unwrap().id;
        store
            .replace_file_properties(
                note,
                &[
                    NewProperty {
                        chunk_seq: DOC_LEVEL,
                        name: "status",
                        value: "draft",
                        kind: Kind::Text,
                        target_file: None,
                    },
                    NewProperty {
                        chunk_seq: DOC_LEVEL,
                        name: "related",
                        value: "other",
                        kind: Kind::Link,
                        target_file: Some(other),
                    },
                ],
            )
            .unwrap();
        (tmp, store, root)
    }

    #[test]
    fn metadata_lists_every_property_row_and_names_the_property_behind_a_link() {
        let (_tmp, store, root) = with_properties();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let meta = metadata_of(context_read(&params, "note.md", None, true).unwrap());
        let names: Vec<&str> = meta.properties.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["related", "status"]);
        assert_eq!(meta.outgoing_links[0].path, "other.md");
        assert_eq!(
            meta.outgoing_links[0].properties,
            vec!["related".to_string()]
        );
        assert!(meta.incoming_links[0].properties.is_empty());

        let other = metadata_of(context_read(&params, "other.md", None, true).unwrap());
        assert!(other.properties.is_empty());
        assert_eq!(
            other.incoming_links[0].properties,
            vec!["related".to_string()]
        );
    }

    /// `ada` files two `employer` links, to `acme` and to `beta`, and one
    /// `mentor` link to `bob`. `bob` carries no property of its own.
    fn with_link_properties() -> (TempDir, Store, std::path::PathBuf) {
        use crate::properties::Kind;
        use crate::store::NewProperty;
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let store = Store::open_memory().unwrap();
        let add = |name: &str| {
            let rel = format!("{name}.md");
            std::fs::write(root.join(&rel), format!("# {name}\n")).unwrap();
            store
                .insert_file(&rel, "h", 0, &generate_docid(&rel), None, None)
                .unwrap()
        };
        let ada = add("ada");
        let acme = add("acme");
        let beta = add("beta");
        let bob = add("bob");
        store
            .replace_file_properties(
                ada,
                &[
                    NewProperty {
                        chunk_seq: DOC_LEVEL,
                        name: "employer",
                        value: "acme",
                        kind: Kind::Link,
                        target_file: Some(acme),
                    },
                    NewProperty {
                        chunk_seq: DOC_LEVEL,
                        name: "employer",
                        value: "beta",
                        kind: Kind::Link,
                        target_file: Some(beta),
                    },
                    NewProperty {
                        chunk_seq: 0,
                        name: "mentor",
                        value: "bob",
                        kind: Kind::Link,
                        target_file: Some(bob),
                    },
                ],
            )
            .unwrap();
        for target in [acme, beta, bob] {
            store
                .insert_edge(ada, DOC_LEVEL, target, DOC_LEVEL, "wikilink")
                .unwrap();
        }
        (tmp, store, root)
    }

    /// The rows shown are the rows the clause matched: a note carrying two
    /// `employer` links shows the one that names the note asked for (#66).
    #[test]
    fn a_listing_under_links_to_shows_the_rows_that_name_that_note() {
        let (_tmp, store, root) = with_link_properties();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let scope = crate::tags::Scope::default()
            .with_filters(Some("employer"), Some("acme"), None)
            .unwrap();
        let items = context_list(&params, &scope, None, None, false).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "ada.md");
        let rows = items[0].properties.as_ref().unwrap();
        let values: Vec<&str> = rows.iter().map(|r| r.value.as_str()).collect();
        assert_eq!(values, ["acme"], "the beta row did not match: {rows:?}");
    }

    /// Under `linked_from` the matched row belongs to the naming note, so
    /// no row of the listed note answers the term and the field is absent
    /// rather than empty (#66).
    #[test]
    fn a_listing_under_linked_from_carries_no_property_rows() {
        let (_tmp, store, root) = with_link_properties();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let scope = crate::tags::Scope::default()
            .with_filters(Some("mentor"), None, Some("ada"))
            .unwrap();
        let items = context_list(&params, &scope, None, None, false).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "bob.md");
        assert!(
            items[0].properties.is_none(),
            "bob carries no mentor row: {:?}",
            items[0].properties
        );
        let json = serde_json::to_string(&items).unwrap();
        assert!(!json.contains("\"properties\""), "{json}");
    }

    #[test]
    fn a_listing_carries_the_matched_rows_only_under_a_property_term() {
        let (_tmp, store, root) = with_properties();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let plain =
            context_list(&params, &crate::tags::Scope::default(), None, None, false).unwrap();
        assert!(plain.iter().all(|i| i.properties.is_none()));
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("\"properties\""), "{json}");

        let scope = crate::tags::Scope::default()
            .with_filters(Some("status=draft"), None, None)
            .unwrap();
        let items = context_list(&params, &scope, None, None, false).unwrap();
        assert_eq!(items.len(), 1);
        let rows = items[0].properties.as_ref().unwrap();
        assert_eq!(
            rows.len(),
            1,
            "only the matched row, not every row: {rows:?}"
        );
        assert_eq!(rows[0].value, "draft");
    }
}
