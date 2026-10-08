use std::path::Path;

use crate::fault::Fault;
use crate::params::Include;
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
    /// stripped, or one section's body alone — the heading is named in
    /// `section` rather than carried here, so the text can be written
    /// straight back through `update` (#80, #96).
    pub content: String,
    /// The section's span, only when a section was asked for (#80).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<SectionSpan>,
    /// The note's frontmatter, only under `include = all` (#130). Absent
    /// under `content`, so that mode's JSON is what it always was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontmatter: Option<String>,
}

/// A note's frontmatter alone: the `include = frontmatter` read (#130). It
/// is the note's own YAML, so it takes no section, and it carries no link
/// graph — that is what separates it from a metadata read.
#[derive(Debug, Serialize)]
pub struct NoteFrontmatter {
    pub path: String,
    pub docid: Option<String>,
    pub frontmatter: String,
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

/// What a read returns: a note's content, its frontmatter, or its metadata.
/// A caller receives exactly one, chosen by `include` (#80, #130) — `all` is
/// `Content` with its `frontmatter` filled, not a variant of its own, so the
/// default mode's JSON is unchanged by the mode existing. Serialized
/// untagged, so the JSON is the inner object with no wrapper.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ReadResult {
    Content(NoteContent),
    Frontmatter(NoteFrontmatter),
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
    /// The other names the note's frontmatter records for it, as written
    /// (#142). `read` and the link filters take any of them in place of the
    /// path, so a listing is also a registry of the names a caller can use.
    pub aliases: Vec<String>,
    /// When this note was last indexed, as `YYYY-MM-DDTHH:MM:SSZ` (#121).
    pub indexed_at: String,
    /// How many distinct notes link to this note (#121). A note named from
    /// eight sections of one note counts one, and a link written in
    /// frontmatter counts like any other. The number is over the whole vault,
    /// not over the scope this listing names.
    pub links_in: usize,
    /// How many chunks the note is indexed as, and so how many distinct units
    /// `search` can answer it with (#131). It is not a size: a note of four
    /// chunks may hold 300 tokens or 3000.
    pub chunk_count: usize,
    /// The note's indexed size in tokens, summed over its chunks (#131).
    /// Read beside `links_in`: a note many others point at that holds little
    /// is the one to write, and a long note of few chunks is the one to
    /// section. `store::ListRow` states what the number counts.
    pub token_count: usize,
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
    pub tagged_notes: usize,
    pub tag_axes: usize,
    pub top_notes: Vec<TopNote>,
    pub recent_files: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct TopNote {
    pub path: String,
    pub links_in: usize,
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
    if let Some(f) = params.store.find_file_by_basename(file_or_docid)? {
        return Ok(Some(f));
    }

    // Alias last (#142), so a note's filename beats another note's alias.
    params.store.find_file_by_alias(file_or_docid)
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
///
/// A failed property read is the caller's error, the policy the
/// `properties` module states: `context_read` returns `Result`, so it
/// carries one rather than reporting a link no property names.
fn link_refs<T>(
    store: &Store,
    edges: &[(i64, T)],
    names: impl Fn(i64) -> Result<Vec<String>>,
) -> Result<Vec<LinkRef>> {
    edges
        .iter()
        .filter_map(|(fid, _)| store.get_file_by_id(*fid).ok().flatten())
        .map(|f| {
            Ok(LinkRef {
                properties: names(f.id)?,
                path: f.path,
                docid: f.docid,
            })
        })
        .collect()
}

/// Read a note, in one of four modes (#80, #130).
///
/// `content` is the default and the cheapest: the whole note's body with the
/// frontmatter stripped, or one section's markdown. `frontmatter` is the
/// note's YAML alone. `all` is both, which is the call a caller makes to
/// narrate from a note whose frontmatter carries canon the prose does not
/// repeat. `metadata` is everything that is not prose — the frontmatter, the
/// link graph, the properties and the size.
///
/// Two of the four refuse a section, and for the same reason each: the answer
/// they give is the whole note's and could not narrow to one heading.
/// `metadata`'s link graph is note-level, and `frontmatter` carries no body at
/// all. `all` takes a section happily — frontmatter is note-level and a
/// section body is section-level, so returning both is no contradiction — and
/// that is the read the section case wanted, since a section's body otherwise
/// arrives with nothing but the path to say whose it is.
pub fn context_read(
    params: &ContextParams,
    file_or_docid: &str,
    section: Option<&str>,
    include: Include,
) -> Result<ReadResult> {
    if section.is_some() {
        match include {
            Include::Metadata => anyhow::bail!(Fault::InvalidInput(
                "--section cannot be combined with --include metadata: \
                 metadata describes the whole note"
                    .into()
            )),
            // Refused rather than ignored: a caller who meant `all` would
            // otherwise be handed a plausible answer with the section they
            // named silently dropped.
            Include::Frontmatter => anyhow::bail!(Fault::InvalidInput(
                "--section cannot be combined with --include frontmatter: \
                 frontmatter is the note's own and not a section's — \
                 use --include all for both"
                    .into()
            )),
            Include::Content | Include::All => {}
        }
    }

    let record = resolve_file(params, file_or_docid)?.ok_or_else(|| {
        anyhow::anyhow!(Fault::NotFound(format!("file not found: {file_or_docid}")))
    })?;

    let full_path = params.vault_path.join(&record.path);
    let disk = std::fs::read_to_string(&full_path).ok();

    if include == Include::Metadata {
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
                |to| params.store.property_names_for_link(id, to),
            )?,
            incoming_links: link_refs(
                params.store,
                &params.store.get_incoming(id, Some("wikilink"))?,
                |from| params.store.property_names_for_link(from, id),
            )?,
            properties: params.store.file_properties(id)?,
            byte_count,
        }));
    }

    // The note's YAML alone, with none of the link work above: the two
    // lookups and the property read are what makes a metadata read expensive,
    // and this mode exists because a caller wanting four lines of frontmatter
    // should not pay for the link graph to get them (#130).
    if include == Include::Frontmatter {
        return Ok(ReadResult::Frontmatter(NoteFrontmatter {
            path: record.path,
            docid: record.docid,
            // A row whose file is gone on disk has no frontmatter to report.
            // Empty rather than invented, the rule the metadata read follows.
            frontmatter: disk.map(|c| split_frontmatter(&c).0).unwrap_or_default(),
        }));
    }

    // `all` carries the frontmatter beside the content, and carries the key
    // even where the note has none: the caller asked for it, an empty string
    // is the honest answer, and an absent key would be indistinguishable from
    // a `content` reply.
    let frontmatter = (include == Include::All).then(|| {
        disk.as_deref()
            .map(|c| split_frontmatter(c).0)
            .unwrap_or_default()
    });

    // Content mode. A file the store holds and the disk does not answers the
    // re-index note in place of content, the way it always has (#62).
    let Some(content_str) = disk else {
        return Ok(ReadResult::Content(NoteContent {
            path: record.path,
            docid: record.docid,
            content: "[File not found on disk. Re-run 'knapper index' to update.]".to_string(),
            section: None,
            frontmatter,
        }));
    };

    // The whole note's body with the frontmatter stripped, or one section's
    // body with its heading named beside it (#80, #96). `find_section`
    // resolves a section by its heading text or its full heading path, and a
    // promoted bold line is one it reaches (#53, #69).
    let (content, span) = match section {
        None => (note_body(&content_str), None),
        Some(heading) => {
            let found = crate::markdown::find_section(&content_str, heading).ok_or_else(|| {
                anyhow::anyhow!(Fault::NotFound(format!("Section not found: {heading}")))
            })?;
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
        frontmatter,
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

/// The stored `indexed_at` as a timestamp a reader does not have to convert
/// (#121).
///
/// The column holds epoch seconds as text: `health` parses it back with
/// `parse::<u64>()` for index age and `list_recent` orders on it as text, so
/// the conversion belongs on the way out and not in the column. A value that
/// is not a number is passed through as it stands, so a row this cannot read
/// is reported rather than dropped.
fn indexed_at_iso(stored: &str) -> String {
    let Ok(seconds) = stored.parse::<i64>() else {
        return stored.to_string();
    };
    let Ok(dt) = time::OffsetDateTime::from_unix_timestamp(seconds) else {
        return stored.to_string();
    };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        dt.year(),
        dt.month() as u8,
        dt.day(),
        dt.hour(),
        dt.minute(),
        dt.second(),
    )
}

/// The notes a scope admits, in the order `order` names (#68, #121). A
/// caller's directory filter is a scope term, which is a case-sensitive range
/// and not a `LIKE`.
///
/// `detailed` costs one file read per listed note, and only then; an
/// undetailed listing touches no file. `limit` and `after` page the query
/// itself, so a page reads the files of its own notes and no others (#143).
pub fn context_list(
    params: &ContextParams,
    tags: &crate::tags::Scope,
    created_by: Option<&str>,
    limit: Option<usize>,
    after: Option<&str>,
    order: crate::store::ListOrder,
    detailed: bool,
) -> Result<Vec<NoteListItem>> {
    // The count comes back with the row rather than from a second query, so
    // the number a listing reports is the number it was ranked by (#121).
    let files = params
        .store
        .list_files_with_links_in(tags, created_by, limit, after, order)?;
    let file_ids: Vec<i64> = files.iter().map(|r| r.file.id).collect();
    let mut matched = matched_properties(params, tags, &file_ids)?;
    let mut aliases = params.store.aliases_for_files(&file_ids)?;
    let mut items = Vec::new();
    for row in files {
        let f = row.file;
        let headings = detailed.then(|| outline(params.vault_path, &f.path));
        let properties = matched
            .as_mut()
            .map(|m| m.remove(&f.id).unwrap_or_default());
        items.push(NoteListItem {
            path: f.path,
            docid: f.docid,
            tags: f.tags,
            aliases: aliases.remove(&f.id).unwrap_or_default(),
            indexed_at: indexed_at_iso(&f.indexed_at),
            links_in: row.links_in,
            chunk_count: row.chunk_count,
            token_count: row.token_count,
            headings,
            properties,
        });
    }
    Ok(items)
}

/// How many hubs the map names. Ten is enough to read a vault's subject off
/// and short enough that the map stays a map (#138).
const TOP_NOTES: usize = 10;

/// High-level vault overview: folders, the tag vocabulary and its reach, the
/// most-linked notes, recently changed files, counts.
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
    let tagged_notes = params.store.tagged_file_count()?;
    // How many facets the vocabulary spans, from the store's own axis rollup
    // (#60): one axis says the vault tags a single facet, which is what turns
    // a tag head count into something a caller can read a structure off.
    let tag_axes = params.store.tag_axes()?.len();

    // The ten notes the vault points at most. `chunk_count` and `token_count`
    // are deliberately not carried beside them: a map is a fixed shape, and a
    // caller who wants sizes has `list` (#131, #138).
    let top_notes: Vec<TopNote> = params
        .store
        .top_linked_files(TOP_NOTES)?
        .into_iter()
        .map(|(path, links_in)| TopNote { path, links_in })
        .collect();

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
        tagged_notes,
        tag_axes,
        top_notes,
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

    #[test]
    fn an_indexed_at_reads_as_a_timestamp_and_not_an_epoch() {
        assert_eq!(indexed_at_iso("1789013945"), "2026-09-10T04:19:05Z");
    }

    #[test]
    fn an_indexed_at_this_cannot_read_is_passed_through() {
        // Whatever the column holds reaches the caller. A row this drops is a
        // row a caller cannot tell from a note that was never indexed.
        assert_eq!(indexed_at_iso("not a number"), "not a number");
    }

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

    /// A content read in these tests, unwrapped. `content` and `all` both
    /// answer this variant; `all` fills its `frontmatter`.
    fn content_of(res: ReadResult) -> NoteContent {
        match res {
            ReadResult::Content(note) => note,
            other => panic!("expected content mode, got {other:?}"),
        }
    }

    /// The one metadata-mode read in these tests, unwrapped.
    fn metadata_of(res: ReadResult) -> NoteMetadata {
        match res {
            ReadResult::Metadata(meta) => meta,
            other => panic!("expected metadata mode, got {other:?}"),
        }
    }

    /// A frontmatter-mode read, unwrapped.
    fn frontmatter_of(res: ReadResult) -> NoteFrontmatter {
        match res {
            ReadResult::Frontmatter(fm) => fm,
            other => panic!("expected frontmatter mode, got {other:?}"),
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
        let note = content_of(context_read(&params, "note.md", None, Include::Content).unwrap());
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
        let note = content_of(
            context_read(&params, &format!("#{}", docid), None, Include::Content).unwrap(),
        );
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
        let note = content_of(context_read(&params, "ghost.md", None, Include::Content).unwrap());
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
        let note = content_of(context_read(&params, "note", None, Include::Content).unwrap());
        assert_eq!(note.path, "note.md");
    }

    // ── A caller names a note by its alias (#142) ───────────────

    /// Give a note of `setup_vault` its aliases, the rows the edge pass writes.
    fn give_aliases(store: &Store, path: &str, aliases: &[&str]) {
        let id = store.get_file(path).unwrap().unwrap().id;
        let aliases: Vec<String> = aliases.iter().map(|a| a.to_string()).collect();
        store.replace_file_aliases(id, &aliases).unwrap();
    }

    fn list_under(params: &ContextParams, scope: &crate::tags::Scope) -> Result<Vec<String>> {
        Ok(context_list(
            params,
            scope,
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )?
        .into_iter()
        .map(|item| item.path)
        .collect())
    }

    #[test]
    fn a_note_is_read_by_an_alias_it_carries() {
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "other.md", &["The Other One"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note =
            content_of(context_read(&params, "the other one", None, Include::Content).unwrap());
        assert_eq!(
            note.path, "other.md",
            "the reply names the note that answered"
        );
    }

    #[test]
    fn a_basename_beats_another_notes_alias_on_read() {
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "other.md", &["note"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = content_of(context_read(&params, "note", None, Include::Content).unwrap());
        assert_eq!(note.path, "note.md");
    }

    #[test]
    fn a_read_by_an_alias_two_notes_carry_is_refused_and_names_both() {
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "note.md", &["Twin"]);
        give_aliases(&store, "other.md", &["twin"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let err = context_read(&params, "Twin", None, Include::Content)
            .unwrap_err()
            .to_string();
        assert!(err.contains("note.md") && err.contains("other.md"), "{err}");
    }

    #[test]
    fn a_link_filter_takes_an_alias() {
        // `setup_vault` links note.md and other.md both ways.
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "other.md", &["Otto"]);
        give_aliases(&store, "note.md", &["Nora"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let to_otto = crate::tags::Scope::default()
            .with_filters(None, Some("otto"), None)
            .unwrap();
        assert_eq!(list_under(&params, &to_otto).unwrap(), ["note.md"]);
        let from_nora = crate::tags::Scope::default()
            .with_filters(None, None, Some("Nora"))
            .unwrap();
        assert_eq!(list_under(&params, &from_nora).unwrap(), ["other.md"]);
    }

    #[test]
    fn a_link_filter_naming_an_alias_two_notes_carry_is_refused_and_names_both() {
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "note.md", &["Twin"]);
        give_aliases(&store, "other.md", &["Twin"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let scope = crate::tags::Scope::default()
            .with_filters(None, Some("Twin"), None)
            .unwrap();
        let err = list_under(&params, &scope).unwrap_err().to_string();
        assert!(err.contains("note.md") && err.contains("other.md"), "{err}");
    }

    #[test]
    fn a_basename_beats_another_notes_alias_in_a_link_filter() {
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "note.md", &["other"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let scope = crate::tags::Scope::default()
            .with_filters(None, Some("other"), None)
            .unwrap();
        // Notes linking to other.md, which is note.md — not notes linking
        // to note.md, which would be other.md.
        assert_eq!(list_under(&params, &scope).unwrap(), ["note.md"]);
    }

    #[test]
    fn a_listing_carries_each_notes_aliases() {
        let (_tmp, store, root) = setup_vault();
        give_aliases(&store, "other.md", &["Otto", "The Other One"]);
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
        let other = items.iter().find(|i| i.path == "other.md").unwrap();
        assert_eq!(other.aliases, ["Otto", "The Other One"]);
        let note = items.iter().find(|i| i.path == "note.md").unwrap();
        assert!(note.aliases.is_empty());
        // The field is there when it is empty, as `tags` is.
        let json = serde_json::to_value(note).unwrap();
        assert_eq!(json["aliases"], serde_json::json!([]));
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
        let note = content_of(context_read(&params, "note.md", None, Include::Content).unwrap());
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
        let res = context_read(&params, "note.md", None, Include::Content).unwrap();
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
        let meta = metadata_of(context_read(&params, "note.md", None, Include::Metadata).unwrap());
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
        assert!(context_read(&params, "note.md", Some("Note"), Include::Metadata).is_err());
    }

    /// `frontmatter` answers the YAML alone. It is the cheap per-note
    /// property check `metadata` is too heavy for, so the link graph — the
    /// bulk of a metadata read — must not be in it (#130).
    #[test]
    fn frontmatter_mode_answers_the_yaml_with_no_link_graph() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let res = context_read(&params, "note.md", None, Include::Frontmatter).unwrap();
        let fm = frontmatter_of(res);
        assert_eq!(fm.path, "note.md");
        assert!(fm.frontmatter.contains("tags:"));

        let json = serde_json::to_string(
            &context_read(&params, "note.md", None, Include::Frontmatter).unwrap(),
        )
        .unwrap();
        for absent in [
            "outgoing_links",
            "incoming_links",
            "byte_count",
            "properties",
            "content",
        ] {
            assert!(
                !json.contains(absent),
                "frontmatter mode leaked {absent}: {json}"
            );
        }
    }

    /// `all` is the `cat` case: the prose and the frontmatter that qualifies
    /// it, in one call and with no link graph (#130).
    #[test]
    fn all_mode_answers_content_and_frontmatter_together() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = content_of(context_read(&params, "note.md", None, Include::All).unwrap());
        assert!(note.content.contains("Content here."));
        assert_eq!(
            note.frontmatter.as_deref().map(str::trim),
            Some("tags:\n  - rust".trim())
        );

        let json =
            serde_json::to_string(&context_read(&params, "note.md", None, Include::All).unwrap())
                .unwrap();
        for absent in ["outgoing_links", "incoming_links", "byte_count"] {
            assert!(!json.contains(absent), "all mode leaked {absent}: {json}");
        }
    }

    /// `all` returns the two halves apart. The content is the body it always
    /// was, so what `update` takes back is unchanged by asking for the YAML
    /// beside it (#96, #130).
    #[test]
    fn all_mode_keeps_the_content_frontmatter_stripped() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let all = content_of(context_read(&params, "note.md", None, Include::All).unwrap());
        let plain = content_of(context_read(&params, "note.md", None, Include::Content).unwrap());
        assert!(
            !all.content.contains("tags:"),
            "frontmatter leaked into content"
        );
        assert_eq!(all.content, plain.content);
    }

    /// A note with no YAML still carries the key in `all`, empty. The caller
    /// asked for the frontmatter, and "this note has none" is the answer;
    /// omitting the key would make `all` indistinguishable from `content`.
    #[test]
    fn all_mode_answers_an_empty_frontmatter_where_a_note_has_none() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note = content_of(context_read(&params, "other.md", None, Include::All).unwrap());
        assert_eq!(note.frontmatter.as_deref(), Some(""));
    }

    /// The union is most useful on a section read, where the path is
    /// otherwise the only thing saying whose section it is. Frontmatter is
    /// note-level and a section body is section-level, so there is nothing to
    /// refuse (#130).
    #[test]
    fn a_section_read_carries_the_note_frontmatter_under_all() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let note =
            content_of(context_read(&params, "note.md", Some("Note"), Include::All).unwrap());
        assert!(note.content.contains("Content here."));
        assert!(
            note.frontmatter
                .as_deref()
                .is_some_and(|f| f.contains("tags:"))
        );
        assert_eq!(
            note.section.as_ref().map(|s| s.heading.as_str()),
            Some("Note")
        );
    }

    /// `frontmatter` answers the note's own YAML and no body, so a section
    /// beside it names something the answer cannot carry. Refused rather than
    /// ignored: a caller who meant `all` would otherwise get a plausible
    /// answer with the section silently dropped (#130).
    #[test]
    fn section_and_frontmatter_cannot_be_combined() {
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let err = context_read(&params, "note.md", Some("Note"), Include::Frontmatter).unwrap_err();
        assert!(
            err.to_string().contains("all"),
            "the refusal should name the mode that answers both: {err}"
        );
    }

    /// A row whose file is gone on disk has no frontmatter to report, which
    /// is the rule the metadata read already follows: empty rather than
    /// invented.
    #[test]
    fn frontmatter_mode_answers_empty_for_a_file_gone_from_disk() {
        let (_tmp, store, root) = setup_vault();
        store
            .insert_file("ghost.md", "h3", 100, "ggg333", None, None)
            .unwrap();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let fm =
            frontmatter_of(context_read(&params, "ghost.md", None, Include::Frontmatter).unwrap());
        assert_eq!(fm.frontmatter, "");
    }

    /// Both size numbers reach the surface beside `links_in`, so a caller can
    /// weigh what points at a note against what the note holds — in one call
    /// and with no client-side join (#131).
    #[test]
    fn a_listing_carries_each_notes_size_beside_its_links_in() {
        let (_tmp, store, root) = setup_vault();
        let note = store.get_file("note.md").unwrap().unwrap().id;
        for (seq, tokens) in [(0i64, 120i64), (1, 80)] {
            store
                .insert_chunk(&crate::store::NewChunk {
                    file_id: note,
                    seq,
                    text: "body",
                    vector_id: seq as u64 + 1,
                    token_count: tokens,
                    ..Default::default()
                })
                .unwrap();
        }
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let items = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
        let listed = items.iter().find(|i| i.path == "note.md").unwrap();
        assert_eq!(listed.chunk_count, 2);
        assert_eq!(listed.token_count, 200);
        // The note with no chunks still reports a size, and it is zero.
        let other = items.iter().find(|i| i.path == "other.md").unwrap();
        assert_eq!(other.chunk_count, 0);
        assert_eq!(other.token_count, 0);
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
            None,
            crate::store::ListOrder::Path,
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
            None,
            crate::store::ListOrder::Path,
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
            None,
            crate::store::ListOrder::Path,
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
        let items = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            true,
        )
        .unwrap();
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
        let items = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            true,
        )
        .unwrap();
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
        let items = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
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
    fn vault_map_names_the_notes_the_vault_points_at() {
        // Folder counts answer where notes are filed. `top_notes` answers
        // which ones matter, which is the question an orienting caller has
        // and the one that used to take a second `list` call (#138).
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let store = Store::open_memory().unwrap();
        let mut ids = Vec::new();
        for name in ["hub.md", "mid.md", "leaf.md", "lonely.md"] {
            let id = store
                .insert_file(name, "h", 100, &generate_docid(name), None, None)
                .unwrap();
            ids.push(id);
        }
        let (hub, mid, leaf) = (ids[0], ids[1], ids[2]);
        store
            .insert_edge(leaf, DOC_LEVEL, hub, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(mid, DOC_LEVEL, hub, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(leaf, DOC_LEVEL, mid, DOC_LEVEL, "wikilink")
            .unwrap();

        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let map = vault_map(&params).unwrap();

        assert_eq!(
            map.top_notes,
            vec![
                TopNote {
                    path: "hub.md".into(),
                    links_in: 2
                },
                TopNote {
                    path: "mid.md".into(),
                    links_in: 1
                },
            ]
        );
    }

    #[test]
    fn vault_map_gives_the_tag_counts_a_denominator() {
        // `top_tags` alone reads against `total_files` and overstates the
        // vocabulary's reach; `tag_axes` says whether the vocabulary is a
        // structure or one facet (#138).
        let (_tmp, store, root) = setup_vault();
        let params = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let map = vault_map(&params).unwrap();

        // One of the two notes carries a tag, on the single axis `rust`.
        assert_eq!(map.total_files, 2);
        assert_eq!(map.tagged_notes, 1);
        assert_eq!(map.tag_axes, 1);
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

        let whole = content_of(context_read(&params, "person.md", None, Include::Content).unwrap());
        let part = content_of(
            context_read(&params, "person.md", Some("Interactions"), Include::Content).unwrap(),
        );

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
        assert!(context_read(&params, "person.md", Some("Nope"), Include::Content).is_err());
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

        let part = content_of(
            context_read(&params, "creature.md", Some("Spells"), Include::Content).unwrap(),
        );
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
            content_of(context_read(&params, "repro.md", Some("Alpha"), Include::Content).unwrap())
                .content;
        let written = crate::writer::apply_note_edits(
            &original,
            &[crate::writer::NoteEdit {
                target: crate::writer::EditTarget::Section("Alpha".into()),
                heading: None,
                mode: crate::writer::EditMode::Replace,
                content: Some(crate::writer::EditContent::Text(body)),
                placement: crate::frontmatter::KeyPlacement::End,
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
        let meta =
            metadata_of(context_read(&params, "person.md", None, Include::Metadata).unwrap());
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

        let body =
            content_of(context_read(&params, "repro.md", None, Include::Content).unwrap()).content;
        let written = crate::writer::apply_note_edits(
            &original,
            &[crate::writer::NoteEdit {
                target: crate::writer::EditTarget::Body,
                heading: None,
                mode: crate::writer::EditMode::Replace,
                content: Some(crate::writer::EditContent::Text(body)),
                placement: crate::frontmatter::KeyPlacement::End,
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
        let meta = metadata_of(context_read(&params, "note.md", None, Include::Metadata).unwrap());
        let names: Vec<&str> = meta.properties.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["related", "status"]);
        assert_eq!(meta.outgoing_links[0].path, "other.md");
        assert_eq!(
            meta.outgoing_links[0].properties,
            vec!["related".to_string()]
        );
        assert!(meta.incoming_links[0].properties.is_empty());

        let other =
            metadata_of(context_read(&params, "other.md", None, Include::Metadata).unwrap());
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
        let items = context_list(
            &params,
            &scope,
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
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
        let items = context_list(
            &params,
            &scope,
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
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
        let plain = context_list(
            &params,
            &crate::tags::Scope::default(),
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
        assert!(plain.iter().all(|i| i.properties.is_none()));
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("\"properties\""), "{json}");

        let scope = crate::tags::Scope::default()
            .with_filters(Some("status=draft"), None, None)
            .unwrap();
        let items = context_list(
            &params,
            &scope,
            None,
            None,
            None,
            crate::store::ListOrder::Path,
            false,
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        let rows = items[0].properties.as_ref().unwrap();
        assert_eq!(
            rows.len(),
            1,
            "only the matched row, not every row: {rows:?}"
        );
        assert_eq!(rows[0].value, "draft");
    }

    /// `read` resolves through the store's order: a basename names the live
    /// note first and still reaches an archived note when it is the only one
    /// (#151).
    #[test]
    fn a_read_by_basename_prefers_the_live_note_and_still_reaches_an_archived_only_one() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().to_path_buf();
        let store = Store::open_memory()
            .unwrap()
            .with_archive_folder("04-Archive");
        for (path, text) in [
            ("lore/deeper/still/n.md", "# Live\n"),
            ("04-Archive/n.md", "# Old\n"),
            ("04-Archive/only.md", "# Only\n"),
        ] {
            std::fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
            std::fs::write(root.join(path), text).unwrap();
            store
                .insert_file(path, "h", 1, &generate_docid(path), None, None)
                .unwrap();
        }
        let ctx = ContextParams {
            store: &store,
            vault_path: &root,
            profile: None,
        };
        let read = |name: &str| {
            content_of(context_read(&ctx, name, None, crate::params::Include::Content).unwrap())
                .path
        };
        assert_eq!(read("n"), "lore/deeper/still/n.md");
        assert_eq!(read("only"), "04-Archive/only.md");
    }
}
