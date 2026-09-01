use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, bail};
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::chunker::{ChunkOptions, chunk_markdown, split_oversized_chunks};
use crate::docid::generate_docid;
use crate::indexer::build_edges_for_file;
use crate::links;
use crate::llm::{EmbedDoc, EmbedModel};
use crate::placement::{self, PlacementHints};
use crate::prefix::{DocContext, EmbedComposition};
use crate::profile::VaultProfile;
use crate::store::Store;

// ── Input / Output types ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CreateNoteInput {
    pub content: String,
    pub filename: String,
    pub type_hint: Option<String>,
    pub tags: Vec<String>,
    pub folder: Option<String>,
    pub created_by: String,
    pub auto_link: Option<bool>,
}

/// How one edit changes what it addresses. `Remove` is only for a property:
/// a section has no "remove" that is not a replace with nothing, so the body
/// and section paths reject it (#62). The enum is fieldless, so it is `Copy`
/// and an edit can hand its mode to a transform without a clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditMode {
    Replace,
    Prepend,
    Append,
    Remove,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EditResult {
    pub path: String,
    pub heading: String,
    pub mode: String,
}

/// What one edit addresses. A note has three addressable things: its body,
/// a section of its body, and a frontmatter property. One edit names one of
/// them, so two targets are unrepresentable rather than rejected (#62).
#[derive(Debug, Clone)]
pub enum EditTarget {
    /// The note's body. The content is the body **alone**: a body edit always
    /// keeps the note's frontmatter, so content that starts with its own
    /// `---` block gives the note two frontmatter blocks. Edit the
    /// frontmatter with `Property` edits in the same list (#62).
    Body,
    Section(String),
    Property(String),
}

/// What an edit writes. A property is scalar or list valued, and which one
/// it is decides whether `Replace` sets a value or a whole sequence.
#[derive(Debug, Clone)]
pub enum EditContent {
    Text(String),
    List(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct NoteEdit {
    pub target: EditTarget,
    pub mode: EditMode,
    pub content: Option<EditContent>,
}

#[derive(Debug, Clone)]
pub struct UpdateInput {
    pub file: String,
    pub edits: Vec<NoteEdit>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WriteResult {
    pub path: String,
    pub docid: String,
    pub tags: Vec<String>,
    pub links_added: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub links_suggested: Vec<String>,
    pub folder: String,
    pub confidence: f64,
    pub strategy: String,
    /// Why the note landed where it did. `create` fills it from placement;
    /// a write with no placement to explain leaves it empty.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

// ── Helper functions ────────────────────────────────────────────

/// Strip characters that are invalid in filenames, keeping alphanumeric, spaces, dashes, underscores, and dots.
pub fn generate_filename(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_' || *c == '.')
        .collect()
}

/// A caller-supplied filename, sanitized and given a `.md` extension. A bare
/// name gets `.md` appended; a name that already ends in `.md` is kept. The
/// caller names the file — knapper does not guess one from content, because
/// since #46 the filename is the breadcrumb root of every chunk (#47).
pub fn normalize_filename(name: &str) -> String {
    let cleaned = generate_filename(name);
    if cleaned.ends_with(".md") {
        cleaned
    } else {
        format!("{cleaned}.md")
    }
}

/// Split content into (frontmatter_string, body_string).
/// If no frontmatter, returns ("", content).
pub fn split_frontmatter(content: &str) -> (String, String) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (String::new(), content.to_string());
    }

    // Find the closing ---
    let after_open = &trimmed[3..];
    // Skip past any remaining dashes and the newline
    let after_open = after_open.trim_start_matches('-');
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);

    if let Some(end_pos) = after_open.find("\n---") {
        let fm_content = &after_open[..end_pos];
        let rest_start = end_pos + 4; // "\n---"
        let rest = &after_open[rest_start..];
        // Skip trailing dashes and newline after closing ---
        let rest = rest.trim_start_matches('-');
        let rest = rest.strip_prefix('\n').unwrap_or(rest);

        let fm = format!("---\n{}\n---\n", fm_content);
        (fm, rest.to_string())
    } else {
        (String::new(), content.to_string())
    }
}

/// Parse frontmatter YAML string (without the --- delimiters) into a map of
/// scalar fields plus separate lists for `tags` and `aliases`.
///
/// Returns (scalars, tags, aliases).
pub(crate) fn parse_frontmatter_fields(
    fm_block: &str,
) -> (BTreeMap<String, String>, Vec<String>, Vec<String>) {
    let mut scalars: BTreeMap<String, String> = BTreeMap::new();
    let mut tags: Vec<String> = Vec::new();
    let mut aliases: Vec<String> = Vec::new();

    // Strip the --- delimiters
    let inner = fm_block
        .trim()
        .strip_prefix("---")
        .unwrap_or(fm_block)
        .trim_start_matches('-')
        .trim();
    let inner = inner.strip_suffix("---").unwrap_or(inner).trim();

    if inner.is_empty() {
        return (scalars, tags, aliases);
    }

    // Try to parse as YAML via serde_yaml
    if let Ok(yaml) = serde_yaml::from_str::<serde_yaml::Value>(inner)
        && let Some(map) = yaml.as_mapping()
    {
        for (k, v) in map {
            let key = match k.as_str() {
                Some(s) => s.to_string(),
                None => continue,
            };
            match key.as_str() {
                "tags" => {
                    if let Some(seq) = v.as_sequence() {
                        for item in seq {
                            if let Some(s) = item.as_str() {
                                tags.push(s.to_string());
                            }
                        }
                    } else if let Some(s) = v.as_str() {
                        // Handle inline `tags: foo` or `tags: [a, b]` parsed as string
                        for t in s.split(',') {
                            let t = t.trim();
                            if !t.is_empty() {
                                tags.push(t.to_string());
                            }
                        }
                    }
                }
                "aliases" => {
                    if let Some(seq) = v.as_sequence() {
                        for item in seq {
                            if let Some(s) = item.as_str() {
                                aliases.push(s.to_string());
                            }
                        }
                    } else if let Some(s) = v.as_str() {
                        for a in s.split(',') {
                            let a = a.trim();
                            if !a.is_empty() {
                                aliases.push(a.to_string());
                            }
                        }
                    }
                }
                _ => {
                    // Serialize value back to a string representation
                    let val_str = match v {
                        serde_yaml::Value::String(s) => s.clone(),
                        serde_yaml::Value::Number(n) => n.to_string(),
                        serde_yaml::Value::Bool(b) => b.to_string(),
                        serde_yaml::Value::Null => String::new(),
                        other => {
                            // serde_yaml may parse dates/timestamps as tagged
                            // values. Serialize and clean up the output.
                            let raw = serde_yaml::to_string(other)
                                .unwrap_or_default()
                                .trim_start_matches("---")
                                .trim()
                                .to_string();
                            // Strip YAML sequence prefix artifacts (e.g., "- - 2026-03-31" → "2026-03-31")
                            raw.trim_start_matches("- ").trim().to_string()
                        }
                    };
                    if !val_str.is_empty() {
                        scalars.insert(key, val_str);
                    }
                }
            }
        }
    }

    (scalars, tags, aliases)
}

/// Returns today's date as "YYYY-MM-DD".
pub fn today_date() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}",
        now.year(),
        now.month() as u8,
        now.day()
    )
}

/// Compute SHA-256 hash of content bytes, returned as hex string.
fn compute_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Get file mtime as seconds since epoch.
fn file_mtime(path: &Path) -> Result<i64> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Ok(mtime.as_secs() as i64)
}

/// Pre-computed chunk data ready for store insertion.
///
/// Named fields rather than a tuple because `text` and `snippet` are adjacent
/// A chunk ready for insertion.
///
/// It used to carry the snippet alongside the text, two `String`s bound for
/// different columns — which is how one transposition made chunk text
/// unrecoverable (issue #11). Since #14 the store derives the snippet from the
/// text, so there is only one to pass and nothing to transpose.
struct ChunkData {
    heading: String,
    /// The whole chunk, for `chunks.text` — which is also what the keyword
    /// index reads, since it is external content over that table (issue #37).
    text: String,
    /// What the keyword lane indexes beside the body (issue #37).
    lexical: crate::prefix::LexicalFields,
    vector: Vec<f32>,
    token_count: i64,
}

impl ChunkData {
    /// The row to write, less the vector id the store assigns.
    fn record<'a>(&'a self, file_id: i64, seq: i64, vector_id: u64) -> crate::store::NewChunk<'a> {
        crate::store::NewChunk {
            file_id,
            seq,
            heading: &self.heading,
            heading_path: &self.lexical.heading_path,
            tags_text: &self.lexical.tags_text,
            text: &self.text,
            vector_id,
            token_count: self.token_count,
        }
    }
}

/// Chunk content, embed, and return pre-computed data ready for store insertion.
///
/// `embed` must match what [`crate::indexer::index_file`] used, or notes
/// written through this path land in the same vector space as the indexed ones
/// while having been embedded a different way. That is why both settings travel
/// as one [`EmbedComposition`] and share one composition function.
///
/// `chunk_opts` is `[chunk_min_chars]` and `[promote_bold_headings]` and it has
/// to match for the same reason, one step earlier: at a different pair this
/// path writes different *rows* for the file than a re-index of it would
/// (issue #43).
fn precompute_chunks(
    rel_path: &str,
    content: &str,
    embedder: &mut impl EmbedModel,
    embed: EmbedComposition,
    chunk_opts: ChunkOptions,
) -> Result<Vec<ChunkData>> {
    let parsed = chunk_markdown(content, chunk_opts);
    let chunks = split_oversized_chunks(
        parsed.chunks,
        &|s| embedder.token_count(s),
        embedder.max_context(),
        crate::chunker::OVERLAP_TOKENS,
    );

    let doc = DocContext::from_file(rel_path, content);
    let inputs = crate::prefix::embed_inputs(&doc, &chunks, embed);
    let docs: Vec<EmbedDoc<'_>> = inputs
        .iter()
        .map(crate::prefix::EmbedInput::as_doc)
        .collect();
    let embeddings = embedder.embed_batch(&docs)?;
    // Composed from the same `doc` as the embed inputs, so a note written here
    // and the same note indexed by `index_file` carry one breadcrumb (issue #37).
    let lexical = crate::prefix::lexical_fields(&doc, &chunks, embed.root);

    let mut results = Vec::with_capacity(chunks.len());
    for ((chunk, embedding), lexical) in chunks.into_iter().zip(embeddings).zip(lexical) {
        results.push(ChunkData {
            heading: chunk.heading.unwrap_or_default(),
            token_count: chunk.text.split_whitespace().count() as i64,
            text: chunk.text,
            lexical,
            vector: embedding,
        });
    }
    Ok(results)
}

/// Write content to a temp file and atomically rename to final path.
/// Returns error if final_path already exists and `allow_overwrite` is false.
fn atomic_write(final_path: &Path, content: &str, allow_overwrite: bool) -> Result<()> {
    if !allow_overwrite && final_path.exists() {
        bail!(
            "file already exists at {}, refusing to overwrite",
            final_path.display()
        );
    }

    // Ensure parent directory exists
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let temp_path = final_path.with_extension("md.tmp");
    std::fs::write(&temp_path, content)?;
    std::fs::rename(&temp_path, final_path)?;
    Ok(())
}

/// Clean up incomplete writes from a previous crash.
/// Scans vault for .md.tmp files and removes them.
pub fn cleanup_temp_files(vault_path: &Path) -> Result<usize> {
    let mut cleaned = 0;
    for entry in WalkBuilder::new(vault_path).standard_filters(true).build() {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path.extension().is_some_and(|e| e == "tmp")
            && path.to_string_lossy().ends_with(".md.tmp")
        {
            std::fs::remove_file(path)?;
            cleaned += 1;
        }
    }
    Ok(cleaned)
}

// ── Pipeline functions ──────────────────────────────────────────

/// Create a new note via the 5-step write pipeline.
pub fn create_note(
    input: CreateNoteInput,
    store: &Store,
    embedder: &mut impl EmbedModel,
    embed: EmbedComposition,
    chunk_opts: ChunkOptions,
    vault_path: &Path,
    profile: Option<&VaultProfile>,
) -> Result<WriteResult> {
    // Step 1: Sanitize the caller's filename and ensure a `.md` extension.
    let filename = normalize_filename(&input.filename);

    // Step 2: Resolve tags
    let resolved_tags = store.resolve_tags(&input.tags)?;

    // Step 3: Discover links and apply them (unless auto_link is explicitly false)
    let people_folder = profile.and_then(|p| p.structure.folders.people.as_deref());
    let discovered = links::discover_links(store, &input.content, vault_path, people_folder)?;

    // Split discovered links into auto-apply and suggestion-only
    let (auto_apply, suggestions): (Vec<_>, Vec<_>) = if input.auto_link.unwrap_or(true) {
        discovered.into_iter().partition(|l| match &l.match_type {
            links::LinkMatchType::ExactName | links::LinkMatchType::Alias => true,
            links::LinkMatchType::FuzzyName { confidence_bp } => *confidence_bp >= 920,
            links::LinkMatchType::FirstName { .. } => false,
        })
    } else {
        // auto_link disabled: all discovered links go to suggestions only
        (Vec::new(), discovered)
    };

    let links_added: Vec<String> = auto_apply.iter().map(|l| l.target_path.clone()).collect();
    let links_suggested: Vec<String> = suggestions
        .iter()
        .map(|l| {
            let target_name = l
                .target_path
                .rsplit('/')
                .next()
                .unwrap_or(&l.target_path)
                .trim_end_matches(".md");
            if let Some(ref display) = l.display {
                format!("[[{}|{}]]", target_name, display)
            } else {
                format!("[[{}]]", target_name)
            }
        })
        .collect();

    // Apply auto-apply links to content via apply_links (respects protected regions)
    let content_with_links = links::apply_links(&input.content, &auto_apply);

    // Step 4: Determine folder placement
    let placement_result = if let Some(ref folder) = input.folder {
        placement::PlacementResult {
            folder: folder.clone(),
            confidence: 1.0,
            strategy: placement::PlacementStrategy::TypeRule,
            reason: "Explicit folder".to_string(),
            suggestion: None,
        }
    } else {
        let hints = PlacementHints {
            type_hint: input.type_hint.clone(),
            tags: resolved_tags.clone(),
        };
        placement::place_note(&content_with_links, &hints, profile, store, Some(embedder))?
    };

    // Step 5: The caller's frontmatter is the note's frontmatter. The only
    // key create writes is `tags`, and only what `--tags` resolved to (#92).
    let mut block = crate::frontmatter::Block::parse_or_open(&content_with_links)?;
    for tag in &resolved_tags {
        block.add_to_list("tags", tag)?;
    }
    let full_content = if block.is_empty() {
        content_with_links.clone()
    } else {
        block.render()
    };

    let rel_path = format!("{}/{}", placement_result.folder, filename);
    let final_path = vault_path.join(&rel_path);

    // Check for existing file before doing expensive work
    if final_path.exists() {
        bail!(
            "file already exists at {}; use update to change an existing note, not create",
            final_path.display()
        );
    }

    // Step 6: Pre-compute chunks + embeddings BEFORE transaction
    let chunk_data = precompute_chunks(&rel_path, &full_content, embedder, embed, chunk_opts)?;

    let content_hash = compute_content_hash(&full_content);
    let docid = generate_docid(&rel_path);

    // Write to temp file first
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = final_path.with_extension("md.tmp");
    std::fs::write(&temp_path, &full_content)?;

    // Step 7: BEGIN IMMEDIATE transaction
    store.begin_transaction()?;
    let result = (|| -> Result<i64> {
        let mtime = file_mtime(&temp_path).unwrap_or(0);
        let file_id = store.insert_file(
            &rel_path,
            &content_hash,
            mtime,
            &docid,
            Some(&input.created_by),
            None,
        )?;

        let start_vid = store.next_vector_id()?;
        for (chunk_seq, c) in chunk_data.iter().enumerate() {
            let vid = start_vid + chunk_seq as u64;
            store.insert_chunk_with_vector(&c.record(file_id, chunk_seq as i64, vid), &c.vector)?;
            store.insert_vec(vid, &c.vector)?;
        }

        build_edges_for_file(store, file_id, &full_content)?;

        // The writer is not an author of the vocabulary (#60). It writes the
        // file; the reconciler owns the rows, and reads the tags back out of
        // the content that was written, so the property and the body are peers
        // here exactly as they are on the index path.
        store.reconcile_file_tags(file_id, &crate::tags::extract(&full_content))?;

        Ok(file_id)
    })();

    match result {
        Ok(_) => {
            // Step 8: COMMIT
            store.commit()?;
            // Step 9: Atomic rename temp → final
            std::fs::rename(&temp_path, &final_path)?;
            // Update stored mtime to match the actual file after rename
            // (OS may adjust mtime during rename)
            let actual_mtime = file_mtime(&final_path).unwrap_or(0);
            store.insert_file(
                &rel_path,
                &content_hash,
                actual_mtime,
                &docid,
                Some(&input.created_by),
                None,
            )?;

            // Incrementally update folder centroid with new note's mean vector
            {
                let folder = &placement_result.folder;
                let new_vecs: Vec<&[f32]> =
                    chunk_data.iter().map(|c| c.vector.as_slice()).collect();
                if !new_vecs.is_empty() {
                    let dim = new_vecs[0].len();
                    let mut mean_vec = vec![0.0f32; dim];
                    for v in &new_vecs {
                        for (i, val) in v.iter().enumerate() {
                            mean_vec[i] += val;
                        }
                    }
                    let n = new_vecs.len() as f32;
                    for val in &mut mean_vec {
                        *val /= n;
                    }
                    let _ = store.adjust_folder_centroid(folder, &mean_vec, true);
                }
            }
        }
        Err(e) => {
            let _ = store.rollback();
            let _ = std::fs::remove_file(&temp_path);
            return Err(e);
        }
    }

    let strategy_name = format!("{:?}", placement_result.strategy);
    Ok(WriteResult {
        path: rel_path,
        docid,
        tags: resolved_tags,
        links_added,
        links_suggested,
        folder: placement_result.folder,
        confidence: placement_result.confidence,
        strategy: strategy_name,
        reason: placement_result.reason.clone(),
    })
}

/// Apply one section edit to a note's text. The transform is separate from
/// the I/O so that `update_note` can apply a list of them to one string and
/// write once (#62).
///
/// Finds the target section by heading name, then applies the edit based on mode:
/// - Replace: replace the entire section body with new content
/// - Append: add new content at the end of the section body
/// - Prepend: add new content at the start of the section body
pub fn apply_section_edit(
    content: &str,
    heading: &str,
    new: &str,
    mode: EditMode,
) -> Result<String> {
    // Find the target section
    let section = crate::markdown::find_section(content, heading)
        .ok_or_else(|| anyhow::anyhow!("section '{}' not found", heading))?;

    // Apply the edit based on mode
    let lines: Vec<&str> = content.lines().collect();
    let before = &lines[..section.body_start];
    let body = &lines[section.body_start..section.body_end];
    let after = &lines[section.body_end..];

    let new_body = match mode {
        EditMode::Replace => {
            format!("\n{}\n", new.trim_end())
        }
        EditMode::Append => {
            let existing = body.join("\n");
            let trimmed_existing = existing.trim_end();
            if trimmed_existing.is_empty() {
                format!("\n{}\n", new.trim_end())
            } else {
                format!("{}\n{}\n", trimmed_existing, new.trim_end())
            }
        }
        EditMode::Prepend => {
            let existing = body.join("\n");
            let trimmed_existing = existing.trim_start();
            if trimmed_existing.is_empty() {
                format!("\n{}\n", new.trim_end())
            } else {
                format!("\n{}\n{}", new.trim_end(), trimmed_existing)
            }
        }
        EditMode::Remove => bail!("Remove has no meaning for a section"),
    };

    // Reconstruct the file
    let mut result_parts: Vec<String> = Vec::new();
    if !before.is_empty() {
        result_parts.push(before.join("\n"));
    }
    result_parts.push(new_body);
    if !after.is_empty() {
        result_parts.push(after.join("\n"));
    }
    // Join with newlines, ensuring we don't double up
    Ok(result_parts.join("\n"))
}

/// The display name of an edit mode, for `EditResult::mode`.
fn edit_mode_name(mode: &EditMode) -> &'static str {
    match mode {
        EditMode::Replace => "Replace",
        EditMode::Append => "Append",
        EditMode::Prepend => "Prepend",
        EditMode::Remove => "Remove",
    }
}

/// Apply a body edit to a note's text. The transform is separate from the I/O
/// so that `update_note` can apply a list of them to one string and write once
/// (#62).
///
/// With `preserve_frontmatter`, the frontmatter is split off, `mode` is
/// applied to the body alone, and the two are reassembled. Without it,
/// `Replace` returns `new` and `Append`/`Prepend` join the whole text.
///
/// `Remove` is a property mode and has no meaning for a body, so it returns
/// the text unchanged. `apply_note_edits` rejects it before it reaches here
/// and no other caller passes it, so the arm exists to keep a mode a body
/// cannot express from deleting one (#62).
pub fn apply_body_edit(
    content: &str,
    new: &str,
    mode: EditMode,
    preserve_frontmatter: bool,
) -> String {
    if mode == EditMode::Remove {
        return content.to_string();
    }
    if preserve_frontmatter {
        let (maybe_frontmatter, old_body) = crate::markdown::split_frontmatter(content);
        if let Some(frontmatter) = maybe_frontmatter {
            // `split_frontmatter` rejoins the body as `lines[i+1..].join("\n")`,
            // so a note written `---\nfm\n---\n\nbody` hands back a body that
            // still carries the break after the closing `---`. The reassembly
            // below writes its own `\n\n`, so this break has to go: keeping it
            // added one blank line to the note on every append, and they
            // accumulated call after call (#62).
            let old_body = old_body.trim_start_matches('\n');
            let new_body = match mode {
                EditMode::Replace => new.to_string(),
                EditMode::Append => format!("{}\n{}", old_body.trim_end(), new),
                EditMode::Prepend => format!("{}\n{}", new.trim_end(), old_body),
                EditMode::Remove => old_body.to_string(),
            };
            return format!("---\n{}\n---\n\n{}", frontmatter, new_body);
        }
        // No existing frontmatter to preserve — fall through to the whole-text
        // join below, so Append and Prepend still keep the note's own body.
    }
    match mode {
        EditMode::Replace => new.to_string(),
        EditMode::Append => format!("{}\n{}", content.trim_end(), new),
        EditMode::Prepend => format!("{}\n{}", new.trim_end(), content),
        EditMode::Remove => content.to_string(),
    }
}

/// The text one edit writes. A body and a section take one string, so a list
/// is content for a property alone (#62).
fn text_of(edit: &NoteEdit) -> Result<String> {
    match &edit.content {
        Some(EditContent::Text(t)) => Ok(t.clone()),
        Some(EditContent::List(_)) => {
            bail!("a list is content for a property and not for a body or a section")
        }
        None => bail!(
            "a {} edit of a body or a section needs content",
            edit_mode_name(&edit.mode)
        ),
    }
}

/// Apply one property edit to `block`. Every mode names one operation on
/// one key, so an append to `status` cannot reach the `tags` list (#62).
fn apply_property_edit(
    block: &mut crate::frontmatter::Block,
    key: &str,
    mode: EditMode,
    content: Option<&EditContent>,
) -> Result<()> {
    match (mode, content) {
        (EditMode::Replace, Some(EditContent::Text(v))) => block.set_scalar(key, v),
        (EditMode::Replace, Some(EditContent::List(vs))) => block.set_list(key, vs),
        (EditMode::Append, Some(EditContent::Text(v))) => block.add_to_list(key, v),
        (EditMode::Remove, None) => block.remove(key),
        (EditMode::Remove, Some(EditContent::Text(v))) => block.remove_from_list(key, v),
        (mode, content) => anyhow::bail!(
            "a {} on property '{key}' with {} content has no meaning",
            edit_mode_name(&mode),
            if content.is_some() { "this" } else { "no" }
        ),
    }
}

/// Apply every edit to a note's text, in order. Pure, so `update_note` can
/// write the result once — one file write, one conflict check and one
/// re-index for a whole batch (#62).
///
/// A run of property edits becomes one pass over one `Block`, so the run is
/// one parse and one render. The block splices onto the note's own body, so
/// no number of property edits moves the body a line.
pub fn apply_note_edits(content: &str, edits: &[NoteEdit]) -> Result<String> {
    let mut text = content.to_string();
    let mut rest = edits;
    while let Some(edit) = rest.first() {
        match &edit.target {
            EditTarget::Property(_) => {
                let run = rest
                    .iter()
                    .position(|e| !matches!(e.target, EditTarget::Property(_)))
                    .unwrap_or(rest.len());
                let (properties, tail) = rest.split_at(run);
                let mut block = crate::frontmatter::Block::parse_or_open(&text)?;
                for property in properties {
                    let EditTarget::Property(key) = &property.target else {
                        bail!("a run of property edits holds property edits alone");
                    };
                    apply_property_edit(&mut block, key, property.mode, property.content.as_ref())?;
                }
                text = block.render();
                rest = tail;
            }
            EditTarget::Body => {
                if edit.mode == EditMode::Remove {
                    bail!("Remove has no meaning for a body");
                }
                let new = text_of(edit)?;
                text = apply_body_edit(&text, &new, edit.mode, true);
                rest = &rest[1..];
            }
            EditTarget::Section(heading) => {
                if edit.mode == EditMode::Remove {
                    bail!("Remove has no meaning for a section");
                }
                let new = text_of(edit)?;
                text = apply_section_edit(&text, heading, &new, edit.mode)?;
                rest = &rest[1..];
            }
        }
    }
    Ok(text)
}

/// Apply a list of edits to one note in one write.
///
/// The list is one write however long it is: one conflict check, one
/// `atomic_write` and one store update. `apply_note_edits` transforms the
/// text in memory, so an edit that fails part way through the list leaves
/// the file as it was — there is nothing to roll back, because nothing is
/// written until every edit applied (#62).
///
/// Does NOT re-index chunks — that is for the MCP layer, as with the other
/// edit calls.
pub fn update_note(store: &Store, vault_path: &Path, input: &UpdateInput) -> Result<EditResult> {
    // Step 1: Resolve file via store
    let file_record = store
        .resolve_file(&input.file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", input.file))?;

    let full_path = vault_path.join(&file_record.path);

    // Step 2: Mtime conflict check — one check for the whole list
    let disk_mtime = file_mtime(&full_path)?;
    if disk_mtime != file_record.mtime {
        bail!(
            "mtime conflict: file {} was modified outside knapper (disk={}, indexed={})",
            file_record.path,
            disk_mtime,
            file_record.mtime
        );
    }

    // Step 3: Apply every edit to the text the file holds
    let content = std::fs::read_to_string(&full_path)?;
    let new_content = apply_note_edits(&content, &input.edits)
        .map_err(|e| anyhow::anyhow!("{e} in {}", input.file))?;

    // Step 4: Write atomically — once
    atomic_write(&full_path, &new_content, true)?;

    // Step 5: Update the store's content hash, mtime and tag rows
    let content_hash = compute_content_hash(&new_content);
    let mtime = file_mtime(&full_path)?;
    let docid = file_record
        .docid
        .clone()
        .unwrap_or_else(|| generate_docid(&file_record.path));

    let file_id = store.insert_file(
        &file_record.path,
        &content_hash,
        mtime,
        &docid,
        file_record.created_by.as_deref(),
        None,
    )?;
    store.reconcile_file_tags(file_id, &crate::tags::extract(&new_content))?;

    Ok(EditResult {
        path: file_record.path,
        heading: String::new(),
        mode: "Update".to_string(),
    })
}

/// Move a note to a new folder.
pub fn move_note(
    file: &str,
    new_folder: &str,
    store: &Store,
    vault_path: &Path,
) -> Result<WriteResult> {
    // Step 1: Resolve file
    let file_record = store
        .resolve_file(file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", file))?;

    let old_path = vault_path.join(&file_record.path);
    let basename = file_record
        .path
        .rsplit('/')
        .next()
        .unwrap_or(&file_record.path);
    let new_rel_path = format!("{}/{}", new_folder, basename);
    let new_full_path = vault_path.join(&new_rel_path);

    if new_full_path.exists() {
        bail!("target path already exists: {}", new_full_path.display());
    }

    // The content does not change, so the stored hash still describes the file
    // and only the path-derived docid needs recomputing.
    let new_docid = generate_docid(&new_rel_path);

    // Ensure target directory exists
    if let Some(parent) = new_full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Step 2: Transaction — move the record to the new path.
    //
    // A move changes the folder, not the basename, and wikilinks resolve by
    // basename — so every `[[note]]` pointing here still resolves and every
    // edge stays valid. This used to delete the row and insert a fresh one,
    // which cascaded all those edges away *and* took the note's chunks with
    // them, leaving a moved note indexed but unsearchable (issue #27).
    // `update_file_path` is the primitive that exists for exactly this: it
    // keeps the id, so chunks, vectors, FTS rows and edges all follow.
    store.begin_transaction()?;
    let result = (|| -> Result<()> {
        store.update_file_path(&file_record.path, &new_rel_path, &new_docid)?;
        let mtime = file_mtime(&old_path)?;
        store.update_file_mtime(&new_rel_path, mtime)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            store.commit()?;
            // Step 3: Rename file on disk
            std::fs::rename(&old_path, &new_full_path)?;
        }
        Err(e) => {
            let _ = store.rollback();
            return Err(e);
        }
    }

    Ok(WriteResult {
        path: new_rel_path,
        docid: new_docid,
        tags: file_record.tags,
        links_added: vec![],
        links_suggested: vec![],
        folder: new_folder.to_string(),
        confidence: 1.0,
        strategy: "Move".to_string(),
        reason: String::new(),
    })
}

// ── Delete ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum DeleteMode {
    /// Move the file to the archive folder, update the store path.
    Soft,
    /// Remove the file from disk and purge all store data.
    Hard,
}

/// Delete a note from the vault.
///
/// - `Soft`: move the file to `archive_folder` and update the store record (path only).
///   The note remains on disk but is relocated. No index rebuild — it stays searchable
///   under its new path.
/// - `Hard`: remove the file from disk and call `store.delete_file_hard()` to purge all
///   associated chunks, edges, FTS, and vector data.
pub fn delete_note(
    store: &Store,
    vault_path: &Path,
    file: &str,
    mode: DeleteMode,
    archive_folder: &str,
) -> Result<()> {
    let file_record = store
        .resolve_file(file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", file))?;

    let old_path = vault_path.join(&file_record.path);

    match mode {
        DeleteMode::Soft => {
            // Build destination path inside archive_folder
            let basename = std::path::Path::new(&file_record.path)
                .file_name()
                .ok_or_else(|| {
                    anyhow::anyhow!("cannot determine filename for: {}", file_record.path)
                })?;
            let new_rel_path = format!(
                "{}/{}",
                archive_folder.trim_end_matches('/'),
                basename.to_string_lossy()
            );
            let new_full_path = vault_path.join(&new_rel_path);

            // Ensure target directory exists
            if let Some(parent) = new_full_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Move file on disk
            std::fs::rename(&old_path, &new_full_path)?;

            // Update store: remove old record, insert under new path
            let docid = file_record.docid.as_deref().unwrap_or("").to_string();
            let created_by = file_record.created_by.clone();
            let mtime = file_record.mtime;

            let content = std::fs::read_to_string(&new_full_path)?;
            let content_hash = compute_content_hash(&content);

            // The row is replaced, so the new id holds no links: read the ids
            // the old row releases, write the new row's rows, then prune (#60).
            let released_tags = store.file_tag_ids(file_record.id)?;
            store.delete_file(file_record.id)?;
            let file_id = store.insert_file(
                &new_rel_path,
                &content_hash,
                mtime,
                &docid,
                created_by.as_deref(),
                None,
            )?;
            store.reconcile_file_tags(file_id, &crate::tags::extract(&content))?;
            store.prune_unused_tags(&released_tags)?;

            Ok(())
        }
        DeleteMode::Hard => {
            // Delete disk file first, then purge store
            let released_tags = store.file_tag_ids(file_record.id)?;
            std::fs::remove_file(&old_path)?;
            store.delete_file_hard(&file_record.path)?;
            store.prune_unused_tags(&released_tags)?;
            Ok(())
        }
    }
}

// ── Archive / Unarchive ─────────────────────────────────────────

/// Archive a note: move to archive folder, add archived frontmatter, remove from index.
/// The note becomes invisible to search/context but is physically preserved.
pub fn archive_note(
    file: &str,
    store: &Store,
    vault_path: &Path,
    profile: Option<&crate::profile::VaultProfile>,
) -> Result<WriteResult> {
    let file_record = store
        .resolve_file(file)?
        .ok_or_else(|| anyhow::anyhow!("file not found: {}", file))?;

    let archive_folder = profile
        .and_then(|p| p.structure.folders.archive.as_deref())
        .unwrap_or("04-Archive");

    // Don't archive something already in the archive
    if file_record.path.starts_with(archive_folder) {
        bail!("note is already archived: {}", file_record.path);
    }

    let old_path = vault_path.join(&file_record.path);
    let new_rel_path = format!("{}/{}", archive_folder, file_record.path);
    let new_full_path = vault_path.join(&new_rel_path);

    // Archive's three keys go into the block the note already has, so every
    // key the note carried is still there when it comes back (#92).
    let content = std::fs::read_to_string(&old_path)?;
    let mut block = crate::frontmatter::Block::parse_or_open(&content)?;
    // A note that already holds one of these three keys cannot be archived
    // without losing something: overwriting the note's own value, or —
    // since `unarchive` removes exactly these three — leaving no way to
    // tell the note's own key from the one this write adds. Refusing is
    // the honest answer; the file is not touched (#92, I7).
    for key in ["archived", "archived_at", "archived_from"] {
        if block.value(key).is_some() {
            bail!(
                "note already holds `{key}`; knapper cannot archive it without losing that note's own value"
            );
        }
    }
    let tags = block.list("tags");
    block.set_bool("archived", true)?;
    block.set_scalar("archived_at", &today_date())?;
    block.set_scalar("archived_from", &file_record.path)?;
    let new_content = block.render();

    // Ensure target directory
    if let Some(parent) = new_full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write archived file to new location
    atomic_write(&new_full_path, &new_content, false)?;

    // Remove from index (note disappears from search)
    let old_vids = store.get_vector_ids_for_file(file_record.id)?;
    for vid in &old_vids {
        store.delete_vec(*vid)?;
    }
    let released_tags = store.file_tag_ids(file_record.id)?;
    store.delete_edges_for_file(file_record.id)?;
    store.delete_file(file_record.id)?;
    store.prune_unused_tags(&released_tags)?;

    // Remove original file
    std::fs::remove_file(&old_path)?;

    let docid = file_record.docid.unwrap_or_default();

    Ok(WriteResult {
        path: new_rel_path,
        docid,
        tags,
        links_added: vec![],
        links_suggested: vec![],
        folder: archive_folder.to_string(),
        confidence: 1.0,
        strategy: "Archive".to_string(),
        reason: String::new(),
    })
}

/// Unarchive a note: move back to original location, strip archive frontmatter, re-index.
pub fn unarchive_note(
    file: &str,
    store: &Store,
    embedder: &mut impl EmbedModel,
    embed: EmbedComposition,
    chunk_opts: ChunkOptions,
    vault_path: &Path,
) -> Result<WriteResult> {
    // Resolve — the file may not be in the index (archived notes are excluded).
    // Try resolving by direct path on disk.
    let archive_path = vault_path.join(file);
    if !archive_path.exists() {
        bail!("archived note not found: {}", file);
    }

    let content = std::fs::read_to_string(&archive_path)?;
    let mut block = crate::frontmatter::Block::parse(&content)?.ok_or_else(|| {
        anyhow::anyhow!("no archived_from in frontmatter — cannot determine original location")
    })?;
    let original_path = block.scalar("archived_from").ok_or_else(|| {
        anyhow::anyhow!("no archived_from in frontmatter — cannot determine original location")
    })?;

    let restore_full_path = vault_path.join(&original_path);

    if restore_full_path.exists() {
        bail!(
            "cannot unarchive: a file already exists at {}",
            original_path
        );
    }

    block.remove("archived")?;
    block.remove("archived_at")?;
    block.remove("archived_from")?;
    // A note archived by a version of knapper before #92 got `archived`
    // written into its own `tags` list, alongside the three keys above.
    // This build's `archive` no longer does that, but an old note still
    // carries the tag, and it must not come back into the vocabulary just
    // because the note is unarchived.
    block.remove_from_list("tags", "archived")?;
    let tags = block.list("tags");
    // A note that had no block before it was archived gets none back.
    // `is_empty` counts keys only, so a block holding just a comment or a
    // blank line still reports empty; checking `is_blank` instead keeps
    // those bytes rather than discarding the fences around them (#92, I2).
    let restored_content = if block.is_blank() {
        block.body().to_string()
    } else {
        block.render()
    };

    // Ensure target directory
    if let Some(parent) = restore_full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write restored file
    atomic_write(&restore_full_path, &restored_content, false)?;

    // Index the restored note
    let chunk_data = precompute_chunks(
        &original_path,
        &restored_content,
        embedder,
        embed,
        chunk_opts,
    )?;
    let content_hash = compute_content_hash(&restored_content);
    let docid = generate_docid(&original_path);
    let mtime = file_mtime(&restore_full_path).unwrap_or(0);

    store.begin_transaction()?;
    let result = (|| -> Result<()> {
        let file_id = store.insert_file(
            &original_path,
            &content_hash,
            mtime,
            &docid,
            Some("unarchive"),
            None,
        )?;

        let start_vid = store.next_vector_id()?;
        for (seq, c) in chunk_data.iter().enumerate() {
            let vid = start_vid + seq as u64;
            store.insert_chunk_with_vector(&c.record(file_id, seq as i64, vid), &c.vector)?;
            store.insert_vec(vid, &c.vector)?;
        }

        build_edges_for_file(store, file_id, &restored_content)?;

        store.reconcile_file_tags(file_id, &crate::tags::extract(&restored_content))?;

        Ok(())
    })();

    match result {
        Ok(()) => store.commit()?,
        Err(e) => {
            let _ = store.rollback();
            let _ = std::fs::remove_file(&restore_full_path);
            return Err(e);
        }
    }

    // Remove archived file
    std::fs::remove_file(&archive_path)?;

    let folder = original_path
        .rsplit_once('/')
        .map(|(f, _)| f.to_string())
        .unwrap_or_default();

    Ok(WriteResult {
        path: original_path,
        docid,
        tags,
        links_added: vec![],
        links_suggested: vec![],
        folder,
        confidence: 1.0,
        strategy: "Unarchive".to_string(),
        reason: String::new(),
    })
}

// ── Index integrity ─────────────────────────────────────────────

/// Verify that all indexed files still exist on disk.
/// Removes orphan DB entries for files that no longer exist.
/// Returns the number of orphan entries cleaned up.
pub fn verify_index_integrity(store: &Store, vault_path: &Path) -> Result<usize> {
    let all_files = store.get_all_files()?;
    let mut orphans = 0;
    for file in &all_files {
        let full_path = vault_path.join(&file.path);
        if !full_path.exists() {
            // Clean up orphan: vectors, edges, file record. The chunks and
            // their keyword index go with the `files` row (issue #37), and
            // so do the file's `file_tags` rows. The tag ids are read
            // before the cascade takes the links away, and each id no
            // other note holds is deleted after it, the way every other
            // removal path does it (#60).
            let released = store.file_tag_ids(file.id)?;
            let vids = store.get_vector_ids_for_file(file.id)?;
            for vid in &vids {
                store.delete_vec(*vid)?;
            }
            store.delete_edges_for_file(file.id)?;
            store.delete_file(file.id)?;
            store.prune_unused_tags(&released)?;
            orphans += 1;
        }
    }
    Ok(orphans)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_section_edit_is_a_pure_transform_of_the_text() {
        let doc = "# Note\n\n## Spells\n\nold body\n\n## Rank\n\nS\n";
        let out = apply_section_edit(doc, "Spells", "new body", EditMode::Replace).unwrap();
        assert!(out.contains("new body"));
        assert!(!out.contains("old body"));
        assert!(out.contains("## Rank"), "the rest of the note survives");
    }

    /// A promoted section is edited the way an ATX one is: the transform
    /// works from `body_start` and `body_end`, so the bold line stays where
    /// it is and the body under it is what changes (#69).
    #[test]
    fn a_replace_under_a_promoted_heading_keeps_the_bold_line() {
        let doc = "## Stat Block\n\nAC 20\n\n**Spells**\n\nFireball\n\n## Lore\n\nOld\n";
        let out = apply_section_edit(doc, "Spells", "Meteor", EditMode::Replace).unwrap();
        assert!(out.contains("**Spells**"));
        assert!(out.contains("Meteor"));
        assert!(!out.contains("Fireball"));
        assert!(out.contains("Old"));
        assert!(out.contains("AC 20"));
    }

    /// An empty section is what a caller fills, so an append to a bodyless
    /// promoted heading writes the first content under it (#69).
    #[test]
    fn an_append_fills_a_bodyless_promoted_section() {
        let doc = "## Stat Block\n\n**Spells**\n**Notes**\n\nSee below\n";
        let out = apply_section_edit(doc, "Spells", "Fireball", EditMode::Append).unwrap();
        let spells_at = out.find("**Spells**").expect("the bold line survives");
        let fireball_at = out.find("Fireball").expect("the new body is written");
        let notes_at = out.find("**Notes**").expect("the next section survives");
        assert!(spells_at < fireball_at && fireball_at < notes_at);
        assert!(out.contains("See below"));
    }

    /// A section the caller names by path is the section that is edited (#69).
    #[test]
    fn a_path_names_the_section_an_edit_reaches() {
        let doc =
            "# Empire\n\n## History\n\nFounding\n\n## Current Events\n\n### History\n\nRecent\n";
        let out = apply_section_edit(
            doc,
            "Empire > Current Events > History",
            "Newer",
            EditMode::Replace,
        )
        .unwrap();
        assert!(out.contains("Founding"));
        assert!(out.contains("Newer"));
        assert!(!out.contains("Recent"));
    }

    #[test]
    fn a_missing_section_is_an_error_and_not_a_silent_append() {
        let doc = "# Note\n\n## Spells\n\nbody\n";
        let err = apply_section_edit(doc, "Nowhere", "x", EditMode::Replace).unwrap_err();
        assert!(format!("{err}").contains("Nowhere"));
    }

    #[test]
    fn a_property_edit_keeps_the_key_in_place_and_in_its_style() {
        let note = "---\nname: Probe\naliases: []\ntags: [type/lore, realm/rudd]\n---\n\nBody.\n";
        let edits = vec![NoteEdit {
            target: EditTarget::Property("tags".into()),
            mode: EditMode::Replace,
            content: Some(EditContent::List(vec![
                "type/lore".into(),
                "realm/skaldi".into(),
            ])),
        }];
        assert_eq!(
            apply_note_edits(note, &edits).unwrap(),
            "---\nname: Probe\naliases: []\ntags: [type/lore, realm/skaldi]\n---\n\nBody.\n"
        );
    }

    #[test]
    fn a_property_replaced_with_an_empty_list_keeps_the_key() {
        let note = "---\nname: Probe\naliases: [Old]\n---\n\nBody.\n";
        let edits = vec![NoteEdit {
            target: EditTarget::Property("aliases".into()),
            mode: EditMode::Replace,
            content: Some(EditContent::List(vec![])),
        }];
        assert_eq!(
            apply_note_edits(note, &edits).unwrap(),
            "---\nname: Probe\naliases: []\n---\n\nBody.\n"
        );
    }

    #[test]
    fn a_run_of_property_edits_is_one_pass_and_does_not_move_the_body() {
        let note = "---\nname: Probe\ntags: [a]\n---\n\nBody.\n";
        let edits = vec![
            NoteEdit {
                target: EditTarget::Property("tags".into()),
                mode: EditMode::Append,
                content: Some(EditContent::Text("b".into())),
            },
            NoteEdit {
                target: EditTarget::Property("status".into()),
                mode: EditMode::Replace,
                content: Some(EditContent::Text("draft".into())),
            },
            NoteEdit {
                target: EditTarget::Property("name".into()),
                mode: EditMode::Remove,
                content: None,
            },
        ];
        assert_eq!(
            apply_note_edits(note, &edits).unwrap(),
            "---\ntags: [a, b]\nstatus: draft\n---\n\nBody.\n"
        );
    }

    #[test]
    fn a_body_edit_after_a_property_edit_still_sees_the_frontmatter() {
        let note = "---\nname: Probe\n---\n\nOld body.\n";
        let edits = vec![
            NoteEdit {
                target: EditTarget::Property("name".into()),
                mode: EditMode::Replace,
                content: Some(EditContent::Text("Renamed".into())),
            },
            NoteEdit {
                target: EditTarget::Body,
                mode: EditMode::Replace,
                content: Some(EditContent::Text("New body.".into())),
            },
        ];
        let out = apply_note_edits(note, &edits).unwrap();
        assert!(out.starts_with("---\nname: Renamed\n---\n"), "{out}");
        assert!(out.contains("New body."), "{out}");
        assert!(!out.contains("Old body."), "{out}");
    }

    #[test]
    fn a_property_edit_on_a_value_it_cannot_address_writes_nothing() {
        let note = "---\nname: Probe\nnested:\n  inner: 1\n---\n\nBody.\n";
        let edits = vec![NoteEdit {
            target: EditTarget::Property("nested".into()),
            mode: EditMode::Replace,
            content: Some(EditContent::Text("flat".into())),
        }];
        let err = apply_note_edits(note, &edits).unwrap_err();
        assert!(err.to_string().contains("nested mapping"), "{err}");
    }

    /// `split_frontmatter` rejoins the body as `lines[i+1..].join("\n")`, so
    /// the usual `---\nfm\n---\n\nbody` layout hands back a body that still
    /// carries the line break after the closing `---`. The reassembly below
    /// supplies its own `\n\n`, so adding both put one blank line into the
    /// note per append and they accumulated. The `write append` call this
    /// replaced added none, and `update`'s body append is byte-identical to it
    /// (#62).
    #[test]
    fn successive_body_appends_add_no_blank_line_of_their_own() {
        let doc = "---\ntags:\n  - a\n---\n\nbody line\n";

        let once = apply_body_edit(doc, "first", EditMode::Append, true);
        assert_eq!(once, "---\ntags:\n  - a\n---\n\nbody line\nfirst");

        let twice = apply_body_edit(&once, "second", EditMode::Append, true);
        assert_eq!(twice, "---\ntags:\n  - a\n---\n\nbody line\nfirst\nsecond");
    }

    /// The same text with and without the frontmatter split, which is what
    /// separated the two body calls `update` replaced. The two must agree byte
    /// for byte (#62).
    #[test]
    fn a_body_append_matches_the_call_it_replaces() {
        let doc = "---\ntags:\n  - a\n---\n\nbody line\n";
        assert_eq!(
            apply_body_edit(doc, "first", EditMode::Append, true),
            apply_body_edit(doc, "first", EditMode::Append, false),
        );
    }

    #[test]
    fn appending_with_preserve_frontmatter_on_a_note_with_none_keeps_the_body() {
        let doc = "old body\n";
        let out = apply_body_edit(doc, "new stuff", EditMode::Append, true);
        assert!(out.contains("old body"), "the existing body must survive");
        assert!(out.contains("new stuff"));
    }

    #[test]
    fn prepending_with_preserve_frontmatter_on_a_note_with_none_keeps_the_body() {
        let doc = "old body\n";
        let out = apply_body_edit(doc, "new stuff", EditMode::Prepend, true);
        assert!(out.contains("old body"), "the existing body must survive");
        assert!(out.contains("new stuff"));
    }

    #[test]
    fn test_generate_filename() {
        assert_eq!(generate_filename("My Great Note"), "My Great Note");
        assert_eq!(generate_filename("Note/With:Bad*Chars"), "NoteWithBadChars");
    }

    #[test]
    fn normalize_filename_accepts_a_bare_or_a_full_name() {
        assert_eq!(normalize_filename("my-file"), "my-file.md");
        assert_eq!(normalize_filename("my-file.md"), "my-file.md");
    }

    #[test]
    fn normalize_filename_strips_path_separators() {
        // The character filter is the write path's only defence against a
        // caller naming a file outside its folder, so pin it (#47 scope note).
        let out = normalize_filename("../etc/passwd");
        assert!(!out.contains('/'), "a slash survived: {out}");
        assert_eq!(out, "..etcpasswd.md");
    }

    #[test]
    fn test_split_frontmatter() {
        let content = "---\ntags: [a]\n---\n\nBody text";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.contains("tags"));
        assert_eq!(body.trim(), "Body text");
    }

    #[test]
    fn test_split_frontmatter_no_fm() {
        let content = "Just body text";
        let (fm, body) = split_frontmatter(content);
        assert!(fm.is_empty());
        assert_eq!(body, "Just body text");
    }

    #[test]
    fn test_cleanup_temp_files() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("note.md.tmp"), "incomplete").unwrap();
        std::fs::write(dir.path().join("good.md"), "complete").unwrap();
        std::fs::write(dir.path().join("other.tmp"), "not md tmp").unwrap();

        let cleaned = cleanup_temp_files(dir.path()).unwrap();
        assert_eq!(cleaned, 1);
        assert!(!dir.path().join("note.md.tmp").exists());
        assert!(dir.path().join("good.md").exists());
        assert!(dir.path().join("other.tmp").exists()); // .tmp but not .md.tmp
    }

    #[test]
    fn test_today_date_format() {
        let date = today_date();
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        assert_eq!(&date[7..8], "-");
    }

    #[test]
    fn test_verify_index_integrity() {
        let dir = tempfile::TempDir::new().unwrap();
        let vault = dir.path();
        std::fs::create_dir_all(vault.join("notes")).unwrap();
        std::fs::write(vault.join("notes/existing.md"), "# Exists").unwrap();

        let store = crate::store::Store::open_memory().unwrap();
        // Insert two files: one exists on disk, one does not
        store
            .insert_file(
                "notes/existing.md",
                "hash1",
                100,
                &crate::docid::generate_docid("notes/existing.md"),
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "notes/gone.md",
                "hash2",
                100,
                &crate::docid::generate_docid("notes/gone.md"),
                None,
                None,
            )
            .unwrap();

        let orphans = verify_index_integrity(&store, vault).unwrap();
        assert_eq!(orphans, 1);

        // The gone file should be removed from the store
        assert!(store.get_file("notes/gone.md").unwrap().is_none());
        // The existing file should still be there
        assert!(store.get_file("notes/existing.md").unwrap().is_some());
    }

    /// A note removed from disk releases its tags, and a tag no other note
    /// carries leaves the vocabulary with it (#60). `resolve_tag` reads
    /// `tags` with no join, so a row left behind keeps being offered as a
    /// match for a tag the vault no longer holds.
    #[test]
    fn verify_index_integrity_prunes_the_tags_the_removed_note_released() {
        let dir = tempfile::TempDir::new().unwrap();
        let vault = dir.path();
        std::fs::write(vault.join("kept.md"), "---\ntags: [shared]\n---\nbody\n").unwrap();

        let store = crate::store::Store::open_memory().unwrap();
        let kept = store
            .insert_file(
                "kept.md",
                "hash1",
                100,
                &crate::docid::generate_docid("kept.md"),
                None,
                None,
            )
            .unwrap();
        let gone = store
            .insert_file(
                "gone.md",
                "hash2",
                100,
                &crate::docid::generate_docid("gone.md"),
                None,
                None,
            )
            .unwrap();
        store
            .reconcile_file_tags(
                kept,
                &crate::tags::extract("---\ntags: [shared]\n---\nbody\n"),
            )
            .unwrap();
        store
            .reconcile_file_tags(
                gone,
                &crate::tags::extract("---\ntags: [shared, solitary]\n---\nbody\n"),
            )
            .unwrap();
        assert!(matches!(
            store.resolve_tag("solitary").unwrap(),
            crate::tags::TagResolution::Exact(_)
        ));

        assert_eq!(verify_index_integrity(&store, vault).unwrap(), 1);

        // The tag only the removed note carried is gone from the vocabulary.
        assert!(
            matches!(
                store.resolve_tag("solitary").unwrap(),
                crate::tags::TagResolution::New(_)
            ),
            "a tag whose only note was removed must not stay in the vocabulary"
        );
        // The tag the surviving note carries stays.
        assert!(matches!(
            store.resolve_tag("shared").unwrap(),
            crate::tags::TagResolution::Exact(_)
        ));
        assert_eq!(store.top_tags(10).unwrap(), vec![("shared".to_string(), 1)]);
    }

    #[test]
    fn test_compute_content_hash() {
        let h1 = compute_content_hash("hello");
        let h2 = compute_content_hash("hello");
        let h3 = compute_content_hash("world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }

    fn setup_vault() -> (tempfile::TempDir, Store, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::open_memory().unwrap();
        let root = tmp.path().to_path_buf();
        (tmp, store, root)
    }

    #[test]
    fn moving_a_note_keeps_it_indexed_and_keeps_its_backlinks() {
        // A move changes the folder, not the basename, so `[[hub]]` still
        // resolves — nothing about the graph should change. The old code
        // deleted the `files` row and inserted a fresh one, which cascaded the
        // backlink away and took the note's chunks with it, leaving a moved
        // note present in the index but absent from every search (issue #27).
        use crate::llm::MockLlm;

        let (_tmp, store, root) = setup_vault();
        std::fs::create_dir_all(root.join("inbox")).unwrap();
        std::fs::create_dir_all(root.join("areas")).unwrap();
        std::fs::write(root.join("inbox/hub.md"), "# Hub\nBody text here.\n").unwrap();
        std::fs::write(root.join("a.md"), "# A\nSee [[hub]].\n").unwrap();

        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            &root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let before = store.get_file("inbox/hub.md").unwrap().unwrap();
        assert_eq!(store.get_chunks_by_file(before.id).unwrap().len(), 1);
        assert_eq!(store.get_incoming(before.id, None).unwrap().len(), 1);

        move_note("inbox/hub.md", "areas", &store, &root).unwrap();

        assert!(root.join("areas/hub.md").exists());
        assert!(store.get_file("inbox/hub.md").unwrap().is_none());
        let after = store.get_file("areas/hub.md").unwrap().unwrap();
        assert_eq!(after.id, before.id, "a move must not re-key the file");
        assert_eq!(
            store.get_chunks_by_file(after.id).unwrap().len(),
            1,
            "a moved note must stay searchable"
        );
        assert_eq!(
            store.get_incoming(after.id, None).unwrap().len(),
            1,
            "[[hub]] resolves by basename, which the move did not change"
        );
        assert_eq!(after.docid, Some(generate_docid("areas/hub.md")));
    }

    #[test]
    fn test_delete_note_soft() {
        let (tmp, store, root) = setup_vault();
        std::fs::create_dir_all(root.join("04-Archive")).unwrap();
        std::fs::write(root.join("deleteme.md"), "# Delete me").unwrap();
        store
            .insert_file("deleteme.md", "hash", 100, "del123", None, None)
            .unwrap();

        delete_note(
            &store,
            &root,
            "deleteme.md",
            DeleteMode::Soft,
            "04-Archive/",
        )
        .unwrap();

        assert!(!root.join("deleteme.md").exists());
        assert!(root.join("04-Archive/deleteme.md").exists());
        drop(tmp);
    }

    #[test]
    fn test_delete_note_hard() {
        let (tmp, store, root) = setup_vault();
        std::fs::write(root.join("gone.md"), "# Gone forever").unwrap();
        store
            .insert_file("gone.md", "hash", 100, "gon123", None, None)
            .unwrap();

        delete_note(&store, &root, "gone.md", DeleteMode::Hard, "").unwrap();

        assert!(!root.join("gone.md").exists());
        assert!(store.get_file("gone.md").unwrap().is_none());
        drop(tmp);
    }

    #[test]
    fn test_parse_frontmatter_fields_empty() {
        let (scalars, tags, aliases) = parse_frontmatter_fields("");
        assert!(scalars.is_empty());
        assert!(tags.is_empty());
        assert!(aliases.is_empty());
    }

    /// Issue #11, on the write path. `precompute_chunks` is shared by
    /// `create_note` and `unarchive_note`, and both hand the same field to
    /// FTS — so this covers the wiring for both of them.
    #[test]
    fn precompute_keeps_the_whole_chunk_for_fts_and_truncates_only_the_snippet() {
        use crate::llm::MockLlm;

        let filler = "The coast road runs north through salt marsh and low dune. ".repeat(8);
        let content = format!("## The Coast Road\n\n{filler}\n\nIt ends at Saltmere.\n");
        let mut embedder = MockLlm::new(256);

        let data = precompute_chunks(
            "places/coast.md",
            &content,
            &mut embedder,
            EmbedComposition::default(),
            ChunkOptions {
                min_chars: 0,
                promote_bold: false,
                carry_orphan_headings: false,
            },
        )
        .unwrap();

        let c = &data[0];
        assert!(c.text.contains("Saltmere"), "text was truncated");

        // The write paths hand this text to the store, which keeps it whole and
        // derives the display snippet from it (issue #14).
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("places/coast.md", "h", 0, "d", None, None)
            .unwrap();
        store.insert_chunk(&c.record(file_id, 0, 1)).unwrap();

        let stored = store.get_chunk_by_seq(file_id, 0).unwrap().unwrap();
        assert!(
            stored.text.contains("Saltmere"),
            "stored text was truncated"
        );
        assert!(
            !stored.snippet.contains("Saltmere"),
            "snippet should still stop at 200 characters"
        );
        assert!(stored.snippet.len() <= 203);
    }

    /// Issue #75. The word-count split at a hardcoded 512 tears a block the
    /// model itself would accept whole. `precompute_chunks` must size-split
    /// against the embedder's own `token_count` and `max_context`, the same
    /// wall `index_file` reads.
    #[test]
    fn a_mid_size_block_is_one_chunk_through_the_write_pipeline() {
        use crate::llm::MockLlm;

        let mut embedder = MockLlm::new(8);
        // ~3200 chars => ~800 approx tokens: over 512, under the 2048 wall.
        let body = "alpha bravo charlie delta ".repeat(128);
        let content = format!("## Note\n{body}");
        let chunks = precompute_chunks(
            "notes/mid.md",
            &content,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
        )
        .unwrap();
        assert_eq!(
            chunks.len(),
            1,
            "a single block under the wall is one chunk"
        );
    }

    // ── The tag store (#60) ──────────────────────────────────────

    fn stored_tags(store: &Store, path: &str) -> Vec<String> {
        let file = store.get_file(path).unwrap().unwrap();
        store.file_tags(file.id).unwrap()
    }

    fn test_chunk_opts() -> ChunkOptions {
        ChunkOptions {
            min_chars: 0,
            promote_bold: false,
            carry_orphan_headings: false,
        }
    }

    #[test]
    fn a_created_note_writes_its_tag_rows() {
        use crate::llm::MockLlm;

        let (_tmp, store, root) = setup_vault();
        let mut embedder = MockLlm::new(256);
        let result = create_note(
            CreateNoteInput {
                content: "# Swamp\n\nA #type/undead lives here.\n".to_string(),
                filename: "swamp.md".to_string(),
                type_hint: None,
                tags: vec!["habitat/swamp".to_string()],
                folder: None,
                created_by: "test".to_string(),
                auto_link: Some(false),
            },
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &root,
            None,
        )
        .unwrap();

        assert_eq!(
            stored_tags(&store, &result.path),
            vec!["habitat/swamp", "type/undead"],
            "the property and the body are peers"
        );
    }

    /// Create one note and hand back what landed on disk.
    fn created_text(content: &str, tags: Vec<String>, filename: &str) -> String {
        use crate::llm::MockLlm;

        let (_tmp, store, root) = setup_vault();
        let mut embedder = MockLlm::new(256);
        let result = create_note(
            CreateNoteInput {
                content: content.to_string(),
                filename: filename.to_string(),
                type_hint: None,
                tags,
                folder: Some("lore".to_string()),
                created_by: "test".to_string(),
                auto_link: Some(false),
            },
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &root,
            None,
        )
        .unwrap();
        std::fs::read_to_string(root.join(&result.path)).unwrap()
    }

    #[test]
    fn create_writes_the_callers_frontmatter_as_it_was_given() {
        let content = "---\nname: Tidewatch Tower\naliases: []\ntags: [type/location]\n---\n\nPart of the coast.\n";
        assert_eq!(created_text(content, vec![], "Tidewatch Tower.md"), content);
    }

    #[test]
    fn create_adds_the_resolved_tags_to_the_notes_own_list() {
        let written = created_text(
            "---\nname: X\ntags: [type/lore]\n---\n\nBody.\n",
            vec!["realm/rudd".to_string()],
            "X.md",
        );
        assert_eq!(
            written,
            "---\nname: X\ntags: [type/lore, realm/rudd]\n---\n\nBody.\n"
        );
    }

    #[test]
    fn create_writes_no_block_when_it_has_nothing_to_put_in_one() {
        assert_eq!(
            created_text("Just a body.\n", vec![], "Bare.md"),
            "Just a body.\n"
        );
    }

    #[test]
    fn create_writes_no_created_stamp_and_no_placement_keys() {
        let written = created_text("---\nname: X\n---\n\nBody.\n", vec![], "X.md");
        for key in [
            "created:",
            "created_by:",
            "suggested_folder:",
            "confidence:",
            "reason:",
        ] {
            assert!(!written.contains(key), "{key} is in the note:\n{written}");
        }
    }

    #[test]
    fn create_refuses_a_colliding_name_and_points_at_update() {
        use crate::llm::MockLlm;

        let (_tmp, store, root) = setup_vault();
        let mut embedder = MockLlm::new(256);
        let input = || CreateNoteInput {
            content: "# Note\n\nBody.\n".to_string(),
            filename: "dup".to_string(),
            type_hint: None,
            tags: vec![],
            folder: Some("notes".to_string()),
            created_by: "test".to_string(),
            auto_link: Some(false),
        };
        create_note(
            input(),
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &root,
            None,
        )
        .expect("the first create writes the note");

        let err = create_note(
            input(),
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &root,
            None,
        )
        .expect_err("a second note at the same path must be refused");
        let msg = err.to_string();
        assert!(msg.contains("already exists"), "message was: {msg}");
        assert!(
            msg.contains("update"),
            "the error must point the caller at update: {msg}"
        );
    }

    /// `archive` overwrites the note's own value and `unarchive` then
    /// removes the key, so a note that already holds `archived_at` loses
    /// it in the round trip. Refusing is the only choice that neither
    /// loses the note's own value nor leaves a key behind (#92, I7).
    #[test]
    fn archiving_a_note_that_already_holds_archived_at_is_refused() {
        let (_tmp, store, root) = setup_vault();
        let content = "---\nname: X\narchived_at: 1999-01-01\n---\n\nBody\n";
        std::fs::write(root.join("n.md"), content).unwrap();
        store
            .insert_file("n.md", "hash", 100, "refuse1", None, None)
            .unwrap();

        let err = archive_note("n.md", &store, &root, None).unwrap_err();
        assert!(err.to_string().contains("archived_at"), "{err}");
        assert_eq!(
            std::fs::read_to_string(root.join("n.md")).unwrap(),
            content,
            "a refused archive must not touch the file"
        );
    }

    #[test]
    fn archiving_a_note_takes_the_tags_that_go_unused_with_it() {
        let (_tmp, store, root) = setup_vault();
        let content = "---\ntags:\n  - solo\n---\n\nBody.\n";
        let file_path = root.join("n.md");
        std::fs::write(&file_path, content).unwrap();
        let mtime = file_mtime(&file_path).unwrap();
        let id = store
            .insert_file("n.md", "hash", mtime, "arctag", None, None)
            .unwrap();
        store
            .reconcile_file_tags(id, &crate::tags::extract(content))
            .unwrap();

        archive_note("n.md", &store, &root, None).unwrap();

        let remaining: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM tags", [], |row| row.get(0))
            .unwrap();
        assert_eq!(remaining, 0, "the note was the tag's only carrier");
    }

    /// A note whose property holds one tag and whose body holds another.
    ///
    /// The junction holds both, because the property and the body are peers
    /// (#60). A write to the user's frontmatter must carry the property alone.
    fn note_with_a_body_hashtag(store: &Store, root: &std::path::Path) {
        let content = "---\ntags:\n  - work\n---\n\nBlocked on #todo today.\n";
        let file_path = root.join("n.md");
        std::fs::write(&file_path, content).unwrap();
        let mtime = file_mtime(&file_path).unwrap();
        let id = store
            .insert_file("n.md", "hash", mtime, "bodytg", None, None)
            .unwrap();
        store
            .reconcile_file_tags(id, &crate::tags::extract(content))
            .unwrap();
        assert_eq!(
            stored_tags(store, "n.md"),
            vec!["todo", "work"],
            "the junction holds the property tag and the body tag"
        );
    }

    #[test]
    fn archiving_keeps_a_body_hashtag_out_of_the_property() {
        let (_tmp, store, root) = setup_vault();
        note_with_a_body_hashtag(&store, &root);

        archive_note("n.md", &store, &root, None).unwrap();

        let written = std::fs::read_to_string(root.join("04-Archive/n.md")).unwrap();
        let (fm, _) = split_frontmatter(&written);
        let (_, property_tags, _) = parse_frontmatter_fields(&fm);
        // archive no longer writes an `archived` tag (#92): `archived: true`
        // and the note's place under the archive folder already say it.
        assert_eq!(property_tags, vec!["work"]);
        assert!(written.contains("#todo"), "the body tag stays in the body");
    }

    /// `archive` and `undo: true` are one capability and its reverse (#62):
    /// this is the durable coverage for that round trip, now that the MCP
    /// and HTTP handlers wire `undo` straight through instead of guarding it.
    #[test]
    fn archiving_and_undoing_it_return_the_note_to_its_folder() {
        use crate::llm::MockLlm;

        let (_tmp, store, vault) = setup_vault();
        std::fs::create_dir_all(vault.join("Projects")).unwrap();
        std::fs::write(vault.join("Projects/n.md"), "# N\n\nbody\n").unwrap();
        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            &vault,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        archive_note("Projects/n.md", &store, &vault, None).unwrap();
        assert!(!vault.join("Projects/n.md").exists());
        let archived = vault.join("04-Archive/Projects/n.md");
        assert!(archived.exists());

        unarchive_note(
            "04-Archive/Projects/n.md",
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &vault,
        )
        .unwrap();
        assert!(vault.join("Projects/n.md").exists());
        assert!(!archived.exists());
        assert_eq!(
            std::fs::read_to_string(vault.join("Projects/n.md")).unwrap(),
            "# N\n\nbody\n"
        );
    }

    /// A vault holding one note at `rel`, indexed and ready to archive.
    fn vault_with(
        rel: &str,
        text: &str,
    ) -> (
        tempfile::TempDir,
        Store,
        std::path::PathBuf,
        crate::llm::MockLlm,
    ) {
        use crate::llm::MockLlm;

        let (tmp, store, vault) = setup_vault();
        if let Some(parent) = std::path::Path::new(rel).parent() {
            std::fs::create_dir_all(vault.join(parent)).unwrap();
        }
        std::fs::write(vault.join(rel), text).unwrap();
        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            &vault,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, vault, embedder)
    }

    #[test]
    fn archiving_keeps_every_key_the_note_already_carried() {
        let note = "---\nname: Probe\naliases: []\ntags: [type/lore]\n---\n\nBody.\n";
        let (_tmp, store, vault, _embedder) = vault_with("lore/Probe.md", note);

        archive_note("lore/Probe.md", &store, &vault, None).unwrap();

        let archived = std::fs::read_to_string(vault.join("04-Archive/lore/Probe.md")).unwrap();
        assert!(
            archived.starts_with("---\nname: Probe\naliases: []\ntags: [type/lore]\n"),
            "{archived}"
        );
        assert!(archived.contains("archived: true"), "{archived}");
        assert!(
            archived.contains("archived_from: lore/Probe.md"),
            "{archived}"
        );
        assert!(archived.ends_with("---\n\nBody.\n"), "{archived}");
    }

    #[test]
    fn an_archive_round_trip_returns_the_file_byte_for_byte() {
        let note = "---\nname: Probe\naliases: []\ntags: [type/lore]\n---\n\nBody.\n";
        let (_tmp, store, vault, mut embedder) = vault_with("lore/Probe.md", note);

        archive_note("lore/Probe.md", &store, &vault, None).unwrap();
        unarchive_note(
            "04-Archive/lore/Probe.md",
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &vault,
        )
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(vault.join("lore/Probe.md")).unwrap(),
            note
        );
    }

    /// A note with no frontmatter block at all takes `Block::parse_or_open`
    /// down the `open` path on archive and back through `parse` on
    /// unarchive — the two paths must agree on where the block ends and the
    /// body begins, or the round trip gains or loses the blank line between
    /// them (#92).
    #[test]
    fn an_archive_round_trip_on_a_note_with_no_frontmatter_returns_the_file_byte_for_byte() {
        let note = "# N\n\nbody\n";
        let (_tmp, store, vault, mut embedder) = vault_with("n.md", note);

        archive_note("n.md", &store, &vault, None).unwrap();
        unarchive_note(
            "04-Archive/n.md",
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &vault,
        )
        .unwrap();

        assert_eq!(std::fs::read_to_string(vault.join("n.md")).unwrap(), note);
    }

    /// `Block::is_empty` counts keys only, so a block holding just a
    /// comment reported empty the same way a note with no block at all
    /// does. `unarchive_note` used to fall back to `block.body()` for both,
    /// discarding the fences and the comment along with them (#92, I2).
    #[test]
    fn an_archive_round_trip_on_a_note_with_a_comment_only_block_returns_the_file_byte_for_byte() {
        let note = "---\n# why this note exists\n---\n\nBody\n";
        let (_tmp, store, vault, mut embedder) = vault_with("n.md", note);

        archive_note("n.md", &store, &vault, None).unwrap();
        unarchive_note(
            "04-Archive/n.md",
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &vault,
        )
        .unwrap();

        assert_eq!(std::fs::read_to_string(vault.join("n.md")).unwrap(), note);
    }

    /// A block that is present but holds no key at all — `---\n---\n` — and
    /// a note with no block are indistinguishable once archived: both
    /// leave the archived block holding exactly the three archive keys and
    /// nothing else, so `unarchive_note`'s only honest choice between them
    /// is the one that does not regress the no-block round trip above.
    /// This does not restore the original `---\n---\n` fences — a known
    /// limit of a fix confined to `unarchive_note` (#92, I2 — see the final
    /// fix report for why `archive_note` would have to change too).
    #[test]
    fn a_truly_empty_block_and_no_block_restore_the_same_way() {
        let note = "---\n---\n\nBody\n";
        let (_tmp, store, vault, mut embedder) = vault_with("n.md", note);

        archive_note("n.md", &store, &vault, None).unwrap();
        unarchive_note(
            "04-Archive/n.md",
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &vault,
        )
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(vault.join("n.md")).unwrap(),
            "Body\n",
            "an empty block cannot be told apart from no block at unarchive time"
        );
    }

    /// A note archived by a version of knapper before #92 wrote `archived`
    /// into its own `tags` list, on top of the `archived: true` key this
    /// build reads (see `git show 9bae927:src/writer.rs`, the
    /// `.filter(|t| t != "archived")` this restores). Unarchiving it must
    /// not let the leftover tag back into the vocabulary (#92, I6).
    #[test]
    fn unarchiving_a_legacy_note_strips_the_archived_tag_it_carried() {
        use crate::llm::MockLlm;

        let (_tmp, store, vault) = setup_vault();
        std::fs::create_dir_all(vault.join("04-Archive")).unwrap();
        std::fs::write(
            vault.join("04-Archive/n.md"),
            "---\ntags: [work, archived]\narchived: true\narchived_at: 2020-01-01\narchived_from: n.md\n---\n\nBody.\n",
        )
        .unwrap();
        let mut embedder = MockLlm::new(256);

        let result = unarchive_note(
            "04-Archive/n.md",
            &store,
            &mut embedder,
            EmbedComposition::default(),
            test_chunk_opts(),
            &vault,
        )
        .unwrap();

        assert_eq!(
            result.tags,
            vec!["work"],
            "the leftover archived tag must not report back"
        );
        let written = std::fs::read_to_string(vault.join("n.md")).unwrap();
        assert_eq!(
            written, "---\ntags: [work]\n---\n\nBody.\n",
            "got {written}"
        );
        assert_eq!(stored_tags(&store, "n.md"), vec!["work"]);
    }

    /// A vault of one note, indexed, ready for `update_note`.
    fn indexed_note(body: &str) -> (tempfile::TempDir, Store, std::path::PathBuf) {
        use crate::llm::MockLlm;

        let (tmp, store, vault) = setup_vault();
        std::fs::write(vault.join("note.md"), body).unwrap();
        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            &vault,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, vault)
    }

    fn one_edit(target: EditTarget, mode: EditMode, content: Option<&str>) -> UpdateInput {
        UpdateInput {
            file: "note.md".into(),
            edits: vec![NoteEdit {
                target,
                mode,
                content: content.map(|c| EditContent::Text(c.to_string())),
            }],
        }
    }

    /// Ported from the three `edit_note` section tests that went with it (#62).
    /// `update` is the one call that edits a section now, so the three modes
    /// have to reach it through `update_note` and not only through
    /// `apply_section_edit`.
    ///
    /// The second element is what the mode does to the body it found: `None`
    /// for the mode that replaces it, and otherwise the side the new text
    /// takes — `Greater` for the mode that writes after it. Both facts are
    /// load-bearing: without them `append` and `prepend` can swap arms, or
    /// either can discard the old body, with every test still green (#62).
    #[test]
    fn a_section_edit_reaches_the_section_it_names_in_every_mode() {
        use std::cmp::Ordering;

        for (mode, old_body) in [
            (EditMode::Replace, None),
            (EditMode::Append, Some(Ordering::Greater)),
            (EditMode::Prepend, Some(Ordering::Less)),
        ] {
            let (_tmp, store, vault) = indexed_note(
                "# Person\n\n## Interactions\n\nOld entry\n\n## Links\n\nSome links\n",
            );
            update_note(
                &store,
                &vault,
                &one_edit(
                    EditTarget::Section("Interactions".into()),
                    mode,
                    Some("New entry"),
                ),
            )
            .unwrap();

            let out = std::fs::read_to_string(vault.join("note.md")).unwrap();
            let new_at = out
                .find("New entry")
                .unwrap_or_else(|| panic!("{mode:?} lost its content: {out}"));
            match old_body {
                None => assert!(
                    !out.contains("Old entry"),
                    "{mode:?} kept the old body: {out}"
                ),
                Some(side) => {
                    let old_at = out
                        .find("Old entry")
                        .unwrap_or_else(|| panic!("{mode:?} discarded the old body: {out}"));
                    assert_eq!(
                        new_at.cmp(&old_at),
                        side,
                        "{mode:?} wrote the new text on the wrong side of the old body: {out}"
                    );
                }
            }
            assert!(
                out.contains("## Links") && out.contains("Some links"),
                "{mode:?} disturbed the next section: {out}"
            );
            // Whatever the mode, the edit lands inside the section it named.
            assert!(
                new_at < out.find("## Links").unwrap(),
                "{mode:?} wrote outside the section: {out}"
            );
        }
    }

    /// Ported from `test_edit_note_file_not_found` (#62).
    #[test]
    fn an_update_of_a_note_the_store_does_not_hold_is_an_error() {
        let (_tmp, store, vault) = setup_vault();
        let err = update_note(
            &store,
            &vault,
            &UpdateInput {
                file: "nonexistent.md".into(),
                edits: vec![NoteEdit {
                    target: EditTarget::Body,
                    mode: EditMode::Append,
                    content: Some(EditContent::Text("x".into())),
                }],
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("file not found"), "got {err}");
    }

    /// Ported from `tests/write_pipeline.rs::test_conflict_detection`, which
    /// was `#[ignore]` behind a model download and so never ran. The guard is
    /// wider since #62: one check covers every call `update` absorbed, where
    /// `edit`, `rewrite` and `edit_frontmatter` each made none.
    ///
    /// The disk mtime is stamped rather than slept for. `file_mtime` reads
    /// whole seconds, so an outside write in the same second as the index
    /// passes the check and the test would assert nothing.
    #[test]
    fn an_update_of_a_note_edited_outside_knapper_is_a_conflict() {
        let (_tmp, store, vault) = indexed_note("# Note\n\nIndexed body\n");
        let indexed_mtime = store.get_file("note.md").unwrap().unwrap().mtime;

        let outside = "# Note\n\nWritten outside knapper\n";
        std::fs::write(vault.join("note.md"), outside).unwrap();
        std::fs::File::options()
            .write(true)
            .open(vault.join("note.md"))
            .unwrap()
            .set_modified(
                std::time::SystemTime::UNIX_EPOCH
                    + std::time::Duration::from_secs(indexed_mtime as u64 + 2),
            )
            .unwrap();

        let err = update_note(
            &store,
            &vault,
            &one_edit(EditTarget::Body, EditMode::Append, Some("appended")),
        )
        .unwrap_err();
        assert!(err.to_string().contains("mtime conflict"), "got {err}");
        assert_eq!(
            std::fs::read_to_string(vault.join("note.md")).unwrap(),
            outside,
            "the update wrote over an edit made outside knapper"
        );
    }

    /// Ported from `test_rewrite_preserves_frontmatter` (#62). A body edit is
    /// the design's `rewrite` row, and it always keeps the note's frontmatter.
    #[test]
    fn a_body_replace_keeps_the_notes_frontmatter() {
        let (_tmp, store, vault) = indexed_note(
            "---\ntags:\n  - project\nstatus: active\n---\n\n# Old Content\n\nOld body\n",
        );
        update_note(
            &store,
            &vault,
            &one_edit(
                EditTarget::Body,
                EditMode::Replace,
                Some("# New Content\n\nNew body\n"),
            ),
        )
        .unwrap();

        let out = std::fs::read_to_string(vault.join("note.md")).unwrap();
        assert!(out.contains("status: active"), "got {out}");
        assert!(out.contains("# New Content"), "got {out}");
        assert!(!out.contains("Old body"), "got {out}");
    }

    /// Ported from `test_edit_frontmatter_no_existing_frontmatter` (#62): the
    /// one frontmatter case the pure-transform tests do not cover is a note
    /// that has none at all.
    #[test]
    fn a_property_edit_on_a_note_with_no_frontmatter_writes_a_block() {
        let (_tmp, store, vault) = indexed_note("# Content\n\nJust body, no frontmatter.\n");
        update_note(
            &store,
            &vault,
            &UpdateInput {
                file: "note.md".into(),
                edits: vec![
                    NoteEdit {
                        target: EditTarget::Property("status".into()),
                        mode: EditMode::Replace,
                        content: Some(EditContent::Text("active".into())),
                    },
                    NoteEdit {
                        target: EditTarget::Property("tags".into()),
                        mode: EditMode::Append,
                        content: Some(EditContent::Text("new-tag".into())),
                    },
                ],
            },
        )
        .unwrap();

        let out = std::fs::read_to_string(vault.join("note.md")).unwrap();
        assert!(out.starts_with("---\n"), "got {out}");
        assert!(out.contains("status: active"), "got {out}");
        assert!(out.contains("new-tag"), "got {out}");
        assert!(out.contains("# Content"), "got {out}");
    }

    /// Ported from `editing_frontmatter_moves_the_tag_rows_with_it` (#62).
    /// `update` is the one frontmatter write path now, and it reconciles the
    /// junction the same way (#60).
    #[test]
    fn a_property_edit_moves_the_tag_rows_with_it() {
        let (_tmp, store, vault) = indexed_note("---\ntags:\n  - habitat/swamp\n---\n\nBody.\n");
        update_note(
            &store,
            &vault,
            &one_edit(
                EditTarget::Property("tags".into()),
                EditMode::Append,
                Some("type/undead"),
            ),
        )
        .unwrap();

        assert_eq!(
            stored_tags(&store, "note.md"),
            vec!["habitat/swamp", "type/undead"]
        );
    }

    /// Ported from `update_metadata_keeps_a_body_hashtag_out_of_the_property`
    /// (#62). The property and the body are peers (#60), so a write to the
    /// user's frontmatter must carry the property alone.
    #[test]
    fn a_property_edit_keeps_a_body_hashtag_out_of_the_property() {
        let (_tmp, store, vault) =
            indexed_note("---\ntags:\n  - work\n---\n\nBlocked on #todo today.\n");
        assert_eq!(
            stored_tags(&store, "note.md"),
            vec!["todo", "work"],
            "the junction holds the property tag and the body tag"
        );

        update_note(
            &store,
            &vault,
            &one_edit(
                EditTarget::Property("status".into()),
                EditMode::Replace,
                Some("done"),
            ),
        )
        .unwrap();

        let written = std::fs::read_to_string(vault.join("note.md")).unwrap();
        let (fm, _) = split_frontmatter(&written);
        let (_, property_tags, _) = parse_frontmatter_fields(&fm);
        assert_eq!(property_tags, vec!["work"]);
        assert!(written.contains("#todo"), "the body tag stays in the body");
    }

    /// Ported from `appended_text_past_the_snippet_boundary_is_searchable`
    /// (#62). The append reaches the index through `update_note` and then
    /// `reindex_written_file`, which is the pair every surface calls.
    #[test]
    fn appended_text_past_the_snippet_boundary_is_searchable() {
        use crate::llm::MockLlm;

        let (_tmp, store, vault) = indexed_note("# Coast\n\n## The Coast Road\n\nOriginal.\n");
        let mut embedder = MockLlm::new(256);

        let filler = "The coast road runs north through salt marsh and low dune. ".repeat(8);
        update_note(
            &store,
            &vault,
            &one_edit(
                EditTarget::Body,
                EditMode::Append,
                Some(&format!("\n{filler}\n\nIt ends at Saltmere.\n")),
            ),
        )
        .unwrap();
        crate::indexer::reindex_written_file(
            "note.md",
            &store,
            &mut embedder,
            &vault,
            crate::indexer::IndexSettings {
                chunk: test_chunk_opts(),
                embed: EmbedComposition::default(),
            },
        )
        .unwrap();

        let file = store.get_file("note.md").unwrap().unwrap();
        assert!(
            store
                .best_matching_chunk_seq(file.id, &["Saltmere".to_string()])
                .unwrap()
                .is_some(),
            "a term past character 200 must still be searchable"
        );
    }

    #[test]
    fn one_call_applies_a_section_edit_and_a_property_edit_in_one_write() {
        use crate::llm::MockLlm;

        let (_tmp, store, vault) = setup_vault();
        std::fs::write(
            vault.join("note.md"),
            "---\ntags:\n  - a\n---\n\n## Spells\n\nold\n",
        )
        .unwrap();
        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            &vault,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let before = std::fs::metadata(vault.join("note.md")).unwrap().len();

        update_note(
            &store,
            &vault,
            &UpdateInput {
                file: "note.md".into(),
                edits: vec![
                    NoteEdit {
                        target: EditTarget::Section("Spells".into()),
                        mode: EditMode::Replace,
                        content: Some(EditContent::Text("new".into())),
                    },
                    NoteEdit {
                        target: EditTarget::Property("tags".into()),
                        mode: EditMode::Append,
                        content: Some(EditContent::Text("b".into())),
                    },
                ],
            },
        )
        .unwrap();

        let out = std::fs::read_to_string(vault.join("note.md")).unwrap();
        assert!(out.contains("new"));
        assert!(!out.contains("old"));
        assert!(out.contains("- a") && out.contains("- b"));
        assert_ne!(before, out.len() as u64);
    }

    #[test]
    fn every_frontmatter_operation_has_an_edit_that_does_it() {
        let cases: Vec<(NoteEdit, &str, bool)> = vec![
            // (edit, substring, expected present after)
            (
                NoteEdit {
                    target: EditTarget::Property("status".into()),
                    mode: EditMode::Replace,
                    content: Some(EditContent::Text("done".into())),
                },
                "status: done",
                true,
            ),
            (
                NoteEdit {
                    target: EditTarget::Property("tags".into()),
                    mode: EditMode::Replace,
                    content: Some(EditContent::List(vec!["x".into(), "y".into()])),
                },
                "- x",
                true,
            ),
            (
                NoteEdit {
                    target: EditTarget::Property("keep".into()),
                    mode: EditMode::Remove,
                    content: None,
                },
                "keep:",
                false,
            ),
            (
                NoteEdit {
                    target: EditTarget::Property("tags".into()),
                    mode: EditMode::Remove,
                    content: Some(EditContent::Text("a".into())),
                },
                "- a",
                false,
            ),
            // An append to a property that is not `tags` goes to that
            // property. Routing appends by name, with `tags` as the
            // fallback, writes this value into the note's tag list instead
            // (#62).
            (
                NoteEdit {
                    target: EditTarget::Property("status".into()),
                    mode: EditMode::Append,
                    content: Some(EditContent::Text("wip".into())),
                },
                "status:\n  - wip",
                true,
            ),
            (
                NoteEdit {
                    target: EditTarget::Property("status".into()),
                    mode: EditMode::Append,
                    content: Some(EditContent::Text("wip".into())),
                },
                "- a\n- wip",
                false,
            ),
        ];

        for (edit, needle, present) in cases {
            let doc = "---\ntags:\n  - a\nkeep: yes\n---\n\nbody\n";
            let out = apply_note_edits(doc, std::slice::from_ref(&edit)).unwrap();
            assert_eq!(out.contains(needle), present, "edit {edit:?} on {doc:?}");
        }
    }

    #[test]
    fn a_list_of_property_edits_reassembles_the_note_once() {
        // A run of property edits becomes one pass over one `Block` (#62), so
        // a two-edit call must give the same bytes as a one-edit call gives,
        // plus the tag, with the block's own two-space list style kept. A
        // `contains` assertion cannot see this, so pin the bytes.
        let doc = "---\ntags:\n  - a\n---\n\n## Spells\n\nold\n";

        let one = apply_note_edits(
            doc,
            &[NoteEdit {
                target: EditTarget::Property("tags".into()),
                mode: EditMode::Append,
                content: Some(EditContent::Text("b".into())),
            }],
        )
        .unwrap();
        assert_eq!(one, "---\ntags:\n  - a\n  - b\n---\n\n## Spells\n\nold\n");

        let two = apply_note_edits(
            doc,
            &[
                NoteEdit {
                    target: EditTarget::Property("tags".into()),
                    mode: EditMode::Append,
                    content: Some(EditContent::Text("b".into())),
                },
                NoteEdit {
                    target: EditTarget::Property("status".into()),
                    mode: EditMode::Replace,
                    content: Some(EditContent::Text("done".into())),
                },
            ],
        )
        .unwrap();
        assert_eq!(
            two,
            "---\ntags:\n  - a\n  - b\nstatus: done\n---\n\n## Spells\n\nold\n"
        );
    }

    /// The Task 12 fix trimmed the leading break `split_frontmatter` leaves
    /// on a body edit; the frontmatter block reassembles the same way and
    /// carried the same defect (#62). `update` is the only frontmatter write
    /// path, so it accumulates one blank line per call across successive
    /// `update`s on the same note — pin the bytes across two of them.
    #[test]
    fn two_successive_property_updates_add_no_blank_line() {
        use crate::llm::MockLlm;

        let (_tmp, store, vault) = setup_vault();
        std::fs::write(
            vault.join("note.md"),
            "---\nstatus: draft\n---\n\n# Content\nbody text\n",
        )
        .unwrap();
        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            &vault,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let input = UpdateInput {
            file: "note.md".into(),
            edits: vec![NoteEdit {
                target: EditTarget::Property("status".into()),
                mode: EditMode::Replace,
                content: Some(EditContent::Text("active".into())),
            }],
        };
        update_note(&store, &vault, &input).unwrap();
        update_note(&store, &vault, &input).unwrap();

        let updated = std::fs::read_to_string(vault.join("note.md")).unwrap();
        assert_eq!(
            updated,
            "---\nstatus: active\n---\n\n# Content\nbody text\n"
        );
    }
}
