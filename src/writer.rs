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
    pub filename: Option<String>,
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

/// One change to a note's frontmatter. `AddTo` and `RemoveFrom` name the
/// property they change, so an edit to `status` cannot reach the `tags` list.
/// The four variants that wrote the key into their own name — `AddTag`,
/// `RemoveTag`, `AddAlias`, `RemoveAlias` — were exact aliases of these two
/// with `tags` or `aliases` inlined, and a routing table that falls back to
/// one of them writes to the wrong property whenever the key is a third one
/// (#62).
#[derive(Debug, Clone)]
pub enum FrontmatterOp {
    Set(String, String),
    Remove(String),
    AddTo(String, String),
    RemoveFrom(String, String),
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
}

// ── Helper functions ────────────────────────────────────────────

/// Strip characters that are invalid in filenames, keeping alphanumeric, spaces, dashes, underscores, and dots.
pub fn generate_filename(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_' || *c == '.')
        .collect()
}

/// Extract a title from content: first `# heading` or first non-empty line, truncated to 50 chars.
pub fn extract_title(content: &str) -> String {
    // If content has frontmatter, check for a title field and skip FM for heading search
    let (fm, body) = split_frontmatter(content);
    if !fm.is_empty() {
        // Check for title: field in frontmatter
        let (scalars, _, _) = parse_frontmatter_fields(&fm);
        if let Some(title) = scalars.get("title") {
            let title = title.trim();
            if !title.is_empty() {
                if title.len() > 50 {
                    return title[..50].to_string();
                }
                return title.to_string();
            }
        }
    }

    // Search body (or full content if no FM) for heading or first non-empty line
    let search_content = if fm.is_empty() {
        content
    } else {
        body.as_str()
    };
    for line in search_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("# ") {
            let heading = heading.trim();
            if heading.len() > 50 {
                return heading[..50].to_string();
            }
            return heading.to_string();
        }
        // First non-empty line
        if trimmed.len() > 50 {
            return trimmed[..50].to_string();
        }
        return trimmed.to_string();
    }
    "Untitled".to_string()
}

/// Optional placement suggestion metadata for inbox notes.
pub struct PlacementSuggestion {
    pub suggested_folder: String,
    pub confidence: f64,
    pub reason: String,
}

/// Build YAML frontmatter string.
pub fn build_frontmatter(
    tags: &[String],
    created_by: Option<&str>,
    aliases: Option<&[String]>,
    suggestion: Option<&PlacementSuggestion>,
) -> String {
    let mut fm = String::from("---\n");

    if !tags.is_empty() {
        fm.push_str("tags:\n");
        for tag in tags {
            fm.push_str(&format!("  - {}\n", tag));
        }
    }

    if let Some(aliases) = aliases
        && !aliases.is_empty()
    {
        fm.push_str("aliases:\n");
        for alias in aliases {
            fm.push_str(&format!("  - {}\n", alias));
        }
    }

    fm.push_str(&format!("created: {}\n", today_date()));

    if let Some(by) = created_by {
        fm.push_str(&format!("created_by: {}\n", by));
    }

    // Placement suggestion for inbox notes — user sees why it landed here
    if let Some(s) = suggestion {
        fm.push_str(&format!("suggested_folder: {}\n", s.suggested_folder));
        fm.push_str(&format!("confidence: {:.2}\n", s.confidence));
        fm.push_str(&format!("reason: \"{}\"\n", s.reason));
    }

    fm.push_str("---\n\n");
    fm
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

/// Build a merged frontmatter block from auto-generated fields + user-provided fields.
///
/// - `tags` and `aliases` are merged (deduplicated), user values included
/// - `created` and `created_by` always use auto-generated values
/// - All other user fields are passed through
fn build_merged_frontmatter(
    auto_tags: &[String],
    created_by: Option<&str>,
    suggestion: Option<&PlacementSuggestion>,
    user_scalars: &BTreeMap<String, String>,
    user_tags: &[String],
    user_aliases: &[String],
) -> String {
    // Merge tags: auto first, then user, deduplicated
    let mut merged_tags: Vec<String> = auto_tags.to_vec();
    for t in user_tags {
        if !merged_tags.iter().any(|existing| existing == t) {
            merged_tags.push(t.clone());
        }
    }

    // Merge aliases: just user aliases (auto has none by default from create_note)
    let merged_aliases: Vec<String> = user_aliases.to_vec();

    let mut fm = String::from("---\n");

    if !merged_tags.is_empty() {
        fm.push_str("tags:\n");
        for tag in &merged_tags {
            fm.push_str(&format!("  - {}\n", tag));
        }
    }

    if !merged_aliases.is_empty() {
        fm.push_str("aliases:\n");
        for alias in &merged_aliases {
            fm.push_str(&format!("  - {}\n", alias));
        }
    }

    // Always auto-generated
    fm.push_str(&format!("created: {}\n", today_date()));

    if let Some(by) = created_by {
        fm.push_str(&format!("created_by: {}\n", by));
    }

    // User scalar fields (skip created/created_by — always auto-generated)
    for (key, val) in user_scalars {
        match key.as_str() {
            "created" | "created_by" => continue,
            _ => fm.push_str(&format!("{}: {}\n", key, val)),
        }
    }

    // Placement suggestion for inbox notes
    if let Some(s) = suggestion {
        fm.push_str(&format!("suggested_folder: {}\n", s.suggested_folder));
        fm.push_str(&format!("confidence: {:.2}\n", s.confidence));
        fm.push_str(&format!("reason: \"{}\"\n", s.reason));
    }

    fm.push_str("---\n\n");
    fm
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
    let chunks = split_oversized_chunks(parsed.chunks, &|s| s.split_whitespace().count(), 512, 50);

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
    // Step 1: Determine filename
    let title = if let Some(ref name) = input.filename {
        name.clone()
    } else {
        extract_title(&input.content)
    };
    let filename = generate_filename(&title);
    let filename = if filename.ends_with(".md") {
        filename
    } else {
        format!("{}.md", filename)
    };

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

    // Step 5: Build frontmatter and assemble content
    // Split user frontmatter from body so we can merge instead of duplicate
    let (user_fm, body) = split_frontmatter(&content_with_links);
    let (user_scalars, user_tags, user_aliases) = if !user_fm.is_empty() {
        parse_frontmatter_fields(&user_fm)
    } else {
        (BTreeMap::new(), Vec::new(), Vec::new())
    };

    // If placement fell back to inbox with a suggestion, inject suggested_folder metadata
    let suggestion = if placement_result.strategy == placement::PlacementStrategy::InboxFallback {
        placement_result
            .suggestion
            .as_ref()
            .map(|(folder, conf)| PlacementSuggestion {
                suggested_folder: folder.clone(),
                confidence: *conf,
                reason: format!("semantic similarity: {conf:.3}"),
            })
    } else {
        None
    };
    let frontmatter = build_merged_frontmatter(
        &resolved_tags,
        Some(&input.created_by),
        suggestion.as_ref(),
        &user_scalars,
        &user_tags,
        &user_aliases,
    );
    let full_content = format!("{}{}", frontmatter, body);

    let rel_path = format!("{}/{}", placement_result.folder, filename);
    let final_path = vault_path.join(&rel_path);

    // Check for existing file before doing expensive work
    if final_path.exists() {
        bail!(
            "file already exists at {}, refusing to overwrite",
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

/// Apply a list of frontmatter operations to a note's text. The transform is
/// separate from the I/O so that `update_note` can apply a list of them to
/// one string and write once (#62).
///
/// Uses `crate::markdown::split_frontmatter()` to extract raw YAML, then applies
/// operations sequentially using `serde_yaml`.
pub fn apply_frontmatter_ops(content: &str, ops: &[FrontmatterOp]) -> Result<String> {
    // Split frontmatter using crate::markdown::split_frontmatter (returns raw YAML without delimiters)
    let (maybe_fm, body) = crate::markdown::split_frontmatter(content);

    // Parse YAML into a Mapping (create empty mapping if no frontmatter)
    let mut mapping: serde_yaml::Mapping = if let Some(ref fm) = maybe_fm {
        let val: serde_yaml::Value = serde_yaml::from_str(fm)
            .unwrap_or(serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
        match val {
            serde_yaml::Value::Mapping(m) => m,
            _ => serde_yaml::Mapping::new(),
        }
    } else {
        serde_yaml::Mapping::new()
    };

    // Apply operations sequentially
    for op in ops {
        match op {
            FrontmatterOp::Set(key, value) => {
                mapping.insert(
                    serde_yaml::Value::String(key.clone()),
                    serde_yaml::Value::String(value.clone()),
                );
            }
            FrontmatterOp::Remove(key) => {
                mapping.remove(serde_yaml::Value::String(key.clone()));
            }
            FrontmatterOp::AddTo(key, value) => {
                apply_add_to_sequence(&mut mapping, key, value);
            }
            FrontmatterOp::RemoveFrom(key, value) => {
                apply_remove_from_sequence(&mut mapping, key, value);
            }
        }
    }

    // Serialize back to YAML
    let yaml_str = serde_yaml::to_string(&serde_yaml::Value::Mapping(mapping))?;

    // Reassemble: ---\n{yaml}---\n\n{body}
    // serde_yaml::to_string adds a trailing newline, so we don't need an extra one before ---.
    // `split_frontmatter`'s found-delimiter branch rejoins the body with `lines().join("\n")`,
    // which drops the body's own final line break — restore it here so the file keeps ending
    // in one, matching every other write path in this module. The same rejoin also leaves the
    // body carrying the break after the closing `---` (the defect `apply_body_edit` fixes the
    // same way for its own reassembly, #62); the line below writes its own `\n\n` separator, so
    // that leading break has to go, or it pushes the body one line further down on every call.
    let body = body.trim_start_matches('\n');
    let body = if body.is_empty() || body.ends_with('\n') {
        body.to_string()
    } else {
        format!("{}\n", body)
    };
    Ok(format!("---\n{}---\n\n{}", yaml_str, body))
}

/// Helper: add a value to a YAML sequence field (create if missing, skip duplicates).
fn apply_add_to_sequence(mapping: &mut serde_yaml::Mapping, key: &str, value: &str) {
    let key_val = serde_yaml::Value::String(key.to_string());
    let new_item = serde_yaml::Value::String(value.to_string());

    let seq = mapping
        .entry(key_val)
        .or_insert_with(|| serde_yaml::Value::Sequence(vec![]));

    if let serde_yaml::Value::Sequence(items) = seq
        && !items.contains(&new_item)
    {
        items.push(new_item);
    }
}

/// Helper: remove a value from a YAML sequence field.
fn apply_remove_from_sequence(mapping: &mut serde_yaml::Mapping, key: &str, value: &str) {
    let key_val = serde_yaml::Value::String(key.to_string());
    let remove_item = serde_yaml::Value::String(value.to_string());

    if let Some(serde_yaml::Value::Sequence(items)) = mapping.get_mut(&key_val) {
        items.retain(|item| item != &remove_item);
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

/// The frontmatter operations one property edit means. Every key routes
/// through `AddTo` and `RemoveFrom`, which carry the key, so an append to
/// `status` cannot reach the `tags` list (#62).
fn property_ops(
    key: &str,
    mode: EditMode,
    content: Option<&EditContent>,
) -> Result<Vec<FrontmatterOp>> {
    Ok(match (mode, content) {
        (EditMode::Replace, Some(EditContent::Text(v))) => {
            vec![FrontmatterOp::Set(key.to_string(), v.clone())]
        }
        // `Set` carries a scalar, so a whole sequence is a remove and then
        // one add per item.
        (EditMode::Replace, Some(EditContent::List(vs))) => {
            let mut ops = vec![FrontmatterOp::Remove(key.to_string())];
            ops.extend(
                vs.iter()
                    .map(|v| FrontmatterOp::AddTo(key.to_string(), v.clone())),
            );
            ops
        }
        (EditMode::Append, Some(EditContent::Text(v))) => {
            vec![FrontmatterOp::AddTo(key.to_string(), v.clone())]
        }
        (EditMode::Remove, None) => vec![FrontmatterOp::Remove(key.to_string())],
        (EditMode::Remove, Some(EditContent::Text(v))) => {
            vec![FrontmatterOp::RemoveFrom(key.to_string(), v.clone())]
        }
        (mode, content) => anyhow::bail!(
            "a {} on property '{key}' with {} content has no meaning",
            edit_mode_name(&mode),
            if content.is_some() { "this" } else { "no" }
        ),
    })
}

/// Apply every edit to a note's text, in order. Pure, so `update_note` can
/// write the result once — one file write, one conflict check and one
/// re-index for a whole batch (#62).
///
/// A run of property edits becomes one `apply_frontmatter_ops` call. The ops
/// already apply in order over one YAML mapping, so the result is the same,
/// but each call reassembles the note and each reassembly adds a line break
/// between the frontmatter and the body. One call per edit therefore pushes
/// the body one line down for every property edit in the list.
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
                let mut ops = Vec::new();
                for property in properties {
                    let EditTarget::Property(key) = &property.target else {
                        bail!("a run of property edits holds property edits alone");
                    };
                    ops.extend(property_ops(key, property.mode, property.content.as_ref())?);
                }
                text = apply_frontmatter_ops(&text, &ops)?;
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
            "mtime conflict: file {} was modified outside engraph (disk={}, indexed={})",
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

    // Read content and inject archive frontmatter
    let content = std::fs::read_to_string(&old_path)?;
    let (old_fm, body) = split_frontmatter(&content);

    // Keep the note's own `tags:` property and add the archive tag. The source
    // is the property, not `FileRecord.tags`: the junction also holds the body
    // hashtags (#60), which stay in the body.
    let (_, mut tags, _) = parse_frontmatter_fields(&old_fm);
    if !tags.contains(&"archived".to_string()) {
        tags.push("archived".to_string());
    }

    let archive_fm = format!(
        "---\n\
         archived: true\n\
         archived_at: {}\n\
         archived_from: {}\n\
         tags:\n{}\
         ---\n\n",
        today_date(),
        file_record.path,
        tags.iter()
            .map(|t| format!("  - {}\n", t))
            .collect::<String>(),
    );
    let new_content = format!("{}{}", archive_fm, body);

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
    let (fm_str, body) = split_frontmatter(&content);

    // Extract archived_from from frontmatter
    let original_path = fm_str
        .lines()
        .find(|l| l.starts_with("archived_from:"))
        .and_then(|l| l.strip_prefix("archived_from:"))
        .map(|s| s.trim().to_string())
        .ok_or_else(|| {
            anyhow::anyhow!("no archived_from in frontmatter — cannot determine original location")
        })?;

    let restore_full_path = vault_path.join(&original_path);

    if restore_full_path.exists() {
        bail!(
            "cannot unarchive: a file already exists at {}",
            original_path
        );
    }

    // Rebuild frontmatter without archive fields
    let mut tags: Vec<String> = fm_str
        .lines()
        .skip_while(|l| !l.starts_with("tags:"))
        .skip(1)
        .take_while(|l| l.starts_with("  - "))
        .filter_map(|l| l.strip_prefix("  - "))
        .map(|s| s.trim().to_string())
        .filter(|t| t != "archived")
        .collect();
    if tags.is_empty() {
        // Try inline tags format
        if let Some(line) = fm_str.lines().find(|l| l.starts_with("tags:"))
            && let Some(rest) = line.strip_prefix("tags:")
        {
            let rest = rest.trim().trim_start_matches('[').trim_end_matches(']');
            tags = rest
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "archived")
                .collect();
        }
    }

    let new_fm = build_frontmatter(&tags, None, None, None);
    let restored_content = format!("{}{}", new_fm, body);

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

    #[test]
    fn a_missing_section_is_an_error_and_not_a_silent_append() {
        let doc = "# Note\n\n## Spells\n\nbody\n";
        let err = apply_section_edit(doc, "Nowhere", "x", EditMode::Replace).unwrap_err();
        assert!(format!("{err}").contains("Nowhere"));
    }

    #[test]
    fn frontmatter_ops_are_a_pure_transform_of_the_text() {
        let doc = "---\ntags:\n  - a\n---\n\nbody\n";
        let out = apply_frontmatter_ops(
            doc,
            &[
                FrontmatterOp::AddTo("tags".into(), "b".into()),
                FrontmatterOp::Set("status".into(), "done".into()),
            ],
        )
        .unwrap();
        assert!(out.contains("- a"));
        assert!(out.contains("- b"));
        assert!(out.contains("status: done"));
        assert!(out.ends_with("body\n"));
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
    fn test_extract_title() {
        assert_eq!(extract_title("# Hello World\nBody"), "Hello World");
        assert_eq!(extract_title("Just some text"), "Just some text");
    }

    #[test]
    fn test_extract_title_empty() {
        assert_eq!(extract_title(""), "Untitled");
    }

    #[test]
    fn test_extract_title_truncation() {
        let long_title = "a".repeat(100);
        let content = format!("# {}\nBody", long_title);
        assert_eq!(extract_title(&content).len(), 50);
    }

    #[test]
    fn test_build_frontmatter() {
        let fm = build_frontmatter(
            &["work".to_string(), "engraph".to_string()],
            Some("claude-code"),
            None,
            None,
        );
        assert!(fm.starts_with("---\n"));
        assert!(fm.ends_with("---\n\n"));
        assert!(fm.contains("work"));
        assert!(fm.contains("created_by: claude-code"));
    }

    #[test]
    fn test_build_frontmatter_with_aliases() {
        let fm = build_frontmatter(
            &["test".to_string()],
            Some("writer"),
            Some(&["alias1".to_string(), "alias2".to_string()]),
            None,
        );
        assert!(fm.contains("aliases:"));
        assert!(fm.contains("  - alias1"));
        assert!(fm.contains("  - alias2"));
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
    fn test_build_frontmatter_with_suggestion() {
        let suggestion = PlacementSuggestion {
            suggested_folder: "02-Areas/Development".to_string(),
            confidence: 0.58,
            reason: "semantic similarity: 0.580".to_string(),
        };
        let fm = build_frontmatter(
            &["work".to_string()],
            Some("claude-code"),
            None,
            Some(&suggestion),
        );
        assert!(fm.contains("suggested_folder: 02-Areas/Development"));
        assert!(fm.contains("confidence: 0.58"));
        assert!(fm.contains("reason: \"semantic similarity: 0.580\""));
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
        crate::indexer::run_index_shared(&root, &config, &store, &mut embedder, false, None)
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

    // ── Frontmatter merge tests ────────────────────────────────────

    #[test]
    fn test_merge_user_frontmatter_produces_single_block() {
        let user_content =
            "---\ntitle: My Note\ntags:\n  - project\n  - work\n---\n\n# My Note content\n";
        let (user_fm, body) = split_frontmatter(user_content);
        assert!(!user_fm.is_empty());

        let (user_scalars, user_tags, user_aliases) = parse_frontmatter_fields(&user_fm);
        let auto_tags = vec!["project".to_string()];

        let merged = build_merged_frontmatter(
            &auto_tags,
            Some("mcp"),
            None,
            &user_scalars,
            &user_tags,
            &user_aliases,
        );
        let full = format!("{}{}", merged, body);

        // Count frontmatter blocks: should be exactly one
        let fm_count = full.matches("\n---\n").count();
        // The opening ---\n at the start + the closing \n---\n = pattern appears once for closing
        assert!(full.starts_with("---\n"));
        assert_eq!(fm_count, 1, "Should have exactly one closing --- delimiter");
        assert!(full.contains("# My Note content"));
    }

    #[test]
    fn test_merge_tags_deduplicated() {
        let user_fm = "---\ntags:\n  - project\n  - work\n  - rust\n---\n";
        let (user_scalars, user_tags, user_aliases) = parse_frontmatter_fields(user_fm);
        let auto_tags = vec!["project".to_string(), "engraph".to_string()];

        let merged = build_merged_frontmatter(
            &auto_tags,
            Some("mcp"),
            None,
            &user_scalars,
            &user_tags,
            &user_aliases,
        );

        // "project" should appear once, "engraph" from auto, "work" and "rust" from user
        let tag_lines: Vec<&str> = merged.lines().filter(|l| l.starts_with("  - ")).collect();
        assert_eq!(tag_lines.len(), 4);
        assert!(merged.contains("  - project\n"));
        assert!(merged.contains("  - engraph\n"));
        assert!(merged.contains("  - work\n"));
        assert!(merged.contains("  - rust\n"));

        // "project" tag line should appear only once
        let project_count = merged.matches("  - project\n").count();
        assert_eq!(
            project_count, 1,
            "Duplicate tag 'project' should be deduplicated"
        );
    }

    #[test]
    fn test_merge_preserves_user_custom_fields() {
        let user_fm =
            "---\ntitle: My Project\nstatus: active\npriority: high\ntags:\n  - work\n---\n";
        let (user_scalars, user_tags, user_aliases) = parse_frontmatter_fields(user_fm);
        let auto_tags = vec!["project".to_string()];

        let merged = build_merged_frontmatter(
            &auto_tags,
            Some("mcp"),
            None,
            &user_scalars,
            &user_tags,
            &user_aliases,
        );

        assert!(merged.contains("title: My Project"));
        assert!(merged.contains("status: active"));
        assert!(merged.contains("priority: high"));
        assert!(merged.contains("  - work"));
        assert!(merged.contains("  - project"));
    }

    #[test]
    fn test_merge_created_always_auto_generated() {
        let user_fm = "---\ncreated: 2020-01-01\ncreated_by: user\ntitle: Test\n---\n";
        let (user_scalars, user_tags, user_aliases) = parse_frontmatter_fields(user_fm);
        let auto_tags = vec![];

        let merged = build_merged_frontmatter(
            &auto_tags,
            Some("mcp"),
            None,
            &user_scalars,
            &user_tags,
            &user_aliases,
        );

        // created should be today's date, not 2020-01-01
        assert!(!merged.contains("2020-01-01"));
        assert!(merged.contains(&format!("created: {}", today_date())));
        // created_by should be "mcp", not "user"
        assert!(merged.contains("created_by: mcp"));
        assert!(!merged.contains("created_by: user"));
        // But title should still be preserved
        assert!(merged.contains("title: Test"));
    }

    #[test]
    fn test_merge_content_without_frontmatter_unchanged() {
        let content = "# Just a heading\n\nSome body text.\n";
        let (user_fm, body) = split_frontmatter(content);
        assert!(user_fm.is_empty());

        let (user_scalars, user_tags, user_aliases) = parse_frontmatter_fields(&user_fm);
        let auto_tags = vec!["inbox".to_string()];

        let merged = build_merged_frontmatter(
            &auto_tags,
            Some("mcp"),
            None,
            &user_scalars,
            &user_tags,
            &user_aliases,
        );
        let full = format!("{}{}", merged, body);

        // Should have frontmatter from auto-gen only
        assert!(full.starts_with("---\n"));
        assert!(full.contains("  - inbox"));
        assert!(full.contains("created_by: mcp"));
        // Body should be intact
        assert!(full.contains("# Just a heading"));
        assert!(full.contains("Some body text."));
    }

    #[test]
    fn test_merge_user_aliases_preserved() {
        let user_fm = "---\naliases:\n  - My Alias\n  - Another Name\ntags:\n  - test\n---\n";
        let (user_scalars, user_tags, user_aliases) = parse_frontmatter_fields(user_fm);
        let auto_tags = vec!["auto".to_string()];

        let merged = build_merged_frontmatter(
            &auto_tags,
            Some("mcp"),
            None,
            &user_scalars,
            &user_tags,
            &user_aliases,
        );

        assert!(merged.contains("aliases:"));
        assert!(merged.contains("  - My Alias"));
        assert!(merged.contains("  - Another Name"));
        assert!(merged.contains("  - test"));
        assert!(merged.contains("  - auto"));
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

    // ── The tag store (#60) ──────────────────────────────────────

    fn stored_tags(store: &Store, path: &str) -> Vec<String> {
        let file = store.get_file(path).unwrap().unwrap();
        store.file_tags(file.id).unwrap()
    }

    fn test_chunk_opts() -> ChunkOptions {
        ChunkOptions {
            min_chars: 0,
            promote_bold: false,
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
                filename: Some("swamp.md".to_string()),
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
        assert_eq!(property_tags, vec!["work", "archived"]);
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
        crate::indexer::run_index_shared(&vault, &config, &store, &mut embedder, false, None)
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
    }

    /// A vault of one note, indexed, ready for `update_note`.
    fn indexed_note(body: &str) -> (tempfile::TempDir, Store, std::path::PathBuf) {
        use crate::llm::MockLlm;

        let (tmp, store, vault) = setup_vault();
        std::fs::write(vault.join("note.md"), body).unwrap();
        let mut embedder = MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(&vault, &config, &store, &mut embedder, false, None)
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
    #[test]
    fn a_section_edit_reaches_the_section_it_names_in_every_mode() {
        for (mode, wanted, unwanted) in [
            (EditMode::Replace, "New entry", Some("Old entry")),
            (EditMode::Append, "New entry", None),
            (EditMode::Prepend, "New entry", None),
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
            assert!(out.contains(wanted), "{mode:?} lost its content: {out}");
            if let Some(gone) = unwanted {
                assert!(!out.contains(gone), "{mode:?} kept the old body: {out}");
            }
            assert!(
                out.contains("## Links") && out.contains("Some links"),
                "{mode:?} disturbed the next section: {out}"
            );
            // Whatever the mode, the edit lands inside the section it named.
            assert!(
                out.find("New entry").unwrap() < out.find("## Links").unwrap(),
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
            EmbedComposition::default(),
            test_chunk_opts(),
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
        crate::indexer::run_index_shared(&vault, &config, &store, &mut embedder, false, None)
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
                "status:\n- wip",
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
        // A run of property edits becomes one `apply_frontmatter_ops` call
        // (#62), so a two-edit call must give the same bytes as a one-edit
        // call gives, plus the tag. A `contains` assertion cannot see this,
        // so pin the bytes.
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
        assert_eq!(one, "---\ntags:\n- a\n- b\n---\n\n## Spells\n\nold\n");

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
            "---\ntags:\n- a\n- b\nstatus: done\n---\n\n## Spells\n\nold\n"
        );
    }

    /// The Task 12 fix trimmed the leading break `split_frontmatter` leaves
    /// on a body edit; `apply_frontmatter_ops` reassembles the same way and
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
        crate::indexer::run_index_shared(&vault, &config, &store, &mut embedder, false, None)
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
