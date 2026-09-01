use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::chunker::{chunk_markdown, split_oversized_chunks};
use crate::config::{Config, db_path};
use crate::docid::generate_docid;
use crate::exclude::ExcludeMatcher;
use crate::graph::{Wikilink, extract_wikilinks};
use crate::llm::EmbedModel;
use crate::profile::VaultProfile;
use crate::store::{DOC_LEVEL, FileRecord, Store};

/// Summary of an indexing run.
pub struct IndexResult {
    pub new_files: usize,
    pub updated_files: usize,
    pub deleted_files: usize,
    pub total_chunks: usize,
    pub duration: Duration,
}

/// Result of indexing a single file.
pub struct IndexFileResult {
    pub file_id: i64,
    pub total_chunks: usize,
    pub docid: String,
}

/// The index-time settings a store is built at: how a chunk is cut, and the
/// vector it is embedded as. Both must be the same for every row in one store —
/// a row cut or embedded at settings the rest of the store was not built at is
/// a row nothing downstream can tell apart, because both are fingerprint
/// components and a full index records whatever it ran at (issue #72).
///
/// It bundles [`ChunkOptions`] and [`EmbedComposition`] — each already one value
/// so a caller cannot thread half of it — into one value for the same reason,
/// one level up: every indexing entry point takes this, so a caller cannot pass
/// the chunking and forget the embedding. [`IndexSettings::from_config`] is the
/// one place both are read out of a `Config`; a session that has captured them
/// carries this value instead, so a fresh `Config::load` cannot be a second
/// source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSettings {
    pub chunk: crate::chunker::ChunkOptions,
    pub embed: crate::prefix::EmbedComposition,
}

impl IndexSettings {
    /// Read both settings off a `Config`. For a caller whose config is the
    /// session — the CLI, a test — this is the whole of it; a long-running
    /// server captures the result once and carries it, so a later handler's
    /// fresh load cannot drift the store's chunking or vector space.
    pub fn from_config(config: &Config) -> Self {
        Self {
            chunk: config.chunk_options(),
            embed: crate::prefix::EmbedComposition::from_config(config),
        }
    }
}

/// Walk a vault directory and collect all `.md` file paths.
///
/// When `respect_gitignore` is true, the `ignore` crate honors `.gitignore` /
/// `.ignore` rules (within a git repo, per the crate's `require_git` default).
/// Set it false to index files those VCS rules would skip; hidden entries
/// (`.git/`, dotfiles) and the explicit `exclude` patterns are always skipped.
///
/// `exclude` holds globs — see [`ExcludeMatcher`] for the pattern syntax. An
/// unparseable pattern is an error here rather than a filter that quietly passes
/// everything through.
pub fn walk_vault(
    path: &Path,
    exclude: &[String],
    respect_gitignore: bool,
) -> Result<Vec<PathBuf>> {
    let matcher = ExcludeMatcher::new(exclude)?;
    let mut builder = WalkBuilder::new(path);
    if respect_gitignore {
        builder.standard_filters(true); // respect .gitignore, .ignore, hidden, etc.
    } else {
        // Stop honoring .gitignore / .ignore so VCS-ignored files get indexed,
        // but still skip hidden entries (.git/, dotfiles) and apply the
        // explicit exclude patterns below.
        builder
            .hidden(true)
            .parents(true)
            .git_ignore(false)
            .git_global(false)
            .git_exclude(false)
            .ignore(false);
    }
    let walker = builder.build();

    let mut files = Vec::new();
    for entry in walker {
        let entry = entry.context("error reading directory entry")?;
        let entry_path = entry.path();

        // Only regular files.
        if !entry_path.is_file() {
            continue;
        }

        // Only .md files.
        match entry_path.extension() {
            Some(ext) if ext == "md" => {}
            _ => continue,
        }

        // Check exclude patterns.
        if matcher.matches_under(entry_path, path) {
            continue;
        }

        files.push(entry_path.to_path_buf());
    }

    files.sort();
    Ok(files)
}

/// Compute the SHA-256 hash of a file's contents, returned as a hex string.
pub fn compute_file_hash(path: &Path) -> Result<String> {
    let content = std::fs::read(path)
        .with_context(|| format!("reading file for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compare vault files against the store to find new, changed, and deleted files.
///
/// Returns `(new_files, changed_files, deleted_file_records)`.
pub fn diff_vault(
    files: &[PathBuf],
    vault_root: &Path,
    store: &Store,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>, Vec<FileRecord>)> {
    let stored_files = store.get_all_files()?;
    let stored_map: HashMap<String, &FileRecord> =
        stored_files.iter().map(|f| (f.path.clone(), f)).collect();

    let mut new_files = Vec::new();
    let mut changed_files = Vec::new();

    // Track which stored paths we've seen on disk.
    let mut seen_paths = std::collections::HashSet::new();

    for file_path in files {
        let rel = file_path.strip_prefix(vault_root).unwrap_or(file_path);
        let rel_str = rel.to_string_lossy().to_string();

        seen_paths.insert(rel_str.clone());

        match stored_map.get(&rel_str) {
            None => {
                new_files.push(file_path.clone());
            }
            Some(record) => {
                let current_hash = compute_file_hash(file_path)?;
                if current_hash != record.content_hash {
                    changed_files.push(file_path.clone());
                }
            }
        }
    }

    // Files in store but not on disk are deleted.
    let deleted: Vec<FileRecord> = stored_files
        .into_iter()
        .filter(|f| !seen_paths.contains(&f.path))
        .collect();

    Ok((new_files, changed_files, deleted))
}

/// Re-derive every edge in the vault from source, without embedding anything.
///
/// The action `link_fingerprint` declares (issue #31). A resolver change is not
/// confined to the files that changed on disk: the same `[[Note]]` may now point
/// somewhere else in every file that writes it, and no content hash can see
/// that, because no content changed.
///
/// This reads the vault rather than re-deriving from `chunks.text` the way
/// `backfill_edges_from_chunks` does, because the chunker strips frontmatter —
/// a link written there belongs to no chunk, and a chunks-only rebuild would
/// silently drop it. That is a vault read of every file and no model call, so it
/// costs seconds where a reindex costs minutes.
fn rebuild_all_edges(store: &Store, vault_path: &Path, files: &[PathBuf]) -> Result<usize> {
    store.clear_edges()?;
    let mut rebuilt = 0usize;
    for path in files {
        let rel = path.strip_prefix(vault_path).unwrap_or(path);
        let rel_str = rel.to_string_lossy().to_string();
        let Some(file_record) = store.get_file(&rel_str)? else {
            continue;
        };
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        build_edges_for_file(store, file_record.id, &content)?;
        rebuilt += 1;
    }
    Ok(rebuilt)
}

/// Resolve a wikilink target name to a file ID in the store.
fn resolve_link_target(store: &Store, target: &str) -> Result<Option<i64>> {
    let with_ext = if target.ends_with(".md") {
        target.to_string()
    } else {
        format!("{}.md", target)
    };

    // Try exact path match
    if let Some(f) = store.get_file(&with_ext)? {
        return Ok(Some(f.id));
    }

    // Try basename match (case-insensitive)
    let all_files = store.get_all_files()?;
    let target_lower = with_ext.to_lowercase();
    let mut matches: Vec<&FileRecord> = all_files
        .iter()
        .filter(|f| {
            let path_lower = f.path.to_lowercase();
            path_lower == target_lower || path_lower.ends_with(&format!("/{}", target_lower))
        })
        .collect();

    matches.sort_by_key(|f| f.path.len());
    Ok(matches.first().map(|f| f.id))
}

/// Build wikilink edges for a single file, at chunk granularity on both ends.
///
/// For each `[[target]]` wikilink:
/// - If target resolves: insert ONE directed edge from source → target.
///   Wikilinks are directional — the reverse edge should only exist if
///   the target's own content contains a wikilink back to source. (That
///   reverse edge gets inserted when `build_edges_for_file` is called on
///   the target file with its own content.)
/// - If target doesn't resolve: record in `unresolved_links` for
///   downstream broken-wikilink tooling.
///
/// Both ends of the edge name a passage where one can be named (issue #28):
///
/// - **Source**: the chunk whose text held the link. Read from the store, so
///   **the file's chunks must already be inserted when this is called** — every
///   caller does that, and `an_edges_source_chunk_is_the_one_that_held_the_link`
///   is the guard. A link in text no chunk holds — frontmatter, which the
///   chunker strips — is attributed to [`DOC_LEVEL`], which is why `content` is
///   still a parameter: the chunks alone cannot see it.
/// - **Target**: the chunks under the named `#Heading`, or [`DOC_LEVEL`] for a
///   plain `[[Note]]`. A heading that no longer resolves degrades to
///   [`DOC_LEVEL`] as well — never to nothing.
///
/// Clears pre-existing `unresolved_links` entries for the source file
/// before re-recording, so this is safe to call repeatedly during
/// incremental indexing.
pub fn build_edges_for_file(store: &Store, file_id: i64, content: &str) -> Result<()> {
    if store.get_file_by_id(file_id)?.is_none() {
        return Ok(()); // file vanished mid-index; no-op
    }

    // Clear stale unresolved entries for this file before re-recording.
    store.clear_unresolved_links_for_file(file_id)?;

    // Which passage each link came from. A link in more than one chunk gets an
    // edge from each — the multiplicity the old file-level UNIQUE discarded.
    let mut sources: HashMap<Wikilink, Vec<i64>> = HashMap::new();
    for chunk in store.get_chunks_by_file(file_id)? {
        for link in extract_wikilinks(&chunk.text) {
            sources.entry(link).or_default().push(chunk.seq);
        }
    }
    // Anything the whole file contains but no chunk claims is unattributable.
    for link in extract_wikilinks(content) {
        sources.entry(link).or_insert_with(|| vec![DOC_LEVEL]);
    }

    for (link, from_seqs) in sources {
        let Some(target_id) = resolve_link_target(store, &link.target)? else {
            store.insert_unresolved_link(file_id, &link.target)?;
            continue;
        };
        if target_id == file_id {
            continue; // self-link
        }
        // A deep link names chunks; a plain one, or a heading that has since
        // been renamed, names the document.
        let to_seqs = match &link.heading {
            Some(h) => match store.chunk_seqs_with_heading(target_id, h)? {
                seqs if seqs.is_empty() => vec![DOC_LEVEL],
                seqs => seqs,
            },
            None => vec![DOC_LEVEL],
        };
        for &from_seq in &from_seqs {
            for &to_seq in &to_seqs {
                store.insert_edge(file_id, from_seq, target_id, to_seq, "wikilink")?;
            }
        }
    }
    Ok(())
}

/// Re-derive every wikilink edge from `chunks.text`, with no vault read (#28).
///
/// The adoption path for a store written before edges had chunk granularity.
/// `chunks.text` has held every chunk's content since #14, so the whole edge
/// table can be rebuilt from the database — nothing is re-read, re-chunked or
/// re-embedded, and a 250-file vault takes well under a second.
///
/// Document-level rows the migration carried over are cleared first, *except*
/// for pairs no chunk turns out to account for. Those are the links the chunker
/// never sees — frontmatter — and dropping them would lose edges a full reindex
/// keeps. That exception is the only reason this is not a plain
/// `clear_edges` + rebuild.
pub fn backfill_edges_from_chunks(store: &Store) -> Result<usize> {
    let files = store.get_all_files()?;
    let before: HashSet<(i64, i64)> = store.wikilink_pairs()?.into_iter().collect();

    for file in &files {
        store.delete_outgoing_edges_for_file(file.id)?;
    }
    for file in &files {
        // `content` is the chunks and nothing else here — there is no vault read
        // to find frontmatter in, so unattributable links are restored below
        // from what the pre-#28 table already knew.
        let chunks = store.get_chunks_by_file(file.id)?;
        let content = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        build_edges_for_file(store, file.id, &content)?;
    }

    let after: HashSet<(i64, i64)> = store.wikilink_pairs()?.into_iter().collect();
    let mut restored = 0;
    for (from_file, to_file) in before.difference(&after) {
        store.insert_edge(*from_file, DOC_LEVEL, *to_file, DOC_LEVEL, "wikilink")?;
        restored += 1;
    }
    store.set_meta("edges_backfill_pending", "0")?;
    Ok(restored)
}

/// Re-derive one file's index rows from what a write left on disk: chunks,
/// vectors, keyword rows, tag rows and outgoing edges.
///
/// A write path has to call this itself. `writer::update_note` stores the new
/// content hash, so `diff_vault` will not list the file as changed at the next
/// index, and a server that records the write makes the watcher skip its one
/// event. Without this call the note's rows keep the text it held before the
/// edit, and nothing later re-derives them (#62).
///
/// The chunker settings and the embedding composition both come from the
/// caller's session, not from the config this loads: a load that fails falls
/// back to the defaults, and one file re-chunked — or re-embedded — at settings
/// the rest of the store was not built at is a set of rows nothing downstream
/// can tell apart. `EmbedComposition` carries `[embedding_prefix]`,
/// `[embedding_prompt] document_title` and `breadcrumb_root` together for that
/// reason: a caller that threads one and forgets the others writes vectors into
/// a space the store does not share, which is what `prefix::EmbedComposition`
/// exists to prevent. It takes both as one [`IndexSettings`], so a caller
/// cannot thread the chunking and forget the embedding.
pub fn reindex_written_file(
    rel_path: &str,
    store: &Store,
    embedder: &mut impl EmbedModel,
    vault_path: &Path,
    settings: IndexSettings,
) -> Result<IndexFileResult> {
    let full_path = vault_path.join(rel_path);
    let content = std::fs::read_to_string(&full_path)
        .with_context(|| format!("reading written file for re-index: {rel_path}"))?;
    let content_hash = {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    };

    let mut config = Config::load().unwrap_or_default();
    config.set_chunk_options(settings.chunk);
    config.set_embed_composition(settings.embed);

    let result = index_file(
        rel_path,
        &content,
        &content_hash,
        store,
        embedder,
        vault_path,
        &config,
    )?;

    // Outgoing only — see issue #27.
    store.delete_outgoing_edges_for_file(result.file_id)?;
    build_edges_for_file(store, result.file_id, &content)?;

    Ok(result)
}

/// Process a single file: chunk, embed, and store in a single transaction.
///
/// This is the self-contained per-file indexing unit. If the file already exists
/// in the store, old entries (vec, FTS, file record) are cleaned up first.
pub fn index_file(
    rel_path: &str,
    content: &str,
    content_hash: &str,
    store: &Store,
    embedder: &mut impl EmbedModel,
    vault_path: &Path,
    config: &Config,
) -> Result<IndexFileResult> {
    let overlap_tokens = crate::chunker::OVERLAP_TOKENS;

    // 1. Parse frontmatter for created_by
    let parsed = chunk_markdown(content, config.chunk_options());
    let chunks = {
        let tc = |s: &str| embedder.token_count(s);
        split_oversized_chunks(parsed.chunks, &tc, embedder.max_context(), overlap_tokens)
    };

    // Extract created_by from frontmatter
    let (frontmatter, _body) = crate::writer::split_frontmatter(content);
    let created_by: Option<String> = frontmatter.lines().find_map(|line| {
        let trimmed = line.trim();
        if let Some(val) = trimmed.strip_prefix("created_by:") {
            let val = val.trim().trim_matches('"').trim_matches('\'');
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
        None
    });

    // Extract note_date from frontmatter or filename
    let note_date = crate::temporal::extract_note_date(&frontmatter, rel_path);

    // 2. Embed all chunks
    let token_counts: Vec<usize> = chunks
        .iter()
        .map(|c| embedder.token_count(&c.text))
        .collect();
    // What the embedder is shown: the title field (issue #36) and a body that
    // carries the contextual prefix when it is on (issue #2). `inputs` is held
    // so the batch below can borrow from it; nothing here reaches storage, which
    // still persists `chunk.text` / `chunk.snippet` verbatim.
    let doc = crate::prefix::DocContext::from_file(rel_path, content);
    let inputs = crate::prefix::embed_inputs(
        &doc,
        &chunks,
        crate::prefix::EmbedComposition::from_config(config),
    );
    // What the keyword lane is shown, stored on the chunk row itself (issue
    // #37). The same `doc`, so the breadcrumb here and the breadcrumb in the
    // title field above are one string.
    let lexical = crate::prefix::lexical_fields(&doc, &chunks, config.breadcrumb_root);
    let docs: Vec<crate::llm::EmbedDoc<'_>> = inputs
        .iter()
        .map(crate::prefix::EmbedInput::as_doc)
        .collect();
    let mut all_vectors = Vec::with_capacity(docs.len());
    for batch in docs.chunks(config.batch_size) {
        let vectors = embedder.embed_batch(batch)?;
        all_vectors.extend(vectors);
    }

    // 3. Compute mtime
    let mtime = std::fs::metadata(vault_path.join(rel_path))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let docid = generate_docid(rel_path);

    // 4. Begin transaction (skip if caller already opened one)
    let owns_transaction = store.conn().is_autocommit();
    if owns_transaction {
        store.conn().execute_batch("BEGIN DEFERRED")?;
    }

    // 5. If file already exists, clean up old entries.
    //
    // Deliberately NOT `delete_file`: the row must survive so the file keeps its
    // id, and the id is what every backlink into this file is keyed on. Dropping
    // the row cascades those edges away and nothing puts them back, because the
    // files that own them are not being re-indexed (issue #27). `insert_file`
    // below upserts on `path` and returns the same id.
    if let Some(record) = store.get_file(rel_path)? {
        let vector_ids = store.get_vector_ids_for_file(record.id)?;
        for &vid in &vector_ids {
            store.delete_vec(vid)?;
        }
        // The keyword index goes with the chunks: `chunks_fts` is external
        // content over `chunks`, and its delete trigger fires on every row this
        // statement removes (issue #37).
        store.delete_chunks_for_file(record.id)?;
    }

    // 6. Insert new file and chunks
    let file_id = store.insert_file(
        rel_path,
        content_hash,
        mtime,
        &docid,
        created_by.as_deref(),
        note_date,
    )?;

    let start_vector_id: u64 = store.next_vector_id()?;
    let total_chunks = chunks.len();

    for (chunk_seq, chunk) in chunks.iter().enumerate() {
        let heading = chunk.heading.clone().unwrap_or_default();
        let vector = &all_vectors[chunk_seq];
        let vector_id = start_vector_id + chunk_seq as u64;

        // The whole chunk goes to storage; the store derives the snippet from
        // it (issue #14). The keyword index needs no write of its own — the
        // insert trigger indexes this row from the columns below (issue #37).
        store.insert_chunk_with_vector(
            &crate::store::NewChunk {
                file_id,
                seq: chunk_seq as i64,
                heading: &heading,
                heading_path: &lexical[chunk_seq].heading_path,
                tags_text: &lexical[chunk_seq].tags_text,
                text: &chunk.text,
                vector_id,
                token_count: token_counts[chunk_seq] as i64,
            },
            vector,
        )?;
        store.insert_vec(vector_id, vector)?;
    }

    // 7. Reconcile the file's tags (#60)
    //
    // The extractor reads the property and the body, and the store owns this
    // file's rows the way step 5 owns its chunks. The watcher calls this
    // function, so the warm path needs no second implementation.
    store.reconcile_file_tags(file_id, &crate::tags::extract(content))?;

    // 8. Commit (only if we own the transaction)
    if owns_transaction {
        store.commit()?;
    }

    Ok(IndexFileResult {
        file_id,
        total_chunks,
        docid,
    })
}

/// Remove a file from the store, cleaning up vec, FTS, and cascading chunks/edges.
///
/// sqlite-vec virtual tables don't participate in CASCADE deletes, so we must
/// manually delete vector entries before removing the file record.
pub fn remove_file(rel_path: &str, store: &Store) -> Result<()> {
    let file = store
        .get_file(rel_path)?
        .ok_or_else(|| anyhow!("File not found: '{}'", rel_path))?;

    let owns_transaction = store.conn().is_autocommit();
    if owns_transaction {
        store.conn().execute_batch("BEGIN DEFERRED")?;
    }

    let vector_ids = store.get_vector_ids_for_file(file.id)?;
    for &vid in &vector_ids {
        store.delete_vec(vid)?;
    }
    // The ids this file is about to release. `file_tags` cascades off the
    // `files` row below, so steps 1 and 2 of reconciliation need no code here;
    // step 3 does, and it needs the ids read before the cascade (#60).
    let released_tags = store.file_tag_ids(file.id)?;
    // No FTS delete: `chunks` CASCADEs off the `files` row below, and the
    // keyword index follows the chunks (issue #37). `unresolved_links` rides
    // the same cascade since #98, so the file's broken links need no call of
    // their own either.
    store.delete_file(file.id)?;
    store.prune_unused_tags(&released_tags)?;

    if owns_transaction {
        store.commit()?;
    }
    Ok(())
}

/// Rename a file in the store, preserving its file_id and all edge integrity.
///
/// Recomputes the docid from the new path and delegates to `Store::update_file_path`
/// which performs a collision check and updates the path in place.
pub fn rename_file(old_rel: &str, new_rel: &str, store: &Store) -> Result<()> {
    let new_docid = generate_docid(new_rel);
    store.update_file_path(old_rel, new_rel, &new_docid)?;
    Ok(())
}

/// Main indexing orchestrator.
///
/// Walks the vault, diffs against the store, processes new/changed/deleted files,
/// embeds chunks in parallel, and writes everything to the store.
pub fn run_index(
    vault_path: &Path,
    config: &Config,
    settings: IndexSettings,
    rebuild: bool,
) -> Result<IndexResult> {
    let data_dir = Config::data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    let db_path = db_path(&data_dir);
    let store = Store::open(&db_path)?;

    let models_dir = data_dir.join("models");
    let mut embedder = crate::llm::load_embedder(&models_dir, config)?;

    let profile = crate::config::Config::load_vault_profile().ok().flatten();
    run_index_inner(
        vault_path,
        config,
        settings,
        &store,
        &mut embedder,
        rebuild,
        profile.as_ref(),
    )
}

/// Like [`run_index`], but accepts shared `Store` and `Embedder` references.
///
/// Useful when the caller already owns these resources (e.g. a file watcher
/// performing a full rescan without re-opening the database or reloading the model).
pub fn run_index_shared(
    vault_path: &Path,
    config: &Config,
    settings: IndexSettings,
    store: &Store,
    embedder: &mut impl EmbedModel,
    rebuild: bool,
    profile: Option<&VaultProfile>,
) -> Result<IndexResult> {
    run_index_inner(
        vault_path, config, settings, store, embedder, rebuild, profile,
    )
}

/// Shared implementation for [`run_index`] and [`run_index_shared`].
fn run_index_inner(
    vault_path: &Path,
    config: &Config,
    settings: IndexSettings,
    store: &Store,
    embedder: &mut impl EmbedModel,
    rebuild: bool,
    profile: Option<&VaultProfile>,
) -> Result<IndexResult> {
    // The single application point for the index-time settings (#72). Everything
    // below — the fingerprint that decides a rebuild, and `index_file`'s chunk
    // and embed — reads them back off this reconciled config, the way
    // `reindex_written_file` already does for one file. A caller reaches this
    // function only through a signature that asks for `settings`, so a bare
    // `Config` cannot be a second source of either.
    let mut config = config.clone();
    config.set_chunk_options(settings.chunk);
    config.set_embed_composition(settings.embed);
    let config = &config;

    let start = Instant::now();

    // Size vector storage to the model before anything is written. On a width
    // change this discards the index, so a full rebuild is not optional
    // (issue #12).
    let mut rebuild = rebuild;
    if let Some(previous) = store.ensure_embedding_dim(embedder.dim())? {
        eprintln!(
            "Embedding dimension changed ({previous} -> {}). \
             Re-indexing vault (this may take a few minutes)...",
            embedder.dim()
        );
        rebuild = true;
    }

    // Reconcile what built this index against what is running now (issue #31).
    //
    // No reranker fingerprint here, and deliberately none: nothing the index
    // path writes passes through a cross-encoder, so this path has no business
    // deciding a reranker is current. Passing `None` leaves whatever a reranked
    // run recorded untouched rather than overwriting it with a guess.
    let fingerprints =
        crate::fingerprint::Fingerprints::compute(config, &embedder.fingerprint(), None);
    let staleness = crate::fingerprint::compare(store, &fingerprints)?;
    crate::fingerprint::warn_unrecorded(&staleness);
    let actions = staleness.actions();
    for mismatch in &staleness.mismatches {
        eprintln!(
            "{} changed since the index was built. Will {}.",
            mismatch.key,
            mismatch.action.describe()
        );
    }
    if actions.contains(&crate::fingerprint::Action::Reindex) {
        rebuild = true;
    }
    // Before the file loop, so anything indexed below lands in the new schema
    // rather than being written into the old one and then thrown away.
    if actions.contains(&crate::fingerprint::Action::RebuildFts) {
        let rows = store.rebuild_fts(&config.fts)?;
        info!(rows, "keyword index rebuilt from stored chunks");
    }
    // A store with no recorded fingerprints is adopted rather than rebuilt
    // (issue #31), and a fresh one gets its keyword index from `Store::init`,
    // which has read no config. Either way the declaration can still disagree
    // with `[fts]`, and this is the path that holds the config, so it is the
    // path that reconciles the two.
    if let Some(rows) = store.sync_fts_objects(&config.fts)? {
        info!(rows, "keyword index rebuilt to match [fts]");
    }

    let cleaned = crate::writer::cleanup_temp_files(vault_path)?;
    if cleaned > 0 {
        info!(cleaned, "cleaned up incomplete writes from previous run");
    }

    let orphans = crate::writer::verify_index_integrity(store, vault_path)?;
    if orphans > 0 {
        info!(orphans, "cleaned up orphan DB entries for missing files");
    }

    // Build exclude list: config excludes + archive folder (if detected)
    let mut exclude = config.exclude.clone();
    if let Some(p) = profile
        && let Some(archive) = &p.structure.folders.archive
    {
        let archive_pattern = format!("{}/", archive);
        if !exclude.contains(&archive_pattern) {
            exclude.push(archive_pattern);
        }
    }

    // If rebuild, treat everything as new.
    let files = walk_vault(vault_path, &exclude, config.respect_gitignore)?;

    let (new_files, changed_files, deleted_files) = if rebuild {
        // On rebuild we skip diffing — all files are "new".
        store.clear_vec()?;
        (files.clone(), Vec::new(), Vec::new())
    } else {
        let (n, c, d) = diff_vault(&files, vault_path, store)?;
        (n, c, d)
    };

    info!(
        new = new_files.len(),
        changed = changed_files.len(),
        deleted = deleted_files.len(),
        "diff complete"
    );

    // Step 4: Handle deleted files — remove vectors from vec0, FTS, and store.
    for record in &deleted_files {
        remove_file(&record.path, store)?;
    }

    // Step 5: Handle changed files — just queue them.
    //
    // `remove_file` used to run here first. It is a *deletion*: it drops the
    // `files` row, and `edges` CASCADEs off that row in both directions, so
    // every backlink into a changed file died on each incremental index and
    // nothing put it back (issue #27). Everything it cleaned up is cleaned up
    // anyway — `index_file` clears the file's vectors, FTS rows and chunks, and
    // `build_edges_for_file` clears its `unresolved_links` — and skipping it is
    // what lets the file keep its id.
    let mut files_to_index: Vec<PathBuf> = new_files.clone();
    for file_path in &changed_files {
        files_to_index.push(file_path.clone());
    }

    // Step 6: Read content, index each file via index_file.
    // Read all file contents and compute hashes.
    let file_contents: Vec<(String, String, String)> = files_to_index
        .iter()
        .filter_map(|p| {
            let hash = compute_file_hash(p).ok()?;
            let content = std::fs::read_to_string(p).ok()?;
            let rel = p.strip_prefix(vault_path).unwrap_or(p);
            let rel_str = rel.to_string_lossy().to_string();
            Some((rel_str, content, hash))
        })
        .collect();

    // Preserve raw content for edge building (wikilink extraction needs full text).
    let content_by_path: HashMap<String, String> = file_contents
        .iter()
        .map(|(rel_str, content, _hash)| (rel_str.clone(), content.clone()))
        .collect();

    // Serial: chunk, embed, and write each file via index_file.
    // Wrap in a single transaction so we get one fsync instead of N.
    let mut total_chunks = 0usize;
    let mut indexed_rel_paths: Vec<String> = Vec::new();

    let pb = ProgressBar::new(file_contents.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("  [{bar:40.cyan/blue}] {pos}/{len} {msg} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    // A failing file is skipped rather than propagated: an API embedder can die
    // mid-index, and one bad file must not cost the whole run. `index_file`
    // embeds before it writes anything (see its step 2 vs. steps 4-8), so an
    // embed failure leaves the store untouched for this file on its own — but
    // a failure *after* the embed, inside one of `index_file`'s own store
    // writes (insert_file / insert_chunk_with_vector / reconcile_file_tags),
    // would otherwise leave that file's partial rows sitting in the ambient
    // transaction below, to be persisted by the final `store.commit()` along
    // with everything that actually succeeded. A per-file SAVEPOINT closes
    // that gap: on `Err` it rolls back exactly this file's writes (a no-op
    // for the embed-failure case, since nothing was written) so a skipped
    // file is left exactly as it was either way. A new file then has no row,
    // a changed file keeps its prior row, chunks and content hash, and
    // `diff_vault` re-lists it next run and it is re-attempted (#84).
    let mut skipped: Vec<String> = Vec::new();
    store.conn().execute_batch("BEGIN DEFERRED")?;
    for (rel_str, content, hash) in &file_contents {
        pb.set_message(rel_str.clone());
        store.conn().execute_batch("SAVEPOINT file_sp")?;
        match index_file(rel_str, content, hash, store, embedder, vault_path, config) {
            Ok(result) => {
                store.conn().execute_batch("RELEASE file_sp")?;
                total_chunks += result.total_chunks;
                indexed_rel_paths.push(rel_str.clone());
            }
            Err(e) => {
                store
                    .conn()
                    .execute_batch("ROLLBACK TO file_sp; RELEASE file_sp")?;
                tracing::warn!(path = %rel_str, error = %e, "embedding failed, skipping file");
                skipped.push(rel_str.clone());
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("done");
    store.commit()?;

    // Step 9: Build vault graph edges.
    info!("building vault graph edges");
    if rebuild {
        store.clear_edges()?;
    }

    for rel_path in &indexed_rel_paths {
        if let Some(file_record) = store.get_file(rel_path)?
            && let Some(content) = content_by_path.get(rel_path)
        {
            // Clear what this file used to own before recomputing it. Without
            // this the incremental path (no `clear_edges` above) can only ever
            // add edges, because `insert_edge` is INSERT OR IGNORE — a wikilink
            // deleted from a file would stay in the graph forever (issue #27).
            // Incoming edges are left alone: they belong to other files.
            store.delete_outgoing_edges_for_file(file_record.id)?;
            build_edges_for_file(store, file_record.id, content)?;
        }
    }

    // A link-resolver change rewrites every edge, including those of files that
    // did not change (issue #31). A full rebuild has already done this from
    // source above, so this is the incremental path's version of it: a vault
    // read and no model, which is what makes it cheap enough to do eagerly.
    if actions.contains(&crate::fingerprint::Action::RebuildEdges) && !rebuild {
        let rebuilt = rebuild_all_edges(store, vault_path, &files)?;
        info!(
            files = rebuilt,
            "vault graph re-derived after a link-resolver change"
        );
    }

    // Adopt chunk-granular edges on a store that predates them (issue #28).
    // Only the files touched above were rebuilt at the fine grain; the rest are
    // still document-level, and re-deriving them needs no vault read, so this
    // costs a fraction of a second even when nothing else changed. `--rebuild`
    // has already done the work from source.
    if store.needs_edge_backfill()? {
        if rebuild {
            // `clear_edges` above plus the loop just run rebuilt every edge from
            // the vault itself, which is strictly better than the backfill.
            store.set_meta("edges_backfill_pending", "0")?;
        } else {
            info!("re-deriving edges at chunk granularity from stored chunks");
            let unattributed = backfill_edges_from_chunks(store)?;
            info!(
                unattributed,
                "edges re-derived; the remainder are links no chunk contains"
            );
        }
    }

    // Step 10: Store vault path in meta.
    store.set_meta("vault_path", &vault_path.to_string_lossy())?;
    store.set_meta(
        "last_indexed_at",
        &format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
    )?;

    // Step 11: Compute folder centroids for placement engine.
    // Recompute from all chunks in the store for indexed files.
    info!("computing folder centroids");
    let mut folder_vecs: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for rel_path in &indexed_rel_paths {
        let folder = rel_path.split('/').next().unwrap_or("(root)").to_string();
        if let Some(file_record) = store.get_file(rel_path)? {
            let chunk_vectors = store.get_chunk_vectors_for_file(file_record.id)?;
            for vector in chunk_vectors {
                folder_vecs.entry(folder.clone()).or_default().push(vector);
            }
        }
    }

    for (folder, vectors) in &folder_vecs {
        if vectors.is_empty() {
            continue;
        }
        let dim = embedder.dim();
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            for (i, val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let n = vectors.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }
        store.upsert_folder_centroid(folder, &centroid, vectors.len())?;
    }

    // Extract L1 identity facts from the freshly indexed vault
    if let Some(p) = profile
        && let Err(e) = crate::identity::extract_l1_facts(store, p)
    {
        tracing::warn!("L1 identity extraction failed (non-fatal): {e:#}");
    }

    // Last, and only on the way out (issue #31). A crash anywhere above leaves
    // the previous fingerprints standing, so the next run repeats the work — a
    // store never claims to match code that never finished running against it.
    // A skipped file is the same case: this pass did not finish embedding the
    // vault, so recording now would tell the next run the index is current when
    // it is not (#84).
    if skipped.is_empty() {
        crate::fingerprint::record(store, &fingerprints)?;
    } else {
        eprintln!(
            "{} file(s) were not embedded and were left for a later run: {}",
            skipped.len(),
            skipped.join(", ")
        );
        eprintln!("Index left unfingerprinted; re-run `knapper index` to complete it.");
    }

    let duration = start.elapsed();
    info!(
        new = new_files.len(),
        updated = changed_files.len(),
        deleted = deleted_files.len(),
        chunks = total_chunks,
        duration_secs = duration.as_secs_f64(),
        "indexing complete"
    );

    Ok(IndexResult {
        new_files: new_files.len(),
        updated_files: changed_files.len(),
        deleted_files: deleted_files.len(),
        total_chunks,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn index_settings_from_config_reads_both_sides() {
        // The one place both index-time settings are read off a `Config` reads
        // both, not one (#72). A config whose chunking and whose embedding
        // composition are each moved off the defaults must round-trip through
        // `IndexSettings::from_config` whole.
        let mut config = Config::default();
        config.chunk_min_chars = 0;
        config.promote_bold_headings = false;
        config.carry_orphan_headings = false;
        config.breadcrumb_root = crate::config::BreadcrumbRoot::Name;
        config.embedding_prompt.document_title = crate::llm::DocumentTitle::Breadcrumb;

        let settings = IndexSettings::from_config(&config);
        assert_eq!(settings.chunk, config.chunk_options());
        assert_eq!(
            settings.embed,
            crate::prefix::EmbedComposition::from_config(&config)
        );
        // Both sides actually moved, so neither assertion passes by defaulting.
        assert_ne!(settings.chunk, Config::default().chunk_options());
        assert_ne!(
            settings.embed,
            crate::prefix::EmbedComposition::from_config(&Config::default())
        );
    }

    /// Helper: create a file with given content inside a temp directory.
    fn write_file(dir: &Path, rel_path: &str, content: &str) {
        let full = dir.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full, content).unwrap();
    }

    #[test]
    fn test_walk_collects_md_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note1.md", "# Note 1");
        write_file(root, "note2.md", "# Note 2");
        write_file(root, "sub/note3.md", "# Note 3");
        write_file(root, "image.png", "not markdown");
        write_file(root, "readme.txt", "text file");

        let files = walk_vault(root, &[], true).unwrap();
        assert_eq!(files.len(), 3, "expected 3 .md files, got {:?}", files);
        for f in &files {
            assert_eq!(f.extension().unwrap(), "md");
        }
    }

    #[test]
    fn test_walk_excludes_patterns() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note.md", "# Note");
        write_file(root, ".obsidian/workspace.md", "obsidian internal");
        write_file(root, ".obsidian/plugins/plugin.md", "plugin data");

        let files = walk_vault(root, &[".obsidian/".to_string()], true).unwrap();
        assert_eq!(files.len(), 1, "expected 1 file, got {:?}", files);
        assert!(files[0].ends_with("note.md"));
    }

    #[test]
    fn test_walk_excludes_globs_at_any_depth() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "lore/archdragon.md", "# Archdragon");
        write_file(root, "lore/lore-index.md", "# Lore index");
        write_file(root, "rules/spell-index.md", "# Spell index");
        write_file(root, "templates/npc.md", "# NPC template");

        let exclude = vec!["*-index.md".to_string(), "templates/".to_string()];
        let files = walk_vault(root, &exclude, true).unwrap();

        assert_eq!(files.len(), 1, "expected 1 file, got {:?}", files);
        assert!(files[0].ends_with("archdragon.md"));
    }

    #[test]
    fn test_walk_rejects_invalid_exclude_pattern() {
        let tmp = TempDir::new().unwrap();
        write_file(tmp.path(), "note.md", "# Note");

        let err = walk_vault(tmp.path(), &["[unclosed.md".to_string()], true).unwrap_err();
        assert!(
            err.to_string().contains("invalid exclude pattern"),
            "unexpected error: {err}"
        );
    }

    /// `writer::update_note` writes the file and the new content hash and no
    /// chunks at all, so the store keeps the pre-edit text unless the write
    /// path re-indexes. The hash it stored is the same SHA-256 `diff_vault`
    /// computes, so a later index finds nothing to do — this is the call that
    /// stops a note from staying searchable as text it no longer holds (#62).
    #[test]
    fn re_indexing_a_written_file_replaces_the_chunk_text() {
        use crate::llm::MockLlm;

        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note.md", "---\ntags:\n  - a\n---\n\nthe old line\n");

        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        let config = Config::default();
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let input = crate::writer::UpdateInput {
            file: "note.md".into(),
            edits: vec![crate::writer::NoteEdit {
                target: crate::writer::EditTarget::Body,
                heading: None,
                mode: crate::writer::EditMode::Append,
                content: Some(crate::writer::EditContent::Text("the new line".into())),
            }],
        };
        crate::writer::update_note(&store, root, &input).unwrap();

        // The written hash already matches disk, which is what makes the
        // staleness permanent without the re-index below.
        let record = store.get_file("note.md").unwrap().unwrap();
        assert_eq!(
            record.content_hash,
            compute_file_hash(&root.join("note.md")).unwrap(),
            "update_note stores the hash of what it wrote"
        );

        reindex_written_file(
            "note.md",
            &store,
            &mut embedder,
            root,
            IndexSettings::from_config(&config),
        )
        .unwrap();

        let file_id = store.get_file("note.md").unwrap().unwrap().id;
        let text: String = store
            .get_chunks_by_file(file_id)
            .unwrap()
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("the new line"),
            "the store must hold the appended text, got: {text:?}"
        );
    }

    #[test]
    fn test_exclude_purges_previously_indexed_files() {
        // Adding an exclude pattern must remove what is already in the store, not
        // just stop future ingestion. `diff_vault` treats "in the store, absent
        // from the walk" as deleted, which makes this work — this pins it.
        use crate::llm::MockLlm;

        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(
            root,
            "lore/archdragon.md",
            "# Archdragon\nSee [[lore-index]].",
        );
        write_file(
            root,
            "lore/lore-index.md",
            "# Lore index\n\n### Archdragon\nSee [[archdragon]].",
        );

        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        let mut config = Config::default();

        let result = run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        assert_eq!(result.new_files, 2);

        let indexed = store
            .get_file("lore/lore-index.md")
            .unwrap()
            .expect("index file should be in the store");
        let file_id = indexed.id;
        // The keyword index is external content over `chunks` (#37), so a
        // file's entries are counted through the join and not through a column
        // of the index.
        let fts_rows = |id: i64| -> i64 {
            store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM chunks_fts
                     JOIN chunks c ON c.id = chunks_fts.rowid
                     WHERE chunks_fts MATCH ?1 AND c.file_id = ?2",
                    rusqlite::params![r#""the" OR "a" OR "of" OR "lore""#, id],
                    |row| row.get(0),
                )
                .unwrap()
        };

        assert!(!store.get_chunks_by_file(file_id).unwrap().is_empty());
        assert!(!store.get_vector_ids_for_file(file_id).unwrap().is_empty());
        assert!(fts_rows(file_id) > 0);
        assert!(
            !store.get_outgoing(file_id, None).unwrap().is_empty(),
            "index file should have contributed graph edges"
        );

        // Exclude it and re-index.
        config.exclude.push("*-index.md".to_string());
        let result = run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(result.deleted_files, 1);
        assert!(store.get_file("lore/lore-index.md").unwrap().is_none());
        assert!(
            store.get_file("lore/archdragon.md").unwrap().is_some(),
            "excluding the index file must not disturb the canonical note"
        );
        assert!(store.get_chunks_by_file(file_id).unwrap().is_empty());
        assert!(store.get_vector_ids_for_file(file_id).unwrap().is_empty());
        assert_eq!(fts_rows(file_id), 0);
        assert!(store.get_outgoing(file_id, None).unwrap().is_empty());
    }

    #[test]
    fn test_removed_file_leaves_no_unresolved_links() {
        // `unresolved_links` is keyed by path rather than file id, so it is not
        // reached by the usual cascade — a removed file used to keep reporting
        // broken links forever.
        use crate::llm::MockLlm;

        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "keeper.md", "# Keeper\nPoints at [[nowhere]].");
        write_file(
            root,
            "lore-index.md",
            "# Index\nPoints at [[also-nowhere]].",
        );

        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        let mut config = Config::default();

        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        let sources: Vec<String> = store
            .get_unresolved_links()
            .unwrap()
            .into_iter()
            .map(|(source, _)| source)
            .collect();
        assert_eq!(sources.len(), 2, "got {sources:?}");

        config.exclude.push("*-index.md".to_string());
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let sources: Vec<String> = store
            .get_unresolved_links()
            .unwrap()
            .into_iter()
            .map(|(source, _)| source)
            .collect();
        assert_eq!(
            sources,
            vec!["keeper.md".to_string()],
            "the excluded file's unresolved links should be gone"
        );
    }

    /// Every edge as `"source#seq -> target#seq (type)"`, so two stores can be
    /// compared without depending on file ids matching. Chunk seqs included:
    /// they are half the table since #28.
    fn edge_snapshot(store: &Store) -> Vec<String> {
        let paths: HashMap<i64, String> = store
            .get_all_files()
            .unwrap()
            .into_iter()
            .map(|f| (f.id, f.path))
            .collect();
        let name = |id: i64, seq: i64| {
            let path = paths.get(&id).cloned().unwrap_or_else(|| format!("?{id}"));
            if seq == DOC_LEVEL {
                path
            } else {
                format!("{path}#{seq}")
            }
        };
        let mut stmt = store
            .conn()
            .prepare(
                "SELECT from_file, from_chunk_seq, to_file, to_chunk_seq, edge_type FROM edges",
            )
            .unwrap();
        let mut edges: Vec<String> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .unwrap()
            .map(|r| {
                let (from, from_seq, to, to_seq, kind) = r.unwrap();
                format!("{} -> {} ({kind})", name(from, from_seq), name(to, to_seq))
            })
            .collect();
        edges.sort();
        edges
    }

    #[test]
    fn an_incremental_edit_and_a_full_index_agree_on_the_edges_table() {
        // The invariant issue #27 is really about. Editing `hub.md` — which has
        // no outgoing links at all — used to wipe both backlinks into it: the
        // `files` row was deleted and re-inserted, and `edges` CASCADEs off it.
        // Nothing restored them, because the files that own those edges were not
        // re-indexed.
        use crate::llm::MockLlm;

        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[hub]].");
        write_file(root, "c.md", "# C\nAlso [[hub]].");
        write_file(root, "hub.md", "# Hub\nNo links out.");

        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        let config = Config::default();

        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        let hub_id = store.get_file("hub.md").unwrap().unwrap().id;
        assert_eq!(
            store.get_incoming(hub_id, Some("wikilink")).unwrap().len(),
            2,
            "both backlinks should exist after a full index"
        );

        // Edit the hub. Its own content still has no wikilinks, so a correct
        // incremental index changes nothing about the graph.
        write_file(
            root,
            "hub.md",
            "# Hub\nStill no links out, one word changed.",
        );
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let hub = store.get_file("hub.md").unwrap().unwrap();
        assert_eq!(
            hub.id, hub_id,
            "re-indexing must keep the file's id — every backlink is keyed on it"
        );
        assert_eq!(
            store.get_incoming(hub.id, Some("wikilink")).unwrap().len(),
            2,
            "editing a file must not destroy the links other files point at it"
        );
        assert!(
            !store.get_chunks_by_file(hub.id).unwrap().is_empty(),
            "the edited file should still be indexed"
        );

        // The real acceptance criterion: incremental and from-scratch agree.
        let fresh = Store::open_memory().unwrap();
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &fresh,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        assert_eq!(edge_snapshot(&store), edge_snapshot(&fresh));
    }

    #[test]
    fn an_incremental_index_drops_wikilinks_the_author_removed() {
        // The other half of #27. `insert_edge` is INSERT OR IGNORE and the
        // incremental path skips `clear_edges`, so without an explicit delete
        // the graph could only ever grow: a link you deleted stayed forever.
        use crate::llm::MockLlm;

        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[hub]] and [[c]].");
        write_file(root, "c.md", "# C");
        write_file(root, "hub.md", "# Hub");

        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        let config = Config::default();

        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        let a_id = store.get_file("a.md").unwrap().unwrap().id;
        assert_eq!(store.get_outgoing(a_id, Some("wikilink")).unwrap().len(), 2);

        write_file(root, "a.md", "# A\nSee [[c]]. The hub link is gone.");
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let out = store.get_outgoing(a_id, Some("wikilink")).unwrap();
        assert_eq!(out.len(), 1, "the removed wikilink should be gone: {out:?}");

        let fresh = Store::open_memory().unwrap();
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &fresh,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        assert_eq!(edge_snapshot(&store), edge_snapshot(&fresh));
    }

    // ── Fingerprints (issue #31) ──────────────────────────────────────────
    //
    // A changed constant and a poked `meta` row are the same thing to the code
    // under test: a stored fingerprint that disagrees with the running one. The
    // tests poke, because a `const` cannot be changed at runtime and rebuilding
    // the crate per assertion is not a test.
    //
    // Every assertion here keys on `path`, never on `file_id`. A rebuild hands
    // out rowids in vault-walk order, so two rebuilds at identical settings
    // disagree on any digest keyed by id — which read as corruption once
    // already (issue #20).

    /// A three-file vault with links, indexed once, plus a recording embedder
    /// so a later run can prove nothing reached the model.
    fn fingerprint_fixture() -> (TempDir, Store, RecordingEmbed, Config) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[hub]].");
        write_file(root, "c.md", "# C\nAlso [[hub]].");
        write_file(root, "hub.md", "# Hub\nNo links out.");

        let store = Store::open_memory().unwrap();
        let mut embedder = RecordingEmbed::new(256);
        let config = Config::default();
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        embedder.seen.clear();
        (tmp, store, embedder, config)
    }

    /// Every chunk as `"path#seq: text"`, sorted. Keyed on path, not id.
    fn chunk_snapshot(store: &Store) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for file in store.get_all_files().unwrap() {
            for chunk in store.get_chunks_by_file(file.id).unwrap() {
                out.push(format!("{}#{}: {}", file.path, chunk.seq, chunk.text));
            }
        }
        out.sort();
        out
    }

    /// How many rows the keyword index actually holds.
    ///
    /// A MATCH and not `count(*)`: since #37 `chunks_fts` is external content,
    /// so `count(*)` counts the rows of the *content* table and reads the same
    /// whether the index is populated or empty. Every fixture file contains
    /// `#`, which the tokenizer drops, so the term below is one every chunk
    /// carries.
    fn fts_row_count(store: &Store) -> i64 {
        store
            .conn()
            .query_row(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?1",
                [r#""a" OR "c" OR "hub""#],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[test]
    fn an_unchanged_configuration_rebuilds_nothing() {
        // The half of the acceptance criteria that actually matters. A
        // fingerprint that fires on every startup is as useless as none, just
        // slower — so this asserts on the *absence* of work, from three angles:
        // no file re-indexed, no string handed to the model, and a keyword index
        // emptied by hand still empty afterwards.
        let (tmp, store, mut embedder, config) = fingerprint_fixture();
        let before = chunk_snapshot(&store);

        store.conn().execute("DELETE FROM chunks_fts", []).unwrap();
        let result = run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(result.new_files, 0);
        assert_eq!(result.updated_files, 0);
        assert!(
            embedder.seen.is_empty(),
            "nothing should have reached the embedder: {:?}",
            embedder.seen
        );
        assert_eq!(
            fts_row_count(&store),
            0,
            "no fingerprint changed, so nothing should have rebuilt the keyword index"
        );
        assert_eq!(before, chunk_snapshot(&store));
    }

    #[test]
    fn an_index_records_its_fingerprints() {
        let (_tmp, store, _embedder, _config) = fingerprint_fixture();
        for key in [
            crate::fingerprint::PARSER,
            crate::fingerprint::CHUNKER,
            crate::fingerprint::LINK,
            crate::fingerprint::FTS,
            crate::fingerprint::EMBEDDING,
        ] {
            assert!(
                store.get_meta(key.name).unwrap().is_some(),
                "{} should be recorded after an index",
                key.name
            );
        }
        assert!(
            store
                .get_meta(crate::fingerprint::RERANKER.name)
                .unwrap()
                .is_none(),
            "the index path scores nothing, so it must not claim a reranker is current"
        );
    }

    #[test]
    fn a_changed_chunker_constant_reindexes_every_file() {
        let (tmp, store, mut embedder, config) = fingerprint_fixture();
        store
            .set_meta(crate::fingerprint::CHUNKER.name, "built-by-other-constants")
            .unwrap();

        let result = run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(
            result.new_files, 3,
            "a rebuild treats every file as new, and no file changed on disk"
        );
        assert!(
            !embedder.seen.is_empty(),
            "a rechunk must re-embed: the vectors describe text that no longer exists"
        );
        assert_ne!(
            store.get_meta(crate::fingerprint::CHUNKER.name).unwrap(),
            Some("built-by-other-constants".to_string()),
            "the fingerprint should be current once the work is done"
        );
    }

    #[test]
    fn a_changed_fts_schema_rebuilds_the_keyword_index_without_re_embedding() {
        // The action `fts_fingerprint` declares, and the reason it is cheap:
        // `chunks.text` holds the whole chunk, so the keyword index is derivable
        // from what is already stored. Emptying it by hand first is what proves
        // the rebuild ran rather than that it was never needed.
        let (tmp, store, mut embedder, config) = fingerprint_fixture();
        let expected = chunk_snapshot(&store).len() as i64;
        store.conn().execute("DELETE FROM chunks_fts", []).unwrap();
        store
            .set_meta(crate::fingerprint::FTS.name, "built-by-another-schema")
            .unwrap();

        let result = run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(
            fts_row_count(&store),
            expected,
            "every chunk should be back"
        );
        assert_eq!(result.new_files, 0);
        assert_eq!(result.updated_files, 0);
        assert!(
            embedder.seen.is_empty(),
            "an FTS schema change must not re-embed: {:?}",
            embedder.seen
        );
    }

    #[test]
    fn a_changed_link_resolver_rebuilds_edges_without_re_embedding() {
        // A resolver change is not confined to the files that changed on disk:
        // the same `[[Note]]` may resolve elsewhere in every file that writes
        // it, and no content hash can see that because no content changed.
        let (tmp, store, mut embedder, config) = fingerprint_fixture();
        let expected = edge_snapshot(&store);
        assert!(!expected.is_empty(), "fixture should have edges to lose");

        store.clear_edges().unwrap();
        store
            .set_meta(crate::fingerprint::LINK.name, "built-by-another-resolver")
            .unwrap();

        let result = run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(edge_snapshot(&store), expected, "every edge should be back");
        assert_eq!(result.new_files, 0);
        assert_eq!(result.updated_files, 0);
        assert!(
            embedder.seen.is_empty(),
            "a resolver change must not re-embed: {:?}",
            embedder.seen
        );
    }

    #[test]
    fn a_changed_reranker_never_reaches_the_index_path() {
        // The one key whose action is not a rebuild. Getting it wrong buys a
        // needless full re-embed for a change that touched no stored byte.
        let (tmp, store, mut embedder, config) = fingerprint_fixture();
        store
            .set_meta(crate::fingerprint::RERANKER.name, "some-other-reranker")
            .unwrap();

        let result = run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(result.new_files, 0);
        assert_eq!(result.updated_files, 0);
        assert!(embedder.seen.is_empty(), "{:?}", embedder.seen);
        assert_eq!(
            store
                .get_meta(crate::fingerprint::RERANKER.name)
                .unwrap()
                .as_deref(),
            Some("some-other-reranker"),
            "the index path loads no reranker, so it must not overwrite what one recorded"
        );
    }

    #[test]
    fn a_pre_fingerprint_store_is_adopted_rather_than_rebuilt() {
        // Every store written before #31 has no fingerprints. Forcing them all
        // through a full reindex to find out whether they were stale is the same
        // uselessness as rebuilding on every startup, just once — and there is
        // no evidence to act on. Fingerprints protect what happens after they
        // are first recorded.
        let (tmp, store, mut embedder, config) = fingerprint_fixture();
        for key in [
            crate::fingerprint::PARSER,
            crate::fingerprint::CHUNKER,
            crate::fingerprint::LINK,
            crate::fingerprint::FTS,
            crate::fingerprint::EMBEDDING,
        ] {
            store
                .conn()
                .execute("DELETE FROM meta WHERE key = ?1", [key.name])
                .unwrap();
        }

        let result = run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(result.new_files, 0);
        assert_eq!(result.updated_files, 0);
        assert!(embedder.seen.is_empty(), "{:?}", embedder.seen);
        assert!(
            store
                .get_meta(crate::fingerprint::CHUNKER.name)
                .unwrap()
                .is_some(),
            "and they are recorded on the way out, so the next change is caught"
        );
    }

    #[test]
    fn a_dimension_change_still_discards_the_index() {
        // `embedding_fingerprint` subsumes `ensure_embedding_dim`, and this is
        // the half that must not regress: a width change is a shape error, not
        // merely a stale index, and it is caught before anything is written.
        let (tmp, store, _embedder, config) = fingerprint_fixture();
        assert!(!chunk_snapshot(&store).is_empty());

        let mut wider = RecordingEmbed::new(512);
        run_index_shared(
            tmp.path(),
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut wider,
            false,
            None,
        )
        .unwrap();

        assert_eq!(store.vec_table_dim().unwrap(), Some(512));
        assert!(
            !wider.seen.is_empty(),
            "a width change must re-embed everything"
        );
        assert_eq!(chunk_snapshot(&store).len(), 3);
    }

    #[test]
    fn test_walk_respects_gitignore() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Initialize a git repo so the ignore crate respects .gitignore.
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(root)
            .output()
            .expect("git init failed");

        write_file(root, ".gitignore", "drafts/\n");
        write_file(root, "note.md", "# Note");
        write_file(root, "drafts/note.md", "# Draft");

        let files = walk_vault(root, &[], true).unwrap();
        assert_eq!(
            files.len(),
            1,
            "expected 1 file (drafts/ gitignored), got {:?}",
            files
        );
        assert!(files[0].ends_with("note.md"));
    }

    #[test]
    fn test_walk_gitignore_toggle() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Git repo so the ignore crate honors .gitignore (require_git default).
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(root)
            .output()
            .expect("git init failed");

        write_file(root, ".gitignore", "drafts/\n");
        write_file(root, "note.md", "# Note");
        write_file(root, "drafts/draft.md", "# Draft");

        // respect_gitignore = true: the gitignored dir is skipped.
        let respected = walk_vault(root, &[], true).unwrap();
        assert_eq!(
            respected.len(),
            1,
            "gitignored dir should be skipped, got {:?}",
            respected
        );
        assert!(respected[0].ends_with("note.md"));

        // respect_gitignore = false: the gitignored file is indexed too.
        let ignored = walk_vault(root, &[], false).unwrap();
        let names: Vec<String> = ignored
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            ignored.len(),
            2,
            "expected gitignored file to be included, got {:?}",
            ignored
        );
        assert!(names.contains(&"note.md".to_string()));
        assert!(names.contains(&"draft.md".to_string()));
    }

    #[test]
    fn test_detect_new_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A");
        write_file(root, "b.md", "# B");

        let store = Store::open_memory().unwrap();
        let files = walk_vault(root, &[], true).unwrap();
        let (new, changed, deleted) = diff_vault(&files, root, &store).unwrap();

        assert_eq!(new.len(), 2, "all files should be new");
        assert_eq!(changed.len(), 0);
        assert_eq!(deleted.len(), 0);
    }

    #[test]
    fn test_detect_changed_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "note.md", "# Original content");

        let store = Store::open_memory().unwrap();
        // Insert file with an old/different hash.
        store
            .insert_file(
                "note.md",
                "old_hash_that_wont_match",
                100,
                &generate_docid("note.md"),
                None,
                None,
            )
            .unwrap();

        let files = walk_vault(root, &[], true).unwrap();
        let (new, changed, deleted) = diff_vault(&files, root, &store).unwrap();

        assert_eq!(new.len(), 0);
        assert_eq!(
            changed.len(),
            1,
            "file with different hash should be changed"
        );
        assert_eq!(deleted.len(), 0);
    }

    #[test]
    fn test_detect_deleted_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "surviving.md", "# I exist");

        let store = Store::open_memory().unwrap();
        // Insert a file that no longer exists on disk.
        store
            .insert_file(
                "surviving.md",
                &compute_file_hash(&root.join("surviving.md")).unwrap(),
                100,
                &generate_docid("surviving.md"),
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "deleted.md",
                "some_hash",
                100,
                &generate_docid("deleted.md"),
                None,
                None,
            )
            .unwrap();

        let files = walk_vault(root, &[], true).unwrap();
        let (new, changed, deleted) = diff_vault(&files, root, &store).unwrap();

        assert_eq!(new.len(), 0);
        assert_eq!(changed.len(), 0);
        assert_eq!(
            deleted.len(),
            1,
            "missing file should be detected as deleted"
        );
        assert_eq!(deleted[0].path, "deleted.md");
    }

    #[test]
    fn test_compute_file_hash_deterministic() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.md");
        std::fs::write(&path, "hello world").unwrap();

        let h1 = compute_file_hash(&path).unwrap();
        let h2 = compute_file_hash(&path).unwrap();
        assert_eq!(h1, h2, "same content should produce same hash");

        // Verify it's the known SHA-256 of "hello world".
        assert_eq!(
            h1,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_edge_building_during_index() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[b]] for details.");
        write_file(root, "b.md", "# B\nLinks to [[a]].");
        write_file(root, "c.md", "# C\nNo links here.");

        let store = Store::open_memory().unwrap();
        let f_a = store
            .insert_file("a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let f_b = store
            .insert_file("b.md", "h2", 100, "bbb222", None, None)
            .unwrap();
        let _f_c = store
            .insert_file("c.md", "h3", 100, "ccc333", None, None)
            .unwrap();

        let content_a = std::fs::read_to_string(root.join("a.md")).unwrap();
        let content_b = std::fs::read_to_string(root.join("b.md")).unwrap();

        build_edges_for_file(&store, f_a, &content_a).unwrap();
        build_edges_for_file(&store, f_b, &content_b).unwrap();

        let a_out = store.get_outgoing(f_a, Some("wikilink")).unwrap();
        assert_eq!(a_out.len(), 1);
        assert_eq!(a_out[0].0, f_b);

        let b_out = store.get_outgoing(f_b, Some("wikilink")).unwrap();
        assert_eq!(b_out.len(), 1);
        assert_eq!(b_out[0].0, f_a);
    }

    #[test]
    fn test_wikilink_edges_are_directional_not_bidirectional() {
        // Regression test for the "edges stored bidirectionally" bug.
        // A has [[B]]; B has NO wikilink to A. Expected: A→B edge exists,
        // B→A edge does NOT exist. Pre-fix, the indexer fabricated the
        // reverse edge regardless of B's actual content.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[b]] for details.");
        write_file(root, "b.md", "# B\nNo backlink here.");

        let store = Store::open_memory().unwrap();
        let f_a = store
            .insert_file("a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let f_b = store
            .insert_file("b.md", "h2", 100, "bbb222", None, None)
            .unwrap();

        let content_a = std::fs::read_to_string(root.join("a.md")).unwrap();
        let content_b = std::fs::read_to_string(root.join("b.md")).unwrap();

        build_edges_for_file(&store, f_a, &content_a).unwrap();
        build_edges_for_file(&store, f_b, &content_b).unwrap();

        // A → B exists (A's content has [[b]])
        let a_out = store.get_outgoing(f_a, Some("wikilink")).unwrap();
        assert_eq!(a_out.len(), 1, "A should have 1 outgoing wikilink");
        assert_eq!(a_out[0].0, f_b);

        // B → A does NOT exist (B's content has no wikilink to A)
        let b_out = store.get_outgoing(f_b, Some("wikilink")).unwrap();
        assert_eq!(
            b_out.len(),
            0,
            "B should have 0 outgoing wikilinks (B has no [[a]] in content)"
        );

        // But B should have 1 INCOMING from A
        let b_in = store.get_incoming(f_b, Some("wikilink")).unwrap();
        assert_eq!(b_in.len(), 1, "B should have 1 incoming wikilink (from A)");
        assert_eq!(b_in[0].0, f_a);
    }

    #[test]
    fn test_unresolved_wikilinks_are_recorded() {
        // Regression test for the "unresolved_links table never populated" bug.
        // A has [[b]] (resolves) and [[nonexistent-target]] (doesn't).
        // Expected: the unresolved target is recorded in unresolved_links.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(
            root,
            "a.md",
            "# A\nSee [[b]] for details.\nAlso [[nonexistent-target]] for nothing.",
        );
        write_file(root, "b.md", "# B");

        let store = Store::open_memory().unwrap();
        let f_a = store
            .insert_file("a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let _f_b = store
            .insert_file("b.md", "h2", 100, "bbb222", None, None)
            .unwrap();

        let content_a = std::fs::read_to_string(root.join("a.md")).unwrap();
        build_edges_for_file(&store, f_a, &content_a).unwrap();

        // Unresolved target should be recorded
        let unresolved = store.get_unresolved_links().unwrap();
        assert_eq!(
            unresolved.len(),
            1,
            "Should have 1 unresolved wikilink (nonexistent-target)"
        );
        assert_eq!(unresolved[0].0, "a.md");
        assert_eq!(unresolved[0].1, "nonexistent-target");
    }

    #[test]
    fn test_unresolved_links_cleared_on_re_index() {
        // When build_edges_for_file is called again on the same source
        // (incremental update / re-index), stale unresolved entries for
        // that source should be cleared before re-recording. Otherwise
        // entries accumulate even after the user fixes broken links.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "a.md", "# A\nSee [[broken-target]] for nothing.");

        let store = Store::open_memory().unwrap();
        let f_a = store
            .insert_file("a.md", "h1", 100, "aaa111", None, None)
            .unwrap();

        let content_a_v1 = std::fs::read_to_string(root.join("a.md")).unwrap();
        build_edges_for_file(&store, f_a, &content_a_v1).unwrap();
        assert_eq!(store.get_unresolved_links().unwrap().len(), 1);

        // Now A is edited to remove the broken wikilink entirely.
        let content_a_v2 = "# A\nNo wikilinks here now.";
        build_edges_for_file(&store, f_a, content_a_v2).unwrap();
        let unresolved = store.get_unresolved_links().unwrap();
        assert_eq!(
            unresolved.len(),
            0,
            "Stale unresolved entry should be cleared after re-index"
        );
    }

    /// Wraps an embedder and keeps every string it was asked to embed, so a
    /// test can compare what reached the model against what reached the store.
    struct RecordingEmbed {
        inner: crate::llm::MockLlm,
        seen: Vec<String>,
        /// The title field each body was paired with (issue #36).
        titles: Vec<String>,
    }

    impl RecordingEmbed {
        fn new(dim: usize) -> Self {
            Self {
                inner: crate::llm::MockLlm::new(dim),
                seen: Vec::new(),
                titles: Vec::new(),
            }
        }
    }

    impl EmbedModel for RecordingEmbed {
        fn embed_batch(&mut self, docs: &[crate::llm::EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
            self.seen.extend(docs.iter().map(|d| d.text.to_string()));
            self.titles.extend(docs.iter().map(|d| d.title.to_string()));
            self.inner.embed_batch(docs)
        }
        fn token_count(&self, text: &str) -> usize {
            self.inner.token_count(text)
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn max_context(&self) -> usize {
            self.inner.max_context()
        }
        fn fingerprint(&self) -> String {
            self.inner.fingerprint()
        }
    }

    /// A `MockLlm`-shaped embedder that fails any batch containing a poisoned
    /// marker, so a test can simulate an API embedder that dies mid-index on
    /// one file and recovers on a later run (issue #84).
    struct FlakyEmbed {
        dim: usize,
        fail_marker: Option<String>,
    }

    impl EmbedModel for FlakyEmbed {
        fn embed_batch(&mut self, docs: &[crate::llm::EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
            if let Some(m) = &self.fail_marker
                && docs.iter().any(|d| d.text.contains(m.as_str()))
            {
                anyhow::bail!("simulated API failure");
            }
            Ok(docs.iter().map(|_| vec![0.1f32; self.dim]).collect())
        }
        fn token_count(&self, text: &str) -> usize {
            text.len() / 4
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn max_context(&self) -> usize {
            100_000
        }
        fn fingerprint(&self) -> String {
            "flaky/v1".into()
        }
    }

    #[test]
    fn a_failing_file_is_skipped_and_re_attempted_next_run() {
        // Two files: good.md embeds cleanly, bad.md carries a marker that a
        // flaky embedder trips on. Run 1 must not error out — it skips bad.md,
        // indexes good.md, and leaves fingerprints unrecorded so a later run
        // knows the index is not clean. Run 2, with the embedder fixed,
        // re-attempts bad.md because it was never written to the store.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "good.md", "# Good\nNothing wrong with this one.");
        write_file(
            root,
            "bad.md",
            "# Bad\nThis note contains POISON in its body.",
        );

        let store = Store::open_memory().unwrap();
        let config = Config::default();

        let mut flaky = FlakyEmbed {
            dim: 256,
            fail_marker: Some("POISON".to_string()),
        };
        let result = run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut flaky,
            false,
            None,
        );
        if let Err(e) = &result {
            panic!("a failing file must be skipped, not propagated as an error: {e:#}");
        }

        assert!(
            store.get_file("good.md").unwrap().is_some(),
            "good.md should be indexed despite bad.md's failure"
        );
        assert!(
            store.get_file("bad.md").unwrap().is_none(),
            "bad.md must be left untouched in the store after its embed failed"
        );
        assert!(
            store.get_meta("embedding_fingerprint").unwrap().is_none(),
            "fingerprints must not be recorded on a pass that skipped a file"
        );

        // Run 2: the embedder is fixed, so bad.md is re-attempted and indexes.
        let mut fixed = FlakyEmbed {
            dim: 256,
            fail_marker: None,
        };
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut fixed,
            false,
            None,
        )
        .unwrap();

        assert!(store.get_file("good.md").unwrap().is_some());
        assert!(
            store.get_file("bad.md").unwrap().is_some(),
            "bad.md should be re-attempted and indexed once the failure clears"
        );
        assert!(
            store.get_meta("embedding_fingerprint").unwrap().is_some(),
            "fingerprints should be recorded once a pass has zero skips"
        );
    }

    /// The other half of issue #84's contract: a failure *inside* one of
    /// `index_file`'s own store writes — after the embed already succeeded —
    /// must not leave that file's partial rows sitting in the loop's ambient
    /// transaction to be persisted by the final `commit()`.
    ///
    /// The module has no injectable `Store` (it is a concrete `rusqlite`
    /// wrapper), so nothing lets a test make `insert_file` or
    /// `insert_chunk_with_vector` fail from the outside. A `BEFORE INSERT`
    /// trigger that `RAISE(ABORT, ...)`s on a poisoned marker gets the same
    /// effect with no production code touched: it fails the *store write* for
    /// one chunk, after `index_file` has already embedded successfully and
    /// already inserted the file's `files` row (step 6 runs before the chunk
    /// loop), which is exactly the ordering the review finding described.
    #[test]
    fn a_failed_store_write_leaves_no_partial_file_row() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "good.md", "# Good\nNothing wrong with this one.");
        write_file(
            root,
            "poison.md",
            "# Poison\nThis note contains STORE_WRITE_POISON in its body.",
        );

        let store = Store::open_memory().unwrap();
        let config = Config::default();
        let mut embedder = RecordingEmbed::new(256);

        // A single index_file call embeds first (unaffected), then writes the
        // file row, then the chunk row — the chunk insert is where this fails.
        store
            .conn()
            .execute_batch(
                "CREATE TRIGGER poison_chunk_write
                 BEFORE INSERT ON chunks
                 FOR EACH ROW
                 WHEN NEW.text LIKE '%STORE_WRITE_POISON%'
                 BEGIN
                     SELECT RAISE(ABORT, 'simulated store write failure');
                 END;",
            )
            .unwrap();

        let result = run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        );
        if let Err(e) = &result {
            panic!("a failed store write must be skipped, not propagated as an error: {e:#}");
        }

        assert!(
            store.get_file("good.md").unwrap().is_some(),
            "good.md should be indexed despite poison.md's failure"
        );
        assert!(
            store.get_file("poison.md").unwrap().is_none(),
            "poison.md's files row must be rolled back with its failed chunk write, \
             not committed as a partial (chunk-less) file — the exact leak the \
             per-file SAVEPOINT closes"
        );
        assert!(
            store.get_meta("embedding_fingerprint").unwrap().is_none(),
            "fingerprints must not be recorded on a pass that skipped a file"
        );

        // Run 2: the trigger is gone, so poison.md is re-attempted and indexes —
        // proving diff_vault still sees it as new, because it holds no row.
        store
            .conn()
            .execute_batch("DROP TRIGGER poison_chunk_write;")
            .unwrap();
        run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert!(store.get_file("good.md").unwrap().is_some());
        assert!(
            store.get_file("poison.md").unwrap().is_some(),
            "poison.md should be re-attempted and indexed once the write succeeds"
        );
        assert!(
            store.get_meta("embedding_fingerprint").unwrap().is_some(),
            "fingerprints should be recorded once a pass has zero skips"
        );
    }

    fn prefixed_config() -> Config {
        Config {
            embedding_prefix: crate::prefix::PrefixConfig::full(),
            ..Config::default()
        }
    }

    fn title_config(title: crate::llm::DocumentTitle) -> Config {
        let mut config = Config::default();
        config.embedding_prompt.document_title = title;
        config
    }

    /// Index one file whose body never names its own subject — the archdragon
    /// case from issue #2 — and report what the embedder saw.
    fn index_prefixed_vault(config: &Config) -> (Store, RecordingEmbed) {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "lore/bestiary/archdragon.md",
            "---\nname: Archdragon\naliases:\n  - Elder Wyrm\ntags:\n  - apex\n---\n\n\
             ## Definition\n\nRank SS, levels 150-511. The apex of the ladder, and the \
             only rank whose stat cap needs an eleven-bit field to hold it, and the ladder \
             has no rung above it.\n\n\
             ## Abilities\n\nFlight and breath. Both are innate rather than learned, and \
             neither costs the mana a spell of the same reach would, which is most of \
             what makes the rank what it is.\n\n\
             ### Combat\n\nOpens at range. Closes only against something that survived the \
             opening, which in the whole of the recorded history of the continent is very \
             little indeed.\n",
        );

        let store = Store::open_memory().unwrap();
        let mut embedder = RecordingEmbed::new(256);
        run_index_shared(
            tmp.path(),
            config,
            crate::indexer::IndexSettings::from_config(config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (store, embedder)
    }

    #[test]
    fn embedded_text_carries_document_identity() {
        let seen = index_prefixed_vault(&prefixed_config()).1.seen;
        assert_eq!(seen.len(), 3, "one embed call per chunk: {seen:#?}");

        // Every chunk, including the two whose bodies never say "Archdragon".
        assert!(seen.iter().all(|t| t.contains("Archdragon")), "{seen:#?}");
        assert!(
            seen.iter()
                .all(|t| t.contains("lore/bestiary/archdragon.md"))
        );
        assert!(seen.iter().all(|t| t.contains("aliases: Elder Wyrm")));
        assert!(seen.iter().all(|t| t.contains("tags: apex")));

        // `### Combat` is a sibling chunk of `## Abilities`, so its own text has
        // lost the parent heading. The prefix is the only thing that carries it.
        let combat = seen.iter().find(|t| t.contains("Opens at range")).unwrap();
        assert!(combat.contains("Abilities > Combat"), "{combat}");
    }

    /// The other half of the contract: the prefix reaches the embedder and
    /// nothing else. If it leaked, `## Definition` would be displayed with a
    /// frontmatter preamble and `apex` would be a keyword hit on a chunk that
    /// never mentions it.
    #[test]
    fn the_prefix_does_not_leak_into_storage_or_fts() {
        let (store, _embedder) = index_prefixed_vault(&prefixed_config());
        let file = store
            .get_file("lore/bestiary/archdragon.md")
            .unwrap()
            .unwrap();

        let definition = store.get_chunk_by_seq(file.id, 0).unwrap().unwrap();
        assert_eq!(definition.heading, "## Definition");
        assert!(definition.snippet.starts_with("## Definition"));
        assert!(!definition.snippet.contains("Archdragon"));
        assert!(!definition.snippet.contains("aliases"));

        // FTS indexes the chunk's own text, so prefix-only terms — which are
        // never part of it — must not be searchable.
        let by_alias = store
            .best_matching_chunk_seq(file.id, &["Wyrm".to_string()])
            .unwrap();
        assert_eq!(by_alias, None, "alias became a keyword hit");

        // A term the chunk really does contain still matches, so this is not
        // just an empty index.
        let by_body = store
            .best_matching_chunk_seq(file.id, &["breath".to_string()])
            .unwrap();
        assert_eq!(by_body, Some(1));
    }

    /// Issue #11: FTS used to be given `chunk.snippet`, the leading 200
    /// characters, so a term appearing later in a chunk was unreachable by
    /// keyword search — 70% of the eval corpus, in practice. The display field
    /// is still truncated; only what FTS is given changed.
    #[test]
    fn keyword_search_reaches_past_the_snippet_boundary() {
        let tmp = TempDir::new().unwrap();
        let filler = "The coast road runs north through salt marsh and low dune. ".repeat(8);
        write_file(
            tmp.path(),
            "places/coast.md",
            &format!("## The Coast Road\n\n{filler}\n\nIt ends at Saltmere.\n"),
        );

        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        run_index_shared(
            tmp.path(),
            &Config::default(),
            crate::indexer::IndexSettings::from_config(&Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let file = store.get_file("places/coast.md").unwrap().unwrap();
        let chunk = store.get_chunk_by_seq(file.id, 0).unwrap().unwrap();

        // The premise: "Saltmere" is well past where the snippet stops.
        assert!(
            chunk.snippet.len() <= 203,
            "snippet grew: {}",
            chunk.snippet
        );
        assert!(
            !chunk.snippet.contains("Saltmere"),
            "fixture is too short to test anything"
        );

        assert_eq!(
            store
                .best_matching_chunk_seq(file.id, &["Saltmere".to_string()])
                .unwrap(),
            Some(0),
            "a term past character 200 must still be searchable"
        );
    }

    #[test]
    fn disabling_the_prefix_embeds_exactly_what_is_stored() {
        // The shipped default, not a special case.
        let (store, embedder) = index_prefixed_vault(&Config::default());
        let seen = embedder.seen;
        assert!(!Config::default().embedding_prefix.enabled);

        assert!(!seen.iter().any(|t| t.contains("aliases:")), "{seen:#?}");
        let file = store
            .get_file("lore/bestiary/archdragon.md")
            .unwrap()
            .unwrap();
        let definition = store.get_chunk_by_seq(file.id, 0).unwrap().unwrap();
        assert!(seen.iter().any(|t| t.starts_with(&definition.snippet)));
    }

    // ── The title field (#36) ────────────────────────────────────

    /// The breadcrumb reaches the embedder as a *field*, chunk by chunk, with
    /// the body untouched. The setting is off by default (#38), so the arm has
    /// to be named.
    #[test]
    fn the_breadcrumb_reaches_the_embedder_as_the_title_field() {
        let (_store, embedder) =
            index_prefixed_vault(&title_config(crate::llm::DocumentTitle::Breadcrumb));
        let (titles, seen) = (embedder.titles, embedder.seen);
        assert_eq!(titles.len(), 3, "one title per chunk: {titles:#?}");

        // `### Combat` is a sibling chunk of `## Abilities`, so its own text has
        // lost the parent heading. Here the title field carries the whole
        // lineage, note name included.
        let combat = titles
            .iter()
            .zip(&seen)
            .find(|(_, body)| body.contains("Opens at range"))
            .map(|(title, _)| title)
            .unwrap();
        assert_eq!(combat, "lore/bestiary/archdragon.md > Abilities > Combat");

        // Distinct per section, which is the property #2's per-file prefix
        // lacked.
        let mut sorted = titles.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "{titles:#?}");

        // And the body is the raw chunk: no prefix, no breadcrumb.
        assert!(
            seen.iter().all(|t| !t.contains(" > ")),
            "the title leaked into the body: {seen:#?}"
        );
    }

    /// `none` sends no title, which is what every store built before this key
    /// existed holds. It is the control every measurement in `eval/probes.md` is
    /// taken against, so it has to stay reachable.
    #[test]
    fn the_none_setting_sends_no_title_at_all() {
        let (_store, embedder) =
            index_prefixed_vault(&title_config(crate::llm::DocumentTitle::None));
        assert!(
            embedder.titles.iter().all(String::is_empty),
            "{:#?}",
            embedder.titles
        );
    }

    /// `note` is the arm that behaves like #2's prefix: one constant for the
    /// whole file.
    #[test]
    fn the_note_arm_sends_one_constant_title_for_every_chunk() {
        let (_store, embedder) =
            index_prefixed_vault(&title_config(crate::llm::DocumentTitle::Note));

        assert_eq!(embedder.titles.len(), 3);
        assert!(
            embedder.titles.iter().all(|t| t == "Archdragon"),
            "{:#?}",
            embedder.titles
        );
    }

    /// A note written through the write pipeline lands in the same vector space
    /// as an indexed one, so both paths have to compose the embedder's input the
    /// same way. Two compositions that *can* disagree eventually do, which is
    /// why they share one `EmbedComposition` and one `embed_inputs`.
    #[test]
    fn the_write_pipeline_and_the_indexer_embed_a_chunk_identically() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let store = Store::open_memory().unwrap();
        let mut embedder = RecordingEmbed::new(256);
        let config = title_config(crate::llm::DocumentTitle::Breadcrumb);
        let content = "---\nname: Archdragon\n---\n\n\
                       ## Definition\n\nRank SS.\n\n\
                       ## Abilities\n\nFlight.\n";

        let written = crate::writer::create_note(
            crate::writer::CreateNoteInput {
                content: content.to_string(),
                filename: "archdragon".into(),
                type_hint: None,
                tags: vec![],
                folder: Some("lore".into()),
                created_by: "test".into(),
                auto_link: Some(false),
            },
            &store,
            &mut embedder,
            crate::prefix::EmbedComposition::from_config(&config),
            config.chunk_options(),
            root,
            None,
        )
        .unwrap();
        let from_write = std::mem::take(&mut embedder.titles);
        assert!(
            from_write.iter().any(|t| t.contains(" > ")),
            "the write path sent no breadcrumb: {from_write:#?}"
        );

        let path = written.path.clone();
        let on_disk = std::fs::read_to_string(root.join(&path)).unwrap();
        index_file(
            &path,
            &on_disk,
            "a-new-hash",
            &store,
            &mut embedder,
            root,
            &config,
        )
        .unwrap();

        assert_eq!(from_write, embedder.titles);
    }

    // ── The lexical limb of the breadcrumb rule (#37) ────────────

    /// The breadcrumb is stored on the chunk row, and it is the same string the
    /// title field carries when that limb is on. One rule, one composition — the
    /// two limbs cannot drift, because they call one function.
    ///
    /// The storage does not depend on the embedding limb. This runs the arm that
    /// has both, because a comparison needs both.
    #[test]
    fn the_breadcrumb_is_stored_on_the_chunk_row() {
        let (store, embedder) =
            index_prefixed_vault(&title_config(crate::llm::DocumentTitle::Breadcrumb));
        let file = store
            .get_file("lore/bestiary/archdragon.md")
            .unwrap()
            .unwrap();

        let stored: Vec<String> = store
            .get_chunks_by_file(file.id)
            .unwrap()
            .iter()
            .map(|c| c.heading_path.clone())
            .collect();
        assert_eq!(
            stored,
            vec![
                "lore/bestiary/archdragon.md > Definition",
                "lore/bestiary/archdragon.md > Abilities",
                "lore/bestiary/archdragon.md > Abilities > Combat",
            ]
        );

        let mut titles = embedder.titles.clone();
        let mut lexical = stored.clone();
        titles.sort();
        lexical.sort();
        assert_eq!(titles, lexical, "the two limbs carry one string");

        // And on the shipped default, where the embedding limb is off, the
        // chunk row carries the breadcrumb just the same. This is the whole of
        // the rule after #38, so it cannot depend on the other limb.
        let (default_store, default_embedder) = index_prefixed_vault(&Config::default());
        let file = default_store
            .get_file("lore/bestiary/archdragon.md")
            .unwrap()
            .unwrap();
        let default_stored: Vec<String> = default_store
            .get_chunks_by_file(file.id)
            .unwrap()
            .iter()
            .map(|c| c.heading_path.clone())
            .collect();
        assert_eq!(default_stored, stored);
        assert!(default_embedder.titles.iter().all(String::is_empty));
    }

    /// The tags of the file, on every chunk of it, sorted. Frontmatter order is
    /// not information, and a stable string keeps two indexes of one file
    /// byte-identical.
    #[test]
    fn the_files_tags_are_stored_on_every_chunk() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "n.md",
            "---\ntags:\n  - zebra\n  - apex\n---\n\n## One\n\nBody.\n\n## Two\n\nMore.\n",
        );
        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        run_index_shared(
            tmp.path(),
            &Config::default(),
            crate::indexer::IndexSettings::from_config(&Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let file = store.get_file("n.md").unwrap().unwrap();
        for chunk in store.get_chunks_by_file(file.id).unwrap() {
            assert_eq!(chunk.tags_text, "apex zebra");
        }
    }

    /// The point of the issue: a term that appears **only** in a heading is
    /// reachable through the keyword lane. The tag is not, under the shipped
    /// default — `[fts] tags` is off and the column is not declared — and it
    /// becomes reachable the moment it is turned on.
    #[test]
    fn a_heading_term_and_a_tag_are_matchable_without_appearing_in_a_body() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "rules/spells.md",
            "---\ntags:\n  - grimoire\n---\n\n## Abjuration\n\nStops a caster.\n",
        );
        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        run_index_shared(
            tmp.path(),
            &Config::default(),
            crate::indexer::IndexSettings::from_config(&Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let file = store.get_file("rules/spells.md").unwrap().unwrap();
        let body = store.get_chunk_by_seq(file.id, 0).unwrap().unwrap().text;
        assert!(
            !body.contains("grimoire"),
            "fixture puts the tag in the body"
        );

        let shipped = crate::config::FtsConfig::default();
        for term in ["Abjuration", "spells"] {
            let hits = store
                .fts_search_any(term, 10, &shipped.weights(), None)
                .unwrap();
            assert_eq!(hits.len(), 1, "no keyword hit for {term:?}");
            assert_eq!(hits[0].file_id, file.id);
        }
        assert!(
            store
                .fts_search_any("grimoire", 10, &shipped.weights(), None)
                .unwrap()
                .is_empty(),
            "the tags column ships undeclared"
        );

        let with_tags = crate::config::FtsConfig {
            tags: true,
            ..shipped
        };
        store.rebuild_fts(&with_tags).unwrap();
        assert_eq!(
            store
                .fts_search_any("grimoire", 10, &with_tags.weights(), None)
                .unwrap()
                .len(),
            1,
            "turning the column on is a rebuild and nothing more"
        );
    }

    /// The control. With both columns off the index is declared over the body
    /// alone, and every score is the one the lane returned before this issue.
    /// A weight of zero is not this: BM25 normalises over every token in the
    /// row, so a populated column moves every score whatever its weight.
    #[test]
    fn the_control_scores_exactly_as_a_body_only_index_did() {
        let files = [
            (
                "a.md",
                "## Abjuration\n\nStops a caster from casting a spell.\n",
            ),
            ("b.md", "## Bread\n\nA short line about casting bread.\n"),
        ];
        let scores = |cfg: crate::config::FtsConfig| -> Vec<(String, f64)> {
            let tmp = TempDir::new().unwrap();
            let mut config = Config {
                fts: cfg,
                ..Config::default()
            };
            config.exclude.clear();
            for (path, content) in &files {
                write_file(tmp.path(), path, content);
            }
            let store = Store::open_memory().unwrap();
            let mut embedder = crate::llm::MockLlm::new(256);
            run_index_shared(
                tmp.path(),
                &config,
                crate::indexer::IndexSettings::from_config(&config),
                &store,
                &mut embedder,
                false,
                None,
            )
            .unwrap();
            store
                .fts_search_any("casting", 10, &config.fts.weights(), None)
                .unwrap()
                .iter()
                .map(|r| {
                    let path = store.get_file_by_id(r.file_id).unwrap().unwrap().path;
                    (path, r.score)
                })
                .collect()
        };

        let control = scores(crate::config::FtsConfig::CONTROL);
        let indexed = scores(crate::config::FtsConfig::default());
        assert_eq!(control.len(), 2);
        // Same rows, same order, and the control's scores are the pre-#37 ones.
        assert_eq!(
            control.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>(),
            indexed.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>(),
        );
        assert!(
            control
                .iter()
                .zip(&indexed)
                .any(|((_, a), (_, b))| (a - b).abs() > f64::EPSILON),
            "the declared columns changed nothing at all, which cannot be right: \
             {control:?} {indexed:?}"
        );
    }

    /// `[fts]` is a schema change and nothing more. It rebuilds the keyword
    /// index, which reads no files and runs no model, and it leaves every
    /// vector where it was.
    #[test]
    fn changing_the_declared_columns_rebuilds_the_keyword_index_only() {
        let (tmp, store, mut embedder, _config) = fingerprint_fixture();
        let before = chunk_snapshot(&store);
        let vectors: Vec<(u64, Vec<f32>)> = store.get_all_vectors().unwrap();

        let control = Config {
            fts: crate::config::FtsConfig::CONTROL,
            ..Config::default()
        };
        let result = run_index_shared(
            tmp.path(),
            &control,
            crate::indexer::IndexSettings::from_config(&control),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(result.new_files, 0);
        assert_eq!(result.updated_files, 0);
        assert!(
            embedder.seen.is_empty(),
            "a column list is not an embedding input: {:?}",
            embedder.seen
        );
        assert_eq!(before, chunk_snapshot(&store));
        assert_eq!(vectors, store.get_all_vectors().unwrap());
        assert_eq!(
            store.fts_columns().unwrap(),
            Some(vec!["text".to_string()]),
            "the index should have been redeclared"
        );
        assert!(fts_row_count(&store) > 0, "and repopulated");
    }

    // ── Chunk-granular edges (#28) ───────────────────────────────

    /// Index a vault of `(path, content)` with the mock embedder and hand back
    /// the store.
    fn index_vault(root: &Path, files: &[(&str, &str)]) -> Store {
        use crate::llm::MockLlm;
        for (path, content) in files {
            write_file(root, path, content);
        }
        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        run_index_shared(
            root,
            &Config::default(),
            crate::indexer::IndexSettings::from_config(&Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        store
    }

    #[test]
    fn an_edges_source_chunk_is_the_one_that_held_the_link() {
        // The point of #28. `hub.md` links to `near` from `## Role` and to
        // `far` from `## Session History`; before this, a seed matching either
        // section expanded to both.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                (
                    "hub.md",
                    "# Hub\n## Role\nIt does the thing, with [[near]]. That is the whole of \
                     what it is for, and the role reaches no further than that in any direction \
                     that anyone has yet found.\n\n\
                     ## Session History\nLong ago, [[far]] happened. Nobody who was there \
                     has told the story the same way twice since, so the record of it is thin and \
                     the parts that agree are thinner.\n",
                ),
                ("near.md", "# Near\nnothing here"),
                ("far.md", "# Far\nnothing here"),
            ],
        );

        let hub = store.get_file("hub.md").unwrap().unwrap();
        let seq_of = |heading: &str| {
            store
                .get_chunks_by_file(hub.id)
                .unwrap()
                .into_iter()
                .find(|c| c.heading.contains(heading))
                .unwrap_or_else(|| panic!("no chunk headed {heading}"))
                .seq
        };

        assert_eq!(
            edge_snapshot(&store),
            vec![
                format!("hub.md#{} -> near.md (wikilink)", seq_of("Role")),
                format!("hub.md#{} -> far.md (wikilink)", seq_of("Session History")),
            ]
        );
    }

    #[test]
    fn the_document_level_view_is_what_it_always_was() {
        // #28's first acceptance criterion: storing the finer grain must lose
        // nothing. A document's link set is the union of its chunks'.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                (
                    "hub.md",
                    "# Hub\n## Role\nSee [[near]] and [[far]]. Between them they cover every \
                     part of what this hub is answerable for, and no other section of the file \
                     covers any of it.\n\n\
                     ## History\nStill [[near]], plus [[far]]. The pair have been named \
                     together in this section for as long as the section has existed, and neither \
                     has ever been named here alone.\n",
                ),
                ("near.md", "# Near\nBack to [[hub]]."),
                ("far.md", "# Far\nnothing here"),
            ],
        );

        let paths: HashMap<i64, String> = store
            .get_all_files()
            .unwrap()
            .into_iter()
            .map(|f| (f.id, f.path))
            .collect();
        let mut pairs: Vec<(String, String)> = store
            .wikilink_pairs()
            .unwrap()
            .into_iter()
            .map(|(from, to)| (paths[&from].clone(), paths[&to].clone()))
            .collect();
        pairs.sort();

        assert_eq!(
            pairs,
            vec![
                ("hub.md".to_string(), "far.md".to_string()),
                ("hub.md".to_string(), "near.md".to_string()),
                ("near.md".to_string(), "hub.md".to_string()),
            ],
            "the derived document view must match what the old file-level table held"
        );
        // And the fine grain really is finer: hub links to each target twice,
        // from two different sections.
        assert_eq!(edge_snapshot(&store).len(), 5);
    }

    #[test]
    fn a_deep_link_lands_on_the_chunks_under_that_heading() {
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                ("hub.md", "# Hub\nSee [[dragon#Human Forms]] for detail."),
                (
                    "dragon.md",
                    "# Dragon\n## Origin\nBorn of fire, in the age before the first cities, \
                     older than any record that survives to name it, and older than the language \
                     those records were written in.\n\n\
                     ## Human Forms\nIt walks as a man, and holds the shape for decades at \
                     a time, long enough to be born, to marry and to be buried under a human name \
                     that nobody thinks to question.\n",
                ),
            ],
        );

        let dragon = store.get_file("dragon.md").unwrap().unwrap();
        let human_forms = store
            .get_chunks_by_file(dragon.id)
            .unwrap()
            .into_iter()
            .find(|c| c.heading.contains("Human Forms"))
            .unwrap()
            .seq;

        assert_eq!(
            edge_snapshot(&store),
            vec![format!("hub.md#0 -> dragon.md#{human_forms} (wikilink)")]
        );
    }

    #[test]
    fn a_deep_link_to_a_renamed_heading_degrades_to_the_document() {
        // "Never drop an edge on a failed resolve." A deep link is more fragile
        // than a plain one and the graph must not lose recall over a retitle.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                ("hub.md", "# Hub\nSee [[dragon#Alternate Form]]."),
                (
                    "dragon.md",
                    "# Dragon\n## Origin\nBorn of fire, in the age before the first cities, \
                     older than any record that survives to name it, and older than the language \
                     those records were written in.\n\n\
                     ## Human Forms\nIt walks as a man, and holds the shape for decades at \
                     a time, long enough to be born, to marry and to be buried under a human name \
                     that nobody thinks to question.\n",
                ),
            ],
        );

        assert_eq!(
            edge_snapshot(&store),
            vec!["hub.md#0 -> dragon.md (wikilink)".to_string()],
            "the link must survive at document level, not vanish"
        );
    }

    #[test]
    fn a_link_no_chunk_contains_is_attributed_to_the_document() {
        // Frontmatter is stripped before chunking, so a link there belongs to no
        // passage. Obsidian allows them, and dropping them would shrink the
        // document-level view the previous test pins.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                (
                    "hub.md",
                    "---\nrelated: \"[[near]]\"\n---\n# Hub\nNo links in the body.\n",
                ),
                ("near.md", "# Near\nnothing here"),
            ],
        );

        assert_eq!(
            edge_snapshot(&store),
            vec!["hub.md -> near.md (wikilink)".to_string()]
        );
    }

    #[test]
    fn backfilling_from_stored_chunks_reproduces_a_full_index() {
        // #28's second acceptance criterion, and the whole adoption story: the
        // edge table can be rebuilt from `chunks.text` alone (#14), so no vault
        // is re-read, nothing is re-chunked and nothing is re-embedded.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                (
                    "hub.md",
                    "# Hub\n## Role\nSee [[near]] and [[dragon#Human Forms]].\n\n## History\n[[far]] once.\n",
                ),
                ("near.md", "# Near\nBack to [[hub]]."),
                ("far.md", "# Far\nnothing here"),
                (
                    "dragon.md",
                    "# Dragon\n## Origin\nBorn of fire, in the age before the first cities, \
                     older than any record that survives to name it, and older than the language \
                     those records were written in.\n\n\
                     ## Human Forms\nIt walks as a man, and holds the shape for decades at \
                     a time, long enough to be born, to marry and to be buried under a human name \
                     that nobody thinks to question.\n",
                ),
            ],
        );
        let from_index = edge_snapshot(&store);
        assert!(!from_index.is_empty());

        // Wipe every edge and rebuild from the database alone.
        store.clear_edges().unwrap();
        backfill_edges_from_chunks(&store).unwrap();

        assert_eq!(edge_snapshot(&store), from_index);
    }

    #[test]
    fn the_backfill_keeps_edges_no_chunk_can_account_for() {
        // The backfill sees `chunks.text`, and the chunker strips frontmatter,
        // so a frontmatter link is invisible to it. It is visible to the
        // document-level row the #28 migration carried over, and that row is
        // what stands in for it — otherwise adopting the new grain would silently
        // drop edges a full reindex keeps.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                (
                    "hub.md",
                    "---\nrelated: \"[[near]]\"\n---\n# Hub\nAlso [[far]] in the body.\n",
                ),
                ("near.md", "# Near\nnothing here"),
                ("far.md", "# Far\nnothing here"),
            ],
        );
        let from_index = edge_snapshot(&store);

        backfill_edges_from_chunks(&store).unwrap();
        assert_eq!(edge_snapshot(&store), from_index);
    }

    #[test]
    fn an_index_run_adopts_the_new_grain_without_reindexing() {
        // #28's last acceptance criterion. The store is complete and unchanged;
        // the only work left is re-deriving edges, and that touches no file.
        let tmp = TempDir::new().unwrap();
        let store = index_vault(
            tmp.path(),
            &[
                (
                    "hub.md",
                    "# Hub\n## Role\nSee [[near]].\n\n## History\n[[far]] once.\n",
                ),
                ("near.md", "# Near\nnothing here"),
                ("far.md", "# Far\nnothing here"),
            ],
        );
        let expected = edge_snapshot(&store);

        // Coarsen the table to exactly what the migration leaves behind, and
        // raise the flag it raises.
        store
            .conn()
            .execute_batch(&format!(
                "DELETE FROM edges;
                 INSERT INTO edges (from_file, from_chunk_seq, to_file, to_chunk_seq, edge_type)
                     SELECT DISTINCT f.id, {DOC_LEVEL}, t.id, {DOC_LEVEL}, 'wikilink'
                     FROM files f, files t WHERE f.path = 'hub.md' AND t.path <> 'hub.md';"
            ))
            .unwrap();
        store.set_meta("edges_backfill_pending", "1").unwrap();
        assert_ne!(edge_snapshot(&store), expected);

        use crate::llm::MockLlm;
        let mut embedder = MockLlm::new(256);
        let result = run_index_shared(
            tmp.path(),
            &Config::default(),
            crate::indexer::IndexSettings::from_config(&Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        assert_eq!(
            (result.new_files, result.updated_files),
            (0, 0),
            "nothing should have been re-indexed"
        );
        assert_eq!(edge_snapshot(&store), expected);
        assert!(!store.needs_edge_backfill().unwrap());
    }

    // ── The tag store (#60) ──────────────────────────────────────

    fn tag_counts(store: &Store) -> (i64, i64) {
        let tags = store
            .conn()
            .query_row("SELECT COUNT(*) FROM tags", [], |row| row.get(0))
            .unwrap();
        let links = store
            .conn()
            .query_row("SELECT COUNT(*) FROM file_tags", [], |row| row.get(0))
            .unwrap();
        (tags, links)
    }

    fn index_once(dir: &std::path::Path, store: &Store) {
        let mut embedder = crate::llm::MockLlm::new(256);
        run_index_shared(
            dir,
            &Config::default(),
            crate::indexer::IndexSettings::from_config(&Config::default()),
            store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
    }

    #[test]
    fn a_property_tag_and_a_body_tag_both_reach_the_store() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "n.md",
            "---\ntags:\n  - habitat/swamp\n---\n\n## One\n\nA #type/undead lives here.\n",
        );
        let store = Store::open_memory().unwrap();
        index_once(tmp.path(), &store);

        let file = store.get_file("n.md").unwrap().unwrap();
        assert_eq!(
            store.file_tags(file.id).unwrap(),
            vec!["habitat/swamp", "type/undead"]
        );
    }

    #[test]
    fn a_file_indexed_twice_holds_the_rows_it_held_once() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "n.md",
            "---\ntags: [alpha, beta]\n---\n\n## One\n\nBody.\n",
        );
        let store = Store::open_memory().unwrap();
        index_once(tmp.path(), &store);
        let first = tag_counts(&store);

        // A second full pass over an unchanged vault.
        index_once(tmp.path(), &store);
        assert_eq!(tag_counts(&store), first);
        assert_eq!(first, (2, 2));
    }

    #[test]
    fn a_vault_indexed_twice_reports_the_usage_of_one_index() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "a.md",
            "---\ntags: [shared]\n---\n\n## A\n\nBody.\n",
        );
        write_file(
            tmp.path(),
            "b.md",
            "---\ntags: [shared]\n---\n\n## B\n\nBody.\n",
        );
        let store = Store::open_memory().unwrap();
        index_once(tmp.path(), &store);
        index_once(tmp.path(), &store);

        let usage: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM file_tags ft JOIN tags t ON t.id = ft.tag_id
                  WHERE t.path = 'shared'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(usage, 2, "usage counts notes, not index events");
    }

    #[test]
    fn editing_a_note_to_drop_its_last_tag_deletes_the_tag_row() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "n.md",
            "---\ntags: [habitat/swamp]\n---\n\n## One\n\nBody.\n",
        );
        let store = Store::open_memory().unwrap();
        index_once(tmp.path(), &store);
        assert_eq!(tag_counts(&store), (1, 1));

        write_file(
            tmp.path(),
            "n.md",
            "---\ntags: []\n---\n\n## One\n\nBody.\n",
        );
        index_once(tmp.path(), &store);
        assert_eq!(tag_counts(&store), (0, 0));
    }

    #[test]
    fn a_tag_a_second_note_carries_survives_the_first_dropping_it() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "a.md",
            "---\ntags: [habitat/swamp]\n---\n\n## A\n\nBody.\n",
        );
        write_file(
            tmp.path(),
            "b.md",
            "---\ntags: [habitat/swamp]\n---\n\n## B\n\nBody.\n",
        );
        let store = Store::open_memory().unwrap();
        index_once(tmp.path(), &store);
        assert_eq!(tag_counts(&store), (1, 2));

        write_file(tmp.path(), "a.md", "---\ntags: []\n---\n\n## A\n\nBody.\n");
        index_once(tmp.path(), &store);
        assert_eq!(tag_counts(&store), (1, 1));
    }

    #[test]
    fn removing_a_file_takes_its_links_and_the_tags_that_go_unused_with_it() {
        let tmp = TempDir::new().unwrap();
        write_file(
            tmp.path(),
            "a.md",
            "---\ntags: [solo, shared]\n---\n\n## A\n\nBody.\n",
        );
        write_file(
            tmp.path(),
            "b.md",
            "---\ntags: [shared]\n---\n\n## B\n\nBody.\n",
        );
        let store = Store::open_memory().unwrap();
        index_once(tmp.path(), &store);
        assert_eq!(tag_counts(&store), (2, 3));

        remove_file("a.md", &store).unwrap();
        assert_eq!(tag_counts(&store), (1, 1));
        let left: String = store
            .conn()
            .query_row("SELECT path FROM tags", [], |row| row.get(0))
            .unwrap();
        assert_eq!(left, "shared");
    }

    /// The orphan sweep at the head of every index removes the `files` row of
    /// a note the vault no longer has, and the note's broken links go with it.
    ///
    /// They used to stay: the sweep is `verify_index_integrity`, one more
    /// `delete_file` caller that a table keyed on the source **path** escaped.
    /// Nothing else could reach the row either — only a file's own re-index
    /// deletes one, and a file that is gone is never re-indexed — so a rebuild
    /// re-read all 239 notes and reported the same ghosts afterwards (#98).
    #[test]
    fn a_rebuild_removes_a_note_that_is_no_longer_on_disk() {
        use crate::llm::MockLlm;

        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write_file(root, "keeper.md", "# Keeper\n\nStays.\n");
        write_file(root, "vanishes.md", "# Vanishes\n\n[[Nowhere]]\n");

        let store = Store::open_memory().unwrap();
        let mut embedder = MockLlm::new(256);
        let config = Config::default();
        let index = |store: &Store, embedder: &mut MockLlm, rebuild: bool| {
            run_index_shared(
                root,
                &config,
                crate::indexer::IndexSettings::from_config(&config),
                store,
                embedder,
                rebuild,
                None,
            )
            .unwrap()
        };

        index(&store, &mut embedder, false);
        assert_eq!(store.get_unresolved_links().unwrap().len(), 1);

        std::fs::remove_file(root.join("vanishes.md")).unwrap();
        index(&store, &mut embedder, true);

        assert!(
            store.get_file("vanishes.md").unwrap().is_none(),
            "a rebuild drops the row of a file the vault no longer has"
        );
        assert!(
            store.get_unresolved_links().unwrap().is_empty(),
            "and the broken links it reported go with it: {:?}",
            store.get_unresolved_links().unwrap()
        );
        assert!(
            store.get_file("keeper.md").unwrap().is_some(),
            "the file that is still there is still indexed"
        );
    }
}
