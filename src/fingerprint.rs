//! What built the index, so the store can tell when that changed (issue #31).
//!
//! engraph fingerprinted exactly one thing: embedding *dimension*. Everything
//! else that decides what is in the store — chunk boundaries, the embedding
//! prompt template, the tokenizer, the FTS schema, the link resolver, the model
//! artifact itself — could change while the store reported itself healthy, and
//! the index would quietly stop matching the code that reads it.
//!
//! That matters more here than the usual cache-invalidation argument. With a
//! five-probe instrument, **a stale index and a real effect look the same in the
//! output**: both move ranks, neither announces itself. #27 is the precedent —
//! every graph-lane number recorded before it was silently taken on a best-case
//! graph, and it took a hand-written invariant to notice.
//!
//! Six keys live in `meta`, each with one declared [`Action`] on mismatch.
//! Section references in the form §N point at `docs/vault-search-design-v1_3.md`.
//!
//! ## Two kinds of input
//!
//! - **Data**, hashed exactly: config values, the FTS schema text, model and
//!   tokenizer artifact digests. A change here is detected with no human step.
//! - **Algorithm**, carried by a version constant below. There is no runtime
//!   view of what a function does, so changing one of them means bumping its
//!   constant. That is the weak joint in this design and it is deliberate: the
//!   alternative, hashing module source, fires on a comment edit and on every
//!   test-only change, which is the "rebuilds on every startup" failure the
//!   acceptance criteria call out.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::store::Store;

// ── Algorithm versions ───────────────────────────────────────────────────────

/// Bump when the markdown parser changes what it extracts from a file:
/// frontmatter splitting, heading detection, tag parsing.
pub const PARSER_VERSION: u32 = 2;

/// Bump when the chunker's *rules* change — break-point scoring, section
/// structuring, sentence splitting, overlap construction.
///
/// The chunker's *numbers* are hashed directly from [`crate::chunker::limits`]
/// and need no bump.
///
/// Version 2 is issue #44: a heading line skipped for an empty body is carried
/// into the promoted section that follows it. A promoted line is an ancestor of
/// nothing, so the skipped heading has no descendant breadcrumb to survive in,
/// and a promoted section under `chunk_min_chars` merges into a chunk that keeps
/// the host's breadcrumb. Without the carry the heading is in no row at all.
pub const CHUNKER_VERSION: u32 = 2;

/// Bump when what a chunk **row** holds changes, even though the chunk
/// boundaries do not.
///
/// Separate from [`CHUNKER_VERSION`] because the two answer different
/// questions: that one asks whether the chunks would come out in the same
/// places, this one asks whether the row written for a chunk still holds
/// everything a reader now needs. Both declare `Reindex`, since the row is
/// written by the same pass that embeds it.
///
/// Version 2 is issue #37: `heading_path` and `tags_text`. Neither can be
/// derived from the columns already stored — the breadcrumb's ancestors are in
/// the vault, not in `chunks.heading` — so a store built before them has to be
/// read again, and a keyword index declared over empty columns would look
/// healthy while matching nothing.
pub const CHUNK_RECORD_VERSION: u32 = 2;

/// Bump when wikilink resolution changes: extraction, the exact → basename →
/// shortest-path ladder, or how an end that names no passage is stored.
pub const LINK_RESOLVER_VERSION: u32 = 1;

/// Bump when the *text* of a [`crate::llm::PromptFormat`] template changes
/// while the template keeps its name.
///
/// *Which* template wrote a vector is data, not algorithm, and is hashed
/// exactly from [`crate::llm::PromptFormat::template_id`] — so selecting
/// EmbeddingGemma's documented document template over nomic's convention
/// (issue #10) re-indexes with no bump here. Rewording one of them does need a
/// bump, because the id would not move.
///
/// The query template is deliberately absent from both: a query is embedded and
/// discarded, so changing it makes nothing in the store stale.
pub const PROMPT_TEMPLATE_VERSION: u32 = 1;

/// Bump when what happens to a vector after inference changes — currently L2
/// normalisation, applied in `LlamaEmbed::embed_formatted`.
pub const EMBEDDING_NORMALIZATION_VERSION: u32 = 1;

// ── Actions ──────────────────────────────────────────────────────────────────

/// What a mismatched fingerprint requires before the index means anything again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    /// Re-read, rechunk and re-embed every file.
    ///
    /// §4 distinguishes reparse from rechunk from re-embed. engraph does not:
    /// `index_file` parses, chunks and embeds as one unit, and re-embedding is
    /// where essentially all of the cost is — rechunking on top of it is free.
    /// Splitting them would buy nothing and add three code paths.
    Reindex,
    /// Rebuild `chunks_fts` from `chunks.text`. No vault read, no model.
    RebuildFts,
    /// Re-derive every edge from the vault. A vault read, but no model.
    RebuildEdges,
    /// Nothing stored is stale. Any calibrated threshold is.
    ///
    /// The only action that must **not** reindex. Getting this one wrong costs
    /// a full re-embed for a change that touched no stored bytes.
    InvalidateThresholds,
}

impl Action {
    /// Whether this action means the stored index no longer answers correctly.
    ///
    /// Read paths refuse to run when one of these is outstanding, the way
    /// `verify_embedding_dim` refuses on a width change. `InvalidateThresholds`
    /// is not one: there is nothing to rebuild, so failing a search over it
    /// would be theatre.
    pub fn blocks_reads(self) -> bool {
        self != Action::InvalidateThresholds
    }

    pub fn describe(self) -> &'static str {
        match self {
            Action::Reindex => "re-index the vault",
            Action::RebuildFts => "rebuild the keyword index",
            Action::RebuildEdges => "rebuild the vault graph",
            Action::InvalidateThresholds => "discard calibrated thresholds",
        }
    }
}

// ── The fingerprints ─────────────────────────────────────────────────────────

/// One `meta` key: its name, what it covers, and what a change to it costs.
pub struct Key {
    pub name: &'static str,
    pub action: Action,
}

pub const PARSER: Key = Key {
    name: "parser_fingerprint",
    action: Action::Reindex,
};
pub const CHUNKER: Key = Key {
    name: "chunker_fingerprint",
    action: Action::Reindex,
};
pub const LINK: Key = Key {
    name: "link_fingerprint",
    action: Action::RebuildEdges,
};
pub const FTS: Key = Key {
    name: "fts_fingerprint",
    action: Action::RebuildFts,
};
pub const EMBEDDING: Key = Key {
    name: "embedding_fingerprint",
    action: Action::Reindex,
};
pub const RERANKER: Key = Key {
    name: "reranker_fingerprint",
    action: Action::InvalidateThresholds,
};

/// The six §4 fingerprints for the code and configuration currently running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprints {
    pub parser: String,
    pub chunker: String,
    pub link: String,
    pub fts: String,
    pub embedding: String,
    /// `None` when no reranker is loaded — intelligence off, or a read path
    /// that never needed one. An absent value is *not* an empty one: it must
    /// not overwrite what a reranked run recorded.
    pub reranker: Option<String>,
}

impl Fingerprints {
    /// Compute all six from the running configuration.
    ///
    /// `embed_model` and `rerank_model` are [`crate::llm::EmbedModel::fingerprint`]
    /// and [`crate::llm::RerankModel::fingerprint`] — the artifact digests, which
    /// only the loaded model knows the path of. Everything else is read here.
    pub fn compute(config: &Config, embed_model: &str, rerank_model: Option<&str>) -> Self {
        use crate::chunker::limits;

        Self {
            parser: digest(&[&PARSER_VERSION.to_string()]),
            chunker: digest(&[
                &CHUNKER_VERSION.to_string(),
                &CHUNK_RECORD_VERSION.to_string(),
                // #46: the breadcrumb root is written into `chunks.heading_path`,
                // a stored column, so changing it rewrites every chunk row.
                &format!("{:?}", config.breadcrumb_root),
                // #43: the minimum decides which sections become rows at all,
                // so it sits with the limits it belongs among rather than with
                // the embedding inputs.
                &config.chunk_min_chars.to_string(),
                // #44: promotion decides where a section starts, so it decides
                // which rows exist at all — the same class of change as the
                // minimum, and the same action.
                &config.promote_bold_headings.to_string(),
                &limits::TARGET_TOKENS.to_string(),
                &limits::OVERLAP_PCT.to_string(),
                &limits::MAX_TOKENS.to_string(),
                &limits::OVERLAP_TOKENS.to_string(),
            ]),
            link: digest(&[
                &LINK_RESOLVER_VERSION.to_string(),
                &crate::store::DOC_LEVEL.to_string(),
            ]),
            fts: digest(&[&crate::store::fts_objects_sql(&config.fts)]),
            // The prefix and the title field are embedding input, not chunk
            // content: they change the vector and leave `chunks.text`, the
            // snippet and FTS untouched (issues #2 and #36). So they belong here
            // and not in `chunker`.
            embedding: digest(&[
                embed_model,
                &prefix_identity(config),
                &document_title_identity(config),
            ]),
            reranker: rerank_model.map(|model| {
                digest(&[
                    model,
                    &config.rerank.document_title.to_string(),
                    &config.rerank.max_document_chars.to_string(),
                ])
            }),
        }
    }

    /// Every key with a value to compare, paired with what it covers.
    fn entries(&self) -> Vec<(Key, &str)> {
        let mut out = vec![
            (PARSER, self.parser.as_str()),
            (CHUNKER, self.chunker.as_str()),
            (LINK, self.link.as_str()),
            (FTS, self.fts.as_str()),
            (EMBEDDING, self.embedding.as_str()),
        ];
        if let Some(reranker) = &self.reranker {
            out.push((RERANKER, reranker.as_str()));
        }
        out
    }
}

/// How the contextual prefix is composed, as a stable string.
///
/// Spelled out rather than derived so that adding a `PrefixConfig` field is a
/// visible edit here: a silently unhashed component is a vector change nobody
/// sees, which is the whole defect.
fn prefix_identity(config: &Config) -> String {
    let p = &config.embedding_prefix;
    format!(
        "prefix(enabled={},path={},heading={},tags={},aliases={})",
        p.enabled, p.path, p.heading, p.tags, p.aliases
    )
}

/// Which string fills the document template's `title:` field (issue #36).
///
/// A separate component from [`prefix_identity`] and from the model's own
/// `template_id`, because it is a third independent choice: the same model and
/// the same template write different vectors for `none`, `note` and
/// `breadcrumb`.
fn document_title_identity(config: &Config) -> String {
    format!(
        "document_title({})",
        config.embedding_prompt.document_title.id()
    )
}

// ── Comparison ───────────────────────────────────────────────────────────────

/// A stored fingerprint that disagrees with the running code.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub key: &'static str,
    pub action: Action,
    pub stored: String,
    pub computed: String,
}

/// The result of checking the store against the running code.
#[derive(Debug, Clone, Default)]
pub struct Comparison {
    pub mismatches: Vec<Mismatch>,
    /// Keys the store has never held.
    ///
    /// Adopted without a rebuild. There is no evidence the index disagrees, and
    /// forcing every pre-#31 store through a full reindex to find out is the
    /// same uselessness as rebuilding on every startup — just once. Fingerprints
    /// protect against changes made *after* they are first recorded, and the
    /// warning says so.
    pub unrecorded: Vec<&'static str>,
}

impl Comparison {
    pub fn is_clean(&self) -> bool {
        self.mismatches.is_empty()
    }

    /// The distinct actions the mismatches require, deduplicated and ordered.
    pub fn actions(&self) -> BTreeSet<Action> {
        self.mismatches.iter().map(|m| m.action).collect()
    }

    /// Whether any outstanding action means a read would answer from a stale
    /// index.
    pub fn blocks_reads(&self) -> bool {
        self.mismatches.iter().any(|m| m.action.blocks_reads())
    }

    fn summary(&self) -> String {
        self.mismatches
            .iter()
            .map(|m| format!("{} ({})", m.key, m.action.describe()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Check the store's recorded fingerprints against `computed`.
pub fn compare(store: &Store, computed: &Fingerprints) -> Result<Comparison> {
    let mut comparison = Comparison::default();
    for (key, value) in computed.entries() {
        match store.get_meta(key.name)? {
            Some(stored) if stored != value => comparison.mismatches.push(Mismatch {
                key: key.name,
                action: key.action,
                stored,
                computed: value.to_string(),
            }),
            Some(_) => {}
            None => comparison.unrecorded.push(key.name),
        }
    }
    Ok(comparison)
}

/// Write `computed` into `meta`.
///
/// Called **after** the work it describes has finished, not alongside it. A
/// crash part-way through an index leaves the old fingerprints in place, so the
/// next run redoes the action — a store never claims to match code that never
/// finished running against it.
pub fn record(store: &Store, computed: &Fingerprints) -> Result<()> {
    for (key, value) in computed.entries() {
        store.set_meta(key.name, value)?;
    }
    Ok(())
}

/// Refuse to read from an index the running code did not build.
///
/// The same contract as `Store::verify_embedding_dim`, and for the same reason:
/// the failure it prevents is silent. A mismatch whose action is
/// [`Action::InvalidateThresholds`] is a warning, because nothing stored is
/// wrong.
pub fn verify(store: &Store, computed: &Fingerprints) -> Result<()> {
    let comparison = compare(store, computed)?;
    if comparison.blocks_reads() {
        anyhow::bail!(
            "the index was built by different code or configuration than is \
             running now: {}. Run 'engraph index' to bring it up to date.",
            comparison.summary()
        );
    }
    for mismatch in &comparison.mismatches {
        tracing::warn!(
            key = mismatch.key,
            "{} changed since the index was built; any threshold calibrated \
             against it is stale",
            mismatch.key
        );
    }
    adopt_unrecorded(store, computed, &comparison)?;
    Ok(())
}

/// Stamp keys the store has never held, leaving every key it has alone.
///
/// This is the only path that ever writes `reranker_fingerprint`. The index
/// loads no cross-encoder, so it cannot honestly claim one is current; a search
/// that loads one can, and does, the first time it runs. Without this the
/// reranker key would never exist and a model swap would never be noticed.
///
/// A *mismatched* key is deliberately not overwritten. Recording the new
/// reranker here would consume the very signal a calibrated threshold needs to
/// invalidate itself, so the mismatch keeps warning until whatever owns that
/// threshold clears it and updates the key together.
pub fn adopt_unrecorded(
    store: &Store,
    computed: &Fingerprints,
    comparison: &Comparison,
) -> Result<()> {
    if comparison.unrecorded.is_empty() {
        return Ok(());
    }
    warn_unrecorded(comparison);
    for (key, value) in computed.entries() {
        if comparison.unrecorded.contains(&key.name) {
            store.set_meta(key.name, value)?;
        }
    }
    Ok(())
}

/// Say once, out loud, which keys were adopted without verification.
pub fn warn_unrecorded(comparison: &Comparison) {
    if comparison.unrecorded.is_empty() {
        return;
    }
    tracing::info!(
        keys = comparison.unrecorded.join(", "),
        "recording index fingerprints for the first time; \
         this store's contents are taken on trust, and are protected from here on"
    );
}

// ── Artifact digests ─────────────────────────────────────────────────────────

/// SHA-256 of a model or tokenizer file, cached beside it.
///
/// **A filename is not identity** (§7.1). Swapping a GGUF behind an unchanged
/// filename is precisely the silent case this module exists to catch, so the
/// bytes are what gets hashed.
///
/// Hashing 640 MB costs ~0.3 s warm and ~8 s cold, on every process start, for
/// three models — too much to pay per query. So the digest is cached in a
/// `<file>.sha256` sidecar keyed on `(size, mtime)`, and the steady-state cost
/// is a `stat`. The cache is keyed on metadata but the *fingerprint* is keyed on
/// content, which is the right way round: touching a file re-hashes it and finds
/// the same digest, so a re-download of identical bytes rebuilds nothing.
pub fn artifact_digest(path: &Path) -> Result<String> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("reading model artifact {}", path.display()))?;
    let stamp = format!(
        "{} {}",
        meta.len(),
        meta.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let sidecar = sidecar_path(path);
    if let Ok(cached) = std::fs::read_to_string(&sidecar)
        && let Some((cached_stamp, cached_digest)) = cached.trim().rsplit_once(' ')
        && cached_stamp == stamp
        && !cached_digest.is_empty()
    {
        return Ok(cached_digest.to_string());
    }

    let digest = hash_file(path)?;
    // Best effort: a read-only models directory costs a rehash, not a failure.
    let _ = std::fs::write(&sidecar, format!("{stamp} {digest}\n"));
    Ok(digest)
}

fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".sha256");
    PathBuf::from(name)
}

fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening model artifact {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buf)
            .with_context(|| format!("reading model artifact {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// SHA-256 over `parts`, length-delimited.
///
/// The delimiting is not decoration: without it `("ab", "c")` and `("a", "bc")`
/// hash alike, and two different configurations would share a fingerprint.
pub fn digest(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_le_bytes());
        hasher.update(part.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn fps() -> Fingerprints {
        Fingerprints::compute(
            &Config::default(),
            "embed-model-abc",
            Some("rerank-model-xyz"),
        )
    }

    #[test]
    fn digest_delimits_its_parts() {
        assert_ne!(digest(&["ab", "c"]), digest(&["a", "bc"]));
        assert_eq!(digest(&["ab", "c"]), digest(&["ab", "c"]));
    }

    #[test]
    fn a_fresh_store_records_rather_than_rebuilds() {
        // The negative half of the acceptance criteria, in its first instance: a
        // store that has never held a fingerprint has no evidence of staleness,
        // and must not be rebuilt on suspicion.
        let store = Store::open_memory().unwrap();
        let computed = fps();

        let comparison = compare(&store, &computed).unwrap();
        assert!(comparison.is_clean());
        assert!(!comparison.blocks_reads());
        assert_eq!(comparison.unrecorded.len(), 6);

        verify(&store, &computed).unwrap();
    }

    #[test]
    fn an_unchanged_configuration_rebuilds_nothing() {
        // The other half of the same criterion, and the one that matters: a
        // fingerprint that fires on every startup is as useless as none, just
        // slower.
        let store = Store::open_memory().unwrap();
        let computed = fps();
        record(&store, &computed).unwrap();

        let comparison = compare(&store, &computed).unwrap();
        assert!(comparison.is_clean(), "{:?}", comparison.mismatches);
        assert!(comparison.unrecorded.is_empty());
        verify(&store, &computed).unwrap();
    }

    #[test]
    fn a_changed_embedding_model_demands_a_reindex() {
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let swapped = Fingerprints::compute(
            &Config::default(),
            "embed-model-DIFFERENT",
            Some("rerank-model-xyz"),
        );
        let comparison = compare(&store, &swapped).unwrap();

        assert_eq!(comparison.mismatches.len(), 1);
        assert_eq!(comparison.mismatches[0].key, EMBEDDING.name);
        assert_eq!(comparison.actions(), BTreeSet::from([Action::Reindex]));
        assert!(verify(&store, &swapped).is_err());
    }

    #[test]
    fn a_changed_prefix_composition_demands_a_reindex() {
        // #2's prefix never reaches storage, so no content hash and no chunk
        // text can see it change. Only this can.
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let mut config = Config::default();
        config.embedding_prefix.tags = !config.embedding_prefix.tags;
        let changed = Fingerprints::compute(&config, "embed-model-abc", Some("rerank-model-xyz"));

        let comparison = compare(&store, &changed).unwrap();
        assert_eq!(comparison.mismatches.len(), 1);
        assert_eq!(comparison.mismatches[0].key, EMBEDDING.name);
    }

    /// The breadcrumb root is written into `chunks.heading_path`, a stored
    /// column, so it has to reach the chunker's key and not only the embedding
    /// one — the keyword index is declared over that column and would otherwise
    /// look healthy while holding the other root's strings (issue #46).
    #[test]
    fn a_changed_breadcrumb_root_demands_a_reindex() {
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let mut config = Config::default();
        config.breadcrumb_root = crate::config::BreadcrumbRoot::Name;
        let changed = Fingerprints::compute(&config, "embed-model-abc", Some("rerank-model-xyz"));

        let comparison = compare(&store, &changed).unwrap();
        assert_eq!(comparison.mismatches[0].key, CHUNKER.name);
        assert_eq!(comparison.actions(), BTreeSet::from([Action::Reindex]));
    }

    /// The minimum decides which sections become rows of their own, so it
    /// changes `chunks.text`, every vector derived from it and the keyword
    /// index over it (issue #43).
    #[test]
    fn a_changed_chunk_minimum_demands_a_reindex() {
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        // 0 is the control arm, and it is a different index from the default.
        let mut config = Config::default();
        config.chunk_min_chars = 0;
        let changed = Fingerprints::compute(&config, "embed-model-abc", Some("rerank-model-xyz"));

        let comparison = compare(&store, &changed).unwrap();
        assert_eq!(comparison.mismatches[0].key, CHUNKER.name);
        assert_eq!(comparison.actions(), BTreeSet::from([Action::Reindex]));
    }

    /// Promotion decides where a section starts, so it decides which rows exist
    /// at all — the same class of change as the minimum (issue #44).
    #[test]
    fn a_changed_promotion_setting_demands_a_reindex() {
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        // `true` ships, so the control arm `false` is a different index.
        let mut config = Config::default();
        config.promote_bold_headings = false;
        let changed = Fingerprints::compute(&config, "embed-model-abc", Some("rerank-model-xyz"));

        let comparison = compare(&store, &changed).unwrap();
        assert_eq!(comparison.mismatches[0].key, CHUNKER.name);
        assert_eq!(comparison.actions(), BTreeSet::from([Action::Reindex]));
    }

    /// Each root is a distinct fingerprint, not merely distinct from the
    /// default, so a switch between two non-default arms re-indexes too.
    #[test]
    fn every_breadcrumb_root_fingerprints_differently() {
        use crate::config::BreadcrumbRoot;

        let digests: Vec<String> = [
            BreadcrumbRoot::Path,
            BreadcrumbRoot::Name,
            BreadcrumbRoot::Stem,
        ]
        .into_iter()
        .map(|root| {
            let mut config = Config::default();
            config.breadcrumb_root = root;
            Fingerprints::compute(&config, "embed-model-abc", None).chunker
        })
        .collect();

        let unique: BTreeSet<&String> = digests.iter().collect();
        assert_eq!(unique.len(), 3, "{digests:?}");
    }

    #[test]
    fn a_changed_document_title_demands_a_reindex() {
        // The title field is embedding input on the same terms as #2's prefix:
        // no stored byte and no content hash can see it change (issue #36).
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let mut config = Config::default();
        config.embedding_prompt.document_title = crate::llm::DocumentTitle::Breadcrumb;
        let changed = Fingerprints::compute(&config, "embed-model-abc", Some("rerank-model-xyz"));

        let comparison = compare(&store, &changed).unwrap();
        assert_eq!(comparison.mismatches.len(), 1);
        assert_eq!(comparison.mismatches[0].key, EMBEDDING.name);
        assert_eq!(comparison.actions(), BTreeSet::from([Action::Reindex]));
    }

    /// Each of the three arms of #36 must be a distinct fingerprint, not merely
    /// distinct from the default: `note` and `breadcrumb` write different
    /// vectors from each other as well.
    #[test]
    fn every_document_title_setting_fingerprints_differently() {
        use crate::llm::DocumentTitle;

        let digests: Vec<String> = [
            DocumentTitle::None,
            DocumentTitle::Note,
            DocumentTitle::Breadcrumb,
        ]
        .into_iter()
        .map(|title| {
            let mut config = Config::default();
            config.embedding_prompt.document_title = title;
            Fingerprints::compute(&config, "embed-model-abc", None).embedding
        })
        .collect();

        assert_ne!(digests[0], digests[1]);
        assert_ne!(digests[1], digests[2]);
        assert_ne!(digests[0], digests[2]);
    }

    #[test]
    fn a_changed_reranker_invalidates_thresholds_and_nothing_else() {
        // The one key whose action is not a rebuild. Getting it wrong buys a
        // needless full re-embed for a change that touched no stored byte, so
        // it is asserted explicitly rather than left to follow from the table.
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let swapped = Fingerprints::compute(
            &Config::default(),
            "embed-model-abc",
            Some("rerank-model-DIFFERENT"),
        );
        let comparison = compare(&store, &swapped).unwrap();

        assert_eq!(comparison.mismatches.len(), 1);
        assert_eq!(comparison.mismatches[0].key, RERANKER.name);
        assert!(!comparison.blocks_reads());
        assert!(
            verify(&store, &swapped).is_ok(),
            "a reranker change must not fail a read: there is nothing to rebuild"
        );
    }

    #[test]
    fn a_rerank_input_limit_counts_as_a_reranker_change() {
        // `max_document_chars` decides what the cross-encoder is shown, so it
        // moves the score a threshold would be calibrated against — even though
        // the reranker artifact itself is untouched.
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let mut config = Config::default();
        config.rerank.max_document_chars += 1;
        let changed = Fingerprints::compute(&config, "embed-model-abc", Some("rerank-model-xyz"));

        let comparison = compare(&store, &changed).unwrap();
        assert_eq!(comparison.mismatches.len(), 1);
        assert_eq!(comparison.mismatches[0].key, RERANKER.name);
    }

    #[test]
    fn a_read_path_is_where_the_reranker_key_first_appears() {
        // The index path loads no cross-encoder, so if a read did not stamp this
        // the key would never exist and a swapped reranker would never be seen.
        let store = Store::open_memory().unwrap();
        let indexed = Fingerprints::compute(&Config::default(), "embed-model-abc", None);
        record(&store, &indexed).unwrap();
        assert!(store.get_meta(RERANKER.name).unwrap().is_none());

        let searched = fps();
        verify(&store, &searched).unwrap();
        assert_eq!(
            store.get_meta(RERANKER.name).unwrap().as_deref(),
            Some(searched.reranker.as_deref().unwrap())
        );

        // And once it exists, a swap is a mismatch rather than another adoption.
        let swapped = Fingerprints::compute(
            &Config::default(),
            "embed-model-abc",
            Some("rerank-model-DIFFERENT"),
        );
        let comparison = compare(&store, &swapped).unwrap();
        assert_eq!(comparison.mismatches.len(), 1);
        assert!(comparison.unrecorded.is_empty());
    }

    #[test]
    fn a_mismatched_key_is_never_silently_adopted() {
        // Stamping the new reranker here would consume the signal a calibrated
        // threshold needs in order to invalidate itself. The warning has to keep
        // firing until whatever owns that threshold clears it.
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();

        let swapped = Fingerprints::compute(
            &Config::default(),
            "embed-model-abc",
            Some("rerank-model-DIFFERENT"),
        );
        verify(&store, &swapped).unwrap();
        verify(&store, &swapped).unwrap();

        assert_eq!(
            store.get_meta(RERANKER.name).unwrap().as_deref(),
            Some(fps().reranker.as_deref().unwrap()),
            "the recorded value must survive a read that disagrees with it"
        );
        assert_eq!(compare(&store, &swapped).unwrap().mismatches.len(), 1);
    }

    #[test]
    fn an_absent_reranker_leaves_the_recorded_one_alone() {
        // A search run with intelligence off must not erase what a reranked run
        // recorded, or the next reranked run reads its own change as a mismatch.
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();
        let recorded = store.get_meta(RERANKER.name).unwrap();
        assert!(recorded.is_some());

        let without = Fingerprints::compute(&Config::default(), "embed-model-abc", None);
        record(&store, &without).unwrap();

        assert_eq!(store.get_meta(RERANKER.name).unwrap(), recorded);
        assert!(compare(&store, &without).unwrap().is_clean());
    }

    #[test]
    fn the_fts_fingerprint_tracks_the_schema_text() {
        // No version constant guards this one: the schema is the input, so the
        // digest moves exactly when the declaration does.
        assert_eq!(
            fps().fts,
            digest(&[&crate::store::fts_objects_sql(
                &crate::config::FtsConfig::default()
            )]),
            "fts_fingerprint must be a digest of the declaration itself"
        );
    }

    /// `[fts]` decides the column list, the column list is the declaration, and
    /// the declaration is the digest. So turning a column off rebuilds the
    /// keyword index and re-embeds nothing (issue #37).
    #[test]
    fn the_declared_fts_columns_reach_the_fts_fingerprint() {
        let control = Config {
            fts: crate::config::FtsConfig::CONTROL,
            ..Config::default()
        };
        let shipped = fps();
        let other = Fingerprints::compute(&control, "embed-model-abc", None);

        assert_ne!(shipped.fts, other.fts);
        assert_eq!(
            (shipped.chunker, shipped.embedding),
            (other.chunker, other.embedding),
            "a column list is neither a chunk boundary nor an embedding input"
        );
    }

    /// The columns a chunk *row* holds are a reindex, and they are the chunker's
    /// key: nothing else declares `Reindex` for a change the vault has to be
    /// read again to satisfy (issue #37).
    #[test]
    fn the_chunk_record_version_is_part_of_the_chunker_fingerprint() {
        assert!(
            fps().chunker
                != digest(&[
                    &CHUNKER_VERSION.to_string(),
                    &(CHUNK_RECORD_VERSION + 1).to_string(),
                    &crate::chunker::limits::TARGET_TOKENS.to_string(),
                    &crate::chunker::limits::OVERLAP_PCT.to_string(),
                    &crate::chunker::limits::MAX_TOKENS.to_string(),
                    &crate::chunker::limits::OVERLAP_TOKENS.to_string(),
                ]),
        );
        assert_eq!(CHUNKER.action, Action::Reindex);
    }

    #[test]
    fn every_key_has_its_declared_action() {
        // §4's table, asserted rather than described.
        let store = Store::open_memory().unwrap();
        record(&store, &fps()).unwrap();
        for key in [PARSER, CHUNKER, LINK, FTS, EMBEDDING, RERANKER] {
            store.set_meta(key.name, "stale").unwrap();
        }
        let comparison = compare(&store, &fps()).unwrap();

        let found: Vec<(&str, Action)> = comparison
            .mismatches
            .iter()
            .map(|m| (m.key, m.action))
            .collect();
        assert_eq!(
            found,
            vec![
                (PARSER.name, Action::Reindex),
                (CHUNKER.name, Action::Reindex),
                (LINK.name, Action::RebuildEdges),
                (FTS.name, Action::RebuildFts),
                (EMBEDDING.name, Action::Reindex),
                (RERANKER.name, Action::InvalidateThresholds),
            ]
        );
    }

    #[test]
    fn artifact_digest_is_content_not_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"weights").unwrap();

        let first = artifact_digest(&path).unwrap();
        assert!(sidecar_path(&path).exists(), "digest should be cached");

        // Same bytes written again: mtime moves, so the cache misses and the
        // file is rehashed — and the fingerprint must not move.
        std::fs::write(&path, b"weights").unwrap();
        assert_eq!(artifact_digest(&path).unwrap(), first);

        // Different bytes behind the same filename: the case §7.1 warns about.
        std::fs::write(&path, b"other weights").unwrap();
        assert_ne!(artifact_digest(&path).unwrap(), first);
    }

    #[test]
    fn artifact_digest_survives_an_unwritable_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"weights").unwrap();
        let expected = artifact_digest(&path).unwrap();

        // A corrupt or truncated sidecar must be ignored, not trusted or fatal.
        std::fs::write(sidecar_path(&path), "garbage").unwrap();
        assert_eq!(artifact_digest(&path).unwrap(), expected);
    }
}
