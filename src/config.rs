use anyhow::{Context, Result};
use rmcp::schemars;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use toml_edit::{DocumentMut, Item, Table};

use crate::llm::EmbeddingPromptConfig;
use crate::prefix::PrefixConfig;

/// Model override configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Override embedding model URI (e.g., "hf:repo/file.gguf").
    pub embed: Option<String>,
    /// Override reranker model URI.
    pub rerank: Option<String>,
    /// How many threads llama.cpp may use per forward pass.
    ///
    /// `None` means the machine's **physical core count** — SMT siblings are
    /// not counted, and that distinction is worth more than it sounds: on an
    /// 8-core/16-thread box, 8 threads runs a query in 8.4 s and 16 runs it in
    /// 15 s, worse than the 4 this replaced. llama.cpp's library default is the
    /// constant `GGML_DEFAULT_N_THREADS = 4` regardless of hardware, carrying
    /// its own `// TODO: better default`, and knapper used to inherit it —
    /// so every model call ran on four threads of whatever box it was on
    /// (issue #20).
    ///
    /// Set this to override: to leave headroom for other work, or because a
    /// sweep found a better number for the machine (12 beat 8 on the box above).
    /// Threads change only how the arithmetic is scheduled, never its result —
    /// see [`crate::llm::resolve_n_threads`].
    pub n_threads: Option<usize>,
    /// Non-secret knobs for an API-backed embedder (#84). The key is never
    /// here; it is read from the provider's environment variable at load.
    #[serde(default)]
    pub embed_api: EmbedApiConfig,
}

/// Non-secret configuration for an external embedding API (#84).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedApiConfig {
    /// Matryoshka-truncated output width. `None` uses the model's native width.
    pub dim: Option<usize>,
    /// Per-request timeout in seconds.
    pub timeout_secs: u64,
    /// Retry ceiling on rate-limit, server, and transport errors.
    pub max_retries: u32,
    /// Endpoint override for a proxy or a test server.
    pub endpoint: Option<String>,
}

impl Default for EmbedApiConfig {
    fn default() -> Self {
        Self {
            dim: None,
            timeout_secs: 30,
            max_retries: 4,
            endpoint: None,
        }
    }
}

/// ChatGPT Actions plugin metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginConfig {
    pub name: Option<String>,
    pub description: Option<String>,
    pub contact_email: Option<String>,
    pub public_url: Option<String>,
}

/// User identity for AI agent context.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct IdentityConfig {
    pub name: Option<String>,
    pub role: Option<String>,
    pub vault_purpose: Option<String>,
}

/// Memory layer feature flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub identity_enabled: bool,
    pub timeline_enabled: bool,
    pub mining_enabled: bool,
    pub mining_strategy: String,
    pub mining_on_index: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            identity_enabled: true,
            timeline_enabled: true,
            mining_enabled: true,
            mining_strategy: "auto".into(),
            mining_on_index: true,
        }
    }
}

/// Output packaging settings (#35).
///
/// Query-time settings; neither key reaches a fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Default token budget for `search`, overridden per call by `--tokens`.
    pub budget_tokens: u32,
    /// Whether the MCP result includes the text rendering beside the structured
    /// content. The CLI renders text unconditionally; HTTP returns JSON alone.
    pub emit_text_rendering: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            budget_tokens: 8192,
            emit_text_rendering: true,
        }
    }
}

/// HTTP REST API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpConfig {
    pub enabled: bool,
    pub port: u16,
    pub host: String,
    pub rate_limit: u32, // requests per minute per key, 0 = unlimited
    pub cors_origins: Vec<String>,
    pub api_keys: Vec<ApiKeyConfig>,
    #[serde(default)]
    pub plugin: PluginConfig,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 3000,
            host: "127.0.0.1".to_string(),
            rate_limit: 60,
            cors_origins: vec![],
            api_keys: vec![],
            plugin: PluginConfig::default(),
        }
    }
}

/// API key entry for HTTP authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    pub key: String,
    pub name: String,
    pub permissions: String, // "read" | "write"
}

/// Granularity of search results.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    clap::ValueEnum,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum GroupBy {
    /// One result per matching section — a document may appear more than once,
    /// bounded by `max_chunks_per_file`.
    #[default]
    Chunk,
    /// One result per document, represented by its best-scoring section. This is
    /// what knapper returned before sections were addressable.
    File,
}

/// Default ceiling on how many sections of one document a result set may hold.
///
/// Three is enough for a rules page to answer with the spell, its prerequisite,
/// and its counter, and few enough that a 33-section document cannot fill a page.
pub fn default_max_chunks_per_file() -> usize {
    3
}

/// Default ceiling on how much of one candidate the cross-encoder reads.
///
/// **Unlimited.** #25 fit a 1000-character cap on a CPU build, where it kept 82%
/// of the text and gave back 12.3% of query latency. #33 moved the cross-encoder
/// to CUDA and the trade collapsed: measured over the eighteen calibration
/// queries against one warm server, the cap now saves 3.6% — 18 ms on a 490 ms
/// query — and it truncates 14.8% of the corpus, because 333 of 1598 chunks are
/// longer than it (`eval/probes.md`, #42).
///
/// What it truncates is answers. Probe 7's answering sentence starts at
/// character 1038 of a 1061-character chunk and probe 2's best answer at
/// character 1098 of 1207, so the cross-encoder scored both chunks on their
/// subject alone and never read the line that answers the question.
///
/// The key stays for a CPU build, where #25's trade still holds.
pub fn default_max_document_chars() -> usize {
    0
}

/// Default shortest section body that becomes a chunk of its own.
///
/// **120 characters**, which is design §5.4's `chunk_min_tokens_est = 30` at the
/// chunker's own `chars / 4` estimate. It takes the eval corpus from 1598 chunks
/// to 1461 and its 72 rows under 60 characters to none — `## Threads\n_None
/// yet._` and the rest of the scaffolding a later workflow will fill. BM25
/// normalises by row length, so each of those rows scores enormously on any
/// query term it happens to carry.
///
/// Measured over the eighteen calibration queries the value costs nothing: every
/// negative holds or falls, every responsive set holds its coverage, and P2's
/// window shortens by two ranks (`eval/probes.md`, #43).
///
/// `0` is the control and reproduces the pre-#43 chunking exactly.
pub fn default_chunk_min_chars() -> usize {
    120
}

/// Whether a bold-only line opens a section (issue #44).
///
/// `true` ships. `false` is the control and reproduces the pre-#44 chunking
/// exactly.
pub fn default_promote_bold_headings() -> bool {
    true
}

/// Whether a bodyless heading whose line would otherwise be lost is carried
/// into a neighbouring chunk (issue #54).
///
/// A `#` heading with no body of its own is dropped when the next heading is
/// not strictly deeper — a same-level sibling, a shallower heading, or the end
/// of the file — because no descendant breadcrumb keeps it and #44's carry only
/// covers a promoted next line. The line then leaves the corpus. `true` carries
/// it: forward into the next section, or backward into the previous chunk when
/// nothing deeper follows.
///
/// `true` ships. `false` is the control and reproduces the pre-#54 chunking
/// exactly.
pub fn default_carry_orphan_headings() -> bool {
    true
}

/// How the rerank lane presents a candidate to the cross-encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// Prepend the document's title to the chunk before scoring it.
    ///
    /// This is not the experiment issue #2 lost: a prefix added to every chunk
    /// of a file moves that file's *vectors* together and costs within-document
    /// separation, whereas a cross-encoder scores each pair on its own and
    /// shares no space to flatten.
    ///
    /// It shipped off and unmeasured until the cross-encoder started deciding
    /// the order (#30), at which point the probes could see it: **probe 2's
    /// answer moves four ranks**, 8 to 4, and nothing else moves. The reason is
    /// visible in the candidate — `## Evolution\n- Previous: Medium Dragon\n-
    /// Next: Archdragon` is a section of `archdragon.md` that never says
    /// "archdragon" outside a link, so without its title the model is judging
    /// an unidentified fragment. The legacy stage is unmoved by it, which is
    /// the same measurement #32 makes: a voter's input barely reaches the
    /// output.
    ///
    /// The chunk's own heading is not included: the chunker already makes it
    /// the first line of the text.
    pub document_title: bool,

    /// Ceiling on how much of one candidate's text the cross-encoder reads.
    /// 0 means unlimited, and 0 is the default — see
    /// `default_max_document_chars()` for why the cap came off (#42).
    ///
    /// The cross-encoder is 85–96% of a query *on a CPU build*, and there its
    /// cost is very nearly
    /// linear in the tokens it is handed: measured over four queries of thirty
    /// candidates, `ctx.decode()` was 99.0–99.3% of the call and a least-squares
    /// fit put the fixed per-candidate term at *negative* 48 ms. So this is the
    /// one knob that bounds query latency, and bounding it in candidates —
    /// `rerank_candidates` — does not, since thirty candidates is anywhere
    /// between 4 s and 17 s depending which thirty.
    ///
    /// Characters rather than tokens because at this boundary the tokens do not
    /// exist yet and the tokenizer belongs to whichever `RerankModel` is loaded.
    /// Measured char/token ratio on real candidates is 3.16–3.43.
    ///
    /// Setting it is a `reranker_fingerprint` change, whose action is
    /// `InvalidateThresholds` — no re-index and no keyword-index rebuild, so a
    /// sweep of this key costs one config edit per arm.
    ///
    /// A cap also bounds a second cost: `n_ctx` and the batch are sized to the
    /// *longest* pair in the set (`llm.rs`), so one outlier inflates the
    /// allocation for every candidate, not only its own.
    pub max_document_chars: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            document_title: true,
            max_document_chars: default_max_document_chars(),
        }
    }
}

/// Calibrated score fusion for the model-free path
/// (docs/specs/2026-08-30-calibrated-fusion-design.md).
///
/// Read only when `[ranking] mode = "sorted"` runs with no cross-encoder
/// configured; with one configured the whole section is inert. Every key is
/// query-time and reaches no fingerprint, so a sweep is a config edit.
///
/// **The four numbers below are EmbeddingGemma's fit, not a global default.**
/// `bm25n` normalizes itself per query and per corpus, but raw cosine is one
/// model's similarity scale: at `bm25n = 0` the shipped floor asks for
/// `cos >= 0.4746`, a threshold read off that model's distribution. A
/// different `models.embed` — an API embedder or another `hf:` GGUF — moves
/// the scale in an unknown direction, and the failure is silent: the path
/// abstains on every query, or on none (#103). `config::section_trailer`
/// carries a measured fit for each other embedder `models list` offers, as a
/// commented block under the header in the generated file (#8).
///
/// The coefficients and the floor are **one fit** and do not move
/// independently. Scaling the weights down without moving the floor tightens
/// the threshold — at `semantic = 11` it becomes `cos >= 0.8964`, above every
/// positive the pin fit was built from, whose top cosines run 0.446 to 0.633.
/// Refit all four together with `scripts/calibrated-fusion-eval.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibratedConfig {
    /// The calibrated sort itself. `false` restores the pre-change routing —
    /// a no-model build takes the legacy stage — byte for byte. The control.
    pub enabled: bool,
    /// w_s: what raw cosine is worth to the logistic.
    pub semantic: f64,
    /// w_k: what upper-bound-normalized BM25 is worth.
    pub keyword: f64,
    /// b: the logistic's intercept.
    pub intercept: f64,
    /// The probability below which a candidate is not an answer — the
    /// `answer_floor` of this path. `0.0` removes nothing and is the floor's
    /// own control.
    pub floor: f64,
}

impl Default for CalibratedConfig {
    fn default() -> Self {
        // The pin fit: 33 tier-1 positives against 1228 labeled negatives,
        // leave-one-query-out validated. Provenance in
        // eval/calibrated-fusion-report-2026-08-30.txt; the fit's tool is
        // scripts/calibrated-fusion-eval.py.
        //
        // The three coefficients are the tool's `fused-raw` fit, which is the
        // variant `probability` computes: it multiplies the candidate's raw
        // cosine. The tool fits three other semantic features beside it, and a
        // fit taken from one of those measures a quantity this function never
        // forms (#8).
        Self {
            enabled: true,
            semantic: 20.777,
            keyword: 13.377,
            intercept: -8.762,
            floor: 0.75,
        }
    }
}

/// What leads a chunk's breadcrumb — the segment before the heading stack
/// (issue #46).
///
/// The breadcrumb is one string with two readers: the embedding limb puts it in
/// the document template's `title:` field, and the lexical limb stores it in
/// `chunks.heading_path`, which is a column the keyword index is declared over.
/// So this key decides what both of them see.
///
/// It is a component of the **chunker** digest, because `heading_path` is a
/// stored column: changing it rewrites every chunk row, which is
/// `Action::Reindex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreadcrumbRoot {
    /// The vault-relative path, extension included —
    /// `lore/bestiary/lesser-dragon.md > Stat Block`. The default.
    ///
    /// A breadcrumb trail that resolves: the first segment names a file on
    /// disk, so a breadcrumb quoted away from its result object is still
    /// actionable. FTS5's unicode61 tokeniser dissolves the separators, so the
    /// folders are matchable terms — which is how `memories/tandi/session-009.md`
    /// yields `tandi` with no frontmatter at all.
    #[default]
    Path,
    /// Frontmatter `name`, else the filename stem. What shipped before #46.
    ///
    /// **`name` is not knapper's key and not Obsidian's.** Obsidian gives
    /// meaning to `aliases`, `tags` and `cssclasses`; a note's title is its
    /// filename. `name:` is a cc-isekai convention that this engine reads and
    /// never writes, and it is a natural property key for a note *about* a
    /// person, so another vault's `name: Aragorn` would become that note's
    /// breadcrumb whatever the file is called. Kept as the control #46 was
    /// measured against, and for a vault that wants it.
    Name,
    /// The filename stem alone, no extension and no folders.
    ///
    /// **Not identifying.** 14 stems in the calibration vault are shared by more
    /// than one file, covering 36 of 259 — `session-002` names five different
    /// notes. Here for completeness, not for use.
    Stem,
}

/// What the keyword lane indexes beside a chunk's body, and how each part is
/// weighted (issue #37).
///
/// The breadcrumb rule of design §5.4 has three limbs that carry the same
/// string. #36 shipped the embedding limb, where the breadcrumb is averaged
/// into a vector. This is the lexical limb, where a heading term is *matched*.
///
/// Both flags are components of `fts_fingerprint`, because `chunks_fts` is
/// declared with one column per enabled flag and the fingerprint hashes that
/// declaration. A change to either therefore rebuilds the keyword index, which
/// reads no files and runs no model. The weights are query-time and reach no
/// fingerprint at all, so a weight sweep costs nothing.
///
/// `heading_path = false, tags = false` is the control: the declaration is then
/// the single body column, and BM25 returns the same scores it returned before
/// this issue. Measured: the whole eighteen-query pool is identical to the
/// pre-#37 binary, to nine decimal places. A zero *weight* is not the same
/// thing — BM25 normalises over the whole row's tokens, so a populated column
/// at weight 0.0 still moves every score, which is why the flags exist and a
/// weight of zero would not do.
///
/// The shipped values are measured rather than designed. `docs/vault-search-/// convergence.md` §6.2 asks for both columns at `bm25(chunks_fts, 1.0, 3.0,
/// 4.0)`; the arms in `eval/probes.md` (#37) give the breadcrumb at equal
/// weight and leave the tags out. See the two fields for what each one did.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FtsConfig {
    /// Index each chunk's breadcrumb — `Note Title > H1 > H2 > H3`, the same
    /// string `[embedding_prompt] document_title = "breadcrumb"` embeds.
    ///
    /// **On, and the reason this issue exists.** It returns `## Level 4
    /// Silence` — *"the silenced target cannot cast spells of any school"* — to
    /// probe 3 at rank 4, one of the two correct answers the embedding limb of
    /// the same rule dropped in #36, and it drops a summoning spell that
    /// answers nothing. It costs one swap below the answer on probe 4. No
    /// tracked answer moves.
    pub heading_path: bool,
    /// Index the file's frontmatter tags, sorted and space separated.
    ///
    /// **Off, and measured off.** A tag records an attribute of a note rather
    /// than something the note discusses, and the keyword lane cannot tell the
    /// two apart. `npcs/tandi.md` is tagged `velthos`, a city, so the adjacent
    /// negative N11 — *"in which city is Tandi's brother a blacksmith?"*, whose
    /// correct answer is nothing — gains a 97.06% result at rank 2. On probe 2
    /// it drops the one section that describes a dragon in human shape, at
    /// 94.94%, and the tracked answer's rank 4 → 3 is that drop and not a gain.
    ///
    /// This is half of #17 attempted the cheap way, and the half that did not
    /// work. Resolving a query against the tag *registry* is the other half and
    /// is untouched by this result.
    pub tags: bool,
    /// BM25 weight on the chunk body.
    pub body_weight: f64,
    /// BM25 weight on the breadcrumb. Ignored when `heading_path` is false.
    ///
    /// 1.0, because 3.0 and 5.0 were measured and bought nothing: probe 3's
    /// recovered answer arrives at rank 4 at all three, and the only other
    /// difference is churn — 62 of 360 result slots move at 1.0, 117 at 3.0,
    /// 123 at 5.0.
    pub heading_path_weight: f64,
    /// BM25 weight on the tags. Ignored when `tags` is false.
    pub tags_weight: f64,
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            heading_path: true,
            tags: false,
            body_weight: 1.0,
            heading_path_weight: 1.0,
            tags_weight: 1.0,
        }
    }
}

impl FtsConfig {
    /// The control: body only, at the weight a one-column table has anyway.
    pub const CONTROL: Self = Self {
        heading_path: false,
        tags: false,
        body_weight: 1.0,
        heading_path_weight: 1.0,
        tags_weight: 1.0,
    };

    /// The declared columns and their BM25 weights, in the order `chunks_fts`
    /// declares them. What `--explain` prints, so that a measurement cannot
    /// silently be taken at weights nobody meant.
    pub fn columns(&self) -> Vec<(&'static str, f64)> {
        let mut cols = vec![("text", self.body_weight)];
        if self.heading_path {
            cols.push(("heading_path", self.heading_path_weight));
        }
        if self.tags {
            cols.push(("tags_text", self.tags_weight));
        }
        cols
    }

    /// The BM25 weights alone, for the query.
    pub fn weights(&self) -> Vec<f64> {
        self.columns().iter().map(|(_, w)| *w).collect()
    }
}

/// Which ranking stage runs (issue #30).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RankingMode {
    /// Five lanes fused by weighted RRF, the cross-encoder among them as a
    /// voter. What knapper did before #30.
    ///
    /// Kept as the control the change is measured against: a switch that
    /// reproduces prior output byte-for-byte is what proves nothing incidental
    /// leaked into the shared retrieval code. `OFF_CHUNK_LINK_WEIGHT = 1.0`
    /// did the same job for #28 and stayed for the same reason.
    Legacy,
    /// Two content lanes fused, graph and temporal routed by reserved quota,
    /// and the cross-encoder sorts what reaches it.
    #[default]
    Sorted,
}

/// What decides between candidates the cross-encoder scored identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tiebreak {
    /// Fused rank first, then the chunk's identity.
    ///
    /// §8.6 says flatly not to reblend, and this is not reblending: a softmax
    /// over two token logits collides only on exact equality, which is rare
    /// enough to be measurable and is dominated by the degenerate case — a
    /// reranker that returned the same number for everything. Falling back to
    /// alphabetical order there would throw away the retrieval ordering for no
    /// reason.
    #[default]
    Rrf,
    /// The chunk's identity alone: pure cross-encoder ordering, with a
    /// deterministic fallback and nothing of fusion in it.
    Identity,
}

/// The ranking stage: what reaches the cross-encoder, and what it does there.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct RankingConfig {
    pub mode: RankingMode,
    /// How many rows each content lane retrieves, per expanded query.
    ///
    /// The semantic and keyword lanes each fetch this many rows for every
    /// expansion, and what they bring back is what `candidates` selects from.
    /// So this value decides *which* chunks the cross-encoder can ever see, and
    /// `top_n` decides only how many of the sorted results are returned.
    ///
    /// **The two were one number, and that was issue #49.** Each lane fetched
    /// `top_n * 3`, so a caller who asked for more results got a different pile
    /// of candidates and a different ranking: probe 3 held
    /// `## Level 4 Silence` at rank 3 and `## Level 6 Antimagic Shell` at rank
    /// 4 at `top_n = 20`, and neither at any rank at 25. The default of 60 is
    /// `20 * 3`, which is the width every table in `eval/probes.md` was
    /// measured at.
    ///
    /// The value is query-time and reaches no fingerprint, so a sweep is a
    /// config edit with no index work, no vault read and no model reload.
    pub retrieval_width: usize,
    /// How many candidates the cross-encoder is shown.
    ///
    /// This is the knob that sets query cost: the cross-encoder is 85–96% of a
    /// reranked query and its cost is very nearly linear in the text it reads,
    /// so `candidates × max_document_chars` is the budget and the two trade
    /// against each other.
    ///
    /// **§8.6 specifies 64 and it was measured and rejected.** At 64 the
    /// assembled input doubles — 18k characters per query to 37k — and the
    /// tracked targets do not move, except probe 2's answer which is *worse* by
    /// two ranks. Thirty is what the legacy stage showed the model, so the
    /// ranking change costs nothing at the stage that dominates the query.
    pub candidates: usize,
    /// Slots reserved for graph candidates in reach order.
    ///
    /// A routing guarantee, not a score bonus. The graph lane's fusion weight
    /// is gone; this replaces it, one stage later, where the shortlist is
    /// actually decided. Held at §8.6's ratio of the budget, 8 of 30.
    ///
    /// **Setting this to 0 leaves the five seed probes byte-identical.** The
    /// reserve admits 8–19 candidates no content lane found, the model scores
    /// every one of them below the content candidates, and none reaches the
    /// output. It ships anyway, for the reason #9 measured: a lane that cannot
    /// reach the model is a lane whose failures are invisible. What is bought
    /// here is that the graph's contribution is now a number that can be read
    /// off a log line rather than inferred from a ranking.
    ///
    /// **It is not free, because the slots come out of the content lanes'
    /// share.** At 8 they get 22, and 22 is a gate: the cross-encoder scores
    /// `rules/restoration-spells.md > ## Level 5 Purify Body` 0.98 against N10
    /// whenever it is shown it, so whether that negative abstains is decided by
    /// what competes for those 22 slots and not by the ranking (#58). Read a
    /// changed abstention here before reading it as a quality result.
    pub graph_reserve: usize,
    /// Slots reserved for date-matching candidates the content order cut.
    ///
    /// **Unmeasured** — no probe covers the temporal lane. See
    /// [`crate::ranking::Reserves::temporal`].
    pub temporal_reserve: usize,
    /// At most this many sections of one document may reach the model.
    ///
    /// Bound what the model is *shown*, because it cannot rank what it never
    /// saw; do not bound what it returns, because ranking is its job. The
    /// default of 3 comes from #6, where a 33-section document took 33 of the
    /// ranks its lane handed to RRF — under sorting there is no vote mechanic
    /// and that reason is gone.
    ///
    /// **Swept: 32 leaves every probe's ranking unchanged** while letting 9–108
    /// more chunks into the fused order, so the loose cap #30 argued for buys
    /// nothing measurable here. Three stays, as the cheaper of two settings the
    /// probes cannot tell apart.
    pub shortlist_cap: usize,
    pub tiebreak: Tiebreak,
    /// The score below which a candidate is not an answer (#34).
    ///
    /// The code applies the floor to each candidate after the sort. It skips a
    /// candidate that has no score — see
    /// [`crate::ranking::apply_answer_floor`]. A value of `0.0` removes nothing
    /// and is the control.
    ///
    /// The value comes from the pool in `eval/probes.md`. On the GPU baseline
    /// store the highest negative a floor can reject scores 5.33% on its best
    /// candidate (N6). The lowest score of a correct answer is 52.52%, at probe
    /// 1's `archivist-lenne.md`, rank 5. The default is the midpoint of those
    /// two values, 28.93%, and CPU and GPU scores differ by about 1 point.
    ///
    /// **Eight of the eleven negatives return nothing at this floor**, and so
    /// does the nonsense control. N4, N10 and N11 return results. N10 is above
    /// any usable floor because the cross-encoder scores a body-purifying spell
    /// 97.87% against a query about cleaning clothing — see
    /// [`RankingConfig::graph_reserve`] for the shortlist gate that used to keep
    /// that candidate out of the model's view.
    ///
    /// **What "a correct answer" means here is #34's reading and it is open.**
    /// The responsive sets of #45 put two more P1 members below 52.52%, at
    /// 23.50% and 1.33%. Read against those, no floor both keeps every
    /// responsive chunk and rejects anything. The pool section in
    /// `eval/probes.md` states both readings.
    ///
    /// #34 specifies a fit against the best score of each query, and the probes
    /// show that this is wrong. That fit gives 89%, the midpoint of 86.2% (N4)
    /// and 91.7% (P6). A floor of 89% removes probe 2's correct answer, which
    /// scores 81% below two better results from a different file. A floor that
    /// applies to each candidate must be below the lowest correct answer, and
    /// not below the best result of each query.
    ///
    /// Three negatives score above any usable floor, and the cause is the
    /// reranker. N11 asks which city Tandi's brother works in as a blacksmith.
    /// It scores 97.1% on a passage that names Tandi, a brother, a blacksmith
    /// and a city, but the brother is Mira's. N4 asks for a Precept who does not
    /// exist, and scores 86.2% on a section that gives the person who runs the
    /// location. N10 asks for a spell that cleans clothing and scores 97.87% on
    /// one that purifies a body. No floor can reject any of them and keep P6 and
    /// P7. The pool table records all three, and #58 measured
    /// Qwen3-Reranker-4B separating the N10 pair that the 0.6B inverts, so this
    /// is a property of the model and not of the class.
    pub answer_floor: f64,
    /// How many sections of one document can appear in the results. `0` means no
    /// limit.
    ///
    /// The default is no limit, which is the current behaviour and #30's
    /// position. Limit what the model reads with `shortlist_cap`, but do not
    /// limit what it returns: if one document holds the ten best sections, then
    /// ten sections is the correct answer. §9.1 of the vault-search design
    /// limits a note to three. The two positions do not combine.
    ///
    /// This key makes the decision a sweep instead of a code change. It waits
    /// for a probe where one document correctly holds the top of the results.
    /// Probe 4 is the candidate: `archdragon.md` holds ranks 1, 3 and 5, and all
    /// 20 of its results are above the floor.
    pub per_note_cap: usize,
    /// Present a section and its subsections, where they abut in one
    /// document, as one result block, after ranking (#39). Query-time: it
    /// reaches no fingerprint and re-indexes nothing. The block takes its
    /// strongest member's score, so abstention is unchanged. `false`
    /// reproduces the per-chunk output byte for byte.
    ///
    /// The merge stops at a sibling section (#101). It exists because a
    /// section is one topic, so a weaker follow-on chunk still carries that
    /// topic's context; a subsection subdivides the same topic and keeps the
    /// premise, while a sibling starts a new one and breaks it.
    pub coalesce_adjacent: bool,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            mode: RankingMode::default(),
            retrieval_width: default_retrieval_width(),
            candidates: 30,
            graph_reserve: 8,
            temporal_reserve: 4,
            shortlist_cap: default_max_chunks_per_file(),
            tiebreak: Tiebreak::default(),
            answer_floor: default_answer_floor(),
            per_note_cap: 0,
            coalesce_adjacent: true,
        }
    }
}

/// What each lane's rank is worth to the RRF fusion step.
///
/// The default vector is the best configuration the seventeen pool cells in the
/// #57 section of `eval/probes.md` measured. Every value is query-time and
/// reaches no fingerprint, so a sweep is a config edit with no index work, no
/// vault read and no model reload — which is what #9 and #19 both needed and
/// neither had.
///
/// Under the shipped ranking stage the cross-encoder sorts the shortlist, so
/// `semantic` and `fts` are the two weights that decide anything: `graph`,
/// `rerank` and `temporal` are read by [`RankingMode::Legacy`] alone, where all
/// five lanes vote. The graph lane reaches the shortlist by `graph_reserve`
/// instead.
///
/// `graph` sits at 0.8, below the content lanes, and that is deliberate (#9).
/// `graph_expand` skips any neighbour already in the seed set, so the graph
/// lane's results are disjoint from the other lanes' by construction. RRF scores
/// agreement between rankings of the same corpus; a disjoint set can never
/// accumulate any, so a graph result's fused score is a pure function of this
/// number. At 1.5 its 20 capped expansions swept the whole top 20, since
/// 1.5/(60+20) beats 0.8/(60+1) by 43%, and every content result was locked out
/// of every "who" query.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct LaneWeights {
    pub semantic: f64,
    pub fts: f64,
    pub graph: f64,
    pub rerank: f64,
    pub temporal: f64,
}

impl Default for LaneWeights {
    fn default() -> Self {
        Self {
            semantic: 1.0,
            fts: 1.0,
            graph: 0.8,
            rerank: 1.0,
            temporal: 0.0,
        }
    }
}

/// The lane width every table in `eval/probes.md` was measured at.
///
/// A function and not a literal in `Default`, so the number and the run that
/// produced it stay together: `top_n = 20` on the pre-#49 binary, where each
/// lane fetched `top_n * 3`.
pub fn default_retrieval_width() -> usize {
    60
}

/// The floor value from the pool fit: the midpoint of 6.77% and 52.52%.
///
/// This is a function and not a literal in `Default`, so the value and the fit
/// stay together. The pool table in `eval/probes.md` gives percentages, and this
/// is the same value as a probability.
pub fn default_answer_floor() -> f64 {
    0.30
}

/// Application configuration, loaded from `~/.knapper/config.toml` with CLI overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Path to the Obsidian vault to index.
    pub vault_path: Option<PathBuf>,
    /// Number of results to return from search.
    pub top_n: usize,
    /// Maximum number of sections one document may contribute to a result set.
    /// 0 means unlimited.
    pub max_chunks_per_file: usize,
    /// Whether search results address sections or whole documents.
    pub group_by: GroupBy,
    /// What document identity is prepended to a chunk's text before embedding.
    /// Affects the vector only — storage, snippets and FTS see the raw chunk.
    /// Changing this needs `knapper index --rebuild`; the incremental path
    /// compares content hashes and will not notice.
    #[serde(default)]
    pub embedding_prefix: PrefixConfig,
    /// Which prompt template each half of an asymmetric embedding model is fed
    /// through (issue #10). `document` is a fingerprint component, so changing
    /// it re-indexes on the next `knapper index`; `query` costs nothing.
    #[serde(default)]
    pub embedding_prompt: EmbeddingPromptConfig,
    /// Glob patterns to exclude from indexing.
    pub exclude: Vec<String>,
    /// Number of files to process per embedding batch.
    pub batch_size: usize,
    /// Honor `.gitignore` / `.ignore` files when walking the vault. Set false
    /// to index files those VCS rules would otherwise skip.
    pub respect_gitignore: bool,
    /// Whether intelligence features are enabled. None = not yet configured.
    pub intelligence: Option<bool>,
    /// Model override URIs.
    pub models: ModelConfig,
    /// HTTP REST API settings.
    #[serde(default)]
    pub http: HttpConfig,
    /// How the rerank lane is fed. Distinct from `models.rerank`, which only
    /// says which reranker to load.
    #[serde(default)]
    pub rerank: RerankConfig,
    /// What reaches the cross-encoder, and what it does there (issue #30).
    #[serde(default)]
    pub ranking: RankingConfig,
    /// What each lane's rank is worth to fusion (issue #59).
    #[serde(default)]
    pub lane_weights: LaneWeights,
    /// What the keyword lane indexes beside the chunk body (issue #37).
    #[serde(default)]
    pub fts: FtsConfig,
    /// Calibrated score fusion for the model-free path
    /// (docs/specs/2026-08-30-calibrated-fusion-design.md).
    #[serde(default)]
    pub calibrated: CalibratedConfig,
    /// What leads a chunk's breadcrumb, before the heading stack (issue #46).
    /// A chunker-digest component, so changing it re-indexes the vault.
    #[serde(default)]
    pub breadcrumb_root: BreadcrumbRoot,
    /// The shortest section body that becomes a chunk of its own (issue #43).
    /// A shorter one merges into the preceding chunk of the same file.
    ///
    /// The unit is characters, because the chunker's own size estimate is
    /// `chars / 4` — design §5.4's `chunk_min_tokens_est = 30` is 120 here. It
    /// is a key rather than a `chunker::limits` constant so that finding the
    /// right value is a config edit and not a recompile; like
    /// [`Config::breadcrumb_root`] it reaches the chunker digest, so changing
    /// it re-indexes the vault.
    ///
    /// The default is [`default_chunk_min_chars`]; `0` is no minimum, which is
    /// the pre-#43 chunking exactly.
    #[serde(default = "default_chunk_min_chars")]
    pub chunk_min_chars: usize,
    /// A line that is one bold span and nothing else opens a section, one level
    /// below the enclosing `#` heading (issue #44). Like [`Config::chunk_min_chars`]
    /// it reaches the chunker digest, so a change to it re-indexes the vault.
    #[serde(default = "default_promote_bold_headings")]
    pub promote_bold_headings: bool,
    /// Carry a bodyless heading's line into a neighbouring chunk rather than
    /// dropping it when no descendant breadcrumb or #44 carry keeps it (issue
    /// #54). Like [`Config::chunk_min_chars`] it reaches the chunker digest, so
    /// a change to it re-indexes the vault.
    #[serde(default = "default_carry_orphan_headings")]
    pub carry_orphan_headings: bool,
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub watcher: WatcherConfig,
}

/// Which filesystem-watch backend `knapper serve` uses to keep the index warm
/// (issue #83).
///
/// `Auto` uses native OS notifications — inotify on Linux, FSEvents on macOS —
/// unless the vault sits on a filesystem those cannot service: a Docker bind
/// mount, an overlay, 9p, fuse, or a network share. On such a filesystem the
/// native watcher registers without error and then delivers nothing, so
/// external edits go unseen until an explicit `index`. There `Auto` falls back
/// to interval polling. `Native` and `Poll` force one backend regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WatcherBackend {
    #[default]
    Auto,
    Native,
    Poll,
}

impl WatcherBackend {
    /// Parse the `KNAPPER_WATCHER_BACKEND` override — the container's way to
    /// select a backend with no config file. Unset or unrecognised reads as
    /// `None`, so the config value stands.
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "native" => Some(Self::Native),
            "poll" => Some(Self::Poll),
            _ => None,
        }
    }
}

/// `[watcher]` — the warm-sync file watcher (issue #83).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WatcherConfig {
    /// Which backend to run. See [`WatcherBackend`].
    pub backend: WatcherBackend,
    /// How often the poll backend restats the vault, in seconds. Ignored by the
    /// native backend.
    pub poll_interval_secs: u64,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            backend: WatcherBackend::default(),
            poll_interval_secs: 10,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vault_path: None,
            top_n: 5,
            max_chunks_per_file: default_max_chunks_per_file(),
            group_by: GroupBy::default(),
            breadcrumb_root: BreadcrumbRoot::default(),
            chunk_min_chars: default_chunk_min_chars(),
            promote_bold_headings: default_promote_bold_headings(),
            carry_orphan_headings: default_carry_orphan_headings(),
            embedding_prefix: PrefixConfig::default(),
            embedding_prompt: EmbeddingPromptConfig::default(),
            exclude: vec![".obsidian/".to_string()],
            batch_size: 64,
            respect_gitignore: true,
            intelligence: None,
            models: ModelConfig::default(),
            rerank: RerankConfig::default(),
            ranking: RankingConfig::default(),
            lane_weights: LaneWeights::default(),
            fts: FtsConfig::default(),
            calibrated: CalibratedConfig::default(),
            http: HttpConfig::default(),
            identity: IdentityConfig::default(),
            memory: MemoryConfig::default(),
            output: OutputConfig::default(),
            watcher: WatcherConfig::default(),
        }
    }
}

/// The store's file name before the rename.
const LEGACY_DB_NAME: &str = "engraph.db";

/// The store file inside `dir`: `knapper.db`. A directory holding only an
/// [`LEGACY_DB_NAME`] written before the rename keeps it, so no store
/// migrates.
pub fn db_path(dir: &Path) -> PathBuf {
    let new = dir.join("knapper.db");
    if new.exists() {
        return new;
    }
    let legacy = dir.join(LEGACY_DB_NAME);
    if legacy.exists() { legacy } else { new }
}

/// Override for the data directory, set once by `--data-dir` before any read.
static DATA_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Set the data-directory override from the top-level `--data-dir` flag.
///
/// First value wins; call once, before any [`Config::data_dir`] read. It sits
/// ahead of `KNAPPER_HOME` in resolution, so a flag beats the environment.
pub fn set_data_dir_override(dir: PathBuf) {
    let _ = DATA_DIR_OVERRIDE.set(dir);
}

/// Resolve the data directory from its three inputs, in order: the `--data-dir`
/// override, then `KNAPPER_HOME`, then `~/.knapper`.
///
/// The override and the environment value are used verbatim (not joined with
/// `.knapper`); an empty `KNAPPER_HOME` reads as unset. `home` is consulted only
/// when both are absent, so a container that sets `KNAPPER_HOME` needs no home
/// directory.
fn resolve_data_dir(
    override_dir: Option<PathBuf>,
    env_home: Option<OsString>,
    home: Option<PathBuf>,
) -> Result<PathBuf> {
    if let Some(dir) = override_dir {
        return Ok(dir);
    }
    if let Some(env) = env_home
        && !env.is_empty()
    {
        return Ok(PathBuf::from(env));
    }
    let home = home.context("could not determine home directory")?;
    Ok(home.join(".knapper"))
}

impl Config {
    /// Canonical data directory: `~/.knapper/`, or an override from `--data-dir`
    /// or `KNAPPER_HOME`.
    pub fn data_dir() -> Result<PathBuf> {
        resolve_data_dir(
            DATA_DIR_OVERRIDE.get().cloned(),
            std::env::var_os("KNAPPER_HOME"),
            dirs::home_dir(),
        )
    }

    /// Load config from `~/.knapper/config.toml`, falling back to defaults.
    pub fn load() -> Result<Self> {
        let config_path = Self::data_dir()?.join("config.toml");

        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?;
            let config: Config = toml::from_str(&contents)
                .with_context(|| format!("failed to parse {}", config_path.display()))?;
            config.validate_exclude(&config_path)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Reject `exclude` patterns that will not compile, at load time.
    ///
    /// Without this a typo becomes a glob that matches nothing, and the only
    /// symptom is files quietly staying in the index.
    fn validate_exclude(&self, source: &Path) -> Result<()> {
        crate::exclude::ExcludeMatcher::new(&self.exclude)
            .with_context(|| format!("in {}", source.display()))?;
        Ok(())
    }

    /// Merge CLI-provided values over the loaded config.
    pub fn merge_vault_path(&mut self, path: Option<PathBuf>) {
        if path.is_some() {
            self.vault_path = path;
        }
    }

    /// The chunker settings that are config keys, as one value.
    ///
    /// Every path that chunks a file takes this rather than the settings
    /// separately, so no path can carry one and forget the other.
    pub fn chunk_options(&self) -> crate::chunker::ChunkOptions {
        crate::chunker::ChunkOptions {
            min_chars: self.chunk_min_chars,
            promote_bold: self.promote_bold_headings,
            carry_orphan_headings: self.carry_orphan_headings,
        }
    }

    /// Put the chunker settings of `opts` back on this config.
    ///
    /// The inverse of [`Config::chunk_options`], and it lives beside it so that
    /// a further chunker key cannot be added to one and forgotten in the other. A
    /// long-running session captures `chunk_options()` once at startup, and a
    /// path that has to hand a whole `Config` to the indexer uses this to carry
    /// the session's settings rather than a fresh load's.
    pub fn set_chunk_options(&mut self, opts: crate::chunker::ChunkOptions) {
        self.chunk_min_chars = opts.min_chars;
        self.promote_bold_headings = opts.promote_bold;
        self.carry_orphan_headings = opts.carry_orphan_headings;
    }

    /// Put the embedding composition of `cfg` back on this config.
    ///
    /// The inverse of [`crate::prefix::EmbedComposition::from_config`], and it
    /// sits beside [`Config::set_chunk_options`] for the same reason: a
    /// long-running session captures the composition once at startup, and a
    /// path that hands a whole `Config` to the indexer uses this to carry the
    /// session's settings rather than a fresh load's. The three keys travel as
    /// one value, so no path can take one and forget another and write vectors
    /// into a space the store does not share (issues #2, #36, #46).
    pub fn set_embed_composition(&mut self, cfg: crate::prefix::EmbedComposition) {
        self.embedding_prefix = cfg.prefix;
        self.embedding_prompt.document_title = cfg.title;
        self.breadcrumb_root = cfg.root;
    }

    /// Merge CLI-provided top_n over the loaded config.
    pub fn merge_top_n(&mut self, n: Option<usize>) {
        if let Some(n) = n {
            self.top_n = n;
        }
    }

    /// Load vault profile from `~/.knapper/vault.toml`, if it exists.
    pub fn load_vault_profile() -> Result<Option<crate::profile::VaultProfile>> {
        let dir = Self::data_dir()?;
        crate::profile::load_vault_toml(&dir)
    }

    /// Whether intelligence is enabled (defaults to false if not configured).
    pub fn intelligence_enabled(&self) -> bool {
        self.intelligence.unwrap_or(false)
    }

    /// Whether `[calibrated]` holds the shipped fit while `models.embed` names
    /// an embedder that fit does not cover (#103).
    ///
    /// Three conditions, and all three have to hold before there is anything
    /// to say. The path has to run at all: with a cross-encoder configured, or
    /// with `enabled = false`, the section is inert and its numbers decide
    /// nothing. The embedder has to be one the fit does not cover, since
    /// cosine is one model's scale and the shipped coefficients read
    /// EmbeddingGemma's. And the numbers have to still be the shipped ones —
    /// a user who refit set their own, and telling them their own fit is stale
    /// would be false.
    ///
    /// `None` resolves through [`crate::llm::ModelDefaults`], the same value
    /// `load_embedder` resolves it through, so the unset case cannot drift
    /// from what actually loads.
    pub fn calibration_needs_refit(&self) -> bool {
        if self.intelligence_enabled() || !self.calibrated.enabled {
            return false;
        }
        if self.calibrated != CalibratedConfig::default() {
            return false;
        }
        let uri = self
            .models
            .embed
            .clone()
            .unwrap_or_else(|| crate::llm::ModelDefaults::default().embed_uri);
        !uri.to_lowercase().contains("embeddinggemma")
    }

    /// Save config to a specific path.
    ///
    /// The file is edited, not rewritten. A serialized `Config` holds every key
    /// the binary ships with, so writing one over the file pinned each of those
    /// values into the user's config: a later release that moved a default
    /// never reached them, and nothing told either party (#90). Two rules
    /// decide what a save writes:
    ///
    /// - a key whose value differs from its default, because that value is the
    ///   user's or a `configure` call's and the file is where it lives;
    /// - a key the file already holds, because the user wrote it there and an
    ///   explicit setting is not knapper's to remove — even where it happens to
    ///   equal today's default.
    ///
    /// Everything else stays out, so an unset key follows the binary. The rest
    /// of the file — its comments, its key order, its blank lines, and any key
    /// this build does not know — is left exactly as it stands. A key the
    /// config no longer holds at all is left too: nothing clears one today, and
    /// deleting on that rule would take a user's typo with it.
    ///
    /// A path with no file yet is given [`commented_defaults`], so the file on
    /// disk always shows what there is to set.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => commented_defaults()?,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let mut doc = text
            .parse::<DocumentMut>()
            .with_context(|| format!("parsing {}", path.display()))?;
        let full = as_document(self)?;
        let defaults = as_document(&Config::default())?;
        let mut position = next_position(doc.as_table());
        write_changed(
            full.as_table(),
            defaults.as_table(),
            doc.as_table_mut(),
            &mut position,
        );
        std::fs::write(path, doc.to_string())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Load config from a specific path.
    pub fn load_from(path: &Path) -> Result<Self> {
        let contents =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: Config =
            toml::from_str(&contents).with_context(|| format!("parsing {}", path.display()))?;
        config.validate_exclude(path)?;
        Ok(config)
    }

    /// Save to the default config path (`~/.knapper/config.toml`).
    pub fn save(&self) -> Result<()> {
        let path = Self::data_dir()?.join("config.toml");
        std::fs::create_dir_all(path.parent().unwrap())?;
        self.save_to(&path)
    }
}

/// The banner that introduces the commented catalogue.
///
/// It sits above the commented keys rather than at the top of the file: TOML
/// puts a table's own keys before any `[section]`, so a root key knapper writes
/// lands above whatever leads the file, and a banner that called itself the
/// header would be wrong the first time one did.
const CONFIG_BANNER: &str = "\
# Everything commented out below carries the value this build defaults to.
# Uncomment a line to set it. A key left commented follows the binary, so a
# release that moves a default moves it here too, and a key written here is
# yours and is kept. Section headers are live: uncommenting one key is enough.

";

/// The prose a generated config carries above one section.
///
/// The banner says what a commented key means; this says what a section's
/// values are *tied to*, which the values themselves cannot. Only a section
/// whose defaults stop being right when something outside them changes needs
/// one, so the map holds what it holds and no more.
fn section_note(path: &str) -> Option<&'static str> {
    match path {
        // The numbers are one embedder's fit, and nothing in the file says so
        // (#103). A user who changes `models.embed` keeps a floor read off a
        // scale that is gone, and the failure is quiet.
        // Both templates and the field they fill are one family's, and nothing
        // in the file says so (#8). Point `models.embed` at another family and
        // every key here stops doing anything, quietly.
        "embedding_prompt" => Some(
            "These templates are EmbeddingGemma's, the embedder this build\n\
             installs. The query and document templates are the two halves of\n\
             its documented pair, and document_title fills a title: field that\n\
             only its document template has.\n\
             \n\
             Point models.embed at another family and every key here stops\n\
             doing anything. Qwen3-Embedding, for one, takes its instruct on\n\
             the query alone and embeds a document as itself.",
        ),
        "calibrated" => Some(
            "These coefficients and this floor are fit against EmbeddingGemma,\n\
             the embedder this build installs. They are one fit: cosine is one\n\
             model's similarity scale, and the floor is a threshold read off\n\
             that scale.\n\
             \n\
             Point models.embed at a different embedder and it is yours to\n\
             refit all four numbers against it. Until you do, the floor cuts in\n\
             the wrong place, and it fails quietly: the path abstains on every\n\
             query, or on none.",
        ),
        _ => None,
    }
}

/// Commented lines written under a section's own keys.
///
/// A note goes above the header and a trailer below the keys, and the
/// difference decides what an uncommented line does. A key uncommented above
/// a header lands in whichever table precedes it, so anything a reader is
/// invited to switch on has to sit under the header it belongs to.
fn section_trailer(path: &str) -> Option<&'static str> {
    match path {
        // The shipped numbers are one embedder's, and a user who changes
        // `models.embed` is told to refit. For the two embedders `models list`
        // offers beside the default, the fit is already taken, so the file
        // carries it and the refit is a line to uncomment (#8).
        "calibrated" => Some(
            "Fits for the other embedders `knapper models list` offers, taken\n\
             on one corpus and one query pool with\n\
             scripts/calibrated-fusion-eval.py. Uncomment one block, and set\n\
             models.embed to the model that block names. The four numbers of a\n\
             block move together. Two live blocks set one key twice, which is a\n\
             TOML error.\n\
             \n\
             Qwen3-Embedding-0.6B\n\
             hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf\n\
             semantic = 17.197\n\
             keyword = 12.181\n\
             intercept = -9.057\n\
             floor = 0.78\n\
             \n\
             Qwen3-Embedding-4B\n\
             hf:Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf\n\
             semantic = 11.618\n\
             keyword = 11.298\n\
             intercept = -6.419\n\
             floor = 0.76",
        ),
        _ => None,
    }
}

/// Write one comment block, one `# ` per line.
///
/// A bare `#` for a blank line: a trailing space is invisible in the file and
/// visible in every diff of it.
fn push_comment_block(out: &mut String, text: &str) {
    for line in text.lines() {
        match line.trim_start() {
            "" => out.push_str("#\n"),
            body => out.push_str(&format!("# {body}\n")),
        }
    }
}

/// A config file holding every default, commented out.
///
/// The section headers stay live and only the key lines are commented, so
/// uncommenting one line sets that key in the table it sits under. Commenting
/// the headers too would put an uncommented key in whichever table precedes it
/// — valid TOML, silently the wrong setting.
fn commented_defaults() -> Result<String> {
    let body = toml::to_string_pretty(&Config::default()).context("serializing defaults")?;
    let mut out = String::with_capacity(CONFIG_BANNER.len() + body.len() * 2);
    out.push_str(CONFIG_BANNER);
    let mut emitted: Vec<String> = Vec::new();
    // A trailer belongs under its own section's keys, so it is held until the
    // next header arrives — and the blank lines that separate two sections are
    // held with it, or the trailer would print against the next header instead
    // of the keys it describes.
    let mut open_section = String::new();
    let mut blanks = 0usize;
    for line in body.lines() {
        if let Some(path) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            if let Some(trailer) = section_trailer(&open_section) {
                push_comment_block(&mut out, trailer);
            }
            out.push_str(&"\n".repeat(blanks));
            blanks = 0;
            // `to_string_pretty` writes `[models.embed_api]` and no `[models]`,
            // because `models` holds no key of its own. Write the parent too:
            // a table the file already names is one no save has to create, and
            // creating one moves the comments that sat where it lands.
            let mut prefix = String::new();
            for segment in path.split('.') {
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(segment);
                if prefix != path && !emitted.iter().any(|e| e == &prefix) {
                    out.push_str(&format!("[{prefix}]\n\n"));
                    emitted.push(prefix.clone());
                }
            }
            emitted.push(path.to_string());
            if let Some(note) = section_note(path) {
                push_comment_block(&mut out, note);
            }
            out.push_str(line);
            out.push('\n');
            open_section = path.to_string();
        } else if line.is_empty() {
            blanks += 1;
        } else {
            out.push_str(&"\n".repeat(blanks));
            blanks = 0;
            out.push_str("# ");
            out.push_str(line);
            out.push('\n');
        }
    }
    if let Some(trailer) = section_trailer(&open_section) {
        push_comment_block(&mut out, trailer);
    }
    out.push_str(&"\n".repeat(blanks));
    Ok(out)
}

/// One config as a document of `[section]` tables.
///
/// Through `to_string_pretty` rather than `toml_edit::ser::to_document`, which
/// renders a nested struct as an inline table — `ranking = { mode = "sorted",
/// … }` — and would leave [`write_changed`] no table to descend into.
fn as_document<T: Serialize>(value: &T) -> Result<DocumentMut> {
    toml::to_string_pretty(value)
        .context("serializing config")?
        .parse::<DocumentMut>()
        .context("re-reading serialized config")
}

/// Write into `doc` every key of `full` that differs from `defaults`, and every
/// key `doc` already holds, leaving the rest of the document alone (#90).
///
/// `full` and `defaults` come from one serializer through [`as_document`], so
/// two equal values render to one string and `to_string` compares them.
fn write_changed(full: &Table, defaults: &Table, doc: &mut Table, next_position: &mut usize) {
    for (key, item) in full.iter() {
        let default_item = defaults.get(key);
        let Some(full_child) = item.as_table() else {
            let differs = default_item.is_none_or(|d| d.to_string() != item.to_string());
            if differs || doc.contains_key(key) {
                set_value(doc, key, item);
            }
            continue;
        };
        let empty = Table::new();
        let default_child = default_item.and_then(Item::as_table).unwrap_or(&empty);
        match doc.get_mut(key).map(Item::as_table_mut) {
            // The document holds the table: descend and leave its own text be.
            Some(Some(doc_child)) => {
                write_changed(full_child, default_child, doc_child, next_position)
            }
            // The document holds something else under this key. `load` reads
            // the file before any save, so it has already refused anything the
            // config cannot parse; leave it rather than overwrite it.
            Some(None) => {}
            // No table yet — add one only if a key inside it has to be written,
            // and put it past every table the document already places. Without
            // a position of its own a new table renders in key order, which can
            // put it between a block of comments and the table they introduce.
            None => {
                let mut fresh = Table::new();
                write_changed(full_child, default_child, &mut fresh, next_position);
                if !fresh.is_empty() {
                    fresh.set_implicit(false);
                    fresh.set_position(*next_position);
                    *next_position += 1;
                    doc.insert(key, Item::Table(fresh));
                }
            }
        }
    }
}

/// One past the last document position any table in `table` claims.
fn next_position(table: &Table) -> usize {
    table
        .iter()
        .filter_map(|(_, item)| item.as_table())
        .map(|t| t.position().map_or(0, |p| p + 1).max(next_position(t)))
        .max()
        .unwrap_or(0)
}

/// Set one key, keeping the formatting the document already gave it.
///
/// A key the document holds is written through the item that is already there,
/// so the key itself is never replaced. Replacing it drops the key's decor —
/// which is where the parser puts the comment lines above it, and the spacing
/// its author chose — and a save would quietly eat the note the user wrote to
/// explain their own setting.
fn set_value(doc: &mut Table, key: &str, item: &Item) {
    let Some(existing) = doc.get_mut(key) else {
        let mut item = item.clone();
        if let Some(value) = item.as_value_mut() {
            value.decor_mut().set_prefix(" ");
            value.decor_mut().set_suffix("");
        }
        doc.insert(key, item);
        return;
    };
    match (item.as_value(), existing.as_value()) {
        (Some(new_value), Some(old_value)) => {
            let mut new_value = new_value.clone();
            *new_value.decor_mut() = old_value.decor().clone();
            *existing = Item::Value(new_value);
        }
        _ => *existing = item.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{DocumentTemplate, QueryTemplate};

    #[test]
    fn default_config_has_sane_values() {
        let cfg = Config::default();
        assert_eq!(cfg.top_n, 5);
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.exclude, vec![".obsidian/"]);
        assert!(cfg.vault_path.is_none());
    }

    /// Issues #25 and #42. The cap ships off, because on a CUDA build it saves
    /// 3.6% of query time and truncates 14.8% of the corpus — including the
    /// sentences that answer two of the calibration probes. It stays settable
    /// for a CPU build, where #25's trade still holds.
    #[test]
    fn the_rerank_character_cap_defaults_off_and_a_cap_is_opt_in() {
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.rerank.max_document_chars, 0, "0 means unlimited");
        assert!(
            bare.rerank.document_title,
            "the candidate has to name the document it came from once the \
             cross-encoder decides the order (#30)"
        );

        let titled: Config = toml::from_str("[rerank]\ndocument_title = true\n").unwrap();
        assert_eq!(
            titled.rerank.max_document_chars, 0,
            "naming one key in the section must not silently cap the other"
        );

        let capped: Config = toml::from_str("[rerank]\nmax_document_chars = 1000\n").unwrap();
        assert_eq!(capped.rerank.max_document_chars, 1000);
    }

    #[test]
    fn data_dir_ends_with_knapper() {
        let dir = Config::data_dir().unwrap();
        assert!(dir.ends_with(".knapper"));
    }

    #[test]
    fn resolve_prefers_override_over_env_and_home() {
        let got = resolve_data_dir(
            Some(PathBuf::from("/flag/dir")),
            Some(OsString::from("/env/dir")),
            Some(PathBuf::from("/home/user")),
        )
        .unwrap();
        assert_eq!(got, PathBuf::from("/flag/dir"));
    }

    #[test]
    fn resolve_uses_env_verbatim_not_joined() {
        let got = resolve_data_dir(
            None,
            Some(OsString::from("/data")),
            Some(PathBuf::from("/home/user")),
        )
        .unwrap();
        assert_eq!(got, PathBuf::from("/data"));
    }

    #[test]
    fn resolve_empty_env_falls_back_to_default() {
        let got = resolve_data_dir(
            None,
            Some(OsString::from("")),
            Some(PathBuf::from("/home/user")),
        )
        .unwrap();
        assert_eq!(got, PathBuf::from("/home/user/.knapper"));
    }

    #[test]
    fn resolve_defaults_to_home_dot_knapper() {
        let got = resolve_data_dir(None, None, Some(PathBuf::from("/home/user"))).unwrap();
        assert_eq!(got, PathBuf::from("/home/user/.knapper"));
    }

    #[test]
    fn resolve_env_needs_no_home() {
        // A container sets KNAPPER_HOME and has no home directory.
        let got = resolve_data_dir(None, Some(OsString::from("/data")), None).unwrap();
        assert_eq!(got, PathBuf::from("/data"));
    }

    #[test]
    fn resolve_errors_when_default_and_no_home() {
        assert!(resolve_data_dir(None, None, None).is_err());
    }

    #[test]
    fn db_path_prefers_the_new_name_and_falls_back_to_a_legacy_store() {
        let dir = tempfile::tempdir().unwrap();
        // A fresh directory gets the new name.
        assert!(db_path(dir.path()).ends_with("knapper.db"));
        // A directory holding only a pre-rename store keeps it.
        std::fs::write(dir.path().join(LEGACY_DB_NAME), b"x").unwrap();
        assert!(db_path(dir.path()).ends_with(LEGACY_DB_NAME));
        // The new name wins when both exist.
        std::fs::write(dir.path().join("knapper.db"), b"x").unwrap();
        assert!(db_path(dir.path()).ends_with("knapper.db"));
    }

    #[test]
    fn parse_config_toml() {
        let toml_str = r#"
vault_path = "/tmp/vault"
top_n = 10
exclude = ["*.canvas", ".obsidian"]
batch_size = 128
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.vault_path.unwrap(), PathBuf::from("/tmp/vault"));
        assert_eq!(cfg.top_n, 10);
        assert_eq!(cfg.exclude, vec!["*.canvas", ".obsidian"]);
        assert_eq!(cfg.batch_size, 128);
    }

    #[test]
    fn load_rejects_invalid_exclude_pattern() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "exclude = [\"[unclosed.md\"]\n").unwrap();

        let err = Config::load_from(&path).unwrap_err();
        let chain = format!("{err:#}");
        assert!(
            chain.contains("invalid exclude pattern"),
            "unexpected error: {chain}"
        );
    }

    #[test]
    fn load_accepts_glob_exclude_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "exclude = [\"*-index.md\", \"templates/\"]\n").unwrap();

        let cfg = Config::load_from(&path).unwrap();
        assert_eq!(cfg.exclude, vec!["*-index.md", "templates/"]);
    }

    #[test]
    fn retrieval_granularity_round_trips() {
        let cfg: Config = toml::from_str("max_chunks_per_file = 5\ngroup_by = \"file\"\n").unwrap();
        assert_eq!(cfg.max_chunks_per_file, 5);
        assert_eq!(cfg.group_by, GroupBy::File);

        // Chunk-level results with a cap of 3 are what a config without them means.
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.max_chunks_per_file, 3);
        assert_eq!(bare.group_by, GroupBy::Chunk);
    }

    #[test]
    fn embedding_prefix_round_trips_and_defaults_off() {
        let cfg: Config =
            toml::from_str("[embedding_prefix]\nenabled = true\ntags = false\n").unwrap();
        assert!(cfg.embedding_prefix.enabled);
        assert!(cfg.embedding_prefix.aliases);
        assert!(cfg.embedding_prefix.heading);
        assert!(!cfg.embedding_prefix.tags);

        // Off unless asked for: it regressed the seed probes on the eval vault
        // (eval/probes.md, "Contextual embedding prefix (#2)").
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.embedding_prefix, PrefixConfig::default());
        assert!(!bare.embedding_prefix.enabled);

        // Turning it on without naming components gives every component.
        let on: Config = toml::from_str("[embedding_prefix]\nenabled = true\n").unwrap();
        assert_eq!(on.embedding_prefix, PrefixConfig::full());
    }

    #[test]
    fn embedding_prompt_round_trips_and_each_half_is_separate() {
        let cfg: Config =
            toml::from_str("[embedding_prompt]\ndocument = \"documented\"\nquery = \"legacy\"\n")
                .unwrap();
        assert_eq!(cfg.embedding_prompt.document, DocumentTemplate::Documented);
        assert_eq!(cfg.embedding_prompt.query, QueryTemplate::Legacy);

        // Naming one half leaves the other alone. The two carry very different
        // costs — the document half re-indexes the vault (issue #10).
        let half: Config = toml::from_str("[embedding_prompt]\ndocument = \"legacy\"\n").unwrap();
        assert_eq!(half.embedding_prompt.document, DocumentTemplate::Legacy);
        assert_eq!(half.embedding_prompt.query, QueryTemplate::Documented);

        // A config that names neither gets the model card's own pair.
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.embedding_prompt, EmbeddingPromptConfig::default());
        assert_eq!(bare.embedding_prompt.document, DocumentTemplate::Documented);
        assert_eq!(bare.embedding_prompt.query, QueryTemplate::Documented);
    }

    #[test]
    fn document_title_round_trips_and_defaults_to_none() {
        use crate::llm::DocumentTitle;

        let cfg: Config =
            toml::from_str("[embedding_prompt]\ndocument_title = \"none\"\n").unwrap();
        assert_eq!(cfg.embedding_prompt.document_title, DocumentTitle::None);
        // Naming the third key leaves the other two at the model card's pair.
        assert_eq!(cfg.embedding_prompt.document, DocumentTemplate::Documented);

        let note: Config =
            toml::from_str("[embedding_prompt]\ndocument_title = \"note\"\n").unwrap();
        assert_eq!(note.embedding_prompt.document_title, DocumentTitle::Note);

        // The design's breadcrumb has to be spelled out (§5.4). It ships in the
        // lexical lane instead — #37 indexes the same string, and #38 measured
        // this limb as a loss beside it.
        let breadcrumb: Config =
            toml::from_str("[embedding_prompt]\ndocument_title = \"breadcrumb\"\n").unwrap();
        assert_eq!(
            breadcrumb.embedding_prompt.document_title,
            DocumentTitle::Breadcrumb
        );

        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.embedding_prompt.document_title, DocumentTitle::None);
    }

    #[test]
    fn retrieval_width_is_settable_and_defaults_to_the_measured_value() {
        // 60 is the width every table in `eval/probes.md` was taken at (#49).
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.ranking.retrieval_width, 60);

        let swept: Config = toml::from_str("[ranking]\nretrieval_width = 120\n").unwrap();
        assert_eq!(swept.ranking.retrieval_width, 120);
        assert_eq!(swept.ranking.candidates, 30, "the other keys keep defaults");
    }

    #[test]
    fn the_chunk_minimum_ships_at_the_measured_value_and_zero_is_the_control() {
        // 120 characters is design §5.4's 30 tokens at the chunker's own
        // `chars / 4`. 0 is the control: it reproduces the pre-#43 index.
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.chunk_min_chars, 120);

        let control: Config = toml::from_str("chunk_min_chars = 0\n").unwrap();
        assert_eq!(control.chunk_min_chars, 0);
    }

    #[test]
    fn promotion_ships_on_and_false_is_the_control_and_it_travels_with_the_minimum() {
        // `true` ships. `false` is the control: it reproduces the pre-#44
        // index.
        let bare: Config = toml::from_str("").unwrap();
        assert!(bare.promote_bold_headings);
        assert_eq!(bare.chunk_options().min_chars, 120);
        assert!(bare.chunk_options().promote_bold);
    }

    #[test]
    fn orphan_carry_ships_on_and_false_is_the_control() {
        // `true` ships. `false` is the control: it reproduces the pre-#54
        // index, and it reaches `chunk_options` so every chunking path sees it.
        let bare: Config = toml::from_str("").unwrap();
        assert!(bare.carry_orphan_headings);
        assert!(bare.chunk_options().carry_orphan_headings);

        let control: Config = toml::from_str("carry_orphan_headings = false\n").unwrap();
        assert!(!control.carry_orphan_headings);
        assert!(!control.chunk_options().carry_orphan_headings);
    }

    #[test]
    fn output_defaults_are_8192_and_text_on() {
        let c = OutputConfig::default();
        assert_eq!(c.budget_tokens, 8192);
        assert!(c.emit_text_rendering);
    }

    #[test]
    fn output_section_parses() {
        let c: Config = toml::from_str("[output]\nbudget_tokens = 4096\n").unwrap();
        assert_eq!(c.output.budget_tokens, 4096);
        assert!(c.output.emit_text_rendering); // serde(default) fills the rest
    }

    #[test]
    fn the_chunker_settings_travel_back_onto_a_config_whole() {
        // A session captures `chunk_options()` once and puts it back on a
        // freshly loaded config before it hands it to the indexer, so a load
        // that failed cannot re-chunk one file at the defaults.
        let session = crate::chunker::ChunkOptions {
            min_chars: 0,
            promote_bold: false,
            carry_orphan_headings: false,
        };
        let mut fresh = Config::default();
        assert_ne!(fresh.chunk_options(), session);
        fresh.set_chunk_options(session);
        assert_eq!(fresh.chunk_options(), session);
    }

    #[test]
    fn the_embedding_composition_travels_back_onto_a_config_whole() {
        // The twin of `the_chunker_settings_travel_back_onto_a_config_whole`
        // for the embedding side: a session captures the composition once and
        // puts it back on a freshly loaded config, so a load that failed cannot
        // re-embed one file at the defaults and into a space the store does not
        // share. All three keys travel as one value (#2, #36, #46, #72).
        let session = crate::prefix::EmbedComposition {
            prefix: crate::prefix::PrefixConfig::default(),
            title: crate::llm::DocumentTitle::Breadcrumb,
            root: BreadcrumbRoot::Name,
        };
        let mut fresh = Config::default();
        assert_ne!(
            crate::prefix::EmbedComposition::from_config(&fresh),
            session
        );
        fresh.set_embed_composition(session);
        assert_eq!(
            crate::prefix::EmbedComposition::from_config(&fresh),
            session
        );
    }

    #[test]
    fn parse_partial_config_uses_defaults() {
        let toml_str = r#"top_n = 20"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.top_n, 20);
        assert_eq!(cfg.batch_size, 64); // default
        assert!(cfg.vault_path.is_none());
    }

    #[test]
    fn merge_overrides_when_present() {
        let mut cfg = Config::default();
        cfg.merge_vault_path(Some(PathBuf::from("/my/vault")));
        cfg.merge_top_n(Some(42));
        assert_eq!(cfg.vault_path.unwrap(), PathBuf::from("/my/vault"));
        assert_eq!(cfg.top_n, 42);
    }

    #[test]
    fn merge_preserves_when_none() {
        let mut cfg = Config {
            top_n: 10,
            ..Config::default()
        };
        cfg.merge_top_n(None);
        assert_eq!(cfg.top_n, 10);
    }

    #[test]
    fn load_from_nonexistent_file_returns_defaults() {
        // Config::load() reads from ~/.knapper/config.toml.
        // If it doesn't exist, defaults are fine. We test the parsing path
        // separately above. This just ensures load() doesn't panic.
        let cfg = Config::load().unwrap();
        assert_eq!(cfg.batch_size, 64);
    }

    #[test]
    fn parse_intelligence_config() {
        let toml_str = r#"
intelligence = true

[models]
embed = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
rerank = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.intelligence, Some(true));
        assert!(cfg.models.embed.is_some());
        assert!(cfg.models.rerank.is_some());
    }

    #[test]
    fn the_lane_weights_are_the_measured_vector_and_each_is_settable() {
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.lane_weights.semantic, 1.0);
        assert_eq!(bare.lane_weights.fts, 1.0);
        assert_eq!(bare.lane_weights.graph, 0.8);
        assert_eq!(bare.lane_weights.rerank, 1.0);
        assert_eq!(bare.lane_weights.temporal, 0.0);

        // Naming one weight leaves the rest at the measured vector, so a sweep
        // states what it changed rather than restating the whole table.
        let swept: Config = toml::from_str("[lane_weights]\nsemantic = 1.2\nfts = 0.8\n").unwrap();
        assert_eq!(swept.lane_weights.semantic, 1.2);
        assert_eq!(swept.lane_weights.fts, 0.8);
        assert_eq!(swept.lane_weights.graph, 0.8);
    }

    /// The graph lane must not outweigh a content lane by default.
    ///
    /// Not a style rule — issue #9. `graph_expand` excludes seed files, so graph
    /// results share no documents with the semantic or FTS lanes and can never
    /// gain an agreement term in RRF. Their fused score is therefore just
    /// `weight/(60+rank)`, and once that number is large enough for the lane's
    /// *worst* result to beat a content lane's *best*, the graph lane takes the
    /// entire ranking. With expansions capped at 20 and `k = 60`, the crossover
    /// is at `graph/80 > content/61`, a ratio of about 1.31.
    ///
    /// A sweep can set whatever it likes; this holds the shipped vector.
    #[test]
    fn the_graph_lane_never_outweighs_a_content_lane() {
        let w = LaneWeights::default();
        let content = w.semantic.max(w.fts);
        assert!(
            w.graph <= content,
            "graph {} exceeds the strongest content lane {content} — disjoint graph \
             results would crowd the ranking out (#9)",
            w.graph,
        );
    }

    #[test]
    fn intelligence_defaults_to_none() {
        let cfg = Config::default();
        assert!(cfg.intelligence.is_none());
        assert!(cfg.models.embed.is_none());
    }

    #[test]
    fn intelligence_false_disables_features() {
        let toml_str = r#"intelligence = false"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.intelligence, Some(false));
        assert!(!cfg.intelligence_enabled());
    }

    #[test]
    fn test_config_backward_compat() {
        // Old format: intelligence = true at top level
        let toml = r#"intelligence = true"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.intelligence, Some(true));
    }

    #[test]
    fn watcher_backend_parses_the_env_override() {
        assert_eq!(
            WatcherBackend::from_env_value("poll"),
            Some(WatcherBackend::Poll)
        );
        assert_eq!(
            WatcherBackend::from_env_value("NATIVE"),
            Some(WatcherBackend::Native)
        );
        assert_eq!(
            WatcherBackend::from_env_value("  Auto  "),
            Some(WatcherBackend::Auto)
        );
        assert_eq!(WatcherBackend::from_env_value("nonsense"), None);
        assert_eq!(WatcherBackend::from_env_value(""), None);
    }

    #[test]
    fn watcher_config_defaults_to_auto_with_a_ten_second_poll() {
        let w = WatcherConfig::default();
        assert_eq!(w.backend, WatcherBackend::Auto);
        assert_eq!(w.poll_interval_secs, 10);
    }

    #[test]
    fn a_config_without_a_watcher_section_defaults_it() {
        let cfg: Config = toml::from_str("intelligence = true").unwrap();
        assert_eq!(cfg.watcher.backend, WatcherBackend::Auto);
        assert_eq!(cfg.watcher.poll_interval_secs, 10);
    }

    #[test]
    fn a_watcher_section_deserializes() {
        let cfg: Config =
            toml::from_str("[watcher]\nbackend = \"poll\"\npoll_interval_secs = 3\n").unwrap();
        assert_eq!(cfg.watcher.backend, WatcherBackend::Poll);
        assert_eq!(cfg.watcher.poll_interval_secs, 3);
    }

    #[test]
    fn test_config_with_http() {
        let toml = r#"
[http]
enabled = true
port = 8080
host = "0.0.0.0"
rate_limit = 120
cors_origins = ["https://chat.openai.com"]

[[http.api_keys]]
key = "eg_test123"
name = "test-key"
permissions = "read"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert!(config.http.enabled);
        assert_eq!(config.http.port, 8080);
        assert_eq!(config.http.api_keys.len(), 1);
        assert_eq!(config.http.api_keys[0].permissions, "read");
    }

    #[test]
    fn test_config_http_defaults() {
        let toml = r#"top_n = 5"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert!(!config.http.enabled);
        assert_eq!(config.http.port, 3000);
        assert_eq!(config.http.host, "127.0.0.1");
        assert_eq!(config.http.rate_limit, 60);
        assert!(config.http.cors_origins.is_empty());
        assert!(config.http.api_keys.is_empty());
    }

    #[test]
    fn test_config_roundtrip_with_intelligence() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");

        let mut cfg = Config {
            intelligence: Some(true),
            ..Config::default()
        };
        cfg.models.embed = Some("hf:custom/model/embed.gguf".into());

        cfg.save_to(&config_path).unwrap();

        let loaded = Config::load_from(&config_path).unwrap();
        assert_eq!(loaded.intelligence, Some(true));
        assert_eq!(
            loaded.models.embed,
            Some("hf:custom/model/embed.gguf".into())
        );
    }

    #[test]
    fn test_config_with_plugin() {
        let toml = r#"
[http.plugin]
name = "my-vault"
public_url = "https://vault.example.com"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.http.plugin.name.as_deref(), Some("my-vault"));
    }

    #[test]
    fn test_identity_config_deserializes() {
        let toml_str = r#"
[identity]
name = "Test User"
role = "Developer"
vault_purpose = "notes"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.identity.name, Some("Test User".into()));
        assert_eq!(config.identity.role, Some("Developer".into()));
        assert_eq!(config.identity.vault_purpose, Some("notes".into()));
    }

    #[test]
    fn test_identity_config_defaults_to_empty() {
        let config = Config::default();
        assert!(config.identity.name.is_none());
        assert!(config.identity.role.is_none());
    }

    #[test]
    fn test_memory_config_defaults() {
        let config = Config::default();
        assert!(config.memory.identity_enabled);
        assert!(config.memory.timeline_enabled);
        assert!(config.memory.mining_enabled);
    }

    #[test]
    fn coalesce_adjacent_defaults_on() {
        assert!(RankingConfig::default().coalesce_adjacent);
    }

    #[test]
    fn coalesce_adjacent_reads_false_and_omission_stays_on() {
        let off: Config = toml::from_str("[ranking]\ncoalesce_adjacent = false\n").unwrap();
        assert!(!off.ranking.coalesce_adjacent);

        let omitted: Config = toml::from_str("[ranking]\nanswer_floor = 0.0\n").unwrap();
        assert!(
            omitted.ranking.coalesce_adjacent,
            "a [ranking] table that omits the key keeps the default on"
        );
    }

    #[test]
    fn embed_api_section_round_trips_and_defaults() {
        // Absent section -> defaults.
        let cfg: Config = toml::from_str("").unwrap();
        assert_eq!(cfg.models.embed_api.timeout_secs, 30);
        assert_eq!(cfg.models.embed_api.max_retries, 4);
        assert!(cfg.models.embed_api.dim.is_none());

        // Present section parses.
        let cfg: Config =
            toml::from_str("[models.embed_api]\ndim = 1536\ntimeout_secs = 10\nmax_retries = 2\n")
                .unwrap();
        assert_eq!(cfg.models.embed_api.dim, Some(1536));
        assert_eq!(cfg.models.embed_api.timeout_secs, 10);
        assert_eq!(cfg.models.embed_api.max_retries, 2);
    }

    #[test]
    fn calibrated_defaults_are_the_pin_fit() {
        let c = CalibratedConfig::default();
        assert!(c.enabled);
        assert_eq!(c.semantic, 20.777);
        assert_eq!(c.keyword, 13.377);
        assert_eq!(c.intercept, -8.762);
        assert_eq!(c.floor, 0.75);
    }

    #[test]
    fn calibrated_reads_partial_tables_and_omission_keeps_the_defaults() {
        let off: Config = toml::from_str("[calibrated]\nenabled = false\n").unwrap();
        assert!(!off.calibrated.enabled);
        assert_eq!(
            off.calibrated.floor, 0.75,
            "an omitted key keeps its default"
        );

        let omitted: Config = toml::from_str("[ranking]\nanswer_floor = 0.0\n").unwrap();
        assert!(
            omitted.calibrated.enabled,
            "a config with no [calibrated] table ships enabled"
        );
    }

    /// Whether `table` names a value anywhere. A live `[section]` header with
    /// nothing under it sets nothing, and the catalogue is made of those.
    fn sets_any_value(table: &toml::Table) -> bool {
        table.values().any(|v| match v {
            toml::Value::Table(t) => sets_any_value(t),
            _ => true,
        })
    }

    /// A serialized `Config` holds every key the binary ships with. Writing one
    /// over the file pinned each of those values into the user's config, so a
    /// later release that moved a default never reached them (#90). A value
    /// equal to its default is not the user's, and is not written.
    #[test]
    fn a_value_at_its_default_is_not_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        Config::default().save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let live: toml::Table = toml::from_str(&text).unwrap();
        assert!(
            !sets_any_value(&live),
            "a default config sets nothing, so the file names no value: {live:?}"
        );
        assert_eq!(
            Config::load_from(&path).unwrap().top_n,
            Config::default().top_n
        );
    }

    /// The other half of the rule: a key the file already holds is the user's,
    /// and stays theirs even where it happens to equal today's default (#90).
    #[test]
    fn a_key_the_file_holds_is_kept_at_its_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                "top_n = {}\n\n[calibrated]\nfloor = {}\n",
                Config::default().top_n,
                Config::default().calibrated.floor
            ),
        )
        .unwrap();

        let mut cfg = Config::load_from(&path).unwrap();
        cfg.intelligence = Some(true);
        cfg.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("top_n = 5"), "{text}");
        assert!(text.contains("floor = 0.75"), "{text}");
        assert!(text.contains("intelligence = true"), "{text}");
    }

    /// The file is edited, not rewritten, so everything the save does not name
    /// survives: the user's comments, their key order and spacing, and a key
    /// this build does not know — deleting that last one would take a typo with
    /// it, and a typo is better left where its author can see it (#90).
    #[test]
    fn a_save_keeps_the_text_it_did_not_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "\
# my vault, my rules
top_n    =    30
some_key_from_a_later_build = 7

[ranking]
# tuned against my own pool
answer_floor = 0.5
";
        std::fs::write(&path, original).unwrap();

        let mut cfg = Config::load_from(&path).unwrap();
        cfg.models.embed = Some("hf:custom/e.gguf".into());
        cfg.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# my vault, my rules"), "{text}");
        assert!(text.contains("# tuned against my own pool"), "{text}");
        assert!(
            text.contains("top_n    =    30"),
            "spacing is the user's: {text}"
        );
        assert!(text.contains("some_key_from_a_later_build = 7"), "{text}");
        assert!(text.contains("answer_floor = 0.5"), "{text}");

        let back = Config::load_from(&path).unwrap();
        assert_eq!(back.top_n, 30);
        assert_eq!(back.ranking.answer_floor, 0.5);
        assert_eq!(back.models.embed.as_deref(), Some("hf:custom/e.gguf"));
    }

    /// A path with no file is given the commented catalogue, so the file on
    /// disk shows what there is to set without setting any of it (#90).
    #[test]
    fn a_generated_file_comments_every_default_under_a_live_header() {
        let text = commented_defaults().unwrap();

        let live: toml::Table = toml::from_str(&text).unwrap();
        assert!(
            !sets_any_value(&live),
            "nothing in the catalogue is set: {live:?}"
        );

        let mut headers: Vec<&str> = Vec::new();
        for line in text.lines() {
            if let Some(path) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                // A table whose parent the file never names is one a save has
                // to create, and creating it moves the comments where it lands.
                if let Some((parent, _)) = path.rsplit_once('.') {
                    assert!(headers.contains(&parent), "no header for {parent}: {text}");
                }
                headers.push(path);
            } else {
                assert!(
                    line.is_empty() || line.starts_with('#'),
                    "an uncommented key line: {line}"
                );
            }
        }
        assert!(headers.contains(&"models"), "{text}");
        assert!(headers.contains(&"calibrated"), "{text}");
    }

    /// The shipped `[calibrated]` numbers are EmbeddingGemma's fit, so an
    /// index that now embeds with something else puts a different cosine scale
    /// under a floor read off that one (#103). Three things have to be true at
    /// once before there is anything to tell the user, and each of them alone
    /// is a reason to stay quiet.
    #[test]
    fn a_refit_is_needed_only_where_the_shipped_fit_decides_something() {
        let mut cfg = Config::default();
        assert!(
            !cfg.calibration_needs_refit(),
            "the default install runs the embedder the fit covers"
        );

        cfg.models.embed = Some("gemini:gemini-embedding-2".into());
        assert!(
            cfg.calibration_needs_refit(),
            "a hosted embedder is not the one the numbers were fit against"
        );

        // A cross-encoder sorts instead, so the section is inert.
        cfg.intelligence = Some(true);
        assert!(!cfg.calibration_needs_refit());
        cfg.intelligence = Some(false);
        assert!(cfg.calibration_needs_refit());

        // The legacy stage runs instead, so the numbers decide nothing.
        cfg.calibrated.enabled = false;
        assert!(!cfg.calibration_needs_refit());
        cfg.calibrated.enabled = true;
        assert!(cfg.calibration_needs_refit());

        // Their own fit is not stale, whatever it says.
        cfg.calibrated.semantic = 8.0;
        assert!(
            !cfg.calibration_needs_refit(),
            "a user who refit set these; calling their numbers stale is false"
        );
        cfg.calibrated = CalibratedConfig::default();
        assert!(cfg.calibration_needs_refit());

        // Naming the shipped model explicitly is the same as leaving it unset.
        cfg.models.embed =
            Some("hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".into());
        assert!(!cfg.calibration_needs_refit());

        cfg.models.embed = Some("hf:someone/other-model/other-Q8_0.gguf".into());
        assert!(
            cfg.calibration_needs_refit(),
            "another local GGUF is another cosine scale"
        );
    }

    /// The unset case resolves through the same default the loader reads, so
    /// the two cannot drift apart.
    #[test]
    fn the_unset_embedder_is_the_one_the_fit_covers() {
        assert!(
            crate::llm::ModelDefaults::default()
                .embed_uri
                .to_lowercase()
                .contains("embeddinggemma")
        );
    }

    /// The `[calibrated]` numbers are one embedder's fit, and the file that
    /// carries them says so above the header they sit under (#103). Without it
    /// a user who changes `models.embed` keeps a floor fit to a scale that is
    /// gone, and nothing in their config tells them.
    #[test]
    fn the_calibrated_section_carries_the_embedder_its_fit_belongs_to() {
        let text = commented_defaults().unwrap();
        let (before, _) = text.split_once("[calibrated]").expect("a header: {text}");
        let note = before
            .rsplit("\n\n")
            .next()
            .expect("text before the header");

        assert!(
            note.contains("EmbeddingGemma"),
            "the note names the embedder: {note}"
        );
        assert!(
            note.contains("refit"),
            "the note says what a different embedder costs: {note}"
        );
        assert!(
            note.lines().all(|l| l.starts_with('#')),
            "every note line is a comment: {note}"
        );
    }

    /// A fit for another embedder is only useful where uncommenting it sets
    /// the key. Above the header the same line sets it in the table before,
    /// which is the trap the note/trailer split exists for (#8).
    #[test]
    fn the_calibrated_section_carries_a_fit_for_every_catalogued_embedder() {
        let text = commented_defaults().unwrap();
        let (_, after) = text.split_once("[calibrated]").expect("a header: {text}");
        let section = after.split("\n[").next().expect("text after the header");

        for model in ["Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B"] {
            assert!(
                section.contains(model),
                "a fit for {model} sits under the header: {section}"
            );
        }
        assert!(
            section.matches("# semantic = ").count() == 3,
            "the shipped fit and one per catalogued alternative: {section}"
        );
        assert!(
            section.lines().all(|l| l.is_empty() || l.starts_with('#')),
            "every line under the header is commented: {section}"
        );
    }

    /// `[embedding_prompt]` is one family's templates, and the file has to say
    /// so (#8). Both templates and the `title:` field they fill are
    /// EmbeddingGemma's; point `models.embed` at Qwen3-Embedding and every key
    /// in the section stops doing anything, with nothing in the output to show
    /// it.
    #[test]
    fn the_embedding_prompt_section_names_the_family_it_belongs_to() {
        let text = commented_defaults().unwrap();
        let (before, _) = text
            .split_once("[embedding_prompt]")
            .expect("a header: {text}");
        let note = before
            .rsplit("\n\n")
            .next()
            .expect("text before the header");

        assert!(
            note.contains("EmbeddingGemma"),
            "the note names the family: {note}"
        );
        assert!(
            note.lines().all(|l| l.starts_with('#')),
            "every note line is a comment: {note}"
        );
    }

    /// An array of tables is one value, so `http.api_keys` survives the round
    /// trip whole and a second save writes the same file as the first.
    #[test]
    fn a_save_is_stable_over_an_array_of_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let mut cfg = Config::default();
        cfg.http.api_keys.push(ApiKeyConfig {
            key: "kn_x".into(),
            name: "chatgpt".into(),
            permissions: "read".into(),
        });
        cfg.save_to(&path).unwrap();
        let once = std::fs::read_to_string(&path).unwrap();

        let back = Config::load_from(&path).unwrap();
        assert_eq!(back.http.api_keys.len(), 1);
        assert_eq!(back.http.api_keys[0].name, "chatgpt");

        back.save_to(&path).unwrap();
        assert_eq!(
            once,
            std::fs::read_to_string(&path).unwrap(),
            "a save of what was loaded rewrites nothing"
        );
    }
}
