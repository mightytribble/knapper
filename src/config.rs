use anyhow::{Context, Result};
use rmcp::schemars;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    /// its own `// TODO: better default`, and engraph used to inherit it —
    /// so every model call ran on four threads of whatever box it was on
    /// (issue #20).
    ///
    /// Set this to override: to leave headroom for other work, or because a
    /// sweep found a better number for the machine (12 beat 8 on the box above).
    /// Threads change only how the arithmetic is scheduled, never its result —
    /// see [`crate::llm::resolve_n_threads`].
    pub n_threads: Option<usize>,
}

/// Obsidian integration configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObsidianConfig {
    #[serde(default)]
    pub enabled: bool,
    pub vault_name: Option<String>,
    pub cli_path: Option<PathBuf>,
}

/// Agent integration configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentsConfig {
    #[serde(default)]
    pub claude_code: bool,
    #[serde(default)]
    pub cursor: bool,
    #[serde(default)]
    pub windsurf: bool,
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
    /// what engraph returned before sections were addressable.
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
    /// **`name` is not engraph's key and not Obsidian's.** Obsidian gives
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
    /// voter. What engraph did before #30.
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

/// Application configuration, loaded from `~/.engraph/config.toml` with CLI overrides.
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
    /// Changing this needs `engraph index --rebuild`; the incremental path
    /// compares content hashes and will not notice.
    #[serde(default)]
    pub embedding_prefix: PrefixConfig,
    /// Which prompt template each half of an asymmetric embedding model is fed
    /// through (issue #10). `document` is a fingerprint component, so changing
    /// it re-indexes on the next `engraph index`; `query` costs nothing.
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
    /// Obsidian integration settings.
    #[serde(default)]
    pub obsidian: ObsidianConfig,
    /// Agent integration settings.
    #[serde(default)]
    pub agents: AgentsConfig,
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
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub output: OutputConfig,
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
            obsidian: ObsidianConfig::default(),
            agents: AgentsConfig::default(),
            http: HttpConfig::default(),
            identity: IdentityConfig::default(),
            memory: MemoryConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Config {
    /// Canonical data directory: `~/.engraph/`.
    pub fn data_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("could not determine home directory")?;
        Ok(home.join(".engraph"))
    }

    /// Load config from `~/.engraph/config.toml`, falling back to defaults.
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

    /// The two chunker settings that are config keys, as one value.
    ///
    /// Every path that chunks a file takes this rather than the settings
    /// separately, so no path can carry one and forget the other.
    pub fn chunk_options(&self) -> crate::chunker::ChunkOptions {
        crate::chunker::ChunkOptions {
            min_chars: self.chunk_min_chars,
            promote_bold: self.promote_bold_headings,
        }
    }

    /// Put the chunker settings of `opts` back on this config.
    ///
    /// The inverse of [`Config::chunk_options`], and it lives beside it so that
    /// a third chunker key cannot be added to one and forgotten in the other. A
    /// long-running session captures `chunk_options()` once at startup, and a
    /// path that has to hand a whole `Config` to the indexer uses this to carry
    /// the session's settings rather than a fresh load's.
    pub fn set_chunk_options(&mut self, opts: crate::chunker::ChunkOptions) {
        self.chunk_min_chars = opts.min_chars;
        self.promote_bold_headings = opts.promote_bold;
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

    /// Load vault profile from `~/.engraph/vault.toml`, if it exists.
    pub fn load_vault_profile() -> Result<Option<crate::profile::VaultProfile>> {
        let dir = Self::data_dir()?;
        crate::profile::load_vault_toml(&dir)
    }

    /// Whether intelligence is enabled (defaults to false if not configured).
    pub fn intelligence_enabled(&self) -> bool {
        self.intelligence.unwrap_or(false)
    }

    /// Save config to a specific path.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))?;
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

    /// Save to the default config path (`~/.engraph/config.toml`).
    pub fn save(&self) -> Result<()> {
        let path = Self::data_dir()?.join("config.toml");
        std::fs::create_dir_all(path.parent().unwrap())?;
        self.save_to(&path)
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
    fn data_dir_ends_with_engraph() {
        let dir = Config::data_dir().unwrap();
        assert!(dir.ends_with(".engraph"));
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
        };
        let mut fresh = Config::default();
        assert_ne!(fresh.chunk_options(), session);
        fresh.set_chunk_options(session);
        assert_eq!(fresh.chunk_options(), session);
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
        // Config::load() reads from ~/.engraph/config.toml.
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
        // New fields default to None/false
        assert!(!config.obsidian.enabled);
    }

    #[test]
    fn test_config_with_obsidian() {
        let toml = r#"
intelligence = true
[obsidian]
enabled = true
vault_name = "Personal"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert!(config.obsidian.enabled);
        assert_eq!(config.obsidian.vault_name.as_deref(), Some("Personal"));
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
}
