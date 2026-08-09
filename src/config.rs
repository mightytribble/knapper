use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::prefix::PrefixConfig;

/// Model override configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Override embedding model URI (e.g., "hf:repo/file.gguf").
    pub embed: Option<String>,
    /// Override reranker model URI.
    pub rerank: Option<String>,
    /// Override expansion/orchestrator model URI.
    pub expand: Option<String>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, clap::ValueEnum)]
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
/// Picked from the sweep in `eval/probes.md` (#25), three rounds over the five
/// seed probes: 1000 characters keeps 82% of the text, gives back 12% of query
/// latency, and moves no probe down — two move up. 600 gives back 26% but costs
/// probe 4 a rank, which is the guard #15 and #25 both name.
pub fn default_max_document_chars() -> usize {
    1000
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
    /// 0 means unlimited, which is what shipped before issue #25.
    ///
    /// Defaults to `default_max_document_chars()`, which is measured rather
    /// than chosen — see there.
    ///
    /// The cross-encoder is 85–96% of a query, and its cost is very nearly
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
    /// The cross-encoder score below which a candidate is not an answer (#34).
    ///
    /// Applied per candidate after the sort, and skipped wherever there is no
    /// probability to threshold — see [`crate::ranking::apply_answer_floor`].
    /// `0.0` disables it and is the inert control.
    ///
    /// **Fit against the 17-query calibration pool in `eval/probes.md`**, not
    /// chosen. On the GPU baseline store nine of the eleven verified negatives
    /// score below **6.8%** on their best candidate, and the lowest score at
    /// which any positive's tracked answer sits is **52.5%** — probe 1's
    /// `archivist-lenne.md` at rank 5. The default is the midpoint of that gap,
    /// with ~23 points of margin either side against ~1 point of CPU/GPU kernel
    /// disagreement.
    ///
    /// **The ticket said to fit on best-score-per-query and the probes overruled
    /// it.** That fit gives 89% — the midpoint of 86.2% (N4) to 91.7% (P6) — and
    /// 89% cuts probe 2's tracked answer, which sits at 81% behind two better
    /// results from a different file. A gate applied per candidate has to clear
    /// the weakest *answer*, not the strongest *result*, and those are different
    /// distributions; the ticket named the risk and the measurement found it.
    ///
    /// **Two negatives sit above any usable floor and that is a finding about
    /// the reranker, not about this number.** N11 asks which city Tandi's
    /// brother smiths in and scores 97.1% on a passage naming Tandi, a brother,
    /// a blacksmith and a city with the sibling bound to the wrong person; N4
    /// asks after a Precept who does not exist and scores 86.2% on a section
    /// saying who runs the place. Neither is separable from P6 and P7 by score.
    /// The pool table records both.
    pub answer_floor: f64,
    /// At most this many sections of one document may appear in the results.
    /// `0` is unbounded.
    ///
    /// **Ships unbounded, which is today's behaviour and #30's position:** bound
    /// what the model is *shown* (`shortlist_cap`), not what it returns, because
    /// if a document holds the ten best sections then ten sections is the right
    /// answer. §9.1 of the vault-search design caps a note at three and the two
    /// are not reconcilable by splitting the difference.
    ///
    /// The key exists so the deferred decision is a one-key sweep rather than a
    /// code change. It waits on a probe where one document legitimately owns the
    /// top of the ranking; probe 4 is the candidate, where `archdragon.md` holds
    /// ranks 1, 3 and 5 and every one of its twenty results clears the floor.
    pub per_note_cap: usize,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            mode: RankingMode::default(),
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

/// The fitted abstention floor: the midpoint of the pool's only constrained gap.
///
/// Kept as a function rather than a literal in `Default` so the number and the
/// fit that produced it live in one place — the pool table in `eval/probes.md`
/// is denominated in percent and this is the same number as a probability.
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
    /// Changing this needs `engraph index --reindex`; the incremental path
    /// compares content hashes and will not notice.
    #[serde(default)]
    pub embedding_prefix: PrefixConfig,
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
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vault_path: None,
            top_n: 5,
            max_chunks_per_file: default_max_chunks_per_file(),
            group_by: GroupBy::default(),
            embedding_prefix: PrefixConfig::default(),
            exclude: vec![".obsidian/".to_string()],
            batch_size: 64,
            respect_gitignore: true,
            intelligence: None,
            models: ModelConfig::default(),
            rerank: RerankConfig::default(),
            ranking: RankingConfig::default(),
            obsidian: ObsidianConfig::default(),
            agents: AgentsConfig::default(),
            http: HttpConfig::default(),
            identity: IdentityConfig::default(),
            memory: MemoryConfig::default(),
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

    #[test]
    fn default_config_has_sane_values() {
        let cfg = Config::default();
        assert_eq!(cfg.top_n, 5);
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.exclude, vec![".obsidian/"]);
        assert!(cfg.vault_path.is_none());
    }

    /// Issue #25. The cap is the only knob that bounds query latency, so a
    /// config that never mentions `[rerank]` still has to get one — and an
    /// explicit `0` still has to mean unlimited.
    #[test]
    fn the_rerank_character_cap_defaults_on_and_zero_means_unlimited() {
        let bare: Config = toml::from_str("").unwrap();
        assert_eq!(bare.rerank.max_document_chars, 1000);
        assert!(
            bare.rerank.document_title,
            "the candidate has to name the document it came from once the \
             cross-encoder decides the order (#30)"
        );

        let titled: Config = toml::from_str("[rerank]\ndocument_title = true\n").unwrap();
        assert_eq!(
            titled.rerank.max_document_chars, 1000,
            "naming one key in the section must not silently unlimit the other"
        );

        let unlimited: Config = toml::from_str("[rerank]\nmax_document_chars = 0\n").unwrap();
        assert_eq!(unlimited.rerank.max_document_chars, 0);
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
        let mut cfg = Config::default();
        cfg.top_n = 10;
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
        assert!(cfg.models.expand.is_none());
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

        let mut cfg = Config::default();
        cfg.intelligence = Some(true);
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
