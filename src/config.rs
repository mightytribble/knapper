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

/// How the rerank lane presents a candidate to the cross-encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// Prepend the document's title to the chunk before scoring it.
    ///
    /// Off, and unmeasured. This is not the experiment issue #2 lost: a prefix
    /// added to every chunk of a file moves that file's *vectors* together and
    /// costs within-document separation, whereas a cross-encoder scores each
    /// pair on its own and shares no space to flatten. But "different failure
    /// mode" is not "known to help", and the five seed probes cannot tell —
    /// #12 changed 76 of 100 result slots without moving a single probe
    /// verdict. So this waits on the probe battery in #3 and ships as a switch.
    ///
    /// The chunk's own heading is not included: the chunker already makes it
    /// the first line of the text.
    pub document_title: bool,
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
