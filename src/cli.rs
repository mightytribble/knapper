//! What the CLI accepts. The definitions live in the library and not in the
//! binary so that `surface.rs`'s parity test can walk them under
//! `cargo test --lib` (#62). The dispatch stays in `main.rs`.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "engraph",
    version,
    about = "Local semantic search for Obsidian vaults"
)]
pub struct Cli {
    /// Output results as JSON.
    #[arg(long, global = true)]
    pub json: bool,

    /// Enable verbose logging.
    #[arg(long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Index a vault directory for semantic search.
    Index {
        /// Path to the vault (overrides config).
        path: Option<PathBuf>,

        /// Rebuild the index from scratch.
        #[arg(long)]
        rebuild: bool,

        /// Index files that `.gitignore` / `.ignore` would normally exclude.
        #[arg(long)]
        no_gitignore: bool,
    },

    /// Search the indexed vault.
    Search {
        /// The search query.
        query: String,

        /// Number of results to return.
        #[arg(short = 'n', long)]
        top_n: Option<usize>,

        /// Show per-lane RRF score breakdown for each result.
        #[arg(long, conflicts_with = "json")]
        explain: bool,

        /// Return one result per matching section, or one per document.
        #[arg(long, value_enum)]
        group_by: Option<crate::config::GroupBy>,

        /// Filter to notes with all listed tags (comma-separated). A trailing
        /// `/` or `/*` matches the tag and its descendants.
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Filter to notes carrying every term (comma-separated). A trailing
        /// `/` or `/*` matches the tag and its descendants.
        #[arg(long, value_delimiter = ',')]
        all: Vec<String>,
        /// Filter to notes carrying at least one term (comma-separated). An
        /// unknown term is an error naming the nearest tag the vault holds.
        #[arg(long, value_delimiter = ',')]
        any: Vec<String>,
        /// Filter out notes carrying any of these terms (comma-separated).
        /// An unknown term here is ignored.
        #[arg(long, value_delimiter = ',')]
        none: Vec<String>,
    },

    /// Show index status and statistics.
    Status,

    /// Clear cached data.
    Clear {
        /// Remove everything including the database and embeddings.
        #[arg(long)]
        all: bool,
    },

    /// Initialize vault profile, identity, and search index.
    Init {
        /// Path to vault directory.
        path: Option<PathBuf>,
        /// `detect` inspects the vault and writes nothing; `apply` configures
        /// identity and indexes. Omit it to run the interactive flow, which
        /// is the one thing the other surfaces cannot do (#62).
        #[arg(long)]
        mode: Option<String>,
        /// Only run identity setup (skip indexing).
        #[arg(long)]
        identity: bool,
        /// Only re-index (skip identity prompts).
        #[arg(long)]
        reindex: bool,
        /// Detect vault without writing anything (agent mode).
        #[arg(long)]
        detect: bool,
        /// Output as JSON (agent mode).
        #[arg(long)]
        json: bool,
        /// Suppress interactive prompts.
        #[arg(long)]
        quiet: bool,
        /// User name (non-interactive mode).
        #[arg(long)]
        name: Option<String>,
        /// User role (non-interactive mode).
        #[arg(long)]
        role: Option<String>,
        /// Vault purpose (non-interactive mode).
        #[arg(long)]
        purpose: Option<String>,
    },

    /// Print identity block (L0 + L1 context for AI agents).
    Identity {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
        /// Force L1 re-extraction without full reindex.
        #[arg(long)]
        refresh: bool,
    },

    /// Configure engraph settings.
    Configure {
        /// Enable intelligence features.
        #[arg(long, conflicts_with = "disable_intelligence")]
        enable_intelligence: bool,

        /// Disable intelligence features.
        #[arg(long, conflicts_with = "enable_intelligence")]
        disable_intelligence: bool,

        /// Override a model: --model embed|rerank <uri>
        #[arg(long, num_args = 2, value_names = &["TYPE", "URI"])]
        model: Option<Vec<String>>,

        /// Enable Obsidian CLI integration.
        #[arg(long, conflicts_with = "disable_obsidian_cli")]
        enable_obsidian_cli: bool,

        /// Disable Obsidian CLI integration.
        #[arg(long, conflicts_with = "enable_obsidian_cli")]
        disable_obsidian_cli: bool,

        /// Register with an AI agent: "claude-code", "cursor", or "windsurf".
        #[arg(long)]
        register: Option<String>,

        /// Generate and add a new API key.
        #[arg(long)]
        add_api_key: bool,

        /// Name for the new API key (requires --add-api-key).
        #[arg(long, requires = "add_api_key")]
        key_name: Option<String>,

        /// Permissions for the new key: "read" or "write" (requires --add-api-key).
        #[arg(long, requires = "add_api_key")]
        key_permissions: Option<String>,

        /// List all API keys.
        #[arg(long)]
        list_api_keys: bool,

        /// Revoke an API key by name.
        #[arg(long)]
        revoke_api_key: Option<String>,

        /// Interactive setup for ChatGPT Actions integration.
        #[arg(long)]
        setup_chatgpt: bool,
    },

    /// Manage embedding models.
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Start MCP stdio server for AI agent access.
    Serve {
        /// Enable HTTP REST API alongside MCP.
        #[arg(long)]
        http: bool,
        /// HTTP port (default: from config or 3000).
        #[arg(long)]
        port: Option<u16>,
        /// HTTP host to bind to (default: 127.0.0.1).
        #[arg(long)]
        host: Option<String>,
        /// Disable API key authentication (local development only, 127.0.0.1 only).
        #[arg(long)]
        no_auth: bool,
        /// Read-only mode: only expose search and read MCP tools, disable all write operations.
        #[arg(long)]
        read_only: bool,
    },

    /// Query vault context.
    Context {
        #[command(subcommand)]
        action: ContextAction,
    },

    /// Write a note to the vault.
    Write {
        #[command(subcommand)]
        action: WriteAction,
    },

    /// Migrate vault structure into PARA.
    Migrate {
        /// `preview` classifies every note and saves the proposed moves;
        /// `apply` performs them; `undo` reverses the last migration. PARA
        /// is the one strategy, so it is not a leaf of its own (#62).
        #[arg(long)]
        mode: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum ContextAction {
    /// Read a note's full content with metadata.
    Read {
        /// File path, basename, or #docid.
        file: String,
    },
    /// List notes by metadata filters.
    List {
        /// Filter to folder path prefix.
        #[arg(long)]
        folder: Option<String>,
        /// Filter to notes with all listed tags (comma-separated). A trailing
        /// `/` or `/*` matches the tag and its descendants.
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Filter to notes carrying every term (comma-separated). A trailing
        /// `/` or `/*` matches the tag and its descendants.
        #[arg(long, value_delimiter = ',')]
        all: Vec<String>,
        /// Filter to notes carrying at least one term (comma-separated). An
        /// unknown term is an error naming the nearest tag the vault holds.
        #[arg(long, value_delimiter = ',')]
        any: Vec<String>,
        /// Filter out notes carrying any of these terms (comma-separated).
        /// An unknown term here is ignored.
        #[arg(long, value_delimiter = ',')]
        none: Vec<String>,
        /// Filter to notes created by a specific agent.
        #[arg(long)]
        created_by: Option<String>,
        /// Maximum results.
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// List the vault's tag vocabulary.
    Tags {
        /// Limit to one tag and its descendants, as `type/` or `type/*`.
        #[arg(long)]
        under: Option<String>,
    },
    /// Vault structure overview.
    VaultMap,
    /// Person context bundle.
    Who {
        /// Person name (matches filename in People folder).
        name: String,
    },
    /// Project context bundle.
    Project {
        /// Project name (matches filename).
        name: String,
    },
    /// Rich topic context with budget.
    Topic {
        /// Search query for the topic.
        query: String,
        /// Character budget (default 32000, ~8000 tokens).
        #[arg(long, default_value = "32000")]
        budget: usize,
    },
}

#[derive(Subcommand, Debug)]
pub enum WriteAction {
    /// Create a new note.
    Create {
        /// Note content (reads from stdin if omitted).
        #[arg(long)]
        content: Option<String>,
        /// Filename (without .md).
        #[arg(long)]
        filename: Option<String>,
        /// Type hint for placement.
        #[arg(long)]
        type_hint: Option<String>,
        /// Tags (comma-separated).
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Explicit folder (skips placement).
        #[arg(long)]
        folder: Option<String>,
    },
    /// Append content to an existing note.
    Append {
        /// Target note (path, basename, or #docid).
        file: String,
        /// Content to append (reads from stdin if omitted).
        #[arg(long)]
        content: Option<String>,
    },
    /// Archive a note (soft delete — moves to archive, removes from index).
    Archive {
        /// Target note (path, basename, or #docid).
        file: String,
    },
    /// Restore an archived note to its original location.
    Unarchive {
        /// Archived note path (e.g., "04-Archive/01-Projects/note.md").
        file: String,
    },
    /// Edit a specific section of a note.
    Edit {
        /// Target note (path, basename, or #docid).
        #[arg(long)]
        file: String,
        /// Section heading to edit (case-insensitive).
        #[arg(long)]
        heading: String,
        /// Content to add/replace in the section.
        #[arg(long)]
        content: String,
        /// Edit mode: "replace", "prepend", or "append" (default: "append").
        #[arg(long, default_value = "append")]
        mode: String,
    },
    /// Rewrite a note's body content (preserves frontmatter by default).
    Rewrite {
        /// Target note (path, basename, or #docid).
        #[arg(long)]
        file: String,
        /// New body content.
        #[arg(long)]
        content: String,
        /// Preserve existing frontmatter (default: true).
        #[arg(long, default_value_t = true)]
        preserve_frontmatter: bool,
    },
    /// Edit a note's frontmatter properties.
    EditFrontmatter {
        /// Target note (path, basename, or #docid).
        #[arg(long)]
        file: String,
        /// Operations as JSON string: [{"op":"add_tag","value":"rust"},{"op":"set","key":"status","value":"done"}]
        #[arg(long)]
        operations: String,
    },
    /// Delete a note.
    Delete {
        /// Target note (path, basename, or #docid).
        file: String,
        /// Delete mode: "soft" (archive, default) or "hard" (permanent).
        #[arg(long, default_value = "soft")]
        mode: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelsAction {
    /// List available models.
    List,
    /// Show info about a model.
    Info { name: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_tree_is_reachable_from_the_library() {
        let cmd = Cli::command();
        let names: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        assert!(names.contains(&"search"), "got {names:?}");
        assert!(names.contains(&"index"), "got {names:?}");
    }

    #[test]
    fn migrate_takes_the_mode_the_servers_take() {
        // PARA is the only strategy, so `migrate` is a leaf that takes the
        // same three words every surface takes (#62).
        let cli = Cli::try_parse_from(["engraph", "migrate", "--mode", "apply"]).unwrap();
        match cli.command {
            Command::Migrate { mode } => assert_eq!(mode, "apply"),
            other => panic!("got {other:?}"),
        }
        assert!(
            Cli::try_parse_from(["engraph", "migrate", "para", "--apply"]).is_err(),
            "the PARA leaf is gone"
        );
        assert!(
            Cli::try_parse_from(["engraph", "migrate"]).is_err(),
            "the mode is required"
        );
    }

    #[test]
    fn init_takes_a_mode_and_runs_the_prompts_without_one() {
        // `init` is one capability: `--mode` on every surface, and the
        // interactive flow when the CLI is given none (#62).
        let cli = Cli::try_parse_from(["engraph", "init", "--mode", "detect"]).unwrap();
        match cli.command {
            Command::Init { mode, .. } => assert_eq!(mode.as_deref(), Some("detect")),
            other => panic!("got {other:?}"),
        }
        let cli = Cli::try_parse_from(["engraph", "init"]).unwrap();
        match cli.command {
            Command::Init { mode, .. } => assert_eq!(mode, None),
            other => panic!("got {other:?}"),
        }
    }
}
