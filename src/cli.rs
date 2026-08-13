//! What the CLI accepts. The definitions live in the library and not in the
//! binary so that `surface.rs`'s parity test can walk them under
//! `cargo test --lib` (#62). The dispatch stays in `main.rs`.
//!
//! `Command` holds one variant per capability, at the top level. A
//! capability's arguments come from its `crate::params` struct, so the command
//! line, the MCP schema and the HTTP route read one declaration. A variant
//! carries fields of its own only where the CLI has an argument the other
//! surfaces cannot have; `surface::CAPABILITIES` names each of those with its
//! reason.
//!
//! Every doc comment below is user-facing text: clap prints an item's doc
//! comment as its help, and an enum's reaches the top-level `--help`. Notes
//! about the code are `//` comments for that reason.

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
    // `group(skip)` because the flattened struct already declares a group of
    // its own name, and clap gives a variant with named fields one too.
    #[group(skip)]
    Index {
        #[command(flatten)]
        args: crate::params::Index,

        /// Path to the vault (overrides config). A running server is bound to
        /// its configured vault, so this argument is the CLI's alone.
        path: Option<PathBuf>,
    },

    /// Search the indexed vault.
    Search(crate::params::Search),

    /// Read a note's full content, or one of its sections.
    Read(crate::params::Read),

    /// List notes by metadata filters.
    List(crate::params::List),

    /// List the vault's tag vocabulary.
    Tags(crate::params::Tags),

    /// Vault structure overview.
    VaultMap(crate::params::VaultMap),

    /// Person context bundle.
    Who(crate::params::Who),

    /// Project context bundle.
    Project(crate::params::Project),

    /// Rich topic context with a character budget.
    Topic(crate::params::Topic),

    /// Create a new note.
    Create(crate::params::Create),

    /// Update a note's body, one of its sections, or one of its properties.
    // The one capability that declares its arguments here and not in
    // `crate::params`: a list of edits is not a clap-parsable type, so the
    // flags below are the one-edit form of the same grammar and `--edits` is
    // the whole of it (#62).
    Update {
        /// File path, basename, or #docid.
        file: String,
        /// The section to edit. Omit for the note's body.
        #[arg(long)]
        section: Option<String>,
        /// The frontmatter property to edit.
        #[arg(long)]
        property: Option<String>,
        /// What the edit does to what it names. `remove` is for a property
        /// alone.
        #[arg(long, value_enum, default_value = "replace")]
        mode: crate::params::EditMode,
        /// The text to write, or a comma-separated list for a list-valued
        /// property such as tags or aliases. A comma is how the command line
        /// spells a sequence, as it does for `--tags`; a value that has to
        /// hold a comma is written with `--edits`.
        #[arg(long, value_delimiter = ',')]
        content: Vec<String>,
        /// A JSON array of edits, applied in one write. It replaces the flags
        /// above, which are the one-edit form of the same grammar (#62).
        #[arg(long, conflicts_with_all = ["section", "property", "mode", "content"])]
        edits: Option<String>,
    },

    /// Delete a note.
    Delete(crate::params::Delete),

    /// Move a note to another folder.
    // `move` is a Rust keyword, so the variant is `Move` and the command name
    // is declared.
    #[command(name = "move")]
    Move(crate::params::Move),

    /// Archive a note, or restore one the archive holds.
    Archive(crate::params::Archive),

    /// Re-index one file after an edit made outside engraph.
    ReindexFile(crate::params::ReindexFile),

    /// Show index status and statistics.
    Status(crate::params::Status),

    /// Vault health report: orphans, broken links, stale notes, tag hygiene.
    Health(crate::params::Health),

    /// Clear cached data.
    Clear {
        /// Remove everything including the database and embeddings.
        #[arg(long)]
        all: bool,
    },

    /// Initialize vault profile, identity, and search index.
    // `group(skip)` for the reason `index` gives: the flattened struct
    // declares the group this variant would declare again.
    #[group(skip)]
    Init {
        #[command(flatten)]
        args: crate::params::Init,
        /// Path to vault directory.
        path: Option<PathBuf>,
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
    },

    /// Print identity block (L0 + L1 context for AI agents).
    Identity(crate::params::Identity),

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

    /// Migrate vault structure into PARA.
    Migrate(crate::params::Migrate),
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

    /// Clap builds a subcommand only when something parses it, so a duplicate
    /// argument or group in an arm no test parses would otherwise reach a
    /// user first. This builds all of them and runs clap's own asserts (#62).
    #[test]
    fn every_command_passes_claps_own_asserts() {
        Cli::command().debug_assert();
    }

    #[test]
    fn migrate_takes_the_mode_the_servers_take() {
        // PARA is the only strategy, so `migrate` is a leaf that takes the
        // same three words every surface takes (#62).
        let cli = Cli::try_parse_from(["engraph", "migrate", "--mode", "apply"]).unwrap();
        match cli.command {
            Command::Migrate(args) => assert_eq!(args.mode, "apply"),
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
            Command::Init { args, .. } => assert_eq!(args.mode.as_deref(), Some("detect")),
            other => panic!("got {other:?}"),
        }
        let cli = Cli::try_parse_from(["engraph", "init"]).unwrap();
        match cli.command {
            Command::Init { args, .. } => assert_eq!(args.mode, None),
            other => panic!("got {other:?}"),
        }
    }

    /// The two command groups are gone: every capability is one word (#62).
    #[test]
    fn the_capabilities_are_top_level_commands() {
        for group in [
            ["engraph", "context", "read"],
            ["engraph", "write", "create"],
        ] {
            assert!(
                Cli::try_parse_from(group).is_err(),
                "{group:?} still parses"
            );
        }

        let cli = Cli::try_parse_from(["engraph", "read", "note.md", "--section", "Spells"])
            .expect("read is a command");
        match cli.command {
            Command::Read(args) => {
                assert_eq!(args.file, "note.md");
                assert_eq!(args.section.as_deref(), Some("Spells"));
            }
            other => panic!("got {other:?}"),
        }

        let cli =
            Cli::try_parse_from(["engraph", "vault-map"]).expect("vault-map is one command name");
        assert!(matches!(cli.command, Command::VaultMap(_)), "{cli:?}");
    }

    /// `move` is a Rust keyword, so the variant carries the command name.
    #[test]
    fn move_is_spelled_the_way_the_table_spells_it() {
        let cli = Cli::try_parse_from(["engraph", "move", "note.md", "--new-folder", "02-Areas"])
            .expect("move is a command");
        match cli.command {
            Command::Move(args) => {
                assert_eq!(args.file, "note.md");
                assert_eq!(args.new_folder, "02-Areas");
            }
            other => panic!("got {other:?}"),
        }
    }

    /// The flags are the one-edit form of the edit list, and `--edits` is the
    /// whole grammar. The two cannot be given together (#62).
    #[test]
    fn update_takes_one_edit_as_flags_or_a_list_as_json() {
        let cli = Cli::try_parse_from([
            "engraph",
            "update",
            "note.md",
            "--property",
            "tags",
            "--content",
            "a,b",
        ])
        .expect("the one-edit form parses");
        match cli.command {
            Command::Update {
                file,
                property,
                content,
                mode,
                ..
            } => {
                assert_eq!(file, "note.md");
                assert_eq!(property.as_deref(), Some("tags"));
                assert_eq!(content, vec!["a".to_string(), "b".to_string()]);
                assert!(matches!(mode, crate::params::EditMode::Replace));
            }
            other => panic!("got {other:?}"),
        }

        let cli = Cli::try_parse_from(["engraph", "update", "note.md", "--edits", "[]"])
            .expect("the list form parses");
        match cli.command {
            Command::Update { edits, .. } => assert_eq!(edits.as_deref(), Some("[]")),
            other => panic!("got {other:?}"),
        }

        assert!(
            Cli::try_parse_from([
                "engraph",
                "update",
                "note.md",
                "--edits",
                "[]",
                "--content",
                "x"
            ])
            .is_err(),
            "the two forms name the same edits twice"
        );
    }
}
