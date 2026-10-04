//! What every capability is called on each surface (#62).
//!
//! A capability is one top-level CLI command, one MCP tool and one HTTP
//! route. The name is written in kebab-case, and each surface spells it its
//! own way: the CLI command as written, the MCP tool with `-` as `_`, and
//! the HTTP route under `/api/`. One transform gets from any spelling to
//! any other, so a caller who learns one surface can predict the others.
//!
//! The tests below compare this table with what each surface registers.
//! Where a surface has no such call, the absence is declared with its
//! reason, and an undeclared absence fails.

/// Whether a capability reaches a surface, and why not when it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    On,
    Exempt(&'static str),
}

/// The method a capability's route serves, or the reason it has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Http {
    Get,
    Post,
    Exempt(&'static str),
}

/// One capability, and its spelling on each surface.
pub struct Capability {
    /// The one name, in kebab-case.
    pub name: &'static str,
    pub cli: Presence,
    pub mcp: Presence,
    pub http: Http,
    /// Arguments this capability takes on the CLI alone, each with its
    /// reason. The parameter parity test reads them as allowed absences.
    pub cli_only_args: &'static [(&'static str, &'static str)],
    /// Arguments this capability takes on the servers alone, each with its
    /// reason. The asymmetry can run both ways, so the parity test needs a
    /// word for it, or the only way to make it pass is to stop reading a
    /// whole direction (#62).
    pub server_only_args: &'static [(&'static str, &'static str)],
}

/// The MCP spelling of a capability's name: `-` written as `_`.
///
/// `Capability::mcp_name` is this transform on a table row, and this is it
/// for a caller holding the name alone — the orientation table in `serve.rs`
/// holds names, not rows — so the transform still has one home (#110).
pub fn mcp_spelling(name: &str) -> String {
    name.replace('-', "_")
}

impl Capability {
    /// The MCP tool name.
    pub fn mcp_name(&self) -> String {
        mcp_spelling(self.name)
    }

    /// The HTTP route path.
    pub fn http_path(&self) -> String {
        format!("/api/{}", self.name)
    }
}

/// A difference between a surface and this table that #62 has not closed
/// yet. Every list below empties by the end of the sweep, and the last
/// task asserts that they are empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pending {
    /// The table names it; the surface does not register it yet.
    NotYetAdded(&'static str),
    /// The surface registers it; the table does not name it.
    NotYetRemoved(&'static str),
}

/// Commands that configure the process and not the vault. They stay on the
/// CLI alone, and the CLI parity test expects them beside the table.
pub const CLI_ONLY: &[(&str, &str)] = &[
    ("configure", "configures the process, not the vault"),
    ("models", "configures the process, not the vault"),
    ("clear", "configures the process, not the vault"),
    ("serve", "configures the process, not the vault"),
];

pub const CAPABILITIES: &[Capability] = &[
    // ── Reading ──
    Capability {
        name: "search",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    // Literal matching, not ranked retrieval (#106). `search` answers what a
    // note is about; this answers whether a note still says a given string,
    // and its POST body carries the scope operators as arrays the way
    // `search`'s does.
    Capability {
        name: "match",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "read",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "list",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "tags",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "properties",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "vault-map",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    // ── Writing ──
    Capability {
        name: "create",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "update",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        // The flags are the one-edit spelling of `edits`, which a command
        // line cannot carry as a list. They are what `cli.rs` declares by hand,
        // so naming them here is what lets the parity test read `update` at all
        // (#62).
        cli_only_args: &[
            (
                "section",
                "the one edit's target section; a command line carries no `edits` list",
            ),
            (
                "property",
                "the one edit's target property; a command line carries no `edits` list",
            ),
            (
                "heading",
                "the one edit's new heading for the section it renames; a command line carries no `edits` list",
            ),
            (
                "after",
                "where the one edit places a new property key; a command line carries no `edits` list",
            ),
            (
                "before",
                "where the one edit places a new property key; a command line carries no `edits` list",
            ),
            (
                "mode",
                "what the one edit does; a command line carries no `edits` list",
            ),
            (
                "content",
                "what the one edit writes; a command line carries no `edits` list",
            ),
        ],
        server_only_args: &[],
    },
    Capability {
        name: "delete",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "move",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "archive",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    // ── Indexing and diagnostics ──
    Capability {
        name: "index",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[("path", "a running server is bound to its configured vault")],
        server_only_args: &[],
    },
    Capability {
        name: "reindex-file",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "status",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "health",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "validate",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[(
            "vault",
            "the pre-init vault root; a running server is bound to its configured vault",
        )],
        server_only_args: &[],
    },
    Capability {
        name: "init",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[("path", "a running server is bound to its configured vault")],
        server_only_args: &[],
    },
];

/// What a capability's HTTP operation says about itself in the OpenAPI
/// document `openapi.rs` generates.
///
/// The parameters are not here: they come from the capability's `params`
/// struct. This table holds the three things a struct cannot say — the
/// operation's id, one summary, and what the 200 reply holds.
pub struct Operation {
    /// The capability name, as `CAPABILITIES` spells it.
    pub name: &'static str,
    /// The `operationId`. A client configured against it keeps working, so
    /// it does not change when the summary does.
    pub id: &'static str,
    /// At most 300 characters. ChatGPT's Actions importer refuses more (#87).
    pub summary: &'static str,
    /// What the 200 reply holds. Handlers build their replies as
    /// `serde_json::Value`, so there is no struct to derive this from.
    pub response: &'static str,
}

pub const OPERATIONS: &[Operation] = &[
    Operation {
        name: "search",
        id: "searchVault",
        summary: "Hybrid semantic + full-text search across the vault. A query pairs with a filter: {\"query\": \"the disagreement over the schedule\", \"links_to\": \"project-atlas\"} cuts the pool to the notes linking to that note and ranks the query inside them. The filter runs first; the ranking runs inside it.",
        response: "An envelope: status ('ok' or 'no_results'); degraded (bool, true when no cross-encoder ranked the results); warnings (array of strings); notes, each answering note's frontmatter properties keyed by path, as a name-to-value map — a value carries its own JSON type, so a number is a number, a checkbox a bool, a key with no value null, a wikilink {link, path} with path present when it resolves, and a name the note carries more than once an array — omitted when no answering note carries a property; blocks, the results that answered, each {id, path, heading_path, lanes (the lanes that account for the result, any of 'semantic', 'keyword', 'linked'), text, untrusted_content, truncated, and score when scores was requested}, with a block's properties read from notes under its path — text is absent on a row the caller asked no text for, which is every row under summaries; overflow, the results the budget excluded, each {id, path, heading_path, lanes, and score when requested} with no text; and less_relevant, the candidates the answer floor rejected, in whatever slots top_n had left after the answers, each carrying its score whether or not scores was requested, beside answer_floor, the floor they missed, on the same 0-100 scale — both omitted when the floor rejected nothing, and a no_results reply that carries them is still no_results. explain, the per-lane breakdown, rides beside the envelope when the request asked for it",
    },
    Operation {
        name: "match",
        id: "matchLiteral",
        summary: "Confirm whether a literal string still appears in the vault's note text, and count the notes holding it.",
        response: "{pattern, notes (how many notes hold it — 0 means nothing in scope says it), lines (distinct matched lines across every note), hits (the matched lines, capped by limit, each {file, in, heading_path, line})}. The scan is exhaustive and unranked over a note's prose and its frontmatter both; `in` names which half a hit came from, and `scan` narrows the reading to one of them. A wikilink is compared as its display text as well as its markup, so a phrase spanning one is found; the reported line is the note as written.",
    },
    Operation {
        name: "read",
        id: "readNote",
        summary: "Read a note's content, its frontmatter, both, or its metadata, chosen with include.",
        response: "content returns {path, docid, content, and section when a section was read, which is {heading, level, line_start, line_end} — level absent for a promoted bold line}. all returns that plus frontmatter, the note's YAML without its --- fences and present even when the note has none, in which case it is an empty string. frontmatter returns {path, docid, frontmatter} and nothing else. metadata returns {path, docid, frontmatter, byte_count, properties (every property row the note holds), and outgoing_links/incoming_links as arrays of {path, docid, properties} — properties names the custom properties that link is filed under, empty for a plain wikilink}.",
    },
    Operation {
        name: "list",
        id: "listNotes",
        summary: "List notes by scope operators, creator or limit, or with no filter at all to enumerate the whole vault.",
        response: "Array of note summaries, each with aliases, the names the note's frontmatter lists, which read, links_to and linked_from accept in place of its path; links_in, the number of distinct notes that link to it counted over the whole vault, and two sizes from the index: chunk_count, how many units search can return the note as, and token_count, the note's indexed size in tokens. Read the sizes against links_in — a note many others point at that holds little is underwritten, and a long note of few chunks needs sectioning. Under a property filter each note also carries properties, the rows that term matched — narrowed to the links that name the note when links_to is set beside it, and omitted under linked_from, where the matched row belongs to the naming note",
    },
    Operation {
        name: "tags",
        id: "listTags",
        summary: "The vault's tag vocabulary, whole or under one term, each tag with the notes carrying it.",
        response: "Array of tag rows: path, note_count, and display where the vault spells the tag differently from its path",
    },
    Operation {
        name: "properties",
        id: "listProperties",
        summary: "The vault's custom properties: every name with its note count, the kinds seen and Obsidian's declared type, or one property's values.",
        response: "Without name, an array of {name, note_count, kinds, declared_type}; with name, an array of {value, kind, note_count}",
    },
    Operation {
        name: "vault-map",
        id: "getVaultMap",
        summary: "Get vault structure overview: folders, counts, the tag vocabulary and the share of notes it covers, the most-linked notes, and recently changed files.",
        response: "Vault structure map",
    },
    Operation {
        name: "create",
        id: "createNote",
        summary: "Create a new note: tags resolved against the vault's vocabulary, links discovered, filed under folder or at the vault root.",
        response: "Created note path and metadata",
    },
    Operation {
        name: "update",
        id: "updateNote",
        summary: "Change an existing note. Applies a list of edits in order, in one write.",
        response: "Updated note path",
    },
    Operation {
        name: "delete",
        id: "deleteNote",
        summary: "Delete a note. Supports soft (archive) and hard (permanent) modes.",
        response: "Deletion confirmation",
    },
    Operation {
        name: "move",
        id: "moveNote",
        summary: "Move a note to a different folder within the vault.",
        response: "New note path",
    },
    Operation {
        name: "archive",
        id: "archiveNote",
        summary: "Archive a note (soft delete), or restore one previously archived with `undo: true`. Archiving moves the note to the archive folder and removes it from the index; `undo` reverses that and re-indexes it.",
        response: "Archived (or restored) note path",
    },
    Operation {
        name: "index",
        id: "indexVault",
        summary: "Index the server's vault: walk it, diff it against the store, and re-embed what changed. Send {} for the defaults; one file is cheaper through /api/reindex-file. It runs to completion holding the store and the embedder, so searches and writes wait while reads answer; a rebuild takes minutes.",
        response: "Counts of new, updated and deleted files, total chunks and the elapsed seconds",
    },
    Operation {
        name: "reindex-file",
        id: "reindexFile",
        summary: "Re-index a single file after external edits. Re-reads, re-embeds, and updates search index.",
        response: "Re-indexed file info (chunks, docid)",
    },
    Operation {
        name: "status",
        id: "getStatus",
        summary: "What the index holds: file and chunk counts, edge and connectivity counts, date coverage, index size, whether intelligence is enabled, and pending_events, the watcher events not yet applied.",
        response: "Index status fields",
    },
    Operation {
        name: "health",
        id: "getHealth",
        summary: "Get vault health report with orphans, broken links, stale notes, and inbox status.",
        response: "Vault health report",
    },
    Operation {
        name: "validate",
        id: "validateVault",
        summary: "Check vault markdown for structural and indexing-quality problems.",
        response: "A report: findings (each {file, line, severity, rule, message}), files_checked, error_count, warning_count, and ok",
    },
    Operation {
        name: "init",
        id: "init",
        summary: "Write the vault profile and index. 'detect' inspects the vault and writes nothing; 'apply' writes vault.toml and indexes. The apply reply carries restart_required: true, because the server reads the profile once, at start.",
        response: "Setup result as JSON",
    },
];

/// What the CLI has yet to bring onto the table (#62). Empty: every
/// capability the table names is one top-level command.
pub const PENDING_CLI: &[Pending] = &[];

/// What the MCP server has yet to bring onto the table (#62). Empty: every
/// capability the table names is one tool.
pub const PENDING_MCP: &[Pending] = &[];

/// What the HTTP API has yet to bring onto the table (#62). Empty: every
/// capability the table names is one route.
pub const PENDING_HTTP: &[Pending] = &[];

/// Capabilities the parameter parity test cannot compare, and why.
///
/// Empty (#62). `update` is the one capability that declares its CLI
/// arguments apart from `params::Update`, but the four extra flags are
/// `cli_only_args` with a reason each, so both sides of `update` reduce to
/// `file` and `edits` and the test reads it like every other capability —
/// which is where a second declaration can drift, so it is the last one to
/// exempt. The list stays as the declaration point for the next capability
/// that has to opt out, and `every_capability_with_a_split_declaration_is_named`
/// holds each entry to a real capability with a reason.
pub const PARAMS_NOT_SHARED: &[(&str, &str)] = &[];

/// Capabilities the MCP orientation string leaves out, each with its
/// reason. The shape `PARAMS_NOT_SHARED` uses — a name and a non-empty
/// reason — so an omission is a decision on the record rather than the
/// oversight that left `match`, `properties` and `validate` unmentioned for
/// three releases (#110).
pub const ORIENTATION_OMITTED: &[(&str, &str)] = &[];

/// Routes the transport serves for itself. They name no capability.
pub const HTTP_TRANSPORT_ROUTES: &[(&str, &str)] = &[
    ("/api/health-check", "a liveness probe for the transport"),
    ("/openapi.json", "the transport describing itself"),
];

/// The set a surface should register: every capability the table puts on it,
/// less what is not yet added, plus what is not yet removed.
pub fn expected(
    on_surface: impl Fn(&Capability) -> bool,
    spell: impl Fn(&Capability) -> String,
    pending: &[Pending],
) -> std::collections::BTreeSet<String> {
    let mut set: std::collections::BTreeSet<String> = CAPABILITIES
        .iter()
        .filter(|c| on_surface(c))
        .map(spell)
        .collect();
    for p in pending {
        match p {
            Pending::NotYetAdded(n) => {
                set.remove(*n);
            }
            Pending::NotYetRemoved(n) => {
                set.insert((*n).to_string());
            }
        }
    }
    set
}

/// The capability table as markdown, for `surfaces.md`. Generating it
/// is what stops the documentation and the code from drifting (#62).
pub fn render_table() -> String {
    let mut out = String::from(
        "# One name per capability\n\n\
         Generated by `surface::render_table`. Do not edit by hand.\n\n\
         | capability | CLI | MCP | HTTP |\n|---|---|---|---|\n",
    );
    for c in CAPABILITIES {
        let cli = match c.cli {
            Presence::On => format!("`knapper {}`", c.name),
            Presence::Exempt(r) => format!("— ({r})"),
        };
        let mcp = match c.mcp {
            Presence::On => format!("`{}`", c.mcp_name()),
            Presence::Exempt(r) => format!("— ({r})"),
        };
        let http = match c.http {
            Http::Get => format!("`GET {}`", c.http_path()),
            Http::Post => format!("`POST {}`", c.http_path()),
            Http::Exempt(r) => format!("— ({r})"),
        };
        out.push_str(&format!("| `{}` | {cli} | {mcp} | {http} |\n", c.name));
    }
    out.push_str("\n## CLI only\n\n| command | reason |\n|---|---|\n");
    for (name, reason) in CLI_ONLY {
        out.push_str(&format!("| `knapper {name}` | {reason} |\n"));
    }
    out.push_str("\n## Transport routes\n\n| route | reason |\n|---|---|\n");
    for (path, reason) in HTTP_TRANSPORT_ROUTES {
        out.push_str(&format!("| `{path}` | {reason} |\n"));
    }
    out.push_str(
        "\n## Errors\n\n\
         A fault the caller can repair carries one of six kinds, built in `fault.rs` \
         where the fault is known: `invalid_input`, `not_found`, `ambiguous`, `conflict`, \
         `stale_index` and `read_only`. HTTP answers 400, 404, 400, 409, 500 and 403 \
         for them in that order, with `kind` beside `error` in the body. MCP answers \
         `INVALID_PARAMS` for the first three, `INVALID_REQUEST` for `conflict` and \
         `read_only`, and `INTERNAL_ERROR` for `stale_index`, with `data.kind` naming \
         the kind. The CLI prints the message and exits 1. Anything without a kind is \
         the server's own: 500 and `INTERNAL_ERROR`.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::collections::BTreeSet;

    #[test]
    fn the_cli_registers_what_the_table_names() {
        let cmd = crate::cli::Cli::command();
        let actual: BTreeSet<String> = cmd
            .get_subcommands()
            .map(|s| s.get_name().to_string())
            .collect();

        let mut want = expected(
            |c| matches!(c.cli, Presence::On),
            |c| c.name.to_string(),
            PENDING_CLI,
        );
        for (name, _reason) in CLI_ONLY {
            want.insert((*name).to_string());
        }

        assert_eq!(
            actual,
            want,
            "\nonly on the CLI: {:?}\nonly in the table: {:?}",
            actual.difference(&want).collect::<Vec<_>>(),
            want.difference(&actual).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_mcp_server_registers_what_the_table_names() {
        let actual: BTreeSet<String> = crate::serve::KnapperServer::tool_router()
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();

        let want = expected(
            |c| matches!(c.mcp, Presence::On),
            |c| c.mcp_name(),
            PENDING_MCP,
        );

        assert_eq!(
            actual,
            want,
            "\nonly on MCP: {:?}\nonly in the table: {:?}",
            actual.difference(&want).collect::<Vec<_>>(),
            want.difference(&actual).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_http_api_registers_what_the_table_names() {
        let actual: BTreeSet<String> = crate::http::routes()
            .into_iter()
            .map(|(path, _)| path.to_string())
            .collect();

        let mut want = expected(
            |c| !matches!(c.http, Http::Exempt(_)),
            |c| c.http_path(),
            PENDING_HTTP,
        );
        for (path, _reason) in HTTP_TRANSPORT_ROUTES {
            want.insert((*path).to_string());
        }

        assert_eq!(
            actual,
            want,
            "\nonly on the router: {:?}\nonly in the table: {:?}",
            actual.difference(&want).collect::<Vec<_>>(),
            want.difference(&actual).collect::<Vec<_>>()
        );
    }

    /// The parameter names a capability's MCP tool publishes, and the ones its
    /// clap command takes, are one set (#62).
    ///
    /// This is the guard the whole sweep exists to keep. `params.rs` derives
    /// `clap::Args`, `Deserialize` and `JsonSchema` from one declaration, so a
    /// capability that reads its struct holds by construction. `update` is the
    /// one that declares its arguments twice — `cli.rs` writes its flags by
    /// hand against `params::Update` — so it is the one capability where the
    /// two can drift, and the test reads it. Its four extra flags are
    /// `cli_only_args`, which leaves `file` and `edits` on both sides.
    ///
    /// Three classes of clap argument are not parameters of the capability and
    /// are subtracted: the `--help` clap adds itself, the global flags that
    /// configure the process rather than the call, and each capability's own
    /// `cli_only_args`, which name an argument the other surfaces cannot have
    /// and say why.
    #[test]
    fn every_tool_takes_the_parameters_its_command_takes() {
        let cmd = crate::cli::Cli::command();
        let mut checked = 0;

        for tool in crate::serve::KnapperServer::tool_router().list_all() {
            let name = tool.name.to_string();
            let capability = CAPABILITIES
                .iter()
                .find(|c| c.mcp_name() == name)
                .unwrap_or_else(|| panic!("the tool {name} names no capability"));

            // A capability that declares its arguments apart from the shared
            // struct opts out here with a reason, which the test above checks.
            if PARAMS_NOT_SHARED.iter().any(|(n, _)| *n == capability.name) {
                continue;
            }

            let server_only: BTreeSet<String> = capability
                .server_only_args
                .iter()
                .map(|(a, _)| (*a).to_string())
                .collect();

            let schema: BTreeSet<String> = tool
                .input_schema
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|o| o.keys().cloned().collect::<BTreeSet<String>>())
                .unwrap_or_default()
                .difference(&server_only)
                .cloned()
                .collect();

            let subcommand = cmd
                .get_subcommands()
                .find(|s| s.get_name() == capability.name)
                .unwrap_or_else(|| panic!("{} is not a CLI command", capability.name));

            let exempt: BTreeSet<String> = capability
                .cli_only_args
                .iter()
                .map(|(a, _)| (*a).to_string())
                .collect();

            let clap: BTreeSet<String> = subcommand
                .get_arguments()
                // `help` is clap's own, and a global flag configures the
                // process and not the call — the design names `--json` and
                // `--verbose` as CLI-only for that reason.
                .filter(|a| a.get_id() != "help" && !a.is_global_set())
                .map(|a| a.get_id().to_string())
                .filter(|id| !exempt.contains(id))
                .collect();

            assert_eq!(
                clap,
                schema,
                "\n{}: only on the CLI: {:?}\n{}: only in the tool schema: {:?}",
                capability.name,
                clap.difference(&schema).collect::<Vec<_>>(),
                capability.name,
                schema.difference(&clap).collect::<Vec<_>>()
            );
            checked += 1;
        }

        assert_eq!(
            checked,
            CAPABILITIES.len() - PARAMS_NOT_SHARED.len(),
            "the test skipped a capability it should have compared"
        );
    }

    /// Every exemption names a real argument of the surface it exempts, and
    /// gives a reason. A stale entry would silently widen the parity test
    /// above into an exemption for nothing (#62).
    #[test]
    fn every_exempt_argument_exists_and_says_why() {
        let cmd = crate::cli::Cli::command();
        let tools = crate::serve::KnapperServer::tool_router().list_all();

        for capability in CAPABILITIES {
            let subcommand = cmd
                .get_subcommands()
                .find(|s| s.get_name() == capability.name)
                .unwrap_or_else(|| panic!("{} is not a CLI command", capability.name));
            let clap_args: BTreeSet<String> = subcommand
                .get_arguments()
                .map(|a| a.get_id().to_string())
                .collect();
            for (arg, reason) in capability.cli_only_args {
                assert!(
                    clap_args.contains(*arg),
                    "{}: cli_only_args names {arg}, which the command does not take",
                    capability.name
                );
                assert!(
                    !reason.is_empty(),
                    "{}: {arg} is exempt with no reason",
                    capability.name
                );
            }

            let tool = tools
                .iter()
                .find(|t| t.name == capability.mcp_name())
                .unwrap_or_else(|| panic!("{} is not an MCP tool", capability.name));
            let schema: BTreeSet<String> = tool
                .input_schema
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default();
            for (arg, reason) in capability.server_only_args {
                assert!(
                    schema.contains(*arg),
                    "{}: server_only_args names {arg}, which the tool schema does not publish",
                    capability.name
                );
                assert!(
                    !reason.is_empty(),
                    "{}: {arg} is exempt with no reason",
                    capability.name
                );
            }
        }
    }

    #[test]
    fn every_capability_with_a_split_declaration_is_named() {
        // A capability may only opt out of the shared struct with a reason.
        for (name, reason) in PARAMS_NOT_SHARED {
            assert!(
                CAPABILITIES.iter().any(|c| c.name == *name),
                "{name} is not a capability"
            );
            assert!(!reason.is_empty(), "{name} opts out with no reason");
        }
    }

    /// `folder` was a second directory handle, and it disagreed with the
    /// first: `LIKE 'lore%'` folds case, reads `_` in the argument as a
    /// wildcard and matches `lorekeeper.md`, where a directory term is a
    /// case-sensitive range anchored at the path boundary. The scope
    /// operators are the one handle, on all three surfaces (#68).
    #[test]
    fn list_declares_no_folder_parameter() {
        let cmd = crate::cli::Cli::command();
        let list = cmd
            .get_subcommands()
            .find(|s| s.get_name() == "list")
            .expect("list is a CLI command");
        assert!(
            !list.get_arguments().any(|a| a.get_id() == "folder"),
            "the CLI still declares --folder"
        );

        let tool = crate::serve::KnapperServer::tool_router()
            .list_all()
            .into_iter()
            .find(|t| t.name == "list")
            .expect("list is an MCP tool");
        let properties = tool
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
            .expect("the list tool declares properties");
        assert!(
            !properties.contains_key("folder"),
            "the list tool schema still declares folder"
        );

        let spec = crate::openapi::build_openapi_spec("http://localhost:3000");
        let named: Vec<&str> = spec["paths"]["/api/list"]["get"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert!(
            !named.contains(&"folder"),
            "/api/list still documents a folder parameter"
        );
    }

    /// The orientation string an MCP client is sent on connect names tools,
    /// and a tool name in prose drifts the way a parameter cannot: `topic`,
    /// `who` and `project` outlived their removal in the string, and `match`,
    /// `properties` and `validate` never reached it. The #62 net reads which
    /// calls exist and what they take, and no prose at all (#110).
    #[test]
    fn the_instructions_name_every_tool_the_server_registers() {
        let text = crate::serve::instructions();
        let omitted: BTreeSet<&str> = ORIENTATION_OMITTED.iter().map(|(n, _)| *n).collect();

        for tool in crate::serve::KnapperServer::tool_router().list_all() {
            if omitted.contains(tool.name.as_ref()) {
                continue;
            }
            assert!(
                text.contains(&format!(" {} ", tool.name)),
                "the instructions do not name the `{}` tool",
                tool.name
            );
        }
    }

    /// A filter two tools share is a composition nothing else will teach: the
    /// caller forms the intent to combine two fields before it reads either
    /// field's schema, and the orientation is what it reads first. It
    /// described filtering as `list`'s alone, so link-shaped questions routed
    /// to the tool that cannot rank (#136).
    ///
    /// The tools are read from the schemas and not from a written list, for
    /// the reason #110 gives: a sentence naming tools drifts where a
    /// parameter cannot, and a third tool that gains the filter would leave
    /// this sentence quietly wrong.
    #[test]
    fn the_orientation_names_every_tool_that_takes_a_graph_filter() {
        let text = crate::serve::instructions();
        let sentence = text
            .split_inclusive('.')
            .find(|s| s.contains("links_to"))
            .expect("the orientation never mentions `links_to`");

        let mut named = 0;
        for tool in crate::serve::KnapperServer::tool_router().list_all() {
            let takes_filter = tool
                .input_schema
                .get("properties")
                .and_then(|p| p.as_object())
                .is_some_and(|o| o.contains_key("links_to"));
            if !takes_filter {
                continue;
            }
            assert!(
                sentence.contains(tool.name.as_ref()),
                "`{}` takes `links_to` and the orientation's graph-filter sentence does not name it: {sentence}",
                tool.name
            );
            named += 1;
        }
        assert!(
            named >= 2,
            "the sentence earns its place by naming a composition, so at least two tools take the filter"
        );
    }

    /// `serve::ORIENTATION` is the prose half of the table, so it answers to
    /// the same parity the three registrations answer to: it describes every
    /// tool the MCP server registers, describes each one once, and describes
    /// nothing that is not a capability. That last clause is what a
    /// hand-written string could not be held to — `topic`, `who` and
    /// `project` were named in prose for two releases after they stopped
    /// existing, and no rule separates a retired tool's name from an
    /// ordinary word in a sentence (#110).
    #[test]
    fn the_orientation_describes_every_mcp_tool_and_no_other() {
        use crate::serve::ORIENTATION;

        let table: BTreeSet<&str> = CAPABILITIES.iter().map(|c| c.name).collect();

        let mut described: BTreeSet<&str> = BTreeSet::new();
        let mut seen_groups: Vec<&str> = Vec::new();
        for row in ORIENTATION {
            assert!(
                table.contains(row.capability),
                "the orientation describes `{}`, which is not a capability",
                row.capability
            );
            assert!(
                described.insert(row.capability),
                "the orientation describes `{}` twice",
                row.capability
            );
            assert!(
                !row.clause.trim().is_empty(),
                "`{}` has an empty clause",
                row.capability
            );
            assert!(
                !row.clause.ends_with('.'),
                "`{}`'s clause ends the sentence the assembler ends",
                row.capability
            );
            if seen_groups.last() != Some(&row.group) {
                assert!(
                    !seen_groups.contains(&row.group),
                    "the `{}` group is split; rows of one group are contiguous",
                    row.group
                );
                seen_groups.push(row.group);
            }
        }

        let omitted: BTreeSet<&str> = ORIENTATION_OMITTED.iter().map(|(n, _)| *n).collect();
        for (name, reason) in ORIENTATION_OMITTED {
            assert!(
                table.contains(name),
                "`{name}` is left out of the orientation but is not a capability"
            );
            assert!(
                !reason.trim().is_empty(),
                "`{name}` is left out of the orientation with no reason"
            );
            assert!(
                !described.contains(name),
                "`{name}` is both described and declared left out"
            );
        }

        let want: BTreeSet<&str> = CAPABILITIES
            .iter()
            .filter(|c| matches!(c.mcp, Presence::On))
            .map(|c| c.name)
            .filter(|n| !omitted.contains(n))
            .collect();

        assert_eq!(
            described,
            want,
            "\nonly in the orientation: {:?}\nonly in the table: {:?}",
            described.difference(&want).collect::<Vec<_>>(),
            want.difference(&described).collect::<Vec<_>>()
        );
    }

    /// #62 is finished when no surface differs from the table.
    #[test]
    fn nothing_is_pending() {
        assert!(PENDING_CLI.is_empty(), "{PENDING_CLI:?}");
        assert!(PENDING_MCP.is_empty(), "{PENDING_MCP:?}");
        assert!(PENDING_HTTP.is_empty(), "{PENDING_HTTP:?}");
    }

    #[test]
    fn there_are_eighteen_capabilities() {
        assert_eq!(CAPABILITIES.len(), 18);
    }

    /// The operation table is one row per capability the HTTP surface
    /// serves: a capability without a row has no `operationId`, and a row
    /// without a capability describes a route that does not exist.
    #[test]
    fn every_http_capability_has_one_operation_and_no_operation_is_an_orphan() {
        let capabilities: BTreeSet<&str> = CAPABILITIES
            .iter()
            .filter(|c| !matches!(c.http, Http::Exempt(_)))
            .map(|c| c.name)
            .collect();
        let rows: Vec<&str> = OPERATIONS.iter().map(|o| o.name).collect();
        let distinct: BTreeSet<&str> = rows.iter().copied().collect();
        assert_eq!(
            rows.len(),
            distinct.len(),
            "a capability has two rows: {rows:?}"
        );
        assert_eq!(
            distinct,
            capabilities,
            "\nrows with no capability: {:?}\ncapabilities with no row: {:?}",
            distinct.difference(&capabilities).collect::<Vec<_>>(),
            capabilities.difference(&distinct).collect::<Vec<_>>()
        );
    }

    #[test]
    fn operation_ids_are_unique() {
        let mut seen = BTreeSet::new();
        for o in OPERATIONS {
            assert!(seen.insert(o.id), "{} repeats operationId {}", o.name, o.id);
        }
    }

    /// ChatGPT's Actions importer refuses a summary over 300 characters, and
    /// the hand-written spec shipped one of 412 (#87). The cap is a test so
    /// the next long summary fails here and not at import.
    #[test]
    fn no_summary_exceeds_300_characters() {
        for o in OPERATIONS {
            let len = o.summary.chars().count();
            assert!(
                len <= 300,
                "{}: the summary is {len} characters; the cap is 300",
                o.name
            );
            assert!(!o.summary.is_empty(), "{}: the summary is empty", o.name);
            assert!(
                !o.response.is_empty(),
                "{}: the 200 description is empty",
                o.name
            );
        }
    }

    #[test]
    fn the_committed_table_matches_the_rendered_one() {
        let want = render_table();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/surfaces.md");
        let got = std::fs::read_to_string(path).unwrap_or_default();
        assert_eq!(
            got, want,
            "surfaces.md is stale. Write this to it:\n\n{want}"
        );
    }
}
