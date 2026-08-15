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
    /// reason. The asymmetry runs both ways — `migrate` takes a `preview` that
    /// a command line has no spelling for — so the parity test needs a word
    /// for it, or the only way to make it pass is to stop reading a whole
    /// direction (#62).
    pub server_only_args: &'static [(&'static str, &'static str)],
}

impl Capability {
    /// The MCP tool name.
    pub fn mcp_name(&self) -> String {
        self.name.replace('-', "_")
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
        name: "vault-map",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "who",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "project",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "topic",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
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
        // The four flags are the one-edit spelling of `edits`, which a command
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
        name: "identity",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
        server_only_args: &[],
    },
    Capability {
        name: "init",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[
            ("path", "a running server is bound to its configured vault"),
            ("identity", "the interactive flow the CLI alone can run"),
            ("reindex", "the interactive flow the CLI alone can run"),
            (
                "detect",
                "the CLI spelling of mode=detect, kept for its own flow",
            ),
            (
                "json",
                "the CLI spelling of mode=apply, kept for its own flow",
            ),
            ("quiet", "suppresses prompts the other surfaces never show"),
        ],
        server_only_args: &[],
    },
    Capability {
        name: "migrate",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
        server_only_args: &[(
            "preview",
            "the plan `mode=preview` returned; the CLI saves its own copy to disk instead",
        )],
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

/// Routes the transport serves for itself. They name no capability.
pub const HTTP_TRANSPORT_ROUTES: &[(&str, &str)] = &[
    ("/api/health-check", "a liveness probe for the transport"),
    ("/openapi.json", "the transport describing itself"),
    (
        "/.well-known/ai-plugin.json",
        "the transport describing itself",
    ),
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

/// The capability table as markdown, for `docs/surfaces.md`. Generating it
/// is what stops the documentation and the code from drifting (#62).
pub fn render_table() -> String {
    let mut out = String::from(
        "# One name per capability\n\n\
         Generated by `surface::render_table`. Do not edit by hand.\n\n\
         | capability | CLI | MCP | HTTP |\n|---|---|---|---|\n",
    );
    for c in CAPABILITIES {
        let cli = match c.cli {
            Presence::On => format!("`engraph {}`", c.name),
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
        out.push_str(&format!("| `engraph {name}` | {reason} |\n"));
    }
    out.push_str("\n## Transport routes\n\n| route | reason |\n|---|---|\n");
    for (path, reason) in HTTP_TRANSPORT_ROUTES {
        out.push_str(&format!("| `{path}` | {reason} |\n"));
    }
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
        let actual: BTreeSet<String> = crate::serve::EngraphServer::tool_router()
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

        for tool in crate::serve::EngraphServer::tool_router().list_all() {
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
        let tools = crate::serve::EngraphServer::tool_router().list_all();

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

        let tool = crate::serve::EngraphServer::tool_router()
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

    /// #62 is finished when no surface differs from the table.
    #[test]
    fn nothing_is_pending() {
        assert!(PENDING_CLI.is_empty(), "{PENDING_CLI:?}");
        assert!(PENDING_MCP.is_empty(), "{PENDING_MCP:?}");
        assert!(PENDING_HTTP.is_empty(), "{PENDING_HTTP:?}");
    }

    #[test]
    fn there_are_twenty_capabilities() {
        assert_eq!(CAPABILITIES.len(), 20);
    }

    #[test]
    fn the_committed_table_matches_the_rendered_one() {
        let want = render_table();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/surfaces.md");
        let got = std::fs::read_to_string(path).unwrap_or_default();
        assert_eq!(
            got, want,
            "docs/surfaces.md is stale. Write this to it:\n\n{want}"
        );
    }
}
