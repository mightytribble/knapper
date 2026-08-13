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
    },
    Capability {
        name: "read",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "list",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "tags",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "vault-map",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "who",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "project",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "topic",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    // ── Writing ──
    Capability {
        name: "create",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    Capability {
        name: "update",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    Capability {
        name: "delete",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    Capability {
        name: "move",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    Capability {
        name: "archive",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    // ── Indexing and diagnostics ──
    Capability {
        name: "index",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[("path", "a running server is bound to its configured vault")],
    },
    Capability {
        name: "reindex-file",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
    Capability {
        name: "status",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "health",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "identity",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Get,
        cli_only_args: &[],
    },
    Capability {
        name: "init",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[
            ("identity", "the interactive flow the CLI alone can run"),
            ("reindex", "the interactive flow the CLI alone can run"),
            (
                "detect",
                "the CLI spelling of mode=detect, kept for its own flow",
            ),
            ("quiet", "suppresses prompts the other surfaces never show"),
        ],
    },
    Capability {
        name: "migrate",
        cli: Presence::On,
        mcp: Presence::On,
        http: Http::Post,
        cli_only_args: &[],
    },
];

/// What the CLI has yet to bring onto the table (#62).
pub const PENDING_CLI: &[Pending] = &[
    Pending::NotYetAdded("health"),
    Pending::NotYetAdded("reindex-file"),
    Pending::NotYetAdded("move"),
    Pending::NotYetAdded("update"),
    Pending::NotYetAdded("read"),
    Pending::NotYetAdded("list"),
    Pending::NotYetAdded("tags"),
    Pending::NotYetAdded("vault-map"),
    Pending::NotYetAdded("who"),
    Pending::NotYetAdded("project"),
    Pending::NotYetAdded("topic"),
    Pending::NotYetAdded("create"),
    Pending::NotYetAdded("delete"),
    Pending::NotYetAdded("archive"),
    Pending::NotYetRemoved("context"),
    Pending::NotYetRemoved("write"),
    Pending::NotYetRemoved("graph"),
];

/// What the MCP server has yet to bring onto the table (#62).
pub const PENDING_MCP: &[Pending] = &[
    Pending::NotYetAdded("index"),
    Pending::NotYetAdded("status"),
    Pending::NotYetAdded("topic"),
    Pending::NotYetAdded("update"),
    Pending::NotYetAdded("move"),
    Pending::NotYetAdded("init"),
    Pending::NotYetAdded("migrate"),
    Pending::NotYetRemoved("context"),
    Pending::NotYetRemoved("append"),
    Pending::NotYetRemoved("edit"),
    Pending::NotYetRemoved("rewrite"),
    Pending::NotYetRemoved("edit_frontmatter"),
    Pending::NotYetRemoved("update_metadata"),
    Pending::NotYetRemoved("unarchive"),
    Pending::NotYetRemoved("move_note"),
    Pending::NotYetRemoved("setup"),
    Pending::NotYetRemoved("migrate_preview"),
    Pending::NotYetRemoved("migrate_apply"),
    Pending::NotYetRemoved("migrate_undo"),
];

/// What the HTTP API has yet to bring onto the table (#62). Entries are
/// route paths, because that is what the router registers.
pub const PENDING_HTTP: &[Pending] = &[
    Pending::NotYetAdded("/api/index"),
    Pending::NotYetAdded("/api/status"),
    Pending::NotYetAdded("/api/topic"),
    Pending::NotYetAdded("/api/update"),
    Pending::NotYetAdded("/api/init"),
    Pending::NotYetAdded("/api/migrate"),
    Pending::NotYetRemoved("/api/context"),
    Pending::NotYetRemoved("/api/append"),
    Pending::NotYetRemoved("/api/edit"),
    Pending::NotYetRemoved("/api/rewrite"),
    Pending::NotYetRemoved("/api/edit-frontmatter"),
    Pending::NotYetRemoved("/api/update-metadata"),
    Pending::NotYetRemoved("/api/unarchive"),
    Pending::NotYetRemoved("/api/setup"),
    Pending::NotYetRemoved("/api/migrate/preview"),
    Pending::NotYetRemoved("/api/migrate/apply"),
    Pending::NotYetRemoved("/api/migrate/undo"),
];

/// Capabilities whose CLI arguments are declared apart from the shared
/// parameter struct, and why. The parity test checks these by name,
/// because they are the only ones where two declarations can drift.
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
