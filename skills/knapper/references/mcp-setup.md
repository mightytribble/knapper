# Knapper MCP Server Setup

## Install

```bash
brew install mightytribble/tap/knapper
knapper index /path/to/documents
```

## Configure MCP Client

**Claude Code** — register with the CLI:

```bash
claude mcp add knapper -- knapper serve                  # this project only
claude mcp add --scope user knapper -- knapper serve     # everywhere
```

`--scope` picks who sees it: `local` (the default) is this project and private
to you, `project` writes `.mcp.json` for the whole team to share, and `user`
loads knapper in every project you open.

**Scope it to one workspace when the vault belongs to one.** A vault of work
notes has nothing to say about an unrelated codebase, and a user-scoped server
loads its tools into every session whether or not they can help.

Or add it by hand to `~/.claude.json` (user scope) or a project's `.mcp.json`
(project scope):

```json
{
  "mcpServers": {
    "knapper": { "type": "stdio", "command": "knapper", "args": ["serve"] }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "knapper": { "command": "knapper", "args": ["serve"] }
  }
}
```

## More than one vault

knapper is single-vault: **one data directory holds one vault.** Pointing
`knapper index` at a different path replaces the index — the CLI warns first
(`Index was built for '…'. Re-indexing will replace it. Continue? [y/N]`), but
the answer is not to switch back and forth.

To keep several vaults, give each its own data directory. `KNAPPER_HOME` sets
it, and the `--data-dir` flag overrides that; absent both it is `~/.knapper`.
Everything derives from it — the store, `config.toml`, `vault.toml` and the
downloaded models — so one variable moves the whole set.

```bash
KNAPPER_HOME=~/.knapper-work    knapper index ~/vaults/work
KNAPPER_HOME=~/.knapper-fiction knapper index ~/vaults/fiction
```

Combine the two ideas and a workspace gets its own vault automatically. In a
project's `.mcp.json`:

```json
{
  "mcpServers": {
    "knapper": {
      "type": "stdio",
      "command": "knapper",
      "args": ["serve"],
      "env": { "KNAPPER_HOME": "/home/you/.knapper-work" }
    }
  }
}
```

Or with the CLI, where `-e` sets the same variable:

```bash
claude mcp add knapper -e KNAPPER_HOME=$HOME/.knapper-work -- knapper serve
```

Use an absolute path: `~` is not expanded in a JSON config value.

**Each data directory downloads its own models.** `models/` sits inside it, so
a second `KNAPPER_HOME` means a second copy of the embedder. Symlink the
directory at the new home to the one you already have if you would rather not
fetch it twice.

## HTTP Mode

```bash
knapper serve --http              # Port 3000
```
