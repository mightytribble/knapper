# Using knapper with Claude Code

Connect knapper to Claude Code so it can search, read, and write to your Obsidian vault.

## Setup

### 1. Install and index

```bash
brew install mightytribble/tap/knapper
knapper index ~/path/to/vault
```

### 2. Register the MCP server

```bash
claude mcp add --scope user knapper -- knapper serve
```

`--scope user` registers knapper for every project; `--scope project` writes a
shared `.mcp.json` in the current project's root instead.

Or add it by hand to `~/.claude.json` (user scope) or a project's `.mcp.json`
(project scope).

```json
{
  "mcpServers": {
    "knapper": {
      "type": "stdio",
      "command": "knapper",
      "args": ["serve"]
    }
  }
}
```

### 3. Start using

Claude Code now has access to 18 vault tools. Each one is named after the CLI
command it answers, with `-` written as `_`.

**Read tools:**
- `search` — hybrid search across the vault
- `read` — read a full note with metadata, or one section of it (`section`)
- `list` — every note the scope admits, in path order (tag terms, directory
  terms, creator); `detailed` adds each note's heading outline
- `tags` — the vault's tag vocabulary, whole or under one term
- `vault_map` — vault structure overview

**Write tools:**
- `create` — create a note with smart filing
- `update` — a list of edits to one note in one write: the body, a section
  (`section`) or a frontmatter property (`property`), each with a `mode` of
  `replace`, `prepend`, `append` or `remove`
- `move` — move a note to a different folder
- `archive` — soft-delete to the archive folder, or restore with `undo`
- `delete` — delete a note (`mode`: soft or hard)

**Index and diagnostic tools:**
- `index` — index the configured vault
- `reindex_file` — re-index one file after an edit made outside knapper
- `status` — index status and statistics
- `health` — vault health diagnostics, read from the index
- `validate` — check vault markdown for structural and indexing problems —
  one note, a scope, or the whole vault — read from disk, with no model loaded
- `identity` — user identity (L0) and current context (L1)
- `init` — first-time onboarding (`mode`: detect or apply)
- `migrate` — PARA migration (`mode`: preview, apply or undo)

## Example interactions

**"What do I know about authentication?"**
Claude will call `search("authentication")` and get results from semantic, keyword, and graph lanes.

**"What is in my projects folder?"**
Claude will call `list` with `scope: ["/01-Projects/"]` and `detailed: true` to see every note there with its heading outline, then `read` the ones that matter.

**"Create a meeting note for today's standup"**
Claude will call `create` with content, tags, and type hint. knapper resolves tags against your registry, discovers wikilinks in the content, and places the note in the best folder.

## Real-time sync

The MCP server includes a file watcher. When you edit notes in Obsidian, knapper re-indexes them automatically (2-second debounce). No need to manually re-run `knapper index`.

## Tips

- Scope a `search` or `list` with `all`, `any` and `none` — a tag term such as `type/person`, or a directory term such as `/03-Resources/People/` — to keep a query inside one part of the vault; `tags` lists the vocabulary to scope on
- `vault_map` helps Claude understand your vault structure before searching
- `read` with a `section` returns one heading's content instead of the whole note, and `read` with `metadata` returns the note's frontmatter and its links both ways
- The `--explain` flag on CLI search shows per-lane score breakdown — useful for debugging search quality
