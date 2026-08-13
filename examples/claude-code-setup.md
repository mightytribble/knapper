# Using engraph with Claude Code

Connect engraph to Claude Code so it can search, read, and write to your Obsidian vault.

## Setup

### 1. Install and index

```bash
brew install devwhodevs/tap/engraph
engraph index ~/path/to/vault
```

### 2. Add to Claude Code settings

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "engraph": {
      "command": "engraph",
      "args": ["serve"]
    }
  }
}
```

### 3. Start using

Claude Code now has access to 20 vault tools. Each one is named after the CLI
command it answers, with `-` written as `_`.

**Read tools:**
- `search` — hybrid search across the vault
- `read` — read a full note with metadata, or one section of it (`section`)
- `list` — every note the scope admits, in path order (tag terms, directory
  terms, creator); `detailed` adds each note's heading outline
- `tags` — the vault's tag vocabulary, whole or under one term
- `vault_map` — vault structure overview
- `who` — person context bundle (their note, the notes mentioning them, their
  wikilinks both ways; the mention list needs a People folder in `vault.toml`)
- `project` — project context bundle
- `topic` — whole notes on a topic, and their one-hop neighbours, within a
  character budget

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
- `reindex_file` — re-index one file after an edit made outside engraph
- `status` — index status and statistics
- `health` — vault health diagnostics
- `identity` — user identity (L0) and current context (L1)
- `init` — first-time onboarding (`mode`: detect or apply)
- `migrate` — PARA migration (`mode`: preview, apply or undo)

## Example interactions

**"What do I know about authentication?"**
Claude will call `search("authentication")` and get results from semantic, keyword, and graph lanes.

**"Who is working on the API project?"**
Claude will call `project("API")` to get the project bundle — related notes, team members, active tasks.

**"Create a meeting note for today's standup"**
Claude will call `create` with content, tags, and type hint. engraph resolves tags against your registry, discovers wikilinks in the content, and places the note in the best folder.

## Real-time sync

The MCP server includes a file watcher. When you edit notes in Obsidian, engraph re-indexes them automatically (2-second debounce). No need to manually re-run `engraph index`.

## Tips

- Use `topic` with a `budget` (for example 8000 characters, about 2000 tokens) to feed whole notes into a prompt; use `search` when you want the best-ranked passages instead
- `vault_map` helps Claude understand your vault structure before searching
- `who("Person Name")` gathers a person's note with everything that names or links to it, which is one call for someone's involvement across the vault
- The `--explain` flag on CLI search shows per-lane score breakdown — useful for debugging search quality
