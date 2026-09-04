---
name: knapper
description: Index and search document collections using hybrid semantic + graph + full-text search. Use when users need to search knowledge bases, find connections between documents, discover related content via link graphs, or query indexed markdown collections.
license: MIT
compatibility: Requires knapper CLI. Install via `brew install mightytribble/tap/knapper` or from GitHub releases.
metadata:
  author: jsynowiec
  version: "1.0.0"
allowed-tools: Bash(knapper:*), mcp__knapper__*
---

# Knapper — Hybrid Semantic & Graph Search for Document Collections

Local knowledge engine for markdown document collections. Combines semantic embeddings, full-text search (BM25), wikilink graph traversal, temporal scoring, and cross-encoder reranking.

## Status

!`knapper --version 2>/dev/null || echo "Not installed: brew install mightytribble/tap/knapper"`

## Indexing

```bash
knapper index /path/to/documents        # Incremental; only changed files re-embedded
knapper index /path/to/documents --rebuild
knapper status                          # File count, stats, index freshness
knapper clear                           # Drop the index (--all also removes models)
```

## Search

```bash
knapper search "how does the auth flow work"
knapper search "performance regressions last month" --explain
knapper search "architecture decisions" -n 5 --json
knapper search "warding" --all type/undead      # Only notes carrying the tag
```

| Flag              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `-n, --top-n <N>` | Number of results (default: `top_n` in `config.toml`) |
| `--explain`       | Show per-lane RRF score breakdown              |
| `--json`          | Machine-readable JSON output                   |
| `--all`/`--any`/`--none` | Answer from the notes these tag terms admit. `--scope` is an alias of `--all` |
| `--property`/`--links-to`/`--linked-from` | Answer from notes carrying a property (`NAME` or `NAME=VALUE`), linking to a note, or linked from one; one value each |

### Query Tips

- **Conceptual / vague**: Use natural language. The cross-encoder reads each candidate jointly with the query, so a full question works better than keywords.
- **Keyword-heavy**: Exact terms, identifiers, and names work well via the BM25 lane.
- **Temporal**: "last week", "yesterday", "March 2026" — the temporal lane activates automatically.

## Graph Inspection

```bash
knapper read "path/to/note.md"          # Content, metadata, incoming and outgoing links
knapper read "#docid"                   # By document ID
knapper status                          # Files, chunks, edges, wikilinks, mentions
```

## Context Queries

```bash
knapper topic "authentication" --budget 8000
knapper topic "warding" --all type/undead   # Gather the bundle from tagged notes only
knapper who "Person Name"
knapper project "Project Name"
knapper vault-map                 # Collection structure overview
knapper read "path/to/note.md"    # Full content + metadata
knapper read "path/to/note.md" --section "Action Items"   # One section
knapper list --scope architecture      # Every document the scope admits, one path per line
knapper list --scope /locations/ --detailed  # ...each with its heading outline
knapper tags --under type/        # The tag vocabulary, whole or under one term
knapper properties                # The custom-property registry: name, note count, kinds, declared type
knapper properties --name status  # One property's values with counts — call before --property NAME=VALUE
knapper list --property status=draft
knapper list --property employer --links-to Acme   # Links filed under one property
knapper read "ada.md" --metadata  # ...also lists the note's properties and names the property behind each link
```

`topic` fills a character budget with whole documents: the five that best match the query, and then the documents one wikilink hop from the top three. It returns documents and not sections, and no cross-encoder scores them, so `search` ranks more accurately. `who` returns a person's document, the documents that mention them, and their wikilinks in both directions; the mention list needs a People folder in `vault.toml`, and without one the bundle holds the document and its links alone.

## Writing

```bash
knapper create --content "# Meeting Notes" --tags meeting
knapper update "Meeting Notes" --section "Action Items" --mode append --content="- [ ] Follow up"
knapper update "Meeting Notes" --property tags --mode append --content "actionable"
knapper move "Meeting Notes" --new-folder 02-Areas
knapper archive "Old Draft"          # --undo restores it
knapper delete "Old Draft" --mode soft
```

For a list-valued property, prefer `--mode append` and `--mode remove` over `replace`: those two cannot drop a sibling value the model failed to reproduce.

> One capability, one name, three surfaces: `knapper tags` is the MCP `tags` tool and `GET /api/tags`; `knapper list` is the MCP `list` tool and `GET /api/list`. A CLI command's name becomes the MCP tool by writing `-` as `_`, and the HTTP route by putting it under `/api/`. The tag operators are `--all`/`--any`/`--none` on the CLI, spelled `all`/`any`/`none` on both other surfaces, with `scope` an alias of `all`. `list` answers every document the scope admits, in path order and with no default cap, and `detailed` adds each document's heading outline. `knapper properties` is the MCP `properties` tool and `GET /api/properties`. The property filters are `--property`, `--links-to` and `--linked-from` on the CLI, spelled `property`, `links_to` and `linked_from` on both other surfaces, one value each.

> Health diagnostics (orphans, broken links, stale notes, tag hygiene) are `knapper health`, the MCP `health` tool and the HTTP `GET /api/health` endpoint — see `references/http-rest-api.md`.

## Setup

```bash
knapper index /path/to/documents
knapper search "your query"
```

## References

- `references/mcp-setup.md` — configure knapper as an MCP server (Claude Code, Claude Desktop).
- `references/http-rest-api.md` — HTTP REST API endpoints, authentication, and examples for web agents and scripts.
