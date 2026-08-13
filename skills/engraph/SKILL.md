---
name: engraph
description: Index and search document collections using hybrid semantic + graph + full-text search. Use when users need to search knowledge bases, find connections between documents, discover related content via link graphs, or query indexed markdown collections.
license: MIT
compatibility: Requires engraph CLI. Install via `brew install devwhodevs/tap/engraph` or from GitHub releases.
metadata:
  author: jsynowiec
  version: "1.0.0"
allowed-tools: Bash(engraph:*), mcp__engraph__*
---

# Engraph — Hybrid Semantic & Graph Search for Document Collections

Local knowledge engine for markdown document collections. Combines semantic embeddings, full-text search (BM25), wikilink graph traversal, temporal scoring, and cross-encoder reranking.

## Status

!`engraph --version 2>/dev/null || echo "Not installed: brew install devwhodevs/tap/engraph"`

## Indexing

```bash
engraph index /path/to/documents        # Incremental; only changed files re-embedded
engraph index /path/to/documents --rebuild
engraph status                          # File count, stats, index freshness
engraph clear                           # Drop the index (--all also removes models)
```

## Search

```bash
engraph search "how does the auth flow work"
engraph search "performance regressions last month" --explain
engraph search "architecture decisions" -n 5 --json
engraph search "warding" --all type/undead      # Only notes carrying the tag
```

| Flag              | Description                                    |
| ----------------- | ---------------------------------------------- |
| `-n, --top-n <N>` | Number of results (default: `top_n` in `config.toml`) |
| `--explain`       | Show per-lane RRF score breakdown              |
| `--json`          | Machine-readable JSON output                   |
| `--all`/`--any`/`--none` | Answer from the notes these tag terms admit. `--tags` is an alias of `--all` |

### Query Tips

- **Conceptual / vague**: Use natural language. The cross-encoder reads each candidate jointly with the query, so a full question works better than keywords.
- **Keyword-heavy**: Exact terms, identifiers, and names work well via the BM25 lane.
- **Temporal**: "last week", "yesterday", "March 2026" — the temporal lane activates automatically.

## Graph Inspection

```bash
engraph read "path/to/note.md"          # Content, metadata, incoming and outgoing links
engraph read "#docid"                   # By document ID
engraph status                          # Files, chunks, edges, wikilinks, mentions
```

## Context Queries

```bash
engraph topic "authentication" --budget 8000
engraph topic "warding" --all type/undead   # Gather the bundle from tagged notes only
engraph who "Person Name"
engraph project "Project Name"
engraph vault-map                 # Collection structure overview
engraph read "path/to/note.md"    # Full content + metadata
engraph read "path/to/note.md" --section "Action Items"   # One section
engraph list --tags architecture  # Filter by tags, folder, created_by, etc.
engraph tags --under type/        # The tag vocabulary, whole or under one term
```

## Writing

```bash
engraph create --content "# Meeting Notes" --tags meeting
engraph update "Meeting Notes" --section "Action Items" --mode append --content="- [ ] Follow up"
engraph update "Meeting Notes" --property tags --mode append --content "actionable"
engraph move "Meeting Notes" --new-folder 02-Areas
engraph archive "Old Draft"          # --undo restores it
engraph delete "Old Draft" --mode soft
```

> One capability, one name, three surfaces: `engraph tags` is the MCP `tags` tool and `GET /api/tags`; `engraph list` is the MCP `list` tool and `GET /api/list`. A CLI command's name becomes the MCP tool by writing `-` as `_`, and the HTTP route by putting it under `/api/`. The tag operators are `--all`/`--any`/`--none` on the CLI, spelled `all`/`any`/`none` on both other surfaces, with `tags` an alias of `all`.

> Health diagnostics (orphans, broken links, stale notes, tag hygiene) are `engraph health`, the MCP `health` tool and the HTTP `GET /api/health` endpoint — see `references/http-rest-api.md`.

## Setup

```bash
engraph index /path/to/documents
engraph search "your query"
```

## References

- `references/mcp-setup.md` — configure engraph as an MCP server (Claude Code, Claude Desktop).
- `references/http-rest-api.md` — HTTP REST API endpoints, authentication, and examples for web agents and scripts.
