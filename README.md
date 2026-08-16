<p align="center">
  <img src="assets/logo.png" alt="engraph logo" width="180">
</p>

<h1 align="center">engraph — Vault Intelligence for AI Agents</h1>

<p align="center"><strong>Turn your Obsidian vault into a knowledge API.</strong> 5-lane hybrid search, MCP server, HTTP REST API, ChatGPT Actions — all local, all offline.</p>

[![CI](https://github.com/devwhodevs/engraph/actions/workflows/ci.yml/badge.svg)](https://github.com/devwhodevs/engraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/devwhodevs/engraph)](https://github.com/devwhodevs/engraph/releases)

engraph turns your markdown vault into a searchable knowledge graph that any AI agent can query — Claude Code via [MCP](https://modelcontextprotocol.io), ChatGPT via [Actions](https://platform.openai.com/docs/actions), or any tool via REST API. It combines semantic embeddings, full-text search, wikilink graph traversal, temporal awareness, and LLM-powered reranking into a single local binary. Same model stack as [qmd](https://github.com/tobi/qmd). No API keys, no cloud — everything runs on your machine.

<p align="center">
  <img src="assets/demo.gif" alt="engraph demo: 4-lane hybrid search with LLM intelligence, person context bundles, Metal GPU" width="800">
</p>

## Why engraph?

Plain vector search treats your notes as isolated documents. But knowledge isn't flat — your notes link to each other, share tags, reference the same people and projects. engraph understands these connections.

- **5-lane hybrid search** — semantic embeddings + BM25 full-text + graph expansion + cross-encoder reranking + temporal scoring, fused via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Lane weights are configurable. Time-aware queries like "what happened last week" or "March 2026 notes" activate the temporal lane automatically.
- **MCP server for AI agents** — `engraph serve` exposes 20 tools (search, read, list, tags, vault_map, who, project, topic, create, update, delete, move, archive, index, reindex_file, status, health, identity, init, migrate) that Claude, Cursor, or any MCP client can call directly.
- **HTTP REST API** — `engraph serve --http` adds an axum-based HTTP server alongside MCP with 21 REST endpoints, API key authentication, rate limiting, and CORS. Web-based agents and scripts can query your vault with simple `curl` calls.
- **Section-level editing** — AI agents can read, replace, prepend, or append to a section by heading, to the note's body, or to a frontmatter property — every change is one `update` call carrying a list of edits.
- **Vault health diagnostics** — detect orphan notes, broken wikilinks, stale content, and tag hygiene issues. Available as MCP tool and CLI command.
- **Obsidian CLI integration** — auto-detects running Obsidian and delegates compatible operations. Circuit breaker (Closed/Degraded/Open) ensures graceful fallback.
- **Real-time sync** — file watcher keeps the index fresh as you edit in Obsidian. No manual re-indexing needed.
- **Smart write pipeline** — AI agents can create, edit, rewrite, and delete notes with automatic tag resolution, wikilink discovery, and folder placement based on semantic similarity.
- **Fully local** — [llama.cpp](https://github.com/ggml-org/llama.cpp) inference with GGUF models (~300MB mandatory, ~650MB optional for the cross-encoder). Metal GPU-accelerated on macOS (88 files indexed in 70s). No API keys, no cloud.

## What problem it solves

You have hundreds of markdown notes. You want your AI coding assistant to understand what you've written — not just search keywords, but follow the connections between notes, understand context, and write new notes that fit your vault's structure.

Existing options are either cloud-dependent (Notion AI, Mem), limited to keyword search (Obsidian's built-in), or require you to copy-paste context manually. engraph gives AI agents direct, structured access to your entire vault through a standard protocol.

## How it works

```
Your vault (markdown files)
        │
        ▼
┌─────────────────────────────────────────────┐
│              engraph index                   │
│                                             │
│  Walk → Chunk → Embed (llama.cpp) → Store   │
│                                             │
│  SQLite: files, chunks, FTS5, vectors,      │
│          edges, centroids, tags, LLM cache  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              engraph serve                   │
│                                             │
│  MCP Server (stdio) + File Watcher          │
│  + HTTP REST API (--http, optional)         │
│                                             │
│  Search: 3-lane retrieval → Reranker        │
│          → Two-pass RRF fusion              │
│                                             │
│  20 MCP tools + 21 REST endpoints           │
└─────────────────────────────────────────────┘
        │
        ▼
  Claude / Cursor / any MCP client / curl / web agents
```

1. **Index** — walks your vault, chunks markdown by headings, embeds with a local GGUF model via llama.cpp (Metal GPU on macOS), stores everything in SQLite with FTS5 + sqlite-vec + a wikilink graph
2. **Search** — runs the query through up to five lanes (semantic KNN, BM25 keyword, graph expansion, cross-encoder reranking, temporal scoring), fused via RRF at configurable per-lane weights
3. **Serve** — starts an MCP server that AI agents connect to, with a file watcher that re-indexes changes in real time

## Quick start

**Install:**

```bash
# Homebrew (macOS)
brew install devwhodevs/tap/engraph

# Pre-built binaries (macOS arm64, Linux x86_64)
# → https://github.com/devwhodevs/engraph/releases

# From source (requires CMake for llama.cpp)
cargo install --git https://github.com/devwhodevs/engraph
```

**Index your vault:**

```bash
engraph index ~/path/to/vault
# Downloads embedding model on first run (~300MB)
# Incremental — only re-embeds changed files on subsequent runs
```

**Search:**

```bash
engraph search "how does the auth system work"
```

```
 1. [97%] 02-Areas/Development/Auth-Architecture.md > # Auth Architecture  #6e1b70
    OAuth 2.0 with PKCE for all client types. Session tokens stored in HTTP-only cookies...

 2. [95%] 01-Projects/API-Design.md > # API Design  #e3e350
    All endpoints require Bearer token authentication. Tokens are issued by the OAuth 2.0...

 3. [91%] 03-Resources/People/Sarah-Chen.md > # Sarah Chen  #4adb39
    Senior Backend Engineer. Tech lead for authentication and security systems...
```

Note how result #3 was found via **graph expansion** — Sarah's note doesn't mention "auth system" directly, but she's linked from the auth architecture doc via `[[Sarah Chen]]`.

**Claude Code** — Install the plugin (recommended):

```bash
claude plugin marketplace add devwhodevs/engraph
claude plugin install engraph@engraph
```

**Connect to Claude Code:**

Or configure MCP manually in `~/.claude/settings.json`:

```bash
{
  "mcpServers": {
    "engraph": {
      "command": "engraph",
      "args": ["serve"]
    }
  }
}
```

Now Claude can search your vault, read notes, build context bundles, and create new notes — all through structured tool calls.

**AI Agent Skills** — Install the Engraph skills using the skills CLI (recommended):

```bash
npx skills add devwhodevs/engraph
```

**Enable HTTP REST API:**

```bash
# Start MCP + HTTP server on port 3000
engraph serve --http

# Custom port and host
engraph serve --http --port 8080 --host 0.0.0.0

# Local development without API keys (127.0.0.1 only)
engraph serve --http --no-auth
```

**API key management:**

```bash
# Add a new API key (read or write permission)
engraph configure --add-api-key

# List existing keys
engraph configure --list-api-keys

# Revoke a key
engraph configure --revoke-api-key eg_abc123...
```

**Enable intelligence (optional, ~650MB download):**

```bash
engraph configure --enable-intelligence
# Downloads Qwen3-Reranker (cross-encoder)
# Adds the reranker lane to search
```

## Example usage

**Search with the cross-encoder lane:**

```bash
engraph search "how does authentication work" --explain
```
```
 1. [97%] 01-Projects/API-Design.md > # API Design  #e3e350
    All endpoints require Bearer token authentication...

--- Explain ---
01-Projects/API-Design.md
  RRF: 0.0387
    semantic: rank #2, raw 0.38, +0.0194
    rerank: rank #2, raw 0.01, +0.0194
02-Areas/Development/Auth-Architecture.md
  RRF: 0.0384
    semantic: rank #1, raw 0.51, +0.0197
    rerank: rank #4, raw 0.00, +0.0187
```

The reranker scored each result for relevance as the 4th RRF lane.

**Rich context for AI agents:**

```bash
engraph topic "authentication" --budget 8000
```

Returns a context bundle to paste into a prompt or serve over MCP: the five notes that best match the query, each read whole, and then the notes one wikilink hop from the top three. The budget is characters, so 8000 of them are about 2000 tokens, and a note that overruns it is cut and marked. A bundle holds whole notes, so the query runs at one result per note, and no cross-encoder scores it: `engraph search` is the sharper ranking. Tag terms hold both steps to the notes they admit, and a term starting with `/` is a directory path from the vault root instead, a trailing `/` its subtree:

```bash
engraph topic "warding" --all type/undead --budget 8000
```

**Person context:**

```bash
engraph who "Sarah Chen"
```

Returns Sarah's note, the notes that mention her, and her wikilinks in both directions. The name resolves as a docid, a path or a basename, and then by keyword search, which prefers a hit under the People folder. The mention list comes from that folder: engraph writes a mention edge when a note's text holds a person note's filename or one of its frontmatter aliases. A vault whose `vault.toml` names no People folder has no mention edges, and `who` answers with the note and its links alone.

**Vault structure overview:**

```bash
engraph vault-map
```

Returns folder counts, top tags, recent files — gives an AI agent orientation before it starts searching.

**See what the vault holds:**

```bash
engraph list --scope /locations/ --detailed
```
```
locations/aurelian-empire.md
# About the Empire
## History
### The founding of the Empire
## Current Events
```

One bare path per line, in path order, so a folder's notes arrive together and `wc -l` is the total. A bare `engraph list` answers every indexed note; `--scope`, `--all`, `--any` and `--none` narrow it by tag or by directory, and `--limit` keeps the first n. `--detailed` reads each listed note and prints its headings, which is how an agent finds the section to read or write before it calls `read` or `update`.

**Create a note via the write pipeline:**

```bash
engraph create --content "# Meeting Notes\n\nDiscussed auth timeline with Sarah." --tags meeting,auth
```

engraph resolves tags against the registry (fuzzy matching), discovers potential wikilinks (`[[Sarah Chen]]`), suggests the best folder based on semantic similarity to existing notes, and writes atomically.

**Edit a specific section:**

```bash
engraph update "Meeting Notes" --section "Action Items" --mode append --content="- [ ] Follow up with Sarah"
```

Targets the "Action Items" section by heading, appends content without touching the rest of the note. Write `--content=` with an equals sign when the value starts with a `-`: the shell passes the text through untouched, and clap reads a leading `-` as a flag.

**Rewrite a note (preserves frontmatter):**

```bash
engraph update "Meeting Notes" --mode replace --content "# Meeting Notes\n\nRevised content here."
```

Replaces the entire body while keeping existing frontmatter (tags, dates, metadata) intact.

**Edit frontmatter:**

```bash
engraph update "Meeting Notes" --property tags --mode append --content "actionable"
```

A property takes `--mode replace`, `append` or `remove`; a body or a section takes `replace`, `prepend` or `append`. Repeat `--content` to write a list-valued property such as tags or aliases.

**Delete a note:**

```bash
engraph delete "Old Draft" --mode soft   # moves to archive
engraph delete "Old Draft" --mode hard   # permanent removal
```

**Check vault health:**

```bash
engraph health
```

Returns orphan notes (no links in or out), broken wikilinks, stale notes, and tag hygiene issues.

## HTTP REST API

`engraph serve --http` adds a full REST API alongside the MCP server, exposing the same capabilities over HTTP for web agents, scripts, and integrations.

**21 endpoints:**

Every capability is one route, and the route is the CLI command's name under `/api/`. `docs/surfaces.md` is the generated table of all three surfaces.

| Method | Endpoint | Permission | Description |
|--------|----------|------------|-------------|
| GET | `/api/health-check` | read | Server health check |
| POST | `/api/search` | read | Hybrid search (semantic + FTS5 + graph + reranker + temporal), scoped by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`) |
| GET | `/api/read` | read | Read a note (`file`), or one of its sections (`section`) |
| GET | `/api/list` | read | List notes by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`), creator, limit, and `detailed=true` for each note's heading outline |
| GET | `/api/tags` | read | The tag vocabulary, whole or under one term (`under`) |
| GET | `/api/vault-map` | read | Vault structure overview (folders, tags, recent files) |
| GET | `/api/who` | read | Person context bundle (`name`) |
| GET | `/api/project` | read | Project context bundle (`name`) |
| POST | `/api/topic` | read | Topic context bundle: whole notes and their one-hop neighbours, within a character budget |
| GET | `/api/status` | read | Index status and statistics |
| GET | `/api/health` | read | Vault health diagnostics |
| POST | `/api/create` | write | Create a new note |
| POST | `/api/update` | write | Apply a list of edits to one note in one write |
| POST | `/api/move` | write | Move note to different folder |
| POST | `/api/archive` | write | Archive a note, or restore one with `undo` |
| POST | `/api/delete` | write | Delete note (soft or hard) |
| POST | `/api/index` | write | Index the configured vault |
| POST | `/api/reindex-file` | write | Re-index a single file after external edits |
| GET | `/api/identity` | read | User identity (L0) and current context (L1). `?refresh=true` re-extracts the L1 facts and takes a write key |
| POST | `/api/init` | write | First-time onboarding setup (`mode`: detect or apply) |
| POST | `/api/migrate` | write | PARA migration (`mode`: preview, apply or undo). `apply` requires the `preview` that `preview` returned |

**Authentication:**

All requests require an API key via the `Authorization` header:

```bash
curl -H "Authorization: Bearer eg_abc123..." http://localhost:3000/api/vault-map
```

Keys have either `read` or `write` permission. Write keys can access all endpoints; read keys are restricted to read-only endpoints. Use `--no-auth` for local development without keys (127.0.0.1 only).

**curl examples:**

```bash
# Search
curl -X POST http://localhost:3000/api/search \
  -H "Authorization: Bearer eg_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication architecture", "top_n": 5}'

# Search, scoped to a tag or directory filter (scope/all, any, none; a
# leading / reads a term as a directory path from the vault root)
curl -X POST http://localhost:3000/api/search \
  -H "Authorization: Bearer eg_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication architecture", "top_n": 5, "scope": ["project/auth", "/01-Projects/"], "all": ["type/decision"], "any": ["status/reviewed", "status/draft"], "none": ["status/archived"]}'

# Read a note, or one of its sections
curl "http://localhost:3000/api/read?file=01-Projects/API-Design.md" \
  -H "Authorization: Bearer eg_..."
curl "http://localhost:3000/api/read?file=01-Projects/API-Design.md&section=Endpoints" \
  -H "Authorization: Bearer eg_..."

# Create a note
curl -X POST http://localhost:3000/api/create \
  -H "Authorization: Bearer eg_..." \
  -H "Content-Type: application/json" \
  -d '{"content": "# Meeting Notes\n\nDiscussed auth timeline.", "tags": ["meeting", "auth"]}'
```

**Rate limiting:** Configurable per-key token bucket (requests per minute). Defaults to 60 req/min. Returns `429 Too Many Requests` when exceeded.

**CORS:** Configurable allowed origins in `config.toml` under `[http]`. Defaults to allow all origins for local development.

```toml
[http]
port = 3000
host = "127.0.0.1"
cors_origins = ["http://localhost:3000", "https://myapp.example.com"]
rate_limit = 60

[[http.api_keys]]
key = "eg_..."
permission = "write"
```

## PARA Migration

`engraph migrate` restructures your vault into the [PARA method](https://fortelabs.com/blog/para/) (Projects, Areas, Resources, Archive) using heuristic classification. The workflow is non-destructive: preview first, review the plan, then apply.

**Workflow:**

```bash
# 1. Preview — classify notes and generate a migration plan
engraph migrate --mode preview
# Outputs: markdown summary + JSON plan saved to ~/.engraph/

# 2. Review the plan (edit if needed)
cat ~/.engraph/migration_preview.md

# 3. Apply — move files according to the plan
engraph migrate --mode apply

# 4. Undo — reverse the last migration if something looks wrong
engraph migrate --mode undo
```

**Classification signals:**

| Category | Detection signals |
|----------|-------------------|
| **Projects** | Open tasks (`- [ ]`), active/in-progress status in frontmatter, project tags |
| **Areas** | Recurring topic keywords (health, finance, career, learning), area-related tags |
| **Resources** | People notes (People folder, person-like content), reference material, articles, code snippets |
| **Archive** | Done/completed/inactive status, no incoming or outgoing wikilinks, stale content |

Notes that don't match any signal with sufficient confidence stay in place. Daily notes (`YYYY-MM-DD.md`) and templates are always skipped.

**MCP tool:** `migrate`, with the same `mode` — available in `engraph serve` for AI-assisted migration.

**HTTP endpoint:** `POST /api/migrate`, with the same `mode` — available via `engraph serve --http`. On both servers `mode: apply` takes the `preview` that `mode: preview` returned; only the CLI reads the copy saved on disk.

## ChatGPT Actions

Connect your Obsidian vault to ChatGPT as a custom GPT Action. ChatGPT can search, read, create, and edit your notes through engraph's REST API.

### Prerequisites

- engraph installed and indexed (`engraph index ~/your-vault`)
- A tunnel tool: [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/) (recommended) or [ngrok](https://ngrok.com)

### Step 1: Configure engraph

```bash
# Interactive setup — enables HTTP, creates API key, sets CORS
engraph configure --setup-chatgpt
```

Or configure manually in `~/.engraph/config.toml`:

```toml
[http]
enabled = true
port = 3000
host = "127.0.0.1"
rate_limit = 60
cors_origins = ["https://chat.openai.com", "https://chatgpt.com"]

[[http.api_keys]]
key = "eg_your_key_here"    # generate with: engraph configure --add-api-key --key-name chatgpt --key-permissions write
name = "chatgpt"
permissions = "write"        # "read" for search-only, "write" to also create/edit notes

[http.plugin]
name = "My Vault"
description = "Search and manage my Obsidian vault"
public_url = "https://your-tunnel-url.trycloudflare.com"   # set after starting tunnel
```

### Step 2: Start engraph + tunnel

**Terminal 1 — engraph HTTP server:**
```bash
engraph serve --http
```

**Terminal 2 — Cloudflare tunnel:**
```bash
cloudflared tunnel --url http://localhost:3000
# Prints a URL like: https://abc-xyz.trycloudflare.com
```

Or with ngrok:
```bash
ngrok http 3000
# Prints a URL like: https://abc123.ngrok-free.app
```

### Step 3: Update config with tunnel URL

Edit `~/.engraph/config.toml` and set `public_url` to your tunnel URL:

```toml
[http.plugin]
public_url = "https://abc-xyz.trycloudflare.com"
```

Then restart engraph (`Ctrl+C` and re-run `engraph serve --http`). This ensures the OpenAPI spec points to the correct public URL.

### Step 4: Verify endpoints

```bash
# Both should return JSON (no auth required)
curl https://your-tunnel-url/openapi.json
curl https://your-tunnel-url/.well-known/ai-plugin.json

# Search with auth
curl -X POST -H "Authorization: Bearer eg_your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "test search"}' \
  https://your-tunnel-url/api/search
```

### Step 5: Register in ChatGPT

1. Go to [ChatGPT](https://chat.openai.com) → **Explore GPTs** → **Create**
2. Give your GPT a name (e.g., "Vault Assistant")
3. Add these **Instructions**:

```
You are a knowledge assistant connected to the user's Obsidian vault via engraph.

WORKFLOW:
1. Use searchVault to find relevant notes before answering questions
2. Use readNote for full content, and its section parameter for one heading
3. Use getWho for people context, getProject for project context
4. Use getVaultMap to orient yourself in the vault structure
5. Only create or edit notes when explicitly asked

SEARCH TIPS:
- Temporal queries ("last week", "yesterday") activate time-aware search automatically
- Results include confidence % — prefer higher confidence matches
- Fuzzy matching works: typos in names are handled

STYLE:
- Reference vault notes by name when answering
- Quote relevant snippets
- If information isn't in the vault, say so clearly
- Be concise
```

4. Click **Add Action** → **Import from URL**
5. Enter: `https://your-tunnel-url/openapi.json`
6. Click the **gear icon** next to Authentication
7. Select **API Key**, Auth Type: **Bearer**
8. Paste your API key (the `eg_...` key from Step 1)
9. **Save** and test

### Conversation starters

- "What happened in my vault last week?"
- "Summarize my current work projects"
- "Find notes related to [topic]"
- "Create a note about today's meeting with [person]"

### Notes

- **Tunnel URLs are temporary** (Cloudflare quick tunnels change on restart). For persistent URLs, set up a [named Cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/) or use ngrok with a reserved domain.
- **Read-only mode**: set `permissions = "read"` on the API key if you don't want ChatGPT to create or modify notes.
- **Rate limiting**: default is 60 requests/minute per key. Adjust `rate_limit` in config if needed.
- **engraph must be running** on your machine for ChatGPT to access it. If you close the terminal, the connection drops.

## Use cases

**AI-assisted knowledge work** — Give Claude or Cursor deep access to your personal knowledge base. Instead of copy-pasting context, the agent searches, reads, and cross-references your notes directly.

**Developer second brain** — Index architecture docs, decision records, meeting notes, and code snippets. Search by concept across all of them.

**Research and writing** — Find connections between notes that you didn't explicitly link. The graph lane surfaces related content through shared wikilinks and mentions.

**Team knowledge graphs** — Index a shared docs vault. AI agents can answer "who knows about X?" and "what decisions were made about Y?" by traversing the note graph.

## How it compares

| | engraph | Basic RAG (vector-only) | Obsidian search |
|---|---|---|---|
| Search method | 5-lane RRF (semantic + BM25 + graph + reranker + temporal) | Vector similarity only | Keyword only |
| Query understanding | Cross-encoder reads each candidate jointly with the query | None | None |
| Understands note links | Yes (wikilink graph traversal) | No | Limited (backlinks panel) |
| AI agent access | MCP server (20 tools) + HTTP REST API (21 endpoints) | Custom API needed | No |
| Write capability | Create/edit/rewrite/delete with smart filing | No | Manual |
| Vault health | Orphans, broken links, stale notes, tag hygiene | No | Limited |
| Real-time sync | File watcher, 2s debounce | Manual re-index | N/A |
| Runs locally | Yes, llama.cpp + Metal GPU | Depends | Yes |
| Setup | One binary, one command | Framework + code | Built-in |

engraph is not a replacement for Obsidian — it's the intelligence layer that sits between your vault and your AI tools.

## Current capabilities

- 5-lane hybrid search (semantic + FTS5 + graph + cross-encoder reranker + temporal) with two-pass RRF fusion
- Temporal search: natural language date queries ("last week", "March 2026", "recent"), date extraction from frontmatter and filenames, smooth decay scoring
- Confidence % display: search results show normalized 0-100% confidence instead of raw RRF scores
- llama.cpp inference via Rust bindings (GGUF models, Metal GPU on macOS, CUDA on Linux)
- Intelligence opt-in: the cross-encoder lane is off unless enabled
- MCP server with 20 tools (8 read, 5 write, 5 index and diagnostic, 1 setup, 1 migrate) via stdio
- HTTP REST API with 21 endpoints, API key auth (`eg_` prefix), rate limiting, CORS — enabled via `engraph serve --http`
- User identity with L0/L1 tiered context for AI agent session starts
- Section-level reading and editing: target specific headings with replace/prepend/append modes
- Full note rewriting with automatic frontmatter preservation
- Frontmatter property edits: replace, append to or remove a property, scalar or list-valued
- Soft delete (archive) and hard delete (permanent) with audit logging
- Vault health diagnostics: orphan notes, broken wikilinks, stale content, tag hygiene
- Obsidian CLI integration with circuit breaker (Closed/Degraded/Open) for resilient delegation
- Real-time file watching with 2s debounce, startup reconciliation, and watcher coordination to prevent double re-indexing
- Write pipeline: tag resolution, fuzzy link discovery, semantic folder placement
- Context engine: a topic bundle of whole notes within a character budget, plus person and project bundles
- Vault graph: bidirectional wikilink + mention edges with multi-hop expansion
- Placement correction learning from user file moves
- Enhanced file resolution with fuzzy Levenshtein matching fallback
- Content-based folder role detection (people, daily, archive) by content patterns
- PARA migration: AI-assisted vault restructuring into Projects/Areas/Resources/Archive with preview, apply, and undo workflow
- Configurable model overrides for multilingual support
- 846 unit tests, CI on macOS + Ubuntu

## Roadmap

- [x] ~~LLM reranker — optional local model for result quality~~ (v1.0)
- [x] ~~MCP edit/rewrite tools — full note editing for AI agents~~ (v1.1)
- [x] ~~Vault health monitor — orphan notes, broken links, stale content, tag hygiene~~ (v1.1)
- [x] ~~Obsidian CLI integration — auto-detect and delegate with circuit breaker~~ (v1.1)
- [x] ~~Temporal search — find notes by time period, date-aware queries~~ (v1.2)
- [x] ~~HTTP/REST API — complement MCP with a standard web API~~ (v1.3)
- [x] ~~PARA migration — AI-assisted vault restructuring with preview/apply/undo~~ (v1.4)
- [x] ~~ChatGPT Actions — OpenAPI 3.1.0 spec + plugin manifest + `--setup-chatgpt` helper~~ (v1.5)
- [x] ~~Identity — user context at session start, enhanced onboarding~~ (v1.6)
- [ ] Timeline — temporal knowledge graph with point-in-time queries (v1.7)
- [ ] Mining — automatic fact extraction from vault notes (v1.8)

## Configuration

Optional config at `~/.engraph/config.toml`:

```toml
vault_path = "~/Documents/MyVault"
top_n = 10
exclude = [".obsidian/", "node_modules/", ".git/", "*-index.md", "templates/"]

# Results address sections, not whole documents. "file" restores one result
# per document; max_chunks_per_file bounds how much of one document can fill
# a page of results (0 = unlimited).
group_by = "chunk"
max_chunks_per_file = 3

# Enable the cross-encoder rerank lane
intelligence = true

# What each lane's rank is worth to RRF fusion. The values below are the
# defaults. They are query-time, so a sweep costs no re-index. Under the
# default ranking stage the cross-encoder sorts the shortlist, so only
# `semantic` and `fts` decide anything; the rest are read by
# `[ranking] mode = "legacy"`, where all five lanes vote.
[lane_weights]
# semantic = 1.0
# fts = 1.0
# graph = 0.8
# rerank = 1.0
# temporal = 0.0

# Adjacent sections of one document are returned as a single block. Default
# on; set to false for per-section results.
[ranking]
# coalesce_adjacent = true

# Prepend document identity to each chunk before embedding. Off by default —
# it helped conceptual queries and hurt exact-name lookup on the test vault.
# Needs `engraph index --rebuild` to take effect either way.
[embedding_prefix]
enabled = false
# path = true      # "Archdragon — lore/bestiary/archdragon.md"
# heading = true   # "Abilities > Combat" — ancestor sections
# tags = true
# aliases = true

# Override models for multilingual or custom use
[models]
# embed = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/qwen3-embedding-0.6b-q8_0.gguf"
# rerank = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"

# Obsidian CLI integration (auto-detected during init)
[obsidian]
# enabled = true
# cli_path = "/usr/local/bin/obsidian"

# Registered AI agents
[agents]
# names = ["claude-code", "cursor"]
```

Search returns **sections**. A note whose "Counterspell" and "Dispel Magic" sections both answer a query contributes both, up to `max_chunks_per_file`, and each result names the heading it came from. Pass `--group-by file` (or set `group_by = "file"`) for one result per document, representing it by its best-matching section.

`embedding_prefix` puts the document's name, path, tags, aliases and the chunk's ancestor headings into the text that is **embedded**, while what is stored, displayed and keyword-matched stays the raw chunk. It is off by default: because the prefix is the same string for every chunk of a document, it separates documents from each other at the cost of separating a document's own sections, and on the test vault that lost more on exact-name lookup than it gained on conceptual queries (`eval/probes.md`). Each component is switchable so the trade can be measured per vault.

The keyword lane indexes each chunk's **full text**. `chunks.snippet` — the leading 200 characters — is the display field only; it is not what BM25 searches.

`exclude` takes `.gitignore`-style globs, matched against paths relative to the vault root. A pattern with no `/` matches at any depth (`*-index.md` catches `lore/lore-index.md`); a trailing `/` means a directory and everything under it; an embedded `/` anchors the pattern to the vault root (`drafts/**`). Excluding a path that is already indexed removes it from the store — chunks, vectors, FTS entries and graph edges — on the next index run.

All data stored in `~/.engraph/` — single SQLite database (~10MB typical), GGUF models, and vault profile.

## Development

```bash
cargo test --lib          # 852 unit tests, no network (requires CMake for llama.cpp)
cargo clippy -- -D warnings
cargo fmt --check
```

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

The codebase is 35 Rust modules behind a lib crate. `CLAUDE.md` in the repo root has detailed architecture documentation for AI-assisted development.

## License

MIT
