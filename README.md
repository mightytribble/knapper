# knapper

**Local hybrid search, retrieval and editing MCP for Obsidian-format vaults.** knapper indexes a markdown vault into section-level chunks and serves them to AI agents — Claude Code over [MCP](https://modelcontextprotocol.io), any tool over a REST API. Semantic embeddings, full-text search, wikilink graph traversal, and cross-encoder reranking run in one local binary. No API keys, no cloud.

**knapper** will work with any hierachy of markdown files, but it's targeted at Obsidian vaults and leverages Obsidian frontmatter conventions for tagging and other fields. 

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Knapper is under active development and should still be considered experimental.

## Why knapper?

You have notes in Obsidian (or other markdown) and you want an LLM (perhaps via Claude Code or Codex) to work with them. You could use filesystem tools, but grep and glob are slow and token-heavy: fine for a small number of documents but for larger collections they eat context, they're slow, and their effectiveness degrades quickly.

Knapper solves this by providing your LLM with effective search, realtime indexing (so you can edit in Obsidian and your changes are immediately visible to the LLM), and efficient document updating. No more bloating your context with unneccesary full file reads and slow keyword searches! Its hybrid search combines semantic, keyword, and graph results into one pipeline with citations, so the LLM knows exactly where the results came from and how. Section level retrieval and edits make working with large documents fast and token light. And we mention this is all fast and local?

- **5-lane hybrid search...** — semantic embeddings + BM25 full-text + graph expansion + cross-encoder reranking + temporal scoring. The content lanes fuse via Reciprocal Rank Fusion, and graph and temporal candidates join the shortlist by reserved quota. One absolute scorer then sorts it: by default a **calibrated convex combination** of the lanes' own scores extending TM2C2 — [score fusion rather than rank fusion](https://arxiv.org/abs/2210.11934), with BM25 measured against each query's own theoretical maximum so a score means the same thing from one query to the next — and the cross-encoder when intelligence is enabled. Lane weights are configurable. Time-aware queries like "what happened last week" or "March 2026 notes" activate the temporal lane automatically.
- **...that knows what it doesn't know...** — Conventional semantic and BM25 search will return results even if they're not relevant, which not only wastes tokens but can actively lead an LLM astray. Knapper scores every result as a calibrated probability and drops anything below a confidence floor — when nothing clears it you get nothing back, rather than a plausible-looking wrong answer.
- **...quickly.** — Against a 240-note test vault the default path refuses exactly the same unanswerable queries as the local cross-encoder, missing on only one particularly tricky question compared to it — but much, much faster: 22 ms a query instead of 13.8 seconds on a CPU-only system.
- **MCP server for AI agents** — `knapper serve` exposes 18 tools (search, read, list, tags, vault_map, create, update, delete, move, archive, index, reindex_file, status, health, validate, identity, init, migrate) that Claude, Cursor, or any MCP client can call directly.
- **HTTP REST API** — `knapper serve --http` adds an axum-based HTTP server alongside MCP with 19 REST endpoints, API key authentication, rate limiting, and CORS. Web-based agents and scripts can query your vault with simple `curl` calls.
- **Section-level editing** — AI agents can read, replace, prepend, append to or rename a section by heading, edit the note's body, or edit a frontmatter property — every change is one `update` call carrying a list of edits, and what `read` returns is what `update` takes back.
- **Vault health diagnostics** — detect orphan notes, broken wikilinks, stale content, and tag hygiene issues. Available as MCP tool and CLI command.
- **Real-time sync** — file watcher keeps the index fresh as you edit in Obsidian. No manual re-indexing needed.
- **Smart write pipeline** — AI agents can create, edit, rewrite, and delete notes with automatic tag resolution, wikilink discovery, and folder placement based on semantic similarity.
- **Fully local** — [llama.cpp](https://github.com/ggml-org/llama.cpp) inference with GGUF models (~300MB mandatory, ~650MB optional for the cross-encoder). Metal GPU-accelerated on macOS, CUDA build available for local GPU. No API keys, no cloud required, but Google Gemini Embeddings supported if you want them.

## The Split from Engraph

I originally investigated Engraph because it seemed to meet my needs - proper hybrid search + wikilinks search across an Obsidian vault, provided via MCP. Unfortunately the actual search implementation wasn't what I wanted, and many aspects of the API seemed to be the result of organic growth and aspirational design rather than tested, implemented methods. It also at the time seemed abandoned by the developer - 3 months stale and with several critical bugfix PRs outstanding. A private fork was the obvious place to start, but after working on it for a week or so it became obvious that I was ripping out more of the original code to fit it into what I wanted and fix existing bugs. I can see that trend continuing, so it made sense to drop the pretence that this was a fork, acknowledge with gratitude the bones upon which `knapper` is built, and move on.

Main Changes from Engraph:

- Hybrid search has been completely re-written and improved with empirical testing.
- The search lanes now correctly include the `graph` lane, with a 1-hop personal page rank implementation.
- Searches can be filtered by tag and directory either by inclusion, exclusion, or a combination thereof, using a common vocabulary.
- The chunker has been completely re-worked and optimized for markdown ingest (chunking by section, handling empty sections, etc).
- The three endpoints (`knapper CLI`, `MCP` and `http`) have all been normalized to a single command and parameter surface.
- `knapper` can now correctly list, read, and edit content in markdown using resolvable breadcrumbs. 
- breadcrumbs now correctly contribute to search results.
- CUDA builds are now possible, utilizing local GPU resources to vastly improve embedding and cross-encoding times (approx x30 speedup).
- Dead and aspirational code has been removed, pending new implementations.

## How it works

```
Your vault (markdown files)
        │
        ▼
┌─────────────────────────────────────────────┐
│                knapper index                │
│                                             │
│  Walk → Chunk → Embed (llama.cpp) → Store   │
│                                             │
│  SQLite: files, chunks, FTS5, vectors,      │
│          edges, centroids, tags, audit log  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│                knapper serve                │
│                                             │
│  MCP Server (stdio) + File Watcher          │
│  + HTTP REST API (--http, optional)         │
│                                             │
│  Search: retrieval → RRF fuses lanes        │
│          → calibrated probability sorts,    │
│            or the cross-encoder if enabled  │
│                                             │
│  18 MCP tools + 19 REST endpoints           │
└─────────────────────────────────────────────┘
        │
        ▼
  Claude / Cursor / any MCP client / curl / web agents
```

1. **Index** — walks your vault, chunks markdown by headings, embeds with a local GGUF model via llama.cpp (Metal GPU on macOS), stores everything in SQLite with FTS5 + sqlite-vec + a wikilink graph
2. **Search** — runs the query through up to five lanes (semantic KNN, BM25 keyword, graph expansion, cross-encoder reranking, temporal scoring): the content lanes fuse via RRF at configurable per-lane weights, then one absolute scorer sorts the shortlist: a calibrated probability fused from the lane scores, or the cross-encoder when intelligence is enabled
3. **Serve** — starts an MCP server that AI agents connect to, with a file watcher that re-indexes changes in real time

## Quick start

**Install:**

```bash
# Homebrew (macOS)
brew install mightytribble/tap/knapper

# Pre-built binaries (macOS arm64, Linux x86_64)
# → https://github.com/mightytribble/knapper/releases

# From source (requires CMake for llama.cpp)
cargo install --git https://github.com/mightytribble/knapper

# Docker (Linux/WSL2), CPU or CUDA. Images are linux/amd64 only, so x86_64
# hosts — Windows on ARM is not supported. The CUDA image carries its own
# toolkit: the host needs Docker and the NVIDIA Container Toolkit, never nvcc.
docker pull ghcr.io/mightytribble/knapper:cuda    # or :cpu, or :latest for cpu
docker pull ghcr.io/mightytribble/knapper:0.9.2-cuda      # version-pinned
```

[deployment.md](deployment.md) has the full container flow — data volume, `--gpus all`, and wiring the MCP stdio server through `docker run -i`.

**Index your vault:**

```bash
knapper index ~/path/to/vault
# Downloads embedding model on first run (~300MB)
# Incremental — only re-embeds changed files on subsequent runs
```

**Search:**

```bash
knapper search "how does the auth system work" --scores
```

```
--- [6e1b70#0] [99%] 02-Areas/Development/Auth-Architecture.md > Auth Architecture (matched: semantic+keyword)
# Auth Architecture
How authentication works across our services. Owned by [[Sarah Chen]]; the public contract is in [[API Design]].

## Overview
We use OAuth 2.0 with PKCE for every client type, including the web app and the [[Mobile App]]. There are no client secrets in any client. The authorization server issues short-lived access tokens (15 minutes) and rotating refresh tokens (30 days, single use).

--- [9e2d34#0] [84%] 04-Archive/2024-Legacy-Session-Auth.md > 2024 Legacy Session Auth (matched: semantic+keyword)
# 2024 Legacy Session Auth
The pre-v2 approach: a server-side `sessions` table keyed by a random cookie, checked on every request. Retired when [[Auth Architecture]] moved to OAuth 2.0 tokens. Kept for the migration notes.

## Why it was retired
Every request hit the sessions table, and the table was the single point of failure during the March outage.

--- [6e1b70#3] [35%] 02-Areas/Development/Auth-Architecture.md > Auth Architecture > Session storage (matched: semantic+keyword)
## Session storage
The web app keeps the session token in an HTTP-only, SameSite=Strict cookie. Nothing auth-related is stored in localStorage. Mobile clients use the platform keychain. The legacy server-side session table is gone; see [[2024 Legacy Session Auth]] for what it replaced.
```

A result is one section of a note, with its full text. Abutting chunks of one section — and of the subsections below it — arrive as one block: the first result is the note's head and its `## Overview`, presented together at the stronger member's score. The merge stops at a sibling section, so two neighbouring topics stay two results. `6e1b70#0` is the note's docid and the section's ordinal, and `matched:` names the lanes that found it. The percentage is whichever scorer sorted the result's probability — the cross-encoder's when one is configured, otherwise a calibrated probability fused from the semantic and keyword lanes' own scores — and `--scores` prints it either way; a result below its scorer's own floor (`answer_floor`, 30% by default, for the cross-encoder; `[calibrated] floor`, 75% by default, for the calibrated logistic) is dropped, so a query the vault cannot answer prints `No relevant content found for this query in the vault.` rather than its nearest miss. `--explain` adds each result's per-lane ranks and scores.

**Claude Code** — Install the plugin (recommended), which registers the MCP server and the skills:

```bash
claude plugin marketplace add mightytribble/knapper
claude plugin install knapper@knapper
```

Or register the MCP server yourself with the Claude Code CLI:

```bash
claude mcp add --scope user knapper -- knapper serve
```

Or add it by hand to `~/.claude.json` (user scope) or a project's `.mcp.json` (project scope). [deployment.md](deployment.md) has the full setup for each platform.

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

Now Claude can search your vault, read notes, build context bundles, and create new notes — all through structured tool calls.

**AI Agent Skills** — Install the Knapper skills using the skills CLI (recommended):

```bash
npx skills add mightytribble/knapper
```

**Enable HTTP REST API:**

```bash
# Start MCP + HTTP server on port 3000
knapper serve --http

# Custom port and host
knapper serve --http --port 8080 --host 0.0.0.0

# Local development without API keys (127.0.0.1 only)
knapper serve --http --no-auth
```

**API key management:**

```bash
# Add a new API key (read or write permission)
knapper configure --add-api-key

# List existing keys
knapper configure --list-api-keys

# Revoke a key
knapper configure --revoke-api-key kn_abc123...
```

**Enable intelligence (optional, ~650MB download):**

```bash
knapper configure --enable-intelligence
# Downloads Qwen3-Reranker (cross-encoder)
# Adds the reranker lane to search
```

Without it, search already ranks by a calibrated probability fused from the semantic and keyword lanes' own scores, and abstains (`No relevant content found for this query in the vault.`) below its own floor. What the cross-encoder still buys is ordering on the hardest queries — the ones where a candidate merely mentions the query's words rather than answering it, which no fusion of lane scores alone can tell apart.

That probability's coefficients and floor are **EmbeddingGemma's fit**, not a global default — `[calibrated]` in your config says so above the numbers. The keyword half normalizes itself against each query's own upper bound, but the semantic half is one model's cosine scale, so another `models.embed` moves the floor in an unknown direction and the path then abstains on every query or on none. The four numbers are one fit and do not move apart: point `models.embed` elsewhere and refit them together with `scripts/calibrated-fusion-eval.py`, or set `[calibrated] floor = 0.0` until you do.

A refit runs on your labels, and writing them is the work. The tool takes a built store, query vectors from `examples/eval_query_embed.rs`, and two files you write yourself: the queries, and which chunks of your vault answer each one. Nothing ships those and nothing can — they are a judgement about your own notes. The shipped fit rests on 33 labeled answers against 1228 labeled non-answers, and a fit is worth what its labels are worth. Note that the labels come from your corpus while the thing that goes stale is the embedder: both lane scores normalize themselves per query, so a new corpus changes what the features say and not what the coefficients mean. Refit when you change the embedder, not when the vault grows.

## Example usage

**Search with the cross-encoder lane:**

```bash
knapper search "how does authentication work" --explain
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

**Vault structure overview:**

```bash
knapper vault-map
```

Returns folder counts, top tags, recent files — gives an AI agent orientation before it starts searching.

**See what the vault holds:**

```bash
knapper list --scope /locations/ --detailed
```
```
locations/aurelian-empire.md
# About the Empire
## History
### The founding of the Empire
## Current Events
```

One bare path per line, in path order, so a folder's notes arrive together and `wc -l` is the total. A bare `knapper list` returns every indexed note; `--scope`, `--all`, `--any` and `--none` narrow it by tag or by directory, and `--limit` keeps the first n. `--detailed` reads each listed note and prints its headings, which is how an agent finds the section to read or write before it calls `read` or `update`.

**Create a note via the write pipeline:**

```bash
knapper create --content "# Meeting Notes\n\nDiscussed auth timeline with Sarah." --tags meeting,auth
```

knapper resolves tags against the registry (fuzzy matching), discovers potential wikilinks (`[[Sarah Chen]]`), suggests the best folder based on semantic similarity to existing notes, and writes atomically. The frontmatter in `--content` is written as given; `tags` is the only key `create` adds, and only the tags it resolved.

**Edit a specific section:**

```bash
knapper update "Meeting Notes" --section "Action Items" --mode append --content="- [ ] Follow up with Sarah"
```

Targets the "Action Items" section by heading, appends content without touching the rest of the note. Write `--content=` with an equals sign when the value starts with a `-`: the shell passes the text through untouched, and clap reads a leading `-` as a flag.

A section's content is the body **below** the heading, which is what `knapper read --section` returns: read a section, change it, write it back, and the note is the note it was. Content that opens with a heading at or above the section's own level is refused, because such a line would end the section rather than fill it.

**Rename a section:**

```bash
knapper update "The Roads of New Visland" --section "Norlund to Westport via Bend" --heading "Norlund to Bend"
```

Renames the section and leaves its body alone; add `--content` to rewrite the body in the same write. The note keeps the heading's own markup, so a `###` stays a `###` and a promoted bold line keeps its markers — `--heading` carries the text. A name another section of the note already holds is refused, since two sections of one name leave both unaddressable by name.

**Rewrite a note (preserves frontmatter):**

```bash
knapper update "Meeting Notes" --mode replace --content "# Meeting Notes\n\nRevised content here."
```

Replaces the entire body while keeping existing frontmatter (tags, dates, metadata) intact.

**Edit frontmatter:**

```bash
knapper update "Meeting Notes" --property tags --mode append --content "actionable"
```

A property takes `--mode replace`, `append` or `remove`; a body or a section takes `replace`, `prepend` or `append`. Repeat `--content` to write a list-valued property such as tags or aliases.

A write changes only the keys it names. A property edit keeps the key's place and the note's own list style, and a list with no items writes `[]` instead of deleting the key. A value knapper cannot edit as a line — a nested mapping, an anchor, a block scalar — refuses the write and names what it found, rather than re-styling the block.

**Delete a note:**

```bash
knapper delete "Old Draft" --mode soft   # moves to archive
knapper delete "Old Draft" --mode hard   # permanent removal
```

**Check vault health:**

```bash
knapper health
```

Returns orphan notes (no links in or out), broken wikilinks, stale notes, and tag hygiene issues.

## HTTP REST API

`knapper serve --http` adds a full REST API alongside the MCP server, exposing the same capabilities over HTTP for web agents, scripts, and integrations.

**19 endpoints:**

Every capability is one route, and the route is the CLI command's name under `/api/`. `surfaces.md` is the generated table of all three surfaces.

| Method | Endpoint | Permission | Description |
|--------|----------|------------|-------------|
| GET | `/api/health-check` | read | Server health check |
| POST | `/api/search` | read | Hybrid search (semantic + FTS5 + graph + reranker + temporal), scoped by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`) |
| GET | `/api/read` | read | Read a note (`file`), or one of its sections (`section`) |
| GET | `/api/list` | read | List notes by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`), creator, limit, and `detailed=true` for each note's heading outline |
| GET | `/api/tags` | read | The tag vocabulary, whole or under one term (`under`) |
| GET | `/api/vault-map` | read | Vault structure overview (folders, tags, recent files) |
| GET | `/api/status` | read | Index status and statistics |
| GET | `/api/health` | read | Vault health diagnostics |
| POST | `/api/validate` | read | Check vault markdown for structural and indexing problems — one note (`path`), a scope, or the whole vault; reads the files, not the index |
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
curl -H "Authorization: Bearer kn_abc123..." http://localhost:3000/api/vault-map
```

Keys have either `read` or `write` permission. Write keys can access all endpoints; read keys are restricted to read-only endpoints. Use `--no-auth` for local development without keys (127.0.0.1 only).

**curl examples:**

```bash
# Search
curl -X POST http://localhost:3000/api/search \
  -H "Authorization: Bearer kn_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication architecture", "top_n": 5}'

# Search, scoped to a tag or directory filter (scope/all, any, none; a
# leading / reads a term as a directory path from the vault root)
curl -X POST http://localhost:3000/api/search \
  -H "Authorization: Bearer kn_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication architecture", "top_n": 5, "scope": ["project/auth", "/01-Projects/"], "all": ["type/decision"], "any": ["status/reviewed", "status/draft"], "none": ["status/archived"]}'

# Read a note, or one of its sections
curl "http://localhost:3000/api/read?file=01-Projects/API-Design.md" \
  -H "Authorization: Bearer kn_..."
curl "http://localhost:3000/api/read?file=01-Projects/API-Design.md&section=Endpoints" \
  -H "Authorization: Bearer kn_..."

# Create a note
curl -X POST http://localhost:3000/api/create \
  -H "Authorization: Bearer kn_..." \
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
key = "kn_..."
permission = "write"
```

## PARA Migration

`knapper migrate` restructures your vault into the [PARA method](https://fortelabs.com/blog/para/) (Projects, Areas, Resources, Archive) using heuristic classification. The workflow is non-destructive: preview first, review the plan, then apply.

**Workflow:**

```bash
# 1. Preview — classify notes and generate a migration plan
knapper migrate --mode preview
# Outputs: markdown summary + JSON plan saved to ~/.knapper/

# 2. Review the plan (edit if needed)
cat ~/.knapper/migration_preview.md

# 3. Apply — move files according to the plan
knapper migrate --mode apply

# 4. Undo — reverse the last migration if something looks wrong
knapper migrate --mode undo
```

**Classification signals:**

| Category | Detection signals |
|----------|-------------------|
| **Projects** | Open tasks (`- [ ]`), active/in-progress status in frontmatter, project tags |
| **Areas** | Recurring topic keywords (health, finance, career, learning), area-related tags |
| **Resources** | People notes (People folder, person-like content), reference material, articles, code snippets |
| **Archive** | Done/completed/inactive status, no incoming or outgoing wikilinks, stale content |

Notes that don't match any signal with sufficient confidence stay in place. Daily notes (`YYYY-MM-DD.md`) and templates are always skipped.

**MCP tool:** `migrate`, with the same `mode` — available in `knapper serve` for AI-assisted migration.

**HTTP endpoint:** `POST /api/migrate`, with the same `mode` — available via `knapper serve --http`. On both servers `mode: apply` takes the `preview` that `mode: preview` returned; only the CLI reads the copy saved on disk.

## ChatGPT Actions

> **Untested in v0.9.** The API half of this path works, but the GPT import step is known to fail on the shipped OpenAPI spec (two endpoint descriptions exceed ChatGPT's 300-character cap), and the setup flow still references the retired plugin-manifest format. The fixes are tracked in [#87](https://github.com/mightytribble/knapper/issues/87) for v1. The HTTP API itself ([above](#http-rest-api)) is supported.

Connect your Obsidian vault to ChatGPT as a custom GPT Action. ChatGPT can search, read, create, and edit your notes through knapper's REST API.

### Prerequisites

- knapper installed and indexed (`knapper index ~/your-vault`)
- A tunnel tool: [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/) (recommended) or [ngrok](https://ngrok.com)

### Step 1: Configure knapper

```bash
# Interactive setup — enables HTTP, creates API key, sets CORS
knapper configure --setup-chatgpt
```

Or configure manually in `~/.knapper/config.toml`:

```toml
[http]
enabled = true
port = 3000
host = "127.0.0.1"
rate_limit = 60
cors_origins = ["https://chat.openai.com", "https://chatgpt.com"]

[[http.api_keys]]
key = "kn_your_key_here"    # generate with: knapper configure --add-api-key --key-name chatgpt --key-permissions write
name = "chatgpt"
permissions = "write"        # "read" for search-only, "write" to also create/edit notes

[http.plugin]
name = "My Vault"
description = "Search and manage my Obsidian vault"
public_url = "https://your-tunnel-url.trycloudflare.com"   # set after starting tunnel
```

### Step 2: Start knapper + tunnel

**Terminal 1 — knapper HTTP server:**
```bash
knapper serve --http
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

Edit `~/.knapper/config.toml` and set `public_url` to your tunnel URL:

```toml
[http.plugin]
public_url = "https://abc-xyz.trycloudflare.com"
```

Then restart knapper (`Ctrl+C` and re-run `knapper serve --http`). This ensures the OpenAPI spec points to the correct public URL.

### Step 4: Verify endpoints

```bash
# Both should return JSON (no auth required)
curl https://your-tunnel-url/openapi.json
curl https://your-tunnel-url/.well-known/ai-plugin.json

# Search with auth
curl -X POST -H "Authorization: Bearer kn_your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "test search"}' \
  https://your-tunnel-url/api/search
```

### Step 5: Register in ChatGPT

1. Go to [ChatGPT](https://chat.openai.com) → **Explore GPTs** → **Create**
2. Give your GPT a name (e.g., "Vault Assistant")
3. Add these **Instructions**:

```
You are a knowledge assistant connected to the user's Obsidian vault via knapper.

WORKFLOW:
1. Use searchVault to find relevant notes before answering questions
2. Use readNote for full content, and its section parameter for one heading
3. Use getVaultMap to orient yourself in the vault structure
4. Only create or edit notes when explicitly asked

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
8. Paste your API key (the `kn_...` key from Step 1)
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
- **knapper must be running** on your machine for ChatGPT to access it. If you close the terminal, the connection drops.

## Use cases

**AI-assisted knowledge work** — Give Claude or Cursor deep access to your personal knowledge base. Instead of copy-pasting context, the agent searches, reads, and cross-references your notes directly.

**Developer second brain** — Index architecture docs, decision records, meeting notes, and code snippets. Search by concept across all of them.

**Research and writing** — Ask a question in prose and get the sections that answer it, each scored by the cross-encoder, with a note's abutting chunks of one section merged into one block; `read --metadata` then follows a note's wikilinks in both directions.

**Team knowledge graphs** — Index a shared docs vault. AI agents can answer "who knows about X?" and "what decisions were made about Y?" by traversing the note graph.

## How it compares

| | knapper | Basic RAG (vector-only) | Obsidian search |
|---|---|---|---|
| Search method | 5-lane hybrid (semantic + BM25 + graph + reranker + temporal): content lanes RRF-fused, cross-encoder sorts | Vector similarity only | Keyword only |
| Query understanding | Cross-encoder reads each candidate jointly with the query | None | None |
| Understands note links | Yes (wikilink graph traversal) | No | Limited (backlinks panel) |
| AI agent access | MCP server (18 tools) + HTTP REST API (19 endpoints) | Custom API needed | No |
| Write capability | Create/edit/rewrite/delete with smart filing | No | Manual |
| Vault health | Orphans, broken links, stale notes, tag hygiene | No | Limited |
| Real-time sync | File watcher, 2s debounce | Manual re-index | N/A |
| Runs locally | Yes, llama.cpp + Metal GPU | Depends | Yes |
| Setup | One binary, one command | Framework + code | Built-in |

knapper is not a replacement for Obsidian — it's the intelligence layer that sits between your vault and your AI tools.

## When to use CUDA vs. CPU Builds, local embedder vs. cloud

Knapper is designed to be fast and usable with the default settings, using only a local EmbeddingGemma model, on OS X or Linux (and in Windows via WSL or Docker). On Linux and WSL2 the Docker images are the least painful route to either build: `VARIANT=cuda` bundles the CUDA toolkit inside the image, so the host never needs `nvcc` or a matching toolkit — only Docker and the NVIDIA Container Toolkit at run time. Embedding is faster with CUDA, but the real speedup is if you want to use a cross-encoder: if you have the GPU RAM and can afford to run it, a 4B parameter cross-encoder gives marginally better results than default at the cost of ~400ms per query.

Similarly, Gemini Embedding 2 gives superior results to EmbeddingGemma, at the cost of, well, money, latency (it's still plenty fast), and a willingness to send traffic to Google. Options are good: pick what works best for you.

## Current capabilities

- 5-lane hybrid search (semantic + FTS5 + graph + cross-encoder reranker + temporal): content lanes RRF-fused, graph and temporal routed into the shortlist by reserved quota, cross-encoder sorts it (legacy mode fuses all five by weighted RRF)
- Temporal search: natural language date queries ("last week", "March 2026", "recent"), date extraction from frontmatter and filenames, smooth decay scoring
- Confidence % display: search results show normalized 0-100% confidence instead of raw RRF scores
- llama.cpp inference via Rust bindings (GGUF models, Metal GPU on macOS, CUDA on Linux)
- Intelligence opt-in: the cross-encoder lane is off unless enabled
- MCP server with 18 tools (5 read, 5 write, 6 index and diagnostic, 1 setup, 1 migrate) via stdio
- HTTP REST API with 19 endpoints, API key auth (`kn_` prefix), rate limiting, CORS — enabled via `knapper serve --http`
- User identity with L0/L1 tiered context for AI agent session starts
- Section-level reading and editing: target specific headings with replace/prepend/append modes
- Full note rewriting with automatic frontmatter preservation
- Frontmatter property edits: replace, append to or remove a property, scalar or list-valued
- Soft delete (archive) and hard delete (permanent) with audit logging
- Vault health diagnostics: orphan notes, broken wikilinks, stale content, tag hygiene
- Real-time file watching with 2s debounce, startup reconciliation, and watcher coordination to prevent double re-indexing
- Write pipeline: tag resolution, fuzzy link discovery, semantic folder placement
- Vault graph: directional wikilink edges, traversed both ways (outgoing links + backlinks); single-hop personalized-PageRank expansion over the chunk graph
- Placement correction learning from user file moves
- Enhanced file resolution with fuzzy Levenshtein matching fallback
- Content-based folder role detection (people, daily, archive) by content patterns
- PARA migration: AI-assisted vault restructuring into Projects/Areas/Resources/Archive with preview, apply, and undo workflow
- Configurable model overrides for multilingual support

## Configuration

Optional config at `~/.knapper/config.toml`:

```toml
vault_path = "~/Documents/MyVault"
# How many results a search returns, counting a merged block as one. Fewer
# come back only when the vault holds fewer: `[ranking] candidates` caps the
# pool and the answer floor drops what is not an answer.
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

# Abutting chunks of one section — and of the subsections below it — are
# returned as a single block, stopping at a sibling section. Default on; set
# to false for per-chunk results.
[ranking]
# coalesce_adjacent = true

# Prepend document identity to each chunk before embedding. Off by default —
# it helped conceptual queries and hurt exact-name lookup on the test vault.
# Needs `knapper index --rebuild` to take effect either way.
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
# Larger 4B cross-encoder (third-party GGUF): more accurate, slower, larger
# download. It runs at query time, so a switch needs no re-index.
# rerank = "hf:gscoppino/Qwen3-Reranker-4B-GGUF-llama_cpp/Qwen3-Reranker-4B-Q4_K_M.gguf"  # ~2.5 GB
# rerank = "hf:gscoppino/Qwen3-Reranker-4B-GGUF-llama_cpp/Qwen3-Reranker-4B.Q8_0.gguf"    # ~4.3 GB
# Hosted embeddings instead of the local GGUF. The id must be pinned and
# versioned: "gemini:gemini-embedding" and "-latest" are rejected, so a moving
# alias cannot re-point an existing store's vectors at a model that changed
# underneath them. The key is read from GEMINI_API_KEY and never written here.
# embed = "gemini:gemini-embedding-2"

# Non-secret knobs for the hosted embedder above.
# [models.embed_api]
# dim = 1536           # Matryoshka truncation; unset uses the native 3072
# timeout_secs = 30
# max_retries = 4      # on 429, 5xx and transport errors
# endpoint = "..."     # proxy or test-server override

# Registered AI agents
[agents]
# names = ["claude-code", "cursor"]
```

Search returns **sections**. A note whose "Counterspell" and "Dispel Magic" sections both answer a query contributes both, up to `max_chunks_per_file`, and each result names the heading it came from. Pass `--group-by file` (or set `group_by = "file"`) for one result per document, representing it by its best-matching section.

`embedding_prefix` puts the document's name, path, tags, aliases and the chunk's ancestor headings into the text that is **embedded**, while what is stored, displayed and keyword-matched stays the raw chunk. It is off by default: because the prefix is the same string for every chunk of a document, it separates documents from each other at the cost of separating a document's own sections, and on the test vault that lost more on exact-name lookup than it gained on conceptual queries (`eval/probes.md`). Each component is switchable so the trade can be measured per vault.

`models.embed` takes a hosted provider as well as a local GGUF: `gemini:<versioned-id>` routes embedding to the Gemini API, reading the key from `GEMINI_API_KEY` — environment only, never written to config. The id has to end in a version number, so a moving alias cannot silently re-point an existing store's vectors. Documents batch at 100 per request and inputs are capped at 2048 tokens; `[models.embed_api]` bounds the call and can truncate the 3072-wide output. The embedder is a fingerprint component, so switching re-indexes the vault — and see the calibration note above: the model-free floor is fit against the local embedder, so a hosted one wants `[calibrated] floor = 0.0` or a refit.

The keyword lane indexes each chunk's **full text**. `chunks.snippet` — the leading 200 characters — is the display field only; it is not what BM25 searches.

`exclude` takes `.gitignore`-style globs, matched against paths relative to the vault root. A pattern with no `/` matches at any depth (`*-index.md` catches `lore/lore-index.md`); a trailing `/` means a directory and everything under it; an embedded `/` anchors the pattern to the vault root (`drafts/**`). Excluding a path that is already indexed removes it from the store — chunks, vectors, FTS entries and graph edges — on the next index run.

All data stored in `~/.knapper/` — single SQLite database (~10MB typical), GGUF models, and vault profile. Set `KNAPPER_HOME` (used verbatim) or pass `--data-dir` to move that directory; `--data-dir` wins over the environment, which wins over the default.

## Development

```bash
cargo test --lib          # runs unit tests, no network (requires CMake for llama.cpp)
cargo clippy -- -D warnings
cargo fmt --check
```

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

The codebase is 41 Rust modules behind a lib crate. `CLAUDE.md` in the repo root has detailed architecture documentation for AI-assisted development.

## Attribution

knapper began as a fork of [engraph](https://github.com/devwhodevs/engraph) (MIT) at v1.7.2. See `NOTICE`. The full git history carries the original commits and their authorship.

## License

MIT
