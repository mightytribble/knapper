# knapper

**Local hybrid search, retrieval and editing MCP for Obsidian-format vaults.**
knapper indexes a markdown vault into section-level chunks and serves them to
AI agents — either over [MCP](https://modelcontextprotocol.io) or via 
a REST API. Semantic embeddings, full-text search, wikilink graph
traversal and an optional cross-encoder run in one local binary. No API keys,
no cloud.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

It works on any tree of markdown files but it is built for Obsidian: it reads
and writes Obsidian's frontmatter, tag and wikilink conventions. Under active
development — still experimental.

## Why knapper?

You have notes, and you want an agent to work with them. Filesystem tools can
do it, but grep and glob are slow and token-heavy: fine for a handful of files,
but on a large vault they eat the context window and can still miss the section you
wanted.

knapper gives the agent search that ranks, cites its sources, and returns
nothing when the vault holds no answer — so a wrong guess does not arrive
looking like a fact. Retrieval and edits both work below the file — a search
answers with the passage that answers it, and an edit rewrites one heading —
so neither costs a full-file read or a full-file rewrite. Edit in
Obsidian and the index follows within seconds.

## What it does

- **Hybrid search that abstains** — semantic embeddings, BM25, wikilink graph
  expansion and an optional cross-encoder, scored as one calibrated
  probability and dropped below a confidence floor.
- **Sub-file retrieval** — a result is a chunk of a note, named by its heading
  path and docid; retrieved chunks of one section come back merged.
- **Section-level editing** — read a section, change it, write it back; the
  rest of the note and its frontmatter stay byte for byte.
- **MCP server** — `knapper serve` exposes 20 tools to Claude Code, Cursor or
  any other MCP client.
- **HTTP REST API** — `knapper serve --http` serves the same capabilities to
  web agents and scripts, with API keys, rate limiting and CORS.
- **Real-time sync** — a file watcher re-indexes what you change in Obsidian.
- **Vault health** — orphan notes, broken wikilinks, links naming a heading
  that no longer exists, stale notes, tag hygiene.
- **Fully local** — [llama.cpp](https://github.com/ggml-org/llama.cpp) with a
  ~300 MB GGUF model, Metal on macOS and CUDA on Linux. Hosted embeddings are
  an option, never a requirement.

## Quick start

```bash
brew install mightytribble/tap/knapper       # macOS and Linux
```

Pre-built binaries, `cargo install`, and CPU and CUDA Docker images are in the
[install guide](install.md), which also covers wiring the MCP server into each
platform.

**Index your vault:**

```bash
knapper index ~/path/to/vault
# Downloads the embedding model on first run (~300 MB).
# Incremental after that — only changed files re-embed.
# See install.md for how to run multiple vaults or 
# constrain knapper to a single workspace.
```

**Search it:**

```bash
knapper search "how does the auth system work" --scores
```

```
--- [6e1b70#0] [99%] 02-Areas/Development/Auth-Architecture.md > Auth Architecture (matched: semantic+keyword)
# Auth Architecture
How authentication works across our services. Owned by [[Sarah Chen]]; the public contract is in [[API Design]].

## Overview
We use OAuth 2.0 with PKCE for every client type, including the web app and the [[Mobile App]]. There are no client secrets in any client. The authorization server issues short-lived access tokens (15 minutes) and rotating refresh tokens (30 days, single use).
```

A result is a chunk of a note, with its full text — here the note's head and
its `## Overview`, which abut and so come back as one block. The
percentage is the probability that the passage answers the query, and a query
the vault cannot answer prints `No relevant content found for this query in
the vault.` rather than its nearest miss.
[How knapper searches](how-knapper-searches.md) explains both.

**Connect it to Claude Code:**

```bash
claude plugin marketplace add mightytribble/knapper
claude plugin install knapper@knapper
```

That registers the MCP server and the skills. To register the server alone:

```bash
claude mcp add --scope user knapper -- knapper serve
```

Set `--scope project` or `--scope local` to only use knapper on specific
workspaces. See [install](install.md) for multi-vault setup.

Claude can now search your vault, read and edit sections, and create notes
through structured tool calls.

## Documentation

| | |
|---|---|
| [install.md](install.md) | Install on macOS, Linux, WSL2 or Docker, and connect an MCP client |
| [how-knapper-searches.md](how-knapper-searches.md) | Why a query returned what it did, what the score means, what to tune |
| [faq.md](faq.md) | Picking models and builds, running several vaults, writing notes that retrieve well |
| [configuration.md](configuration.md) | Every config key, and what changing it costs |
| [http-rest-api.md](http-rest-api.md) | The REST endpoints, authentication and examples |
| [chatgpt-actions.md](chatgpt-actions.md) | Custom GPT setup — untested in v0.9, see [#87](https://github.com/mightytribble/knapper/issues/87) |
| [surfaces.md](surfaces.md) | What each capability is called on the CLI, on MCP and over HTTP |
| [BUILDING.md](BUILDING.md) · [CONTRIBUTING.md](CONTRIBUTING.md) | Building from source, and contributing |

## The split from engraph

knapper began as a private fork of [engraph](https://github.com/devwhodevs/engraph),
which had the shape I wanted — hybrid search and wikilinks over an Obsidian
vault, served over MCP — but not the implementation. The search was not what it
claimed, parts of the API described methods that were never built, and the
project had been stale for three months with critical fixes outstanding. A week
in, I was replacing more than I was keeping: the search pipeline, the chunker,
the graph lane, and one command surface across all three endpoints. Dropping
the pretence that this was a fork seemed the right thing to do. The bones are engraph's
and gratefully acknowledged; see [NOTICE](NOTICE) and the git history.

## Development

```bash
cargo test --lib          # unit tests, no network (needs CMake for llama.cpp)
cargo clippy -- -D warnings
cargo fmt --check
```

See `CLAUDE.md` for the architecture documentation, module by module.

## Contributing

Contributions welcome — please open an issue first to discuss what you would
like to change. [CONTRIBUTING.md](CONTRIBUTING.md) has the details.

## Attribution

knapper began as a fork of [engraph](https://github.com/devwhodevs/engraph)
(MIT) at v1.7.2. See [NOTICE](NOTICE). The full git history carries the
original commits and their authorship.

## License

MIT
