# Changelog

## Unreleased

### Fixed

- Frontmatter writes preserve the note (#92). `create` writes the caller's frontmatter as given, and no longer adds `created`, `created_by` or placement keys. A property edit keeps the key's position and the note's list style, and an empty list writes an empty list instead of deleting the key. `archive` and `unarchive` edit the block instead of rebuilding it, so an archive round trip keeps the note's other keys, its comments and its blank lines byte for byte. `archive` refuses a note that already holds `archived`, `archived_at` or `archived_from`, naming the key it found, rather than lose the note's own value or drop it silently on unarchive. `unarchive` drops a leftover `archived` tag from a note archived by an earlier version of knapper, so the tag does not return to the vocabulary. A value knapper cannot address as a line — a nested mapping, an anchor, a block scalar — refuses the write and names what it found. The placement-correction learning loop no longer fires for a newly created note, because `create` no longer writes the `suggested_folder` and `created_by` keys the loop reads.

## 0.9.1 (2026-08-30)

### Calibrated score fusion: the model-free default ranks and abstains

With no cross-encoder configured — the default install — the sorted ranking stage now runs and a three-coefficient logistic sorts the candidate pool, where the build fell to the legacy five-lane fusion order before. Each content lane's score is normalized per query onto an absolute `[0, 1]` evidence scale — BM25 against the query's own upper bound, cosine as the near-absolute signal it already is — and `p = σ(w_s·cos + w_k·bm25n + b)` fuses the pair into one probability per candidate. Results sort by that probability, the answer floor applies to it, and the confidence reported is a probability rather than a share of the top result. No model call runs. Design: `docs/specs/2026-08-30-calibrated-fusion-design.md`.

Measured through `eval/pool.sh` at `top_n = 20` on a 240-file / 1501-chunk vault, three arms over one store: the calibrated path and the cross-encoder abstain on the same nine of twelve non-answers and keep the same three negatives, where the previous model-free order returned a full 14-20 block window for every query — the nonsense control included — and reported 100% on all twenty.

#### Added

- **`[calibrated]`**, a config section read only on the model-free path: `enabled` (default `true`), the fitted coefficients `semantic = 13.878`, `keyword = 13.571`, `intercept = -5.848`, and `floor = 0.75`, the probability below which a candidate is not an answer. Every key is query-time and reaches no fingerprint, so a change re-indexes nothing and a sweep is a config edit. `enabled = false` restores the previous routing — a no-model build takes the legacy stage — byte for byte; `floor = 0.0` removes nothing. With a cross-encoder configured the whole section is inert.
- **Abstention on the default install.** A query the vault cannot answer returns the empty set with `NO_RELEVANT_CONTENT` on the CLI, over MCP and on `POST /api/search`. The previous model-free path could not abstain at all: `answer_floor` skips a candidate whose score is not a probability, and every score on that path was a fused rank.
- **`calibrate.rs`**, the fusion arithmetic: FTS5's idf with its clamp, the query's BM25 upper bound `(k1 + 1) · Σ idf`, the `[0, 1]` normalization and the logistic. Pure math with no store and no model, so every formula is tested against hand-computed values.
- **`--explain` reports the path's working**: the query's BM25 upper bound and each term's idf beside the MATCH expression, and every candidate's probability as a `calibrated` lane contribution.

#### Changed

- **Confidence on the model-free path is an absolute probability.** It was `rrf_score` renormalized against the top result, so the first answer of every query printed 100% however bad it was.
- **The graph and temporal lanes are judged on the same scale as content candidates** on this path. A graph admission is scored by the logistic rather than slotted by reserved-quota position, so provenance no longer caps rank; its features come from the store when no content lane fetched the candidate.
- **`into_fused` names its sort lane**, `rerank` or `calibrated`, so `--explain` reports which scorer produced the order and the answer floor applies to whichever probability sorted the pool.
- **Test count: 1014 → 1035.**

## 0.9.0 (2026-08-16) — knapper

The project leaves fork status: engraph, forked at v1.7.2, becomes knapper. The binary, the repository (mightytribble/knapper), the data directory (`~/.knapper/`) and the store file (`knapper.db`, with a read fallback to an existing `engraph.db`) carry the new name. The MIT license and the full git history stay; see `NOTICE`. Versions restart at 0.9.x; 1.0.0 marks the v1 milestone: functional Metal and Docker build pipelines.

### Address a section by its heading path ([#69](https://github.com/mightytribble/knapper/issues/69))

`--section` names a section by its heading text or by its full heading path, and a promoted bold line is one of the sections it reaches. `list --detailed` enumerates the same set, so what the outline prints is what `read` and `update` can name.

#### Added

- **`--section` takes a heading path.** `read` and `update` accept a heading's own text, as before, or its full path from its own root joined with ` > `: `--section "About the Empire > Current Events > History"` reaches the second `History` of a note that holds two. A partial path resolves nothing, so a wrong guess is an error and never an edit to another section. Two same-named siblings under one parent share a path, and the first in document order is the one that resolves.
- **A promoted bold heading is addressable** ([#53](https://github.com/mightytribble/knapper/issues/53)). `--section "Spells"`, `--section "**Spells**"` and `--section "Stat Block > Spells"` all reach a `**Spells**` section, which is 107 of the 1559 chunks in the pinned corpus and was reachable by no name at all. The section ends where the chunker ends it: at the next promoted line, or the next `#` heading of any depth. An ATX heading keeps precedence over a promoted one of the same name under the same parent. The fallback runs whatever `promote_bold_headings` says, because what a caller may name is a property of the file and not of what the indexer chunks.
- **An empty section is addressable**, promoted or ATX, because addressing one is how a caller fills it.

#### Changed

- **`list --detailed` lists promoted headings** beside the ATX ones, and lists a bodyless promoted heading too, so every entry of an outline is a section `read` and `update` can name. `Heading.level` is now absent for a promoted line rather than carrying a depth it does not have; on the CLI such a line prints in its bold form, `**Spells**`.
- **`markdown.rs` owns what a heading is** and `chunker.rs` owns which headings start a chunk. The chunker's set is unchanged, so no fingerprint moves and no store re-indexes.
- **Test count: 893 → 918.**

### List the vault's files ([#68](https://github.com/mightytribble/knapper/issues/68))

`knapper list` is the call an agent makes to see a vault it cannot read: every note the scope admits, in path order, one bare path per line, and each note's heading outline under `--detailed`.

#### Removed

- **`list --folder`**, and the `folder` parameter on the MCP `list` tool and `GET /api/list`. A directory filter is a scope term: `--scope /lore/` — a leading `/` reads a term as a directory path, a trailing `/` its subtree — which is a case-sensitive range anchored at the path boundary, where `--folder lore` was a `LIKE 'lore%'` that folded case, read `_` as a wildcard and matched `lorekeeper.md`. `store::list_files` keeps its `folder` argument for `project`'s sibling gather, which is not a caller-typed filter.

#### Added

- **`list --detailed`** answers each note's ATX heading outline beneath its path. The outline is read from the file, because the index cannot hold it: a short section merges into the chunk before it, an empty heading emits no chunk, and a promoted bold line sits in `chunks.heading` beside real headings. `NoteListItem` gains `headings: Option<Vec<Heading>>` — level, text, and a 1-based line — absent unless `detailed` is set, so an undetailed listing serialises as it did before. On the CLI the headings print as their own `#` markers under the path; over MCP and HTTP they are structured. `detailed=true` is required on the HTTP query string, because `serde_urlencoded` reads no bare flag. A note whose file is missing on disk lists with an empty outline and no error.

#### Changed

- **`list` answers in path order** (`ORDER BY f.path`, SQLite `BINARY` collation, so `Lore/` sorts before `lore/`), where it answered most-recently-indexed first. A folder's notes now arrive together and a subtree scope reads as one block. `project`'s sibling gather takes the same ordering: the first 50 of the folder in path order.
- **`list`'s limit is unbounded by default.** It was capped at 20. An absent `--limit` now emits no `LIMIT` clause, so a bare `knapper list` answers every note the scope admits, and a caller that wants less names a scope or a `--limit`; `--limit 0` answers none. The default is the same on every surface, so no one surface silently caps the whole vault.
- **The CLI's plain `list` output is one bare path per line** and nothing else — no docid, tags, edge count, or trailing total — so it pipes and `wc -l` is the total. The path is relative to the vault root, the form `read`, `update` and `move` take, so a listed path pastes into the next call. `--json` still answers the full `NoteListItem` array.
- **Test count: 877 → 893.**

### One name per capability ([#62](https://github.com/mightytribble/knapper/issues/62))

Every capability now has one name and one parameter set on the CLI, the MCP server and the HTTP API. The name is one word in `kebab-case`, and each surface spells it its own way: the CLI command as written, the MCP tool with `-` as `_`, and the HTTP route under `/api/` and the name. One transform gets from any spelling to any other.

**The renames land with no aliases and with no deprecation window.** Nothing outside this repository named an MCP tool or a CLI command, so no old name is kept working. Everything under "Removed" below is a hard break: an agent or a script that calls an old name gets an error, not a warning. This section is the whole list.

#### Removed

- **CLI command groups.** `knapper context <leaf>` and `knapper write <leaf>` are gone; every leaf is a top-level command. `context read` → `read`, `context list` → `list`, `context tags` → `tags`, `context vault-map` → `vault-map`, `context who` → `who`, `context project` → `project`, `context topic` → `topic`, `write create` → `create`, `write archive` → `archive`, `write delete` → `delete`.
- **`knapper graph`**, both leaves. `graph show` ran the same four queries `read` runs, so `read` answers it and now carries a docid beside each link. `graph stats` is folded into `status`.
- **`knapper migrate para`.** PARA is the only strategy, so the leaf is gone: `knapper migrate --mode preview|apply|undo`.
- **12 MCP tools** (26 → 20, with six added below): `read_section` is a `section` parameter of `read`; `append`, `edit`, `rewrite`, `edit_frontmatter` and `update_metadata` are one `update` tool; `unarchive` is `archive {undo: true}`; `setup` is `init {mode}`; `migrate_preview`, `migrate_apply` and `migrate_undo` are `migrate {mode}`; `context` is renamed `topic`. `move_note` is renamed `move`.
- **12 HTTP routes** (29 → 23, 27 under `/api` → 21, with five added below): `POST /api/read-section`, `/api/append`, `/api/edit`, `/api/rewrite`, `/api/edit-frontmatter`, `/api/update-metadata`, `/api/unarchive`, `/api/setup`, `/api/context`, and `POST /api/migrate/preview`, `/apply` and `/undo`.
- **Path parameters.** `GET /api/read/{*file}`, `/api/who/{name}` and `/api/project/{name}` are `?file=`, `?name=` and `?name=` query parameters, because the shared parameter struct carries its arguments the way it names them.
- **`rewrite`'s `preserve_frontmatter: false`** has no spelling in `update`. A body edit always keeps the note's frontmatter; change the frontmatter with `property` edits in the same list.
- **`update_metadata`'s `modified_by` stamp.** A whole-note tag or alias replacement no longer writes a `modified_by` property into the note.
- **`total_files`** from the `status` JSON. `files` already reports it.
- **The disk fallback for `migrate` `mode: apply`** on the two servers. `apply` now requires the `preview` the caller's own `mode: preview` returned; it no longer falls back to `~/.knapper/migration-preview.json`. The CLI's two-step flow, which saves and reads that file itself, is unchanged.

#### Added

- **`update`**, one capability for every change to an existing note. It takes a list of edits and applies them in order in one write: one mtime conflict check, one file write, one re-index. Each edit names a `section`, a `property`, or neither (the note's body), and carries a `mode` of `replace`, `prepend`, `append` or `remove`. `content` is a string, or a list of strings for a list-valued property. The grammar reaches something no call it replaced could: two sections and a tag change in one atomic write.
- **`read --section`** narrows the content to one ATX heading's body and adds `heading`, `line_start` and `line_end`. The heading match folds case, and `byte_count` measures the section. The note's tags and links are reported either way, because a section's are its file's.
- **The gaps fill.** Six MCP tools are new — `index`, `status`, `topic`, `update`, `init` and `migrate` — and five HTTP routes — `POST /api/index`, `/api/update`, `/api/migrate`, `/api/topic` and `GET /api/status`. On the CLI, `health`, `reindex-file`, `move`, `update`, `status` and `index` each reach a surface that did not have them.
- **`explain` and `group_by` are per call on every surface**, so one query answers the same way whoever asks it.
- **`GET /api/identity?refresh=` and MCP `identity {refresh}`** re-extract the L1 facts, which was a CLI-only flag. It rewrites the `identity_facts` rows, so it takes the write permission and a read-only server refuses it.
- **`docs/surfaces.md`**, generated from `src/surface.rs` and checked by a test, listing what every capability is called on each surface.
- **Parity tests.** Five tests compare the capability table with what `Cli::command()`, `KnapperServer::tool_router()` and `http::routes()` register, including each tool's schema against its clap arguments. A capability added to one surface and forgotten on another fails the build.

#### Changed

- **`search`'s default `top_n` is the configured one** on both servers. It was a hardcoded 10; it is now `top_n` from `config.toml`, whose default is **5** — the same number the CLI has always used. A caller that relied on ten results per query must now ask for `top_n: 10`.
- **`search` over HTTP returns an envelope**, `{"results": [...], "message": ...}`, replacing the bare array. `explain` joins it when the call asked for it. HTTP was the one surface with nowhere to put the answer-floor signal.
- **`update`'s `--mode` defaults to `replace`.** The calls it absorbed were stricter — `write rewrite --content` and `write edit --content` were required, and `write edit`'s mode defaulted to `append`. `knapper update <file>` with no `--content` and no `--edits` still reads stdin, and an empty read is now refused for a body or section `replace` rather than blanking the note. `--content ""` is the deliberate spelling for that.
- **A body edit adds no blank line of its own.** `split_frontmatter` rejoins the body carrying the break after the closing `---`, and the reassembly supplies its own, so successive appends and successive property edits used to push the body one line down per call. Both reassembly paths now normalise it.
- **`update` checks the mtime.** A note changed outside knapper and not yet re-indexed fails with an mtime conflict, which `edit`, `rewrite` and `edit_frontmatter` did not do.
- **`delete`'s `mode` is an enum** on all three surfaces. It read `"hard" => hard, _ => soft`, so `mode: "hardd"` archived the note silently; an unknown word is now refused where the request is read.
- **A read-only server refuses `index` and `init {mode: apply}`** on both servers, as it already refused the write calls.
- **MCP tools: 26 → 20. HTTP routes: 29 → 23** (27 under `/api` → 21, beside the `/api/health-check` liveness probe and the two discovery routes). **CLI top-level commands: 13 → 24** — twenty capabilities plus `configure`, `models`, `clear` and `serve`, which configure the process and not the vault.
- **Test count: 785 → 851.**

---

Entries below this line describe the upstream engraph lineage (https://github.com/devwhodevs/engraph).

## v1.6.1 — Patch Release (2026-04-21)

### Fixed
- **UTF-8 panic in temporal search** — `find_iso_date_in_query()` no longer panics on queries containing multi-byte characters (e.g. Polish diacritics like ą, ś, ź, ż). Added `is_char_boundary()` guards before `&str` slicing, matching the pattern already used by `extract_date_from_filename()`. Thanks [@majkelooo](https://github.com/majkelooo) ([#24](https://github.com/devwhodevs/engraph/pull/24)).
- **CI green on Rust 1.95** — silenced new `clippy::unnecessary_sort_by` and `clippy::explicit_counter_loop` lints promoted to deny-by-default in Rust 1.95.0. No behavior changes ([#29](https://github.com/devwhodevs/engraph/pull/29)).

### Changed
- **rmcp** bumped from 1.4.0 to 1.5.0 — adds MCP protocol version `2025-11-25` via upstream [modelcontextprotocol/rust-sdk#802](https://github.com/modelcontextprotocol/rust-sdk/pull/802). Resolves [#20](https://github.com/devwhodevs/engraph/issues/20): engraph tools now surface in Claude Desktop Cowork and Code-in-Desktop modes ([#30](https://github.com/devwhodevs/engraph/pull/30)).

## v1.6.0 — Onboarding + Identity (2026-04-10)

### Added
- **Interactive onboarding** (`engraph init`) — polished CLI with welcome banner, vault scan checkmarks, identity prompts via dialoguer, progress bars, actionable next steps
- **Agent onboarding** — `engraph init --detect --json` for vault inspection, `--json` for non-interactive apply. Two-phase detect → apply flow for AI agents.
- **`identity` MCP tool + CLI + HTTP** — returns compact L0/L1 identity block (~170 tokens) for AI session context
- **`setup` MCP tool + HTTP** — first-time setup from inside an MCP session (detect/apply modes)
- **`identity_facts` table** — SQLite storage for L0 (static identity) and L1 (dynamic context) facts
- **L1 auto-extraction** — active projects, key people, current focus, OOO status, blocking items extracted during `engraph index`
- **`engraph identity --refresh`** — re-extract L1 facts without full reindex
- **`[identity]` config section** — name, role, vault_purpose in config.toml
- **`[memory]` config section** — feature flags for identity/timeline/mining

### Changed
- MCP tools: 23 → 25
- HTTP endpoints: 24 → 26
- Dependencies: +dialoguer 0.12, +console 0.16, +regex 1

## v1.5.5 — Housekeeping (2026-04-10)

### Added
- **`auto_link` parameter** on `create` — set to `false` to skip automatic wikilink resolution. Applies to MCP, HTTP, and CLI. Discovered links still appear as suggestions in the response.
- **`reindex_file` MCP tool + HTTP endpoint** — re-indexes a single file after external edits. Reads from disk, re-embeds chunks, rebuilds edges. Available as MCP tool, `POST /api/reindex-file`, and OpenAPI operation.

### Changed
- **rmcp** bumped from 1.2.0 to 1.4.0 — host validation, non-Send handler support, transport fixes. Does not yet fix [#20](https://github.com/devwhodevs/engraph/issues/20) (protocol `2025-11-25` needed for Claude Desktop Cowork/Code modes — blocked upstream on [modelcontextprotocol/rust-sdk#800](https://github.com/modelcontextprotocol/rust-sdk/issues/800)).
- MCP tools: 22 → 23
- HTTP endpoints: 23 → 24
- OpenAPI version: 1.5.0 → 1.5.5

## v1.5.0 — ChatGPT Actions (2026-03-26)

### Added
- **OpenAPI 3.1.0 spec** (`openapi.rs`) — hand-written spec for all 23 endpoints, served at `GET /openapi.json`
- **ChatGPT plugin manifest** — served at `GET /.well-known/ai-plugin.json`
- **`--setup-chatgpt` CLI helper** — interactive setup: enables HTTP, creates API key, configures CORS, prompts for public URL
- **Plugin config** — `[http.plugin]` section for name, description, contact_email, public_url

### Changed
- Module count: 25 → 26
- Test count: 417 → 426
- `/openapi.json` and `/.well-known/ai-plugin.json` routes require no authentication

## v1.4.0 — PARA Migration (2026-03-26)

### Added
- **PARA migration engine** (`migrate.rs`) — AI-assisted vault restructuring into Projects/Areas/Resources/Archive
- **Heuristic classification** — priority-ordered rules detect Projects (tasks, active status), Areas (recurring topics), Resources (people, reference), Archive (done, inactive)
- **Preview-then-apply workflow** — generates markdown + JSON preview for review before moving files
- **Migration rollback** — `engraph migrate para --undo` reverses the last migration
- **3 new MCP tools** — `migrate_preview`, `migrate_apply`, `migrate_undo`
- **3 new HTTP endpoints** — `POST /api/migrate/preview`, `/apply`, `/undo`
- **Migration log** — SQLite table tracks all moves for rollback support

### Changed
- Module count: 24 → 25
- MCP tools: 19 → 22
- HTTP endpoints: 20 → 23
- Test count: 385 → 417

## v1.3.0 — HTTP/REST Transport (2026-03-26)

### Added
- **HTTP REST API** (`http.rs`) — axum-based HTTP server alongside MCP, enabled via `engraph serve --http`
- **20 REST endpoints** mirroring all 19 MCP tools + update-metadata
- **API key authentication** — `eg_` prefixed keys with read/write permission levels
- **Rate limiting** — configurable per-key token bucket (requests/minute)
- **CORS** — configurable allowed origins for web-based agents
- **Graceful shutdown** — CancellationToken coordinates MCP + HTTP + watcher exit
- **API key management CLI** — `engraph configure --add-api-key/--list-api-keys/--revoke-api-key`
- **`--no-auth` mode** — local development without API keys (127.0.0.1 only)

### Changed
- `engraph serve` gains `--http`, `--port`, `--host`, `--no-auth` flags
- Module count: 23 → 24
- Test count: 361 → 385
- New dependencies: axum, tower-http, tower, rand, tokio-util

## v1.2.0 — Temporal Search (2026-03-26)

### Added
- **Temporal search lane** (`temporal.rs`) — 5th RRF lane for time-aware queries
- **Date extraction** — from frontmatter `date:` field or `YYYY-MM-DD` filename pattern
- **Heuristic date parsing** — "today", "yesterday", "last week", "this month", "recent", month names, ISO dates, date ranges
- **LLM date extraction** — orchestrator detects temporal intent and extracts date ranges from natural language
- **Temporal scoring** — smooth decay function for files near but outside the target date range
- **Temporal candidate injection** — date-matched files enter candidate pool as graph seeds
- **Confidence % display** — search results show normalized confidence (0-100%) instead of raw RRF scores
- **Date coverage stats** — `engraph status` shows how many files have extractable dates

### Changed
- `QueryIntent` gains `Temporal` variant with custom lane weights (temporal: 1.5)
- `OrchestrationResult` gains `date_range` field (backward-compatible serde)
- `LaneWeights` gains `temporal` field (0.0 for non-temporal intents)
- `insert_file` signature extended with `note_date` parameter
- Module count: 22 → 23
- Test count: 318 → 361

## [1.1.0] - 2026-03-26 — Complete Vault Gateway

### Added
- **Section parser** (`markdown.rs`) — heading detection, section extraction, frontmatter splitting
- **Obsidian CLI wrapper** (`obsidian.rs`) — process detection, circuit breaker (Closed/Degraded/Open), async CLI delegation
- **Vault health** (`health.rs`) — orphan detection, broken link detection, stale notes, tag hygiene
- **Section-level editing** — `edit_note()` with replace/prepend/append modes targeting specific headings
- **Note rewriting** — `rewrite_note()` with frontmatter preservation
- **Frontmatter mutations** — `edit_frontmatter()` with granular set/remove/add_tag/remove_tag/add_alias/remove_alias ops
- **Hard delete** — `delete_note()` with soft (archive) and hard (permanent) modes
- **Section reading** — `read_section()` in context engine for targeted note section access
- **Enhanced file resolution** — fuzzy Levenshtein matching as final fallback in `resolve_file()`
- **6 new MCP tools** — `read_section`, `health`, `edit`, `rewrite`, `edit_frontmatter`, `delete`
- **CLI events table** — audit log for CLI operations
- **Watcher coordination** — `recent_writes` map prevents double re-indexing of MCP-written files
- **Content-based role detection** — detect people/daily/archive folders by content patterns, not just names
- **Enhanced onboarding** — `engraph init` detects Obsidian CLI + AI agents, `engraph configure` has new flags
- **Config sections** — `[obsidian]` and `[agents]` in config.toml

### Changed
- Module count: 19 → 22
- MCP tools: 13 → 19
- Test count: 270 → 318

## [1.0.2] - 2026-03-26

### Fixed
- **Person search uses FTS** — `context who` now finds person notes via full-text search instead of exact filename matching. Handles hyphens, underscores, any vault structure. Prefers People folder → `person` tag → fuzzy filename.
- **llama.cpp logs suppressed** — `backend.void_logs()` silences Metal/model loading output. Clean terminal output by default.
- **Basename resolution** — `find_file_by_basename` normalizes hyphens/underscores/spaces for cross-format matching.

### Changed
- Re-recorded demo GIF with v1.0.2 brew binary (clean output, no `2>/dev/null` workarounds)

## [1.0.1] - 2026-03-26

### Changed
- **Inference backend switched from candle to llama.cpp** — via `llama-cpp-2` Rust bindings. Gets full Metal GPU acceleration on macOS (88 files indexed in 70s vs 37+ minutes on CPU with candle). Same backend as [qmd](https://github.com/tobi/qmd).
- Default embedding model produces 256-dim vectors via embeddinggemma-300M (Matryoshka truncation)
- BERT GGUF architecture support added alongside Gemma (future model flexibility)
- Progress bar during indexing via indicatif (was silent for minutes)
- CI workflow installs CMake on Ubuntu (required for llama.cpp build)

### Fixed
- **Prompt format applied during embedding** — `embed_one` uses search_query prefix, `embed_batch` uses search_document prefix. Without this, embeddinggemma operated in wrong symmetric mode.
- **GGUF tokenizer fallback** — added `shimmytok` crate to extract tokenizer from GGUF metadata when tokenizer.json is unavailable (Google Gemma repos are gated)
- **LlamaBackend singleton** — global `OnceLock` prevents double-initialization crash when loading multiple models
- **Orchestrator/reranker use built-in tokenizer** — llama.cpp reads tokenizer from GGUF metadata, no external tokenizer.json needed
- **Dimension migration clears FTS** — `reset_for_reindex` now also clears `chunks_fts` to prevent duplicate entries
- **LLM cache wired into search** — `search_with_intelligence` checks/populates `llm_cache` table
- **MCP server wires intelligence** — search handler passes orchestrator + reranker via `SearchConfig`
- **CLI search wires intelligence** — `run_search` loads models when intelligence enabled
- **Qwen3 GGUF filename** — fixed case sensitivity (was 404)
- **Embedding batch params** — `n_ubatch >= n_tokens` assertion, use `encode()` not `decode()`, `AddBos::Never` (PromptFormat adds `<bos>`)

### Removed
- `candle-core`, `candle-nn`, `candle-transformers` dependencies (replaced by `llama-cpp-2`)

## [1.0.0] - 2026-03-25

Intelligence release. Replaced ONNX with GGUF model inference, added LLM-powered search intelligence. Immediately followed by v1.0.1 which switched the inference backend from candle to llama.cpp for Metal GPU support.

### Added
- **GGUF model inference** — replaced ONNX (`ort`) with GGUF quantized models for all ML inference
- **Research orchestrator** — LLM-based query classification (exact/conceptual/relationship/exploratory) with adaptive lane weights. Single LLM call returns intent + 2-4 query expansions.
- **Cross-encoder reranker** — 4th RRF lane using Qwen3-Reranker for relevance scoring. Two-pass fusion: 3-lane retrieval → reranker scores top 30 → 4-lane RRF.
- **Query expansion** — each search runs multiple expanded queries through all retrieval lanes, merged via deduplication.
- **Heuristic orchestrator** — fast-path intent classification via pattern matching (docids, ticket IDs, "who" queries) when intelligence is disabled. Zero latency.
- **Intelligence onboarding** — opt-in prompt during `engraph init` and first `engraph index`. Downloads ~1.3GB of optional models.
- **`engraph configure` command** — `--enable-intelligence`, `--disable-intelligence`, `--model embed|rerank|expand <uri>` for model overrides.
- **Dimension migration** — auto-detects embedding dimension changes and triggers re-index.
- **LLM result cache** — SQLite cache for orchestrator results (keyed by query SHA256).
- **Model override support** — configurable embedding, reranker, and expansion model URIs for multilingual support.

### Changed
- Embedding model: `all-MiniLM-L6-v2` (ONNX, 384-dim, 23MB) → `embeddinggemma-300M` (GGUF, 256-dim, ~300MB)
- Search pipeline: hardcoded 3-lane weights → adaptive per-query-intent weights
- `--explain` output now shows query intent and 4-lane breakdown (semantic, FTS, graph, rerank)
- `status` command shows intelligence enabled/disabled state

### Removed
- `ort` (ONNX Runtime) dependency
- `ndarray` dependency
- `src/embedder.rs` and `src/model.rs` (replaced by `src/llm.rs`)
- `ModelBackend` trait (replaced by `EmbedModel`)

## [0.7.0] - 2026-03-25

### Added
- **File watcher** — `engraph serve` now watches the vault for changes and re-indexes automatically (2s debounce)
- **Placement correction learning** — detects when users move notes from suggested folders, updates centroids
- **Fuzzy link matching** — sliding window Levenshtein matching (0.92 threshold) during note creation
- **First-name matching** — matches "Steve" to `[[Steve Barbera]]` for People folder notes (suggestion-only)
- `created_by` column and filter — track note origin, filter with `engraph context list --created-by`
- `placement_corrections` table for observability
- `link_skiplist` table schema (reserved for future use)

### Changed
- Centroid updates use true online mean (was EMA 0.9/0.1)
- Indexer refactored: `index_file`, `remove_file`, `rename_file` extracted as public functions
- Bulk indexing uses batched transactions for performance
- `run_index_shared` variant accepts external store/embedder references

### Fixed
- Content hash consistency between `diff_vault` and `index_file` (BOM handling)

## [0.6.0] - 2026-03-25

### Added
- **Write pipeline** — create, append, update_metadata, move, archive, unarchive notes
- **sqlite-vec** replaces HNSW for vector search (single SQLite database)
- **Tag registry** with fuzzy Levenshtein resolution
- **Link discovery** — exact basename and alias matching during note creation
- **Folder placement** — type rules, semantic centroids, inbox fallback
- **Archive/unarchive** — soft delete with metadata preservation
- 6 new MCP write tools (13 total)

### Changed
- All vectors stored in SQLite vec0 virtual table (was HNSW + separate files)
- Atomic writes via temp file + rename for crash safety
- Mtime-based conflict detection for concurrent edits

## [0.5.0] - 2026-03-24

### Added
- **MCP server** — `engraph serve` starts stdio MCP server via rmcp SDK
- 7 read-only MCP tools: search, read, list, vault_map, who, project, context

## [0.4.0] - 2026-03-24

### Added
- **Context engine** — 6 functions: read, list, vault_map, who, project, topic
- Token-budgeted context bundles for AI agents
- Person and project context assembly from graph + search

## [0.3.0] - 2026-03-24

### Added
- **Vault graph** — bidirectional wikilink + mention edges built during indexing
- **Graph search agent** — 3rd RRF lane with 1-2 hop expansion
- People detection from configured People folder

## [0.2.0] - 2026-03-24

### Added
- **Hybrid search** — semantic (embeddings) + keyword (FTS5 BM25) fused via RRF
- Smart chunking with break-point scoring algorithm
- Docid system (6-char hex file IDs)
- Vault profiles with auto-detection (`engraph init`)
- Pluggable model layer (`ModelBackend` trait)
- `--explain` flag for per-lane score breakdown

## [0.1.0] - 2026-03-19

### Added
- Initial release
- ONNX embedding model (all-MiniLM-L6-v2, 384-dim)
- SQLite metadata storage
- Incremental indexing
- `.gitignore`-aware vault walking
