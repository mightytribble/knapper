# Changelog

Every entry below the upstream-lineage line names the commit it describes.
Links resolve against https://github.com/mightytribble/knapper.

## Unreleased

### Changed

- `SKILL.md` is an operating guide, not a tool list. The plugin ships the skill beside the MCP server, so the tools are already in context; the skill now carries only what a tool description cannot. ([`a8dc3a1`](https://github.com/mightytribble/knapper/commit/a8dc3a1))
- `mcp-setup.md` covers `--scope` and `KNAPPER_HOME`, including a per-project vault via `env` in `.mcp.json`, and warns that a second data directory downloads the embedder again. ([`a8dc3a1`](https://github.com/mightytribble/knapper/commit/a8dc3a1))
- The README is a landing page: the pitch, what knapper does, a quick start that ends with a working search and a registered MCP server, and a table pointing at the rest. It carried the whole manual before — 794 lines down to 153. ([`2ca66bd`](https://github.com/mightytribble/knapper/commit/2ca66bd))
- reference material split into separate documents: `install.md` (renamed from `deployment.md`), `configuration.md`, `how-knapper-searches.md`, `faq.md`, `http-rest-api.md` and `chatgpt-actions.md`. Added previously undocumented details to `configuration.md`.  ([`7e06471`](https://github.com/mightytribble/knapper/commit/7e06471))

### Removed

- `knapper configure --register <agent>` is gone. It printed MCP setup instructions and set a flag under `[agents]` that nothing ever read, so registering an agent changed no behaviour. The setup instructions live in [install.md](install.md). ([`5e6505f`](https://github.com/mightytribble/knapper/commit/5e6505f))

### Fixed

- The MCP server now correctly tells a client which tools it has. Future changes to tool surface descriptions are now declared in code, not prose. ([`005441b`](https://github.com/mightytribble/knapper/commit/005441b))
- A search result is a chunk, not a section. Docs said otherwise; now they don't. ([`a5dec4b`](https://github.com/mightytribble/knapper/commit/a5dec4b))
- Fixed `max_chunks_per_file` comments to reflect code reality. ([`6178f8e`](https://github.com/mightytribble/knapper/commit/6178f8e))
- Fixed `configure --revoke-api-key` and `[[http.api_keys]]` comments to reflect reality. ([`7e06471`](https://github.com/mightytribble/knapper/commit/7e06471))
- Clarified `KNAPPER_HOME` docs for multi-vault setup. ([`2a08156`](https://github.com/mightytribble/knapper/commit/2a08156))
- `--property <list> --mode replace` collapses the list to a scalar, not just drops the siblings. The skill warns with a before and after; use `append` or `remove`. ([`a8dc3a1`](https://github.com/mightytribble/knapper/commit/a8dc3a1))
- Dropped the `topic`, `who` and `project` commands and `POST /api/topic` from the skill and the route table. They went in #73. ([`a8dc3a1`](https://github.com/mightytribble/knapper/commit/a8dc3a1))

## 0.9.7 (2026-09-04)

knapper reads the custom properties your notes carry — frontmatter keys and
Dataview inline fields — and `list` and `search` filter on them. The vault
graph is rebuilt once on the upgrade to fill the properties table, which is a
vault read with no model call. Nothing re-embeds.

### Added

- knapper reads a vault's custom properties (#66): frontmatter keys, and Dataview inline fields such as `Mentor:: [[Bob]]` in a note's body. `knapper properties` lists the vault's property names with their note counts, the kinds seen and Obsidian's declared type, or one property's values with `--name`. `list` and `search` take `--property NAME[=VALUE]`, `--links-to NOTE` and `--linked-from NOTE`; with `--property` beside a link filter, only links filed under that property count. `read --metadata` lists a note's properties and names the property behind each link, and every search hit carries its note's frontmatter properties. `validate` warns on an unquoted `[[link]]` in frontmatter, a value that disagrees with `.obsidian/types.json`, a name that holds more than one kind, two names that differ only in case or separator, and a declared name no note carries. Writing is unchanged: `update --property` edits a frontmatter property, as it already did, and an inline field is edited as body text. Search ranking does not read the table. Same capability on the MCP `properties` tool and `GET /api/properties`. ([`b7d7be8`](https://github.com/mightytribble/knapper/commit/b7d7be8))

### Fixed

- A wikilink written before its target exists resolves once the target is there (#108). Such a link was recorded broken and stayed broken after the target was created and indexed: `health` named it, the graph did not hold it, and both notes counted as orphans. `index` did not repair it. Creating, indexing, restoring or moving a note now re-resolves the links that name it. ([`4ffeaea`](https://github.com/mightytribble/knapper/commit/4ffeaea))

- `health` reports a link whose target has been removed (#108). It reported nothing. Deleting, archiving or moving a note now records the links it leaves naming nothing. ([`4ffeaea`](https://github.com/mightytribble/knapper/commit/4ffeaea))

- knapper serves MCP clients that speak the 2026-07-28 protocol, such as Google Antigravity (#109). Such a client opens the connection with a `server/discover` request; knapper read that as a failed handshake and exited, and the client reported the connection closed. A client that opens with `initialize` connects as before. ([`fd9a9a6`](https://github.com/mightytribble/knapper/commit/fd9a9a6))

- **Test count: 1224 → 1289.**

## 0.9.6 (2026-09-03)

Qwen3-Embedding becomes a local embedder you can point `models.embed` at.
EmbeddingGemma stays the default and its stored vectors are unchanged, so no
existing store re-indexes on the upgrade.

### Added

- Added support for Qwen3-Embedding 0.6B and 4B. `knapper models list` now names embedders that are known to work. ([`6e84edc`](https://github.com/mightytribble/knapper/commit/6e84edc))

- Added default values for Gemma, Qwen3 0.6B, and Qwen 4B embedders to config. Uncomment to activate, adjust to your own corpus. ([`06b95f2`](https://github.com/mightytribble/knapper/commit/06b95f2))

### Fixed

- The default `[calibrated]` coefficients were fit against a value the search path never computes. Refitted against the right one: with no cross-encoder configured, search keeps more correct results and fewer irrelevant ones, and abstains on the same queries as before. Nothing re-indexes. ([`06b95f2`](https://github.com/mightytribble/knapper/commit/06b95f2))

- Pointing `models.embed` at a Qwen3 embedding model crashed the process. ([`6e84edc`](https://github.com/mightytribble/knapper/commit/6e84edc))

- A Qwen3 embedder pooled the last token of the note text instead of the model's end-of-text token, which weakened every stored vector. ([`6e84edc`](https://github.com/mightytribble/knapper/commit/6e84edc))

- The Qwen3 prompt templates did not match the model card. They do now. ([`6e84edc`](https://github.com/mightytribble/knapper/commit/6e84edc))

- **Test count: 1208 → 1224.**

## 0.9.5 (2026-09-02)

A feature release over 0.9.4. Nothing re-indexes on the upgrade.

### Added

- `match` reports whether a literal string is present in your notes, and where. Available on the CLI, MCP and HTTP. It takes the same scope operators as `search`, and reads note bodies only — not frontmatter. ([`6116e2e`](https://github.com/mightytribble/knapper/commit/6116e2e))

- `update` and `health` report stale heading links: a `[[Note#Heading]]` naming a heading that has been renamed or removed. The note still resolves, so these never appeared as broken links. The linking notes are not rewritten for you. ([`6588773`](https://github.com/mightytribble/knapper/commit/6588773))

### Fixed

- An edit no longer rewrites a CRLF note's line endings. Every line an edit does not name keeps the ending it came in with. ([`07d9d4f`](https://github.com/mightytribble/knapper/commit/07d9d4f))

- A section edit no longer adds a stray blank line under the heading. Blank lines at the edges of your content are kept only on the edge the edit joins. ([`216a462`](https://github.com/mightytribble/knapper/commit/216a462))

- **Test count: 1154 → 1208.**

### Changed

- The calibration guidance names note length beside the embedder. `[calibrated]` also wants a refit when your notes are much shorter or longer than the ones it was fit on; the symptom is a query that retrieves everything and returns nothing. No numbers moved. ([`e7858b0`](https://github.com/mightytribble/knapper/commit/e7858b0))

## 0.9.4 (2026-09-02)

A patch release over 0.9.3. Nothing re-indexes on the upgrade.

### Fixed

- `top_n` returns the number of results it promises. Merging ran after the list was cut, so you got fewer than you asked for, by an amount that moved with the data. ([`fc6a171`](https://github.com/mightytribble/knapper/commit/fc6a171))

- A search says when the token budget is what shortened the answer. Raise it with `--tokens` on the CLI, `budget_tokens` over MCP and HTTP. ([`fc6a171`](https://github.com/mightytribble/knapper/commit/fc6a171))

- A merged result block stays inside one section. A block could be labelled with a heading that had not matched, so following its path with a section read landed you in the wrong place. ([`e606401`](https://github.com/mightytribble/knapper/commit/e606401))

- `[calibrated]` says which embedder its numbers were fit against. `knapper index` warns when you change embedder, and a generated config carries the same note above the section. The numbers do not move, so nothing changes on the default embedder. ([`0e71556`](https://github.com/mightytribble/knapper/commit/0e71556))

- **Test count: 1136 → 1154.**

### Changed

- The refit tool ships, as `scripts/calibrated-fusion-eval.py`. Refitting `[calibrated]` is yours to do, and the tool sat in a directory you never received. ([`0e71556`](https://github.com/mightytribble/knapper/commit/0e71556))

## 0.9.3 (2026-09-01)

A patch release over 0.9.2. Nothing re-indexes on the upgrade.

### Added

- `update` renames the section it edits: `--section "old name" --heading "new name"`, and the same field over MCP and HTTP. `content` is optional beside it, because a rename does not restate the body. Renaming a section had no spelling at all before this. ([`7d7190f`](https://github.com/mightytribble/knapper/commit/7d7190f))

### Fixed

- A note you deleted, archived or soft-deleted stops reporting broken links. `health` went on naming a source file that was no longer there, and nothing could clear the row. The table is repaired in place on the upgrade. ([`c2e9396`](https://github.com/mightytribble/knapper/commit/c2e9396))

- `delete --mode soft` keeps the note searchable. It kept the note in `list` and dropped it out of every search. ([`c2e9396`](https://github.com/mightytribble/knapper/commit/c2e9396))

- What `read` returns can be written straight back through `update`. A section read carried its heading, so feeding it back wrote the heading twice; a whole-note read gained a blank line on every round trip. Both corrupted the note silently. ([`9a818cc`](https://github.com/mightytribble/knapper/commit/9a818cc), [`0f08c59`](https://github.com/mightytribble/knapper/commit/0f08c59))

- **Test count: 1116 → 1136.**

## 0.9.2 (2026-08-31)

A patch release over 0.9.1. Nothing re-indexes on the upgrade.

### Added

- The `:cpu` and `:cuda` images publish to GHCR on release, so `docker pull ghcr.io/mightytribble/knapper` is the install. ([`b24d8be`](https://github.com/mightytribble/knapper/commit/b24d8be))

### Changed

- A config save records what you chose, not what the binary shipped with. `knapper configure` wrote every default into `~/.knapper/config.toml`, so a later release that moved one never reached you. A save now edits the file, keeping your comments, key order and spacing. A data directory with no config yet gets the whole catalogue, commented out. ([`421ba77`](https://github.com/mightytribble/knapper/commit/421ba77))

- `brew install knapper` installs the published binary on Apple Silicon and Linux x86_64, where it compiled llama.cpp from source on every platform. macOS Intel and Linux arm64 still build from source, because no binary is published for either. ([`4dcffa6`](https://github.com/mightytribble/knapper/commit/4dcffa6))

- **Test count: 1035 → 1116.**

### Fixed

- A note written through knapper keeps the rest of itself byte for byte. `create` writes your frontmatter as given and adds no keys of its own, a property edit keeps the key's position and the note's list style, and an archive round trip returns the file unchanged. A value knapper cannot edit safely refuses the write and names what it found. ([`710eae1`](https://github.com/mightytribble/knapper/commit/710eae1), the `frontmatter-preserving-writes` branch)

- An edited note stays in the index. Saving over a note made the file watcher read the save as a deletion, so `search`, `list` and `read` lost a note that was still on disk and correct. ([`bdd9965`](https://github.com/mightytribble/knapper/commit/bdd9965))

- An edit keeps the newline the note ended on, so a one-line change reads as one line of diff. ([`8fe61a8`](https://github.com/mightytribble/knapper/commit/8fe61a8))

## 0.9.1 (2026-08-30)

Search ranks and abstains with no model configured. Nothing re-indexes on the upgrade.

### Added

- With no cross-encoder configured — the default install — search scores each result as a probability and sorts by it, where it fell back to a rank-fusion order before. A query your vault cannot answer now returns nothing, on the CLI, over MCP and on `POST /api/search`; the default install could not abstain at all. On the calibration pool it declines the same non-answers the cross-encoder declines. No model call runs. ([`19ba687`](https://github.com/mightytribble/knapper/commit/19ba687))

- `[calibrated]`, the config section behind it: `enabled`, three fitted coefficients, and `floor`, the probability below which a result is not an answer. Every key is query-time, so a change re-indexes nothing. `enabled = false` restores the previous behaviour exactly, and `floor = 0.0` removes nothing. With a cross-encoder configured the section is inert. ([`c450b53`](https://github.com/mightytribble/knapper/commit/c450b53))

- `--explain` reports the working: the query's BM25 upper bound, each term's idf, and every candidate's probability. ([`915e1b8`](https://github.com/mightytribble/knapper/commit/915e1b8))

- `validate` checks your notes for broken structure and reports what it finds: unclosed code fences and wikilink brackets, unparseable frontmatter, missing or duplicated headings, empty sections, over-long paragraphs, unknown tags and wikilinks that name no note. It takes one note or a scope, reaches all three surfaces, and `--strict` makes a warning exit non-zero. ([`e9d665f`](https://github.com/mightytribble/knapper/commit/e9d665f) … [`f4f8cc6`](https://github.com/mightytribble/knapper/commit/f4f8cc6))

- An external embedder. `models.embed = "gemini:<versioned-id>"` embeds through the Gemini API instead of a local GGUF, with the key in `GEMINI_API_KEY` and never in the config file; `[models.embed_api]` holds `dim`, `timeout_secs`, `max_retries` and `endpoint`. The local form is unchanged. ([`3d9e66a`](https://github.com/mightytribble/knapper/commit/3d9e66a), [`ffa61c5`](https://github.com/mightytribble/knapper/commit/ffa61c5), [`2374eab`](https://github.com/mightytribble/knapper/commit/2374eab))

- The data directory moves. `--data-dir` and `KNAPPER_HOME` both relocate `~/.knapper`, and the store, the models and both config files follow it together. ([`3ce87ba`](https://github.com/mightytribble/knapper/commit/3ce87ba))

- `read --metadata` answers a note's frontmatter, its incoming and outgoing links and its size, in place of its content. It describes the whole note, so it cannot be combined with `--section`. ([`81020ca`](https://github.com/mightytribble/knapper/commit/81020ca))

- Docker images, `:cpu` and `:cuda`, built from a two-stage Dockerfile. ([`81bde46`](https://github.com/mightytribble/knapper/commit/81bde46))

### Changed

- Confidence on the default install is an absolute probability. It was measured against the top result, so the first answer of every query printed 100% however bad it was. ([`19ba687`](https://github.com/mightytribble/knapper/commit/19ba687))

- The file watcher polls where inotify cannot serve the filesystem. `knapper serve` on a Docker bind mount, an overlay, fuse, 9p, nfs or cifs vault never saw a change; it now detects that and polls instead. `[watcher] backend` and `KNAPPER_WATCHER_BACKEND` override the choice. ([`e613d54`](https://github.com/mightytribble/knapper/commit/e613d54))

- `status` reports the embedding model you configured, not a compiled-in constant. ([`099c6e8`](https://github.com/mightytribble/knapper/commit/099c6e8))

- `configure` reports the cross-encoder it resolved and its real download size, in place of a fixed figure. ([`71c7925`](https://github.com/mightytribble/knapper/commit/71c7925))

- `configure --register claude-code` registers through `claude mcp add`. It wrote a `settings.json` Claude Code does not read. ([`74a4310`](https://github.com/mightytribble/knapper/commit/74a4310))

- A cross-encoder knapper cannot prompt is refused at load, with the supported models named, instead of scoring every candidate wrongly. ([`78c0302`](https://github.com/mightytribble/knapper/commit/78c0302))

- **Test count: 918 → 1035.**

### Removed

- The Obsidian CLI integration, which nothing called. No surface, config key or behaviour changes with it. ([`485150d`](https://github.com/mightytribble/knapper/commit/485150d))

## 0.9.0 (2026-08-16) — knapper

The project leaves fork status: engraph, forked at v1.7.2, becomes knapper. The
binary, the repository (mightytribble/knapper), the data directory
(`~/.knapper/`) and the store file (`knapper.db`, with a read fallback to an
existing `engraph.db`) carry the new name. The MIT license and the full git
history stay; see `NOTICE`. Versions restart at 0.9.x; 1.0.0 marks the v1
milestone: functional Metal and Docker build pipelines.
([`31a8718`](https://github.com/mightytribble/knapper/commit/31a8718),
[`da2d325`](https://github.com/mightytribble/knapper/commit/da2d325),
[`c68944b`](https://github.com/mightytribble/knapper/commit/c68944b))

Three changes travel with the rename: every capability gets one name on all
three surfaces, `knapper list` answers the vault's files, and `--section`
addresses a section by its heading path.

### Removed

**The renames land with no aliases and no deprecation window.** A script or an agent that calls an old name gets an error, not a warning. This section is the whole list.

- CLI command groups. `knapper context <leaf>` and `knapper write <leaf>` are gone and every leaf is a top-level command: `context read` → `read`, `write create` → `create`, and so on for all ten. ([`c6ea35b`](https://github.com/mightytribble/knapper/commit/c6ea35b))
- `knapper graph`. `graph show` ran the queries `read` runs, so `read` answers it; `graph stats` is folded into `status`. ([`d848cb4`](https://github.com/mightytribble/knapper/commit/d848cb4))
- `knapper migrate para`. PARA is the only strategy: `knapper migrate --mode preview|apply|undo`. ([`61f0824`](https://github.com/mightytribble/knapper/commit/61f0824))
- 15 MCP tools (25 → 17). `read_section` is a `section` parameter of `read`; `append`, `edit`, `rewrite`, `edit_frontmatter` and `update_metadata` are one `update` tool; `unarchive` is `archive {undo: true}`; `setup` is `init {mode}`; the three `migrate_*` tools are `migrate {mode}`. `move_note` is renamed `move`. ([`65e2e9d`](https://github.com/mightytribble/knapper/commit/65e2e9d), [`c32bfb0`](https://github.com/mightytribble/knapper/commit/c32bfb0), [`61f0824`](https://github.com/mightytribble/knapper/commit/61f0824), [`b88e043`](https://github.com/mightytribble/knapper/commit/b88e043))
- `context`, `who` and `project`, the three composite bundle tools, with no replacement. Each one ran the queries the primitives run, in one fixed shape; the composites return as vault-defined commands. `context` was first renamed `topic`, then removed with the other two. ([`d5928b4`](https://github.com/mightytribble/knapper/commit/d5928b4))
- 14 HTTP routes (26 → 18), the same consolidation, plus path parameters: `GET /api/read/{*file}`, `/api/who/{name}` and `/api/project/{name}` now take `?file=` and `?name=`. ([`94a881a`](https://github.com/mightytribble/knapper/commit/94a881a))
- `list --folder`, and `folder` on the MCP tool and `GET /api/list`. A directory is a scope term — `--scope /lore/`, a case-sensitive path range — where `--folder lore` was a `LIKE` that folded case and matched `lorekeeper.md`. ([`a411850`](https://github.com/mightytribble/knapper/commit/a411850), [`7e97141`](https://github.com/mightytribble/knapper/commit/7e97141))
- `rewrite`'s `preserve_frontmatter: false`, `update_metadata`'s `modified_by` stamp, and `total_files` from the `status` JSON. ([`182ff48`](https://github.com/mightytribble/knapper/commit/182ff48), [`dd65285`](https://github.com/mightytribble/knapper/commit/dd65285))
- `migrate` `mode: apply` no longer falls back to a preview file on disk over MCP and HTTP. Pass the preview your own `mode: preview` returned; the CLI's two-step flow is unchanged. ([`eb61dbd`](https://github.com/mightytribble/knapper/commit/eb61dbd))

### Added

- `update`, one capability for every change to an existing note. It takes a list of edits and applies them in order in one write: one conflict check, one file write, one re-index. Each edit names a `section`, a `property`, or neither (the body), with a `mode` of `replace`, `prepend`, `append` or `remove`. Two sections and a tag change in one atomic write is something no call it replaces could do. ([`da4c12a`](https://github.com/mightytribble/knapper/commit/da4c12a), [`c32bfb0`](https://github.com/mightytribble/knapper/commit/c32bfb0))
- `knapper list` answers every note a scope admits, in path order, one bare path per line — so it pipes, and `wc -l` is the total. `--detailed` puts each note's heading outline under its path. ([`1a07a66`](https://github.com/mightytribble/knapper/commit/1a07a66), [`32b4450`](https://github.com/mightytribble/knapper/commit/32b4450), [`9d16472`](https://github.com/mightytribble/knapper/commit/9d16472))
- `--section` names a section by its heading text or by its full heading path: `--section "About the Empire > Current Events > History"` reaches the second `History` of a note that holds two. A partial path resolves nothing, so a wrong guess is an error and never an edit to another section. A promoted bold line such as `**Spells**` is addressable too, and was reachable by no name at all; so is an empty section, because addressing one is how you fill it. ([`aa670a3`](https://github.com/mightytribble/knapper/commit/aa670a3), [`19b4c8a`](https://github.com/mightytribble/knapper/commit/19b4c8a))
- Six MCP tools — `index`, `status`, `tags`, `update`, `init` and `migrate` — and the six HTTP routes beside them fill the gaps, and `health`, `reindex-file`, `move`, `update`, `status` and `index` each reach the CLI. ([`c6ea35b`](https://github.com/mightytribble/knapper/commit/c6ea35b), [`fde3cb6`](https://github.com/mightytribble/knapper/commit/fde3cb6))
- `explain` and `group_by` are per call on every surface, so one query answers the same way whoever asks it. ([`4720fa0`](https://github.com/mightytribble/knapper/commit/4720fa0))
- `GET /api/identity?refresh=` and MCP `identity {refresh}` re-extract the L1 facts, which was a CLI-only flag. It writes, so a read-only server refuses it. ([`521d05f`](https://github.com/mightytribble/knapper/commit/521d05f))
- `docs/surfaces.md`, generated and checked by a test, listing what every capability is called on each surface. ([`cf079cf`](https://github.com/mightytribble/knapper/commit/cf079cf), [`016bc19`](https://github.com/mightytribble/knapper/commit/016bc19))

### Changed

- `search`'s default `top_n` on both servers is the configured one, whose default is **5**. It was a hardcoded 10, so a caller that relied on ten results must now ask for `top_n: 10`. ([`06e03e3`](https://github.com/mightytribble/knapper/commit/06e03e3))
- `search` over HTTP returns `{"results": [...], "message": ...}` in place of a bare array. HTTP was the one surface with nowhere to put the answer-floor signal. ([`0aefc5f`](https://github.com/mightytribble/knapper/commit/0aefc5f))
- `list` has no default limit. It was capped at 20; a bare `knapper list` now answers every note in scope, and `--limit 0` answers none. ([`1a07a66`](https://github.com/mightytribble/knapper/commit/1a07a66))
- `update`'s `--mode` defaults to `replace`, and an empty stdin read is refused for a body or section replace rather than blanking the note. `--content ""` is the deliberate spelling for that. ([`07f1125`](https://github.com/mightytribble/knapper/commit/07f1125))
- `update` checks the mtime, so a note changed outside knapper and not yet re-indexed fails instead of being overwritten. ([`da4c12a`](https://github.com/mightytribble/knapper/commit/da4c12a))
- `delete`'s `mode` is an enum on all three surfaces. It read anything but `"hard"` as soft, so `mode: "hardd"` archived the note silently. ([`06e03e3`](https://github.com/mightytribble/knapper/commit/06e03e3))
- A read-only server refuses `index` and `init {mode: apply}`, as it already refused the write calls. ([`07f1125`](https://github.com/mightytribble/knapper/commit/07f1125), [`521d05f`](https://github.com/mightytribble/knapper/commit/521d05f))
- **CLI commands: 13 → 21. MCP tools: 25 → 17. HTTP routes: 26 → 18.**
- **Test count: 785 → 918.**

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
