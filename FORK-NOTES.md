# Fork notes

Private fork of [devwhodevs/engraph](https://github.com/devwhodevs/engraph) v1.7.2, maintained at
`mightytribble/engraph`. Evaluated 2026-08-06 as a knowledge-lookup layer for Obsidian-format
world stores (`cc-isekai`, `cc-pluribus`).

## Why this fork exists

Upstream is dormant — last commit 2026-05-27, seven PRs open and unmerged, several of which fix
real defects. Rather than wait, this fork carries the fixes we need.

**Divergence from upstream is deliberately minimal.** Track it with:

```bash
git fetch upstream && git diff --stat upstream/main main
```

| commit | what | origin |
|---|---|---|
| `a19f27a` | chunker overlap-stride crawl | cherry-pick of upstream PR #41 (`ec7b06b`, @jdubdevs) |
| `structure_chunk` | structure-first chunking | this fork, issue #1 |
| `src/exclude.rs` | glob `exclude` patterns, shared by indexer + watcher | this fork, issue #7 |
| `chunks.seq` | chunk identity — section-level retrieval | this fork, issue #6 |
| `src/prefix.rs` | contextual embedding prefix, **off by default** | this fork, issue #2 |
| `insert_fts_chunk` | index the whole chunk, not its 200-char snippet | this fork, issue #11 |
| `ensure_embedding_dim` | embed at the model's native width; no hidden truncation | this fork, issue #12 |
| `embed_formatted` / `rerank_batch` | one llama.cpp context per batch, not per call | this fork, issue #13 |
| `chunks.text` | the reranker scores the whole chunk, not a preview | this fork, issue #14 |
| `from_intent` `Relationship` | graph no longer outweighs the content lanes | this fork, issue #9 |
| `resolve_n_threads` | llama.cpp runs on the machine's cores, not a constant 4 | this fork, issue #20 |
| `fts::any_term_expr` | the keyword lane matches terms, not the query as one phrase | this fork, issue #22 |
| `ensure_original` | the user's own query is always searched for | this fork, issue #23 |
| `.github/workflows/ci.yml` | manual dispatch only — upstream runs it on push and PR | this fork, Actions minutes |

Cherry-picked rather than merged: PR #41 branched before upstream's #40 graph fix, so merging the
branch wholesale would have silently reverted `src/graph.rs`.

To rebase on a future upstream release:

```bash
git fetch upstream
git rebase upstream/main        # or: git merge upstream/main
```

Files added by this fork (`FORK-NOTES.md`, `eval/`) are new paths and never conflict on rebase.

## Building

Ubuntu 24.04 / WSL2, **no sudo required**. `llama-cpp-sys-2` compiles llama.cpp from source, which
is where all the friction lives — the Rust itself needs nothing special.

```bash
# 1. Rust (durable, user-local)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# 2. cmake + libclang (Ubuntu 24.04 blocks system pip, so use a venv)
python3 -m venv ~/.engraph-buildenv
~/.engraph-buildenv/bin/pip install cmake libclang

# 3. Build
export PATH="$HOME/.cargo/bin:$HOME/.engraph-buildenv/bin:$PATH"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export LIBCLANG_PATH="$HOME/.engraph-buildenv/lib/python3.12/site-packages/clang/native"
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include -I/usr/include -I/usr/include/x86_64-linux-gnu"

cargo build --release        # ~10 min cold, ~20s incremental
cargo test --lib             # 563 pass
```

Each env var exists for a specific failure. Omit one and you get:

| omitted | failure |
|---|---|
| `CMAKE_POLICY_VERSION_MINIMUM` | cmake 4.x rejects llama.cpp's older `cmake_minimum_required` |
| `LIBCLANG_PATH` | `Unable to find libclang` — bindgen can't run |
| `BINDGEN_EXTRA_CLANG_ARGS` | `fatal error: 'stdbool.h' file not found` (pip libclang ships the .so, not clang's builtin headers) |

Adjust the gcc version in the include path (`13`) and the python version (`python3.12`) to match the box.

### Known pre-existing test failures

`cargo test` (full) fails to compile `tests/integration.rs` and `tests/write_pipeline.rs`:
`unresolved import engraph::embedder`, `engraph::hnsw`, and a `walk_vault` arity mismatch.
**These are broken on pristine upstream** — verify with `git stash && cargo clippy --all-targets`.
Upstream PR #47 addresses them. Use `cargo test --lib` (563 tests) as the working suite.
`cargo clippy -- -D warnings`, which is what CI runs, is clean.

### CI is manual-only in this fork

`.github/workflows/ci.yml` triggers on `workflow_dispatch` and nothing else — upstream runs it on
every push and PR to `main`. The hosted run duplicated checks that take seconds here and billed
Actions minutes for it, two jobs per push, with the macOS leg charged at 10× the minute rate.

**So the gate is local, and it is not optional.** Before every commit:

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test --lib
```

Run the hosted matrix deliberately (`gh workflow run ci.yml`) when a change needs checking against
macOS or a clean Ubuntu toolchain — anything touching llama.cpp bindings, `#[cfg]` branches, or the
build script. `resolve_n_threads` is the current example: its Linux path reads sysfs and its fallback
has never executed on this box.

`release.yml` is untouched. It fires only on `v*` tags, so it cannot go off by accident.

## Runtime gotchas

- **Data dir is hardcoded** to `dirs::home_dir()/.engraph` (`src/config.rs:169`) — no env var, no CLI
  flag, no config key. `vault_path` is a single `Option<PathBuf>`: one vault per instance.
- **Isolation via `$HOME`.** `dirs` reads `$HOME` on Linux, so `HOME=/path/to/store engraph …` gives
  per-vault (and per-git-branch) datastores. Symlink `.engraph/models` between homes or each
  re-downloads 300MB (1.6GB with intelligence enabled).
- **MCP servers launch once per session**, so a mid-session `git checkout` leaves the server pointed
  at the previous branch's store.
- **`engraph status` misreports the model** as `all-MiniLM-L6-v2` while actually loading
  `embeddinggemma-300M`. Upstream PR #48 fixes it.
- **Intelligence is not a quality dial.** Enabling it (query expansion + Qwen3 reranker, 1.6GB)
  *regressed* exact-name lookup in testing. Treat on/off as distinct configurations.
- **A `LlamaContext` spans a batch, not a call** (this fork, issue #13). `embed_formatted` and
  `rerank_batch` each create one context and run the whole slice through it, clearing the KV cache
  between items. Upstream created one per text and per (query, document) pair — 1598 per reindex, 30
  per reranked search — and blamed `!Send`, which is true of the struct *field* and irrelevant to the
  loop; the binding constraint is that `new_context<'a>(&'a self, …)` borrows the model, so a context
  cannot live beside it in one struct. **The latency this recovers is small: context setup for these
  model sizes is 1–3 ms.** See `eval/probes.md`.
- **`batch_size` is nearly inert even after #13.** `index_file` batches *within a file*
  (`texts.chunks(config.batch_size)` sits inside the per-file loop), and the eval vault averages 6.5
  chunks per file, so the default of 64 never fills a batch. Contexts went 1598 → 247, not 1598 → 25.
  Batching across files is part of #13's unbuilt phase 2.
- **The rerank lane is now about half of a reranked query** (this fork, issue #14). Since the
  cross-encoder started reading whole chunks instead of 200-character previews, a warm reranked
  search costs 11.87 s where it cost 7.90 s (n=20 medians, non-overlapping distributions). At
  ~1.3 ms per token through Qwen3-Reranker-0.6B, 30 candidates × ~155 tokens is ~5.9 s of it. The
  obvious lever is `rerank_candidates`, **hardcoded at 30** in all four places a `SearchConfig` is
  built and not currently exposed in `config.toml`.
- **`chunks.snippet` is derived, not supplied** (this fork, issue #14). `Store::insert_chunk` calls
  the chunker's `make_snippet` on the text it is given, so a caller passes one string, not two. That
  is what makes the transposition in #11 — where the preview went to FTS and the text went nowhere —
  unreachable rather than merely fixed. Databases predating the column backfill from `chunks_fts` on
  first open; the backfill is exact on the eval vault (0 mismatches, 0 empties, 1598 chunks).
- **The reranker does not rerank** (issue #15). Its scores go into a second RRF pass as a fourth
  lane, so a calibrated probability becomes a rank and then `weight/(60+rank)`, averaged with the
  lanes it exists to correct.
- **The embedding width is the model's, read from the GGUF at load time** (this fork, issue #12).
  `LlamaEmbed::new` takes it from `LlamaModel::n_embd()`, so a `models.embed` override changes the
  dimension along with the model. Upstream hardcoded 256 in `ModelDefaults::embed_dim` and truncated
  every vector to its first 256 of 768 — silently, with no config key and no relationship to the
  model loaded. **Every measurement in `eval/` recorded before this was taken at a third of the
  model's dimensionality.**
- **Upgrading past #12 re-indexes the vault on the first `engraph index`.** `run_index_inner` calls
  `store.ensure_embedding_dim`, which rebuilds `chunks_vec` at the model's width and discards every
  chunk indexed at the old one. It is automatic and prints what it is doing, but it is a full
  reindex, and until it runs the index is unreadable: `search` and `serve` refuse to start against a
  width the model does not produce rather than let sqlite-vec raise a shape error.
- **`chunks_vec` does not exist until something establishes its width.** `Store::init` runs before
  any model is loaded, so it cannot size the table and no longer guesses; a database that has never
  been indexed simply has no vec table, and the semantic lane returns nothing. The width comes from
  `ensure_embedding_dim` at index time, or from the first vector written.
- **Every embedding is computed with the wrong model's prompt prefix** (issue #10).
  `PromptFormat::EmbeddingGemma` emits `<bos>search_query:` / `<bos>search_document: {title} {text}`,
  which is *nomic-embed-text*'s convention; EmbeddingGemma documents
  `task: search result | query: …` and `title: {title | "none"} | text: …`. Affects queries and
  documents alike, so it is a floor under every retrieval measurement in `eval/`.
  Two traps when fixing it: the documented strings contain **no `<bos>`**, and `str_to_token` is
  called with `AddBos::Never` because the current string supplies one literally
  (`parse_special = true` in llama-cpp-2, so it does become the real BOS token) — swap the strings
  without addressing that and every embedding loses its BOS. Changing the *document* side needs
  `--reindex`; changing the *query* side needs nothing.
- **`exclude` is `.gitignore`-style globs** (`src/exclude.rs`, this fork). No separator means the
  pattern matches at any depth; a trailing `/` means a directory and its contents; an embedded `/`
  anchors to the vault root. Adding a pattern purges what is already indexed on the next run, via
  `diff_vault`'s deleted-files path. Upstream matched patterns as substrings, where `*.canvas` hit
  nothing at all.
- **Derived index files poison the graph lane.** Obsidian-style `*-index.md` cheat sheets link to
  everything and are linked to by nothing, making them the largest out-hubs in the vault. Any two
  notes they list become 2-hop neighbours, so graph expansion returns siblings-of-a-category rather
  than answers. Exclude them.
- **Retrieval is section-level** (this fork, issue #6). Lanes dedup and fuse on `(file_id, seq)`,
  so a document contributes as many sections as it has good ones, up to `max_chunks_per_file`
  (default 3). `group_by = "file"` / `--group-by file` restores upstream's one-result-per-document
  behaviour — it is the same code path with a cap of 1. Upstream collapses every lane to one result
  per `file_path` before fusion, so however good the chunking, the chunk that won a document's only
  slot was often not the relevant one.
- **`chunks.seq` is load-bearing and implicit.** It must equal the `chunk_seq` written to
  `chunks_fts` for the same chunk, because that pair is the only key the semantic and FTS lanes can
  both produce. Nothing enforces it at the type level — all four insert sites pass the loop index to
  both calls. Databases predating the column are backfilled by rowid order within each file, which
  reproduces the original numbering exactly (verified against a from-scratch index).
- **A contextual embedding prefix is available and switched off** (`[embedding_prefix]`, this fork,
  issue #2). It prepends document identity to the text handed to the embedder only — storage,
  snippets and FTS keep the raw chunk. It is off because the prefix is a *per-file constant*: adding
  the same string to every chunk of a document separates documents from each other and de-separates
  a document's own sections, which is the ordering #6 made load-bearing. On the eval vault it moved
  probe 3 from @5 to @2 and knocked probe 4's answer out of the top 20. Full numbers and the
  mechanism are in `eval/probes.md`.
- **Changing `[embedding_prefix]` needs `engraph index --reindex`.** Incremental indexing compares
  content hashes, so a config change alone leaves every existing vector as it was and silently mixes
  two embedding schemes in one vector space.
- **`chunks_fts` is where a chunk's full text lives** (this fork, issue #11) — nowhere else does.
  `chunks` has no `text` column, only `snippet`, the leading 200 characters kept for display. So
  whatever is not passed to `insert_fts_chunk` is unreachable by keyword search, permanently.
  Upstream passed `snippet` at all four sites: 79.7% of eval-vault chunks were truncated and keyword
  search reached 27.6% of the corpus. `Saltmere`, a place name in two notes, returned zero hits.
- **Any lexical measurement taken before #11 was taken against a 27.6%-visible index**, including
  #6's and #7's. It moved 73 of 100 probe result slots. Fixing it changes BM25 length normalisation
  for *every* chunk — average document length goes from a near-constant 200 characters to the real
  distribution — so ranks move in both directions rather than simply improving.
- **The graph lane depends on the FTS index too.** `graph.rs` picks which section a neighbour
  matched via `store.best_matching_chunk_seq`, which queries `chunks_fts`. Before #11 it was choosing
  sections while able to see the first 200 characters of each.
- **Changing what goes into FTS needs `engraph index --reindex`**, for the same reason
  `[embedding_prefix]` does: incremental indexing compares content hashes and will not notice. There
  is no backfill possible here — the full text exists nowhere in the database to recover it from.
- **RRF scores tie constantly**, since every lane hands out the same `weight/(k + rank)` values.
  Sorting them without a tiebreak means `HashMap` order decides the ranking, and results vary
  run-to-run — they did, from about rank 7 down, until #6 added tiebreaks in `fusion.rs` and
  `graph.rs`. Worth remembering before trusting any A/B measurement taken before that.

## Open work

See issues on this repo:

- ~~**#1** structure-first chunking (section → sub-section → paragraph → size)~~ — **done**, 93.1%
  heading attribution (best measured). Retrieval barely moved; see #6 for why.
- ~~**#2** contextual embedding prefix (filename / heading path / tags into every chunk)~~ —
  **built, measured, shipped off by default.** Net negative on the seed probes: +3 ranks on the
  conceptual probe, −4 (out of the window) on the exact-name non-regression probe. Both of its
  retrieval acceptance criteria turned out to have been met already by #1 + #6. Whether a length-scaled or conditional prefix is worth
  having is an open question, and a cheap one — every component is separately switchable.
- ~~**#11** FTS indexes the 200-char snippet, not the chunk~~ — **done.** Keyword search had been
  reaching 27.6% of the corpus; `Saltmere` and four other verified-present terms returned zero hits.
  Probe 3 went @5 → **@1**, the best result any configuration has produced on it, and probe 4 held
  at @2 under the heaviest churn of any probe. Nothing regressed. Moved 73 of 100 probe slots, so
  every lexical number recorded before it is superseded.
- **#3** more probes, drawn from real usage — *not* a battery, and not a prerequisite for anything.
  The five seed probes catch regressions and resolve large effects; what they cannot do is settle a
  one-up-one-down at n=5 (#14). Fix is sample count, plus negative controls and section-vs-file
  scoring recorded separately. Calibrates #4
- **#4** relevance floor — configurable per-lane min scores so nonsense queries return nothing
- **#5** embedding model config — expose output dim, tie max chunk tokens to the model's context window
- **#8** pick a better local embedder — >512 tokens, >768 dim (pairs with #5, which exposes the knobs)
- ~~**#12** embed at the model's native dimension~~ — **done.** Every vector had been truncated to its
  first 256 of 768. The seed probes return identical verdicts at identical ranks and confidences,
  while **76 of 100 slots moved underneath**. Read at the time as the probes being blind; the better
  reading, and the one #14 confirmed by moving them, is that the top of the ranking is robust to this
  change and the churn sat in the tail. Ruled out as the probe 1 explanation (#9). The migration ran
  itself; storage roughly doubles. Optional Matryoshka truncation is deliberately left unbuilt
- **#10** the embedding prompt format is nomic-embed-text's, not EmbeddingGemma's — both query and
  document sides are out-of-distribution. Query-side fix needs **no reindex** and is the cheapest
  open experiment in the repo; document-side costs a reindex per configuration, which is the only
  reason to do it second
- ~~**#13** reuse one llama.cpp context per phase instead of one per call~~ — **done, and the
  performance premise was wrong.** All three behaviour criteria held byte-for-byte: identical index
  content, identical probes with intelligence off, identical probes with intelligence *on*. But the
  latency it was filed for is not there — index time 214.7 s → 210.3 s best-of-3 with the
  distributions overlapping, and a reranked query 8.105 s → 8.096 s (n=20 medians), which is nothing.
  Context creation costs 1–3 ms, so 30 of them cannot show up in an 8 s query. Kept for the API
  shape #14 and #15 need — `rerank_batch` hands the reranker its whole candidate set — and for
  deleting a comment that gave the wrong reason. Phase 2 (true multi-sequence decode) is unbuilt and
  is where an actual speed-up would come from
- ~~**#14** the reranker scores a truncated preview, not the chunk~~ — **done, and it is the first
  change here to cost something certain.** `chunks.text` landed as predicted; the index is
  byte-identical in every pre-existing column, the new column hashes equal to `chunks_fts` in both a
  fresh index and an in-place migration, and intelligence-off output is unchanged. The reranker was
  reading 28% of a chunk (mean 664 chars vs a 185-char preview, 79% of chunks longer than their
  preview). **The cost is +50% on a reranked query — 7.90 s → 11.87 s, distributions not
  overlapping.** The benefit is a wash on the seed probes: probe 3 gains the top slot
  (`developer-console > ## [3] SPELLS` displaced by Counterspell), probe 2 loses its correct section.
  18 of 20 slots moved. One up, one down is genuinely ambiguous at n=5 and is recorded as ambiguous;
  it wants more queries on that question, not a hold on the queue.
  Shipped on and unswitchable, because "200 chars, or a 64-token match window if FTS found it" is not
  an alternative strategy worth preserving. `[rerank] document_title` is a switch and is off
- ~~**#9** the graph lane is fused as if it were an alternative ranking, and its weight locks the
  content lanes out~~ — **done, and it is the largest single move in the record.** One constant:
  `from_intent`'s `Relationship` arm, `graph 1.5 / sem 0.8 / fts 0.8` → `graph 0.8 / sem 1.0 /
  fts 1.0`. Probe 1 lands for the first time — `temple-of-the-architect` at ranks 1, 2 and 5 and
  `archivist-lenne` at 8 with the word "who" *present*, against absent-from-top-20 in every
  configuration ever recorded. Probes 2–5 are byte-identical, intelligence on and off.
  Two things came out of it that were not expected. **`default_no_intelligence()` is never called** —
  intelligence-off still runs `search_with_intelligence`, just with `orchestrator: None`, so the
  heuristic classifier and the weight table govern every search this engine performs and the gate was
  firing in the baseline configuration all along. And **the graph lane's contribution at 0.8 is not
  zero but is not evidence of value either**: with intelligence off its only appearances across the
  five probes are two tail slots on probe 1 and two on probe 5, *the nonsense control*. A disjoint
  set contributes most where the content lanes have least to say. The category error itself is
  untouched and belongs to #15
- ~~**#22** `fts_search` phrase-quotes every query~~ — **done, and it had never worked.** The whole
  query went to FTS5 in double quotes, making it a phrase query, so a multi-word query matched only
  where the caller had already guessed the corpus's wording. **Four of the five seed probes retrieved
  zero rows from the keyword lane**; the only one that worked was the single-word probe. That is the
  answer to #19's open question — probes 2 and 4 contributed nothing from FTS because the lane was
  empty, not because it was outweighed. `fts::any_term_expr` quotes each token and joins with `OR`,
  which keeps every token literal (`BRE-1234`, `#a1b2c3`, `C++`) while letting BM25 do IDF weighting
  inside one expression. `Store::fts_search` keeps phrase semantics for `context.rs`, where identity
  resolution needs them. **Not measurable without #23** — see below
- ~~**#23** the orchestrator drops the original query~~ — **done in part.** The prompt asks for the
  original first and Qwen3-0.6B omitted it in three of three probes it answered, including replacing
  the bare name `Archdragon` with `Archdragon character` and `Archdragon concept`. So the exact-name
  probe ran with no exact name in it, and the 34-hit, 7.068-BM25 query that would have answered it
  was never issued. `ensure_original` repairs the list in `search_with_intelligence`, after **every**
  source — model, cache hit, heuristic — because #9 showed what a rule with two entrances costs.
  Applied on read, so old cache rows are fixed without invalidation. **The provenance half is not
  done**: `orchestrate` still swallows failures into `heuristic_orchestrate` and the caller still
  writes the result as `model = 'orchestrator'`, so two of the five cached probes are heuristic
  word-splits wearing that label
- ~~**#20** llama.cpp runs on 4 threads regardless of the machine~~ — **done, and the first latency
  ticket that actually pays.** All three wrappers inherited `GGML_DEFAULT_N_THREADS = 4` — a
  constant, on every machine, carrying llama.cpp's own `// TODO: better default`. `models.n_threads`
  now defaults to **physical cores**: query 12.32 s → 8.46 s (1.46×), reindex 213 s → 171 s (1.26×),
  with the probes byte-identical and every rebuilt embedding bit-identical.
  **The obvious default would have shipped a regression.** `available_parallelism()` reports 16 on
  this 8-core/16-thread box, and 16 threads measures **15 s a query — slower than the 4 it
  replaces**, with the spread widening from ±0.4 s to ±3 s. The curve peaks at 12 (7.35 s) and falls
  off a cliff by 14. The tempting explanation, "leave the OS headroom", is wrong and was tested:
  pinned to eight *distinct* cores so the box looks non-SMT, 8 threads on 8 visible CPUs is the
  fastest setting and degrades not at all. Threads equal to cores is fine; threads equal to siblings
  is not. So the default counts cores, the way llama.cpp's own CLI does
  (`cpu_get_num_physical_cores`) — only its *library* default is the flat 4. 12 is faster still here
  and the default does not take it: 1.5× physical is a number this box likes, not a rule, which is
  what the config key is for.
  Two things fell out. Every latency number recorded before this was taken at 4 threads — #13's
  210 s reindex and the ~11.9 s query both reproduced exactly at 4, which is independent confirmation
  of the diagnosis. And **`--rebuild` does not produce a reproducible database**: `file_id` is a rowid
  handed out in vault-walk order, so two rebuilds at the *same* thread count disagree on any digest
  keyed by it. Pre-existing and unrelated, but it briefly looked like corruption — index comparisons
  must key on `path`
- **#19** intent classification looks inverted on two probes — with the orchestrator running,
  `dragon that can take human form` classifies `Exact` and the bare noun `Archdragon` classifies
  `Conceptual`. Both then show zero FTS contributions in their top 20 regardless. Picked up from
  #9's `--explain` audit; probe 4's intelligence-on regression may live here rather than in the
  reranker. The sharper fact is that an `fts 1.5` lane contributes nothing to a 20-slot ranking,
  which should be explained before the classifier is touched — likely #18
- **#18** query expansion splits on words against a 16-item stopword list containing no modals and
  no verbs, so `dragon that can take human form` becomes seven expansions including `that` and
  `can` — ~840 result slots for one question, and `collapse_lane` then pools BM25 scores from
  different queries and sorts them as if commensurable
- **#17** resolve queries against a tag/alias registry and expand on what they name. Aliases are
  parsed at index time and **thrown away** — there is no registry to resolve against — and tags reach
  retrieval only as a yes/no admission test on graph expansion. Hangs on the expansion slot rather
  than a new lane because every expansion runs through *both* content lanes, and in RRF a second
  lane's vote outweighs any position within one. Tags resolve semantically (a `tag_centroids` table,
  the same machinery as `folder_centroids`), aliases lexically (`links.rs`'s fuzzy name matching).
  Replaces the withdrawn FTS-injection experiment. Probe 4 is the guard, probe 5 the control
- **#21** multi-sequence decode — phase 2 of #13, which built the batch *API* and then looped inside
  it. `LlamaBatch::new(max_tokens + 16, 1)` sets `n_seq_max = 1` and `n_ctx` is sized for one
  document, so all 30 candidates get their own forward pass. Packing them under distinct `seq_id`s
  and decoding once is where quantized CPU inference gets its throughput. **Acceptance is bit-identical
  scores** — sequence isolation stops being enforced by `clear_kv_cache()` and starts being enforced
  by getting batch construction right, and a silent failure would corrupt every ranking. **Baseline is
  now #20's**: a query is 8.46 s, not 12.3 s, and the reranker's share of it shrank with everything
  else
- ~~**#16** `rerank_candidates` is hardcoded at 30~~ — **superseded, and its premise was wrong.**
  Candidate count is pinned at 30 on *both sides* of a change that doubled rerank time (#22), because
  cost tracks candidate **text**: 12,075 → 29,431 chars, 8,433 → 16,787 ms. Fitting the two points
  gives **87.4 ms per candidate + 0.481 ms per character**, so the count is worth 2.6 s of a 16.8 s
  rerank and the text is worth the other 14.2 s. A budget in candidates cannot bound this — 30
  candidates is anywhere between ~4 s and ~17 s depending which 30. Split into **#25** (the character
  budget, shippable now) and **#24** (the rest)
- ~~**#15** the reranker is fused as a lane~~ — **superseded by #24**, because it cannot be done
  alone: sorting by cross-encoder score while candidates are still selected by fused rank means the
  model sorts a shortlist RRF already decided, which is the constraint #15's own body records.
  Original reasoning kept — sorting on its score
  would give engraph its first **absolute** confidence (today `rrf_score / max_score * 100`, so the
  top hit is always 100%) and is likely the cheapest route into #4. Also amplifies the exact-name
  regression intelligence already causes, so probe 4 is the guard. **Unblocked**: #14 landed, so the
  score it would sort on is now a judgement of the actual chunk. Note that #14 raised the stakes —
  a score that only feeds `weight/(60+rank)` is a cheap thing to be wrong about, and one that
  orders the results is not, while the lane producing it now costs half the query
- **#24 (the ranking-stage redesign, Magnus's design)** — per-lane quota seeding, cross-encoder
  sorts. Semantic and FTS each hand graph *their* top N; graph resolves the union to distinct files
  and expands from those; **rank crosses the lane boundary, never score**; graph becomes a candidate
  source rather than a fusion lane; the cross-encoder sorts instead of voting. Two caps split by job
  — bound one document's share of the **shortlist** (the model cannot rank what it never saw), do
  **not** bound its share of the **results** (if a document holds the ten best sections, ten sections
  is the right answer, and #6's vote-counting reason for capping evaporates once there are no votes).
  Widen retrieval freely: it is 30–50 ms, and `sqlite-vec` is brute-force KNN so `k` is nearly free
- **#25 (extracted from #16)** — budget the cross-encoder in **characters**. Applied to the document
  text before formatting, never to the tokenized input: `rerank_batch` reads the score from the
  *last* token position, which is the assistant scaffolding after the document, so truncating that
  sequence's tail removes the thing being read. Also bounds a third cost — `n_ctx` and
  `LlamaBatch::new` are sized to the **longest** pair, so one 2195-char outlier inflates the
  allocation for all 30. Ships off (`0` = unlimited), then sweep. Honest tension: this is the lever
  **#14** removed, but in a different regime — #14's effective cap truncated 79.7% of chunks, where
  1200/1600 touch 15.3%/7.7% of this vault
- ~~**#6** section-level retrieval granularity — fuse on `(file_id, seq)` so a document can
  contribute more than one section~~ — **done**, probe 3 now returns the correct section and every
  result names its heading. It did not fix probe 1, but #9 has: nothing about the section-per-file
  transform is still unexplained, so **`eval/section-split.py` can be retired and the Path A /
  Path B question is closed in favour of in-place chunking.**
- ~~**#7** exclude derived `*-index.md` / `templates/` from ingest, and make `exclude` glob for
  real~~ — **done**, 14.2% of chunks and 18.3% of edges removed from the eval corpus

**THE SEED MERGE COMPARES SCORES FROM DIFFERENT SCALES (2026-08-07, found while measuring #22; no
ticket of its own — it is defect 2 of #24).** `merge_seeds` keys on file path and keeps the highest
score per file, but the lanes fill that field from incommensurable units: semantic writes
`1.0 - distance`, FTS writes negated BM25. Measured, the ranges **do not overlap** — semantic
0.206–0.686 across four queries, FTS 2.162–16.961. So **every FTS-seeded file outranks every
semantic-seeded file, always**: a total ordering by lane, not a ranking by relevance, with the top
ten seeds 10-from-FTS on 3 of 3 real queries. It propagates, because `graph_expand` computes
`expansion_score = seed.score * decay` and then truncates at 20 — BM25 magnitudes decide which
expansions reach fusion at all. The graph lane is documented as expanding from the best of both
lanes; it expands almost exclusively from keyword hits, which makes it redundant with the FTS lane
rather than complementary, and leaves it contributing least where the semantic lane is the only
thing working. **Dormant until #22** — the FTS lane previously returned zero rows for any multi-word
query, so seeds were purely semantic and all on one scale. Fixing the keyword lane woke it up, and
it should have been in #22's measurement.

**STAGE TIMINGS (2026-08-07, three queries, steady state) — the cross-encoder is 85–96% of a
query.** semantic 28–36 ms (including the embedding), FTS 4–17 ms, graph 536–1548 ms, **rerank
9,490–16,140 ms**; totals 11.1 / 11.9 / 16.9 s. Three consequences: running the content lanes
concurrently would save **~15 ms** and is not worth doing (and would contend for the same cores
#20 already saturates); **graph is the second cost at 5–14%**, which is worth weighing against #9's
finding that at weight 0.8 its main contribution is to probe 5, the nonsense control; and **#21 is
live again** — it attacks the 87 ms/candidate term, worth 2.6 s now the reranker is 16 s rather
than the 4 s it was after #20. Graph is also **not a peer of the other two lanes**: it consumes
`merge_seeds(semantic, fts)`, so it is a second stage by data dependency and could never be in the
same gather.

Suggested order: **#10's query side first.** It needs no reindex, so it can be A/B'd in minutes
against the eval homes that already exist, where everything else here costs a rebuild per column.
Its document side is the same experiment at a reindex per configuration, which is the only reason to
do it second.

### Correction: the probes work, and #3 is smaller than these notes had it

Earlier revisions of this file said, in several places, that the five probes "cannot measure" a
retrieval change and that everything waited on #3. **That was wrong, and it came from one bad
inference.** #12 changed the vector space outright, the probes reported identical verdicts at
identical ranks, and 76 of 100 slots moved underneath — from which these notes concluded the
instrument was blind. The better reading was available at the time: **the top of the ranking is
robust to that change and the churn was in the tail nobody reads.** That is a finding about the
retrieval stack, not a defect in the probes.

#14 settled it. The probes moved — probe 3 to rank 1, probe 2 off its correct section —
deterministically, reproducing #13's recorded hash exactly on the before side. A null result in #12
and a live one in #14 from the same five queries is an instrument working, not a broken one.

What follows is that **#3 is a sampling upgrade, not a capability upgrade.** If the shape
*query → expected document → is it in the top N* resolves anything at n=5, then n=40 is the same
instrument with better statistics. So #3 is not a battery to be built before work resumes; it is
**more probes, sourced from real usage** — session transcripts, `## Coverage` headers, logged
`/research` calls — which is hours, not a project. The three things it adds beyond sample count are
cheap and separable: negative controls (probe 5 already is one, and has never passed in any config),
scoring section accuracy apart from file accuracy (#6 made that observable and it just needs
recording), and sourcing discipline, since invented queries reproduce the corpus's own phrasing and
flatter the results.

**Nothing is blocked on it.** The honest limit of five probes is narrower than "unmeasurable": they
catch regressions and resolve large effects — they caught #2's probe-4 collapse, #11's @5→@1, #14's
two moves — and they do **not** resolve a one-up-one-down at n=5, which is exactly #14's result and
the reason that one is recorded as ambiguous. The fix for an ambiguous result is more queries on the
ambiguous question, not a hold on the queue.

**#13 and #14 are both done and neither needed a battery.** #13 adjudicated on byte-identical
output. #14 established a certain cost (+50% query latency, non-overlapping) against a probe result
that splits — and shipped anyway, because it replaces an incoherence rather than trading one strategy
for another. **#15 does not inherit a hold from that.** It is a strategy trade, so it needs a
measurement rather than an argument, and the probes can give it one: probe 4 is the named guard for
the exact-name regression pure rerank-ordering would amplify, and a blend measured against pure
ordering is a two-column A/B on the homes that already exist.

Then **#5** (makes chunk size and dim configurable, which comparing configurations needs), then #4 —
which #11 made more pressing, not less: the FTS lane now finds genuine keyword matches for nonsense
queries where it used to find none by accident.

**Withdrawn:** putting per-file identity (name, aliases, tags, path segments) into the FTS index,
described here for several revisions as the natural follow-on from #11. It is the wrong shape, and
the reason is the RRF arithmetic. Injecting terms changes BM25 scores, which moves positions *inside*
the FTS lane — and a lane's whole 60-deep spread is worth 2× (`weight/(60+rank)`), while appearing in
a *second* lane adds a whole new term, so two lanes at rank 20 beat one lane at rank 1. It would cost
a reindex to shuffle the one dimension that barely counts, and it reintroduces #2's failure mode in
the lexical lane besides: a term on every chunk of a document is a per-file constant, and BM25 length
normalisation would then favour that document's *shortest* chunk, which is exactly probe 4's shape.
**#17 replaces it** — resolve the query against a tag/alias registry and expand on what it names.
Query-conditional, no per-file constant, and it reaches both lanes through the expansion slot that
already exists.

**Probe 1 is answered (#9), and the answer was never the semantic lane.** Its expected notes sat at
ranks 21+ through #6, #2, #11 and #12 — four investigations, each reporting "no effect" and each
read as eliminating a hypothesis. They were eliminating the same one twice removed: every one of
them tried to make the **semantic lane better**, and probe 1's top 20 contains no semantic result at
all. It is twenty single-vote graph expansions.

The query starts with "who", which `llm.rs:922` classifies `Relationship`, which sets graph to 1.5
against the content lanes' 0.8. Graph is capped at 20 expansions, and `weight/(60+rank)` makes its
*worst* result (1.5/80 = 0.0188) beat the semantic lane's *best* (0.8/61 = 0.0131) by 43%. Every
slot is taken before the semantic lane is consulted. Two-lane agreement is the only way through
(0.8/61 × 2 = 0.0262) and probe 1 is the zero-lexical-overlap probe by construction, so agreement is
impossible. **The probe built to isolate the semantic lane is the one query shape where the semantic
lane cannot reach the results.**

One word demonstrated it. Dropping `who` — same binary, same index — moved
`temple-of-the-architect.md` from **absent from the top 20** to **rank 1 at 100%**.

The general defect is that `graph.rs:78` skips any neighbour already in the seed set, so graph
results are disjoint from the content lanes *by construction*. RRF fuses alternative rankings of the
same corpus; a disjoint set can never accumulate agreement, so its score is a pure function of the
weight constant — invisible at 1.0, total at 1.5. Fusing a recall step as though it were a ranking
is a category error, and #15's pool removes it by construction.

**The weight is now 0.8 and the probe lands** — `temple-of-the-architect` at ranks 1, 2 and 5,
`archivist-lenne` at 8, with `who` present. The demonstration has reversed: keeping the word now
beats dropping it, because `who` is the only token marking this as a question about a person and the
pipeline can finally use it. Two corrections to the paragraphs above came out of that audit. The
gate was firing with intelligence **off** as well — `default_no_intelligence()` is dead code, and
turning intelligence off only sets `orchestrator: None`, leaving `heuristic_orchestrate` and the same
weight table in charge — so this was the baseline configuration's behaviour, not a
models-loaded-only bug. And graph's contribution at 0.8 is not quite "invisible": it is two tail
slots on probe 1 and two on probe 5, *the nonsense control*, which is what a disjoint set does when
the content lanes have nothing to say. Full audit in `eval/probes.md`.

The section-per-file transform's win is no longer mysterious either: its sections are their own
files, dense enough in "temple" for **both** content lanes to find them, which is the one
configuration that clears the graph block. Now confirmed from the other direction — with the weight
corrected, in-place chunking returns the same file at the same ranks — so `eval/section-split.py`
has no open question left and can be retired.

`eval/` holds the probes and the harnesses: `probe.sh` for what came back, `bench-search.sh` for how
long it took.
