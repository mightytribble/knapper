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
cargo test --lib             # 546 pass
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
Upstream PR #47 addresses them. Use `cargo test --lib` (546 tests) as the working suite.
`cargo clippy -- -D warnings`, which is what CI runs, is clean.

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
  retrieval acceptance criteria turned out to have been met already by #1 + #6. Needs #3 to decide
  whether a length-scaled or conditional prefix is worth having.
- ~~**#11** FTS indexes the 200-char snippet, not the chunk~~ — **done.** Keyword search had been
  reaching 27.6% of the corpus; `Saltmere` and four other verified-present terms returned zero hits.
  Probe 3 went @5 → **@1**, the best result any configuration has produced on it, and probe 4 held
  at @2 under the heaviest churn of any probe. Nothing regressed. Moved 73 of 100 probe slots, so
  every lexical number recorded before it is superseded.
- **#3** retrieval eval battery — now adjudicates #2, and calibrates #4
- **#4** relevance floor — configurable per-lane min scores so nonsense queries return nothing
- **#5** embedding model config — expose output dim, tie max chunk tokens to the model's context window
- **#8** pick a better local embedder — >512 tokens, >768 dim (pairs with #5, which exposes the knobs)
- ~~**#12** embed at the model's native dimension~~ — **done.** Every vector had been truncated to its
  first 256 of 768. The seed probes return identical verdicts at identical ranks and confidences,
  while **76 of 100 slots moved underneath** — five hand-picked probes cannot measure a change to the
  vector space, which is #3's job. Ruled out as the probe 1 explanation (#9). The migration ran
  itself; storage roughly doubles. Optional Matryoshka truncation is deliberately left unbuilt
- **#10** the embedding prompt format is nomic-embed-text's, not EmbeddingGemma's — both query and
  document sides are out-of-distribution. Query-side fix needs **no reindex** and is the cheapest
  open experiment in the repo; document-side needs one per configuration, so it waits for #3
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
  18 of 20 slots moved on the benchmark query, which is #12's churn signature — so #3 adjudicates.
  Shipped on and unswitchable, because "200 chars, or a 64-token match window if FTS found it" is not
  an alternative strategy worth preserving. `[rerank] document_title` is a switch and is off
- **#15** the reranker is fused as a lane instead of ordering the results — sorting on its score
  would give engraph its first **absolute** confidence (today `rrf_score / max_score * 100`, so the
  top hit is always 100%) and is likely the cheapest route into #4. Also amplifies the exact-name
  regression intelligence already causes, so probe 4 is the guard. **Unblocked**: #14 landed, so the
  score it would sort on is now a judgement of the actual chunk. Note that #14 raised the stakes —
  a score that only feeds `weight/(60+rank)` is a cheap thing to be wrong about, and one that
  orders the results is not, while the lane producing it now costs half the query
- **#9** section-per-file still beats in-place on probe 1 — #6 ruled out granularity, #2 ruled out
  document identity in the vector, #11 ruled out lexical coverage, #12 ruled out embedding
  dimensionality. Ranks 21+ on that probe are set by graph expansion, and the confidence ladder there
  has survived four changes to the lanes underneath it
- ~~**#6** section-level retrieval granularity — fuse on `(file_id, seq)` so a document can
  contribute more than one section~~ — **done**, probe 3 now returns the correct section and every
  result names its heading. Did **not** fix probe 1, so `eval/section-split.py` cannot be retired
  and the Path A / Path B question stays open.
- ~~**#7** exclude derived `*-index.md` / `templates/` from ingest, and make `exclude` glob for
  real~~ — **done**, 14.2% of chunks and 18.3% of edges removed from the eval corpus

Suggested order: **#10's query side, then #3.** The query-side prompt fix jumps the queue only
because it needs no reindex — it can be A/B'd in minutes against the existing eval homes, where
everything else here costs a rebuild per column. Its document side goes after #3 with the rest.

#12 sharpened the argument for putting #3 next. It changed the vector space outright and the five
probes reported *identical* verdicts at identical ranks and confidences while three quarters of the
result slots moved. Any further work on the semantic lane — #10's document side, #8's model swap,
#5's knobs — is unmeasurable until the battery exists.

**#13 was the exception and is done.** It needed no battery to adjudicate — same index, same
ranking, byte-identical output — which is also how it managed to disprove its own premise without
waiting for #3.

**#14 is done and is the argument for #3 stated as plainly as it can be.** It is the first change
in this fork with a certain cost (+50% query latency, measured, non-overlapping) and an unmeasurable
benefit: on the five probes it wins one and loses one, and 18 of 20 slots moved on the benchmark
query. It shipped anyway because it fixes an incoherence rather than trading one strategy for
another — but that reasoning does not transfer to #15, which *is* a strategy trade. **#15 should
wait for #3.** It is worth keeping in view while scoping #4: a cross-encoder probability
is the calibrated score a relevance floor wants, and RRF scores demonstrably are not (the nonsense
query's top RRF score already exceeds a legitimate third-place result).

Then **#3**, and now with a concrete debt to pay off. Five hand-picked probes were
enough to show that #2 trades one probe for another, and not enough to say which trade is right —
the same five gave contradictory verdicts on three configurations of the same feature. #11 then
moved 73 of the 100 slots those verdicts were read from. #3 supplies the measurements everything
else is judged by. Then **#5** (makes chunk size and dim configurable, which #3 needs to compare
configurations), then #4 — which #11 made more pressing, not less: the FTS lane now finds genuine
keyword matches for nonsense queries where it used to find none by accident.

**One experiment #3 should own:** putting per-file identity (name, aliases, tags, path segments)
into the FTS index. It is the natural follow-on from #11 and was deliberately excluded from it,
because it reintroduces #2's failure mode in the lexical lane — a term on every chunk of a document
is a per-file constant, and BM25 length normalisation would favour that document's *shortest* chunk.
Probe 4 is exactly that shape and is the probe that would catch it.

**Probe 1 is the open question #6 was expected to close** (now #9). The section-per-file transform
puts `temple-of-the-architect` at ranks 1–4; indexing in place leaves it at 21 before #6, after #6,
under all three prefix configurations of #2, and after #11 — where the ranks are identical to the
rank (temple 21/22/23, `archivist-lenne` 25/28/32). Three explanations are now ruled out: the
ranking unit (#6), document identity in the vector (#2), and lexical coverage (#11). What is left of
the transform's difference is
that each section became a whole document — its own embedding computed over that text alone, its own
docid, its own graph node. It needs #3 to diagnose against more than one query.

`eval/` holds the seed material for #3.
