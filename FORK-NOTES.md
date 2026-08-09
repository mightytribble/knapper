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
| `rerank::max_document_chars` | the cross-encoder's input is bounded; upstream has no budget at all | this fork, issue #25 |
| re-index keeps the `files` row | editing a note no longer cascades away every backlink into it | this fork, issue #27 |
| normalised seed pool | graph seeds are ranked by relevance, not by which lane's unit is bigger | this fork, issue #26 |
| chunk-to-chunk `edges` | a link knows which passage wrote it; expansion follows the seed's own links | this fork, issue #28 |
| `DOC_LEVEL` sentinel | an un-headed link stores one row, not one per target chunk | this fork, issue #28 |
| `PprParams` | the graph lane is personalized PageRank over chunks: sum, not max | this fork, issue #29 |
| `incident_wikilink_edges` | one indexed fetch per frontier, not a BFS per seed | this fork, issue #29 |
| `src/fingerprint.rs` | the store knows what built it, and rebuilds only what changed | this fork, issue #31 |
| `src/ranking.rs` | the cross-encoder sorts the shortlist; the graph reaches it by reserved quota | this fork, issue #30 |
| `format_reranker_input` | the cross-encoder is asked the question its model card documents | this fork, issue #32 |
| `cuda` cargo feature | llama.cpp compiles its CUDA backend; off by default | this fork, issue #33 |
| `llm::device_identity` | the compute device is read at load and fingerprinted | this fork, issue #33 |
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
cargo test --lib             # 642 pass
```

Each env var exists for a specific failure. Omit one and you get:

| omitted | failure |
|---|---|
| `CMAKE_POLICY_VERSION_MINIMUM` | cmake 4.x rejects llama.cpp's older `cmake_minimum_required` |
| `LIBCLANG_PATH` | `Unable to find libclang` — bindgen can't run |
| `BINDGEN_EXTRA_CLANG_ARGS` | `fatal error: 'stdbool.h' file not found` (pip libclang ships the .so, not clang's builtin headers) |

Adjust the gcc version in the include path (`13`) and the python version (`python3.12`) to match the box.

### The CUDA build (issue #33)

**Also no sudo.** The `cuda` feature is out of `default`, so the build above is unchanged and CI's
macOS and Ubuntu legs — which have no toolkit — are unaffected. `cargo clippy -- -D warnings` passes
with and without it.

The toolkit goes in `$HOME` because the runfile installer takes a `--toolkitpath`. **Install the
toolkit only**: the `--driver` component is a Linux display driver and installing it breaks the WSL
GPU passthrough that makes any of this work.

```bash
# 1. Toolkit, user-local (~4.4 GB download, ~7 GB installed)
curl -fLO https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
env -u DISPLAY bash cuda_12.6.3_560.35.05_linux.run --nox11 --silent --toolkit \
    --toolkitpath="$HOME/.engraph-cuda" --no-man-page --override --tmpdir=/some/large/tmp

# 2. Build (the four vars above still apply)
export PATH="$HOME/.engraph-cuda/bin:$PATH"
export CUDAToolkit_ROOT="$HOME/.engraph-cuda"   # find_package(CUDAToolkit)
export CUDA_LIBRARY_PATH="$HOME/.engraph-cuda"  # the linker search path — see below
export CUDAARCHS=89                             # Ada; ggml's default is `native`, same answer here

cargo build --release --features cuda
```

Two of these are not guessable from the ticket, and each is a silent build failure:

| trap | what happens |
|---|---|
| `CUDA_LIBRARY_PATH` vs `CUDA_PATH` | `find_cuda_helper::find_cuda_lib_dirs` reads **only** `CUDA_LIBRARY_PATH` on Linux (`CUDA_PATH` is the Windows path), then joins `lib64` onto each entry — so it wants the toolkit **root**, not `lib64`. Set `CUDA_PATH` instead and the link fails on `cudart_static` |
| `--nox11` | the makeself wrapper sees `$DISPLAY` (WSLg sets it) with no tty and tries to `exec xterm`, failing with `exec: -title: not found` before the installer runs at all |
| `--log-file`, `--defaultroot` | not options in the 12.6 installer. It exits `Unknown option:` and installs nothing |
| `sh` instead of `bash` | the runfile is a bash script; dash dies at line 461 |

`llama-cpp-sys-2` links CUDA **statically** on Linux (`cudart_static`, `cublas_static`,
`cublasLt_static`, `culibos`), which is why the PyPI `nvidia-*-cu12` wheels are not a shortcut — they
ship the shared libraries. The runfile toolkit has all four. `-lcuda` resolves against
`lib64/stubs/libcuda.so` at link time and the real driver at run time, from
`/usr/lib/wsl/lib/libcuda.so.1`, which WSL already puts on the loader path.

The CUDA binary is **701 MB** against 25 MB for the CPU one — statically linked kernels. Keep the
two in separate target directories (`CARGO_TARGET_DIR`) if you want both, because a feature change
relinks the same path and a rebuild each way costs the llama.cpp compile.

### Known pre-existing test failures

`cargo test` (full) fails to compile `tests/integration.rs` and `tests/write_pipeline.rs`:
`unresolved import engraph::embedder`, `engraph::hnsw`, and a `walk_vault` arity mismatch.
**These are broken on pristine upstream** — verify with `git stash && cargo clippy --all-targets`.
Upstream PR #47 addresses them. Use `cargo test --lib` (642 tests) as the working suite.
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
- **The eval store lives at `standalone/mcp-isekai/.engraph-eval/`** and the corpus it indexes is
  not frozen. Every measurement from #26 to #29 was taken on a 247-file / 1598-chunk store; the
  pinned vault at `63f33e6` is 266 / 1863 / 1988 edges, because `standalone/mcp-isekai` tracks the
  live `cc-isekai` repo and it grew in between. Rank tables from either side of that are not
  comparable. Earlier stores lived in session scratchpads and expired with the sessions.
- **`top_n` is part of the measurement, not a display setting.** Both content lanes retrieve
  `top_n * 3` per expansion, so a probe table taken at 5 and one taken at 20 are different
  experiments — probe 2's tracked answer is absent at 5 and rank 1 at 20. `eval/probes.md`'s tables
  are at 20. The orchestration cache (`llm_cache`, keyed on the query) is what holds expansions
  constant across variants, so reusing one warm store is the control rather than a shortcut.
- **Backlinks used to rot on every save** (#27, fixed). Any store last written by a pre-#27 build has
  edges missing — 24 of 1084 per three files edited, on the isekai vault — and the fix does not
  reconstruct them. `engraph index --rebuild` is the one-time repair, and any graph-lane number taken
  before that repair is measuring a degraded graph.
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
  `--reindex`; changing the *query* side needs nothing. Since #31 the reindex is no longer something
  to remember: bump `fingerprint::PROMPT_TEMPLATE_VERSION` and the next `engraph index` does it.
- **The cross-encoder sorts the results; it used to vote on them** (this fork, issue #30). Two
  content lanes are fused by RRF, graph and temporal candidates are routed into the shortlist by
  **reserved quota** (`[ranking] candidates = 30`, `graph_reserve = 8`, `temporal_reserve = 4`), and
  the model's probability is the order. Nothing reblends; `tiebreak` decides exact ties only.
  `[ranking] mode = "legacy"` restores the five-lane fusion and reproduces pre-#30 output byte for
  byte — it is the control the change was measured against, and it stays for that.
  - **The shortlist is capped and the results are not.** Bound what the model is shown, because it
    cannot rank what it never saw; do not bound what it returns, because ranking is its job. So one
    document can own the top of the answer, which under the legacy stage `max_chunks_per_file` cut
    at three. `group_by = "file"` still collapses to one result per document.
  - **A build with no reranker configured keeps the legacy stage.** The documented 3:1 interleave is
    the fallback for a model that *should* be there, and it is what a failed `rerank_batch` takes —
    labelled `(degraded ordering)` on stdout, because stderr is where a warning goes to be
    discarded. With `intelligence = false` the interleave measured two tracked targets down and one
    up against the fusion, so a deliberate configuration is not treated as a degradation.
  - **The graph reserve is inert on this corpus and ships anyway.** 20 graph candidates generated
    per query, 8–19 admitted, **none reaching the output** — the model scores every graph-only
    candidate below every content candidate, and `graph_reserve = 0` is byte-identical on all five
    probes. What it buys is that those are now three numbers in a log line rather than an inference
    from a ranking, which is the state #9 could not get out of.
  - **Confidence means two different things and one of them is not a probability.** Under sorting it
    is the cross-encoder's own score — probe 5, the nonsense control, reports 0% where it used to
    report 100%. Under the degraded interleave it is a **position**, since nothing calibrated that
    order. Layer 2 of the convergence plan replaces the percentage with provenance and a status.
- **The reranker was being asked a question its model card does not document** (this fork, issue
  #32). Qwen3-Reranker-0.6B specifies a fixed system prompt, an `<Instruct>/<Query>/<Document>`
  body, an **empty `<think></think>` block** before the answer, and lowercase `yes`/`no` as the
  scored tokens; `format_reranker_input` matched none of them, so the logits being read came from a
  distribution that was never about yes or no. Same family as #10. **It survived because the score
  was a vote**: correcting it leaves the legacy stage's top four unchanged on every probe. Under
  #30 it is the difference between probe 3's answer at rank 16 and at rank 1, and between a nonsense
  query scoring 8% and scoring 0%. `[rerank] document_title` ships on for the same reason — a
  section that reads `## Evolution / - Previous: Medium Dragon` never names the document it is from.
- **The store records what built it, and rebuilds only what changed** (this fork, issue #31). Six
  keys in `meta` — `parser_`, `chunker_`, `link_`, `fts_`, `embedding_`, `reranker_fingerprint` —
  compared at the top of every index and at the top of every read. Three costs are wildly different
  and that is the point: on the 266-file isekai vault a chunker or embedding change is a **183 s**
  full reindex, a link-resolver change re-derives all 1988 edges from the vault in **3.1 s**, and an
  FTS schema change rebuilds all 1863 keyword rows from `chunks.text` in **0.1 s** with no vault read
  at all. `reranker_fingerprint` triggers nothing — it exists to invalidate a calibrated threshold,
  and a read that disagrees with it warns rather than blocking.
  - **A read path refuses rather than warns.** `engraph search` and `engraph serve` fail with
    "Run 'engraph index'" on any rebuild-class mismatch, the way `verify_embedding_dim` already does
    on a width change. Warning to stderr would be useless here: `eval/probe.sh` sends stderr to
    `/dev/null`, and a stale index answering a probe is exactly the silent wrong number this exists
    to stop.
  - **A store with no fingerprints is adopted, not rebuilt.** Every pre-#31 database is in that
    state. There is no evidence it is stale and a forced one-time reindex for everyone is the same
    uselessness as rebuilding on every startup. Protection starts from the first recorded value.
  - **`reranker_fingerprint` is written by a *search*, never by an index** — the index path loads no
    cross-encoder, so it cannot honestly claim one is current. A *mismatched* key is never
    overwritten from a read either: doing so would consume the signal a calibrated threshold needs.
  - **Model identity is the artifact's SHA-256, not its filename.** Hashing 640 MB costs ~0.3 s warm
    and ~8 s cold, so it is cached in a `<model>.sha256` sidecar keyed on `(size, mtime)`. The cache
    is keyed on metadata, the fingerprint on content — so re-downloading identical bytes rehashes and
    rebuilds nothing.
  - **Three algorithm versions are hand-maintained** (`PARSER_VERSION`, `CHUNKER_VERSION`,
    `LINK_RESOLVER_VERSION`, plus `PROMPT_TEMPLATE_VERSION` and
    `EMBEDDING_NORMALIZATION_VERSION` in `llm`). There is no runtime view of what a function does.
    Everything that is *data* — chunk limits, the FTS schema text, config, artifact digests — is
    hashed exactly and needs no bump.
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
- **The GPU is a build-time choice and a runtime reading** (this fork, issue #33). `--features cuda`
  compiles the backend; nothing in `llm.rs` asks for offload, because all three model loads already
  pass `LlamaModelParams::default()` and its `n_gpu_layers` is -1. What device the process actually
  got is read at load by `llm::device_identity` and folded into both model fingerprints, so
  **swapping between a CPU and a CUDA binary forces a re-embed each way.** On a derived store that
  is a wait, not a loss.
- **A hidden or unavailable GPU falls back to CPU silently, and the fingerprint is what catches it.**
  `CUDA_VISIBLE_DEVICES=""` against a GPU-built store loads `device=cpu` with no error and then the
  read path refuses on `embedding_fingerprint`. One binary can therefore produce either device's
  vectors in one session, which is why the component is a runtime value and not
  `cfg!(feature = "cuda")`.
- **VRAM is shared with the Windows session.** 16376 MiB total, ~3.3 GB in use by Windows while
  measuring. `engraph serve` adds **1768 MiB** with all three models resident; `engraph index` adds
  ~570 MiB, because indexing loads the embedder alone. Genuine exhaustion-at-load was **not** induced
  here, so llama.cpp's behaviour in that case is untested on this box — only the device-absent path
  above is known, and it is clean.
- **`engraph status` does not report the device.** The only place the resolved device is visible is
  the `loaded LlamaEmbed …` / `loaded LlamaRerank …` line at `RUST_LOG=engraph=info`.
- **GPU numbers and CPU numbers are different baselines.** `eval/probes.md`'s CUDA section is a fresh
  baseline for exactly this reason: the kernels are not bitwise identical, the embeddings differ in
  the low bits, and the retrieved candidate set differs with them. Comparing a GPU rank table with a
  CPU one measures the backend, not the change under test.

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
  cost tracks candidate **text**: 12,075 → 29,431 chars, 8,433 → 16,787 ms with the count pinned at 30
  on both sides. A budget in candidates cannot bound this — 30 candidates is anywhere between ~4 s and
  ~17 s depending which 30. Split into **#25** (the text budget, shippable now) and **#24** (the rest)
- ~~**#15** the reranker is fused as a lane~~ — **superseded by #30**, because it cannot be done
  alone: sorting by cross-encoder score while candidates are still selected by fused rank means the
  model sorts a shortlist RRF already decided, which is the constraint #15's own body records.
  Original reasoning kept — sorting on its score
  would give engraph its first **absolute** confidence (today `rrf_score / max_score * 100`, so the
  top hit is always 100%) and is likely the cheapest route into #4. Also amplifies the exact-name
  regression intelligence already causes, so probe 4 is the guard. **Unblocked**: #14 landed, so the
  score it would sort on is now a judgement of the actual chunk. Note that #14 raised the stakes —
  a score that only feeds `weight/(60+rank)` is a cheap thing to be wrong about, and one that
  orders the results is not, while the lane producing it now costs half the query
- ~~**#24 (the ranking-stage redesign, Magnus's design)**~~ — **closed, replaced by #30.** Four
  tickets were extracted from it and shipped (#25, #26, #28, #29), and two of its own prescriptions
  were overruled on the way: #26 shipped per-expansion min-max normalisation rather than item 2's
  rank-crossing, and #29 deleted the disjointness skip item 3 argued for keeping. Its acceptance
  criterion that no score crosses a lane boundary is therefore false against `main`
- ~~**#31 — fingerprint the index**~~ — **DONE.** `src/fingerprint.rs`; `meta` carries
  `parser`/`chunker`/`link`/`fts`/`embedding`/`reranker_fingerprint`, each with one declared action,
  compared at the top of every index and every read. Before it, only embedding *dimension* was
  fingerprinted, so chunker constants, the prompt template, the tokenizer, a swapped GGUF behind an
  unchanged filename, the FTS schema and the resolver could all change under a store reporting itself
  healthy. Settled by invariant on the 266-file eval vault, all five actions poked one at a time:
  **chunker → 266 files re-indexed, 1863 chunks byte-identical keyed on path, 182 s; link → 1988
  edges wiped and re-derived identically in 3.1 s with nothing re-embedded; fts → 1863 keyword rows
  wiped and rebuilt in 0.1 s with no vault read; embedding → `search` refuses with "Run 'engraph
  index'"; reranker → `search` proceeds and the recorded value survives.** The negative half —
  poke nothing, and a hand-emptied FTS table stays empty — is the one that matters, and it holds.
  **The tests were checked against a mutant** whose `compare` always returns clean: nine fail, so
  they are evidence rather than decoration. De-risks #10, #5, #8 and #2, each of which changes the
  model input and previously relied on someone remembering to rebuild. Also pinned the eval corpus at
  **`63f33e6`** — see `eval/probes.md`; `standalone/mcp-isekai`'s `origin` is the live `cc-isekai`
  repo, so the pin is a checkout that stops tracking, not a fetchable tag.
  **Layer 0 of `docs/vault-search-convergence.md`**
- ~~**#30 — the ranking stage: the cross-encoder sorts, and graph reaches it by reserved quota**~~
  — **DONE.** `src/ranking.rs`. Graph stops being a fusion lane and becomes a candidate source
  keeping #29's PPR as its reach function; the shortlist is built by **reserved quota, backfilling
  either way**, and the cross-encoder's probability is the final order with nothing reblended.
  Two caps split by job — bound one document's share of the **shortlist**, not of the **results**;
  this conflicts with §9.1 of the vault-search design and #30 is the side shipped.
  **The control holds: `mode = "legacy"` is byte-identical to pre-#30 output on all five probes**,
  through every subsequent edit. The tests were checked against two mutants — a sort that does
  nothing fails three, a reserve that admits nothing fails six, including the end-to-end one.
  **The probes overruled three of the ticket's four prescriptions.** §8.6's 64-candidate pool
  doubles the assembled text for one rank *worse* on probe 2, so the pool stays at 30 and the change
  is **cost-neutral** (55.7 s → 56.8 s of cross-encoder across five probes). The "loose shortlist
  cap" moves nothing at 32. Pure ordering and the RRF tie-break are **byte-identical**, because a
  softmax collides only on exact equality. And the 16-of-64 reserve is inert here: 8–19 graph-only
  candidates enter the pool per query and **none survives the model** — `graph_reserve = 0` is
  byte-identical. It ships as instrumentation rather than as a ranking effect.
  **On rank the change loses three targets to one** (p1 temple 1 → 2 and lenne 2 → 5, p2 4, p3
  holds at 1, **p4 5 → 1**), and that is recorded rather than explained away. What it buys is the
  thing the rank table cannot show: **probe 5's confidence goes 100% → 0%** while every real probe's
  answer scores 98–100%, which is the absolute score layer 2's abstention floor has to exist
  against. Blocked on #32 — without it the same change puts probe 3's answer at rank 16.
  **Layer 1 of `docs/vault-search-convergence.md`**
- ~~**#32 — the cross-encoder is asked a question its model card does not document**~~ — **DONE.**
  Found while measuring #30, and it inverted that measurement. Qwen3-Reranker's documented input is
  a fixed system prompt, an `<Instruct>/<Query>/<Document>` body, an empty `<think></think>` block
  before the answer, and lowercase `yes`/`no`; we matched none of it. **The reason it survived is
  itself the sharpest measurement of #15 in the file**: as a voter, correcting the format changes
  the top four results of no probe at all — two materially different score distributions, one
  ranking. Under #30 it moves probe 3's answer 16 → 1, probe 1's 15 → 2, and the nonsense control's
  best score 8% → 0%. `[rerank] document_title` ships on with it (probe 2's answer, 8 → 4).
- ~~**#26 (defect 2 of #24, extracted)** — normalise each lane's scores before they leave the lane~~
  — **DONE.** Min-max into `[0.1, 1.0]` per `(lane × expansion)`, floor 0.1 because
  `seed.score * decay` feeds a sort that feeds `truncate`, `max == min` to the top of the range.
  **The ticket got the placement wrong and the probes caught it.** Normalising the pooled lanes
  before `collapse_lane` — what #26 specified — moved three of six probe targets down, including
  `archdragon.md` 3 → 6 on probe 4, the probe that exists to catch BM25 regressions. `score` has two
  consumers wanting opposite things: **fusion reads rank** and wants the lane's own ordering intact,
  **seeding reads magnitude** and wants the lanes commensurable. Per-expansion normalisation sets
  every expansion's best hit to exactly 1.0, so a weak expansion's best ties a strong one's and the
  pooled lane is reordered on no evidence — pure loss for the semantic lane, where `1.0 - distance`
  *is* comparable across expansions. **Shipped as a separate seed pool**: `all_*` keeps the lane's
  scores and feeds fusion, `*_seeds` is normalised and feeds `merge_seeds`. Seed composition moves
  (top-10 seeds from FTS: 10/10 → 8, 6, 8 on three of four real probes; probe 1 stays 10/10) and
  **no real probe's output changes at all** — probes 1–4 byte-identical, only the nonsense control
  moves. A correctness fix the pipeline was nearly insensitive to on the day, because #9 holds the
  graph lane at or below the content lanes; **#29 is what made it pay**, since `Σ seed_score × 1/L`
  would otherwise be summing incommensurable units. `merge_seeds` logs `seeds`/`fts_won`/`top10_fts` at
  DEBUG. Knock-on: `LaneWeights::from_intent` is meaningful for the first time
- ~~**#27** — **editing any file destroys every backlink into it**~~ — **DONE.** The ticket named
  `delete_edges_for_file`'s `from_file = ?1 OR to_file = ?1` as the cause. It was a symptom: **the
  real cause is that re-indexing deleted the `files` row**, and both `edges` columns are
  `REFERENCES files(id) ON DELETE CASCADE`. So the backlinks were gone before the explicit deletion
  ran, and pointing that deletion at `from_file` alone would have fixed nothing. Three call paths did
  it — `index_file` step 5, `run_index_inner` step 5 (`remove_file` on every *changed* file), and
  `writer::update_note`. The fix keeps the row: `delete_chunks_for_file` clears the content and
  `insert_file`'s upsert on `path` returns the same id, so every edge keyed on it survives.
  A second, opposite defect fell out of the same reading: on the incremental path `clear_edges()` is
  skipped and `insert_edge` is `INSERT OR IGNORE`, so nothing ever removed a wikilink the author had
  **deleted** — the graph could only grow. `delete_outgoing_edges_for_file` before each
  `build_edges_for_file` closes it. A third: `writer::move_note` deleted and re-inserted the row, so
  a move cascaded the backlinks away **and took the note's chunks with it** — a moved note stayed in
  the index and vanished from every search. It now uses `update_file_path`, which is the primitive
  that already existed for this. **Measured on the 266-file isekai vault: editing three files
  destroyed 24 of 1084 edges; after the fix, 1084 → 1084.** The pinned invariant is
  `an_incremental_edit_and_a_full_index_agree_on_the_edges_table`. **Was invisible to the probe
  harness by construction** — the eval vault is built by one full index, which calls `clear_edges()`,
  so every graph-lane measurement ever taken here was on a best-case graph
- ~~**#28** — **wikilink edges at chunk granularity, both ends**~~ — **DONE.** `edges` gained
  `from_chunk_seq` / `to_chunk_seq` and the unique key widened to the full chunk-to-chunk identity.
  1084 rows became 1697 and the derived `SELECT DISTINCT from_file, to_file` view stayed **identical**,
  which is the criterion that proves nothing was lost. Backfilled from `chunks.text` in **3.4 s with
  no vault read**, byte-identical to a 158.9 s `--rebuild`. **Departed from the ticket on one point:**
  an un-headed `[[Note]]` stores a `DOC_LEVEL` sentinel instead of one row per target chunk —
  materialising would have been 7,215 rows on the to-side alone and would make an edge's row count
  proportional to the target document's size, i.e. a 37-chunk note reading as 37× the endorsement of
  a 1-chunk one under #29's summation. Deep links still keep the ticket's set semantics.
  **Zero deep links in the vault, so that branch has unit tests and no data.** Retrieval scopes
  expansion to the seed's own chunk at `OFF_CHUNK_LINK_WEIGHT = 0.5`; every setting from 0.0 to 0.9
  gave the *same* output, because `graph_expand` took the max across 60–120 seeds and only 1 of 116
  expansions on probe 1 is reachable exclusively off-chunk. Probe 1 does not regress; the entire
  effect was one insertion at rank 13 of probe 4. **The plateau ended with #29** — summing makes the
  discount bite on both sides of the fraction, and 0.5 now beats hard scoping by two tracked targets
- ~~**#29** — **personalized PageRank replaces the graph lane's scoring**~~ — **DONE.** The old code
  was legible as a one-iteration PPR with the accumulation operator wrong: hop decay *is* damping,
  two hops *are* two iterations, seed scores *are* the restart distribution. `sum` not `max` (the
  co-citation signal), out-degree normalisation to stop the sum re-electing the hubs, and the walk
  now runs over **chunks**, so the lane returns the passage it reached instead of guessing one.
  Deleted: the decay table, the `max` merge, `get_best_chunk_for_file`, `get_shared_tags_files` and
  its unordered `LIMIT 100`, the disjointness skip, and **the whole admission filter**.
  **Three tracked targets up, one down, two hold — the best single move in `eval/probes.md` since
  #6.** Probe 2's exact-answer section goes 6 → 1; probe 1, the acceptance criterion, improves 5 → 2;
  probe 4 slips 3 → 4, one rank *inside* the top five, which is not the drop-out that probe guards.
  **The lane's own cost fell from 0.76–2.08 s to 0.83–1.64 ms** — three interleaved rounds, both
  binaries instrumented identically; the ticket's 536–1548 ms estimate was low because #23 and #26
  had since grown the seed pool. End-to-end wall clock could not resolve it and is not claimed.
  **Both normalisation exponents were swept and landed in opposite places**: full `1/L` on the source,
  softened `1/√N` on the target — the ticket specified `1/N` there and it costs two targets. Two
  iterations lose. And **#28's plateau is gone**: `OFF_CHUNK_LINK_WEIGHT` was unmeasurable under
  `max` and now has evidence, since hard scoping costs two targets. The probe 5 diagnostic came back
  the opposite way round from the ticket's prediction and reads worse for the old lane: graph's share
  of the nonsense control *rose*, because **the old lane's entire visible output across five probes
  was five results, four of them on probe 5** — the filter had suppressed it so completely that junk
  was nearly all that survived
- ~~**#25 (extracted from #16)** — budget the cross-encoder in **characters**~~ — **DONE.**
  `[rerank] max_document_chars`, applied to the document text before formatting and before any title
  is prepended, never to the tokenized input: `rerank_batch` reads the score from the *last* token
  position, which is the assistant scaffolding after the document, so truncating that sequence's tail
  removes the thing being read. Also bounds a third cost — `n_ctx` and `LlamaBatch::new` are sized to
  the **longest** pair, so one 2195-char outlier inflates the allocation for all 30. **Swept three
  rounds and shipped defaulting to 1000**, not off: 82% of the text kept, 12% of query latency back,
  two probe targets up and none down. 600 gives back 26% and costs probe 4 a rank. The honest tension
  with **#14** survives measurement — #14's effective cap truncated 79.7% of chunks where 1000 cuts
  18% of the text, and the two ranks #14 won both hold
- ~~**#6** section-level retrieval granularity — fuse on `(file_id, seq)` so a document can
  contribute more than one section~~ — **done**, probe 3 now returns the correct section and every
  result names its heading. It did not fix probe 1, but #9 has: nothing about the section-per-file
  transform is still unexplained, so **`eval/section-split.py` can be retired and the Path A /
  Path B question is closed in favour of in-place chunking.**
- ~~**#7** exclude derived `*-index.md` / `templates/` from ingest, and make `exclude` glob for
  real~~ — **done**, 14.2% of chunks and 18.3% of edges removed from the eval corpus

**THE RERANKER IS 99% ARITHMETIC — MEASURED, after a two-point solve told me otherwise (2026-08-07).**
Timing the stages inside `rerank_batch`, four queries at 30 candidates: **`ctx.decode()` is 99.0–99.3%
of the call.** Tokenizing all 30, creating the context, thirty `clear_kv_cache()` calls, batch
construction, logit extraction and the DB fetch total **96–121 ms for the entire candidate set** —
~3.5 ms each. Per-token cost is flat at 1.816–1.918 ms and a least-squares fit for a fixed
per-candidate term returns **−48 ms**, i.e. there is no such term:

```
rerank ≈ total_tokens × 1.867 ms        (chars/token measured 3.16–3.43, not the assumed 4)
```

**I had published `87.4 ms per candidate + 0.481 ms per char` in #24, #25 and here.** It came from
solving two equations for two unknowns — an exact solve has no residual and will hand back whatever
term it is asked for. Corrected in all three places. **General lesson: an exact solve is not a fit;
it cannot disagree with the data it was built from, so it is evidence of nothing.**

Consequence for **#21**: its premise is mostly gone. The per-candidate overhead it exists to amortise
is ~0.1 s across the batch, not the 2.6 s I claimed one message earlier. The one hypothesis left is
**GEMM efficiency** — one batch of 8,575 tokens versus thirty of ~286 — and that prior is weaker than
it looks, because the intuition that batching wins big comes from autoregressive *generation*
(memory-bandwidth-bound matrix-vector), while this is *prompt processing*, already matrix-matrix and
compute-bound with the cores saturated at `n_threads = 8`. **This is the second time this reasoning
has failed in this subsystem**: #13 amortised context creation and bought nothing because a context
costs 1–3 ms. Both assumed per-call overhead mattered in a workload that is ~99% arithmetic. Test it
as a standalone benchmark before building anything.

Confirmed end-to-end by the **#25 sweep** (three rounds, five caps): query time tracks rerank
characters at **0.44–0.55 ms/char**, which is the same 1.867 ms/token at the measured 3.16–3.43
chars/token. The sweep's own intercept is *not* usable — its caps span only 61%–100% of the text, and
extrapolating to zero puts the fixed term anywhere between 1.9 s and 4.1 s per query. So the fixed
term remains un-measured rather than measured-as-zero, and #21's standalone benchmark is still the
only thing that can settle it.

**LATENCY ON THIS BOX HAS A 15% BETWEEN-PROCESS FLOOR (2026-08-07, #25).** Repeating the same five
probes at the same config in **three fresh `engraph serve` processes** gave 61.9 / 63.4 / 64.5 /
71.3 s on byte-identical input. Within one process the spread is 1–5%. So any latency comparison that
restarts the server between arms needs interleaved rounds and `min` across them, or it is reading
process noise as a result — the first #25 sweep did exactly that and produced a plausible threshold
effect that did not exist. `eval/sweep-rerank-chars.py` does the rounds and now also **aborts if the
server process exits** and **refuses a run whose DEBUG trace count does not match the searches it
timed**: a leftover server holding the port made every subsequent server fail to bind while the
health check kept passing against the orphan, and 75 queries were silently answered by one uncapped
process. The resulting table — every cap identical in latency and rank — is indistinguishable from a
correct measurement of a deterministic reranker.

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
finding that at weight 0.8 its main contribution is to probe 5, the nonsense control; and **#21 is *not* revived** —
the per-candidate term it targets measures ~3.5 ms, not the 87 ms a two-point solve suggested; see
the section above. Graph is also **not a peer of the other two lanes**: it consumes
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
