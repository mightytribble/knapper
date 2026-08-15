# Fork notes

Private fork of [devwhodevs/engraph](https://github.com/devwhodevs/engraph) v1.7.2, maintained at
`mightytribble/engraph`. Evaluated as a knowledge-lookup layer to make for Obsidian-format vaults visible
within Claude Code, instead of relying on filesystem tools. Also wanted viable hybrid search across
vaults of arbitrary size, again to work around limitations of Claude Code grepping for information.

## Why this fork exists

Upstream is dormant — last commit 2026-05-27, seven PRs open and unmerged, several of which fix
real defects.

Upstream over-promised - several features either didn't work, or were implemented
contrary to how they were described.

Upstream search was sub-optimal; both FTS and Graph lanes required significant re-working.

Divergence from upstream can be tracked with:

```bash
git fetch upstream && git diff --stat upstream/main main
```

The divergence is significant and likely to get worse.


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
| `[lane_weights]` | the lane weights are one configured vector, not a classifier's guess; graph never outweighs the content lanes | this fork, issues #9, #59 |
| `resolve_n_threads` | llama.cpp runs on the machine's cores, not a constant 4 | this fork, issue #20 |
| `fts::any_term_expr` | the keyword lane matches terms, not the query as one phrase | this fork, issue #22 |
| no query expansion | the user's own query is the only one searched for | this fork, issues #23, #59 |
| `rerank::max_document_chars` | the cross-encoder's input *can* be bounded; upstream has no budget at all. Ships off — see #42 | this fork, issues #25, #42 |
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
| `ranking::apply_answer_floor` | a query with no candidate above the floor returns no results | this fork, issue #34 |
| `[embedding_prompt]` | EmbeddingGemma is fed the prompt format its model card documents | this fork, issue #10 |
| `DocumentTitle` | the document template's `title:` field is selectable; the vault's breadcrumb is one of its values | this fork, issues #36, #38 |
| `BreadcrumbRoot` | a breadcrumb leads with the file path, so its first segment resolves to a file on disk | this fork, issue #46 |
| `chunks_fts` external content | the keyword index is derived from the chunk table, and indexes the breadcrumb beside the body | this fork, issue #37 |
| `ranking::retrieval_width` | how deep the content lanes dig is a setting; `top_n` truncates the output and nothing else | this fork, issue #49 |
| `chunk_min_chars` | a section too short to stand on its own merges into the preceding chunk rather than becoming a row. Ships at 120 — see #43 | this fork, issue #43 |
| `structure_headings` | a bold-only line opens a section, deeper than any `#` heading and an ancestor of nothing. Ships on — see #44 | this fork, issue #44 |
| `tags` + `file_tags` | a tag is an attribute of a note, and it joins to the note. Usage and last use are derived, so neither can drift. Upstream keys its vocabulary in `tag_registry`, a flat table with no join to `files` whose `usage_count` counts index events | this fork, issue #60 |
| `tags::predicate` / `Scope` / `tags_under` | notes filter by tag term (`all`/`any`/`none`) and the vocabulary lists whole or by subtree | this fork, issue #60 |
| one tag vocabulary on three surfaces | the filter and the vocabulary each carry one name and one parameter set across `engraph list`/`engraph tags`, MCP `list`/`tags` and `GET /api/list`/`GET /api/tags`. A term is a tag path, a trailing `/` names the subtree, `scope` is the alias of `all`, and an unmatched `all`/`any` term is an error naming the nearest tag. Only the container encoding is per-surface: a JSON body reads an array, and a query string reads one comma-separated value, because `serde_urlencoded` reads no sequence | this fork, issue #61 |
| `Store::files_in_scope` / `rarray` scoping | a search can be scoped to the notes a tag filter admits. The scope resolves once to a set of file ids and every lane pre-filters on it: the KNN through vec0's `rowid IN`, the keyword lane through its own join, and the graph lane by dropping out-of-scope candidates before its quota truncation, so the walk still passes through untagged notes. An empty filter emits no clause and reproduces the unscoped pipeline exactly | this fork, issue #60 |
| `src/surface.rs` | every capability has one name and one parameter set on all three surfaces, and a test fails when one does not | this fork, issue #62 |
| directory terms in the scope | a scope term can name a directory as well as a tag: a leading `/` marks a path from the vault root, a trailing `/` its subtree, and the two mix in the same `all`/`any`/`none` operators. A directory resolves to a range predicate on `files.path` beside the tag's junction `EXISTS`, case-sensitive, and reaches `search` and `list`. The single-field alias of `all` is spelled `scope` | this fork, issue #65 |
| `topic` / `who` / `project` removed | the three composite bundles are gone from all three surfaces, and `build_people_edges` with them — the indexer writes no mention edges, and `LINK_RESOLVER_VERSION = 2` clears the stale rows out of an existing store on its next index run. Composites return as vault-defined commands (#71) once #35 defines what a bundle may emit | this fork, issue #73 |
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
cargo test --lib             # 852 pass
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

### There is no `tests/` directory

This fork deleted `tests/integration.rs`, `tests/write_pipeline.rs` and `tests/fixtures/`. They had
not compiled since upstream v1.0.0 — `unresolved import engraph::embedder`, `engraph::hnsw`, and a
`walk_vault` arity mismatch — so `cargo test` (full) and `cargo clippy --all-targets` both failed on
pristine upstream, and every test in them was `#[ignore]` behind a GGUF download. `integration.rs`
also reimplemented the index and search pipeline in its own helpers, so repairing it would have
asserted against a copy of the shipped code rather than against the code. The one behaviour with no
twin in the lib suite, the mtime conflict, moved to `writer::tests` and runs on `MockLlm`.

**A rebase onto upstream brings all three paths back.** Delete them again. Upstream PR #47 repairs
them instead, if that is ever the better answer.

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
- **Re-index before you trust a tag reader (#60).** `PARSER_VERSION = 2` declares the re-index that
  fills `tags` and `file_tags`, and until that index runs the two tables are empty. The paths that
  call `fingerprint::verify` refuse to run at all — `engraph search`, `engraph serve` and the write
  commands (`engraph create/update/move/archive/delete`) all answer "Run 'engraph index'".
  `engraph read/list` and `engraph status` do not verify, so they answer from the
  empty tables with no warning: `health` reports every note as missing its tags. Run `engraph index` once on an upgraded store before reading any of
  them.
- **Do not add a top-level config key with `>>`.** Every arm home's `config.toml` ends in a
  `[section]` table, so a line appended to the end of the file lands **inside that table**. TOML
  parses `promote_bold_headings = true` after `[memory]` as `memory.promote_bold_headings`, serde
  ignores the unknown field, and the top-level key keeps its default. The store then re-indexes,
  reports success, and is a second copy of the arm the key was supposed to change. #44 built one
  wrong arm this way and it was caught only by a fingerprint check against the control. Insert the
  key above the first line that starts with `[`, and read the file back.
- **The eval store lives at `standalone/mcp-isekai/.engraph-eval/`**, indexed by a CUDA build at the
  documented prompt templates (#10) and at `document_title = "breadcrumb"` (#36). Its
  `embedding_fingerprint` is `d41cc349`, which is the same value as the #36 breadcrumb arm — and no
  longer the default, so it re-indexes once when it is next opened (#38). The four
  #36 arm homes stay beside it as `.engraph-i36-*`, and the `.engraph-i36x-*` homes are the same four
  arms with #7's exclusion on. The two control arms name `document_title = "none"` in their own
  `config.toml`, because the default no longer gives it.
  The four #38 arms are `.engraph-i38-*`, built from `.engraph-i37-shipped`. `-shipped` carries no
  `[embedding_prompt]` and no `[fts]` block at all, and it exists to check the defaults #38 changes.
  The #42 arms are `.engraph-i42-*`; `-shipped` was the baseline of its day, defaults only.
  The two #44 arms are `.engraph-i44-off` and `.engraph-i44-on`, both copied from
  `.engraph-i43-min120` and fully re-indexed. `-off` is #44's control and reproduces
  `.engraph-i43-min120` row for row, over one SHA-256 of every
  `(path, seq, heading, heading_path, text)`.
  **`.engraph-i44-on` is the current baseline** — the shipped defaults, the arm
  `eval/ground-truth.json` is stamped against, and what a new arm should be copied from.
  `.engraph-i43-min120` is a record of the previous default, `.engraph-i43-min0` is #43's control,
  and `.engraph-i46-path` is the same configuration as `-min0` under its old name — the last two
  name `chunk_min_chars = 0` in their own `config.toml`, because the default no longer gives it.
  **Every arm home built before #44 names `promote_bold_headings = false` in its own
  `config.toml`** for the same reason: the key is a chunker-digest component, so an open without it
  quietly converts the home to the shipped arm and re-indexes it.
  `.engraph-i43-headings` and `.engraph-i43b` are **not #43's**: they are #44's
  heading probe under its old number, and they point at a **scratch vault under `/tmp`** that did not
  survive the session — the promotion rule is written out in #44 and the arms are 15 s to rebuild.
  The eight #37 arms are `.engraph-i37-*`, built from `.engraph-i36x-breadcrumb`. `-control` is the
  copy the **pre-#37 binary** was run against; the other seven come from re-indexing one home with the
  #37 binary and copying its database, so they share a vector space exactly and differ only in the
  keyword index's declaration. `-off` is the inert control and reproduces `-control` to nine decimal
  places.
- **Every arm home built before #46 refuses to answer, and each arm is reproducible as a
  configuration.** #46 made `breadcrumb_root` a chunker-digest component, because the value is written
  into `chunks.heading_path`, so nine homes fail `chunker_fingerprint`: `.engraph-i36x-control`,
  `-breadcrumb`, `.engraph-i37-shipped`, `-off`, `.engraph-i38-shipped`, `-none`, `-nolex`,
  `.engraph-i42-shipped` and `-cap1000`. The route back is the arm's own settings plus **three** keys,
  and an omission of any of them is silent:
  - `breadcrumb_root = "name"`, which #46 measured as reproducing the pre-#46 binary on 360 of 360
    result slots.
  - `[rerank] max_document_chars = 1000` for every arm before #42, because #25 shipped that cap as
    the default. The key is a `reranker_fingerprint` component, whose action is
    `InvalidateThresholds`, so it rebuilds nothing and warns about nothing at index time.
  - `promote_bold_headings = false` for every arm before #44, because #44 ships that rule on. The
    key is a chunker-digest component, so an open without it re-chunks the whole home.

  The nine homes hold **four** configurations, because each experiment was copied to a new home
  rather than run again, and the fifth is `.engraph-i46-name`. `eval/probes.md` §8 lists the homes;
  the four configurations, the control check for each and the scores are in #45 and its commit.
  **The store itself does not exclude derived files.** Its `exclude` is `[".obsidian/"]`, so
  `*-index.md` and `templates/` are in the corpus. Issue #7 measured that exclusion as a clear gain and
  the baseline never adopted it. Every table recorded for #26 through #10 therefore holds about
  29% derived rows in the window, and the graph lane in each of them runs over six hub files that have
  no incoming edges. #36's tables use the exclusion. The other tables need a re-run before they can be
  compared with them.
  The corpus the store indexes is not frozen. Every measurement from #26 to #29 was taken on a 247-file / 1598-chunk store; the
  pinned vault at `63f33e6` is 266 / 1863 / 1988 edges, because `standalone/mcp-isekai` tracks the
  live `cc-isekai` repo and it grew in between. Rank tables from either side of that are not
  comparable. Earlier stores lived in session scratchpads and expired with the sessions.
- **`[ranking] retrieval_width` is part of the measurement, not a display setting** (#49). Both
  content lanes retrieve this many rows, so a probe table taken at 15 and one taken at 60 are
  different experiments — probe 2's tracked answer is absent at 15 and rank 1 at 60.
  `eval/probes.md`'s readings are taken at 60, which is the default. **`top_n` is a display
  setting**: it
  truncates the result list,
  and the eighteen pool queries are prefix-stable from 20 to 25 and to 100. A table taken before #49
  names `top_n = 20` in place of the width, on a binary whose width was `top_n * 3`, and that is the
  same 60.
- **Backlinks used to rot on every save** (#27, fixed). Any store last written by a pre-#27 build has
  edges missing — 24 of 1084 per three files edited, on the isekai vault — and the fix does not
  reconstruct them. `engraph index --rebuild` is the one-time repair, and any graph-lane number taken
  before that repair is measuring a degraded graph.
- **`engraph status` misreports the model** as `all-MiniLM-L6-v2` while actually loading
  `embeddinggemma-300M`. Upstream PR #48 fixes it.
- **Intelligence is the cross-encoder and nothing else** since #59 — one 640 MB Qwen3-Reranker. It
  is still not a quality dial: treat on/off as distinct configurations.
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
- **`CHUNKER_VERSION` is 2, so every store built by an earlier binary re-indexes once on its next
  open.** The version is the chunker's *algorithm* input to `chunker_fingerprint`, hand-bumped
  because there is no runtime view of what a function does, and #44's carry rule changed the rules
  while every hashed number stayed the same. It applies to every arm home, whatever its settings, and
  it is on top of any key-driven mismatch an arm already has.
- **Upgrading past #12 re-indexes the vault on the first `engraph index`.** `run_index_inner` calls
  `store.ensure_embedding_dim`, which rebuilds `chunks_vec` at the model's width and discards every
  chunk indexed at the old one. It is automatic and prints what it is doing, but it is a full
  reindex, and until it runs the index is unreadable: `search` and `serve` refuse to start against a
  width the model does not produce rather than let sqlite-vec raise a shape error.
- **`chunks_vec` does not exist until something establishes its width.** `Store::init` runs before
  any model is loaded, so it cannot size the table and no longer guesses; a database that has never
  been indexed simply has no vec table, and the semantic lane returns nothing. The width comes from
  `ensure_embedding_dim` at index time, or from the first vector written.
- **Each half of the embedding prompt is a config key** (`[embedding_prompt]`, this fork, issue #10).
  `document` and `query` both ship `documented`, which is EmbeddingGemma's own format —
  `<bos>task: search result | query: …` and `<bos>title: {title | none} | text: …`. `legacy` on
  either side restores nomic-embed-text's `search_query:` / `search_document:` convention and is the
  control each was measured against.
  - **The `<bos>` is written into the template on purpose.** The documented strings contain none, and
    `str_to_token` is called with `AddBos::Never` because the template supplies one literally
    (`parse_special = true` in llama-cpp-2, so it becomes the real BOS token). Removing it from the
    template and switching that call to `AddBos::Always` would also give `QwenEmbedding` and `Raw` a
    BOS, which they have never had.
  - **The two halves have very different costs.** `document` is a component of
    `embedding_fingerprint`, so changing it re-indexes the vault — 68 s on the GPU store, and it
    happens by itself on the next `engraph index`. `query` reaches no fingerprint, because a query is
    embedded and discarded.
  - **Which template built a store is hashed as data**, from `PromptFormat::template_id`. Nothing is
    hand-bumped to switch templates; `fingerprint::PROMPT_TEMPLATE_VERSION` covers a reword of a
    template that keeps its name.
  - **`document_title` decides what fills the `title:` field** (#36, #38). `none` ships, and it is
    the literal the model card gives for a document with no title. `breadcrumb` is design §5.4's
    `Note Title > H1 > H2 > H3`; #36 shipped it and #38 measured it out again — it costs no tracked
    answer either way, below the answers the two arms are a draw on the six positive queries, and it
    scores four of eleven negatives higher. `note` is the note's title alone, and it breaks
    exact-name lookup. The key is a fingerprint component, so a change to it re-indexes the vault.
    **A store built before #38 re-indexes once on upgrade**, because the default is no longer what
    that store holds.
- **The keyword index is derived from the chunk table, and indexes the breadcrumb** (this fork, issue
  #37). `chunks_fts` is an FTS5 **external-content** table over `chunks`, kept in step by three
  triggers, so it holds an index and no text of its own. #11's bug class — the keyword index holding a
  different string from the chunk — stops being a bug that was fixed and becomes a state that cannot
  be reached, and the four hand-written FTS writes and four hand-written FTS deletes are gone with it.
  A delete SQLite performs itself, on the cascade from `files`, reaches the index too.
  - **`[fts]` decides which columns the index is declared over**, and the fingerprint hashes the
    declaration, so a change to a flag is a keyword-index rebuild: 1598 rows in 0.1 s, no vault read
    and no model. The BM25 weights are query-time and reach no fingerprint at all, so a weight sweep
    costs nothing. `engraph search --explain` prints the declaration and the weights.
  - **`heading_path = true` at weight 1.0, `tags = false`** — measured, not designed. §6.2 of the
    convergence document asks for both columns at `1.0, 3.0, 4.0`.
  - **`heading_path = false, tags = false` is the control, and it is exact.** The new binary with both
    columns undeclared reproduces the pre-#37 binary on all eighteen calibration queries, to nine
    decimal places. A zero *weight* is not the same thing: BM25 normalises over every token in the
    row, so a populated column at weight 0.0 still moves every score.
  - **A store built before #37 re-indexes once**, declared by `fingerprint::CHUNK_RECORD_VERSION` in
    the chunker's key. The breadcrumb cannot be derived from anything `chunks` already holds — only
    the leaf heading is stored, and its ancestors are in the vault.
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
- **Changing `[embedding_prefix]` needs `engraph index --rebuild`.** Incremental indexing compares
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
- **Changing what goes into FTS needs `engraph index --rebuild`**, for the same reason
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
  Mid-measurement this is a trap: a bare `cargo build --release` overwrites `target/release/engraph`
  with a 27 MB CPU binary in about 20 s, and the next `index` run silently re-embeds the arm on the
  other device. Two tells — the index takes about 158 s instead of 65 s, and a query against any
  store the other binary built returns zero results.
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
- **A search can return no results** (this fork, issue #34). `[ranking] answer_floor` is 0.30 by
  default. The code removes every candidate that the cross-encoder scored below the floor. An empty
  result set prints *"No relevant content found for this query in the vault."* A floor of 0.0
  removes nothing and gives the same output as the build before #34.
  - **The floor value is only correct for the pinned corpus and the current reranker.** It is fit
    against the eighteen-query pool in `eval/probes.md`. A new pin makes the pool invalid, and the floor
    with it. A new reranker also makes it invalid. This is what `reranker_fingerprint` and its
    `InvalidateThresholds` action are for.
  - **The floor applies to each candidate, and this decides the value.** #34 specifies a fit against
    the best score of each query, which gives 89%. A floor of 89% removes probe 2's correct answer,
    which scores 81% below two better results from a different file. The limit is the lowest correct
    answer, 52.52% at probe 1's `archivist-lenne.md`, rank 5, against 5.33% for the highest negative
    a floor can reject — a 28.93% midpoint.
  - **Three negatives score above any usable floor, and the cause is the reranker.** *In which city
    is Tandi's brother a blacksmith?* scores 97.82% on a passage that names Tandi, a brother, a
    blacksmith and a city, where the brother is Mira's. *What spell can be used to clean clothing?*
    scores 97.87% on one that purifies a body. *Who is the Precept of Alder's Crossing?* scores
    86.20% on the section naming who runs the place. The reranker scores the topic of a passage, and
    one incorrect entity does not lower the score. **Eight of the eleven negatives return no
    results**, and so does the nonsense control; these three do not.
  - **How many results the floor removes depends on the query.** The control goes from 20 to 0, P6 to
    2, P1 to 5, P2 to 8, and P4 stays at 20, because all 20 sections of that exact-name query score
    above the floor.
  - **What counts as a correct answer is #34's reading, and it is open.** The 52.52% anchor is the
    lowest result #34 judged an answer; the responsive sets of #45 put two more P1 members below it,
    at 23.50% and 1.33%. Read against tier 1 no floor both keeps every responsive chunk and rejects
    anything. `eval/probes.md` §6 states both readings.
  - **`[ranking] per_note_cap` is 0, which means no limit.** The conflict between §9.1 and #30 stays
    open, and the key makes a sweep possible without a code change.
- **GPU numbers and CPU numbers are different baselines**, which is a standing rule in
  `eval/probes.md` §7: the kernels are not bitwise identical, the embeddings differ in
  the low bits, and the retrieved candidate set differs with them. Comparing a GPU rank table with a
  CPU one measures the backend, not the change under test.
- **A tag is file-level, because Obsidian defines it that way.** A tag is an attribute of a note, so
  `file_tags` keys on `files(id)` and not on a chunk. A body `#tag` does have a position, and
  attributing it to its chunk would give a tag a meaning Obsidian does not give it. The two sources
  are peers: the `tags` property and the body's `#tags` are a union, and a reader of one alone holds
  a subset of the vault.
- **A term is a tag path, and a trailing `/` — or its synonym `/*` — asks for its subtree.**
  `tags::parse_term` drops one leading `#` and folds case; `tags::predicate` compiles the subtree
  form to a range over `tags.path` rather than a `LIKE` pattern, so the unique index serves it too.
  `Store::list_files` takes a `Scope { all, any, none }` of terms — every `all` term, at least one
  `any` term, none of the `none` terms, ANDed together and with `folder`/`created_by` — as
  `--all/--any/--none` on the CLI, each comma-separated (`--scope` the alias of `--all`), as
  `all`/`any`/`none` on MCP's `list` tool (`scope` the same alias), and as the `scope`/`all`/`any`/`none`
  query parameters on `/api/list`, each one comma-separated value, read through the same term grammar.
  `Store::tags_under` answers the whole vocabulary or
  one subtree, ordered by path, as `TagCount { path, display, note_count }`, the count always the
  exact tag's notes and never a subtree total, through `context tags [--under <term>]`, MCP's `tags`
  tool and `GET /api/tags?under=<term>` (#61). An `all` or `any` term matching no tag in the vault is an error naming the nearest one,
  from `resolve_tag` — except that an over-deep term answers with its longest existing ancestor,
  because `resolve_tag`'s `Extension` variant echoes the proposed tag back rather than naming the tag
  it extends. A `none` term is not checked, and `tags_under` validates no prefix at all: an empty
  subtree is a true answer for a caller exploring.

## Open work

See issues on this repo.
