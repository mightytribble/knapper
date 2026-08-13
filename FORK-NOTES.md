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
| `tags::predicate` / `TagFilter` / `tags_under` | notes filter by tag term (`all`/`any`/`none`) and the vocabulary lists whole or by subtree | this fork, issue #60 |
| one tag vocabulary on three surfaces | the filter and the vocabulary each carry one name and one parameter set across `engraph list`/`engraph tags`, MCP `list`/`tags` and `GET /api/list`/`GET /api/tags`. A term is a tag path, a trailing `/` names the subtree, `tags` is the alias of `all`, and an unmatched `all`/`any` term is an error naming the nearest tag. Only the container encoding is per-surface: a JSON body reads an array, and a query string reads one comma-separated value, because `serde_urlencoded` reads no sequence | this fork, issue #61 |
| `Store::files_in_scope` / `rarray` scoping | a search can be scoped to the notes a tag filter admits. The scope resolves once to a set of file ids and every lane pre-filters on it: the KNN through vec0's `rowid IN`, the keyword lane through its own join, and the graph lane by dropping out-of-scope candidates before its quota truncation, so the walk still passes through untagged notes. An empty filter emits no clause and reproduces the unscoped pipeline exactly | this fork, issue #60 |
| `src/surface.rs` | every capability has one name and one parameter set on all three surfaces, and a test fails when one does not | this fork, issue #62 |
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
  `engraph read/list/who/project` and `engraph status` do not verify, so they answer from the
  empty tables with no warning: `who` finds nobody through its tag pass, and `health` reports every
  note as missing its tags. Run `engraph index` once on an upgraded store before reading any of
  them.
- **A promoted heading is not addressable by `read --section` or `update --section`.** Promotion
  (#44) puts a bold-only line in `chunks.heading`, so a search result prints
  `lore/bestiary/archdragon.md > **Spells**` — 107 of the shipped arm's 1559 chunks are labelled this
  way, `**Abilities**` 67 times and `**Spells**` 22. `markdown::find_section` reads ATX headings only,
  which is deliberate and is what the section editor writes against, so `engraph read --section` with
  `**Spells**` and with `Spells` both answer "Section not found", and `engraph update --section`
  cannot target the passage either. Read the whole note instead, or name the enclosing `#` heading.
  The breadcrumb the result prints is the way back: the ancestor before the bold line is an ATX
  heading and does resolve.
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
  `Store::list_files` takes a `TagFilter { all, any, none }` of terms — every `all` term, at least one
  `any` term, none of the `none` terms, ANDed together and with `folder`/`created_by` — as
  `--all/--any/--none` on the CLI, each comma-separated (`--tags` the older spelling of `--all`), as
  `all`/`any`/`none` on MCP's `list` tool (`tags` the same alias), and as the `tags`/`all`/`any`/`none`
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

See issues on this repo:

- ~~**#1** structure-first chunking (section → sub-section → paragraph → size)~~ — **done**, 93.1%
  heading attribution (best measured). Retrieval barely moved; see #6 for why.
- **#2** contextual embedding prefix (filename / heading path / tags into every chunk) — **built,
  measured, shipped off by default, and open on the question it did not answer.** Net negative on the
  seed probes: +3 ranks on the conceptual probe, −4 (out of the window) on the exact-name
  non-regression probe. Both of its retrieval acceptance criteria turned out to have been met already
  by #1 + #6. Whether a length-scaled or conditional prefix is worth having is what keeps it open, and
  it is cheap to answer — every component is separately switchable.
- ~~**#11** FTS indexes the 200-char snippet, not the chunk~~ — **done.** Keyword search had been
  reaching 27.6% of the corpus; `Saltmere` and four other verified-present terms returned zero hits.
  Probe 3 went @5 → **@1**, the best result any configuration has produced on it, and probe 4 held
  at @2 under the heaviest churn of any probe. Nothing regressed. Moved 73 of 100 probe slots, so
  every lexical number recorded before it is superseded.
- **#3** more probes, drawn from real usage — *not* a battery. The five seed probes catch regressions
  and resolve large effects; what they cannot do is settle a one-up-one-down at n=5 (#14). Fix is
  sample count, plus negative controls and section-vs-file scoring recorded separately. Calibrates #4,
  and **#41 blocks on it**: that question needs two probes the eighteen-query pool does not have —
  one whose answer sits under a heading whose terms are absent from the chunk body, and one naming a
  folder term
- **#4** relevance floor — §8.3's per-lane dense candidate floor. Deferred, and narrower than its
  title. #34 shipped the half that makes a query with no answer return nothing, and it uses the
  calibrated cross-encoder score. The retrieval lanes have no calibrated score, so the per-lane half
  waits for a probe
- ~~**#34** the answer floor~~ — **DONE.** `ranking::apply_answer_floor`, at
  `[ranking] answer_floor = 0.30`, applied to each candidate after `sort_by_rerank` and skipped when
  `rerank_score` is `None`. Probe 5 returns no results, which it had never done before, and it
  prints the message. A floor of 0.0 gives output identical to the build before #34 on all five
  probes. Two mutants break the tests: a floor that rejects nothing fails 4, and a floor that
  rejects everything fails 9.
  The pool disagreed with the ticket twice. First, the ticket expected a clear gap and the pool has
  an overlap: N11 scores 97.1% and N4 scores 86.2%, above the 91.7% of the weakest positive, so no
  floor can reject them and keep P6 and P7. We checked both against the vault. The reranker scores
  the topic of a passage, and one incorrect entity does not lower the score. Second, the fit the
  ticket specifies gives 89%, and 89% removes probe 2's correct answer at 81%. A floor that applies
  to each candidate must be below the lowest correct answer, not below the best result of each
  query. The value is the midpoint of two scores, and both are re-taken on the shipped engine:
  **5.33%**, the highest negative a floor can reject, and **52.52%**, probe 1's `archivist-lenne.md`
  at rank 5, for a 28.93% midpoint. The margin is 23 points above and 5 below. Eight of the eleven
  negatives return no results, and N10 joins N11 and N4 above any usable floor. Every correct answer
  keeps its rank. The result count goes from 20 to 2 on P6, and stays at 20 on probe 4.
  `[ranking] per_note_cap` ships at 0.
  **Layer 2 of `docs/vault-search-convergence.md`, first half**
- **#35** output contract — the scored window is the emitted window (today the model reads
  `chunks.text` capped at `max_document_chars` and the caller gets the 200-char `snippet`, so the
  passage that earned the rank is not the passage that comes back), `budget_tokens` with a greedy
  fill and an 8-item overflow list, structured content with `untrusted_content`, `truncated`,
  `status`, and provenance replacing `score`/`confidence` **on the MCP and HTTP payloads only** —
  §7.3's boundary is the consumer, not the field, and the CLI keeps the percentage `probe.sh` reads.
  Settled by ordering that must not move. **Layer 2, second half**
- ~~**#36** the breadcrumb rule's embedding limb~~ — **DONE.**
  `[embedding_prompt] document_title` selects what fills the `title:` field. The values are `none`,
  `note` and `breadcrumb`. `breadcrumb` is design §5.4's rule and is the current default.
  `EmbedModel::embed_batch` takes `EmbedDoc` pairs, so the title belongs to the document and not to the
  model. `prefix::embed_inputs` composes both fields, and `EmbedComposition` carries the two settings as
  one value. That keeps the indexer and the write pipeline in one vector space. `chunks.heading_path` is
  not stored: the embedding limb reads the chunker's in-process value, and no other code needs the column
  until #37.
  **The arms were measured twice.** The first pass kept the derived files, and derived files then held
  28.6% to 30.0% of the result slots. Two findings were artifacts of that corpus. `note` appeared to
  gain a rank on probe 2, through `lore-index.md`, and the gain is absent from the clean corpus. `note`
  and the body control appeared to lose probe 4 to a list of links in `lore-index.md`. Every table in
  #36 uses `exclude = ["*-index.md", "templates/"]`, which is #7's setting, and
  0 of 1440 result slots hold a derived file.
  `note` and the body control lose the exact-name answer and gain nothing.
  `archdragon.md > ## Definition` has rank 1 at 99.97% in the control, and it is absent from the top 20
  in both. That is #2's failure. A file-level table cannot see it, because `## Stat Block` holds rank 1
  for the file in every arm. The body control also destroys abstention: N10 goes from 1.61% to 97.87%
  and returns one result above the floor, which no floor can reject.
  **Arm A against arm B confirms the mechanism the issue predicted.** The note title is a per-file
  constant, and it loses probe 4. The breadcrumb is different for each section, and it keeps probe 4. The
  body control carries the same variable string and loses probe 4, so the field and the body are not
  equivalent. Probe 3, the named target, had no headroom, because it has had rank 1 since #10.
  `answer_floor = 0.30` needs no new fit for the control, `note` or `breadcrumb`.
  **`breadcrumb` holds every tracked answer and makes the secondary results worse.** It swaps 22 results
  below the answers in the six positive queries. Probe 3 loses `## Level 6 Antimagic Shell` (*"spells
  cannot be cast"*) and `## Level 4 Silence` (*"cannot cast spells of any school"*), and gains
  `## Level 3 Invisibility` at rank 3. Probe 6 loses `## Level 9 Restore Object` and puts
  `## Level 3 Heal` above the floor. Probe 7 loses the right entity. Probe 4 gains a little. Probes 1
  and 2 are noise. Each reading is a manual check against the vault text.
  An aggregate of `confidence` cannot judge that churn, because the score is the sort key and a redundant
  window gets the highest mean. Probe 4 shows the metric with the wrong sign.
  **The default accepts that cost.** `breadcrumb` is §5.4's rule, it costs no tracked answer, and the
  rule's three limbs carry the same string. #37 puts the breadcrumb in the lexical lane, where a heading
  term is matched and not averaged into a vector, so the embedding limb is measured again after #37 and
  is not closed now. #3 is the evidence still owed
- ~~**#37** the breadcrumb rule's lexical limb~~ — **DONE.**
  `chunks_fts` is external content over `chunks` with three sync triggers, and `[fts]` declares it
  over `text` plus whichever of `heading_path` / `tags_text` is on. Both columns are stored on every
  chunk row whatever the config says; the config decides what is *declared*, so switching one is a
  0.1 s rebuild and never a re-index. `prefix::breadcrumb` is the one composition both limbs of the
  rule call. `NewChunk` is the chunk row as it is written, and `CHUNK_RECORD_VERSION` is the
  fingerprint input that asks whether the row still holds what a reader needs — separate from
  `CHUNKER_VERSION`, which asks whether the chunks would fall in the same places.
  **The control is exact.** The new binary with both columns undeclared reproduces the pre-#37 binary
  on all eighteen calibration queries, to nine decimal places, 0 of 360 result slots moved — after a
  schema change, a full re-index and a rebuilt index. A zero weight would not have shown this, because
  BM25 normalises over every token in the row.
  **The breadcrumb returns an answer #36 dropped.** `rules/abjuration-spells.md > ## Level 4 Silence`
  — *"the silenced target cannot cast spells of any school"* — enters probe 3 at rank 4, 94.04%, and a
  summoning spell that answers nothing leaves. It is one of the two correct answers #36's arm B lost,
  and recovering it is the argument that issue made for accepting its own cost. It costs one swap
  below the answer on probe 4, and no tracked target moves.
  **The tags cost two things and gain nothing, so they ship off.** They drop probe 2's most direct
  answer, `the-archdragon-disguise.md > ### Summary` at 94.94%, out of the window entirely, and they
  give the adjacent negative N11 a 97.06% result at rank 2. The tracked answer's 4 → 3 in the tags
  arms is that drop and not a gain. The mechanism is in the vault: `npcs/tandi.md` is tagged
  `velthos`, a city, so a tag records an attribute of a note rather than something the note discusses
  and the keyword lane cannot tell the two apart. That is half of #17 tried the cheap way; the tag
  *registry* half is untouched.
  **The weights buy nothing.** Probe 3's recovered answer arrives at rank 4 at heading weights 1, 3
  and 5 alike, and the only difference is churn — 62, 117 and 123 of 360 slots. §6.2's `1.0, 3.0, 4.0`
  is not supported. `answer_floor` needs no refit: same 6.77% highest rejectable negative, same 52.52%
  lowest correct answer, same 29.64% midpoint.
  **One invariant reads differently from the issue.** `'rebuild'` reproduces the trigger-built index's
  *postings and scores* exactly, not its bytes: an incremental write leaves one segment per batch
  where a rebuild writes a single merged one. The test asserts `fts5vocab` rows and BM25 scores, which
  is what "the same index" can mean. #38 is the evidence still owed
- ~~**#38** re-measure the breadcrumb rule's embedding limb, now that the lexical limb is in~~ —
  **DONE.** `document_title = "none"` ships. #36 shipped `breadcrumb` while making the secondary
  results worse on two of six positive queries, and accepted that cost on the argument that the
  lexical limb would pay for it. It did not.
  **No tracked answer separates the arms** — same rank and same score on all seven, which #36 already
  recorded for this pair. The decision is entirely the results below them.
  **Read against the vault, the positive queries are a draw.** `none` wins probes 6 and 7: probe 6 gets
  `## Level 9 Restore Object` at rank 2 where `breadcrumb` puts `## Level 3 Heal`, which does not touch
  cloth, above the floor, and probe 7 keeps the right species. `breadcrumb` wins probes 3 and 4: it
  returns four answering spells in probe 3's top seven against three, and `none` admits three archdemon
  sections to an archdragon query. Probes 1 and 2 are noise.
  **Abstention decides it.** `none` scores four of the eleven negatives lower, N10 at 1.61% against
  6.77% and probe 5 at 0.02% against 0.23%, and it is the only difference in the pool that is not a
  manual judgment about one passage. Above the floor it takes probe 6's wrong answer out and two of
  N11's ten results with it, and gives back only probe 4. `answer_floor` needs no refit: 29.64%
  midpoint again.
  **The 2×2 says what each limb reaches.** Crossing the two limbs on probe 3: the embedding limb
  decides which pair of answers the query reaches, and its cost is `breadcrumb` with a body-only
  index, which reaches one answer beside Counterspell where every other cell reaches two. The lexical
  limb is what pays that off. With the embedding limb off, the pair is there without it.
  **The defaults are verified end to end.** A home with no `[embedding_prompt]` block and no `[fts]`
  block, indexed from empty, reproduces the explicit arm on 360 of 360 result slots.
  **#45 scored both arms against the responsive sets and confirmed the default on today's binary.**
  The arm that would ship now is `document_title = "breadcrumb"` at `breadcrumb_root = "path"`, and it
  holds 3 of 5 on P3 where the default holds 4 of 5. Two readings above are corrected. The `name`-rooted
  breadcrumb returned `rules/dark-spells.md > ## Level 3 Curse of Tongues` at rank 7, 90.44%, which is a
  tier-1 member no other arm returns and which this issue removed without counting it; #46 then made
  that result unreachable. And the 2×2's `## Level 3 Invisibility` is tier 2, so the cell it separates
  reaches one tier-1 member fewer and not one answer fewer
- ~~**#46** the breadcrumb root should be the file path, not a frontmatter key engraph does not own~~
  — **DONE.** `breadcrumb_root = "path"` ships: `lore/bestiary/lesser-dragon.md > Stat Block`.
  **`name:` was never engraph's key.** The engine read it and never wrote it, it is a cc-isekai
  convention, and Obsidian gives meaning to `aliases` / `tags` / `cssclasses` alone — a note's title
  is its filename. Another vault's `name: Aragorn` on a character sheet would have become that note's
  breadcrumb whatever the file was called.
  **The control is exact**: `name` on the new binary, after a full re-index, reproduces the pre-#46
  binary on 360 of 360 result slots.
  **The arm costs nothing measurable.** Every tracked target holds rank and score, the above-floor
  count is identical on all eighteen queries, and 18 of 360 slots move — on two negatives, everything
  below 0.35%. No positive query changes in any slot. It survives losing `name` because the path
  carries the same terms: of 252 files with `name:`, the path recovers 189 fully, 61 partly and loses
  2, one of which #7 excludes.
  **It cannot show a benefit, and did not test the risk.** The gain is that a breadcrumb quoted away
  from its result object still names a file — a property of the string, not the ranking. And folder
  names now enter the keyword index on every chunk beneath them, but **0 of the 18 pool queries
  contain a folder term**, so that dilution is untested rather than absent.
  Two knots come undone: the 64 `X > X` breadcrumbs are gone, which **closes #41's third arm**, and an
  H1 becomes an ordinary heading level with no consumption rule. `stem` is recorded as unusable — 14
  stems name more than one file. `writer::extract_title` is untouched; naming a file being created is
  a different job
- **#47** *(low priority)* `writer::extract_title` guesses a filename, and since #46 that filename is
  a breadcrumb root — it reads frontmatter `title` (neither Obsidian's key nor engraph's: the
  community "Front Matter Title" plugin's, and no community plugins are installed here), else the
  first `# heading`, else the first non-empty line, **truncated to 50 characters**. It produces a
  filename rather than a label, so it is not a competing answer to `DocContext`'s job — but #46 made
  the filename the first segment of every breadcrumb of every note the write pipeline creates.
  `generate_filename` does not consult the store either, and the write path's only defence against a
  collision is `final_path.exists()`, which refuses the write. **Nothing in `eval/pool.sh` reaches
  this code** — the calibration corpus is authored, not written — so the evidence is unit tests over
  a fixture and no arm can be run. It becomes urgent the first time a workflow writes notes into a
  vault that is then searched
- ~~**#45** score the pool's queries against a ground-truth responsive set, not against churn~~ —
  **DONE.** The two metrics this file used before it are both broken. *"Result slots that moved"*
  counts change and calls it quality; *"read the swaps against the vault"* inspects only what differs
  between arms, so it cannot see a responsive chunk absent from both or a noise chunk high in both.
  `rules/developer-console.md > ## [3] SPELLS`, a console menu, has sat at **rank 2, 95.65%** on probe
  3 in every arm of #36, #37 and #38 and no reading ever saw it. Per query: a tier-1 responsive set
  and a tier-2 informative set drafted from the vault by the author, then coverage, the rank of each
  tier-1 member, and inversions. **It is the instrument #43 and #44 were judged with, and the one
  #41 needs.**
  **The sets are agreed and shipped as data**, one query per comment on the issue with the author
  ruling every reading. `eval/ground-truth.json`, built by `eval/build-ground-truth.py` and read by
  `eval/score-ground-truth.py`. At the pin: P1 **7/7**, inversions 0; P2 **5/5**, 4; P3 **4/5**, 7;
  P4 **6/9**, 1; P6 **1/1**, 0; P7 **1/1**, 0.
  **The pool holds five positives and one false premise.** P7 asks which form Lesser Dragons adopt and
  the vault says they adopt none, so it is neither a positive nor one of the eleven negatives —
  abstention would score it as a failure for returning the refutation, which is the right answer.
  **A responsive set is not a property of the query string**, so each query records the intent it was
  drafted against: P3's `## Level 3 Invisibility` is responsive to *"which spells limit a caster"* and
  not to *"which spells stop a caster casting"*, and the query says neither.
  Five boundary rules: a negative answer is tier 1; tier 1 needs the text to *state* the effect asked
  for, or a piano dropped on a mage qualifies; right topic and wrong entity is tier 2; a bare entity
  name is a keyword search, so tier 1 is the canonical entry in full and nothing else; overkill is
  tier 2.
  **The key is `file`, `heading` as the result JSON prints it, `section`, and the anchor sentence.**
  Neither `path > heading` nor the stored breadcrumb is unique — 27 groups in the corpus collide,
  because a split section gives every piece the same heading. The scorer resolves a member by anchor,
  which survives a re-chunk, and reads `section` as a stamp taken at the pin; when the two disagree it
  reports that the chunking moved rather than scoring a stale list. An anchor that does not occur in
  its chunk fails the build. A result above the lowest tier-1 member that is in no list is reported
  **unclassified** rather than counted as noise, so an incomplete list asks for a ruling instead of
  reading as a clean run.
  **What it has already overturned:** P3 is missing a correct answer and carries seven noise chunks
  above another, while Counterspell holds rank 1 and every older metric reads the query as healthy;
  P4 returns two thirds of its own entity's file, and the keyword index matches all nine, so the loss
  is after the lane; the pool discriminates arms on **four** queries, because P6 and P7 hold one
  member each at rank 1 and their inversion count is inert; #37's reading of
  `the-archdragon-disguise.md > ### Summary` as "probe 2's most direct answer" is wrong; #38's probe-6
  limb compared a tier-2 chunk with a noise chunk, both below the single correct answer, and is
  withdrawn; `[ranking] answer_floor = 0.30` cuts a tier-1 member of P2 that scores 3.45%, against the
  52.52% lowest correct answer #34 fit it on, and the fit is re-read once, at the end, against all six
  sets.
  **Step 4 re-measured the arms on record, and every control held.** No kept JSON existed and all nine
  pre-#46 homes fail `chunker_fingerprint`, so each arm was rebuilt as a configuration — see the entry
  above on the two keys a reconstruction sets. The nine homes are four configurations plus
  `.engraph-i46-name`. **#42's uncapping is the largest gain on record**: three of P2's five tier-1
  members go from about 1.5% to between 78% and 98%, and coverage cannot see it because every arm
  holds 5 of 5. **#37's lexical limb is +1 coverage on P3 and costs nothing**, and #36's embedding limb
  is the −1 it pays back. **P4 holds 6 of 9 in every configuration since #36**, so that loss belongs to
  no arm and goes to #41. **P1, P6 and P7 give the same score in every arm on record**, so the pool
  discriminates on P3 and P4. `document_title = "none"` is confirmed on today's binary: at
  `breadcrumb_root = "path"` the `breadcrumb` arm holds 3 of 5 on P3 against the default's 4 of 5.
  **One defect in the instrument is filed as #50** — the scorer counts inversions above the *lowest*
  tier-1 member, so an arm that loses coverage is judged over a smaller window and its inversion count
  falls. Two arms compare on inversions only when that member holds the same rank.
  #45's tables are in its commit; `eval/probes.md` §4 holds the rule the scorer now applies
- ~~**#49** `top_n` sets retrieval width, so asking for more results returns worse ones~~ — **DONE.**
  `[ranking] retrieval_width` is the lane width, default 60, and `top_n` truncates the output and
  nothing else. Both content lanes had fetched `top_n * 3` while `[ranking] candidates` stayed at 30,
  so the output limit decided *which* thirty candidates the cross-encoder was shown: **twelve of the
  eighteen pool queries changed their ranking between `top_n = 20` and 25**, P3 losing
  `## Level 4 Silence` (95.60%, rank 3) and `## Level 6 Antimagic Shell` (94.82%, rank 4) at any
  rank, and N10 gaining a new best result, which is an input to the `answer_floor` fit. The default
  reproduces the shipped arm on **360 of 360 result slots**, and all eighteen queries are now
  prefix-stable from 20 to 25 and to 100. `candidates` is the ceiling on a result list, so
  `top_n = 100` returns 30. The width is query-time and reaches no fingerprint, so an arm is a re-run.
  The tables are in this issue's commit; `eval/probes.md` §2 holds the shipped width
- ~~**#50** the inversion count falls when an arm loses coverage~~ — **DONE.**
  `eval/score-ground-truth.py` counts noise above the **lowest-ranked tier-1 member present**, so an
  inversion count is taken over a window the arm itself sets: an arm that loses that member shortens
  the window, and its count falls with the coverage it lost. On P3 the shipped arm holds the member at
  rank 15 and reports 7 inversions; the `breadcrumb` arm at `breadcrumb_root = "path"` loses it, holds
  the next member at rank 4, and reports **1** — while carrying 10 noise chunks in the top 20 against
  the shipped arm's 12. The scorer now prints the **anchor rank** beside the inversion count and a
  **noise count over the whole returned window**, which compares whatever the coverage. Both fixes the
  issue proposed, because the anchor rank is what says whether the inversion count may be read at all.
  One row of the #57 grid changes meaning: Exploratory / Qwen reads 3 inversions, the joint lowest in
  the table, over a P4 window three ranks deep, and on the window count it carries 27 noise chunks
  against 25 for the cell returning seven more answers
- ~~**#52** a noise stamp cannot survive a re-chunk, because it carries no anchor~~ — **DONE.**
  `(file, seq)` is a position and a merge moves positions. Tier 1 falls back to its anchor sentence
  since #43; noise had none, so five stamps dropped on #43's arm and P2 read `inversions 3` where the
  truth was 4, the missing row being its own noise row an ordinal lower. Every one of the 56 noise
  entries now carries an anchor, and `eval/pick-noise-anchors.py` reads them from
  `eval/ground-truth.json` rather than from a second copy of the table it held itself. Each entry is
  located **by its anchor** — content, not position — and an anchor naming no chunk, or more than one,
  prints `None` and is reported; a stamp that moved is reported with the ordinal it moved to. A silent
  wrong-chunk pick is unreachable, where the old default store picked 10 of 56 anchors from the wrong
  chunk and said nothing.
  **`eval/chunk-rows-sha.py` had the same class of defect.** Its `0x1f` separator sat between the
  fields of a row and not at the end of one, so one row's `text` abutted the next row's `path` — the
  run the separator exists to stop. The separator terminates the row as well, and the row hash
  `eval/probes.md` §1 records moves with it
- **#51** a short **piece** of a split section escapes the chunk minimum — `emit_section` flushes
  whatever it has packed when the next paragraph alone busts the budget, so a section over the minimum
  can still emit a short row. `summaries/session-013.md > ## NPC Activity` is 68 characters at seq 5,
  and it is the only row under 120 characters in the shipped corpus that is not its file's first
  chunk. One row of 1461, so it is a paper cut, and #43's own mechanism is the fix: merge the flushed
  piece into the preceding chunk on the same terms
- **#39** should neighbouring chunks be coalesced, and when? — when a lane returns chunks that are
  adjacent in their file, does merging them before the answer is presented help, even where they are
  not adjacent in the ranking? The case for it is that a weak claim beside a strong one may be the
  context the strong one needs. It touches what #35 emits and not what the lanes retrieve, and #43's
  merge is the same operation at index time on a different criterion
- **#40** find out what information makes tags useful — #37 measured the tags out of the keyword
  index on the reading that a tag records an attribute of a note rather than something the note
  discusses, and the lane cannot tell the two apart: `npcs/tandi.md` is tagged `velthos`, a city. The
  question this asks is the other direction — what a tag would have to record for the lane to gain by
  matching it. It decides what #17's registry half is worth
- ~~**#48** the per-document shortlist cap hides a responsive chunk from the cross-encoder~~ —
  **closed, not reproducible.** Both of its arms ran in a home at `top_n = 100`, so neither was the
  shipped configuration. The chunk it reported as hidden is rank 2 at 93.50% in the shipped arm, and
  `shortlist_cap` is not binding on either query cited. Superseded by #49
- ~~**#43** sections under the minimum size should not become chunks~~ — **DONE.**
  `chunk_min_chars` is the shortest section body that becomes a chunk of its own; a shorter section
  merges into the **preceding chunk of the same file**, keeping its own heading line inside the
  merged body so its terms stay in `chunks.text` and the keyword index over it. The host keeps its
  `heading` and `heading_path`, the merge stops at `TARGET_TOKENS` — these sections run in streaks —
  and a section with no preceding chunk stays a chunk. The unit is characters because the chunker
  estimates `chars / 4`, so §5.4's 30 tokens is 120. It is a chunker-digest component, so a change
  re-indexes, and **the digest moves at 0 too**: every store re-indexes once on the upgrade.
  **The default is 120.** The corpus goes 1598 → 1461 chunks and the 72 sub-60-character rows go to
  zero, and the pool reads nothing against it: every negative holds or falls, every set holds its
  coverage, and P2's window shortens two ranks. The control is exact — at 0 the new binary reproduces
  the previous default's 1598 chunks with an identical SHA over every row, and its scores.
  **The pin moves**: `.engraph-i43-min120` is the shipped arm at 247 files and 1461 chunks,
  `.engraph-i43-min0` is the control, and `eval/ground-truth.json` is re-stamped against the shipped
  arm — P1 7/7 inv 0, P2 5/5 inv 4, P3 4/5 inv 7, **P4 5/8** inv 1, P6 1/1 inv 0, P7 1/1 inv 0, where
  P4's denominator is 8 because two of its members are now one chunk and nothing it returns changed.
  The tables are in this issue's commit. Two limits found and filed: a short **piece** of a split section
  is not a section, so the rule does not see it (**#51**, one row in the corpus), and
  `eval/build-ground-truth.py` had to learn a re-chunk — a tier-1 member now falls back to its anchor,
  and a noise stamp, which carries none, is dropped and reported (**#52**). Four of the five that
  dropped are the same row an ordinal lower and are re-stamped; `hell-moth.md > ## Resources` merged
  into `## Stat Block` and wants a ruling if it ever surfaces
- ~~**#44** a bold-only line is a heading the chunker cannot see~~ — **DONE, and the default is a
  ruling rather than the measurement's reading.** `**Skills**` is structure to a reader
  and a paragraph to `emit_section`, so **33% of the bestiary's structure markers never reach a
  breadcrumb**. The vault cannot separate two candidate guards: all 219 bold-only lines are inside a
  section and all 73 that are not are preamble stat lines, so the content test and a
  "not before the first heading" test agree on every instance. Splitting cannot fix it: `Chunk::from_section` gives later pieces the parent's
  `heading_path` unchanged, and 112 of 1598 continuation chunks carry no breadcrumb their parent did
  not. Probed as a vault edit on a scratch copy: **P7's answer becomes a 363-char `### Spells` chunk
  at rank 1** where it was buried at character 1038 of a 1061-char `## Stat Block`, and **P2's becomes
  `### Notes` at 99.73%** against 98.02%. The choice is vault-authoring against a chunker rule; the
  chunker version generalises to any Obsidian vault in this house style. **#43 unblocked it**: the
  first probe pass destroyed N10's abstention, that was #43's bug, and `chunk_min_chars = 120` put all
  eleven negatives back to the baseline. It is the default now, so #44's arm inherits it.
  **The rule is a content test and it is flat.** One bold span and nothing else — `**Text**`,
  `__Text__`, or either with one colon after the closing marker — so the bestiary's
  `**Rank**: S • **Levels**: …` preamble is not promoted, and the test separates 219 bold-only lines
  from 73 preamble lines exactly. `structure_headings` merges the promoted lines into what
  `markdown::parse_headings` returns, deeper than any `#` heading and an ancestor of nothing, so the
  enclosing heading stays in the breadcrumb, the next promoted line ends the section and the next `#`
  heading of any depth ends it too. The raw line stays at the head of the chunk body, so the keyword
  index reads `**Spells**` and only the breadcrumb reads the stripped text. A promoted line with no
  body of its own is not promoted, and a `#` heading emptied by promotion is carried into that
  promoted section rather than lost — a flat line has no descendants to carry its text.
  **The control is exact.** `.engraph-i44-off` reproduces `.engraph-i43-min120` row for row over one
  SHA-256 of every `(path, seq, heading, heading_path, text)`, after a full re-chunk. Promotion takes
  the corpus from 1461 chunks to 1559 over the same 247 files, adds no row under 120 characters, and
  moves the longest row not at all.
  **Three of the plan's four conditions are met and the second is not.** P2's best rises 97.9795 to
  99.6961 onto a `**Notes**` chunk that reads as the answer alone, P7's falls 98.3475 to 97.7192 onto
  a `**Spells**` chunk a third the size of the stat block that held it and still rank 1, and P1, P2,
  P6 and P7 hold their coverage. **P3 falls 4/5 to 1/5**: six per-creature `**Spells**` rows take
  ranks 10, 12, 14, 15, 17 and 18, and three members that scored 84–96% leave the window. What removes
  them before the cross-encoder sorts is not diagnosed. P3's inversion count falling 7 to 0 is #50's
  artifact and not a gain, and P4's denominator changes so its two figures do not compare.
  It ships `true` by the repository owner's ruling, taken with these numbers in front of them, and the
  P3 regression is to be debugged rather than to block the rule.
  **`CHUNKER_VERSION` is 2 because of the carry rule**, so every store built by an earlier binary
  re-chunks once on its next open, whatever its settings. The final review filed **#53**, **#54** and
  **#55**, and #51 is the fourth defect of the same pass
- **#53** a promoted heading is not addressable by `read --section` or by `update --section` —
  `chunks.heading` holds the raw promoted line, so a result prints
  `lore/bestiary/archdragon.md > **Spells**` and `markdown::find_section` reads ATX headings only.
  `**Spells**` and `Spells` both answer "Section not
  found", and `writer::update_note` fails the same way. **107 of the shipped arm's 1559 chunks** carry a
  promoted heading, `**Abilities**` 67 times and `**Spells**` 22, so 7% of the corpus breaks the
  search-then-read path that is the main route through the MCP server. Three treatments are on the
  issue: teach `find_section` the promoted form, which would then address a construct the writer
  cannot edit as a section, since a promoted section ends at the next promoted line or the next `#`
  heading and that is the chunker's rule and not the parser's; return the addressable ancestor and
  keep the promoted leaf in the breadcrumb, which gives a reader more than the chunk that matched; or
  report it and have the caller read the whole note
- **#54** a bodyless heading above a same-level sibling loses its heading line — `structure_chunk`
  skips a heading with no body of its own, because the text survives in its descendants'
  `heading_path`. That holds for a deeper heading and fails for a sibling: `## A` above `## B` pops
  from the ancestor stack before `B` pushes itself, so `A` is in no chunk's text and in no breadcrumb.
  It predates #44 and happens at every setting of `promote_bold_headings`, since no bold line is
  involved. #44 fixed the neighbouring case and deliberately did not widen the carry, because
  widening it changes the control arm's rows and #44's result rests on that control being exact. **The
  pinned vault does not hold this shape**, so no measurement is affected, and the fix is
  `Action::Reindex`, a `CHUNKER_VERSION` bump and a new chunk-boundary baseline for `eval/probes.md`'s
  corpus counts, its row hash and every reading taken at the pin. What decides it is whether an empty
  heading is worth a row of text at all
- **#55** the setup and apply reindex path chunks at settings the session did not choose —
  `EngraphServer::setup` and `handle_setup` each load a fresh `Config::load().unwrap_or_default()` and
  hand it to `onboarding::run_apply_json`, which runs a full vault index. Neither applies the
  session's captured `chunk_opts`, so the index runs at whatever is on disk, or at the shipped
  defaults if the load fails, while every later single-file `reindex_file` runs at the session's value.
  That is the mixed chunking `ChunkOptions` exists to make impossible, and the chunker keys are
  `chunker_fingerprint` components, so a full index at the wrong settings records the wrong
  fingerprint too and the store then looks consistent. `reindex_file` had the same defect and is
  fixed, at both call sites, with `Config::set_chunk_options`. The wider fix is to stop handing a bare
  `Config` to anything that indexes. No measurement is affected, because every arm was indexed through
  the CLI
- **#41** does the keyword lane's breadcrumb column still earn its default? — **two arms, on against
  off.** The column exists for one case: a chunk whose answering terms are in its heading and not in
  its body. #37 shipped `[fts] heading_path = true` on one reading — it returns `## Level 4 Silence`
  to probe 3, an answer the embedding limb of the same rule dropped. #38 removed that embedding limb,
  and the answer is present without the column, so the recorded gain is gone. Measured against the
  new default the column moves 80 of 360 slots and nothing the instrument can read, except one result
  against it — probe 4 drops `threads/the-archdragon-disguise.md > ## Objectives` at 97.43% and admits
  a demon-knight section. One tail swap on one query is thinner than the pool can carry, so the column
  stays on and this is the evidence owed.
  **#46 changed the string this measures.** The breadcrumb leads with the file path, so folder names
  enter the keyword index on every chunk beneath them, and 0 of the 18 pool queries contain a folder
  term. The pool needs two probes it does not have: one whose answer sits under a heading whose terms
  are absent from the chunk body, and one naming a folder term. **#45 is the instrument that scores
  them, so take it first.** A flag flip is a 0.1 s keyword-index rebuild, so the arms are nearly free
- ~~**#33** compile llama.cpp's CUDA backend, and read the device at load~~ — **DONE.** The `cuda`
  cargo feature is out of `default`, so the CPU build and CI's two legs are unchanged, and
  `llm::device_identity` reads what device the process actually got and folds it into both model
  fingerprints — so swapping between a CPU and a CUDA binary forces a re-embed each way, and a hidden
  GPU is caught by the read path rather than answering from the wrong vector space. It is a 30×
  speedup on the reranker, which is what closed #21, and a new measurement baseline: the kernels are
  not bitwise identical, so a GPU rank table and a CPU one are not comparable. **See "The CUDA build"
  above** for the toolkit install, the four environment variables and the two silent build failures
- **#5** embedding model config — expose output dim, tie max chunk tokens to the model's context window
- **#8** pick a better local embedder — >512 tokens, >768 dim (pairs with #5, which exposes the knobs)
- ~~**#12** embed at the model's native dimension~~ — **done.** Every vector had been truncated to its
  first 256 of 768. The seed probes return identical verdicts at identical ranks and confidences,
  while **76 of 100 slots moved underneath**. Read at the time as the probes being blind; the better
  reading, and the one #14 confirmed by moving them, is that the top of the ranking is robust to this
  change and the churn sat in the tail. Ruled out as the probe 1 explanation (#9). The migration ran
  itself; storage roughly doubles. Optional Matryoshka truncation is deliberately left unbuilt
- ~~**#10** the embedding prompt format is nomic-embed-text's, not EmbeddingGemma's~~ — **DONE.**
  `[embedding_prompt]`, both halves shipping `documented`. The control is the whole basis of the
  reading: the pre-#10 build reproduces #34's tables exactly, and the #10 build at
  `legacy`/`legacy` — after a forced 72.7 s re-embed, because the template is a fingerprint
  component — is **byte-identical over all eighteen calibration queries**.
  **P7 gains the section that answers it.** `lesser-dragon.md > ## Stat Block` holds *"Lesser
  Dragons cannot take human form"*, and #34 recorded it as unreachable on the GPU store at any
  floor. It was not ranked low: at `top_n = 50` the whole 30-candidate shortlist does not contain
  it. Under the documented format it is **rank 1 at 97.6%**. That was filed against #3 and #8 and
  belonged here.
  **Correcting either half retrieves it**, which is the finding that stops this being a story about
  matched pairs: the vector space moved, and that is all the probes can say. 261–292 of 360 result
  slots moved against seven tracked targets — #12's and #14's shape.
  The grid overruled the ticket twice. **The query half is where the rank table improves** (probe 1's
  `archivist-lenne.md` 5 → 4, nothing lost) and the document half **costs probe 2 a rank**, inserting
  an unrelated 91.0% candidate above `## Human Forms`. And **`per_intent` is inert** — two of
  eighteen queries, no tracked target. Both halves ship documented anyway, because two ranks over
  seven targets is inside what this instrument can resolve, the model card specifies a *pair*, and
  #36 could only be measured from the documented document template.
  **`answer_floor` needs no refit**: every cell gives the same 29.64% midpoint, from an unchanged
  6.77% highest rejectable negative and 52.52% lowest correct answer.
  Also fixed in passing: `placement::try_semantic_placement` embedded a note's content with the
  **query** template and compared it against centroids built from **document** vectors. No probe
  measures placement, so it is recorded rather than demonstrated
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
  firing in the baseline configuration all along. (#59 deleted that classifier and that table; the
  weights are `[lane_weights]` and the constant this issue set is their default.) And **the graph
  lane's contribution at 0.8 is not zero but is not evidence of value either**: with intelligence off
  its only appearances across the five probes are two tail slots on probe 1 and two on probe 5, *the
  nonsense control*. A disjoint
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
  Applied on read, so old cache rows were fixed without invalidation. **#59 closed the gap for
  good**: there is no expansion list to repair, because the user's query is the only one run
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
- ~~**#59** remove query expansion and intent classification; put the lane weights in config~~ —
  **DONE.** Gone: `OrchestratorModel`, `LlamaOrchestrator`, `heuristic_orchestrate`,
  `parse_orchestration_json`, `QueryIntent`, `OrchestrationResult`, `ensure_original`,
  `LaneWeights::from_intent`, the `llm_cache` table and its accessors, `models.expand` and the 600 MB
  Qwen3-0.6B download, `QueryTemplate::PerIntent`, and `temporal::parse_date_range_from_json`. The
  temporal lane keeps its date range, which comes from `parse_date_range_heuristic` and was never the
  model's. A store carried across this drops `llm_cache` on migrate.
  **`[lane_weights]` is what replaces the classifier** — `semantic 1.0 / fts 1.0 / graph 0.8 /
  rerank 1.0 / temporal 0.0`, the vector `Exploratory` named and the best cell the #57 grid measured.
  Query-time, reaching no fingerprint, so a sweep is a config edit with no index work. `--explain`
  keeps one retrieval row: the query as run, its FTS MATCH expression, and per-lane hit counts.
  **Verified against the arm home**: 23/29 coverage, 5 inversions, 25 noise, N10 97.8715 and N11
  97.8242 — the seeded cell reproduced with no seeding. Closes #18, #19 and #57
- ~~**#57** the orchestrator's output holds no JSON on 13 of 18 pool queries, and the heuristic
  fallback is cached as the model's~~ — **DONE, and it decided #59.** `LlamaOrchestrator::orchestrate`
  generated 256 tokens, found no `{` in them, and fell to `heuristic_orchestrate` on 13 of the 18 pool
  queries, every time, warning to stderr — after which `search.rs` cached the fallback under
  `model = "orchestrator"` with nothing recording that no model produced it. The cause is the
  template: `format_prompt` ends the prompt at `<|im_start|>assistant\n` and Qwen3-0.6B opens a
  thinking block it never closes, so the budget is spent inside `<think>` before any JSON is written —
  P3's generation is 1157 characters of reasoning and no object. Same family as #32 and #10.
  **Both treatments work and neither matters.** A 1024-token budget parses 18 of 18 at 736 ms
  median, and a `<think></think>` prefill parses 18 of 18 at **175 ms** against the defect's 5 of 18
  at 610 ms. Measured against the
  responsive sets, no expander this pool has been run with — the word-splitter, Qwen, or a hand-written
  set — ever put a tier-1 member in front of the cross-encoder that the user's own query would not
  have, and against the 22-slot shortlist expansion cost six of them by displacement. So the fix was to
  delete the feature.
  **Forty-four arm homes hold a byte-identical `llm_cache`**, `.engraph-i36-*` through
  `.engraph-i46-*`, the `diag-*` homes and `.engraph-eval`. Every table from #36 onward, the
  ground-truth build and the `answer_floor` fit ran under one set of eighteen rows, fourteen of them
  word-splits. Nothing in `--json` records the intent, so a saved pool run carries no trace of what it
  ran under and the cache is the only record. #59 deleted the table, so those homes are the last
  stores that hold one
- ~~**#58** N10's abstention is the shortlist gate, not the ranking~~ — **DONE, diagnosed and
  reproduced three ways.** N10, *What spell can be used to clean clothing?*, is the near negative the
  chunk minimum was fitted against and shipping condition one of both #43 and #44: "N10 stays at
  1.6135". It stayed there only while the content lanes got 22 shortlist slots. The cross-encoder
  scores `rules/restoration-spells.md > ## Level 5 Purify Body` **0.98** for that query every time it
  is shown it, and `graph_reserve = 0`, `candidates = 90`, dropping the expansions, or lane weights of
  `semantic 1.2 / fts 0.8` each surface it at 97.87%. Widening the gate is one half of the cause: the
  eight slots go to a graph reserve that reaches the output on no probe. Reducing what competes for
  the 22 is the other, and the competitor was the word-split branch — 68 fused candidates with no
  expansion against 409 with the seven word-splits, and 1.61% appears in four of seventeen #57 cells,
  all four word-splits at 22 slots. **After #59 the "N10 stays at 1.6135" condition is dead rather
  than unmeaning.**
  Truncating the lanes before fusion cannot help — the chunk is semantic rank 1 with no keyword-lane
  presence, as P3's three lost members are semantic 2, 3 and 4 with none, so for any non-negative
  weights and any cut, every configuration that admits them admits this first.
  **"No floor can survive it" is narrower than the ticket read.** It holds for Qwen3-Reranker-0.6B and
  not for the model class: the 4B scores the wrong chunk 73.87 against the right one's 86.88 and opens
  a band that separates them, where the 0.6B reads 97.87 against 91.74. It does not solve the general
  case — N11 reads 97.87% under both models, above every positive in the pool — which is #4
- ~~**#19** intent classification looks inverted on two probes~~ — **closed by #59, which deleted the
  classifier.** It selected a weight vector that had never been swept, and twelve of the eighteen pool
  queries took its default branch anyway. The weights are now `[lane_weights]`, one configured vector
- ~~**#18** query expansion splits on words against a 16-item stopword list~~ — **closed by #59, which
  deleted the branch.** The list held no modals and no verbs, so `dragon that can take human form`
  became seven expansions including `that` and `can`. Seventeen pool cells measured the whole feature:
  no expander ever put a tier-1 member in front of the cross-encoder that the query itself would not
  have, and against a 22-slot shortlist expansion cost six of them by displacement
- **#17** resolve queries against a tag/alias registry and expand on what they name. Aliases are
  parsed at index time and **thrown away** — there is no registry to resolve against — and tags reach
  retrieval only as a yes/no admission test on graph expansion. Hangs on the expansion slot rather
  than a new lane because every expansion runs through *both* content lanes, and in RRF a second
  lane's vote outweighs any position within one. Tags resolve semantically (a `tag_centroids` table,
  the same machinery as `folder_centroids`), aliases lexically (`links.rs`'s fuzzy name matching).
  Replaces the withdrawn FTS-injection experiment. Probe 4 is the guard, probe 5 the control
- ~~**#21** multi-sequence decode~~ — **closed: the GPU took the latency the batch was for.** #33's
  CUDA backend is a 30× speedup on the reranker, which is the whole of what this was going to buy.
  The diagnosis stands if a CPU build ever needs the throughput again — phase 2 of #13, which built
  the batch *API* and then looped inside
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
  DEBUG. Knock-on: `LaneWeights::from_intent` is meaningful for the first time — a symbol #59 deleted,
  so the knock-on now reads on `[lane_weights]`
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
  **Three tracked targets up, one down, two hold — the best single move on record since #6.** Probe 2's exact-answer section goes 6 → 1; probe 1, the acceptance criterion, improves 5 → 2;
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
  18% of the text, and the two ranks #14 won both hold.
  **#42 took the cap back off.** The 12% was a CPU number; on CUDA the same cap saves 3.6%
- ~~**#42** the cross-encoder's character cap is a CPU-era trade, and it is cutting answers off
  chunks~~ — **DONE.** `max_document_chars` ships at **0**, unlimited.
  **The cap was truncating answers, not padding.** Probe 7's answering sentence starts at character
  1038 of a 1061-character chunk, and probe 2's best answer at 1098 of 1207 — so the cross-encoder
  scored both on their subject and never read the line that answers the question. Across the vault,
  333 of 1598 chunks are over the cap and **14.8% of the corpus never reached the cross-encoder**.
  **The latency it bought is gone.** Eighteen queries, one warm server per arm, three interleaved
  rounds, `min` per query: 8.745 s capped against 9.064 s uncapped. **3.6%, or 18 ms on a 490 ms
  query**, against #25's 12.3% on the CPU build.
  **Probe 2's two lost ranks are the instrument failing, not the arm.** #25 and #30 both recorded
  `archdragon.md > ## Human Forms` dropping from rank 4 when the cap comes off, and both read it as a
  cost. Its *score* is unchanged at 80.73%; it drops because two previously truncated chunks moved
  above it, and both answer the query — `medium-dragon.md > ## Stat Block` at 98.02% answers it
  better, naming the youngest dragons able to take human form. **A tracked target's rank falls the
  same way when a better result passes it as when a worse one does.** Probe 7 shows the same shape
  with the sign visible: rank 1 held, score 97.56% → 98.35%.
  **The cost is the two negatives #34 already records as unrejectable**, N4 and N11, which gain two
  above-floor results each. Every other negative and probe 5 are identical, best score and every
  slot. `answer_floor` needs no refit: the same 29.64% midpoint.
  The key stays settable, because #25's trade is still real on a CPU build. **Every table recorded
  before #42 was measured at cap 1000**, so rank tables from either side of it are not comparable.
  **#45 measured this as the largest gain on record.** P2 holds five responsive chunks and every arm
  returns all five, so coverage does not separate the arms and the ranks do. Three of the five score
  **1.36% to 1.83% under the cap and 78.76% to 98.02% without it**. At the shipped floor the capped
  arms answer P2 with one of the five and the uncapped arm answers with four
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
models-loaded-only bug. (#59 deleted the orchestrator and the weight table both.) And graph's
contribution at 0.8 is not quite "invisible": it is two tail
slots on probe 1 and two on probe 5, *the nonsense control*, which is what a disjoint set does when
the content lanes have nothing to say. The full audit is in #9 and its commit.

The section-per-file transform's win is no longer mysterious either: its sections are their own
files, dense enough in "temple" for **both** content lanes to find them, which is the one
configuration that clears the graph block. Now confirmed from the other direction — with the weight
corrected, in-place chunking returns the same file at the same ranks — so `eval/section-split.py`
has no open question left and can be retired.

`eval/` holds the probes and the harnesses: `pool.sh` runs the eighteen-query calibration pool and
keeps the JSON, `probe.sh` says what came back for one query, `bench-search.sh` times one query in a
warm server and `bench-pool.sh` times the whole pool the same way. Use `bench-pool.sh` for anything
that touches the rerank lane: a cap or a candidate count does nothing for a query whose candidates
are already short, and an average over five probes hides that.
