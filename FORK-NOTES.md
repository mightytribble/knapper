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
cargo test --lib             # 530 pass
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
Upstream PR #47 addresses them. Use `cargo test --lib` (530 tests) as the working suite.
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
  `embeddinggemma-300M` at `target_dim=256`. Upstream PR #48 fixes it.
- **Intelligence is not a quality dial.** Enabling it (query expansion + Qwen3 reranker, 1.6GB)
  *regressed* exact-name lookup in testing. Treat on/off as distinct configurations.
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
- **#3** retrieval eval battery — now adjudicates #2, and calibrates #4
- **#4** relevance floor — configurable per-lane min scores so nonsense queries return nothing
- **#5** embedding model config — expose output dim, tie max chunk tokens to the model's context window
- **#8** pick a better local embedder — >512 tokens, >768 dim (pairs with #5, which exposes the knobs)
- **#9** section-per-file still beats in-place on probe 1 — #6 ruled out granularity, #2 ruled out
  document identity in the vector
- ~~**#6** section-level retrieval granularity — fuse on `(file_id, seq)` so a document can
  contribute more than one section~~ — **done**, probe 3 now returns the correct section and every
  result names its heading. Did **not** fix probe 1, so `eval/section-split.py` cannot be retired
  and the Path A / Path B question stays open.
- ~~**#7** exclude derived `*-index.md` / `templates/` from ingest, and make `exclude` glob for
  real~~ — **done**, 14.2% of chunks and 18.3% of edges removed from the eval corpus

Suggested order: **#3 next**, and now with a concrete debt to pay off. Five hand-picked probes were
enough to show that #2 trades one probe for another, and not enough to say which trade is right —
the same five gave contradictory verdicts on three configurations of the same feature. #3 supplies
the measurements everything else is judged by. Then **#5** (makes chunk size and dim configurable,
which #3 needs to compare configurations), then #4 (needs #3's negative controls to calibrate).

**Probe 1 is the open question #6 was expected to close** (now #9). The section-per-file transform
puts `temple-of-the-architect` at ranks 1–4; indexing in place leaves it at 21 before #6, after #6,
and under all three prefix configurations of #2. Two explanations are now ruled out: the ranking
unit (#6) and document identity in the vector (#2). What is left of the transform's difference is
that each section became a whole document — its own embedding computed over that text alone, its own
docid, its own graph node. It needs #3 to diagnose against more than one query.

`eval/` holds the seed material for #3.
