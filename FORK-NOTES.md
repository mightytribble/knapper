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
cargo test --lib             # 469 pass
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
`unresolved import engraph::embedder`, `engraph::hnsw`. **These are broken on pristine upstream** —
verify with `git stash && cargo test --test integration`. Upstream PR #47 addresses them.
Use `cargo test --lib` (469 tests) as the working suite.

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

## Open work

See issues on this repo:

- **#1** structure-first chunking (section → sub-section → paragraph → size)
- **#2** contextual embedding prefix (filename / heading path / tags into every chunk)
- **#3** retrieval eval battery — gates #1 and #2

`eval/` holds the seed material for #3.
