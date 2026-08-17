# Building

Ubuntu 24.04 / WSL2, **no sudo required**. `llama-cpp-sys-2` compiles llama.cpp from source, which
is where all the friction lives — the Rust itself needs nothing special.

```bash
# 1. Rust (durable, user-local)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# 2. cmake + libclang (Ubuntu 24.04 blocks system pip, so use a venv)
python3 -m venv ~/.knapper-buildenv
~/.knapper-buildenv/bin/pip install cmake libclang

# 3. Build
export PATH="$HOME/.cargo/bin:$HOME/.knapper-buildenv/bin:$PATH"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export LIBCLANG_PATH="$HOME/.knapper-buildenv/lib/python3.12/site-packages/clang/native"
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include -I/usr/include -I/usr/include/x86_64-linux-gnu"

cargo build --release        # ~10 min cold, ~20s incremental
cargo test --lib             # 922 pass
```

Each env var exists for a specific failure. Omit one and you get:

| omitted | failure |
|---|---|
| `CMAKE_POLICY_VERSION_MINIMUM` | cmake 4.x rejects llama.cpp's older `cmake_minimum_required` |
| `LIBCLANG_PATH` | `Unable to find libclang` — bindgen can't run |
| `BINDGEN_EXTRA_CLANG_ARGS` | `fatal error: 'stdbool.h' file not found` (pip libclang ships the .so, not clang's builtin headers) |

Adjust the gcc version in the include path (`13`) and the python version (`python3.12`) to match the box.

## The CUDA build (issue #33)

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
    --toolkitpath="$HOME/.knapper-cuda" --no-man-page --override --tmpdir=/some/large/tmp

# 2. Build (the four vars above still apply)
export PATH="$HOME/.knapper-cuda/bin:$PATH"
export CUDAToolkit_ROOT="$HOME/.knapper-cuda"   # find_package(CUDAToolkit)
export CUDA_LIBRARY_PATH="$HOME/.knapper-cuda"  # the linker search path — see below
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

## There is no `tests/` directory

This fork deleted `tests/integration.rs`, `tests/write_pipeline.rs` and `tests/fixtures/`. They had
not compiled since upstream v1.0.0 — `unresolved import engraph::embedder`, `engraph::hnsw`, and a
`walk_vault` arity mismatch — so `cargo test` (full) and `cargo clippy --all-targets` both failed on
pristine upstream, and every test in them was `#[ignore]` behind a GGUF download. `integration.rs`
also reimplemented the index and search pipeline in its own helpers, so repairing it would have
asserted against a copy of the shipped code rather than against the code. The one behaviour with no
twin in the lib suite, the mtime conflict, moved to `writer::tests` and runs on `MockLlm`.

**A rebase onto upstream brings all three paths back.** Delete them again. Upstream PR #47 repairs
them instead, if that is ever the better answer.

## CI is manual-only in this fork

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
