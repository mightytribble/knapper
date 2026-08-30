# Deployment guide

How to install knapper and connect it to an MCP client (Claude Code) on your
own machine. Three install tiers, one per target environment:

| Tier | How you run knapper | GPU |
|---|---|---|
| macOS (Apple Silicon) | native binary | Metal |
| Linux / WSL2 + NVIDIA | Docker, `:cuda` tag | CUDA |
| Linux x86_64, CPU only | Docker, `:cpu` tag | none |

One vault per install. Re-indexing a different vault path replaces the
active one; running several vaults side by side needs a separate data
directory per vault (`--data-dir` or `KNAPPER_HOME`) and is not covered here.

**What's live today:**

- The from-source build (macOS) and the local Docker build (Linux/WSL2) work
  now, from this repository.
- `brew install mightytribble/tap/knapper` and `docker pull
  ghcr.io/mightytribble/knapper:...` are not live yet — the tap and the
  container registry push are a separate, gated release step. Both sections
  below show the interim path and note where the future one-liner goes once
  it ships.
- The `git clone https://github.com/mightytribble/knapper` commands below
  assume the repository is public, which is part of that same gated step.
  If you're reading this before it lands, you already have a checkout —
  this file lives in it.

## macOS (Apple Silicon, Metal)

**Intended path**, once the tap is live:

```bash
brew install mightytribble/tap/knapper
knapper configure --enable-intelligence
knapper index ~/Vault
```

**Interim path (works now): build from source.**

Requires [CMake](https://cmake.org/) — `llama.cpp`'s C++ layer builds from
it — and a Rust toolchain.

```bash
git clone https://github.com/mightytribble/knapper
cd knapper
cargo build --release
```

Metal is auto-detected on macOS; no build flag is needed. The binary lands
at `target/release/knapper`. Put it on your `PATH`, or call it by its full
path in the steps below.

**Enable the reranker (optional, ~650 MB download).** Search runs without it;
this adds the cross-encoder lane:

```bash
knapper configure --enable-intelligence
```

**Index your vault:**

```bash
knapper index ~/Vault
# Downloads the embedding model on first run (~300 MB, or ~950 MB total
# with intelligence enabled). Incremental after that — only changed files
# re-embed.
```

**Connect to Claude Code.**

Recommended — register it with the CLI:

```bash
claude mcp add --scope user knapper -- knapper serve
```

`--scope user` registers knapper for every project; `--scope project`
writes a shared `.mcp.json` in the current project's root instead.

Manual alternative — add the same server definition directly to
`~/.claude.json` (user scope) or a project's `.mcp.json` (project scope).
Claude Code does not read MCP servers from `~/.claude/settings.json` —
that file is for permissions, hooks, environment variables, and the model
choice only (see the [MCP quickstart](https://code.claude.com/docs/en/mcp-quickstart)):

```json
{
  "mcpServers": {
    "knapper": {
      "type": "stdio",
      "command": "knapper",
      "args": ["serve"]
    }
  }
}
```

If you built from source and `knapper` is not on your `PATH`, use the full
path to the binary as `command` instead.

## Linux / WSL2 + NVIDIA (CUDA)

**Prerequisites:**

- [Docker](https://docs.docker.com/engine/install/)
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), so `docker run --gpus all` can reach your GPU.

**Two rules, both load-bearing:**

- **Mount the vault at the same container path for both `index` and
  `serve`** — `/vault` in the examples below. `serve` reads the vault path
  that was baked into the store at index time; mount it anywhere else and
  `serve` looks for a vault that isn't there.
- **Keep the vault on the WSL Linux filesystem, not `/mnt/c`.** The
  real-time file watcher uses inotify, and inotify events do not reliably
  cross from a Windows-filesystem vault into a Linux container — edits made
  outside the container go unseen until you re-index by hand. Keep the
  vault under your Linux home (e.g. `~/vault`), not under `/mnt/c/Users/...`.

**Get the image.**

The published image (`ghcr.io/mightytribble/knapper:cuda`) is not pushed
yet. Until it is, build it locally from a checkout of this repository:

```bash
git clone https://github.com/mightytribble/knapper
cd knapper
docker build --build-arg VARIANT=cuda -t knapper:cuda .
```

Building the image needs only Docker and network access — the CUDA
toolkit and `nvcc` live inside the `nvidia/cuda:12.6.3-devel-ubuntu24.04`
build stage, not on your host. (The NVIDIA Container Toolkit above is a
run-time requirement, for `docker run --gpus all`; it is not needed to
build the image.) Once the image is published, this step becomes:

```bash
docker pull ghcr.io/mightytribble/knapper:cuda
```

— substitute `ghcr.io/mightytribble/knapper:cuda` for `knapper:cuda` in the
commands below when you switch to the published tag.

**Create a data volume.** This holds the SQLite store and the downloaded
models, so they survive between container runs:

```bash
docker volume create knapper-data
```

**Index your vault.** This also downloads the models into the volume on
first run:

```bash
docker run --rm --gpus all \
  -v knapper-data:/data \
  -v /home/you/vault:/vault \
  knapper:cuda index /vault
# Downloads the embedding model on first run (~300 MB). Incremental after
# that — only changed files re-embed. Enabling intelligence, below, adds
# another ~650 MB (~950 MB total).
```

Replace `/home/you/vault` with your vault's real path (on the WSL Linux
filesystem, per the rule above).

This first run may flash an `Enable AI-powered search intelligence? [y/N]`
prompt. `index` doesn't run with `-i`, so the prompt gets EOF on stdin
immediately and defaults to No without hanging — indexing proceeds
normally. Enable intelligence explicitly in the next step if you want it.

**Enable the reranker (optional, ~650 MB download):**

```bash
docker run --rm --gpus all \
  -v knapper-data:/data \
  knapper:cuda configure --enable-intelligence
```

The `:cuda` binary is dynamically linked against the NVIDIA driver
(`libcuda.so.1`), so it needs `--gpus all` on every invocation, including
`configure`. `configure` loads no model, but the binary cannot start
without the driver, so a `:cuda` command without `--gpus all` fails with
`libcuda.so.1: cannot open shared object file`. The download happens during
this call itself; there is no need to re-index. The cross-encoder lane is
active on your next `search` or `serve`.

**Connect to Claude Code.**

Recommended — register it with the CLI:

```bash
claude mcp add --scope user knapper -- docker run --rm -i --gpus all -v knapper-data:/data -v /home/you/vault:/vault knapper:cuda serve
```

Everything after `--` is the exact command Claude Code runs to start the
server. For a Docker install that is the whole `docker run … serve` line
above, **not `knapper serve`**. Registering `-- knapper serve` (as the
macOS tier does) points Claude Code at a native binary on your `PATH`,
which a Docker install does not provide, and which would read a different
data directory than the `knapper-data` volume you indexed, so it never
starts the container. Keep the image tag, the `knapper-data` volume, and
the `/vault` mount exactly as in the `index` step above.

`--scope user` registers it for every project; `--scope project` writes a
shared `.mcp.json` in the current project's root instead.

Manual alternative — add the same server definition directly to
`~/.claude.json` (user scope) or a project's `.mcp.json` (project scope),
not `~/.claude/settings.json` (see the note in the macOS section above):

```json
{
  "mcpServers": {
    "knapper": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--gpus", "all",
        "-v", "knapper-data:/data",
        "-v", "/home/you/vault:/vault",
        "knapper:cuda",
        "serve"
      ]
    }
  }
}
```

Use the same `/home/you/vault` path you indexed with. Once the published
image is available, replace `knapper:cuda` with
`ghcr.io/mightytribble/knapper:cuda` here too (in both the `claude mcp
add` command and the JSON).

`-i` keeps stdin open and is required — it's how the MCP stdio handshake
reaches the container. `--rm` cleans up the container when the client
disconnects; Claude Code starts a fresh one per session.

## Linux x86_64 (CPU)

Same Docker path as the CUDA tier, without `--gpus all` and with the `:cpu`
tag. The two rules above (same mount path for index and serve, vault on the
Linux filesystem under WSL2) apply here too.

**Get the image.**

The published image (`ghcr.io/mightytribble/knapper:cpu`) is not pushed
yet. Until it is, build it locally from a checkout of this repository:

```bash
git clone https://github.com/mightytribble/knapper
cd knapper
docker build --build-arg VARIANT=cpu -t knapper:cpu .
```

This needs only Docker and network access — no GPU toolkit is involved on
this tier. Once the image is published, this step becomes:

```bash
docker pull ghcr.io/mightytribble/knapper:cpu
```

— substitute `ghcr.io/mightytribble/knapper:cpu` for `knapper:cpu` in the
commands below when you switch to the published tag.

**Create a data volume:**

```bash
docker volume create knapper-data
```

**Index your vault:**

```bash
docker run --rm \
  -v knapper-data:/data \
  -v /home/you/vault:/vault \
  knapper:cpu index /vault
# Downloads the embedding model on first run (~300 MB). Incremental after
# that — only changed files re-embed. Enabling intelligence, below, adds
# another ~650 MB (~950 MB total).
```

This first run may flash an `Enable AI-powered search intelligence? [y/N]`
prompt. `index` doesn't run with `-i`, so the prompt gets EOF on stdin
immediately and defaults to No without hanging — indexing proceeds
normally. Enable intelligence explicitly in the next step if you want it.

**Enable the reranker (optional, ~650 MB download):**

```bash
docker run --rm \
  -v knapper-data:/data \
  knapper:cpu configure --enable-intelligence
```

`configure` only downloads a file — it loads no model. The download
happens during this call itself; there is no need to re-index. The
cross-encoder lane is active on your next `search` or `serve`.

**Connect to Claude Code.**

Recommended — register it with the CLI:

```bash
claude mcp add --scope user knapper -- docker run --rm -i -v knapper-data:/data -v /home/you/vault:/vault knapper:cpu serve
```

Everything after `--` is the exact command Claude Code runs to start the
server. For a Docker install that is the whole `docker run … serve` line
above, **not `knapper serve`**. Registering `-- knapper serve` (as the
macOS tier does) points Claude Code at a native binary on your `PATH`,
which a Docker install does not provide, and which would read a different
data directory than the `knapper-data` volume you indexed, so it never
starts the container. Keep the image tag, the `knapper-data` volume, and
the `/vault` mount exactly as in the `index` step above.

`--scope user` registers it for every project; `--scope project` writes a
shared `.mcp.json` in the current project's root instead.

Manual alternative — add the same server definition directly to
`~/.claude.json` (user scope) or a project's `.mcp.json` (project scope),
not `~/.claude/settings.json` (see the note in the macOS section above):

```json
{
  "mcpServers": {
    "knapper": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "knapper-data:/data",
        "-v", "/home/you/vault:/vault",
        "knapper:cpu",
        "serve"
      ]
    }
  }
}
```

Use the same `/home/you/vault` path you indexed with. Once the published
image is available, replace `knapper:cpu` with
`ghcr.io/mightytribble/knapper:cpu` here too (in both the `claude mcp add`
command and the JSON).

## Verifying the install

Restart Claude Code after registering the server, then ask it to search
your vault. If the MCP server isn't reachable, run the `serve` command
from your config directly in a terminal and check what it prints — a
container that can't find its vault, or a missing GPU driver, shows up
there before Claude Code ever sees it.

## Data and models

Everything knapper stores — the SQLite index, downloaded GGUF models, vault
profile, and config — lives in one data directory: `~/.knapper` for a native
install, or the `/data` mount (backed by the `knapper-data` volume) for a
container. No API keys, no cloud calls — search, indexing, and inference all
run against local files and local models.
