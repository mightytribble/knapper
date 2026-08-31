# Deployment guide

How to install knapper and connect it to an MCP client (Claude Code) on your
own machine. Three install tiers, one per target environment:

| Tier | How you run knapper | GPU |
|---|---|---|
| macOS (Apple Silicon) | native binary | Metal |
| Linux x86_64, CPU only | native binary, or Docker `:cpu` | none |
| Linux / WSL2 + NVIDIA (x86_64) | Docker `:cuda`, or a source build | CUDA |

**On Linux, Docker is an option and not a requirement.** Every release
carries a native `knapper-linux-x86_64` binary, and Homebrew installs that
same binary on Linux — either is a simpler install than a container, and
the Docker sections below are written for people who want one. If you want
GPU offload, use the `:cuda` image or a source build with
`--features cuda`.

The container images are **x86_64 only** — `linux/amd64`, with no arm64
variant — so Docker Desktop on Windows works on an Intel or AMD machine
and **Windows on ARM is not supported**. Apple Silicon is served by the
native macOS binary, not by a container.

One vault per install. Re-indexing a different vault path replaces the
active one; running several vaults side by side needs a separate data
directory per vault (`--data-dir` or `KNAPPER_HOME`) and is not covered here.

## macOS (Apple Silicon, Metal)

**Homebrew.** On Apple Silicon this installs the released binary. On an
Intel Mac there is no published binary, so the formula builds from source
and pulls in CMake and Rust as build dependencies:

```bash
brew install mightytribble/tap/knapper
```

**Or take the release binary**, which is already compiled:

```bash
curl -sL https://github.com/mightytribble/knapper/releases/latest/download/knapper-macos-arm64.tar.gz | tar xz
```

**Or build it yourself**, which is how you run something newer than the
last release. Requires [CMake](https://cmake.org/) — `llama.cpp`'s C++
layer builds from it — and a Rust toolchain.

```bash
git clone https://github.com/mightytribble/knapper
cd knapper
cargo build --release
```

Metal is auto-detected on macOS; no build flag is needed. A self-built
binary lands at `target/release/knapper`. However you installed it, put it
on your `PATH` or call it by its full path in the steps below.

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

**Two cautions:**

- **Mount the vault at the same container path for both `index` and
  `serve`** — `/vault` in the examples below. `serve` reads the vault path
  that was baked into the store at index time; mount it anywhere else and
  `serve` looks for a vault that isn't there.
- **Prefer the WSL Linux filesystem over `/mnt/c`.** Edits are still seen
  either way: inotify does not cross from a Windows-filesystem vault into a
  Linux container, so the watcher detects that mount and polls instead
  (`[watcher] backend`, default `auto`). What a Windows-side vault costs is
  speed — every read crosses the 9p/virtiofs boundary, which slows indexing
  and makes each poll pass more expensive. Keep the vault under your Linux
  home (e.g. `~/vault`) when you can.

**Get the image.**

A published image exists:

```bash
docker pull ghcr.io/mightytribble/knapper:cuda
docker tag ghcr.io/mightytribble/knapper:cuda knapper:cuda
```

`:cuda` tracks the latest release and `:0.9.1-cuda` pins one. The second
line retags it locally, so every command below can read `knapper:cuda`
whichever way you got the image. To run something newer than the last
release, build from a checkout instead:

```bash
git clone https://github.com/mightytribble/knapper
cd knapper
docker build --build-arg VARIANT=cuda -t knapper:cuda .
```

Building the image needs only Docker and network access — the CUDA
toolkit and `nvcc` live inside the `nvidia/cuda:12.6.3-devel-ubuntu24.04`
build stage, not on your host. (The NVIDIA Container Toolkit above is a
run-time requirement, for `docker run --gpus all`; it is not needed to
build the image.)

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

Use the same `/home/you/vault` path you indexed with.

`-i` keeps stdin open and is required — it's how the MCP stdio handshake
reaches the container. `--rm` cleans up the container when the client
disconnects; Claude Code starts a fresh one per session.

## Linux x86_64 (CPU)

Two ways in: the native binary, or the `:cpu` container.

**Native binary — no Docker involved:**

```bash
curl -sL https://github.com/mightytribble/knapper/releases/latest/download/knapper-linux-x86_64.tar.gz | tar xz
```

`brew install mightytribble/tap/knapper` installs the same binary. Put it
on your `PATH` and the commands are the ones in the macOS section above —
`knapper index`, `knapper configure`, `claude mcp add` — none of which is
macOS-specific apart from Metal.

**Or run it in a container**, which is the rest of this section: the same
Docker path as the CUDA tier, without `--gpus all` and with the `:cpu` tag.
The two rules above (same mount path for index and serve, vault on the
Linux filesystem under WSL2) apply here too.

**Get the image.**

A published image exists:

```bash
docker pull ghcr.io/mightytribble/knapper:cpu
docker tag ghcr.io/mightytribble/knapper:cpu knapper:cpu
```

`:cpu` tracks the latest release, `:0.9.1-cpu` pins one, and `:latest` is
an alias for `:cpu`. The second line retags it locally, so every command
below can read `knapper:cpu` whichever way you got the image. To run
something newer than the last release, build from a checkout instead:

```bash
git clone https://github.com/mightytribble/knapper
cd knapper
docker build --build-arg VARIANT=cpu -t knapper:cpu .
```

This needs only Docker and network access — no GPU toolkit is involved on
this tier.

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

Use the same `/home/you/vault` path you indexed with.

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
