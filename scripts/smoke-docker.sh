#!/usr/bin/env bash
# Container smoke. Builds the image, disables intelligence (so `index` never
# prompts and only the embed model downloads), indexes a fixture vault into a
# named volume, checks the MCP tools/list over `docker run -i … serve`, and
# confirms the volume persists the index across runs.
#
# CPU by default. Pass `cuda` as $1 on the dev box to build --features cuda and
# add --gpus all (needs the NVIDIA Container Toolkit).
set -euo pipefail

VARIANT="${1:-cpu}"
IMAGE="knapper:${VARIANT}"
VOL="knapper-smoke-data"
WORK="$(mktemp -d)"
VAULT="$WORK/vault"
GPU_ARGS=()
[ "$VARIANT" = "cuda" ] && GPU_ARGS=(--gpus all)

cleanup() { docker volume rm -f "$VOL" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

mkdir -p "$VAULT"
printf '# Dragon\nA lesser dragon breathes fire.\n' > "$VAULT/dragon.md"
printf '# Silence\nThe silenced target cannot cast spells.\n' > "$VAULT/silence.md"

echo "== build $IMAGE =="
docker build --build-arg "VARIANT=$VARIANT" -t "$IMAGE" .

docker volume create "$VOL" >/dev/null

echo "== disable intelligence (keeps index non-interactive) =="
docker run --rm -v "$VOL:/data" "$IMAGE" configure --disable-intelligence

echo "== index (first run; downloads the embed model into the volume) =="
docker run --rm "${GPU_ARGS[@]}" \
    -v "$VOL:/data" -v "$VAULT:/vault" \
    "$IMAGE" index /vault

echo "== MCP tools/list over serve =="
REQ='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
OUT="$(printf '%s\n' "$REQ" | timeout 60 docker run --rm -i "${GPU_ARGS[@]}" \
    -v "$VOL:/data" -v "$VAULT:/vault" \
    "$IMAGE" serve 2>/dev/null || true)"
for tool in search read list tags vault_map \
            create update delete move archive \
            index reindex_file status health identity \
            init migrate; do
    echo "$OUT" | grep -q "\"$tool\"" \
        || { echo "FAIL: tool '$tool' missing from tools/list"; exit 1; }
done

echo "== persistence: a second index reuses the volume =="
OUT2="$(docker run --rm "${GPU_ARGS[@]}" \
    -v "$VOL:/data" -v "$VAULT:/vault" \
    "$IMAGE" index /vault)"
echo "$OUT2" | grep -qE '0 new' \
    || { echo "FAIL: second index re-processed files; volume not persisted"; exit 1; }

echo "PASS ($VARIANT)"
