# ChatGPT Actions

> **Untested in v0.9.** The API half of this path works, but the GPT import step is known to fail on the shipped OpenAPI spec (two endpoint descriptions exceed ChatGPT's 300-character cap), and the setup flow still references the retired plugin-manifest format. The fixes are tracked in [#87](https://github.com/mightytribble/knapper/issues/87) for v1. The HTTP API itself ([http-rest-api.md](http-rest-api.md)) is supported.

Connect your Obsidian vault to ChatGPT as a custom GPT Action. ChatGPT can search, read, create, and edit your notes through knapper's REST API.

## Prerequisites

- knapper installed and indexed (`knapper index ~/your-vault`)
- A tunnel tool: [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/) (recommended) or [ngrok](https://ngrok.com)

## Step 1: Configure knapper

```bash
# Interactive setup — enables HTTP, creates API key, sets CORS
knapper configure --setup-chatgpt
```

Or configure manually in `~/.knapper/config.toml`:

```toml
[http]
enabled = true
port = 3000
host = "127.0.0.1"
rate_limit = 60
cors_origins = ["https://chat.openai.com", "https://chatgpt.com"]

[[http.api_keys]]
key = "kn_your_key_here"    # generate with: knapper configure --add-api-key --key-name chatgpt --key-permissions write
name = "chatgpt"
permissions = "write"        # "read" for search-only, "write" to also create/edit notes

[http.plugin]
name = "My Vault"
description = "Search and manage my Obsidian vault"
public_url = "https://your-tunnel-url.trycloudflare.com"   # set after starting tunnel
```

## Step 2: Start knapper + tunnel

**Terminal 1 — knapper HTTP server:**
```bash
knapper serve --http
```

**Terminal 2 — Cloudflare tunnel:**
```bash
cloudflared tunnel --url http://localhost:3000
# Prints a URL like: https://abc-xyz.trycloudflare.com
```

Or with ngrok:
```bash
ngrok http 3000
# Prints a URL like: https://abc123.ngrok-free.app
```

## Step 3: Update config with tunnel URL

Edit `~/.knapper/config.toml` and set `public_url` to your tunnel URL:

```toml
[http.plugin]
public_url = "https://abc-xyz.trycloudflare.com"
```

Then restart knapper (`Ctrl+C` and re-run `knapper serve --http`). This ensures the OpenAPI spec points to the correct public URL.

## Step 4: Verify endpoints

```bash
# Both should return JSON (no auth required)
curl https://your-tunnel-url/openapi.json
curl https://your-tunnel-url/.well-known/ai-plugin.json

# Search with auth
curl -X POST -H "Authorization: Bearer kn_your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "test search"}' \
  https://your-tunnel-url/api/search
```

## Step 5: Register in ChatGPT

1. Go to [ChatGPT](https://chat.openai.com) → **Explore GPTs** → **Create**
2. Give your GPT a name (e.g., "Vault Assistant")
3. Add these **Instructions**:

```
You are a knowledge assistant connected to the user's Obsidian vault via knapper.

WORKFLOW:
1. Use searchVault to find relevant notes before answering questions
2. Use readNote for full content, and its section parameter for one heading
3. Use getVaultMap to orient yourself in the vault structure
4. Only create or edit notes when explicitly asked

SEARCH TIPS:
- Temporal queries ("last week", "yesterday") activate time-aware search automatically
- Results include confidence % — prefer higher confidence matches
- Fuzzy matching works: typos in names are handled

STYLE:
- Reference vault notes by name when answering
- Quote relevant snippets
- If information isn't in the vault, say so clearly
- Be concise
```

4. Click **Add Action** → **Import from URL**
5. Enter: `https://your-tunnel-url/openapi.json`
6. Click the **gear icon** next to Authentication
7. Select **API Key**, Auth Type: **Bearer**
8. Paste your API key (the `kn_...` key from Step 1)
9. **Save** and test

## Conversation starters

- "What happened in my vault last week?"
- "Summarize my current work projects"
- "Find notes related to [topic]"
- "Create a note about today's meeting with [person]"

## Good to know

- **Tunnel URLs are temporary** (Cloudflare quick tunnels change on restart). For persistent URLs, set up a [named Cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/) or use ngrok with a reserved domain.
- **Read-only mode**: set `permissions = "read"` on the API key if you don't want ChatGPT to create or modify notes.
- **Rate limiting**: default is 60 requests/minute per key. Adjust `rate_limit` in config if needed.
- **knapper must be running** on your machine for ChatGPT to access it. If you close the terminal, the connection drops.
