# HTTP REST API

`knapper serve --http` adds a REST API alongside the MCP server, exposing the
same capabilities over HTTP for web agents, scripts and integrations.

```bash
knapper serve --http                        # port 3000
knapper serve --http --port 8080 --host 0.0.0.0
knapper serve --http --no-auth              # local dev only, 127.0.0.1
```

## Endpoints

Every capability is one route, and the route is the CLI command's name under
`/api/`. [surfaces.md](surfaces.md) is the generated table of all three
surfaces; [faq.md](faq.md) and [configuration.md](configuration.md) explain
what the parameters mean.

| Method | Endpoint | Permission | Description |
|--------|----------|------------|-------------|
| GET | `/api/health-check` | read | Server health check |
| POST | `/api/search` | read | Hybrid search (semantic + FTS5 + graph + reranker + temporal), scoped by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`), and `property`, `links_to`, `linked_from` (one value each) |
| POST | `/api/match` | read | Find every note whose text holds a literal string, and count them — scoped the same way. For verification, not discovery: `notes: 0` means nothing in scope says it |
| GET | `/api/read` | read | Read a note (`file`), or one of its sections (`section`) |
| GET | `/api/list` | read | List notes by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`), creator, limit, and `detailed=true` for each note's heading outline, and `property`, `links_to`, `linked_from` (one value each) |
| GET | `/api/tags` | read | The tag vocabulary, whole or under one term (`under`) |
| GET | `/api/properties` | read | The custom-property registry, or one property's values (`name`) |
| GET | `/api/vault-map` | read | Vault structure overview (folders, tags, recent files) |
| GET | `/api/status` | read | Index status and statistics |
| GET | `/api/health` | read | Vault health diagnostics |
| POST | `/api/validate` | read | Check vault markdown for structural and indexing problems — one note (`path`), a scope, or the whole vault; reads the files, not the index |
| POST | `/api/create` | write | Create a new note |
| POST | `/api/update` | write | Apply a list of edits to one note in one write |
| POST | `/api/move` | write | Move note to different folder |
| POST | `/api/archive` | write | Archive a note, or restore one with `undo` |
| POST | `/api/delete` | write | Delete note (soft or hard) |
| POST | `/api/index` | write | Index the configured vault |
| POST | `/api/reindex-file` | write | Re-index a single file after external edits |
| GET | `/api/identity` | read | User identity (L0) and current context (L1). `?refresh=true` re-extracts the L1 facts and takes a write key |
| POST | `/api/init` | write | First-time onboarding setup (`mode`: detect or apply) |
| POST | `/api/migrate` | write | PARA migration (`mode`: preview, apply or undo). `apply` requires the `preview` that `preview` returned |

## Authentication

All requests require an API key via the `Authorization` header:

```bash
curl -H "Authorization: Bearer kn_abc123..." http://localhost:3000/api/vault-map
```

Keys have either `read` or `write` permission. Write keys can access all endpoints; read keys are restricted to read-only endpoints. Use `--no-auth` for local development without keys (127.0.0.1 only).

## Examples

```bash
# Search
curl -X POST http://localhost:3000/api/search \
  -H "Authorization: Bearer kn_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication architecture", "top_n": 5}'

# Search, scoped to a tag or directory filter (scope/all, any, none; a
# leading / reads a term as a directory path from the vault root)
curl -X POST http://localhost:3000/api/search \
  -H "Authorization: Bearer kn_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication architecture", "top_n": 5, "scope": ["project/auth", "/01-Projects/"], "all": ["type/decision"], "any": ["status/reviewed", "status/draft"], "none": ["status/archived"]}'

# The property registry, and a list filtered by a property value
curl "http://localhost:3000/api/properties" -H "Authorization: Bearer kn_..."
curl "http://localhost:3000/api/list?property=status%3Ddraft" -H "Authorization: Bearer kn_..."

# Read a note, or one of its sections
curl "http://localhost:3000/api/read?file=01-Projects/API-Design.md" \
  -H "Authorization: Bearer kn_..."
curl "http://localhost:3000/api/read?file=01-Projects/API-Design.md&section=Endpoints" \
  -H "Authorization: Bearer kn_..."

# Create a note
curl -X POST http://localhost:3000/api/create \
  -H "Authorization: Bearer kn_..." \
  -H "Content-Type: application/json" \
  -d '{"content": "# Meeting Notes\n\nDiscussed auth timeline.", "tags": ["meeting", "auth"]}'
```

## Rate limiting and CORS

**Rate limiting:** Configurable per-key token bucket (requests per minute). Defaults to 60 req/min. Returns `429 Too Many Requests` when exceeded.

**CORS:** Configurable allowed origins in `config.toml` under `[http]`. Defaults to allow all origins for local development.

```toml
[http]
port = 3000
host = "127.0.0.1"
cors_origins = ["http://localhost:3000", "https://myapp.example.com"]
rate_limit = 60

[[http.api_keys]]
key = "kn_..."
name = "web-agent"
permissions = "write"
```

## Reading and editing

```bash
# One section of a note
curl "http://localhost:3000/api/read?file=Meeting%20Notes&section=Action%20Items" \
  -H "Authorization: Bearer kn_abc123..."

# Append to a section, and add a tag, in one write
curl -X POST http://localhost:3000/api/update \
  -H "Authorization: Bearer kn_abc123..." -H "Content-Type: application/json" \
  -d '{"file": "Meeting Notes", "edits": [
        {"section": "Action Items", "mode": "append", "content": "- [ ] Follow up"},
        {"property": "tags", "mode": "append", "content": "actionable"}
      ]}'
```

## Managing API keys

```bash
knapper configure --add-api-key        # interactive: name + read/write
knapper configure --list-api-keys
knapper configure --revoke-api-key <name>
```

Keys are written to `[[http.api_keys]]` in `~/.knapper/config.toml`.
