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

`GET /openapi.json` is the OpenAPI 3.1 document for every route below, and it
takes no key. It is generated from the same parameter declarations the CLI and
MCP read, so a parameter it names is one the server reads, and the `kind` enum
under `components.schemas.Error` is the one the Errors section lists. Set
`public_url` under `[http]` in `config.toml` when the server is reached
through a tunnel; the document names it as its server.

| Method | Endpoint | Permission | Description |
|--------|----------|------------|-------------|
| GET | `/api/health-check` | none | Server health check |
| POST | `/api/search` | read | Hybrid search (semantic + FTS5 + graph + reranker + temporal), scoped by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`), and `property`, `links_to`, `linked_from` (one value each) |
| POST | `/api/match` | read | Find every note whose text holds a literal string, and count them — scoped the same way. For verification, not discovery: `notes: 0` means nothing in scope says it |
| GET | `/api/read` | read | Read a note (`file`), or one of its sections (`section`) |
| GET | `/api/list` | read | List notes by tag or directory terms — a leading `/` reads a term as a directory path (`scope`/`all`, `any`, `none`), creator, limit, `after` (the last path a page answered, which starts the next page), and `detailed=true` for each note's heading outline, and `property`, `links_to`, `linked_from` (one value each) |
| GET | `/api/tags` | read | The tag vocabulary, whole or under one term (`under`) |
| GET | `/api/properties` | read | The custom-property registry, or one property's values (`name`) |
| GET | `/api/vault-map` | read | Vault structure overview (folders, counts, the tag vocabulary and its reach, the most-linked notes, recently changed files) |
| GET | `/api/status` | read | Index status and statistics |
| GET | `/api/health` | read | Vault health diagnostics |
| POST | `/api/validate` | read | Check vault markdown for structural and indexing problems — one note (`path`), a scope, or the whole vault; reads the files, not the index |
| POST | `/api/create` | write | Create a new note, filed under `folder` or at the vault root. A `folder` with a `..` segment is 400 `invalid_input`; a leading or trailing `/` is trimmed, so `/` is the vault root. |
| POST | `/api/update` | write | Apply a list of edits to one note in one write |
| POST | `/api/move` | write | Move note to different folder. A `new_folder` with a `..` segment is 400 `invalid_input`; a leading or trailing `/` is trimmed, so `/` is the vault root. |
| POST | `/api/archive` | write | Archive a note, or restore one with `undo`. With `undo`, `file` takes the archive path, the note's original path, its basename or a `#docid`; one that names several archived notes is 400 `ambiguous`, and a `file` outside the vault, or a note whose `archived_from` is, is 400 `invalid_input`. |
| POST | `/api/delete` | write | Delete note (soft or hard) |
| POST | `/api/index` | write | Index the configured vault |
| POST | `/api/reindex-file` | write | Re-index a single file after external edits. A `file` with a `..` segment or a leading `/` is 400 `invalid_input`. |
| POST | `/api/init` | write | Write the vault profile and index (`mode`: detect or apply) |

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
# leading / reads a term as a directory path from the vault root, and a
# path ending in .md as that one note)
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

## Rate limiting, CORS and limits

**Rate limiting:** Configurable per-key token bucket (requests per minute). Defaults to 60 req/min. Returns `429 Too Many Requests` when exceeded.

**CORS:** Configurable allowed origins in `config.toml` under `[http]`. Defaults to allow all origins for local development.

**Limits:** a request is answered 408 after `[http] request_timeout_secs` (60 by default; `0` disables it), except `/api/index` and `/api/init`, which run to completion. A body over 8 MiB is refused as 400 `invalid_input`. Sixteen requests run at once; the rest wait their turn. The six write routes — `create`, `update`, `move`, `archive`, `delete`, `reindex-file` — are timed only while they read the request and wait for the index lock: a 408 on one means the write did not run and a retry is safe, and a write that started answers with its result however long it takes.

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

## Errors

Every error the server answers is a JSON body with two fields: `error`, the message, and `kind`, one word for what went wrong. That includes a body the server cannot read — malformed JSON, a word that is not one of a parameter's values, a missing field, a query value of the wrong type, a body over the size limit — which is 400 `invalid_input` with the parser's own text, an unknown path, which is 404 `not_found`, and the wrong method on a known path, which is 405 `invalid_input`. The one exception is the request timeout's 408, which has no body. The status says whose fault it is. The OpenAPI document carries the same table for the statuses an error body carries a `kind` under: every operation references `components.responses` for each, and the body is `components.schemas.Error`. The 405 and 408 are the transport's own and are not declared there.

| status | kind | when |
|---|---|---|
| 400 | `invalid_input` | the request's own text named nothing or asked two things at once: a scope term, an `after` cursor, a `links_to` or `linked_from` name, `full` with `summaries`, a `section` beside `include=metadata`, an empty `match` pattern, a `mode` word, a malformed edit list |
| 400 | `ambiguous` | one name, several notes: an alias more than one note carries |
| 404 | `not_found` | the `file` or `section` the call addresses is absent, on `read`, `update`, `move`, `delete`, `archive` and `reindex-file` |
| 405 | `invalid_input` | the wrong method on a known route |
| 409 | `conflict` | the write would clobber: the note changed on disk since it was indexed, a `create` or `move` onto an existing path, an `archive` of an archived note |
| 403 | `read_only` | the server was started with `--read-only` |
| 403 | `forbidden` | the key has no write permission |
| 401 | `unauthorized` | no key, or a key the server does not hold |
| 429 | `rate_limited` | the key's bucket is empty; `retry-after` says when |
| 408 | — | the request ran past `request_timeout_secs`; no body. `/api/index` and `/api/init` are never timed out; a write route is timed only before it starts, reading its body or waiting for the index lock, so its 408 means it did not run |
| 500 | `stale_index` | the index cannot answer until `knapper index` runs |
| 500 | `internal` | anything else; the body carries the whole error chain |

The rule separating 400 from 404: a scope term, cursor or link filter naming nothing is query shape, so it is 400; the note or section a call addresses naming nothing is an absent resource, so it is 404. `search` with `all: ["/nowhere.md"]` is 400 and `read?file=nowhere.md` is 404.

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
