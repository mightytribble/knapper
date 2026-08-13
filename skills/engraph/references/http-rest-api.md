# Engraph HTTP REST API

Enable alongside MCP for web agents, scripts, and integrations:

```bash
engraph serve --http              # Default port 3000
engraph serve --http --port 8080 --host 0.0.0.0
engraph serve --http --no-auth    # Local dev only (127.0.0.1)
```

## Key Endpoints

Every capability is one route, named after the CLI command it answers. The whole table of three surfaces is `docs/surfaces.md`.

| Method | Endpoint                | Description                                                      |
| ------ | ----------------------- | ---------------------------------------------------------------- |
| POST   | `/api/search`           | Hybrid search with semantic + FTS5 + graph + reranker + temporal, scoped by tag terms (`tags`/`all`, `any`, `none`) |
| GET    | `/api/read`             | Read a document (`file`), or one section of it (`section`)        |
| GET    | `/api/list`             | List documents by folder and tag terms (`tags`/`all`, `any`, `none`) |
| GET    | `/api/tags`             | The tag vocabulary, whole or under one term (`under`)            |
| GET    | `/api/vault-map`        | Collection structure overview                                    |
| POST   | `/api/topic`            | Rich topic context with a character budget, scoped by tag terms (`tags`/`all`, `any`, `none`) |
| GET    | `/api/health`           | Collection health diagnostics                                    |
| POST   | `/api/create`           | Create new document                                              |
| POST   | `/api/update`           | Apply a list of edits — body, sections and properties — in one write |
| POST   | `/api/archive`          | Archive a document, or restore one with `undo`                   |
| POST   | `/api/delete`           | Delete a document (`mode`: soft or hard)                         |

## Reading and editing

```bash
# One section of a document
curl "http://localhost:3000/api/read?file=Meeting%20Notes&section=Action%20Items" \
  -H "Authorization: Bearer eg_abc123..."

# Append to a section, and add a tag, in one write
curl -X POST http://localhost:3000/api/update \
  -H "Authorization: Bearer eg_abc123..." -H "Content-Type: application/json" \
  -d '{"file": "Meeting Notes", "edits": [
        {"section": "Action Items", "mode": "append", "content": "- [ ] Follow up"},
        {"property": "tags", "mode": "append", "content": "actionable"}
      ]}'
```

## Authentication

All HTTP requests require an API key via the `Authorization` header:

```bash
curl -H "Authorization: Bearer eg_abc123..." http://localhost:3000/api/search \
  -H "Content-Type: application/json" -d '{"query": "architecture", "top_n": 5}'
```

Generate keys:

```bash
engraph configure --add-api-key       # Interactive
engraph configure --list-api-keys     # List existing
engraph configure --revoke-api-key <name>  # Revoke a key by name
```
