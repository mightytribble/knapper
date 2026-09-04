# Knapper HTTP REST API

Enable alongside MCP for web agents, scripts, and integrations:

```bash
knapper serve --http              # Default port 3000
knapper serve --http --port 8080 --host 0.0.0.0
knapper serve --http --no-auth    # Local dev only (127.0.0.1)
```

## Key Endpoints

Every capability is one route, named after the CLI command it answers. The whole table of three surfaces is `surfaces.md`.

| Method | Endpoint                | Description                                                      |
| ------ | ----------------------- | ---------------------------------------------------------------- |
| POST   | `/api/search`           | Hybrid search with semantic + FTS5 + graph + reranker + temporal, scoped by tag terms (`scope`/`all`, `any`, `none`); `property`, `links_to`, `linked_from` filter by custom property and by link, one value each |
| GET    | `/api/read`             | Read a document (`file`), or one section of it (`section`)        |
| GET    | `/api/list`             | List documents by tag or directory terms (`scope`/`all`, `any`, `none`), in path order; `detailed=true` adds each document's heading outline; `property`, `links_to`, `linked_from` filter by custom property and by link, one value each |
| GET    | `/api/tags`             | The tag vocabulary, whole or under one term (`under`)            |
| GET    | `/api/properties`       | The custom-property registry, or one property's values (`name`) |
| GET    | `/api/vault-map`        | Collection structure overview                                    |
| POST   | `/api/match`            | Every note whose text holds a literal string, and the counts — `notes: 0` is a reliable absence answer |
| POST   | `/api/validate`         | Structural and indexing problems in the vault's markdown          |
| GET    | `/api/health`           | Collection health diagnostics                                    |
| POST   | `/api/create`           | Create new document                                              |
| POST   | `/api/update`           | Apply a list of edits — body, sections and properties — in one write |
| POST   | `/api/archive`          | Archive a document, or restore one with `undo`                   |
| POST   | `/api/delete`           | Delete a document (`mode`: soft or hard)                         |

## Reading and editing

```bash
# One section of a document
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

## Authentication

All HTTP requests require an API key via the `Authorization` header:

```bash
curl -H "Authorization: Bearer kn_abc123..." http://localhost:3000/api/search \
  -H "Content-Type: application/json" -d '{"query": "architecture", "top_n": 5}'
```

Generate keys:

```bash
knapper configure --add-api-key       # Interactive
knapper configure --list-api-keys     # List existing
knapper configure --revoke-api-key <name>  # Revoke a key by name
```
