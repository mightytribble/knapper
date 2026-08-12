# Engraph HTTP REST API

Enable alongside MCP for web agents, scripts, and integrations:

```bash
engraph serve --http              # Default port 3000
engraph serve --http --port 8080 --host 0.0.0.0
engraph serve --http --no-auth    # Local dev only (127.0.0.1)
```

## Key Endpoints

| Method | Endpoint                | Description                                                      |
| ------ | ----------------------- | ---------------------------------------------------------------- |
| POST   | `/api/search`           | Hybrid search with semantic + FTS5 + graph + reranker + temporal |
| GET    | `/api/read/{file}`      | Read full document content + metadata                            |
| GET    | `/api/read-section`     | Read specific section by heading                                 |
| GET    | `/api/list`             | List documents by folder and tag terms (`tags`/`all`, `any`, `none`) |
| GET    | `/api/tags`             | The tag vocabulary, whole or under one term (`under`)            |
| GET    | `/api/vault-map`        | Collection structure overview                                    |
| POST   | `/api/context`          | Rich topic context with token budget                             |
| GET    | `/api/health`           | Collection health diagnostics                                    |
| POST   | `/api/create`           | Create new document                                              |
| POST   | `/api/edit`             | Section-level editing                                            |
| POST   | `/api/rewrite`          | Full body rewrite (preserves frontmatter)                        |
| POST   | `/api/edit-frontmatter` | Granular frontmatter mutations                                   |

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
