use crate::config::HttpConfig;

/// Build the OpenAPI 3.1.0 specification for all HTTP endpoints.
pub fn build_openapi_spec(server_url: &str) -> serde_json::Value {
    let mut paths = serde_json::Map::new();

    // Read endpoints
    paths.insert("/api/health-check".into(), build_health_check());
    paths.insert("/api/search".into(), build_search());
    paths.insert("/api/read".into(), build_read());
    paths.insert("/api/list".into(), build_list());
    paths.insert("/api/tags".into(), build_tags());
    paths.insert("/api/vault-map".into(), build_vault_map());
    paths.insert("/api/health".into(), build_health());
    paths.insert("/api/status".into(), build_status());

    // Write endpoints
    paths.insert("/api/create".into(), build_create());
    paths.insert("/api/update".into(), build_update());
    paths.insert("/api/move".into(), build_move());
    paths.insert("/api/archive".into(), build_archive());
    paths.insert("/api/delete".into(), build_delete());
    paths.insert("/api/index".into(), build_index());
    paths.insert("/api/reindex-file".into(), build_reindex_file());

    // Identity endpoints
    paths.insert("/api/identity".into(), build_identity_endpoint());
    paths.insert("/api/init".into(), build_init_endpoint());

    // Migration endpoints
    paths.insert("/api/migrate".into(), build_migrate());

    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "engraph",
            "version": "1.6.0",
            "description": "AI-powered semantic search and management API for Obsidian vaults."
        },
        "servers": [{ "url": server_url }],
        "security": [{ "bearerAuth": [] }],
        "components": {
            "schemas": {},
            "securitySchemes": {
                "bearerAuth": { "type": "http", "scheme": "bearer" }
            }
        },
        "paths": paths
    })
}

// ---------------------------------------------------------------------------
// Path builders — each returns one path item to keep macro recursion shallow
// ---------------------------------------------------------------------------

fn build_health_check() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "healthCheck",
            "summary": "Simple liveness check. Returns 'ok' when the server is running.",
            "responses": {
                "200": { "description": "Server is alive" }
            }
        }
    })
}

fn build_search() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "searchVault",
            "summary": "Hybrid semantic + full-text search across the vault.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "Search query text" },
                        "top_n": { "type": "integer", "description": "Number of results. Defaults to the configured top_n, the same number on every surface" },
                        "explain": { "type": "boolean", "description": "Return the per-lane score breakdown in the response's explain field" },
                        "group_by": { "type": "string", "enum": ["chunk", "file"], "description": "One result per matching section, or one per document. Defaults to the server's setting" },
                        "scope": { "type": "array", "items": { "type": "string" }, "description": "Tag terms; a trailing / matches the tag and its descendants. A term starting with / is a directory path from the vault root instead, case-sensitive, with a trailing / its subtree. Alias of all" },
                        "all": { "type": "array", "items": { "type": "string" }, "description": "Tag terms a note carries every one of, or directory terms (starting with /, case-sensitive, a trailing / its subtree) it lies under" },
                        "any": { "type": "array", "items": { "type": "string" }, "description": "Tag terms a note carries at least one of, or directory terms (starting with /, case-sensitive, a trailing / its subtree) it lies under" },
                        "none": { "type": "array", "items": { "type": "string" }, "description": "Tag terms a note carries none of, or directory terms (starting with /, case-sensitive, a trailing / its subtree) it does not lie under" },
                        "budget_tokens": { "type": "integer", "description": "Token budget for the returned text. Fill is greedy in rank order and the first result is always included. Defaults to the configured output budget" },
                        "full": { "type": "boolean", "description": "Return every result's full text, ignoring the token budget. Conflicts with summaries" },
                        "summaries": { "type": "boolean", "description": "Return breadcrumb and provenance only, no text, for every result. Conflicts with full" },
                        "scores": { "type": "boolean", "description": "Include the cross-encoder's relevance score on each block and overflow row. Absent by default; null on a degraded row, which has no probability to report" }
                    }
                }}}
            },
            "responses": { "200": { "description": "An envelope: status ('ok' or 'no_results'); degraded (bool, true when no cross-encoder ranked the results); warnings (array of strings); blocks, the results that fit the token budget, each {id, path, heading_path, provenance: {keyword, semantic, graph, linked_from}, text, untrusted_content, truncated, and score when scores was requested}; and overflow, the results the budget excluded, each {id, path, heading_path, provenance, and score when requested} with no text. explain, the per-lane breakdown, rides beside the envelope when the request asked for it" } }
        }
    })
}

fn build_read() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "readNote",
            "summary": "Read a note's full content with metadata and graph connections.",
            "parameters": [
                {
                    "name": "file", "in": "query", "required": true,
                    "description": "File path, basename, or #docid",
                    "schema": { "type": "string" }
                },
                {
                    "name": "section", "in": "query", "required": false,
                    "description": "Read one section by its heading. Omit for the whole note. The heading is an ATX # heading and the match folds case, so 'spells' finds '## Spells'. A section read narrows content and byte_count to that section; the note's tags and links are reported either way.",
                    "schema": { "type": "string" }
                }
            ],
            "responses": { "200": { "description": "Note content with metadata; outgoing_links and incoming_links are arrays of {path, docid}" } }
        }
    })
}

fn build_list() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "listNotes",
            "summary": "List notes filtered by scope operators, creator, or limit.",
            "parameters": [
                { "name": "scope", "in": "query", "required": false, "description": "Comma-separated tag terms; a trailing / matches the tag and its descendants. A term starting with / is a directory path from the vault root instead, case-sensitive, with a trailing / its subtree. Alias of all", "schema": { "type": "string" } },
                { "name": "all", "in": "query", "required": false, "description": "Comma-separated tag terms a note carries every one of, or directory terms (starting with /, case-sensitive, a trailing / its subtree) it lies under", "schema": { "type": "string" } },
                { "name": "any", "in": "query", "required": false, "description": "Comma-separated tag terms a note carries at least one of, or directory terms (starting with /, case-sensitive, a trailing / its subtree) it lies under", "schema": { "type": "string" } },
                { "name": "none", "in": "query", "required": false, "description": "Comma-separated tag terms a note carries none of, or directory terms (starting with /, case-sensitive, a trailing / its subtree) it does not lie under", "schema": { "type": "string" } },
                { "name": "created_by", "in": "query", "required": false, "description": "Agent filter", "schema": { "type": "string" } },
                { "name": "limit", "in": "query", "required": false, "description": "Maximum notes to answer. Absent, every note the scope admits", "schema": { "type": "integer" } },
                { "name": "detailed", "in": "query", "required": false, "description": "detailed=true answers each note's heading outline beside its path. The value is required; a bare `detailed` does not parse", "schema": { "type": "boolean" } }
            ],
            "responses": { "200": { "description": "Array of note summaries" } }
        }
    })
}

fn build_tags() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "listTags",
            "summary": "The vault's tag vocabulary, whole or under one term, each tag with the notes carrying it.",
            "parameters": [
                { "name": "under", "in": "query", "required": false, "description": "One tag term; the rows returned are that tag and its descendants. Omit for the whole vocabulary", "schema": { "type": "string" } }
            ],
            "responses": { "200": { "description": "Array of tag rows: path, display, note_count" } }
        }
    })
}

fn build_vault_map() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getVaultMap",
            "summary": "Get vault structure overview with folder tree, tag cloud, and statistics.",
            "responses": { "200": { "description": "Vault structure map" } }
        }
    })
}

fn build_health() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getHealth",
            "summary": "Get vault health report with orphans, broken links, stale notes, and inbox status.",
            "responses": { "200": { "description": "Vault health report" } }
        }
    })
}

fn build_create() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "createNote",
            "summary": "Create a new note with automatic placement and frontmatter generation.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": { "type": "string", "description": "Note content (markdown)" },
                        "filename": { "type": "string", "description": "Filename without .md" },
                        "type_hint": { "type": "string", "description": "Type hint for placement" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags to apply" },
                        "folder": { "type": "string", "description": "Explicit folder (skips auto-placement)" },
                        "auto_link": { "type": "boolean", "description": "Set to false to skip automatic wikilink resolution. Defaults to true." }
                    }
                }}}
            },
            "responses": { "200": { "description": "Created note path and metadata" } }
        }
    })
}

fn build_update() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "updateNote",
            "summary": "Change an existing note. Applies a list of edits in order, in one write.",
            "description": "Each edit names its target: `section` for one heading, `property` for one frontmatter key, and neither for the note's body. An edit naming both is an error. `content` is a string, or a list of strings for a list-valued property such as tags or aliases; a body edit and a section edit take a string. A body edit always keeps the note's frontmatter, so change the frontmatter with `property` edits in the same list. Three things differ from the append/edit/rewrite/edit-frontmatter/update-metadata calls this replaces. A note changed outside engraph and not yet re-indexed fails with an mtime conflict. Replacing a note's frontmatter wholesale has no spelling here: rewrite's `preserve_frontmatter: false` is gone rather than renamed, and the new frontmatter is written with `property` edits instead. A whole-note tag or alias replacement no longer stamps a `modified_by` property on the note.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "edits"],
                    "additionalProperties": false,
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "edits": {
                            "type": "array",
                            "description": "Edits to apply, in order, in one write",
                            "items": {
                                "type": "object",
                                "required": ["mode"],
                                "additionalProperties": false,
                                "properties": {
                                    "section": { "type": "string", "description": "Heading of the section to edit. Omit this and property to edit the body" },
                                    "property": { "type": "string", "description": "Frontmatter property to edit. Naming a section as well is an error" },
                                    "mode": { "type": "string", "enum": ["replace", "prepend", "append", "remove"], "description": "What the edit does. remove is for a property alone" },
                                    "content": {
                                        "description": "A string, or a list of strings to set a list-valued property",
                                        "oneOf": [
                                            { "type": "string" },
                                            { "type": "array", "items": { "type": "string" } }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }}}
            },
            "responses": { "200": { "description": "Updated note path" } }
        }
    })
}

fn build_move() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "moveNote",
            "summary": "Move a note to a different folder within the vault.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file", "new_folder"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "new_folder": { "type": "string", "description": "Destination folder path" }
                    }
                }}}
            },
            "responses": { "200": { "description": "New note path" } }
        }
    })
}

fn build_archive() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "archiveNote",
            "summary": "Archive a note (soft delete), or restore one previously archived with `undo: true`. Archiving moves the note to the archive folder and removes it from the index; `undo` reverses that and re-indexes it.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid); an archived note's path when undoing" },
                        "undo": { "type": "boolean", "description": "Restore the note instead of archiving it (default false)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Archived (or restored) note path" } }
        }
    })
}

fn build_delete() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "deleteNote",
            "summary": "Delete a note. Supports soft (archive) and hard (permanent) modes.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "Target note (path, basename, or #docid)" },
                        "mode": { "type": "string", "enum": ["soft", "hard"], "description": "'soft' (default) archives the note; 'hard' removes it permanently. A word outside the two is refused." }
                    }
                }}}
            },
            "responses": { "200": { "description": "Deletion confirmation" } }
        }
    })
}

fn build_index() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "indexVault",
            "summary": "Index the server's vault: walk it, diff it against the store, and re-embed what changed.",
            "description": "The vault is the one the server was started on; no path is taken here. Send {} to index with no options. A single file is cheaper through /api/reindex-file. The call runs to completion once it starts: it holds the store and the embedder while it runs, so every other request waits on it, and a graceful shutdown will not interrupt it. On a large vault a rebuild takes minutes. A read-only server refuses it.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "properties": {
                        "rebuild": { "type": "boolean", "description": "Discard the index and build it again from nothing" },
                        "no_gitignore": { "type": "boolean", "description": "Index files that .gitignore or .ignore would exclude" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Counts of new, updated and deleted files, total chunks and the elapsed seconds" } }
        }
    })
}

fn build_status() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getStatus",
            "summary": "What the index holds: file and chunk counts, edge and connectivity counts, date coverage, index size, and whether intelligence is enabled.",
            "responses": { "200": { "description": "Index status fields" } }
        }
    })
}

fn build_reindex_file() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "reindexFile",
            "summary": "Re-index a single file after external edits. Re-reads, re-embeds, and updates search index.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": { "type": "string", "description": "File path relative to vault root" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Re-indexed file info (chunks, docid)" } }
        }
    })
}

fn build_identity_endpoint() -> serde_json::Value {
    serde_json::json!({
        "get": {
            "operationId": "getIdentity",
            "summary": "Returns compact user identity (L0) and current context (L1).",
            "parameters": [
                {
                    "name": "refresh", "in": "query", "required": false,
                    "description": "Re-extract the L1 facts from the index before answering, without a full re-index. It rewrites the identity_facts rows, so it takes a write key and a read-only server refuses it.",
                    "schema": { "type": "boolean" }
                }
            ],
            "responses": { "200": { "description": "Identity block as JSON with 'identity' key" } }
        }
    })
}

fn build_init_endpoint() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "init",
            "summary": "Run first-time setup or update identity. Use 'detect' to inspect, 'apply' to configure.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["mode"],
                    "properties": {
                        "mode": { "type": "string", "enum": ["detect", "apply"], "description": "'detect' inspects the vault and writes nothing; 'apply' configures identity and indexes. A read-only server refuses 'apply', because it reaches the same indexing work /api/index is guarded against." },
                        "name": { "type": "string", "description": "User name (apply mode)" },
                        "role": { "type": "string", "description": "User role (apply mode)" },
                        "purpose": { "type": "string", "description": "Vault purpose (apply mode)" }
                    }
                }}}
            },
            "responses": { "200": { "description": "Setup result as JSON" } }
        }
    })
}

fn build_migrate() -> serde_json::Value {
    serde_json::json!({
        "post": {
            "operationId": "migrate",
            "summary": "Restructure the vault into PARA. 'preview' classifies notes and suggests folder moves, 'apply' performs them, 'undo' restores the last migration.",
            "requestBody": {
                "required": true,
                "content": { "application/json": { "schema": {
                    "type": "object",
                    "required": ["mode"],
                    "properties": {
                        "mode": { "type": "string", "description": "'preview', 'apply' or 'undo'" },
                        "preview": { "type": "object", "description": "The preview to apply. Required for 'apply' mode: send back the plan that 'preview' returned. There is no fallback to a plan saved on the server's disk, because a dropped key would then apply a plan this caller never saw." }
                    }
                }}}
            },
            "responses": { "200": { "description": "Migration preview, migration result or undo result, per mode" } }
        }
    })
}

/// Build the ChatGPT plugin manifest (ai-plugin.json).
pub fn build_plugin_manifest(config: &HttpConfig, server_url: &str) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "v1",
        "name_for_human": config.plugin.name.as_deref().unwrap_or("engraph"),
        "name_for_model": "engraph",
        "description_for_human": config.plugin.description.as_deref()
            .unwrap_or("Search and manage your Obsidian vault with AI-powered hybrid search."),
        "description_for_model": "Access an Obsidian knowledge vault. Use search to find notes by content or time. read for full content. who/project for context bundles. Write tools create, edit, and organize notes.",
        "auth": {
            "type": "service_http",
            "authorization_type": "bearer",
            "verification_tokens": {}
        },
        "api": {
            "type": "openapi",
            "url": format!("{}/openapi.json", server_url)
        },
        "logo_url": "",
        "contact_email": config.plugin.contact_email.as_deref().unwrap_or(""),
        "legal_info_url": ""
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_spec_structure() {
        let spec = build_openapi_spec("http://localhost:3000");
        assert_eq!(spec["openapi"], "3.1.0");
        // How many paths there are is the router's business, and
        // `the_spec_describes_every_route_the_router_serves` reads it from
        // there. A count here is a second declaration that falls out of step
        // every time one name absorbs another (#62). Assert the shape
        // instead: every path item declares a method.
        let paths = spec["paths"].as_object().unwrap();
        assert!(!paths.is_empty());
        for (path, item) in paths {
            let methods = item.as_object().unwrap();
            assert!(
                methods.contains_key("get") || methods.contains_key("post"),
                "{path} declares no method"
            );
        }
        assert_eq!(spec["servers"][0]["url"], "http://localhost:3000");
    }

    #[test]
    fn test_openapi_has_security() {
        let spec = build_openapi_spec("http://localhost:3000");
        assert!(spec["components"]["securitySchemes"]["bearerAuth"].is_object());
    }

    #[test]
    fn test_plugin_manifest() {
        let config = crate::config::HttpConfig::default();
        let manifest = build_plugin_manifest(&config, "https://vault.example.com");
        assert_eq!(manifest["schema_version"], "v1");
        assert_eq!(manifest["name_for_model"], "engraph");
        assert!(
            manifest["api"]["url"]
                .as_str()
                .unwrap()
                .contains("openapi.json")
        );
    }

    /// The tag filter is one capability on three surfaces (#61), so the
    /// spec names every operator the CLI and MCP take.
    #[test]
    fn test_list_documents_every_tag_operator() {
        let spec = build_openapi_spec("http://localhost:3000");
        let named: Vec<&str> = spec["paths"]["/api/list"]["get"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        for operator in ["scope", "all", "any", "none"] {
            assert!(named.contains(&operator), "missing parameter: {operator}");
        }
    }

    /// The outline is a documented parameter of the HTTP surface, not a
    /// CLI-only flag (#68).
    #[test]
    fn test_list_documents_detailed() {
        let spec = build_openapi_spec("http://localhost:3000");
        let named: Vec<&str> = spec["paths"]["/api/list"]["get"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert!(named.contains(&"detailed"), "missing parameter: detailed");
    }

    #[test]
    fn test_tags_documents_under() {
        let spec = build_openapi_spec("http://localhost:3000");
        let named: Vec<&str> = spec["paths"]["/api/tags"]["get"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert_eq!(named, vec!["under"]);
    }

    /// `read` absorbed `graph show`'s docid fact (#62). Every endpoint here
    /// is description-only, with no per-field schema to keep it honest, so
    /// the shape change has to be said in the one line that exists.
    #[test]
    fn test_read_documents_the_link_shape() {
        let spec = build_openapi_spec("http://localhost:3000");
        let description = spec["paths"]["/api/read"]["get"]["responses"]["200"]["description"]
            .as_str()
            .unwrap();
        assert!(
            description.contains("docid"),
            "the response description doesn't say what a link looks like: {description}"
        );
    }

    /// The set the spec publishes is the set named here — equal, not merely
    /// contained. A containment check in one direction lets an `operationId`
    /// be lost with the suite still green, and `reindexFile` and `getIdentity`
    /// were both absent from the list (#62).
    #[test]
    fn test_openapi_has_all_operation_ids() {
        use std::collections::BTreeSet;

        let spec = build_openapi_spec("http://localhost:3000");
        let paths = spec["paths"].as_object().unwrap();
        let mut op_ids: BTreeSet<String> = BTreeSet::new();
        for (path, methods) in paths {
            for (method, details) in methods.as_object().unwrap() {
                let id = details
                    .get("operationId")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| panic!("{method} {path} has no operationId"));
                assert!(
                    op_ids.insert(id.to_string()),
                    "{method} {path} repeats operationId {id}"
                );
            }
        }
        let expected: BTreeSet<String> = [
            "healthCheck",
            "searchVault",
            "readNote",
            "listNotes",
            "listTags",
            "getVaultMap",
            "getHealth",
            "getStatus",
            "createNote",
            "updateNote",
            "moveNote",
            "archiveNote",
            "deleteNote",
            "indexVault",
            "reindexFile",
            "getIdentity",
            "init",
            "migrate",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        assert_eq!(
            op_ids,
            expected,
            "\nonly in the spec: {:?}\nonly in the list: {:?}",
            op_ids.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&op_ids).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_openapi_server_url_passed_through() {
        let spec = build_openapi_spec("https://my-tunnel.example.com");
        assert_eq!(spec["servers"][0]["url"], "https://my-tunnel.example.com");
    }

    #[test]
    fn test_plugin_manifest_custom_config() {
        let mut config = crate::config::HttpConfig::default();
        config.plugin.name = Some("my-vault".into());
        config.plugin.contact_email = Some("test@example.com".into());
        let manifest = build_plugin_manifest(&config, "https://example.com");
        assert_eq!(manifest["name_for_human"], "my-vault");
        assert_eq!(manifest["contact_email"], "test@example.com");
    }

    #[test]
    fn the_spec_describes_every_route_the_router_serves() {
        let spec = build_openapi_spec("http://localhost:7777");
        let described: std::collections::BTreeSet<String> =
            spec["paths"].as_object().unwrap().keys().cloned().collect();

        // The router writes a wildcard as `{*file}`; OpenAPI writes `{file}`.
        let served: std::collections::BTreeSet<String> = crate::http::routes()
            .into_iter()
            .map(|(p, _)| p.replace("{*", "{"))
            .filter(|p| p.starts_with("/api/"))
            .collect();

        assert_eq!(served, described, "the spec and the router disagree");
    }
}
