//! The OpenAPI 3.1 document for the HTTP surface, generated rather than
//! written.
//!
//! Every parameter comes from the capability's `params` struct, through the
//! `JsonSchema` it already derives for MCP, so the document cannot name a
//! parameter the server does not read. The three things a struct cannot say
//! — the operation's id, its summary and what its 200 reply holds — come
//! from `surface::OPERATIONS`. schemars emits the 2020-12 dialect, which
//! OpenAPI 3.1 embeds as is; the one transform is `query_parameters`, for
//! the GET routes (#62).

use rmcp::schemars::JsonSchema;
use rmcp::schemars::generate::{SchemaGenerator, SchemaSettings};
use serde_json::{Value, json};

use crate::surface::{CAPABILITIES, Http, OPERATIONS};

/// Build the OpenAPI 3.1 document for every route under `/api/`.
///
/// `server_url` is `[http] public_url` when set, else the bound address.
pub fn build_openapi_spec(server_url: &str) -> Value {
    let settings = SchemaSettings::draft2020_12().with(|s| {
        s.definitions_path = "/components/schemas".into();
        s.meta_schema = None;
    });
    let mut generator = settings.into_generator();

    let mut paths = serde_json::Map::new();
    paths.insert("/api/health-check".into(), health_check_operation());

    for capability in CAPABILITIES {
        let method = match capability.http {
            Http::Get => "get",
            Http::Post => "post",
            Http::Exempt(_) => continue,
        };
        let row = OPERATIONS
            .iter()
            .find(|o| o.name == capability.name)
            .unwrap_or_else(|| panic!("{} has no operation row", capability.name));
        let schema = params_schema(capability.name, &mut generator)
            .unwrap_or_else(|| panic!("{} has no params struct", capability.name));

        let mut operation = json!({
            "operationId": row.id,
            "summary": row.summary,
            "responses": operation_responses(row.response),
        });
        match capability.http {
            Http::Get => {
                let parameters = query_parameters(&schema);
                if !parameters.is_empty() {
                    operation["parameters"] = Value::Array(parameters);
                }
            }
            Http::Post => {
                operation["requestBody"] = json!({
                    "required": true,
                    "content": { "application/json": { "schema": schema } },
                });
            }
            Http::Exempt(_) => unreachable!(),
        }

        let mut item = serde_json::Map::new();
        item.insert(method.to_string(), operation);
        paths.insert(capability.http_path(), Value::Object(item));
    }

    let mut schemas = generator.take_definitions(true);
    let (error, responses) = error_components();
    schemas.insert("Error".into(), error);

    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "knapper",
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
        },
        "servers": [{ "url": server_url }],
        "security": [{ "bearerAuth": [] }],
        "components": {
            "schemas": schemas,
            "responses": responses,
            "securitySchemes": {
                "bearerAuth": { "type": "http", "scheme": "bearer" }
            }
        },
        "paths": paths
    })
}

/// The liveness probe. Not a capability: it takes no key and no parameters.
fn health_check_operation() -> Value {
    json!({
        "get": {
            "operationId": "healthCheck",
            "summary": "Liveness check. Answers the text ok while the server is running.",
            "security": [],
            "responses": {
                "200": {
                    "description": "Server is alive",
                    "content": { "text/plain": { "schema": { "type": "string" } } }
                }
            }
        }
    })
}

/// One row per status the classifier answers, with the kinds it carries,
/// in the words of http-rest-api.md's table. The test
/// `each_kind_is_listed_under_the_status_the_classifier_answers` holds each
/// kind to the status `From<anyhow::Error> for ApiError` gives it.
const ERROR_RESPONSES: &[(&str, &str)] = &[
    (
        "400",
        "The request's own text named nothing or asked two things at once: a scope term, an after cursor, a links_to or linked_from name, full with summaries, a section beside include=metadata, an empty match pattern, a mode word, a malformed edit list (kind invalid_input); or one name matched several notes, such as an alias more than one note carries (kind ambiguous).",
    ),
    (
        "401",
        "No key, or a key the server does not hold (kind unauthorized).",
    ),
    (
        "403",
        "The key has no write permission (kind forbidden), or the server was started with --read-only (kind read_only).",
    ),
    (
        "404",
        "The file or section the call addresses is absent (kind not_found).",
    ),
    (
        "409",
        "The write would clobber: the note changed on disk since it was indexed, a create or move onto an existing path, an archive of an archived note (kind conflict).",
    ),
    (
        "429",
        "The key's bucket is empty; the retry-after header says when (kind rate_limited).",
    ),
    (
        "500",
        "The index cannot answer until knapper index runs (kind stale_index), or anything else, with the whole error chain in error (kind internal).",
    ),
];

/// `components.schemas.Error` and `components.responses`, built from the
/// kinds the two classifiers declare.
fn error_components() -> (Value, serde_json::Map<String, Value>) {
    let kinds: Vec<&str> = crate::fault::Fault::KINDS
        .iter()
        .chain(crate::http::ApiError::TRANSPORT_KINDS)
        .copied()
        .collect();
    let error = json!({
        "type": "object",
        "required": ["error", "kind"],
        "properties": {
            "error": {
                "type": "string",
                "description": "The message. A 500 carries the whole error chain."
            },
            "kind": {
                "type": "string",
                "enum": kinds,
                "description": "One word for what went wrong. The status says whose fault it is."
            }
        }
    });
    let mut responses = serde_json::Map::new();
    for (status, description) in ERROR_RESPONSES {
        responses.insert(
            (*status).to_string(),
            json!({
                "description": description,
                "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } }
            }),
        );
    }
    (error, responses)
}

/// The `responses` of a capability operation: the 200 and a reference to
/// every error status.
fn operation_responses(response: &str) -> Value {
    let mut responses = serde_json::Map::new();
    responses.insert("200".into(), json!({ "description": response }));
    for (status, _) in ERROR_RESPONSES {
        responses.insert(
            (*status).to_string(),
            json!({ "$ref": format!("#/components/responses/{status}") }),
        );
    }
    Value::Object(responses)
}

/// The schema of the `params` struct a capability reads, or `None` for a
/// name that is not a capability.
///
/// This is the one match from a name to a type. The handler's extractor
/// names the type a second time and `serve.rs`'s tool a third; the test
/// `every_operations_parameters_are_the_mcp_tools_properties` holds this
/// match to the third.
fn params_schema(name: &str, generator: &mut SchemaGenerator) -> Option<Value> {
    use crate::params as p;

    fn of<T: JsonSchema>(generator: &mut SchemaGenerator) -> Value {
        <T as JsonSchema>::json_schema(generator).to_value()
    }

    Some(match name {
        "search" => of::<p::Search>(generator),
        "match" => of::<p::Match>(generator),
        "read" => of::<p::Read>(generator),
        "list" => of::<p::List>(generator),
        "tags" => of::<p::Tags>(generator),
        "properties" => of::<p::Properties>(generator),
        "vault-map" => of::<p::VaultMap>(generator),
        "create" => of::<p::Create>(generator),
        "update" => of::<p::Update>(generator),
        "delete" => of::<p::Delete>(generator),
        "move" => of::<p::Move>(generator),
        "archive" => of::<p::Archive>(generator),
        "index" => of::<p::Index>(generator),
        "reindex-file" => of::<p::ReindexFile>(generator),
        "status" => of::<p::Status>(generator),
        "health" => of::<p::Health>(generator),
        "validate" => of::<p::Validate>(generator),
        "init" => of::<p::Init>(generator),
        _ => return None,
    })
}

/// The GET rule: one query parameter per property of the struct schema.
///
/// A GET route reads its struct from the query string, where
/// `serde_urlencoded` reads no sequence and no null (#61). So an `array`
/// property is one comma-separated `string`, and the `null` a body may send
/// is a parameter's absence. This is the one place that rule is written.
fn query_parameters(schema: &Value) -> Vec<Value> {
    let required: Vec<&str> = schema["required"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let Some(properties) = schema["properties"].as_object() else {
        return Vec::new();
    };

    properties
        .iter()
        .map(|(name, property)| {
            let mut property = property.clone();
            let mut description = property
                .as_object_mut()
                .and_then(|o| o.remove("description"))
                .and_then(|d| d.as_str().map(String::from))
                .unwrap_or_default();
            let schema = query_schema(property, &mut description);
            let mut parameter = json!({
                "name": name,
                "in": "query",
                "required": required.contains(&name.as_str()),
                "schema": schema,
            });
            if !description.is_empty() {
                parameter["description"] = Value::String(description);
            }
            parameter
        })
        .collect()
}

/// One property schema, as a query string can carry it.
fn query_schema(mut schema: Value, description: &mut String) -> Value {
    let Some(object) = schema.as_object_mut() else {
        return schema;
    };

    if let Some(types) = object.get("type").and_then(Value::as_array).cloned() {
        let kept: Vec<Value> = types.into_iter().filter(|t| *t != "null").collect();
        let one = match kept.as_slice() {
            [one] => one.clone(),
            _ => Value::Array(kept),
        };
        object.insert("type".into(), one);
    }

    // schemars writes an optional enum as `[inner, {"type": "null"}]`, null last.
    if let Some(any_of) = object.get("anyOf").and_then(Value::as_array).cloned()
        && let [inner, null] = any_of.as_slice()
        && *null == json!({ "type": "null" })
    {
        object.remove("anyOf");
        if let Some(inner) = inner.as_object() {
            for (k, v) in inner {
                object.insert(k.clone(), v.clone());
            }
        }
    }

    if object.get("default") == Some(&Value::Null) {
        object.remove("default");
    }

    if object.get("type") == Some(&json!("array")) {
        if !description.is_empty() {
            description.push(' ');
        }
        description.push_str("Comma-separated.");
        return json!({ "type": "string" });
    }

    schema
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_openapi_spec_structure() {
        let spec = build_openapi_spec("http://localhost:3000");
        assert_eq!(spec["openapi"], "3.1.0");
        // How many paths there are is the router's business, and
        // `the_spec_describes_every_route_the_router_serves` reads it from
        // there. Assert the shape instead: every path item declares a method.
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
        assert_eq!(spec["security"], serde_json::json!([{ "bearerAuth": [] }]));
    }

    #[test]
    fn test_openapi_server_url_passed_through() {
        let spec = build_openapi_spec("https://my-tunnel.example.com");
        assert_eq!(spec["servers"][0]["url"], "https://my-tunnel.example.com");
    }

    #[test]
    fn the_spec_describes_every_route_the_router_serves() {
        let spec = build_openapi_spec("http://localhost:7777");
        let described: BTreeSet<String> =
            spec["paths"].as_object().unwrap().keys().cloned().collect();

        let served: BTreeSet<String> = crate::http::routes()
            .into_iter()
            .map(|(p, _)| p.to_string())
            .filter(|p| p.starts_with("/api/"))
            .collect();

        assert_eq!(served, described, "the spec and the router disagree");
    }

    /// The names an operation publishes — GET parameters or POST body
    /// properties — are the names the MCP tool publishes for the same
    /// capability. The tool list registers each `params` struct through
    /// `serve.rs`, so this holds `params_schema`'s match to the handlers'
    /// extractors by way of a third registration, and it is the one test
    /// that replaces the per-field ones (#62).
    #[test]
    fn every_operations_parameters_are_the_mcp_tools_properties() {
        use crate::surface::{CAPABILITIES, Http};

        let spec = build_openapi_spec("http://localhost:3000");
        let tools = crate::serve::KnapperServer::tool_router().list_all();
        let mut checked = 0;

        for capability in CAPABILITIES {
            let path = capability.http_path();
            let (method, published): (&str, BTreeSet<String>) =
                match capability.http {
                    Http::Get => (
                        "get",
                        spec["paths"][&path]["get"]["parameters"]
                            .as_array()
                            .cloned()
                            .unwrap_or_default()
                            .iter()
                            .map(|p| p["name"].as_str().unwrap().to_string())
                            .collect(),
                    ),
                    Http::Post => (
                        "post",
                        spec["paths"][&path]["post"]["requestBody"]["content"]["application/json"]
                            ["schema"]["properties"]
                            .as_object()
                            .cloned()
                            .unwrap_or_default()
                            .keys()
                            .cloned()
                            .collect(),
                    ),
                    Http::Exempt(_) => continue,
                };

            let tool = tools
                .iter()
                .find(|t| t.name == capability.mcp_name())
                .unwrap_or_else(|| panic!("{} is not an MCP tool", capability.name));
            let mcp: BTreeSet<String> = tool
                .input_schema
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default();

            assert_eq!(
                published,
                mcp,
                "\n{method} {path}: only in the document: {:?}\n{method} {path}: only in the tool: {:?}",
                published.difference(&mcp).collect::<Vec<_>>(),
                mcp.difference(&published).collect::<Vec<_>>()
            );
            checked += 1;
        }

        assert_eq!(checked, CAPABILITIES.len(), "a capability was skipped");
    }

    /// Every operation carries the id and summary its table row gives it.
    #[test]
    fn every_operation_carries_its_rows_id_and_summary() {
        use crate::surface::{CAPABILITIES, Http, OPERATIONS};

        let spec = build_openapi_spec("http://localhost:3000");
        for capability in CAPABILITIES {
            let method = match capability.http {
                Http::Get => "get",
                Http::Post => "post",
                Http::Exempt(_) => continue,
            };
            let row = OPERATIONS
                .iter()
                .find(|o| o.name == capability.name)
                .unwrap();
            let operation = &spec["paths"][&capability.http_path()][method];
            assert_eq!(operation["operationId"], row.id, "{}", capability.name);
            assert_eq!(operation["summary"], row.summary, "{}", capability.name);
            assert_eq!(
                operation["responses"]["200"]["description"], row.response,
                "{}",
                capability.name
            );
        }
    }

    /// A GET route reads its struct from the query string, where
    /// `serde_urlencoded` reads no sequence, so a list is one comma-separated
    /// value (#61). A POST body keeps the array.
    #[test]
    fn a_get_routes_list_parameter_is_a_comma_separated_string() {
        let spec = build_openapi_spec("http://localhost:3000");
        let all = spec["paths"]["/api/list"]["get"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "all")
            .expect("/api/list takes all");
        assert_eq!(all["schema"], serde_json::json!({ "type": "string" }));
        assert!(
            all["description"]
                .as_str()
                .unwrap()
                .ends_with(" Comma-separated."),
            "{}",
            all["description"]
        );

        let body_all = &spec["paths"]["/api/search"]["post"]["requestBody"]["content"]["application/json"]
            ["schema"]["properties"]["all"];
        assert_eq!(body_all["type"], "array");
        assert_eq!(body_all["items"], serde_json::json!({ "type": "string" }));
    }

    /// A query string carries no null: an absent parameter is the null a
    /// body may send. So the `null` schemars adds for an `Option` is dropped,
    /// a one-of-two `anyOf` collapses to its one schema, and `required` says
    /// what is optional.
    #[test]
    fn a_query_parameter_carries_no_null() {
        let spec = build_openapi_spec("http://localhost:3000");
        let read = spec["paths"]["/api/read"]["get"]["parameters"]
            .as_array()
            .unwrap();
        let by_name = |name: &str| {
            read.iter()
                .find(|p| p["name"] == name)
                .unwrap_or_else(|| panic!("/api/read takes {name}"))
        };

        let file = by_name("file");
        assert_eq!(file["required"], true);
        assert_eq!(file["schema"], serde_json::json!({ "type": "string" }));

        let section = by_name("section");
        assert_eq!(section["required"], false);
        assert_eq!(section["schema"], serde_json::json!({ "type": "string" }));
        assert!(
            section["description"]
                .as_str()
                .unwrap()
                .starts_with("Read one section")
        );

        let include = by_name("include");
        assert_eq!(include["required"], false);
        assert_eq!(include["schema"]["$ref"], "#/components/schemas/Include");

        let limit = spec["paths"]["/api/list"]["get"]["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "limit")
            .unwrap();
        assert_eq!(limit["schema"]["type"], "integer");
        assert!(limit["schema"].get("default").is_none());
    }

    /// The GET rule holds for the two shapes no struct has yet: an optional
    /// list is still one comma-separated string, and an optional enum with a
    /// default loses the null and the null default both.
    #[test]
    fn the_get_rule_reads_an_optional_list_and_an_optional_enum() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "tags": {
                    "description": "Tags.",
                    "type": ["array", "null"],
                    "items": { "type": "string" },
                    "default": null
                },
                "mode": {
                    "anyOf": [{ "$ref": "#/components/schemas/Mode" }, { "type": "null" }],
                    "default": null
                },
                "either": {
                    "anyOf": [{ "type": "string" }, { "type": "integer" }]
                }
            }
        });
        let parameters = query_parameters(&schema);
        let by_name = |name: &str| {
            parameters
                .iter()
                .find(|p| p["name"] == name)
                .unwrap_or_else(|| panic!("no parameter {name}"))
        };

        let tags = by_name("tags");
        assert_eq!(tags["schema"], serde_json::json!({ "type": "string" }));
        assert_eq!(tags["description"], "Tags. Comma-separated.");
        assert_eq!(tags["required"], false);

        let mode = by_name("mode");
        assert_eq!(
            mode["schema"],
            serde_json::json!({ "$ref": "#/components/schemas/Mode" })
        );

        let either = by_name("either");
        assert_eq!(
            either["schema"]["anyOf"],
            serde_json::json!([{ "type": "string" }, { "type": "integer" }])
        );
    }

    /// Nested types are hoisted to `components.schemas`, where OpenAPI puts
    /// them; nothing from the standalone JSON Schema dialect is left behind.
    #[test]
    fn nested_types_live_under_components() {
        let spec = build_openapi_spec("http://localhost:3000");
        let edits = &spec["paths"]["/api/update"]["post"]["requestBody"]["content"]["application/json"]
            ["schema"]["properties"]["edits"];
        assert_eq!(edits["items"]["$ref"], "#/components/schemas/Edit");
        assert!(spec["components"]["schemas"]["Edit"]["properties"]["mode"].is_object());
        assert_eq!(
            spec["components"]["schemas"]["Edit"]["properties"]["mode"]["$ref"],
            "#/components/schemas/EditMode"
        );
        for name in [
            "Edit",
            "EditContent",
            "EditMode",
            "Include",
            "Sort",
            "Scan",
            "DeleteMode",
            "GroupBy",
        ] {
            assert!(
                spec["components"]["schemas"][name].is_object(),
                "components.schemas has no {name}"
            );
        }

        fn walk(value: &serde_json::Value, path: &str, found: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (k, v) in map {
                        if k == "$defs" || k == "$schema" || k == "title" {
                            found.push(format!("{path}/{k}"));
                        }
                        walk(v, &format!("{path}/{k}"), found);
                    }
                }
                serde_json::Value::Array(items) => {
                    for (i, v) in items.iter().enumerate() {
                        walk(v, &format!("{path}/{i}"), found);
                    }
                }
                _ => {}
            }
        }
        let mut found = Vec::new();
        walk(&spec["paths"], "paths", &mut found);
        walk(
            &spec["components"]["schemas"],
            "components/schemas",
            &mut found,
        );
        assert!(
            found.is_empty(),
            "standalone-dialect keys in the document: {found:?}"
        );
    }

    #[test]
    fn the_version_is_the_crates_own() {
        let spec = build_openapi_spec("http://localhost:3000");
        assert_eq!(spec["info"]["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(spec["info"]["description"], env!("CARGO_PKG_DESCRIPTION"));
    }

    /// The liveness probe is the transport's own route. It takes no key, and
    /// the document says so.
    #[test]
    fn health_check_takes_no_key() {
        let spec = build_openapi_spec("http://localhost:3000");
        let op = &spec["paths"]["/api/health-check"]["get"];
        assert_eq!(op["operationId"], "healthCheck");
        assert_eq!(op["security"], serde_json::json!([]));
        assert!(op["responses"]["200"]["content"]["text/plain"].is_object());
        assert!(op.get("parameters").is_none());
    }

    /// Every capability operation documents the error body under every
    /// status the classifier can answer. The classifier is one function and
    /// which status a call reaches depends on its arguments, so the set is
    /// the same on every route.
    #[test]
    fn every_operation_answers_the_error_body() {
        use crate::surface::{CAPABILITIES, Http};

        let spec = build_openapi_spec("http://localhost:3000");
        let statuses = ["400", "401", "403", "404", "409", "429", "500"];

        for status in statuses {
            let response = &spec["components"]["responses"][status];
            assert!(
                response["description"]
                    .as_str()
                    .is_some_and(|d| !d.is_empty()),
                "components.responses.{status} has no description"
            );
            assert_eq!(
                response["content"]["application/json"]["schema"]["$ref"],
                "#/components/schemas/Error",
                "{status}"
            );
        }

        let error = &spec["components"]["schemas"]["Error"];
        assert_eq!(error["type"], "object");
        assert_eq!(error["required"], serde_json::json!(["error", "kind"]));
        assert_eq!(error["properties"]["error"]["type"], "string");
        let kinds: BTreeSet<&str> = error["properties"]["kind"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap())
            .collect();
        let want: BTreeSet<&str> = crate::fault::Fault::KINDS
            .iter()
            .chain(crate::http::ApiError::TRANSPORT_KINDS)
            .copied()
            .collect();
        assert_eq!(kinds, want);

        for capability in CAPABILITIES {
            let method = match capability.http {
                Http::Get => "get",
                Http::Post => "post",
                Http::Exempt(_) => continue,
            };
            let responses = &spec["paths"][&capability.http_path()][method]["responses"];
            for status in statuses {
                assert_eq!(
                    responses[status]["$ref"],
                    format!("#/components/responses/{status}"),
                    "{} {method} {status}",
                    capability.name
                );
            }
        }

        let health = &spec["paths"]["/api/health-check"]["get"]["responses"];
        assert_eq!(
            health.as_object().unwrap().len(),
            1,
            "health-check answers 200 alone"
        );
    }

    /// The description under each status names the kinds the classifier
    /// answers with that status, so the document and `From<anyhow::Error>
    /// for ApiError` cannot say different things.
    #[test]
    fn each_kind_is_listed_under_the_status_the_classifier_answers() {
        use crate::fault::Fault;
        use crate::http::ApiError;
        use std::collections::{BTreeMap, BTreeSet};

        let spec = build_openapi_spec("http://localhost:3000");

        // What the classifier answers: status → the kinds it gives that status.
        let mut classified: BTreeMap<u16, BTreeSet<&str>> = BTreeMap::new();
        let faults = [
            Fault::InvalidInput("x".into()),
            Fault::NotFound("x".into()),
            Fault::Ambiguous("x".into()),
            Fault::Conflict("x".into()),
            Fault::StaleIndex("x".into()),
            Fault::ReadOnly,
        ];
        for fault in faults {
            let kind = fault.kind();
            let api = ApiError::from(anyhow::Error::from(fault));
            classified
                .entry(api.status.as_u16())
                .or_default()
                .insert(kind);
        }
        for api in [
            ApiError::unauthorized("x"),
            ApiError::forbidden("x"),
            ApiError::rate_limited(1),
            ApiError::internal("x"),
        ] {
            classified
                .entry(api.status.as_u16())
                .or_default()
                .insert(api.kind);
        }

        // What the document says: status → the kinds its description names,
        // each written as `(kind <word>)`.
        let responses = spec["components"]["responses"].as_object().unwrap();
        let mut documented: BTreeMap<u16, BTreeSet<&str>> = BTreeMap::new();
        for (status, response) in responses {
            let description = response["description"].as_str().unwrap();
            let kinds: BTreeSet<&str> = description
                .split("(kind ")
                .skip(1)
                .map(|rest| rest.split(')').next().unwrap())
                .collect();
            documented.insert(status.parse().unwrap(), kinds);
        }

        assert_eq!(documented, classified);
    }
}
