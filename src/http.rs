use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, Method};
use axum::{
    Json, Router,
    http::StatusCode,
    response::IntoResponse,
    routing::{MethodRouter, get, post},
};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::config::{ApiKeyConfig, HttpConfig};
use crate::context::{self, ContextParams};
use crate::health;
use crate::llm::{EmbedModel, RerankModel};
use crate::profile::VaultProfile;
use crate::search;
use crate::serve::RecentWrites;
use crate::store::Store;
use crate::writer::{self, CreateNoteInput, DeleteMode, UpdateInput};

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApiState {
    pub store: Arc<Mutex<Store>>,
    pub embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    pub vault_path: Arc<std::path::PathBuf>,
    pub profile: Arc<Option<VaultProfile>>,
    pub reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>>,
    pub http_config: Arc<HttpConfig>,
    pub no_auth: bool,
    pub recent_writes: RecentWrites,
    pub rate_limiter: Arc<RateLimiter>,
    pub read_only: bool,
    /// Retrieval granularity settings from `config.toml`.
    pub max_chunks_per_file: usize,
    pub group_by: crate::config::GroupBy,
    /// How many results a call that names no `top_n` gets. It comes from
    /// `config.toml`, the way the CLI's does: a default that differs per
    /// surface is the last place one query answers two ways (#62).
    pub top_n: usize,
    /// Rerank-lane settings from `config.toml`.
    pub rerank: crate::config::RerankConfig,
    /// Ranking-stage settings from `config.toml`.
    pub ranking: crate::config::RankingConfig,
    pub lane_weights: crate::config::LaneWeights,
    /// Keyword-lane settings from `config.toml` (issue #37).
    pub fts: crate::config::FtsConfig,
    /// The index-time settings — how a note written over HTTP is chunked, and
    /// the vector it is embedded as — captured once at startup so every write
    /// endpoint and every full index this server runs shares one chunking and
    /// one vector space with the indexed vault (issues #43, #44, #72).
    pub index_settings: crate::indexer::IndexSettings,
    /// Output-packaging settings from `config.toml` (#35): the default token
    /// budget a request's `budget_tokens` overrides.
    pub output: crate::config::OutputConfig,
}

// ---------------------------------------------------------------------------
// Rate limiter (in-memory token bucket)
// ---------------------------------------------------------------------------

pub struct RateLimiter {
    buckets: std::sync::Mutex<HashMap<String, RateBucket>>,
    limit: u32, // requests per minute, 0 = unlimited
}

struct RateBucket {
    tokens: u32,
    last_refill: Instant,
}

impl RateLimiter {
    pub fn new(limit: u32) -> Self {
        Self {
            buckets: std::sync::Mutex::new(HashMap::new()),
            limit,
        }
    }

    /// Check if a request is allowed. Returns Ok(()) or Err with retry-after seconds.
    pub fn check(&self, key: &str) -> Result<(), u64> {
        if self.limit == 0 {
            return Ok(());
        }
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets.entry(key.to_string()).or_insert(RateBucket {
            tokens: self.limit,
            last_refill: Instant::now(),
        });
        // Refill tokens based on elapsed time
        let elapsed = bucket.last_refill.elapsed().as_secs_f64();
        let refill = (elapsed * self.limit as f64 / 60.0) as u32;
        if refill > 0 {
            bucket.tokens = (bucket.tokens + refill).min(self.limit);
            bucket.last_refill = Instant::now();
        }
        if bucket.tokens > 0 {
            bucket.tokens -= 1;
            Ok(())
        } else {
            let retry_after = (60.0 / self.limit as f64).ceil() as u64;
            Err(retry_after)
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub headers: Vec<(String, String)>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "error": self.message });
        let mut response = (self.status, Json(body)).into_response();
        for (name, value) in &self.headers {
            if let (Ok(n), Ok(v)) = (
                axum::http::header::HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                response.headers_mut().insert(n, v);
            }
        }
        response
    }
}

impl ApiError {
    pub fn unauthorized(msg: &str) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn forbidden(msg: &str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn bad_request(msg: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn not_found(msg: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn internal(msg: &str) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.to_string(),
            headers: vec![],
        }
    }
    pub fn rate_limited(retry_after: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: format!("Rate limit exceeded. Retry after {retry_after}s"),
            headers: vec![("retry-after".to_string(), retry_after.to_string())],
        }
    }
}

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

/// Validate API key from Authorization header. Returns the matching key config.
pub fn validate_api_key<'a>(key: &str, config: &'a HttpConfig) -> Option<&'a ApiKeyConfig> {
    config.api_keys.iter().find(|k| k.key == key)
}

/// Check if a permission level allows the requested operation.
pub fn check_permission(permission: &str, is_write: bool) -> bool {
    if !is_write {
        return true;
    }
    permission == "write"
}

/// Extract and validate auth from request headers, then check rate limit.
pub fn authorize(
    headers: &axum::http::HeaderMap,
    state: &ApiState,
    is_write: bool,
) -> Result<(), ApiError> {
    if state.no_auth {
        state
            .rate_limiter
            .check("no_auth")
            .map_err(ApiError::rate_limited)?;
        return Ok(());
    }
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError::unauthorized("Missing Authorization header"))?;
    let key = auth
        .strip_prefix("Bearer ")
        .ok_or_else(|| ApiError::unauthorized("Authorization must use Bearer scheme"))?;
    let key_config = validate_api_key(key, &state.http_config)
        .ok_or_else(|| ApiError::unauthorized("Invalid API key"))?;
    if !check_permission(&key_config.permissions, is_write) {
        return Err(ApiError::forbidden(
            "Insufficient permissions: write access required",
        ));
    }
    state
        .rate_limiter
        .check(key)
        .map_err(ApiError::rate_limited)?;
    Ok(())
}

/// Generate a new API key with `eg_` prefix + 32 hex chars.
pub fn generate_api_key() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let hex: String = (0..32)
        .map(|_| format!("{:x}", rng.random_range(0..16u8)))
        .collect();
    format!("eg_{hex}")
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

fn cors_layer(origins: &[String]) -> CorsLayer {
    if origins.is_empty() {
        return CorsLayer::new();
    }
    if origins.iter().any(|o| o == "*") {
        return CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);
    }
    let origins: Vec<HeaderValue> = origins.iter().filter_map(|o| o.parse().ok()).collect();
    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
            axum::http::header::ACCEPT,
        ])
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Every route the API serves, as data. `build_router` folds this list into
/// the `Router`, and `surface.rs`'s parity test reads it — an axum `Router`
/// cannot be inspected once built, so the list is the only way the test can
/// see what is served (#62). `openapi.rs` reads it too, so the spec and the
/// router cannot describe different APIs.
pub fn routes() -> Vec<(&'static str, MethodRouter<ApiState>)> {
    vec![
        ("/api/health-check", get(health_check)),
        ("/api/search", post(handle_search)),
        ("/api/read", get(handle_read)),
        ("/api/list", get(handle_list)),
        ("/api/tags", get(handle_tags)),
        ("/api/vault-map", get(handle_vault_map)),
        ("/api/health", get(handle_health)),
        ("/api/status", get(handle_status)),
        // Write endpoints
        ("/api/create", post(handle_create)),
        ("/api/update", post(handle_update)),
        ("/api/move", post(handle_move)),
        ("/api/archive", post(handle_archive)),
        ("/api/delete", post(handle_delete)),
        // Index maintenance
        ("/api/index", post(handle_index)),
        ("/api/reindex-file", post(handle_reindex_file)),
        // Identity endpoints
        ("/api/identity", get(handle_identity)),
        ("/api/init", post(handle_init)),
        // Migration endpoints
        ("/api/migrate", post(handle_migrate)),
        // OpenAPI / ChatGPT plugin discovery (no auth required)
        ("/openapi.json", get(handle_openapi)),
        ("/.well-known/ai-plugin.json", get(handle_plugin_manifest)),
    ]
}

pub fn build_router(state: ApiState) -> Router {
    let cors = cors_layer(&state.http_config.cors_origins);
    let mut router = Router::new();
    for (path, handler) in routes() {
        router = router.route(path, handler);
    }
    router.layer(cors).with_state(state)
}

async fn health_check() -> &'static str {
    "ok"
}

async fn handle_openapi(State(state): State<ApiState>) -> impl IntoResponse {
    let default_url = format!(
        "http://{}:{}",
        state.http_config.host, state.http_config.port
    );
    let server_url = state
        .http_config
        .plugin
        .public_url
        .as_deref()
        .unwrap_or(&default_url);
    let spec = crate::openapi::build_openapi_spec(server_url);
    Json(spec)
}

async fn handle_plugin_manifest(State(state): State<ApiState>) -> impl IntoResponse {
    let default_url = format!(
        "http://{}:{}",
        state.http_config.host, state.http_config.port
    );
    let server_url = state
        .http_config
        .plugin
        .public_url
        .as_deref()
        .unwrap_or(&default_url);
    let manifest = crate::openapi::build_plugin_manifest(&state.http_config, server_url);
    Json(manifest)
}

// ---------------------------------------------------------------------------
// Read endpoint handlers
// ---------------------------------------------------------------------------

/// Whether an error message is a caller's own scope typo, which is a bad
/// request rather than a server fault. `check_terms` gives the caller the
/// nearest tag or folder in the message, the cheapest honest signal this far
/// from where the error is built (#60, #65).
fn is_scope_typo(message: &str) -> bool {
    message.starts_with("no such tag") || message.starts_with("no such folder")
}

async fn handle_search(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Search>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;

    // `full` and `summaries` both name the whole result set and disagree on
    // its shape, so asking for both is a usage error rather than one flag
    // silently winning. Checked before the pipeline runs, so a caller's
    // typo fails fast instead of paying for embed+retrieve+rerank first
    // (#35).
    if body.full && body.summaries {
        return Err(ApiError::bad_request(
            "--full and --summaries are mutually exclusive",
        ));
    }

    // Per call, with the configured default behind it (#62).
    let top_n = body.top_n.unwrap_or(state.top_n);
    let all_terms = crate::tags::merge_scope_alias(body.scope, body.all);
    let scope = crate::tags::Scope::parse(&all_terms, &body.any, &body.none)
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;

    let mut rerank_guard = match &state.reranker {
        Some(r) => Some(r.lock().await),
        None => None,
    };

    let mut config = search::SearchConfig {
        reranker: rerank_guard
            .as_mut()
            .map(|g| g.as_mut() as &mut dyn RerankModel),
        store: &store,
        rerank_candidates: 30,
        rerank: state.rerank,
        max_chunks_per_file: state.max_chunks_per_file,
        // Per call, with the process setting as the default: one query answers
        // the same way whoever asks it, and the granularity is part of the
        // question rather than of how the server was started (#62).
        group_by: body.group_by.unwrap_or(state.group_by),
        ranking: state.ranking,
        lane_weights: state.lane_weights,
        fts: state.fts,
        scope,
    };

    let output = search::search_with_intelligence(&body.query, top_n, &mut *embedder, &mut config)
        .map_err(|e| {
            // An unknown tag or folder is a caller's typo, not a server
            // fault. The message text is the cheapest honest signal
            // check_terms gives a caller this far from the error's
            // construction (#60, #65).
            if is_scope_typo(&e.to_string()) {
                ApiError::bad_request(&format!("{e:#}"))
            } else {
                ApiError::internal(&format!("{e:#}"))
            }
        })?;

    // Per call, with the configured default behind it, the same pattern
    // `top_n` follows (#35, #62).
    let budget = body.budget_tokens.unwrap_or(state.output.budget_tokens);
    let mut env = crate::packaging::assemble(
        &output.results,
        crate::packaging::AssembleParams {
            budget_tokens: budget,
            full: body.full,
            summaries: body.summaries,
            degraded: output.degraded,
            per_note_cap: state.ranking.per_note_cap,
        },
    );
    // A number invites a caller to trust it as ground truth rather than as a
    // reranker's opinion, so it ships only when asked (#35).
    if body.scores {
        crate::packaging::apply_scores(&mut env, &output.results);
    }
    let mut value =
        serde_json::to_value(&env).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    // The per-lane detail rides beside the envelope, the way the CLI prints
    // it after the rendered results and MCP sends it as a second content
    // block. It is present only when the caller asked, because an agent
    // that did not ask must not have to read past it (#62).
    if body.explain {
        value["explain"] = serde_json::Value::String(search::explain_report(&output, top_n));
    }
    Ok(Json(value))
}

async fn handle_read(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(p): Query<crate::params::Read>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let note = context::context_read(&ctx, &p.file, p.section.as_deref()).map_err(|e| {
        // A file or a section the vault does not hold is the caller's own text
        // naming nothing, not a server fault — the rule `handle_search` and
        // `handle_list` already follow (#60). The message text is the cheapest
        // honest signal `context_read` gives a caller this far from the error's
        // construction.
        let message = e.to_string();
        if message.starts_with("Section not found") || message.starts_with("File not found") {
            ApiError::bad_request(&format!("{e:#}"))
        } else {
            ApiError::internal(&format!("{e:#}"))
        }
    })?;
    Ok(Json(serde_json::json!(note)))
}

async fn handle_list(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<crate::params::List>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let all_terms = crate::tags::merge_scope_alias(params.scope, params.all);
    let filter = crate::tags::Scope::parse(&all_terms, &params.any, &params.none)
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let items = context::context_list(
        &ctx,
        &filter,
        params.created_by.as_deref(),
        params.limit,
        params.detailed,
    )
    .map_err(|e| {
        // An unknown tag or folder is a caller's typo, not a server fault.
        // The message text is the cheapest honest signal check_terms gives a
        // caller this far from the error's construction (#65).
        if is_scope_typo(&e.to_string()) {
            ApiError::bad_request(&format!("{e:#}"))
        } else {
            ApiError::internal(&format!("{e:#}"))
        }
    })?;
    Ok(Json(serde_json::json!(items)))
}

/// The vault's tag vocabulary, whole or under one term — the call to make
/// before filtering with `/api/list` (#61).
async fn handle_tags(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<crate::params::Tags>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let prefix = params.under.as_deref().and_then(crate::tags::parse_term);
    let rows = store
        .tags_under(prefix.as_ref())
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(rows)))
}

async fn handle_vault_map(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let ctx = ContextParams {
        store: &store,
        vault_path: &state.vault_path,
        profile: state.profile.as_ref().as_ref(),
    };
    let map = context::vault_map(&ctx).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(map)))
}

async fn handle_health(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let store = state.store.lock().await;
    let profile_ref = state.profile.as_ref().as_ref();
    let config = health::HealthConfig {
        daily_folder: profile_ref.and_then(|p| p.structure.folders.daily.clone()),
        inbox_folder: profile_ref.and_then(|p| p.structure.folders.inbox.clone()),
    };
    let report = health::generate_health_report(&store, &config)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!(report)))
}

/// What the index holds. It reads and writes nothing, so it takes the read
/// permission, and it answers the fields the CLI's `status --json` prints —
/// one composer for the three surfaces (#62).
async fn handle_status(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let data_dir =
        crate::config::Config::data_dir().map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    // The store is this server's own, so the reads see one snapshot and no
    // second connection runs the schema batch against the writer.
    let store = state.store.lock().await;
    let report = search::status_json(&store, &data_dir)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(report))
}

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

/// Record a write to the recent-writes map so the file watcher skips re-indexing.
async fn record_write(recent_writes: &RecentWrites, path: &std::path::Path) {
    if let Ok(meta) = std::fs::metadata(path)
        && let Ok(mtime) = meta.modified()
    {
        recent_writes.lock().await.insert(path.to_path_buf(), mtime);
    }
}

// ---------------------------------------------------------------------------
// Write endpoint handlers
// ---------------------------------------------------------------------------

async fn handle_create(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Create>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    if state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    // No stdin exists on this surface, so an omitted content is an error
    // here instead of the CLI's fallback read.
    let content = body
        .content
        .ok_or_else(|| ApiError::bad_request("content is required"))?;
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;
    let input = CreateNoteInput {
        content,
        filename: body.filename,
        type_hint: body.type_hint,
        tags: body.tags,
        folder: body.folder,
        created_by: "http-api".into(),
        auto_link: body.auto_link,
    };
    let result = writer::create_note(
        input,
        &store,
        &mut *embedder,
        state.index_settings.embed,
        state.index_settings.chunk,
        &state.vault_path,
        state.profile.as_ref().as_ref(),
    )
    .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

/// One capability for every change to an existing note (#62). The whole
/// edit list is read before anything is written, so a request that names an
/// impossible target answers 400 and writes nothing.
async fn handle_update(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Update>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    if state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    let edits = body
        .to_writer_edits()
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let store = state.store.lock().await;
    let input = UpdateInput {
        file: body.file,
        edits,
    };
    let result = writer::update_note(&store, &state.vault_path, &input)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    // `update_note` stores the new content hash and writes no chunks, so
    // nothing else will re-derive them: not `diff_vault`, which sees a hash
    // that already matches disk, and not the watcher, which the `record_write`
    // below tells to skip this file. Re-index here or the note stays
    // searchable only as the text it held before the edit (#62).
    let mut embedder = state.embedder.lock().await;
    // A failure here happens after the write, so a bare 500 would read as
    // "nothing happened" — say what did. Returning early also skips
    // `record_write` below, so the watcher's event on this file is not
    // suppressed and it re-indexes it on its own; that recovery is
    // deliberate, not accidental.
    crate::indexer::reindex_written_file(
        &result.path,
        &store,
        &mut *embedder,
        &state.vault_path,
        state.index_settings,
    )
    .with_context(|| {
        format!(
            "the file was written; its index rows were not updated for {}",
            result.path
        )
    })
    .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_move(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Move>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    if state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    let store = state.store.lock().await;
    let result = writer::move_note(&body.file, &body.new_folder, &store, &state.vault_path)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

/// Archive a note, or restore one previously archived with `undo: true`.
/// Archiving and restoring are one operation and its reverse, so they are
/// one capability with a flag rather than two routes (#62).
async fn handle_archive(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Archive>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    if state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    let store = state.store.lock().await;
    let result = if body.undo {
        let mut embedder = state.embedder.lock().await;
        writer::unarchive_note(
            &body.file,
            &store,
            &mut *embedder,
            state.index_settings.embed,
            state.index_settings.chunk,
            &state.vault_path,
        )
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?
    } else {
        writer::archive_note(
            &body.file,
            &store,
            &state.vault_path,
            state.profile.as_ref().as_ref(),
        )
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?
    };
    let full_path = state.vault_path.join(&result.path);
    record_write(&state.recent_writes, &full_path).await;
    Ok(Json(serde_json::json!(result)))
}

// ---------------------------------------------------------------------------
// Migration endpoint handlers
// ---------------------------------------------------------------------------

async fn handle_migrate(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Migrate>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    // The CLI already took a mode. MCP and HTTP split it into three names,
    // which is the same capability spelled three ways (#62). The mode is read
    // before the read-only guard, so that one word means the same thing on
    // both servers: `preview` writes nothing and runs, and a word that names
    // no operation is answered as such rather than as a refused write.
    match body.mode.as_str() {
        "preview" => {
            let profile_ref = state.profile.as_ref().as_ref();
            let preview = crate::migrate::generate_preview(&store, &state.vault_path, profile_ref)
                .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
            Ok(Json(serde_json::to_value(&preview).unwrap()))
        }
        "apply" => {
            if state.read_only {
                return Err(ApiError::forbidden(
                    "Write operations disabled in read-only mode",
                ));
            }
            let preview = crate::migrate::resolve_preview(body.preview)
                .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
            let result = crate::migrate::apply_preview(&preview, &store, &state.vault_path)
                .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
            Ok(Json(serde_json::to_value(&result).unwrap()))
        }
        "undo" => {
            if state.read_only {
                return Err(ApiError::forbidden(
                    "Write operations disabled in read-only mode",
                ));
            }
            let result = crate::migrate::undo_last(&store, &state.vault_path)
                .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
            Ok(Json(serde_json::to_value(&result).unwrap()))
        }
        // The mode is the caller's own text, so a word that names no
        // operation is a bad request and not an internal fault.
        other => Err(ApiError::bad_request(&format!(
            "Unknown mode: {other}. Use 'preview', 'apply' or 'undo'."
        ))),
    }
}

async fn handle_delete(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Delete>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    if state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    let store = state.store.lock().await;
    let mode = DeleteMode::from(body.mode);
    let archive_folder = state
        .profile
        .as_ref()
        .as_ref()
        .and_then(|p| p.structure.folders.archive.as_deref())
        .unwrap_or("04-Archive");
    writer::delete_note(&store, &state.vault_path, &body.file, mode, archive_folder)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!({
        "deleted": body.file,
        "mode": body.mode,
    })))
}

/// Index the server's vault.
///
/// An agent that writes a batch of notes needs a way to rebuild the whole
/// index, and a multi-minute call is acceptable for that (#62). It writes the
/// index, so it takes the write permission; the vault it walks is the one the
/// server was started on, and no caller-supplied path reaches it.
async fn handle_index(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Index>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    // A read-only server refuses it like any other write, the way MCP's
    // `index` does: `rebuild: true` discards the index before it builds one
    // again, so this destroys derived state and stalls every other call while
    // it runs (#62).
    if state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;
    let mut config = crate::config::Config::load().unwrap_or_default();
    if body.no_gitignore {
        config.respect_gitignore = false;
    }
    // The index-time settings come from the session, not this load: the
    // signature asks for them, so a fresh `Config::load` cannot be a second
    // source of the store's chunking or vector space (#55, #72). This load
    // supplies only the other index fields, such as `respect_gitignore`.
    let result = crate::indexer::run_index_shared(
        &state.vault_path,
        &config,
        state.index_settings,
        &store,
        &mut *embedder,
        body.rebuild,
        state.profile.as_ref().as_ref(),
    )
    .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!({
        "new_files": result.new_files,
        "updated_files": result.updated_files,
        "deleted_files": result.deleted_files,
        "total_chunks": result.total_chunks,
        "duration_secs": result.duration.as_secs_f64(),
    })))
}

async fn handle_reindex_file(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::ReindexFile>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    let store = state.store.lock().await;
    let mut embedder = state.embedder.lock().await;

    // One helper packages the six steps this used to spell out, so the three
    // callers cannot drift apart (#62). A file the server cannot read is the
    // caller's own text naming nothing, which is the 400 the neighbouring
    // handlers answer for that class (#60).
    let result = crate::indexer::reindex_written_file(
        &body.file,
        &store,
        &mut *embedder,
        &state.vault_path,
        state.index_settings,
    )
    .map_err(|e| match e.downcast_ref::<std::io::Error>() {
        Some(_) => ApiError::bad_request(&format!("Cannot read file {}: {e:#}", body.file)),
        None => ApiError::internal(&format!("{e:#}")),
    })?;

    Ok(Json(serde_json::json!({
        "file": body.file,
        "chunks": result.total_chunks,
        "docid": result.docid,
    })))
}

// ---------------------------------------------------------------------------
// Identity / init endpoint handlers
// ---------------------------------------------------------------------------

/// The identity block, optionally re-extracted first.
///
/// `refresh` is a parameter of the capability on every surface (#62). It
/// re-reads the L1 facts from the store the server already holds, so an agent
/// whose session started before the last write can ask for current ones
/// without a full re-index.
async fn handle_identity(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(p): Query<crate::params::Identity>,
) -> Result<impl IntoResponse, ApiError> {
    // Re-extraction clears the `identity_facts` rows and derives them again,
    // so it takes the write permission and a read-only server refuses it; the
    // block itself is a read either way.
    authorize(&headers, &state, p.refresh)?;
    if p.refresh && state.read_only {
        return Err(ApiError::forbidden(
            "Write operations disabled in read-only mode",
        ));
    }
    let store = state.store.lock().await;
    if p.refresh {
        let profile = state.profile.as_ref().as_ref().ok_or_else(|| {
            ApiError::bad_request("No vault profile found. Run `engraph init` first.")
        })?;
        crate::identity::extract_l1_facts(&store, profile)
            .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    }
    let config = crate::config::Config::load().unwrap_or_default();
    let block = crate::identity::format_identity_block(&config, &store)
        .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(serde_json::json!({ "identity": block })))
}

async fn handle_init(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<crate::params::Init>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    match body.mode.as_deref() {
        Some("detect") => {
            let result = crate::onboarding::run_detect_json(&state.vault_path)
                .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
            Ok(Json(result))
        }
        Some("apply") => {
            // `apply` indexes the vault, which is the work `index` is guarded
            // against on a read-only server. The mode is read first, so
            // `detect` — which writes nothing — still runs (#62).
            if state.read_only {
                return Err(ApiError::forbidden(
                    "Write operations disabled in read-only mode",
                ));
            }
            let mut config = crate::config::Config::load().unwrap_or_default();
            // `apply` indexes the whole vault. The index-time settings come from
            // the session, not this load: `run_apply_json` asks for them, so a
            // fresh load that fell back to the defaults, or drifted from disk,
            // cannot build the index at a chunking or a vector space the rest of
            // the session does not use — the divergence nothing downstream can
            // tell apart (#55, #72). This load supplies only the identity and
            // profile fields `apply` writes.
            let data_dir = crate::config::Config::data_dir()
                .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
            let flags = crate::onboarding::ApplyFlags {
                name: body.name,
                role: body.role,
                purpose: body.purpose,
                identity_only: false,
                reindex_only: false,
            };
            let result = crate::onboarding::run_apply_json(
                &state.vault_path,
                &mut config,
                state.index_settings,
                &data_dir,
                flags,
            )
            .map_err(|e| ApiError::internal(&format!("{e:#}")))?;
            Ok(Json(result))
        }
        Some(other) => Err(ApiError::bad_request(&format!(
            "Unknown mode: {other}. Use 'detect' or 'apply'."
        ))),
        // A server has no interactive flow, so `init` there needs a mode.
        // The CLI's no-mode form is its own prompt sequence and reaches no
        // surface but the CLI (#62).
        None => Err(ApiError::bad_request(
            "init needs mode=detect or mode=apply",
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::SystemTime;

    use axum::body::Body;
    use tower::ServiceExt;

    fn test_http_config() -> HttpConfig {
        HttpConfig {
            enabled: true,
            port: 3000,
            host: "127.0.0.1".to_string(),
            rate_limit: 0,
            cors_origins: vec![],
            api_keys: vec![
                ApiKeyConfig {
                    key: "eg_readkey".into(),
                    name: "reader".into(),
                    permissions: "read".into(),
                },
                ApiKeyConfig {
                    key: "eg_writekey".into(),
                    name: "writer".into(),
                    permissions: "write".into(),
                },
            ],
            plugin: crate::config::PluginConfig::default(),
        }
    }

    /// Dummy embedder that returns zero vectors. Only used for constructing
    /// `ApiState` in tests that don't exercise search/context endpoints.
    struct DummyEmbedder;
    impl crate::llm::EmbedModel for DummyEmbedder {
        fn embed_batch(
            &mut self,
            docs: &[crate::llm::EmbedDoc<'_>],
        ) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(docs.iter().map(|_| vec![0.0; 384]).collect())
        }
        fn token_count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn dim(&self) -> usize {
            384
        }
        fn max_context(&self) -> usize {
            2048
        }
        fn fingerprint(&self) -> String {
            "dummy-embed".to_string()
        }
    }

    fn test_api_state() -> ApiState {
        test_api_state_at(PathBuf::from("/tmp/test-vault"))
    }

    fn test_api_state_at(vault_path: PathBuf) -> ApiState {
        let store = Store::open_memory().expect("in-memory store");
        let config = test_http_config();
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit));
        ApiState {
            store: Arc::new(Mutex::new(store)),
            embedder: Arc::new(Mutex::new(
                Box::new(DummyEmbedder) as Box<dyn EmbedModel + Send>
            )),
            vault_path: Arc::new(vault_path),
            profile: Arc::new(None),
            reranker: None,
            http_config: Arc::new(config),
            no_auth: false,
            recent_writes: Arc::new(Mutex::new(HashMap::<PathBuf, SystemTime>::new())),
            rate_limiter,
            read_only: false,
            max_chunks_per_file: crate::config::default_max_chunks_per_file(),
            group_by: crate::config::GroupBy::default(),
            top_n: crate::config::Config::default().top_n,
            rerank: crate::config::RerankConfig::default(),
            ranking: crate::config::RankingConfig::default(),
            lane_weights: crate::config::LaneWeights::default(),
            fts: crate::config::FtsConfig::default(),
            index_settings: crate::indexer::IndexSettings {
                chunk: crate::chunker::ChunkOptions {
                    min_chars: 0,
                    promote_bold: false,
                    carry_orphan_headings: false,
                },
                embed: crate::prefix::EmbedComposition::default(),
            },
            output: crate::config::OutputConfig::default(),
        }
    }

    #[test]
    fn test_validate_api_key_valid() {
        let config = test_http_config();
        let result = validate_api_key("eg_readkey", &config);
        assert!(result.is_some());
        assert_eq!(result.unwrap().permissions, "read");
    }

    #[test]
    fn test_validate_api_key_invalid() {
        let config = test_http_config();
        assert!(validate_api_key("eg_badkey", &config).is_none());
    }

    #[test]
    fn test_generate_api_key_format() {
        let key = generate_api_key();
        assert!(key.starts_with("eg_"));
        assert_eq!(key.len(), 35); // "eg_" + 32 hex chars
    }

    #[test]
    fn test_check_permission_read_on_read() {
        assert!(check_permission("read", false));
    }

    #[test]
    fn test_check_permission_read_on_write() {
        assert!(!check_permission("read", true));
    }

    #[test]
    fn test_check_permission_write_on_write() {
        assert!(check_permission("write", true));
    }

    #[test]
    fn test_check_permission_write_on_read() {
        assert!(check_permission("write", false));
    }

    // -----------------------------------------------------------------------
    // Integration tests using axum oneshot
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_vault_map_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_vault_map_invalid_key() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer eg_badkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_vault_map_authorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_authorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// `status` reads and writes nothing, so a read key reaches it and no key
    /// does not (#62).
    #[tokio::test]
    async fn status_takes_the_read_permission() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// `index` writes the index, so a read key is refused before the walk
    /// starts (#62).
    #[tokio::test]
    async fn index_takes_the_write_permission() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/index")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_search_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn a_search_scope_naming_no_tag_is_a_bad_request() {
        // #60. The caller's own text named nothing, so this is a 400 and not
        // the 500 every error on this route used to answer.
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::from(r#"{"query":"warding","all":["type/undead"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn a_search_scope_naming_no_folder_is_a_bad_request() {
        // #65. The caller's own text named a folder no note lives under, so
        // this is a 400, the same as an unknown tag.
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::from(r#"{"query":"warding","all":["/Nowhere/"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn a_search_body_with_an_explicit_null_scope_field_is_not_rejected() {
        // #60. `#[serde(default)]` on a bare `Vec<String>` covers a missing
        // field only. A client that serialises an absent optional as JSON
        // `null` — routine in JavaScript and Python — would fail to
        // deserialize and never reach `handle_search`, answering 422 instead
        // of running an unscoped search. `Option<Vec<String>>` reads `null`
        // the same way it reads a missing field.
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::from(r#"{"query":"warding","all":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_authorized_empty() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/list")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Three notes over four tags, written through the calls the indexer
    /// makes, so the tag endpoints answer against rows a vault could hold.
    async fn seed_tags(state: &ApiState) {
        let tag = |p: &str| crate::tags::Tag {
            path: p.to_string(),
            display: p.to_string(),
        };
        let store = state.store.lock().await;
        let wight = store
            .insert_file("wight.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let wolf = store
            .insert_file("wolf.md", "h2", 200, "bbb222", None, None)
            .unwrap();
        let draft = store
            .insert_file("draft.md", "h3", 300, "ccc333", None, None)
            .unwrap();
        store
            .reconcile_file_tags(wight, &[tag("type/undead"), tag("habitat/swamp")])
            .unwrap();
        store
            .reconcile_file_tags(wolf, &[tag("type/beast")])
            .unwrap();
        store
            .reconcile_file_tags(draft, &[tag("type/beast"), tag("status/draft")])
            .unwrap();
    }

    /// The response body as JSON, for the tests that read rows rather than
    /// a status code.
    async fn json_body(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn get(state: ApiState, uri: &str) -> axum::response::Response {
        build_router(state)
            .oneshot(
                axum::http::Request::builder()
                    .uri(uri)
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    /// The paths of an `/api/list` response, in the order it returned them.
    fn paths(items: &serde_json::Value) -> Vec<String> {
        items
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["path"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn test_tags_unauthorized() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_tags_returns_the_whole_vocabulary() {
        let state = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/tags").await;
        assert_eq!(response.status(), StatusCode::OK);
        let rows = json_body(response).await;
        let listed: Vec<&str> = rows
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["path"].as_str().unwrap())
            .collect();
        assert_eq!(
            listed,
            vec!["habitat/swamp", "status/draft", "type/beast", "type/undead"]
        );
        assert_eq!(rows[2]["note_count"], 2);
    }

    #[tokio::test]
    async fn test_tags_under_reads_a_bare_term_as_its_subtree() {
        let state = test_api_state();
        seed_tags(&state).await;
        let slash = json_body(get(state.clone(), "/api/tags?under=type/").await).await;
        let bare = json_body(get(state, "/api/tags?under=type").await).await;
        let listed: Vec<&str> = slash
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["path"].as_str().unwrap())
            .collect();
        assert_eq!(listed, vec!["type/beast", "type/undead"]);
        assert_eq!(bare, slash);
    }

    #[tokio::test]
    async fn test_list_any_matches_either_term() {
        let state = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?any=type/undead,status/draft").await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut listed = paths(&json_body(response).await);
        listed.sort();
        assert_eq!(listed, vec!["draft.md", "wight.md"]);
    }

    #[tokio::test]
    async fn test_list_none_excludes_its_terms() {
        let state = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?all=type/&none=status/draft").await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut listed = paths(&json_body(response).await);
        listed.sort();
        assert_eq!(listed, vec!["wight.md", "wolf.md"]);
    }

    #[tokio::test]
    async fn test_list_merges_tags_into_all() {
        let state = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?tags=type/beast&all=status/draft").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(paths(&json_body(response).await), vec!["draft.md"]);
    }

    #[tokio::test]
    async fn test_no_auth_mode_skips_check() {
        let mut state = test_api_state();
        state.no_auth = true;
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // -----------------------------------------------------------------------
    // Write endpoint permission tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_write_endpoint_read_key_rejected() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/create")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::from(r##"{"content":"# Test","filename":"test"}"##))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_write_endpoint_write_key_accepted() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/update")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_writekey")
                    .body(Body::from(
                        r#"{"file":"nonexistent","edits":[{"section":"Test","mode":"append","content":"new"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Should be 500 (file not found via store) but NOT 403
        assert_ne!(response.status(), StatusCode::FORBIDDEN);
    }

    // -----------------------------------------------------------------------
    // Rate limiter unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let limiter = RateLimiter::new(5);
        for _ in 0..5 {
            assert!(limiter.check("key1").is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_rejects_over_limit() {
        let limiter = RateLimiter::new(2);
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key1").is_err());
    }

    #[test]
    fn test_rate_limiter_unlimited() {
        let limiter = RateLimiter::new(0);
        for _ in 0..1000 {
            assert!(limiter.check("key1").is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_separate_keys() {
        let limiter = RateLimiter::new(1);
        assert!(limiter.check("key1").is_ok());
        assert!(limiter.check("key2").is_ok()); // different key, separate bucket
        assert!(limiter.check("key1").is_err()); // key1 exhausted
    }

    #[tokio::test]
    async fn test_rate_limit_returns_429() {
        let mut state = test_api_state();
        state.rate_limiter = Arc::new(RateLimiter::new(1));
        let app = build_router(state);
        // First request passes (consumes the single token)
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Second request gets 429
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer eg_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().get("retry-after").is_some());
    }

    // -----------------------------------------------------------------------
    // OpenAPI / Plugin manifest tests (no auth required)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_openapi_no_auth_required() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_plugin_manifest_no_auth_required() {
        let state = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/.well-known/ai-plugin.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // -----------------------------------------------------------------------
    // Mode routing (#62)
    // -----------------------------------------------------------------------

    /// POST `body` to `path` as a writer, and return the status and the body.
    async fn post_json(state: ApiState, path: &str, body: &str) -> (StatusCode, serde_json::Value) {
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer eg_writekey")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn each_migrate_mode_reaches_the_operation_it_names() {
        // One name takes three modes (#62), so the one thing worth proving is
        // that each mode string still arrives at the operation it names.
        // Each answer below can come from one of the three and no other.

        // `preview` classifies: an empty index proposes no move, and the
        // response is a preview and not a result.
        let (status, body) =
            post_json(test_api_state(), "/api/migrate", r#"{"mode":"preview"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.get("files").is_some(), "not a preview: {body}");
        assert_eq!(body["files"].as_array().unwrap().len(), 0);

        // `apply` executes the preview it is given: this one moves nothing,
        // and it answers with that preview's own id.
        let (status, body) = post_json(
            test_api_state(),
            "/api/migrate",
            r#"{"mode":"apply","preview":{"migration_id":"m-14","files":[],"uncertain":[],"skipped":0}}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["migration_id"], "m-14");
        assert_eq!(body["moved"], 0);

        // `undo` reads the migration log, which this store has no row in.
        let (status, body) =
            post_json(test_api_state(), "/api/migrate", r#"{"mode":"undo"}"#).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("No migration to undo"),
            "not the undo path: {body}"
        );
    }

    #[tokio::test]
    async fn read_only_refuses_the_migrate_modes_that_write_and_no_others() {
        // One name now carries all three modes, so the guard belongs to the
        // modes that write and not to the route. `preview` writes nothing, so
        // it runs here as it does on MCP, and an unknown mode is still
        // answered as an unknown mode (#62).
        let mut state = test_api_state();
        state.read_only = true;
        let (status, _) = post_json(state, "/api/migrate", r#"{"mode":"preview"}"#).await;
        assert_eq!(status, StatusCode::OK);

        let mut state = test_api_state();
        state.read_only = true;
        let (status, _) = post_json(
            state,
            "/api/migrate",
            r#"{"mode":"apply","preview":{"migration_id":"m","files":[],"uncertain":[],"skipped":0}}"#,
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        let mut state = test_api_state();
        state.read_only = true;
        let (status, _) = post_json(state, "/api/migrate", r#"{"mode":"undo"}"#).await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        let mut state = test_api_state();
        state.read_only = true;
        let (status, body) = post_json(state, "/api/migrate", r#"{"mode":"sideways"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"],
            "Unknown mode: sideways. Use 'preview', 'apply' or 'undo'."
        );
    }

    #[tokio::test]
    async fn a_migrate_mode_naming_nothing_is_a_bad_request() {
        // The mode is the caller's own text, so a word that names no
        // operation is a 400 and not a 500 (#62).
        let (status, body) =
            post_json(test_api_state(), "/api/migrate", r#"{"mode":"sideways"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"],
            "Unknown mode: sideways. Use 'preview', 'apply' or 'undo'."
        );
    }

    #[tokio::test]
    async fn init_detect_reaches_detection_and_writes_nothing() {
        // `detect` is the half of `init` a server can run without touching
        // the vault (#62): it reports what it found and leaves no file.
        let vault = tempfile::tempdir().unwrap();
        std::fs::write(vault.path().join("note.md"), "# Note\n").unwrap();
        let (status, body) = post_json(
            test_api_state_at(vault.path().to_path_buf()),
            "/api/init",
            r#"{"mode":"detect"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.get("structure").is_some(), "not a detection: {body}");
        let left: Vec<_> = std::fs::read_dir(vault.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 1, "detect wrote something: {left:?}");
    }

    #[tokio::test]
    async fn init_without_a_mode_is_a_bad_request() {
        // A server has no interactive flow, so the CLI's no-mode form has no
        // meaning here (#62).
        let (status, body) = post_json(test_api_state(), "/api/init", r#"{}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "init needs mode=detect or mode=apply");
    }

    #[tokio::test]
    async fn an_init_mode_naming_nothing_is_a_bad_request() {
        let (status, body) =
            post_json(test_api_state(), "/api/init", r#"{"mode":"sideways"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"],
            "Unknown mode: sideways. Use 'detect' or 'apply'."
        );
    }

    /// A vault of two notes, indexed in memory. The mock's vectors are hashes,
    /// so the keyword lane carries the meaning here — which is all a
    /// granularity assertion needs.
    fn indexed_state() -> (tempfile::TempDir, ApiState) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("rules")).unwrap();
        std::fs::write(
            root.join("rules/abjuration-spells.md"),
            "# Abjuration\n\n\
             ## Level 3 Counterspell\n\nA warding effect that stops a spell mid-cast. \
             It interrupts the casting itself and does nothing to a spell already in effect.\n\n\
             ## Level 5 Dispel Magic\n\nA warding effect that ends an ongoing spell. \
             It reaches an effect already in place and cannot interrupt one \
             that is still being cast, which is the whole of the difference.\n\n\
             ## Level 9 Dimensional Anchor\n\nA warding effect that pins a creature. \
             It closes every route out of the space the creature \
             currently stands in, and it does not care how that route was opened.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("rules/evocation-spells.md"),
            "# Evocation\n\n## Level 1 Firebolt\n\nA bolt of flame.\n",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        crate::indexer::run_index_shared(
            root,
            &crate::config::Config::default(),
            crate::indexer::IndexSettings::from_config(&crate::config::Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let mut state = test_api_state_at(root.to_path_buf());
        state.store = Arc::new(Mutex::new(store));
        state.embedder = Arc::new(Mutex::new(Box::new(embedder) as Box<dyn EmbedModel + Send>));
        (tmp, state)
    }

    /// How many sections of the one file that holds three matching ones came
    /// back, across the included blocks and the budget's overflow alike —
    /// this counts answers, not what fit under the default budget.
    fn sections_of_the_abjuration_note(body: &serde_json::Value) -> usize {
        let in_array = |key: &str| {
            body[key]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["path"] == "rules/abjuration-spells.md")
                .count()
        };
        in_array("blocks") + in_array("overflow")
    }

    #[tokio::test]
    async fn a_search_takes_its_granularity_from_the_call() {
        // `group_by` is per call, with the process setting as the default
        // (#62). The server here is started on `file`, so a call that names
        // `chunk` proves the override rather than the default.
        let (_tmp, mut state) = indexed_state();
        state.group_by = crate::config::GroupBy::File;
        // This test asserts per-section output. That output is below
        // coalescing. Coalescing has its own tests (#39).
        state.ranking.coalesce_adjacent = false;

        let (status, body) =
            post_json(state.clone(), "/api/search", r#"{"query":"warding"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(sections_of_the_abjuration_note(&body), 1, "got {body}");

        let (status, body) = post_json(
            state,
            "/api/search",
            r#"{"query":"warding","group_by":"chunk"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(sections_of_the_abjuration_note(&body) > 1, "got {body}");
    }

    #[tokio::test]
    async fn the_per_lane_detail_answers_the_call_that_asked_for_it() {
        // `explain` is per call on all three surfaces (#62). A caller that did
        // not ask reads no explain field at all, so the detail costs the
        // callers who did not want it nothing.
        let (_tmp, state) = indexed_state();
        let (status, body) =
            post_json(state.clone(), "/api/search", r#"{"query":"warding"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.get("explain").is_none());

        let (status, body) = post_json(
            state,
            "/api/search",
            r#"{"query":"warding","explain":true}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body["explain"]
                .as_str()
                .unwrap()
                .contains("--- Query run ---")
        );
    }

    /// A section the note does not hold is the caller's own text naming
    /// nothing, so it is a 400 and not the 500 every error on this route used
    /// to answer. It is the first error an agent using the newly documented
    /// `section` parameter meets (#60, #62).
    #[tokio::test]
    async fn a_read_of_a_section_the_note_does_not_hold_is_a_bad_request() {
        let (_tmp, state) = indexed_state();

        let response = get(
            state.clone(),
            "/api/read?file=rules/abjuration-spells.md&section=Level%203%20Counterspell",
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        let response = get(
            state.clone(),
            "/api/read?file=rules/abjuration-spells.md&section=Nowhere",
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        // A file the vault does not hold is the same class of mistake.
        let response = get(state, "/api/read?file=nowhere.md").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// `archive` and `archive {undo: true}` are one operation and its reverse
    /// (#62). The handler's own branch chooses `archive_note` against
    /// `unarchive_note`, and nothing else covers it — an inverted branch would
    /// move the file the opposite way with the whole suite green.
    #[tokio::test]
    async fn the_undo_flag_chooses_the_operation_it_names() {
        let (_tmp, state) = indexed_state();
        let vault = state.vault_path.as_ref().clone();
        let live = vault.join("rules/evocation-spells.md");
        let archived = vault.join("04-Archive/rules/evocation-spells.md");
        assert!(live.exists());

        let (status, _) = post_json(
            state.clone(),
            "/api/archive",
            r#"{"file":"rules/evocation-spells.md"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!live.exists(), "undo: false must archive");
        assert!(archived.exists(), "undo: false must archive");

        let (status, _) = post_json(
            state,
            "/api/archive",
            r#"{"file":"04-Archive/rules/evocation-spells.md","undo":true}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(live.exists(), "undo: true must restore");
        assert!(!archived.exists(), "undo: true must restore");
    }

    /// A read-only server refuses `index` the way MCP's `index` refuses it:
    /// `rebuild: true` discards derived state and stalls every other call
    /// while it runs (#62).
    #[tokio::test]
    async fn a_read_only_server_refuses_index() {
        let mut state = test_api_state();
        state.read_only = true;
        let (status, _) = post_json(state, "/api/index", r#"{}"#).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    /// `init {mode: apply}` indexes the vault, which is the work `index` is
    /// guarded against. `detect` writes nothing and still runs (#62).
    #[tokio::test]
    async fn a_read_only_server_refuses_init_apply_and_runs_init_detect() {
        let vault = tempfile::tempdir().unwrap();
        std::fs::write(vault.path().join("note.md"), "# Note\n").unwrap();

        let mut state = test_api_state_at(vault.path().to_path_buf());
        state.read_only = true;
        let (status, _) = post_json(state, "/api/init", r#"{"mode":"apply"}"#).await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        let mut state = test_api_state_at(vault.path().to_path_buf());
        state.read_only = true;
        let (status, _) = post_json(state, "/api/init", r#"{"mode":"detect"}"#).await;
        assert_eq!(status, StatusCode::OK);
    }

    /// A server's `apply` acts on the plan its caller sends and no other. The
    /// copy `engraph migrate --mode preview` saves belongs to the CLI's own
    /// two-step flow, and an `apply` that fell back to it would move files
    /// against a plan this caller never saw (#62).
    #[tokio::test]
    async fn a_migrate_apply_with_no_preview_is_a_bad_request() {
        let (status, body) =
            post_json(test_api_state(), "/api/migrate", r#"{"mode":"apply"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"].as_str().unwrap().contains("needs a preview"),
            "got {body}"
        );
    }

    /// `identity` takes `refresh` on every surface (#62). Before this the
    /// route had no extractor at all, so the flag the CLI honoured had no
    /// spelling here. `extract_l1_facts` clears tier 1 before it derives it
    /// again, so a stale fact seeded first is what proves the call was made.
    #[tokio::test]
    async fn identity_refresh_re_extracts_the_l1_facts() {
        let (_tmp, mut state) = indexed_state();
        let root = state.vault_path.as_ref().clone();
        state.profile = Arc::new(Some(crate::profile::VaultProfile {
            vault_path: root,
            vault_type: crate::profile::VaultType::Obsidian,
            structure: crate::profile::StructureDetection {
                method: crate::profile::StructureMethod::Para,
                folders: crate::profile::FolderMap::default(),
            },
            stats: crate::profile::VaultStats::default(),
        }));

        let stale = |state: &ApiState| {
            let store = state.store.try_lock().expect("uncontended");
            store
                .get_identity_facts(1)
                .unwrap()
                .into_iter()
                .any(|f| f.key == "stale")
        };
        {
            let store = state.store.try_lock().expect("uncontended");
            store
                .upsert_identity_fact(1, "stale", "from an older session", None)
                .unwrap();
        }
        assert!(stale(&state));

        // A read key reaches the block itself and no re-extraction happens.
        let response = get(state.clone(), "/api/identity").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(stale(&state), "a call that did not ask re-extracts nothing");

        // Re-extraction writes rows, so a read key is refused.
        let response = get(state.clone(), "/api/identity?refresh=true").await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(stale(&state));

        let response = build_router(state.clone())
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/identity?refresh=true")
                    .header("authorization", "Bearer eg_writekey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!stale(&state), "refresh=true must re-derive tier 1");
    }

    /// Five notes that all answer one query. Five is more than the `top_n` the
    /// R21 test configures, so a truncation reads as a truncation and not as a
    /// corpus that had no more to give (#62). Each body is well over
    /// `chunk_min_chars`, so each note is one chunk of its own.
    fn state_over_five_answering_notes() -> (tempfile::TempDir, ApiState) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        for (i, subject) in ["counterspell", "dispel", "anchor", "ward", "seal"]
            .iter()
            .enumerate()
        {
            std::fs::write(
                root.join(format!("{i}-{subject}.md")),
                format!(
                    "# The {subject} rule\n\nA warding effect. Every warding effect in this \
                     ruleset states what it stops, when it may be cast, and what it leaves \
                     alone, and the {subject} rule is one of them among several others.\n"
                ),
            )
            .unwrap();
        }

        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        crate::indexer::run_index_shared(
            root,
            &crate::config::Config::default(),
            crate::indexer::IndexSettings::from_config(&crate::config::Config::default()),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();

        let mut state = test_api_state_at(root.to_path_buf());
        state.store = Arc::new(Mutex::new(store));
        state.embedder = Arc::new(Mutex::new(Box::new(embedder) as Box<dyn EmbedModel + Send>));
        (tmp, state)
    }

    /// R21 (#62): the number of results a call that names no `top_n` gets is
    /// the configured one, and not a literal this server holds. A state built
    /// at three answers three, and the same state answers more when the call
    /// asks for more — which is what separates the configured default from a
    /// corpus that ran out.
    #[tokio::test]
    async fn a_search_that_names_no_top_n_gets_the_configured_number() {
        let (_tmp, mut state) = state_over_five_answering_notes();
        state.top_n = 3;

        // The count is blocks plus overflow: `top_n` bounds how many answers
        // the pipeline returns, before the budget decides which of them carry
        // text.
        let count = |body: &serde_json::Value| {
            body["blocks"].as_array().unwrap().len() + body["overflow"].as_array().unwrap().len()
        };

        let (status, body) =
            post_json(state.clone(), "/api/search", r#"{"query":"warding"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(count(&body), 3, "the configured top_n is 3, got {body}");

        let (status, body) =
            post_json(state, "/api/search", r#"{"query":"warding","top_n":5}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            count(&body) > 3,
            "the corpus holds more than three answers, got {body}"
        );
    }

    /// `full` and `summaries` both name the whole result set and disagree on
    /// its shape, so asking for both is the caller's own contradiction and a
    /// 400, not one flag silently winning (#35).
    #[tokio::test]
    async fn full_and_summaries_together_is_a_bad_request() {
        let (_tmp, state) = indexed_state();
        let (status, _body) = post_json(
            state,
            "/api/search",
            r#"{"query":"warding","full":true,"summaries":true}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// No numeric score reaches the wire by default; `scores` restores it on
    /// every block and overflow row (#35). A reranker is wired in so the run
    /// is not degraded — a degraded row reports no probability at all, and
    /// asserting against one here would pass whether or not `scores` worked.
    #[tokio::test]
    async fn scores_is_absent_by_default_and_present_when_asked() {
        let (_tmp, mut state) = indexed_state();
        state.reranker = Some(Arc::new(Mutex::new(
            Box::new(crate::llm::MockLlm::new(256)) as Box<dyn RerankModel + Send>,
        )));
        // The mock's Jaccard scores run well under the real cross-encoder's
        // range, and the default answer floor exists to gate a real model's
        // probability — not this fixture's stand-in. Zero it so the query
        // still answers (#34's floor is exercised in its own tests).
        state.ranking.answer_floor = 0.0;

        let (status, body) =
            post_json(state.clone(), "/api/search", r#"{"query":"warding"}"#).await;
        assert_eq!(status, StatusCode::OK);
        let blocks = body["blocks"].as_array().unwrap();
        assert!(!blocks.is_empty(), "got {body}");
        assert!(
            blocks.iter().all(|b| b.get("score").is_none()),
            "a block carried a score with no --scores, got {body}"
        );

        let (status, body) =
            post_json(state, "/api/search", r#"{"query":"warding","scores":true}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["degraded"], false, "got {body}");
        let blocks = body["blocks"].as_array().unwrap();
        assert!(!blocks.is_empty(), "got {body}");
        assert!(
            blocks.iter().all(|b| b["score"].is_number()),
            "--scores must fill a number on every block, got {body}"
        );
    }
}
