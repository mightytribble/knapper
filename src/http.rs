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
use tower_http::cors::{Any, CorsLayer};

use crate::config::{ApiKeyConfig, HttpConfig};
use crate::context::{self, ContextParams};
use crate::core::Core;
use crate::health;
use crate::search;
use crate::writer::{self, CreateNoteInput, DeleteMode, UpdateInput};

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ApiState {
    /// What this server shares with the MCP server and the watcher.
    pub core: Core,
    pub http_config: Arc<HttpConfig>,
    pub rate_limiter: Arc<RateLimiter>,
    pub no_auth: bool,
    /// `[http] request_timeout_secs` as a duration, `None` for `0`. The
    /// router's timeout layer and the write routes' wait for the core both
    /// read it.
    pub request_timeout: Option<std::time::Duration>,
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
    /// The fault's kind, from `Fault::kind`, or one of this transport's own:
    /// `unauthorized`, `forbidden`, `rate_limited`, `internal`.
    pub kind: &'static str,
    pub headers: Vec<(String, String)>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        if self.status == StatusCode::REQUEST_TIMEOUT {
            return self.status.into_response();
        }
        let body = serde_json::json!({ "error": self.message, "kind": self.kind });
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
    /// The kinds this transport builds itself, with no `Fault` behind them.
    /// With `Fault::KINDS` it is every word the `kind` field can hold.
    pub const TRANSPORT_KINDS: &'static [&'static str] =
        &["unauthorized", "forbidden", "rate_limited", "internal"];

    fn new(status: StatusCode, kind: &'static str, message: &str) -> Self {
        Self {
            status,
            message: message.to_string(),
            kind,
            headers: vec![],
        }
    }
    /// The request ran past `request_timeout_secs` before it started. It has
    /// no body, like the timeout layer's own 408, so its kind is never read.
    pub fn timed_out() -> Self {
        Self::new(StatusCode::REQUEST_TIMEOUT, "timed_out", "")
    }
    pub fn unauthorized(msg: &str) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "unauthorized", msg)
    }
    pub fn forbidden(msg: &str) -> Self {
        Self::new(StatusCode::FORBIDDEN, "forbidden", msg)
    }
    /// The handler's own parse stage: the caller's text read before any core
    /// call. What comes out of a core call goes through `From` instead.
    pub fn bad_request(msg: &str) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_input", msg)
    }
    pub fn not_found(msg: &str) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", msg)
    }
    pub fn internal(msg: &str) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", msg)
    }
    pub fn rate_limited(retry_after: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: format!("Rate limit exceeded. Retry after {retry_after}s"),
            kind: "rate_limited",
            headers: vec![("retry-after".to_string(), retry_after.to_string())],
        }
    }
    /// A known route asked with the wrong method. The route exists and the
    /// method is the caller's text, so the kind is `invalid_input`.
    pub fn method_not_allowed() -> Self {
        Self::new(
            StatusCode::METHOD_NOT_ALLOWED,
            "invalid_input",
            "method not allowed on this route",
        )
    }
}

/// The one place an error from a core call becomes a status.
///
/// The kind is read once, through whatever context the pipeline added. An
/// error with no `Fault` in its chain is the server's own, and a 500 keeps
/// the whole chain in its body: this surface serves one local agent, and the
/// chain is the most useful thing it can read.
impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        use crate::fault::Fault;
        let message = format!("{e:#}");
        match Fault::of(&e) {
            Some(fault) => {
                let status = match fault {
                    Fault::InvalidInput(_) | Fault::Ambiguous(_) => StatusCode::BAD_REQUEST,
                    Fault::NotFound(_) => StatusCode::NOT_FOUND,
                    Fault::Conflict(_) => StatusCode::CONFLICT,
                    Fault::ReadOnly => StatusCode::FORBIDDEN,
                    Fault::StaleIndex(_) => StatusCode::INTERNAL_SERVER_ERROR,
                };
                Self::new(status, fault.kind(), &message)
            }
            None => Self::internal(&message),
        }
    }
}

// ---------------------------------------------------------------------------
// Extractors whose rejection is an ApiError
// ---------------------------------------------------------------------------

/// `Json<T>` whose rejection is an `ApiError`: 400, `invalid_input`, axum's
/// own text as the message. Every POST handler takes its body through this,
/// so malformed JSON, an unknown enum word and a missing field answer the
/// same body shape every other error does.
pub struct ApiJson<T>(pub T);

impl<T> axum::extract::FromRequest<ApiState> for ApiJson<T>
where
    T: serde::de::DeserializeOwned,
{
    type Rejection = ApiError;

    async fn from_request(
        req: axum::extract::Request,
        state: &ApiState,
    ) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ApiJson(value)),
            Err(rejection) if body_read_timed_out(&rejection) => Err(ApiError::timed_out()),
            Err(rejection) => Err(ApiError::bad_request(&rejection.body_text())),
        }
    }
}

/// Whether a body rejection came from the write routes' body timeout, which
/// means the write did not run and answers like the lock wait's 408.
fn body_read_timed_out(rejection: &(dyn std::error::Error + 'static)) -> bool {
    let mut source = Some(rejection);
    while let Some(err) = source {
        if err.is::<tower_http::timeout::TimeoutError>() {
            return true;
        }
        source = err.source();
    }
    false
}

/// `Query<T>` the same way, for every GET handler.
pub struct ApiQuery<T>(pub T);

impl<T> axum::extract::FromRequestParts<ApiState> for ApiQuery<T>
where
    T: serde::de::DeserializeOwned,
{
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &ApiState,
    ) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(value)) => Ok(ApiQuery(value)),
            Err(rejection) => Err(ApiError::bad_request(&rejection.body_text())),
        }
    }
}

/// An unknown path. The transport's own 404, with a body like every other.
async fn unknown_route() -> ApiError {
    ApiError::not_found("no such route")
}

/// A known path with the wrong method.
async fn wrong_method() -> ApiError {
    ApiError::method_not_allowed()
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

/// Generate a new API key with `kn_` prefix + 32 hex chars.
pub fn generate_api_key() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let hex: String = (0..32)
        .map(|_| format!("{:x}", rng.random_range(0..16u8)))
        .collect();
    format!("kn_{hex}")
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
        ("/api/match", post(handle_match)),
        ("/api/read", get(handle_read)),
        ("/api/list", get(handle_list)),
        ("/api/tags", get(handle_tags)),
        ("/api/properties", get(handle_properties)),
        ("/api/vault-map", get(handle_vault_map)),
        ("/api/health", get(handle_health)),
        ("/api/validate", post(handle_validate)),
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
        // Setup
        ("/api/init", post(handle_init)),
        // The transport describing itself (no auth required)
        ("/openapi.json", get(handle_openapi)),
    ]
}

/// Bodies larger than this are refused before any handler reads them.
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Requests in flight at once; the rest wait. One agent, maybe a few calls.
pub const MAX_IN_FLIGHT: usize = 16;
/// Routes that index the vault to completion and are never timed out.
pub const UNTIMED_ROUTES: &[&str] = &["/api/index", "/api/init"];
/// Routes that write. They are timed while they wait for the core and not
/// after, so a 408 on one means it did not run. See [`core_within`].
pub const WRITE_ROUTES: &[&str] = &[
    "/api/create",
    "/api/update",
    "/api/move",
    "/api/archive",
    "/api/delete",
    "/api/reindex-file",
];

/// `request_timeout_secs` as a duration; `0` is no timeout.
pub fn request_timeout(config: &HttpConfig) -> Option<std::time::Duration> {
    match config.request_timeout_secs {
        0 => None,
        secs => Some(std::time::Duration::from_secs(secs)),
    }
}

/// The limits every route runs under, from inner to outer: the timeout on
/// the timed routes alone, then the body limit and the in-flight limit (one
/// semaphore shared by every route) on all of them, then the two fallbacks.
/// `build_router` also puts a body timeout on the write and untimed routes
/// before it hands them in. A function of its own so a test can hand it two
/// toy routers and a timeout of milliseconds.
pub fn with_limits(
    timed: Router<ApiState>,
    untimed: Router<ApiState>,
    timeout: Option<std::time::Duration>,
) -> Router<ApiState> {
    let timed = match timeout {
        Some(limit) => timed.layer(tower_http::timeout::TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            limit,
        )),
        None => timed,
    };
    timed
        .merge(untimed)
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(tower::limit::GlobalConcurrencyLimitLayer::new(
            MAX_IN_FLIGHT,
        ))
        .fallback(unknown_route)
        .method_not_allowed_fallback(wrong_method)
}

pub fn build_router(state: ApiState) -> Router {
    let cors = cors_layer(&state.http_config.cors_origins);
    let timeout = state.request_timeout;
    let mut timed = Router::new();
    let mut untimed = Router::new();
    let mut writes = Router::new();
    for (path, handler) in routes() {
        if WRITE_ROUTES.contains(&path) {
            writes = writes.route(path, handler);
        } else if UNTIMED_ROUTES.contains(&path) {
            untimed = untimed.route(path, handler);
        } else {
            timed = timed.route(path, handler);
        }
    }
    // A stalled body would hold an in-flight permit for good, and a write or
    // an untimed route is outside the timeout layer, so its body read is
    // timed on its own.
    let body_timed = |router: Router<ApiState>| match timeout {
        Some(limit) => router.layer(tower_http::timeout::RequestBodyTimeoutLayer::new(limit)),
        None => router,
    };
    with_limits(timed, body_timed(untimed.merge(writes)), timeout)
        .layer(cors)
        .with_state(state)
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
        .public_url
        .as_deref()
        .unwrap_or(&default_url);
    Json(crate::openapi::build_openapi_spec(server_url))
}

// ---------------------------------------------------------------------------
// Read endpoint handlers
// ---------------------------------------------------------------------------

async fn handle_match(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Match>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    // Checked before the scan runs: a malformed scope is the caller's own text (#60).
    body.scope()
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let report = state
        .core
        .with_reader(move |store| crate::matching::run(store, &body))
        .await?;
    Ok(Json(report))
}

async fn handle_search(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Search>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    // Checked before the pipeline runs, so a typo fails fast (#35).
    if body.full && body.summaries {
        return Err(ApiError::bad_request(
            "--full and --summaries are mutually exclusive",
        ));
    }
    let scope = search::parse_scope(&body).map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let config = state.core.config.clone();
    let env = state
        .core
        .with_core(move |g| {
            search::run_query(body, scope, &config, g.store, g.embedder, g.reranker)
        })
        .await?;
    let value = serde_json::to_value(&env).map_err(|e| ApiError::internal(&format!("{e:#}")))?;
    Ok(Json(value))
}

async fn handle_read(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiQuery(p): ApiQuery<crate::params::Read>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let vault = state.core.vault_path.clone();
    let profile = state.core.profile.clone();
    let result = state
        .core
        .with_reader(move |store| {
            let ctx = ContextParams {
                store,
                vault_path: &vault,
                profile: profile.as_ref().as_ref(),
            };
            context::context_read(&ctx, &p.file, p.section.as_deref(), p.include)
        })
        .await?;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_list(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiQuery(params): ApiQuery<crate::params::List>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let all_terms = crate::tags::merge_scope_alias(params.scope, params.all);
    let filter = crate::tags::Scope::parse(&all_terms, &params.any, &params.none)
        .and_then(|s| {
            s.with_filters(
                params.property.as_deref(),
                params.links_to.as_deref(),
                params.linked_from.as_deref(),
            )
        })
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let vault = state.core.vault_path.clone();
    let profile = state.core.profile.clone();
    let items = state
        .core
        .with_reader(move |store| {
            let ctx = ContextParams {
                store,
                vault_path: &vault,
                profile: profile.as_ref().as_ref(),
            };
            context::context_list(
                &ctx,
                &filter,
                params.created_by.as_deref(),
                params.limit,
                params.after.as_deref(),
                params.sort.into(),
                params.detailed,
            )
        })
        .await?;
    Ok(Json(serde_json::json!(items)))
}

/// The vault's tag vocabulary, whole or under one term — the call to make
/// before filtering with `/api/list` (#61).
async fn handle_tags(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiQuery(params): ApiQuery<crate::params::Tags>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let prefix = params.under.as_deref().and_then(crate::tags::parse_term);
    let rows = state
        .core
        .with_reader(move |store| store.tags_under(prefix.as_ref()))
        .await?;
    Ok(Json(serde_json::json!(rows)))
}

/// The vault's custom-property registry, or one property's values — the
/// call to make before filtering `/api/list` or `/api/search` with
/// `property` (#66).
async fn handle_properties(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiQuery(params): ApiQuery<crate::params::Properties>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let vault = state.core.vault_path.clone();
    let report = state
        .core
        .with_reader(move |store| crate::properties::run(store, &vault, &params))
        .await?;
    Ok(Json(serde_json::json!(report)))
}

async fn handle_vault_map(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let vault = state.core.vault_path.clone();
    let profile = state.core.profile.clone();
    let map = state
        .core
        .with_reader(move |store| {
            let ctx = ContextParams {
                store,
                vault_path: &vault,
                profile: profile.as_ref().as_ref(),
            };
            context::vault_map(&ctx)
        })
        .await?;
    Ok(Json(serde_json::json!(map)))
}

async fn handle_health(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let profile_ref = state.core.profile.as_ref().as_ref();
    let config = health::HealthConfig {
        daily_folder: profile_ref.and_then(|p| p.structure.folders.daily.clone()),
        inbox_folder: profile_ref.and_then(|p| p.structure.folders.inbox.clone()),
    };
    let report = state
        .core
        .with_reader(move |store| health::generate_health_report(store, &config))
        .await?;
    Ok(Json(serde_json::json!(report)))
}

async fn handle_validate(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Validate>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, false)?;
    let target = body
        .target()
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let limits = crate::validate::ChunkLimits {
        min_chars: state.core.config.chunk_min_chars,
        target_tokens: crate::chunker::limits::TARGET_TOKENS,
    };
    let vault = state.core.vault_path.clone();
    let strict = body.strict;
    let report = crate::core::blocking(move || {
        crate::validate::validate_target(&vault, &target, &limits, strict)
    })
    .await?;
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
    let data_dir = crate::config::Config::data_dir()?;
    let config = state.core.config.clone();
    let pending = state
        .core
        .pending_events
        .load(std::sync::atomic::Ordering::Relaxed);
    let report = state
        .core
        .with_reader(move |store| search::status_json(store, &data_dir, &config, pending))
        .await?;
    Ok(Json(report))
}

// ---------------------------------------------------------------------------
// Write endpoint handlers
// ---------------------------------------------------------------------------

/// Run a write against the core, timed while it waits for the locks and not
/// after. Past the deadline the write has not run, so the 408 is safe to
/// retry; once it holds the core it runs to the end and answers.
async fn core_within<R, F>(state: &ApiState, f: F) -> Result<R, ApiError>
where
    R: Send + 'static,
    F: for<'a> FnOnce(crate::core::CoreGuards<'a>) -> anyhow::Result<R> + Send + 'static,
{
    let locks = match state.request_timeout {
        Some(limit) => tokio::time::timeout(limit, state.core.lock_core())
            .await
            .map_err(|_| ApiError::timed_out())?,
        None => state.core.lock_core().await,
    };
    Ok(locks.run(f).await?)
}

async fn handle_create(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Create>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    state.core.writable()?;
    // No stdin exists on this surface, so an omitted content is an error
    // here instead of the CLI's fallback read.
    let content = body
        .content
        .ok_or_else(|| ApiError::bad_request("content is required"))?;
    let input = CreateNoteInput {
        content,
        filename: body.filename,
        tags: body.tags,
        folder: body.folder,
        created_by: "http-api".into(),
        auto_link: body.auto_link,
    };
    let vault = state.core.vault_path.clone();
    let profile = state.core.profile.clone();
    let settings = state.core.index_settings;
    let result = core_within(&state, move |g| {
        writer::create_note(
            input,
            g.store,
            g.embedder,
            settings.embed,
            settings.chunk,
            &vault,
            profile.as_ref().as_ref(),
        )
    })
    .await?;
    state
        .core
        .record_write(&state.core.vault_path.join(&result.path))
        .await;
    Ok(Json(serde_json::json!(result)))
}

/// One capability for every change to an existing note (#62). The whole
/// edit list is read before anything is written, so a request that names an
/// impossible target answers 400 and writes nothing.
async fn handle_update(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Update>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    state.core.writable()?;
    let edits = body
        .to_writer_edits()
        .map_err(|e| ApiError::bad_request(&format!("{e:#}")))?;
    let input = UpdateInput {
        file: body.file,
        edits,
    };
    let vault = state.core.vault_path.clone();
    let config = state.core.config.clone();
    // `update_note` stores the new content hash and writes no chunks, so the
    // re-index runs here, in the same core call (#62). A failure after the
    // write says so, and `record_write` is skipped, so the watcher's own
    // event on this file re-indexes it.
    let result = core_within(&state, move |g| {
        let result = writer::update_note(g.store, &vault, &input)?;
        crate::indexer::reindex_written_file(&result.path, g.store, g.embedder, &vault, &config)
            .with_context(|| {
                format!(
                    "the file was written; its index rows were not updated for {}",
                    result.path
                )
            })?;
        Ok(result)
    })
    .await?;
    state
        .core
        .record_write(&state.core.vault_path.join(&result.path))
        .await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_move(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Move>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    state.core.writable()?;
    let vault = state.core.vault_path.clone();
    let result = core_within(&state, move |g| {
        writer::move_note(&body.file, &body.new_folder, g.store, &vault)
    })
    .await?;
    state
        .core
        .record_write(&state.core.vault_path.join(&result.path))
        .await;
    Ok(Json(serde_json::json!(result)))
}

/// Archive a note, or restore one previously archived with `undo: true`.
/// Archiving and restoring are one operation and its reverse, so they are
/// one capability with a flag rather than two routes (#62).
async fn handle_archive(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Archive>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    state.core.writable()?;
    let vault = state.core.vault_path.clone();
    let profile = state.core.profile.clone();
    let settings = state.core.index_settings;
    let result = core_within(&state, move |g| {
        if body.undo {
            writer::unarchive_note(
                &body.file,
                g.store,
                g.embedder,
                settings.embed,
                settings.chunk,
                &vault,
                profile.as_ref().as_ref(),
            )
        } else {
            writer::archive_note(&body.file, g.store, &vault, profile.as_ref().as_ref())
        }
    })
    .await?;
    state
        .core
        .record_write(&state.core.vault_path.join(&result.path))
        .await;
    Ok(Json(serde_json::json!(result)))
}

async fn handle_delete(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Delete>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    state.core.writable()?;
    let mode = DeleteMode::from(body.mode);
    let vault = state.core.vault_path.clone();
    let file = body.file.clone();
    core_within(&state, move |g| {
        writer::delete_note(g.store, &vault, &file, mode)
    })
    .await?;
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
/// server was started on, and no caller-supplied path reaches it. It holds
/// the writer and the embedder for its duration; every read answers from the
/// reader throughout.
async fn handle_index(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Index>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    // A read-only server refuses it like any other write, the way MCP's
    // `index` does: `rebuild: true` discards the index before it builds one
    // again (#62).
    state.core.writable()?;
    // The startup config, with the call's one override. The index-time
    // settings come from the session, so nothing here can be a second source
    // of the store's chunking or vector space (#55, #72).
    let mut config = (*state.core.config).clone();
    if body.no_gitignore {
        config.respect_gitignore = false;
    }
    let vault = state.core.vault_path.clone();
    let profile = state.core.profile.clone();
    let settings = state.core.index_settings;
    let result = state
        .core
        .with_core(move |g| {
            crate::indexer::run_index_shared(
                &vault,
                &config,
                settings,
                g.store,
                g.embedder,
                body.rebuild,
                profile.as_ref().as_ref(),
            )
        })
        .await?;
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
    ApiJson(body): ApiJson<crate::params::ReindexFile>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    // It writes the store, so a read-only server refuses it like `index`.
    state.core.writable()?;
    let vault = state.core.vault_path.clone();
    let config = state.core.config.clone();
    let file = body.file.clone();
    // A path not on disk is a Fault::NotFound from the indexer, and the
    // classifier answers 404 for it (#60).
    let result = core_within(&state, move |g| {
        crate::indexer::reindex_written_file(&file, g.store, g.embedder, &vault, &config)
    })
    .await?;
    Ok(Json(serde_json::json!({
        "file": body.file,
        "chunks": result.total_chunks,
        "docid": result.docid,
    })))
}

// ---------------------------------------------------------------------------
// Init endpoint handler
// ---------------------------------------------------------------------------

async fn handle_init(
    State(state): State<ApiState>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<crate::params::Init>,
) -> Result<impl IntoResponse, ApiError> {
    authorize(&headers, &state, true)?;
    match body.mode {
        crate::params::InitMode::Detect => {
            let vault = state.core.vault_path.clone();
            let result =
                crate::core::blocking(move || crate::onboarding::run_detect_json(&vault)).await?;
            Ok(Json(result))
        }
        crate::params::InitMode::Apply => {
            // `apply` indexes the vault, which is the work `index` is guarded
            // against on a read-only server. The mode is read first, so
            // `detect` — which writes nothing — still runs (#62).
            state.core.writable()?;
            let data_dir = crate::config::Config::data_dir()?;
            let vault = state.core.vault_path.clone();
            let config = state.core.config.clone();
            let settings = state.core.index_settings;
            // The running server keeps the profile it started with, which
            // the reply says. Any index it builds takes the session's
            // index-time settings (#55, #72). `run_apply_json` opens its own
            // store, so the core call is taken for exclusion only.
            let result = state
                .core
                .with_core(move |g| {
                    let _ = g;
                    let mut result = crate::onboarding::run_apply_json(
                        &vault,
                        &config,
                        settings,
                        &data_dir,
                        &mut crate::indexer::NoProgress,
                    )?;
                    if let Some(object) = result.as_object_mut() {
                        object.insert("restart_required".into(), serde_json::json!(true));
                    }
                    Ok(result)
                })
                .await?;
            Ok(Json(result))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

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
                    key: "kn_readkey".into(),
                    name: "reader".into(),
                    permissions: "read".into(),
                },
                ApiKeyConfig {
                    key: "kn_writekey".into(),
                    name: "writer".into(),
                    permissions: "write".into(),
                },
            ],
            public_url: None,
            request_timeout_secs: 60,
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

    /// The two abjuration-school notes the search tests index.
    const ABJURATION_NOTES: &[(&str, &str)] = &[
        (
            "rules/abjuration-spells.md",
            "# Abjuration\n\n\
             ## Level 3 Counterspell\n\nA warding effect that stops a spell mid-cast. \
             It interrupts the casting itself and does nothing to a spell already in effect.\n\n\
             ## Level 5 Dispel Magic\n\nA warding effect that ends an ongoing spell. \
             It reaches an effect already in place and cannot interrupt one \
             that is still being cast, which is the whole of the difference.\n\n\
             ## Level 9 Dimensional Anchor\n\nA warding effect that pins a creature. \
             It closes every route out of the space the creature \
             currently stands in, and it does not care how that route was opened.\n",
        ),
        (
            "rules/evocation-spells.md",
            "# Evocation\n\n## Level 1 Firebolt\n\nA bolt of flame.\n",
        ),
    ];

    /// A write still waiting for the core when the deadline passes answers
    /// 408 with no body, and never runs: a retry is safe.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_write_waiting_for_the_core_past_the_deadline_is_a_408_and_does_not_run() {
        let (_tmp, mut state) = indexed_state();
        state.request_timeout = Some(Duration::from_millis(50));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let held = {
            let core = state.core.clone();
            tokio::spawn(async move {
                core.with_core(move |_guards| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("the core call started");

        let (status, body) = post_json(
            state.clone(),
            "/api/create",
            r##"{"filename":"late","content":"# Late\n"}"##,
        )
        .await;
        assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
        assert!(body.is_null(), "a 408 has no body, got {body}");

        release_tx.send(()).unwrap();
        held.await.unwrap().unwrap();
        // Give a write that wrongly ran after the 408 time to land.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!state.core.vault_path.join("late.md").exists());
    }

    /// A write that took the core before the deadline runs to the end and
    /// answers with its result, however long it took.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_write_that_started_before_the_deadline_answers_after_it() {
        let config = crate::core::testing::test_config();
        let (_tmp, vault, db) = crate::core::testing::indexed_vault(ABJURATION_NOTES, &config);
        let (embed, release, entered) = crate::core::testing::GatedEmbed::new(256);
        let mut state = api_state_from(crate::core::Core::for_test(
            &db,
            Box::new(embed),
            config,
            vault.clone(),
        ));
        state.request_timeout = Some(Duration::from_millis(50));

        let writing = {
            let state = state.clone();
            tokio::spawn(async move {
                post_json(
                    state,
                    "/api/create",
                    r##"{"filename":"slow","content":"# Slow\n\nBody.\n"}"##,
                )
                .await
            })
        };
        tokio::task::spawn_blocking(move || entered.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("the create reached the embedder");
        tokio::time::sleep(Duration::from_millis(200)).await;
        release.send(()).unwrap();

        let (status, body) = writing.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(vault.join("slow.md").is_file());
    }

    /// A write or index body that stalls past the deadline answers the
    /// bodyless 408 and never runs; a read route is not touched by the body timeout.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_write_body_that_stalls_is_a_408_and_does_not_run() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (_tmp, mut state) = indexed_state();
        state.request_timeout = Some(Duration::from_millis(50));
        let vault = state.core.vault_path.clone();
        let app = build_router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        // Headers promise 100 bytes; one chunk arrives and the rest never does.
        for path in ["/api/create", "/api/index"] {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            let head = format!(
                "POST {path} HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
                 content-length: 100\r\nconnection: close\r\n\r\n{{\"filename\":\"stall\","
            );
            stream.write_all(head.as_bytes()).await.unwrap();
            let mut raw = Vec::new();
            tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut raw))
                .await
                .expect("the stalled body is cut off")
                .unwrap();
            let text = String::from_utf8_lossy(&raw);
            assert!(text.starts_with("HTTP/1.1 408"), "{path}: got {text}");
            assert!(
                text.ends_with("\r\n\r\n"),
                "{path}: a 408 has no body, got {text}"
            );
            assert!(!vault.join("stall.md").exists());
        }

        // A read route on the same server is untouched.
        let mut read = tokio::net::TcpStream::connect(addr).await.unwrap();
        read.write_all(b"GET /api/health-check HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut raw = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), read.read_to_end(&mut raw))
            .await
            .expect("the read route answers")
            .unwrap();
        assert!(String::from_utf8_lossy(&raw).starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn every_write_route_is_a_real_route() {
        let served: Vec<&str> = routes().into_iter().map(|(path, _)| path).collect();
        for path in WRITE_ROUTES {
            assert!(served.contains(path), "{path} is not a route");
        }
    }

    #[test]
    fn no_route_is_both_untimed_and_a_write_route() {
        for path in WRITE_ROUTES {
            assert!(!UNTIMED_ROUTES.contains(path), "{path} is in both");
        }
    }

    fn api_state_from(core: Core) -> ApiState {
        let config = test_http_config();
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit));
        ApiState {
            core,
            request_timeout: request_timeout(&config),
            http_config: Arc::new(config),
            rate_limiter,
            no_auth: false,
        }
    }

    /// An empty store under a temp dir, for the tests that seed rows by hand
    /// or exercise auth. The dir has to outlive the state, so it is returned.
    fn test_api_state_at(vault_path: PathBuf) -> (tempfile::TempDir, ApiState) {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = tmp.path().join("knapper.db");
        let core = Core::for_test(
            &db,
            Box::new(DummyEmbedder),
            crate::core::testing::test_config(),
            vault_path,
        );
        (tmp, api_state_from(core))
    }

    fn test_api_state() -> (tempfile::TempDir, ApiState) {
        test_api_state_at(PathBuf::from("/tmp/test-vault"))
    }

    #[test]
    fn test_validate_api_key_valid() {
        let config = test_http_config();
        let result = validate_api_key("kn_readkey", &config);
        assert!(result.is_some());
        assert_eq!(result.unwrap().permissions, "read");
    }

    #[test]
    fn test_validate_api_key_invalid() {
        let config = test_http_config();
        assert!(validate_api_key("kn_badkey", &config).is_none());
    }

    #[test]
    fn test_generate_api_key_format() {
        let key = generate_api_key();
        assert!(key.starts_with("kn_"));
        assert_eq!(key.len(), 35); // "kn_" + 32 hex chars
    }

    #[test]
    fn test_check_permission_read_on_read() {
        assert!(check_permission("read", false));
    }

    /// The one mapping from a kind to a status, over every kind and the
    /// plain `anyhow` case, read through a context layer the way a handler
    /// receives it.
    #[test]
    fn each_fault_maps_to_its_status_and_kind() {
        use crate::fault::Fault;
        let cases: Vec<(anyhow::Error, StatusCode, &str)> = vec![
            (
                Fault::InvalidInput("x".into()).into(),
                StatusCode::BAD_REQUEST,
                "invalid_input",
            ),
            (
                Fault::NotFound("x".into()).into(),
                StatusCode::NOT_FOUND,
                "not_found",
            ),
            (
                Fault::Ambiguous("x".into()).into(),
                StatusCode::BAD_REQUEST,
                "ambiguous",
            ),
            (
                Fault::Conflict("x".into()).into(),
                StatusCode::CONFLICT,
                "conflict",
            ),
            (
                Fault::StaleIndex("x".into()).into(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "stale_index",
            ),
            (Fault::ReadOnly.into(), StatusCode::FORBIDDEN, "read_only"),
            (
                anyhow::anyhow!("x"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
            ),
        ];
        for (err, status, kind) in cases {
            let api = ApiError::from(err.context("under context"));
            assert_eq!(api.status, status, "{kind}");
            assert_eq!(api.kind, kind);
            assert!(
                api.message.starts_with("under context: "),
                "{}",
                api.message
            );
        }
    }

    /// Every error body names its kind beside its message, the parse-stage
    /// ones included.
    #[tokio::test]
    async fn an_error_body_names_its_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) = post_json(state, "/api/init", r#"{}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["kind"], "invalid_input");
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
        let (_tmp, state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer kn_badkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_vault_map_authorized() {
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_authorized() {
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_unauthorized() {
        let (_tmp, state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/index")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_search_unauthorized() {
        let (_tmp, state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::from(r#"{"query":"warding","all":["type/undead"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(response).await["kind"], "invalid_input");
    }

    #[tokio::test]
    async fn a_search_scope_naming_no_folder_is_a_bad_request() {
        // #65. The caller's own text named a folder no note lives under, so
        // this is a 400, the same as an unknown tag.
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer kn_readkey")
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
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::from(r#"{"query":"warding","all":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_authorized_empty() {
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/list")
                    .header("authorization", "Bearer kn_readkey")
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
        let writer = state.core.writer();
        let store = writer.lock().await;
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
                    .header("authorization", "Bearer kn_readkey")
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
        let (_tmp, state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
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
    async fn test_properties_lists_the_registry_and_one_names_values() {
        let (_tmp, state) = test_api_state();
        {
            let writer = state.core.writer();
            let store = writer.lock().await;
            let a = store
                .insert_file("ada.md", "h1", 100, "aaa111", None, None)
                .unwrap();
            store
                .replace_file_properties(
                    a,
                    &[crate::store::NewProperty {
                        chunk_seq: crate::store::DOC_LEVEL,
                        name: "status",
                        value: "draft",
                        kind: crate::properties::Kind::Text,
                        target_file: None,
                    }],
                )
                .unwrap();
        }
        let rows = json_body(get(state.clone(), "/api/properties").await).await;
        assert_eq!(rows[0]["name"], "status");
        assert_eq!(rows[0]["note_count"], 1);
        assert_eq!(rows[0]["kinds"][0], "text");
        let vals = json_body(get(state, "/api/properties?name=status").await).await;
        assert_eq!(vals[0]["value"], "draft");
        assert_eq!(vals[0]["kind"], "text");
    }

    #[tokio::test]
    async fn test_list_any_matches_either_term() {
        let (_tmp, state) = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?any=type/undead,status/draft").await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut listed = paths(&json_body(response).await);
        listed.sort();
        assert_eq!(listed, vec!["draft.md", "wight.md"]);
    }

    #[tokio::test]
    async fn test_list_none_excludes_its_terms() {
        let (_tmp, state) = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?all=type/&none=status/draft").await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut listed = paths(&json_body(response).await);
        listed.sort();
        assert_eq!(listed, vec!["wight.md", "wolf.md"]);
    }

    #[tokio::test]
    async fn test_list_starts_a_page_after_the_path_it_names() {
        let (_tmp, state) = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?limit=1&after=draft.md").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(paths(&json_body(response).await), vec!["wight.md"]);
    }

    /// A ranked page after a note the vault does not hold is the caller's
    /// cursor naming nothing, not a server fault (#143).
    #[tokio::test]
    async fn test_list_ranked_after_a_note_the_vault_does_not_hold_is_a_400() {
        let (_tmp, state) = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?sort=links_in&after=gone.md").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_list_merges_tags_into_all() {
        let (_tmp, state) = test_api_state();
        seed_tags(&state).await;
        let response = get(state, "/api/list?tags=type/beast&all=status/draft").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(paths(&json_body(response).await), vec!["draft.md"]);
    }

    #[tokio::test]
    async fn test_no_auth_mode_skips_check() {
        let (_tmp, mut state) = test_api_state();
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
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/create")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::from(r##"{"content":"# Test","filename":"test"}"##))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_write_endpoint_write_key_accepted() {
        let (_tmp, state) = test_api_state();
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/update")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer kn_writekey")
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
        let (_tmp, mut state) = test_api_state();
        state.rate_limiter = Arc::new(RateLimiter::new(1));
        let app = build_router(state);
        // First request passes (consumes the single token)
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/vault-map")
                    .header("authorization", "Bearer kn_readkey")
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
                    .header("authorization", "Bearer kn_readkey")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().get("retry-after").is_some());
    }

    // -----------------------------------------------------------------------
    // OpenAPI document (no auth required)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_openapi_no_auth_required() {
        let (_tmp, state) = test_api_state();
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
    async fn the_well_known_route_is_gone() {
        let (_tmp, state) = test_api_state();
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
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// `[http] public_url` is the document's server URL when set.
    #[tokio::test]
    async fn the_document_names_the_public_url_when_one_is_set() {
        let (_tmp, mut state) = test_api_state();
        let mut config = (*state.http_config).clone();
        config.public_url = Some("https://abc.trycloudflare.com".into());
        state.http_config = Arc::new(config);
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
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let spec: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(spec["servers"][0]["url"], "https://abc.trycloudflare.com");
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
                    .header("authorization", "Bearer kn_writekey")
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
    async fn init_detect_reaches_detection_and_writes_nothing() {
        // `detect` is the half of `init` a server can run without touching
        // the vault (#62): it reports what it found and leaves no file.
        let vault = tempfile::tempdir().unwrap();
        std::fs::write(vault.path().join("note.md"), "# Note\n").unwrap();
        let (_tmp, state) = test_api_state_at(vault.path().to_path_buf());
        let (status, body) = post_json(state, "/api/init", r#"{"mode":"detect"}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.get("structure").is_some(), "not a detection: {body}");
        let left: Vec<_> = std::fs::read_dir(vault.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 1, "detect wrote something: {left:?}");
    }

    #[tokio::test]
    async fn an_unarchive_of_a_file_outside_the_vault_is_a_bad_request() {
        let (tmp, state) = indexed_state();
        let outside = tmp.path().join("outside.md");
        std::fs::write(&outside, "---\narchived_from: x.md\n---\n# O\n").unwrap();
        let (status, body) = post_json(
            state,
            "/api/archive",
            r#"{"file":"../outside.md","undo":true}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(outside.is_file());
    }

    #[tokio::test]
    async fn a_reindex_file_outside_the_vault_is_a_bad_request() {
        let (tmp, state) = indexed_state();
        std::fs::write(tmp.path().join("secret.md"), "# Secret\n").unwrap();
        let (status, body) =
            post_json(state, "/api/reindex-file", r#"{"file":"../secret.md"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
    }

    #[tokio::test]
    async fn init_without_a_mode_is_a_bad_request() {
        // The mode is required on every surface; the extractor refuses the
        // body before the handler runs, with the kind every error carries.
        let (_tmp, state) = test_api_state();
        let (status, body) = post_json(state, "/api/init", r#"{}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(body["error"].as_str().unwrap().contains("mode"), "{body}");
    }

    #[tokio::test]
    async fn an_init_mode_naming_nothing_is_a_bad_request() {
        let (_tmp, state) = test_api_state();
        let (status, body) = post_json(state, "/api/init", r#"{"mode":"sideways"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(
            body["error"].as_str().unwrap().contains("sideways"),
            "{body}"
        );
    }

    /// A vault of two notes, indexed. The mock's vectors are hashes, so the
    /// keyword lane carries the meaning here — which is all a granularity
    /// assertion needs.
    fn indexed_state() -> (tempfile::TempDir, ApiState) {
        let (tmp, core) = crate::core::testing::indexed_core(
            ABJURATION_NOTES,
            crate::core::testing::test_config(),
        );
        (tmp, api_state_from(core))
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
        state.core.config_mut().group_by = crate::config::GroupBy::File;
        // This test asserts per-section output. That output is below
        // coalescing. Coalescing has its own tests (#39).
        state.core.config_mut().ranking.coalesce_adjacent = false;

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

    /// A section the note does not hold, or a note the vault does not hold,
    /// is the resource the call addresses being absent: a 404, with the kind
    /// in the body (#60, #62).
    #[tokio::test]
    async fn a_read_of_a_section_or_a_note_the_vault_does_not_hold_is_a_404() {
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
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(json_body(response).await["kind"], "not_found");

        let response = get(state, "/api/read?file=nowhere.md").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = json_body(response).await;
        assert_eq!(body["kind"], "not_found");
        assert_eq!(body["error"], "file not found: nowhere.md");
    }

    /// A write that names a note the vault does not hold is the same absent
    /// resource.
    #[tokio::test]
    async fn an_update_of_a_missing_note_is_a_404() {
        let (_tmp, state) = indexed_state();
        let (status, body) = post_json(
            state,
            "/api/update",
            r#"{"file":"nowhere.md","edits":[{"mode":"append","content":"x"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(body["kind"], "not_found");
    }

    /// A section the note does not hold is the same absent resource, and the
    /// message names the note.
    #[tokio::test]
    async fn an_update_naming_a_missing_section_is_a_404() {
        let (_tmp, state) = indexed_state();
        let (status, body) = post_json(
            state,
            "/api/update",
            r#"{"file":"rules/evocation-spells.md","edits":[{"section":"Nowhere","mode":"replace","content":"x"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(body["kind"], "not_found");
        assert_eq!(
            body["error"],
            "editing rules/evocation-spells.md: section 'Nowhere' not found"
        );
    }

    /// A scope the parser refuses is the caller's own text, checked before
    /// the scan runs.
    #[tokio::test]
    async fn a_match_with_a_scope_the_parser_refuses_is_a_400() {
        let (_tmp, state) = indexed_state();
        let (status, body) = post_json(state, "/api/match", r#"{"pattern":"x","all":[""]}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
    }

    /// `indexed_state`'s vault is `tmp/vault`, so `tmp` is outside it.
    #[tokio::test]
    async fn a_create_folder_that_climbs_out_is_a_bad_request() {
        let (tmp, state) = indexed_state();
        let (status, body) = post_json(
            state,
            "/api/create",
            r##"{"filename":"out","content":"# Out\n","folder":"../escape"}"##,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert_eq!(
            body["error"],
            "folder must stay inside the vault: ../escape"
        );
        assert!(!tmp.path().join("escape").exists());
    }

    #[tokio::test]
    async fn a_validate_path_that_climbs_out_is_a_bad_request() {
        let (tmp, state) = indexed_state();
        std::fs::write(tmp.path().join("secret.md"), "# Sentinel\n").unwrap();
        let (status, body) = post_json(state, "/api/validate", r#"{"path":"../secret"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert_eq!(body["error"], "path must stay inside the vault: ../secret");
        assert!(!body.to_string().contains("Sentinel"));
    }

    #[tokio::test]
    async fn a_move_folder_that_climbs_out_is_a_bad_request() {
        let (tmp, state) = indexed_state();
        let (status, body) = post_json(
            state,
            "/api/move",
            r#"{"file":"rules/evocation-spells.md","new_folder":"../escape"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(!tmp.path().join("escape").exists());
    }

    /// `reindex-file` of a path not on disk is a 404; a path that exists and
    /// cannot be read stays a 500, which no fixture can provoke without
    /// changing permissions, so only the first is asserted.
    #[tokio::test]
    async fn a_reindex_file_of_a_path_not_on_disk_is_a_404() {
        let (_tmp, state) = indexed_state();
        let (status, body) =
            post_json(state, "/api/reindex-file", r#"{"file":"nowhere.md"}"#).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(body["kind"], "not_found");
    }

    /// Set a file's mtime two minutes ahead of what the index recorded, which
    /// is what a note edited outside knapper looks like to `update_note`.
    fn touch_forward(path: &std::path::Path) {
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(120))
            .unwrap();
    }

    /// The note moved under the caller: the write is refused and the status
    /// says the caller's view is stale, not that the server failed.
    #[tokio::test]
    async fn an_update_of_a_note_changed_on_disk_is_a_409() {
        let (_tmp, state) = indexed_state();
        touch_forward(&state.core.vault_path.join("rules/evocation-spells.md"));
        let (status, body) = post_json(
            state,
            "/api/update",
            r#"{"file":"rules/evocation-spells.md","edits":[{"mode":"append","content":"\nMore.\n"}]}"#,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert_eq!(body["kind"], "conflict");
        assert!(
            body["error"].as_str().unwrap().contains("mtime conflict"),
            "{body}"
        );
    }

    /// A create over a path that exists is refused before anything is
    /// embedded, and it is a conflict with the vault, not a server fault.
    #[tokio::test]
    async fn a_create_over_an_existing_note_is_a_409() {
        let (_tmp, state) = indexed_state();
        let (status, body) = post_json(
            state,
            "/api/create",
            r##"{"content":"# Evocation\n\nAgain.\n","filename":"evocation-spells","folder":"rules"}"##,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert_eq!(body["kind"], "conflict");
    }

    /// `archive` and `archive {undo: true}` are one operation and its reverse
    /// (#62). The handler's own branch chooses `archive_note` against
    /// `unarchive_note`, and nothing else covers it — an inverted branch would
    /// move the file the opposite way with the whole suite green.
    #[tokio::test]
    async fn the_undo_flag_chooses_the_operation_it_names() {
        let (_tmp, state) = indexed_state();
        let vault = state.core.vault_path.as_ref().clone();
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
        let (_tmp, mut state) = test_api_state();
        state.core.read_only = true;
        let (status, _) = post_json(state, "/api/index", r#"{}"#).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    /// `init {mode: apply}` indexes the vault, which is the work `index` is
    /// guarded against. `detect` writes nothing and still runs (#62).
    #[tokio::test]
    async fn a_read_only_server_refuses_init_apply_and_runs_init_detect() {
        let vault = tempfile::tempdir().unwrap();
        std::fs::write(vault.path().join("note.md"), "# Note\n").unwrap();

        let (_tmp, mut state) = test_api_state_at(vault.path().to_path_buf());
        state.core.read_only = true;
        let (status, _) = post_json(state, "/api/init", r#"{"mode":"apply"}"#).await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        let (_tmp, mut state) = test_api_state_at(vault.path().to_path_buf());
        state.core.read_only = true;
        let (status, _) = post_json(state, "/api/init", r#"{"mode":"detect"}"#).await;
        assert_eq!(status, StatusCode::OK);
    }

    /// GET `path` as a writer, and return the status and the body.
    async fn get_json(state: ApiState, path: &str) -> (StatusCode, serde_json::Value) {
        let app = build_router(state);
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(path)
                    .header("authorization", "Bearer kn_writekey")
                    .body(Body::empty())
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

    /// The promise http-rest-api.md makes is that every error body carries
    /// `kind`. axum's own extractor rejections are errors too.
    #[tokio::test]
    async fn malformed_json_is_a_bad_request_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) = post_json(state, "/api/delete", r#"{"file": "#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(body["error"].is_string(), "{body}");
    }

    #[tokio::test]
    async fn an_unknown_enum_word_is_a_bad_request_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) =
            post_json(state, "/api/delete", r#"{"file":"a.md","mode":"sideways"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(
            body["error"].as_str().unwrap().contains("sideways"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_missing_field_is_a_bad_request_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) = post_json(state, "/api/delete", r#"{}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(body["error"].as_str().unwrap().contains("file"), "{body}");
    }

    #[tokio::test]
    async fn a_query_value_of_the_wrong_type_is_a_bad_request_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) = get_json(state, "/api/list?limit=many").await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["kind"], "invalid_input");
        assert!(body["error"].as_str().unwrap().contains("limit"), "{body}");
    }

    #[tokio::test]
    async fn an_unknown_route_is_a_404_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) = get_json(state, "/api/nowhere").await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert_eq!(body["kind"], "not_found");
    }

    #[tokio::test]
    async fn the_wrong_method_on_a_known_route_is_a_405_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let (status, body) = get_json(state, "/api/delete").await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "{body}");
        assert_eq!(body["kind"], "invalid_input");
    }

    /// A toy router through the same limits the real one gets. A route under
    /// the timeout answers 408 when it runs past it; one in `UNTIMED_ROUTES`
    /// does not.
    #[tokio::test]
    async fn a_request_past_the_timeout_is_a_408() {
        async fn slow() -> &'static str {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            "done"
        }
        let (_tmp, state) = test_api_state();
        let timed = Router::new().route("/api/slow", axum::routing::get(slow));
        let untimed = Router::new().route(UNTIMED_ROUTES[0], axum::routing::get(slow));
        let app = with_limits(timed, untimed, Some(std::time::Duration::from_millis(20)))
            .with_state(state);

        let request = |uri: &str| {
            axum::http::Request::builder()
                .uri(uri)
                .body(Body::empty())
                .unwrap()
        };
        let timed_out = app.clone().oneshot(request("/api/slow")).await.unwrap();
        assert_eq!(timed_out.status(), StatusCode::REQUEST_TIMEOUT);
        let completed = app.oneshot(request(UNTIMED_ROUTES[0])).await.unwrap();
        assert_eq!(completed.status(), StatusCode::OK);
    }

    /// One semaphore for the whole server, not one per route: more than
    /// `MAX_IN_FLIGHT` requests spread over two routes never run more than
    /// `MAX_IN_FLIGHT` at once.
    #[tokio::test]
    async fn the_in_flight_limit_is_server_wide() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let slow = {
            let in_flight = in_flight.clone();
            let peak = peak.clone();
            move || {
                let in_flight = in_flight.clone();
                let peak = peak.clone();
                async move {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    "done"
                }
            }
        };
        let (_tmp, state) = test_api_state();
        let timed = Router::new().route("/api/a", axum::routing::get(slow.clone()));
        let untimed = Router::new().route(UNTIMED_ROUTES[0], axum::routing::get(slow));
        let app = with_limits(timed, untimed, None).with_state(state);

        let requests = (0..(MAX_IN_FLIGHT * 3)).map(|i| {
            let app = app.clone();
            let path = if i % 2 == 0 {
                "/api/a"
            } else {
                UNTIMED_ROUTES[0]
            };
            tokio::spawn(async move {
                let response = app
                    .oneshot(
                        axum::http::Request::builder()
                            .uri(path)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
            })
        });
        // Spawn them all before awaiting any: a lazy iterator would run them
        // one at a time.
        let requests: Vec<_> = requests.collect();
        for handle in requests {
            handle.await.unwrap();
        }

        let peak = peak.load(Ordering::SeqCst);
        assert!(
            peak > 1,
            "the requests never overlapped; the test proves nothing"
        );
        assert!(
            peak <= MAX_IN_FLIGHT,
            "{peak} requests ran at once; the limit is {MAX_IN_FLIGHT}"
        );
    }

    #[test]
    fn every_untimed_route_is_a_real_route() {
        let served: Vec<&str> = routes().into_iter().map(|(path, _)| path).collect();
        for path in UNTIMED_ROUTES {
            assert!(served.contains(path), "{path} is not a route");
        }
    }

    #[tokio::test]
    async fn a_body_over_the_limit_is_a_bad_request_with_a_kind() {
        let (_tmp, state) = test_api_state();
        let content = "x".repeat(MAX_BODY_BYTES + 1024);
        let body = format!(r#"{{"filename":"big","content":"{content}"}}"#);
        let (status, reply) = post_json(state, "/api/create", &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(reply["kind"], "invalid_input");
    }

    /// Axum's own `Json` limit is 2 MiB; the layer raises it to
    /// `MAX_BODY_BYTES`, so a body between the two reaches the handler.
    #[tokio::test]
    async fn a_body_between_axums_default_and_the_limit_is_not_refused() {
        let vault = tempfile::tempdir().unwrap();
        let (_tmp, state) = test_api_state_at(vault.path().to_path_buf());
        let content = "x".repeat(3 * 1024 * 1024);
        let body = format!(r#"{{"filename":"big","content":"{content}"}}"#);
        let (status, reply) = post_json(state, "/api/create", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert!(vault.path().join("big.md").is_file());
    }

    /// The transport's two unauthenticated routes are timed like any read.
    /// The membership asserts pin the timing; the 200s show the routes answer
    /// through the real router with a timeout set.
    #[tokio::test]
    async fn the_openapi_document_and_the_health_check_answer_under_the_timeout() {
        for path in ["/openapi.json", "/api/health-check"] {
            assert!(!UNTIMED_ROUTES.contains(&path), "{path}");
            assert!(!WRITE_ROUTES.contains(&path), "{path}");
        }
        let (_tmp, mut state) = test_api_state();
        state.request_timeout = Some(Duration::from_secs(5));
        for path in ["/openapi.json", "/api/health-check"] {
            let response = get(state.clone(), path).await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn a_read_only_server_refuses_reindex_file() {
        let (_tmp, mut state) = indexed_state();
        state.core.read_only = true;
        let (status, body) = post_json(
            state,
            "/api/reindex-file",
            r#"{"file":"rules/evocation-spells.md"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        assert_eq!(body["kind"], "read_only");
    }

    /// Five notes that all answer one query. Five is more than the `top_n` the
    /// R21 test configures, so a truncation reads as a truncation and not as a
    /// corpus that had no more to give (#62). Each body is well over
    /// `chunk_min_chars`, so each note is one chunk of its own.
    fn state_over_five_answering_notes() -> (tempfile::TempDir, ApiState) {
        let notes: Vec<(String, String)> = ["counterspell", "dispel", "anchor", "ward", "seal"]
            .iter()
            .enumerate()
            .map(|(i, subject)| {
                (
                    format!("{i}-{subject}.md"),
                    format!(
                        "# The {subject} rule\n\nA warding effect. Every warding effect in this \
                         ruleset states what it stops, when it may be cast, and what it leaves \
                         alone, and the {subject} rule is one of them among several others.\n"
                    ),
                )
            })
            .collect();
        let borrowed: Vec<(&str, &str)> = notes
            .iter()
            .map(|(p, b)| (p.as_str(), b.as_str()))
            .collect();
        let (tmp, core) =
            crate::core::testing::indexed_core(&borrowed, crate::core::testing::test_config());
        (tmp, api_state_from(core))
    }

    /// #133 reaches HTTP: a query the floor emptied names what it rejected,
    /// with the floor beside it, and the status does not move.
    #[tokio::test]
    async fn a_floored_search_names_what_it_rejected() {
        let (_tmp, mut state) = state_over_five_answering_notes();
        // No cross-encoder here, so the sorted stage needs `[calibrated]
        // enabled` to run and the logistic's own floor is what applies.
        {
            let c = state.core.config_mut();
            c.calibrated.enabled = true;
            c.calibrated.floor = 1.01;
        }

        let (status, body) = post_json(state, "/api/search", r#"{"query":"warding"}"#).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "no_results", "got {body}");
        let rows = body["less_relevant"].as_array().expect("got {body}");
        assert!(!rows.is_empty(), "got {body}");
        assert!(
            rows.iter().all(|r| r.get("score").is_some()),
            "a rejected row carries its score without being asked, got {body}"
        );
        assert_eq!(body["answer_floor"], 101.0, "got {body}");
    }

    /// R21 (#62): the number of results a call that names no `top_n` gets is
    /// the configured one, and not a literal this server holds. A state built
    /// at three answers three, and the same state answers more when the call
    /// asks for more — which is what separates the configured default from a
    /// corpus that ran out.
    #[tokio::test]
    async fn a_search_that_names_no_top_n_gets_the_configured_number() {
        let (_tmp, mut state) = state_over_five_answering_notes();
        state.core.config_mut().top_n = 3;

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
        state
            .core
            .set_reranker(Box::new(crate::llm::MockLlm::new(256)));
        // The mock's Jaccard scores run well under the real cross-encoder's
        // range, and the default answer floor exists to gate a real model's
        // probability — not this fixture's stand-in. Zero it so the query
        // still answers (#34's floor is exercised in its own tests).
        state.core.config_mut().ranking.answer_floor = 0.0;

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

    #[tokio::test]
    async fn test_list_filters_by_property_and_an_unknown_note_is_a_400() {
        let (_tmp, state) = test_api_state();
        {
            let writer = state.core.writer();
            let store = writer.lock().await;
            let a = store
                .insert_file("ada.md", "h1", 100, "aaa111", None, None)
                .unwrap();
            let acme = store
                .insert_file("acme.md", "h2", 200, "bbb222", None, None)
                .unwrap();
            store
                .replace_file_properties(
                    a,
                    &[crate::store::NewProperty {
                        chunk_seq: crate::store::DOC_LEVEL,
                        name: "employer",
                        value: "acme",
                        kind: crate::properties::Kind::Link,
                        target_file: Some(acme),
                    }],
                )
                .unwrap();
        }
        let rows = json_body(get(state.clone(), "/api/list?property=employer%3Dacme").await).await;
        assert_eq!(paths(&rows), vec!["ada.md"]);
        let rows =
            json_body(get(state.clone(), "/api/list?property=employer&links_to=acme").await).await;
        assert_eq!(paths(&rows), vec!["ada.md"]);
        let response = get(state, "/api/list?links_to=nobody").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn an_alias_two_notes_carry_is_a_400_on_read_and_on_list() {
        // The caller's own name matched more than one note, which is the
        // caller's input to repair and not a server fault (#142).
        let (_tmp, state) = test_api_state();
        {
            let writer = state.core.writer();
            let store = writer.lock().await;
            for (path, docid) in [("a.md", "aaa111"), ("b.md", "bbb222")] {
                let id = store
                    .insert_file(path, "h", 100, docid, None, None)
                    .unwrap();
                store
                    .replace_file_aliases(id, &["Twin".to_string()])
                    .unwrap();
            }
        }
        let response = get(state.clone(), "/api/read?file=Twin").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(response).await["kind"], "ambiguous");
        let response = get(state, "/api/list?links_to=Twin").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// `index` holds the writer and the embedder for its duration; a read
    /// answers from the reader throughout (serve-core spec).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_index_tool_leaves_reads_answering() {
        use crate::core::testing::{GatedEmbed, indexed_vault, test_config};
        use std::time::Duration;

        let config = test_config();
        let (_tmp, vault, db) = indexed_vault(ABJURATION_NOTES, &config);
        // A change for `index` to find, so it reaches the embedder.
        std::fs::write(
            vault.join("rules/evocation-spells.md"),
            "# Evocation\n\n## Level 1 Firebolt\n\nA bolt of flame, rewritten.\n",
        )
        .unwrap();
        let (embed, release, entered) = GatedEmbed::new(256);
        let state = api_state_from(crate::core::Core::for_test(
            &db,
            Box::new(embed),
            config,
            vault,
        ));

        let indexing = {
            let state = state.clone();
            tokio::spawn(async move { post_json(state, "/api/index", "{}").await })
        };
        tokio::task::spawn_blocking(move || entered.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("index reached the embedder");

        let response = tokio::time::timeout(
            Duration::from_secs(2),
            get(state.clone(), "/api/read?file=rules/abjuration-spells.md"),
        )
        .await
        .expect("a read waited on index");
        assert_eq!(response.status(), StatusCode::OK);

        release.send(()).unwrap();
        let (status, _) = indexing.await.unwrap();
        assert_eq!(status, StatusCode::OK);
    }
}
