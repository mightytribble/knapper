//! Optional API-backed embedding backend (#84). One provider-agnostic
//! `ApiEmbedder<P>` wraps a small `EmbedProvider` trait; `Gemini` is the first
//! provider. The `EmbedModel` trait stays synchronous and the call blocks
//! through `ureq`, the way local inference already blocks.

use crate::llm::{EmbedDoc, EmbedModel};
use anyhow::{Context, Result};
use std::time::Duration;

pub trait EmbedProvider: Send {
    fn env_var(&self) -> &'static str;
    fn endpoint(&self) -> String;
    fn auth_header(&self, key: &str) -> (String, String);
    fn native_dim(&self) -> usize;
    fn max_input_tokens(&self) -> usize;
    fn batch_cap(&self) -> usize;
    fn identity(&self, dim: usize) -> String;
    fn build_document_request(&self, docs: &[EmbedDoc<'_>], dim: usize) -> serde_json::Value;
    fn build_query_request(&self, text: &str, dim: usize) -> serde_json::Value;
    fn parse_vectors(&self, body: &serde_json::Value) -> Result<Vec<Vec<f32>>>;
}

pub struct ApiEmbedder<P: EmbedProvider> {
    provider: P,
    api_key: String,
    dim: usize,
    agent: ureq::Agent,
    max_retries: u32,
}

// Manual, not derived: a derive would print `api_key` in cleartext on any
// `{:?}` (a `dbg!()`, an error context capturing `self`, a panic message, a
// log line) — the same secret-leak the env-only posture forbids, via a
// different vector. Writing it by hand also drops the `P: Debug` bound a
// derive would impose, so a provider (e.g. `Gemini`) need not derive `Debug`.
impl<P: EmbedProvider> std::fmt::Debug for ApiEmbedder<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiEmbedder")
            .field("provider", &std::any::type_name::<P>())
            .field("api_key", &"[redacted]")
            .field("dim", &self.dim)
            .field("agent", &self.agent)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl<P: EmbedProvider> ApiEmbedder<P> {
    pub fn new(
        provider: P,
        dim_override: Option<usize>,
        timeout_secs: u64,
        max_retries: u32,
    ) -> Result<Self> {
        let var = provider.env_var();
        let api_key =
            std::env::var(var).with_context(|| format!("environment variable {var} is not set"))?;
        let dim = dim_override.unwrap_or_else(|| provider.native_dim());
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs))
            .build();
        Ok(Self {
            provider,
            api_key,
            dim,
            agent,
            max_retries,
        })
    }
}

// Retryable: a 429, any 5xx, or a transport-level failure (connect/timeout/
// read). Any other 4xx is the caller's own bad request and retrying it would
// only repeat the same rejection.
fn retryable(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::Status(code, _) => *code == 429 || *code >= 500,
        ureq::Error::Transport(_) => true,
    }
}

impl<P: EmbedProvider> ApiEmbedder<P> {
    fn post(&self, body: &serde_json::Value) -> Result<serde_json::Value> {
        let (header, value) = self.provider.auth_header(&self.api_key);
        let mut attempt = 0u32;
        loop {
            let result = self
                .agent
                .post(&self.provider.endpoint())
                .set(&header, &value)
                .send_json(body);
            match result {
                Ok(resp) => {
                    return resp
                        .into_json::<serde_json::Value>()
                        .context("decoding embedding response");
                }
                Err(err) if retryable(&err) && attempt < self.max_retries => {
                    let backoff = std::time::Duration::from_millis(50u64 << attempt);
                    std::thread::sleep(backoff);
                    attempt += 1;
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err).context("embedding request failed"));
                }
            }
        }
    }
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl<P: EmbedProvider> EmbedModel for ApiEmbedder<P> {
    fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(docs.len());
        for sub in docs.chunks(self.provider.batch_cap().max(1)) {
            let body = self.provider.build_document_request(sub, self.dim);
            let json = self.post(&body)?;
            let mut vectors = self.provider.parse_vectors(&json)?;
            anyhow::ensure!(
                vectors.len() == sub.len(),
                "provider returned {} vectors for {} documents",
                vectors.len(),
                sub.len()
            );
            for v in &mut vectors {
                normalize(v);
            }
            out.append(&mut vectors);
        }
        Ok(out)
    }
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let body = self.provider.build_query_request(text, self.dim);
        let json = self.post(&body)?;
        let mut vectors = self.provider.parse_vectors(&json)?;
        anyhow::ensure!(
            vectors.len() == 1,
            "provider returned {} vectors for a single query",
            vectors.len()
        );
        let mut v = vectors
            .pop()
            .context("query embedding returned no vector")?;
        normalize(&mut v);
        Ok(v)
    }
    // `token_count` biases high: chars/3 (ceil) over-counts relative to the
    // chunker's chars/4, so a chunk is budgeted under the real input wall and
    // the API never silently truncates its tail (#75).
    fn token_count(&self, text: &str) -> usize {
        text.chars().count().div_ceil(3)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_context(&self) -> usize {
        self.provider.max_input_tokens()
    }
    fn fingerprint(&self) -> String {
        self.provider.identity(self.dim)
    }
}

const GEMINI_DOC_TASK: &str = "RETRIEVAL_DOCUMENT";
const GEMINI_QUERY_TASK: &str = "RETRIEVAL_QUERY";
const GEMINI_NATIVE_DIM: usize = 3072;
const GEMINI_MAX_INPUT_TOKENS: usize = 2048;
const GEMINI_BATCH_CAP: usize = 100;

pub struct Gemini {
    pub model_id: String,
    pub endpoint_override: Option<String>,
}

impl Gemini {
    fn one_request(
        &self,
        text: &str,
        task: &str,
        title: Option<&str>,
        dim: usize,
    ) -> serde_json::Value {
        let mut req = serde_json::json!({
            "model": format!("models/{}", self.model_id),
            "content": { "parts": [ { "text": text } ] },
            "taskType": task,
            "outputDimensionality": dim,
        });
        if let Some(t) = title.filter(|t| !t.is_empty()) {
            req["title"] = serde_json::Value::String(t.to_string());
        }
        req
    }
}

impl EmbedProvider for Gemini {
    fn env_var(&self) -> &'static str {
        "GEMINI_API_KEY"
    }
    fn endpoint(&self) -> String {
        self.endpoint_override.clone().unwrap_or_else(|| {
            format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents",
                self.model_id
            )
        })
    }
    fn auth_header(&self, key: &str) -> (String, String) {
        ("x-goog-api-key".to_string(), key.to_string())
    }
    fn native_dim(&self) -> usize {
        GEMINI_NATIVE_DIM
    }
    fn max_input_tokens(&self) -> usize {
        GEMINI_MAX_INPUT_TOKENS
    }
    fn batch_cap(&self) -> usize {
        GEMINI_BATCH_CAP
    }
    fn identity(&self, dim: usize) -> String {
        format!(
            "gemini/{}/dim={dim}/doc={GEMINI_DOC_TASK}/qry={GEMINI_QUERY_TASK}",
            self.model_id
        )
    }
    fn build_document_request(&self, docs: &[EmbedDoc<'_>], dim: usize) -> serde_json::Value {
        let requests: Vec<_> = docs
            .iter()
            .map(|d| self.one_request(d.text, GEMINI_DOC_TASK, Some(d.title), dim))
            .collect();
        serde_json::json!({ "requests": requests })
    }
    fn build_query_request(&self, text: &str, dim: usize) -> serde_json::Value {
        serde_json::json!({ "requests": [ self.one_request(text, GEMINI_QUERY_TASK, None, dim) ] })
    }
    fn parse_vectors(&self, body: &serde_json::Value) -> Result<Vec<Vec<f32>>> {
        let arr = body["embeddings"]
            .as_array()
            .context("response has no embeddings array")?;
        arr.iter()
            .map(|e| {
                e["values"]
                    .as_array()
                    .context("embedding has no values array")
                    .map(|vals| {
                        vals.iter()
                            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                            .collect()
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::EmbedModel;
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex};

    // Serializes tests that touch the process-global KNAPPER_TEST_KEY env var,
    // since `cargo test --lib` runs tests in parallel by default.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // A provider that needs no network for the non-network methods.
    #[derive(Debug)]
    struct TestProvider {
        endpoint: String,
    }
    impl EmbedProvider for TestProvider {
        fn env_var(&self) -> &'static str {
            "KNAPPER_TEST_KEY"
        }
        fn endpoint(&self) -> String {
            self.endpoint.clone()
        }
        fn auth_header(&self, key: &str) -> (String, String) {
            ("x-test-key".to_string(), key.to_string())
        }
        fn native_dim(&self) -> usize {
            8
        }
        fn max_input_tokens(&self) -> usize {
            2048
        }
        fn batch_cap(&self) -> usize {
            2
        }
        fn identity(&self, dim: usize) -> String {
            format!("test/v1/dim={dim}")
        }
        fn build_document_request(&self, _docs: &[EmbedDoc<'_>], _dim: usize) -> serde_json::Value {
            serde_json::json!({})
        }
        fn build_query_request(&self, _text: &str, _dim: usize) -> serde_json::Value {
            serde_json::json!({})
        }
        fn parse_vectors(&self, _body: &serde_json::Value) -> Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
    }

    fn provider() -> TestProvider {
        TestProvider {
            endpoint: "http://127.0.0.1:1/x".into(),
        }
    }

    #[test]
    fn missing_key_errors_naming_the_variable() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::remove_var("KNAPPER_TEST_KEY") };
        let err = ApiEmbedder::new(provider(), None, 30, 3).unwrap_err();
        assert!(err.to_string().contains("KNAPPER_TEST_KEY"));
    }

    #[test]
    fn dim_defaults_to_native_and_override_wins() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        let native = ApiEmbedder::new(provider(), None, 30, 3).unwrap();
        assert_eq!(EmbedModel::dim(&native), 8);
        let truncated = ApiEmbedder::new(provider(), Some(4), 30, 3).unwrap();
        assert_eq!(EmbedModel::dim(&truncated), 4);
    }

    #[test]
    fn token_count_over_counts_relative_to_chars_over_four() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        let e = ApiEmbedder::new(provider(), None, 30, 3).unwrap();
        let text = "a".repeat(120);
        // chars/4 is the local chunker estimate (30). The API estimate must be
        // at least that, so a chunk is budgeted no larger than the real wall.
        assert!(e.token_count(&text) >= 30);
    }

    #[test]
    fn max_context_and_fingerprint_delegate_to_provider() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        let e = ApiEmbedder::new(provider(), Some(4), 30, 3).unwrap();
        assert_eq!(e.max_context(), 2048);
        assert_eq!(e.fingerprint(), "test/v1/dim=4");
    }

    // A one-shot-per-connection HTTP/1.1 stub. `bodies` is the queue of JSON
    // response bodies; each connection pops one. `count` records requests seen.
    fn spawn_stub(bodies: Vec<String>) -> (String, Arc<Mutex<usize>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(Mutex::new(0usize));
        let seen = count.clone();
        let queue = Arc::new(Mutex::new(bodies));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                *seen.lock().unwrap() += 1;
                let body = queue.lock().unwrap().remove(0);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        (format!("http://{addr}/embed"), count)
    }

    // A provider whose parse reads {"vectors":[[...],...]} and whose endpoint is
    // injected, so it drives the generic half against the stub.
    struct StubProvider {
        endpoint: String,
        cap: usize,
    }
    impl EmbedProvider for StubProvider {
        fn env_var(&self) -> &'static str {
            "KNAPPER_TEST_KEY"
        }
        fn endpoint(&self) -> String {
            self.endpoint.clone()
        }
        fn auth_header(&self, key: &str) -> (String, String) {
            ("x-test-key".into(), key.into())
        }
        fn native_dim(&self) -> usize {
            3
        }
        fn max_input_tokens(&self) -> usize {
            2048
        }
        fn batch_cap(&self) -> usize {
            self.cap
        }
        fn identity(&self, dim: usize) -> String {
            format!("stub/dim={dim}")
        }
        fn build_document_request(&self, docs: &[EmbedDoc<'_>], _dim: usize) -> serde_json::Value {
            serde_json::json!({ "n": docs.len() })
        }
        fn build_query_request(&self, _t: &str, _d: usize) -> serde_json::Value {
            serde_json::json!({})
        }
        fn parse_vectors(&self, body: &serde_json::Value) -> Result<Vec<Vec<f32>>> {
            let arr = body["vectors"].as_array().context("no vectors")?;
            Ok(arr
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect()
                })
                .collect())
        }
    }

    #[test]
    fn embed_batch_splits_at_cap_and_normalizes() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        // cap 2, three docs -> two requests. Each request returns its own rows.
        let (endpoint, count) = spawn_stub(vec![
            r#"{"vectors":[[3.0,0.0,0.0],[0.0,4.0,0.0]]}"#.into(),
            r#"{"vectors":[[0.0,0.0,5.0]]}"#.into(),
        ]);
        let mut e = ApiEmbedder::new(StubProvider { endpoint, cap: 2 }, None, 30, 3).unwrap();
        let docs = vec![
            EmbedDoc::untitled("a"),
            EmbedDoc::untitled("b"),
            EmbedDoc::untitled("c"),
        ];
        let out = e.embed_batch(&docs).unwrap();
        assert_eq!(*count.lock().unwrap(), 2); // split into two requests
        assert_eq!(out.len(), 3); // order preserved across sub-batches
        for v in &out {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5); // L2-normalized
        }
        // Exact values (not just unit norm) prove position, not just magnitude,
        // survived the split: a reversed or scrambled sub-batch order would
        // still pass the unit-norm loop above but fail this.
        assert_eq!(
            out,
            vec![
                vec![1.0f32, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn embed_query_normalizes_and_sends_one_request() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        let (endpoint, count) = spawn_stub(vec![r#"{"vectors":[[0.0,0.0,5.0]]}"#.into()]);
        let mut e = ApiEmbedder::new(StubProvider { endpoint, cap: 2 }, None, 30, 3).unwrap();
        let out = e.embed_query("q").unwrap();
        assert_eq!(*count.lock().unwrap(), 1); // exactly one request
        assert_eq!(out, vec![0.0f32, 0.0, 1.0]); // L2-normalized
    }

    // Extend the stub so it can return non-200 statuses first.
    fn spawn_status_stub(statuses: Vec<(u16, String)>) -> (String, Arc<Mutex<usize>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(Mutex::new(0usize));
        let seen = count.clone();
        let queue = Arc::new(Mutex::new(statuses));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                *seen.lock().unwrap() += 1;
                let (code, body) = queue.lock().unwrap().remove(0);
                let reason = if code == 200 { "OK" } else { "ERR" };
                let resp = format!(
                    "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = std::io::Write::write_all(&mut stream, resp.as_bytes());
            }
        });
        (format!("http://{addr}/embed"), count)
    }

    #[test]
    fn retries_on_429_then_succeeds() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        let (endpoint, count) = spawn_status_stub(vec![
            (429, "{}".into()),
            (200, r#"{"vectors":[[1.0,0.0,0.0]]}"#.into()),
        ]);
        let mut e = ApiEmbedder::new(StubProvider { endpoint, cap: 8 }, None, 30, 3).unwrap();
        let out = e.embed_batch(&[EmbedDoc::untitled("a")]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(*count.lock().unwrap(), 2); // one retry
    }

    #[test]
    fn gives_up_after_max_retries() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("KNAPPER_TEST_KEY", "k") };
        let (endpoint, count) = spawn_status_stub(vec![
            (500, "{}".into()),
            (500, "{}".into()),
            (500, "{}".into()),
            (500, "{}".into()),
        ]);
        let mut e = ApiEmbedder::new(StubProvider { endpoint, cap: 8 }, None, 30, 2).unwrap();
        let err = e.embed_batch(&[EmbedDoc::untitled("a")]).unwrap_err();
        assert!(err.to_string().contains("embedding request failed"));
        assert_eq!(*count.lock().unwrap(), 3); // initial try + 2 retries
    }

    #[test]
    fn gemini_document_request_carries_task_and_title() {
        let g = Gemini {
            model_id: "gemini-embedding-2".into(),
            endpoint_override: None,
        };
        let docs = vec![
            EmbedDoc::new("Dragon", "A large reptile."),
            EmbedDoc::untitled("plain"),
        ];
        let body = g.build_document_request(&docs, 1536);
        let reqs = body["requests"].as_array().unwrap();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0]["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(reqs[0]["title"], "Dragon");
        assert_eq!(reqs[0]["content"]["parts"][0]["text"], "A large reptile.");
        assert_eq!(reqs[0]["outputDimensionality"], 1536);
        // An untitled doc omits the title field rather than sending "".
        assert!(reqs[1].get("title").is_none());
    }

    #[test]
    fn gemini_query_request_uses_query_task_and_no_title() {
        let g = Gemini {
            model_id: "gemini-embedding-2".into(),
            endpoint_override: None,
        };
        let body = g.build_query_request("where do dragons live", 1536);
        assert_eq!(body["requests"][0]["taskType"], "RETRIEVAL_QUERY");
        assert!(body["requests"][0].get("title").is_none());
    }

    #[test]
    fn gemini_parses_values_in_order() {
        let g = Gemini {
            model_id: "gemini-embedding-2".into(),
            endpoint_override: None,
        };
        let body: serde_json::Value =
            serde_json::from_str(r#"{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}"#)
                .unwrap();
        let vectors = g.parse_vectors(&body).unwrap();
        assert_eq!(vectors, vec![vec![0.1f32, 0.2], vec![0.3, 0.4]]);
    }

    #[test]
    fn gemini_identity_pins_model_and_dim() {
        let g = Gemini {
            model_id: "gemini-embedding-2".into(),
            endpoint_override: None,
        };
        assert_eq!(
            g.identity(1536),
            "gemini/gemini-embedding-2/dim=1536/doc=RETRIEVAL_DOCUMENT/qry=RETRIEVAL_QUERY"
        );
    }
}
