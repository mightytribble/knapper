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

// fields api_key/agent/max_retries are wired across Tasks 2-3
#[allow(dead_code)]
#[derive(Debug)]
pub struct ApiEmbedder<P: EmbedProvider> {
    provider: P,
    api_key: String,
    dim: usize,
    agent: ureq::Agent,
    max_retries: u32,
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

impl<P: EmbedProvider> EmbedModel for ApiEmbedder<P> {
    fn embed_batch(&mut self, _docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
        unimplemented!("Task 2")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::EmbedModel;
    use std::sync::Mutex;

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
}
