use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context as _, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
/// Mutex used only during the first initialization of `BACKEND`.
static BACKEND_INIT: Mutex<()> = Mutex::new(());

/// Get or initialize the global llama.cpp backend.
/// Safe to call from multiple places — the backend is initialized at most once.
pub fn llama_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _guard = BACKEND_INIT.lock().unwrap();
    // Double-checked: another thread may have initialized while we waited.
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let mut backend =
        LlamaBackend::init().map_err(|e| anyhow::anyhow!("initializing llama backend: {e}"))?;
    // Suppress llama.cpp's noisy Metal/model loading logs to stderr.
    backend.void_logs();
    Ok(BACKEND.get_or_init(|| backend))
}

// ── Prompt format ────────────────────────────────────────────────────────────

/// Model-family-specific prompt templates for embedding models.
#[derive(Debug, Clone)]
pub enum PromptFormat {
    /// Google embeddinggemma family: uses `<bos>search_query:` / `<bos>search_document:` prefixes.
    EmbeddingGemma,
    /// Qwen embedding family: uses `Instruct:` / `Query:` format.
    QwenEmbedding,
    /// No special formatting — pass text as-is.
    Raw,
}

impl PromptFormat {
    /// Auto-detect prompt format from a GGUF filename.
    pub fn detect(filename: &str) -> Self {
        let lower = filename.to_lowercase();
        if lower.contains("embeddinggemma") {
            Self::EmbeddingGemma
        } else if lower.contains("qwen") && lower.contains("embed") {
            Self::QwenEmbedding
        } else {
            Self::Raw
        }
    }

    /// Format text for a search query.
    pub fn format_query(&self, query: &str) -> String {
        match self {
            Self::EmbeddingGemma => format!("<bos>search_query: {query}"),
            Self::QwenEmbedding => {
                format!("Instruct: Retrieve relevant passages\nQuery: {query}")
            }
            Self::Raw => query.to_string(),
        }
    }

    /// Format text for a document to be indexed.
    pub fn format_document(&self, title: &str, text: &str) -> String {
        match self {
            Self::EmbeddingGemma => format!("<bos>search_document: {title} {text}"),
            Self::QwenEmbedding | Self::Raw => format!("{title}\n{text}"),
        }
    }
}

// ── Types ────────────────────────────────────────────────────────────────────

/// Classified intent of an incoming search query.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QueryIntent {
    /// User wants a precise fact or term match.
    Exact,
    /// User wants related ideas and concepts.
    Conceptual,
    /// User wants to explore connections between entities.
    Relationship,
    /// User is browsing without a clear target.
    Exploratory,
    /// User is asking about a specific time period.
    Temporal,
}

/// Output produced by an orchestrator model for a query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrchestrationResult {
    /// Classified query intent.
    pub intent: QueryIntent,
    /// Query string(s) to actually run (original + any expansions).
    pub expansions: Vec<String>,
    /// Optional unix-timestamp range for temporal queries (start, end).
    #[serde(default)]
    pub date_range: Option<(i64, i64)>,
}

impl OrchestrationResult {
    /// Put the user's own words back at the head of the expansion list if the
    /// orchestrator left them out.
    ///
    /// The system prompt asks for the original query first. Qwen3-0.6B ignores
    /// that routinely — measured on the eval vault, three of the three probes
    /// it answered came back without it, including the bare name `Archdragon`,
    /// which was replaced by `Archdragon character` and `Archdragon concept`.
    /// So the exact-name probe ran with no exact name in it, which is most of
    /// what `eval/probes.md` has been recording as "intelligence breaks probe 4"
    /// since the file was started (#23).
    ///
    /// Applied on read rather than before caching, so `llm_cache` keeps what the
    /// model actually said and rows written before this existed are repaired
    /// without invalidation.
    pub fn ensure_original(&mut self, query: &str) {
        let original = query.trim();
        if original.is_empty() {
            return;
        }
        if self
            .expansions
            .iter()
            .any(|e| e.trim().eq_ignore_ascii_case(original))
        {
            return;
        }
        self.expansions.insert(0, original.to_string());
    }
}

/// Per-lane weights for the RRF fusion step.
#[derive(Debug, Clone)]
pub struct LaneWeights {
    pub semantic: f64,
    pub fts: f64,
    pub graph: f64,
    pub rerank: f64,
    pub temporal: f64,
}

impl LaneWeights {
    /// Map a classified intent to recommended lane weights.
    pub fn from_intent(intent: &QueryIntent) -> Self {
        match intent {
            QueryIntent::Exact => Self {
                fts: 1.5,
                semantic: 0.6,
                graph: 0.6,
                rerank: 0.8,
                temporal: 0.0,
            },
            QueryIntent::Conceptual => Self {
                semantic: 1.2,
                fts: 0.8,
                graph: 1.0,
                rerank: 1.2,
                temporal: 0.0,
            },
            // Graph sits at 0.8 here, not above the content lanes, and this is
            // load-bearing — see issue #9. `graph_expand` skips any neighbour
            // already in the seed set, so the graph lane's results are disjoint
            // from the other lanes' by construction. RRF scores agreement
            // between rankings of the same corpus; a disjoint set can never
            // accumulate any, so a graph result's fused score is a pure function
            // of this constant. At 1.5 its 20 capped expansions swept the whole
            // top 20, since 1.5/(60+20) beats 0.8/(60+1) by 43% — every content
            // result was locked out of every "who" query.
            QueryIntent::Relationship => Self {
                graph: 0.8,
                semantic: 1.0,
                fts: 1.0,
                rerank: 1.0,
                temporal: 0.0,
            },
            QueryIntent::Exploratory => Self {
                semantic: 1.0,
                fts: 1.0,
                graph: 0.8,
                rerank: 1.0,
                temporal: 0.0,
            },
            QueryIntent::Temporal => Self {
                semantic: 0.6,
                fts: 0.8,
                graph: 0.5,
                rerank: 0.8,
                temporal: 1.5,
            },
        }
    }

    /// Weights used when no intelligence layer is available (legacy mode).
    ///
    /// **Nothing calls this outside tests.** Turning intelligence off does not
    /// reach it: `search_with_intelligence` still runs, just with
    /// `orchestrator: None`, so the intent comes from `heuristic_orchestrate`
    /// and the weights from [`Self::from_intent`] exactly as they do with the
    /// models loaded. That is why issue #9's gate fired in *both*
    /// configurations, and why the fix had to be in the table above.
    pub fn default_no_intelligence() -> Self {
        Self {
            semantic: 1.0,
            fts: 1.0,
            graph: 0.8,
            rerank: 0.0,
            temporal: 0.0,
        }
    }
}

// ── Traits ───────────────────────────────────────────────────────────────────

/// Embedding backend — converts text into dense float vectors.
pub trait EmbedModel: Send {
    /// Embed a batch of texts in one call.
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Convenience wrapper for a single text.
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text])?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embed_batch returned empty results"))
    }

    /// Approximate token count for `text` (used for chunk-size budgeting).
    fn token_count(&self, text: &str) -> usize;

    /// Dimensionality of vectors produced by this model.
    fn dim(&self) -> usize;
}

// Blanket impl: `Box<dyn EmbedModel + Send>` itself implements `EmbedModel`.
// This lets `Arc<Mutex<Box<dyn EmbedModel + Send>>>` callers pass
// `&mut *guard` (which is `&mut Box<dyn EmbedModel + Send>`) to any
// function taking `&mut impl EmbedModel`.
impl EmbedModel for Box<dyn EmbedModel + Send> {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_one(text)
    }

    fn token_count(&self, text: &str) -> usize {
        (**self).token_count(text)
    }

    fn dim(&self) -> usize {
        (**self).dim()
    }
}

/// Cross-encoder reranker — scores a (query, document) pair.
pub trait RerankModel: Send {
    /// Return a relevance score in [0.0, 1.0].
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32>;

    /// Score `query` against every document, returning one score per document
    /// in order.
    ///
    /// This is the shape search actually calls in — thirty candidates against
    /// one query — and it exists so that an implementation holding a model can
    /// set up once for the whole set instead of once per pair (issue #13). The
    /// default is the naive loop, which is right for any implementation with no
    /// setup to amortize.
    fn rerank_batch(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        documents
            .iter()
            .map(|document| self.rerank_score(query, document))
            .collect()
    }
}

/// Orchestrator — interprets a query and produces an enriched search plan.
pub trait OrchestratorModel: Send {
    fn orchestrate(&mut self, query: &str) -> Result<OrchestrationResult>;
}

// ── MockLlm ──────────────────────────────────────────────────────────────────

/// Deterministic in-process implementation of all three traits.
/// Suitable for unit tests and CI runs — no model files required.
pub struct MockLlm {
    dim: usize,
}

impl MockLlm {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Produce a deterministic L2-normalised vector from `text` via SHA-256.
    pub fn hash_to_vector(&self, text: &str) -> Vec<f32> {
        let mut raw: Vec<f32> = Vec::with_capacity(self.dim);
        // Seed the first hash from the text itself, then chain hashes to fill
        // vectors wider than 32 bytes (8 f32s per 256-bit hash).
        let mut seed = text.to_owned();
        while raw.len() < self.dim {
            let mut hasher = Sha256::new();
            hasher.update(seed.as_bytes());
            let hash = hasher.finalize();
            // Each hash gives 32 bytes → 8 f32 values.
            for chunk in hash.chunks(4) {
                if raw.len() >= self.dim {
                    break;
                }
                let bytes: [u8; 4] = chunk.try_into().expect("chunk is always 4 bytes");
                // Map u32 → [-1.0, 1.0] for a reasonable spread before normalisation.
                let u = u32::from_le_bytes(bytes);
                let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
                raw.push(f);
            }
            // Next round: hash the previous hash digest (as hex) so values differ.
            seed = format!("{:x}", {
                let mut h2 = Sha256::new();
                h2.update(hash);
                h2.finalize()
            });
        }

        // L2-normalise so the mock behaves like a real embedding model.
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            raw.iter_mut().for_each(|x| *x /= norm);
        }
        raw
    }
}

impl EmbedModel for MockLlm {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vector(t)).collect())
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_vector(text))
    }

    fn token_count(&self, text: &str) -> usize {
        text.len() / 4 + 1
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

impl RerankModel for MockLlm {
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32> {
        // Deterministic score: Jaccard overlap of character 4-grams, clamped to [0,1].
        let ngrams = |s: &str| -> std::collections::HashSet<String> {
            s.chars()
                .collect::<Vec<_>>()
                .windows(4)
                .map(|w| w.iter().collect())
                .collect()
        };

        let q_set = ngrams(&query.to_lowercase());
        let d_set = ngrams(&document.to_lowercase());

        if q_set.is_empty() && d_set.is_empty() {
            return Ok(0.5);
        }

        let intersection = q_set.intersection(&d_set).count();
        let union = q_set.union(&d_set).count();

        let score = intersection as f32 / union as f32;
        Ok(score.clamp(0.0, 1.0))
    }
}

impl OrchestratorModel for MockLlm {
    fn orchestrate(&mut self, query: &str) -> Result<OrchestrationResult> {
        Ok(OrchestrationResult {
            intent: QueryIntent::Exploratory,
            expansions: vec![query.to_owned()],
            date_range: None,
        })
    }
}

// ── HuggingFace model download infrastructure ─────────────────────────────────

/// Parsed HuggingFace model URI: "hf:org/repo/filename.gguf"
#[derive(Debug, Clone)]
pub struct HfModelUri {
    pub repo: String,
    pub filename: String,
}

impl HfModelUri {
    pub fn parse(uri: &str) -> Result<Self> {
        let rest = uri
            .strip_prefix("hf:")
            .ok_or_else(|| anyhow::anyhow!("model URI must start with 'hf:', got: {uri}"))?;
        let last_slash = rest.rfind('/').ok_or_else(|| {
            anyhow::anyhow!("model URI must be 'hf:org/repo/file.gguf', got: {uri}")
        })?;
        let repo = &rest[..last_slash];
        let filename = &rest[last_slash + 1..];
        if repo.is_empty() || filename.is_empty() || !repo.contains('/') {
            bail!("invalid model URI format: {uri}");
        }
        Ok(Self {
            repo: repo.to_string(),
            filename: filename.to_string(),
        })
    }

    pub fn download_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.repo, self.filename
        )
    }

    /// Local cache path: models_dir/repo--filename (slashes replaced with --)
    pub fn cache_path(&self, models_dir: &Path) -> PathBuf {
        let safe_name = format!("{}--{}", self.repo.replace('/', "--"), self.filename);
        models_dir.join(safe_name)
    }
}

/// Download a file with progress bar and optional SHA256 verification. Retries once on failure.
pub fn download_model(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
    fn try_download(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
        tracing::info!("downloading {} -> {}", url, dest.display());

        let resp = ureq::get(url)
            .call()
            .map_err(|e| anyhow::anyhow!("HTTP GET {url}: {e}"))?;

        let total_size: u64 = resp
            .header("Content-Length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
        );
        pb.set_message(format!(
            "downloading {}",
            dest.file_name().and_then(|n| n.to_str()).unwrap_or("model")
        ));

        // Write to a temp file alongside dest, then rename for crash safety.
        let tmp_path = dest.with_extension("tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)
                .map_err(|e| anyhow::anyhow!("creating {}: {e}", tmp_path.display()))?;
            let mut reader = resp.into_reader();
            let mut buffer = [0u8; 8192];
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                std::io::Write::write_all(&mut file, &buffer[..n])?;
                pb.inc(n as u64);
            }
        }
        pb.finish_with_message("done");

        // Verify hash if provided.
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(&tmp_path)?;
            if actual != expected {
                let _ = std::fs::remove_file(&tmp_path);
                bail!(
                    "SHA-256 mismatch for {}: expected {expected}, got {actual}",
                    dest.display()
                );
            }
        }

        std::fs::rename(&tmp_path, dest).map_err(|e| anyhow::anyhow!("renaming temp file: {e}"))?;

        Ok(())
    }

    // Try once, retry on failure.
    match try_download(url, dest, expected_sha256) {
        Ok(()) => Ok(()),
        Err(first_err) => {
            tracing::warn!("download failed, retrying: {first_err:#}");
            let _ = std::fs::remove_file(dest);
            try_download(url, dest, expected_sha256)
        }
    }
}

/// Compute SHA-256 hex digest of a file.
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Ensure a model is present locally, downloading if not cached.
pub fn ensure_model(uri: &HfModelUri, models_dir: &Path) -> Result<PathBuf> {
    let path = uri.cache_path(models_dir);
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        download_model(&uri.download_url(), &path, None)?;
    }
    Ok(path)
}

/// Tokenizer that can be backed by either HuggingFace tokenizers crate or shimmytok (GGUF-embedded).
pub enum FlexTokenizer {
    HuggingFace(Box<tokenizers::Tokenizer>),
    Gguf(Box<shimmytok::Tokenizer>),
}

impl FlexTokenizer {
    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(t) => {
                let enc = t
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
                Ok(enc.get_ids().to_vec())
            }
            Self::Gguf(t) => {
                let ids = t
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
                Ok(ids)
            }
        }
    }

    /// Count tokens in text.
    pub fn token_count(&self, text: &str) -> usize {
        self.encode(text, false).map(|ids| ids.len()).unwrap_or(0)
    }

    /// Look up a token's ID by string (only available with HuggingFace backend).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::HuggingFace(t) => t.token_to_id(token),
            Self::Gguf(_) => None,
        }
    }

    /// Decode token IDs back to text (only available with HuggingFace backend).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t
                .decode(ids, skip_special)
                .map_err(|e| anyhow::anyhow!("decode: {e}")),
            Self::Gguf(_) => bail!("decode not supported with GGUF tokenizer"),
        }
    }
}

/// Load tokenizer for a model. Tries external tokenizer.json first, falls back to GGUF-embedded.
fn load_tokenizer_for_model(uri: &HfModelUri, models_dir: &Path) -> Result<FlexTokenizer> {
    // First try: external tokenizer.json from candidate repos.
    if let Some(tok) = try_external_tokenizer(uri, models_dir) {
        return Ok(FlexTokenizer::HuggingFace(Box::new(tok)));
    }

    // Fallback: load tokenizer from GGUF file metadata.
    let model_path = uri.cache_path(models_dir);
    if model_path.exists() {
        tracing::info!(
            "no external tokenizer found, loading from GGUF: {}",
            model_path.display()
        );
        let tok = shimmytok::Tokenizer::from_gguf_file(&model_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer from GGUF metadata: {e}"))?;
        return Ok(FlexTokenizer::Gguf(Box::new(tok)));
    }

    bail!(
        "could not find tokenizer for model '{}': no external tokenizer.json \
         and GGUF file not yet downloaded",
        uri.repo
    )
}

/// Try downloading tokenizer.json from candidate HuggingFace repos.
fn try_external_tokenizer(uri: &HfModelUri, models_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let mut candidates: Vec<String> = vec![uri.repo.clone()];

    // Non-GGUF variant: "org/model-GGUF" → "org/model"
    let base_repo = uri.repo.trim_end_matches("-GGUF").to_string();
    if base_repo != uri.repo {
        candidates.push(base_repo);
    }

    // Known upstream repos for default models (GGUF repos rarely ship tokenizers).
    let model_lower = uri.repo.to_lowercase();
    if model_lower.contains("all-minilm") {
        candidates.push("sentence-transformers/all-MiniLM-L6-v2".to_string());
    } else if model_lower.contains("embeddinggemma") {
        candidates.push("google/embeddinggemma-300m".to_string());
        candidates.push("google/gemma-2b".to_string());
    } else if model_lower.contains("qwen3") {
        let base_name = uri
            .repo
            .rsplit('/')
            .next()
            .unwrap_or("")
            .trim_end_matches("-GGUF")
            .trim_end_matches("-Q8_0-GGUF");
        if !base_name.is_empty() {
            candidates.push(format!("Qwen/{base_name}"));
        }
    }

    for repo in &candidates {
        let tok_uri = HfModelUri {
            repo: repo.clone(),
            filename: "tokenizer.json".to_string(),
        };
        let tok_path = tok_uri.cache_path(models_dir);

        if tok_path.exists()
            && let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path)
        {
            return Some(tok);
        }

        if let Ok(p) = ensure_model(&tok_uri, models_dir)
            && let Ok(tok) = tokenizers::Tokenizer::from_file(&p)
        {
            return Some(tok);
        }
    }

    None
}

/// Default model URIs for the intelligence layer.
///
/// Deliberately carries no embedding dimensionality: the dimension is the
/// model's, read from the GGUF at load time by [`LlamaEmbed::new`], so that a
/// `models.embed` override changes the dimension along with the model
/// (issue #12).
pub struct ModelDefaults {
    pub embed_uri: String,
    pub rerank_uri: String,
    pub expand_uri: String,
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            embed_uri: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".into(),
            rerank_uri: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
                .into(),
            expand_uri: "hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf".into(),
        }
    }
}

/// llama.cpp's own thread default, used only when the machine will say nothing
/// at all about how many cores it has.
const FALLBACK_N_THREADS: i32 = 4;

/// Physical cores, or `None` if the platform will not say.
///
/// Counts distinct `thread_siblings` masks in sysfs — one entry per physical
/// core, with an SMT pair sharing a mask. This is llama.cpp's own rule, from
/// `common/common.cpp::cpu_get_num_physical_cores`, which is what its CLI feeds
/// to `n_threads`; only the *library* default is the flat 4 that engraph used to
/// inherit.
///
/// Linux only, deliberately. Everywhere else the caller falls back to
/// `available_parallelism()`, which is right on the platforms that matter here:
/// Apple Silicon has no SMT, so its logical count already is a core count.
fn physical_cores() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let mut siblings = std::collections::HashSet::new();
        // Bounded: a missing entry ends the enumeration, and the cap keeps a
        // strange sysfs from spinning.
        for cpu in 0..4096 {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings");
            match std::fs::read_to_string(&path) {
                Ok(mask) => {
                    siblings.insert(mask.trim().to_string());
                }
                Err(_) => break,
            }
        }
        if !siblings.is_empty() {
            return Some(siblings.len());
        }
    }
    None
}

/// How many threads llama.cpp may use for a forward pass (issue #20).
///
/// `models.n_threads` if set, otherwise the machine's **physical** core count.
/// It is not llama.cpp's library default, which is the constant
/// `GGML_DEFAULT_N_THREADS = 4` on every machine — the binding crate pins it in
/// a doctest (`assert_eq!(params.n_threads(), 4)`) and llama.cpp annotates it
/// `// TODO: better default`. Inheriting it ran this 16-thread box on four
/// threads; the #9 audit runs showed a user/wall ratio of 3.42× and 3.34×,
/// which is what four compute threads plus serial overhead looks like.
///
/// **Physical, not logical, and that is measured rather than assumed.** On this
/// 8-core/16-thread machine, query latency falls from 12.3 s at 4 threads to
/// 8.4 s at 8, bottoms at 7.4 s around 12 — and then *collapses to 15 s at 16*,
/// worse than the four-thread default it replaced, with the spread widening
/// from ±0.4 s to ±3 s. Running both SMT siblings of a core turns every ggml
/// barrier into a wait on a thread that is contending for the same execution
/// units.
///
/// The obvious reading of that cliff — "leave the OS some headroom" — is wrong,
/// and the check is in `eval/probes.md`: pinned to eight *distinct* cores so the
/// box looks non-SMT, `n_threads = 8` is the fastest setting tried and does not
/// degrade at all. Threads equal to cores is fine; threads equal to *siblings*
/// is not. So the default keys off cores, which also leaves it safe on the
/// machines where physical and logical are the same number.
///
/// 12 was faster still here (7.4 s vs 8.4 s), but 1.5× physical is a number
/// this box happens to like, not a rule — `models.n_threads` is how a machine
/// that has been swept gets to use its own answer.
///
/// **This must never change a score.** Thread count decides how the arithmetic
/// is scheduled, not what it is. Any output that moves with it is a bug in
/// llama.cpp or in how we build the batch, and would matter more than the
/// latency. Verified: the five seed probes and the served JSON payload are
/// byte-identical at 4, 8, 12 and 16 threads.
///
/// Both context knobs get this value. `n_threads` governs single-token
/// autoregressive decode and `n_threads_batch` governs multi-token forward
/// passes, and engraph is nearly all of the latter: the reranker decodes a
/// whole ~155-token pair and reads the final logits, the embedder encodes a
/// chunk, and only the orchestrator generates — one short JSON object per
/// uncached query.
pub fn resolve_n_threads(config: &crate::config::Config) -> i32 {
    let n = config
        .models
        .n_threads
        .or_else(physical_cores)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(FALLBACK_N_THREADS as usize)
        });
    // A zero or absurd value would be worse than the default it replaced.
    i32::try_from(n.max(1)).unwrap_or(i32::MAX)
}

// ── LlamaEmbed — GGUF embedding model via llama.cpp ──────────────────────────

/// GGUF embedding model loaded via llama.cpp.
///
/// Loads a quantized embedding model from a GGUF file and produces dense float
/// vectors via llama.cpp's built-in embedding support with mean pooling + L2
/// normalization. Supports Metal acceleration on macOS automatically.
///
/// `LlamaModel` is `Send + Sync`, so this struct is `Send`. A `LlamaContext`
/// borrows the model it was made from, so it cannot be a field here — but it
/// can, and does, span a whole batch (issue #13). The global `LlamaBackend` is
/// referenced via `llama_backend()` — no need to store it per-struct.
pub struct LlamaEmbed {
    model: LlamaModel,
    tokenizer: FlexTokenizer,
    dim: usize,
    prompt_format: PromptFormat,
    /// Resolved once at load — see [`resolve_n_threads`].
    n_threads: i32,
}

// Safety: LlamaModel is Send+Sync per llama-cpp-2 docs.
// FlexTokenizer contains only Send types (tokenizers::Tokenizer is Send, shimmytok::Tokenizer is Send).
// We never store a LlamaContext (which is !Send) — it lives inside a single call.
unsafe impl Send for LlamaEmbed {}

impl std::fmt::Debug for LlamaEmbed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaEmbed")
            .field("dim", &self.dim)
            .field("prompt_format", &self.prompt_format)
            .finish()
    }
}

impl LlamaEmbed {
    /// Load a GGUF embedding model from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load tokenizer (try same repo's tokenizer.json, then repo without -GGUF suffix)
    /// 4. Load GGUF model via llama.cpp
    /// 5. Detect prompt format from filename
    pub fn new(models_dir: &Path, config: &crate::config::Config) -> Result<Self> {
        let defaults = ModelDefaults::default();
        let uri_str = config
            .models
            .embed
            .as_deref()
            .unwrap_or(&defaults.embed_uri);
        let uri = HfModelUri::parse(uri_str)?;
        let model_path = ensure_model(&uri, models_dir)?;

        // Load tokenizer: try from the same HF repo, then from the non-GGUF variant.
        let tokenizer = load_tokenizer_for_model(&uri, models_dir)?;

        // Detect prompt format from filename.
        let prompt_format = PromptFormat::detect(&uri.filename);

        // Get or initialize the global llama.cpp backend, then load model.
        let backend = llama_backend()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("loading GGUF model {}: {e}", model_path.display()))?;

        // Output dimensionality is the model's own, read from the GGUF. It is
        // never a constant: whatever `models.embed` points at decides it, and
        // that one value is what gets stored, indexed and queried (issue #12).
        let dim = usize::try_from(model.n_embd())
            .map_err(|_| anyhow::anyhow!("model reported a negative embedding dimension"))?;
        if dim == 0 {
            bail!("model {uri_str} reports an embedding dimension of 0");
        }

        let n_threads = resolve_n_threads(config);
        tracing::info!(
            "loaded LlamaEmbed from {}, dim={}, n_threads={}",
            uri_str,
            dim,
            n_threads
        );

        Ok(Self {
            model,
            tokenizer,
            dim,
            prompt_format,
            n_threads,
        })
    }

    /// Run embedding inference over already prompt-formatted `texts` and return
    /// their L2-normalized vectors, in order.
    ///
    /// **One llama.cpp context serves the whole slice** (issue #13). Creating
    /// one per text was never forced by `!Send` — that constrains what may be a
    /// struct *field*, not what may live in a loop — and it made `batch_size`
    /// decorative, because `index_file` batches the work and this unrolled it
    /// again. The real constraint is that `new_context` borrows the model, so
    /// the context cannot outlive a call; a batch is the largest scope it can
    /// have, and is the one it now gets.
    ///
    /// The context is sized to the longest input in the slice, trading N
    /// right-sized KV allocations for one worst-case allocation. Each text is
    /// encoded from a cleared cache, which is what keeps the vectors identical
    /// to the per-call version — verified: the eval vault's index is
    /// byte-identical across the change.
    ///
    /// The slice is one file's chunks, because that is where `index_file`
    /// batches (`texts.chunks(config.batch_size)`, inside the per-file loop).
    /// On the eval vault that is 247 contexts instead of 1598, for a saving of
    /// at most ~2% of index time — context setup is cheap enough that the win
    /// is small. `batch_size` is no longer ignored, but with a 6.5-chunk mean
    /// file its default of 64 still never fills a batch.
    ///
    /// Vectors come back at the model's full width. Nothing is discarded here —
    /// optional Matryoshka truncation is a separate, opt-in feature and must not
    /// be the silent default (issue #12).
    fn embed_formatted(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize up front: the context has to be sized to the longest input
        // in the batch before any of them can be encoded.
        // Use AddBos::Never because PromptFormat already adds <bos> for embeddinggemma.
        let tokenized = texts
            .iter()
            .map(|text| {
                let tokens = self
                    .model
                    .str_to_token(text, AddBos::Never)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
                if tokens.is_empty() {
                    bail!("tokenizer returned empty token sequence");
                }
                Ok(tokens)
            })
            .collect::<Result<Vec<_>>>()?;

        // n_ubatch must be >= n_tokens for the encoder, and n_ctx must fit all
        // tokens — of the longest member, now that the context is shared.
        let max_tokens = tokenized
            .iter()
            .map(|t| t.len())
            .max()
            .expect("texts is non-empty") as u32;
        let n_ctx = std::num::NonZeroU32::new(max_tokens.max(64) + 16);
        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_n_ctx(n_ctx)
            .with_n_ubatch(max_tokens.max(512))
            .with_n_batch(max_tokens.max(512))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);
        let mut ctx = self
            .model
            .new_context(llama_backend()?, ctx_params)
            .map_err(|e| anyhow::anyhow!("creating embedding context: {e}"))?;

        // One batch buffer, reused. Allocated for the longest input for the
        // same reason the context is.
        let mut batch = LlamaBatch::new(max_tokens as usize + 16, 1);
        let mut vectors = Vec::with_capacity(tokenized.len());

        for tokens in &tokenized {
            // Every text is its own sequence 0, encoded from an empty cache —
            // no state carries over from the previous one.
            batch.clear();
            ctx.clear_kv_cache();

            // Add tokens — mark all as outputs for embedding.
            batch
                .add_sequence(tokens, 0, true)
                .map_err(|e| anyhow::anyhow!("adding sequence to batch: {e}"))?;

            // Encode (compute embeddings). Use encode() for embedding models.
            ctx.encode(&mut batch)
                .map_err(|e| anyhow::anyhow!("embedding encode failed: {e}"))?;

            // Get embeddings for sequence 0 (mean pooled by llama.cpp).
            let embeddings = ctx
                .embeddings_seq_ith(0)
                .map_err(|e| anyhow::anyhow!("getting embeddings: {e}"))?;

            // The width llama.cpp returns must be the width we told the store to
            // expect. A disagreement means `dim` and the model have come apart, and
            // silently storing a short vector is how issue #12 happened.
            if embeddings.len() != self.dim {
                bail!(
                    "model returned {} dimensions, expected {}",
                    embeddings.len(),
                    self.dim
                );
            }

            // L2 normalize — `embeddings_seq_ith` returns the raw pooled vector.
            let norm: f32 = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
            vectors.push(if norm > 0.0 {
                embeddings.iter().map(|x| x / norm).collect()
            } else {
                embeddings.to_vec()
            });
        }

        Ok(vectors)
    }

    /// Embed a single already prompt-formatted text.
    ///
    /// A one-element call into [`LlamaEmbed::embed_formatted`], so single and
    /// batch embedding cannot drift apart.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let owned = [text.to_owned()];
        self.embed_formatted(&owned)?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding returned no vector"))
    }
}

impl EmbedModel for LlamaEmbed {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Apply document prompt format for indexing (asymmetric models need this),
        // then hand the whole batch to one context (issue #13).
        let formatted: Vec<String> = texts
            .iter()
            .map(|t| self.prompt_format.format_document("", t))
            .collect();
        self.embed_formatted(&formatted)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        // Apply query prompt format (asymmetric models like embeddinggemma need this).
        let formatted = self.prompt_format.format_query(text);
        self.embed_text(&formatted)
    }

    fn token_count(&self, text: &str) -> usize {
        self.tokenizer.token_count(text)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ── Heuristic orchestrator ───────────────────────────────────────────────────

/// Heuristic orchestrator — no LLM, fast path when intelligence is off.
pub fn heuristic_orchestrate(query: &str) -> OrchestrationResult {
    let trimmed = query.trim();

    // Temporal: detect date/time references in the query
    let date_range = crate::temporal::parse_date_range_heuristic(query);
    if date_range.is_some() {
        return OrchestrationResult {
            intent: QueryIntent::Temporal,
            expansions: vec![trimmed.to_string()],
            date_range,
        };
    }

    // Exact: docids (#abc123) or ticket IDs (ABC-1234)
    if trimmed.starts_with('#') && trimmed.len() <= 8 {
        return OrchestrationResult {
            intent: QueryIntent::Exact,
            expansions: vec![trimmed.to_string()],
            date_range: None,
        };
    }
    // Ticket ID pattern: PREFIX-1234
    if trimmed.contains('-')
        && let Some(prefix) = trimmed.split('-').next()
        && prefix.chars().all(|c| c.is_ascii_uppercase())
    {
        let after = trimmed.split('-').nth(1).unwrap_or("");
        if after.chars().all(|c| c.is_ascii_digit()) && !after.is_empty() {
            return OrchestrationResult {
                intent: QueryIntent::Exact,
                expansions: vec![trimmed.to_string()],
                date_range: None,
            };
        }
    }

    // Relationship: "who" queries
    let lower = trimmed.to_lowercase();
    if lower.starts_with("who ") || lower.contains(" who ") {
        return OrchestrationResult {
            intent: QueryIntent::Relationship,
            expansions: vec![trimmed.to_string()],
            date_range: None,
        };
    }

    // Default: exploratory with word splitting for multi-word queries
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let mut expansions = vec![trimmed.to_string()];
    if words.len() > 2 {
        let stopwords = [
            "how", "does", "the", "a", "an", "is", "are", "was", "to", "in", "on", "for", "with",
            "what", "when", "where",
        ];
        for word in &words {
            if word.len() > 2 && !stopwords.contains(&word.to_lowercase().as_str()) {
                expansions.push(word.to_string());
            }
        }
    }

    OrchestrationResult {
        intent: QueryIntent::Exploratory,
        expansions,
        date_range: None,
    }
}

// ── Orchestration JSON parsing ────────────────────────────────────────────────

/// Parse orchestration JSON from LLM output.
/// Handles: raw JSON, JSON embedded in text, and partial/malformed responses.
pub fn parse_orchestration_json(text: &str) -> Result<OrchestrationResult> {
    let json_str = extract_json_object(text)
        .ok_or_else(|| anyhow::anyhow!("no JSON object found in LLM response"))?;

    let parsed: serde_json::Value =
        serde_json::from_str(json_str).with_context(|| "parsing orchestration JSON")?;

    let intent_str = parsed["intent"].as_str().unwrap_or("exploratory");
    let intent = match intent_str {
        "exact" => QueryIntent::Exact,
        "conceptual" => QueryIntent::Conceptual,
        "relationship" => QueryIntent::Relationship,
        "temporal" => QueryIntent::Temporal,
        _ => QueryIntent::Exploratory,
    };

    let expansions: Vec<String> = parsed["expansions"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if expansions.is_empty() {
        anyhow::bail!("no expansions in orchestration response");
    }

    let date_range = crate::temporal::parse_date_range_from_json(&parsed);
    let intent = if date_range.is_some() && intent != QueryIntent::Temporal {
        QueryIntent::Temporal
    } else {
        intent
    };

    Ok(OrchestrationResult {
        intent,
        expansions,
        date_range,
    })
}

/// Extract the first JSON object ({...}) from text, handling nested braces.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0;
    for (i, b) in text[start..].bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

// ── LlamaOrchestrator — GGUF text generation via llama.cpp ─────────────────────

const ORCHESTRATOR_SYSTEM_PROMPT: &str = r#"You are a search query analyzer. Given a user's search query, classify it and expand it.

Return JSON with:
- "intent": one of "exact", "conceptual", "relationship", "exploratory", "temporal"
- "expansions": 2-4 alternative phrasings (always include the original query first)
- "date_range": (only for temporal queries) {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}

Use "temporal" intent when the query references a time period (e.g. "yesterday", "last week", "March 2026").

Be concise. Only return the JSON object."#;

/// Quantized Qwen3 model for query orchestration and expansion via llama.cpp.
///
/// Loads a Qwen3 GGUF model and performs autoregressive generation to classify
/// queries and produce expansions. Falls back to `heuristic_orchestrate` if
/// generation or JSON parsing fails. Uses Metal acceleration on macOS automatically.
///
/// Uses llama.cpp's built-in tokenizer for both encoding and decoding — no
/// external tokenizer.json required. The global `LlamaBackend` is used via
/// `llama_backend()`.
pub struct LlamaOrchestrator {
    model: LlamaModel,
    /// Resolved once at load — see [`resolve_n_threads`].
    n_threads: i32,
}

// Safety: LlamaModel is Send+Sync per llama-cpp-2 docs.
// LlamaContext is created per-call and never stored.
unsafe impl Send for LlamaOrchestrator {}

impl std::fmt::Debug for LlamaOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaOrchestrator").finish()
    }
}

impl LlamaOrchestrator {
    /// Load a Qwen3 GGUF model for orchestration from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load GGUF model via llama.cpp (uses built-in tokenizer — no tokenizer.json needed)
    pub fn new(models_dir: &Path, config: &crate::config::Config) -> Result<Self> {
        let defaults = ModelDefaults::default();
        let uri_str = config
            .models
            .expand
            .as_deref()
            .unwrap_or(&defaults.expand_uri);
        let uri = HfModelUri::parse(uri_str)?;
        let model_path = ensure_model(&uri, models_dir)?;

        // Use global backend and llama.cpp's built-in tokenizer (no tokenizer.json required).
        let backend = llama_backend()?;
        let model_params = LlamaModelParams::default();
        let model =
            LlamaModel::load_from_file(backend, &model_path, &model_params).map_err(|e| {
                anyhow::anyhow!("loading orchestrator model {}: {e}", model_path.display())
            })?;

        let n_threads = resolve_n_threads(config);
        tracing::info!(
            "loaded LlamaOrchestrator from {}, n_threads={}",
            uri_str,
            n_threads
        );

        Ok(Self { model, n_threads })
    }

    /// Format a chat prompt in Qwen3 ChatML format.
    fn format_prompt(query: &str) -> String {
        format!(
            "<|im_start|>system\n{ORCHESTRATOR_SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n{query}<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }

    /// Run autoregressive generation (greedy decode) up to `max_tokens`.
    /// Returns the generated text (excluding the prompt).
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Tokenize using llama.cpp's built-in tokenizer.
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        if tokens.is_empty() {
            bail!("tokenizer returned empty token sequence");
        }

        // Create context per-call (LlamaContext is !Send).
        let n_ctx = (tokens.len() + max_tokens + 16) as u32;
        // The one place `n_threads` (as opposed to `n_threads_batch`) does any
        // work: the loop below decodes a single token at a time.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);
        let mut ctx = self
            .model
            .new_context(llama_backend()?, ctx_params)
            .map_err(|e| anyhow::anyhow!("creating orchestrator context: {e}"))?;

        // Process prompt tokens in a batch.
        let mut batch = LlamaBatch::new(tokens.len() + max_tokens + 16, 1);
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| anyhow::anyhow!("adding prompt token to batch: {e}"))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("prompt decode failed: {e}"))?;

        // Autoregressive generation loop.
        let mut sampler = LlamaSampler::greedy();
        let mut output = String::new();
        // Each token may produce multi-byte UTF-8 sequences; use an encoding_rs decoder
        // to correctly reassemble them across token boundaries.
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let prompt_len = tokens.len();

        for step in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            // Check for end-of-generation.
            if self.model.is_eog_token(new_token) {
                break;
            }

            // Decode this token to text using llama.cpp's built-in tokenizer.
            let piece = self
                .model
                .token_to_piece(new_token, &mut decoder, false, None)
                .map_err(|e| anyhow::anyhow!("token_to_piece failed: {e}"))?;
            output.push_str(&piece);

            // Add token to batch for next iteration.
            batch.clear();
            batch
                .add(new_token, (prompt_len + step) as i32, &[0], true)
                .map_err(|e| anyhow::anyhow!("adding generated token to batch: {e}"))?;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("generation decode failed: {e}"))?;
        }

        Ok(output)
    }
}

impl OrchestratorModel for LlamaOrchestrator {
    fn orchestrate(&mut self, query: &str) -> Result<OrchestrationResult> {
        let prompt = Self::format_prompt(query);

        match self.generate(&prompt, 256) {
            Ok(text) => match parse_orchestration_json(&text) {
                Ok(result) => Ok(result),
                Err(e) => {
                    tracing::warn!(
                        "orchestrator JSON parse failed, falling back to heuristic: {e:#}"
                    );
                    Ok(heuristic_orchestrate(query))
                }
            },
            Err(e) => {
                tracing::warn!("orchestrator generation failed, falling back to heuristic: {e:#}");
                Ok(heuristic_orchestrate(query))
            }
        }
    }
}

// ── LlamaRerank — GGUF cross-encoder reranker via llama.cpp ─────────────────────

/// Format query+document for cross-encoder reranking.
pub fn format_reranker_input(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the document is relevant to the search query. \
         Respond only with \"Yes\" or \"No\".<|im_end|>\n\
         <|im_start|>user\nSearch query: {query}\nDocument: {document}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Quantized Qwen3 cross-encoder for reranking search results via llama.cpp.
///
/// Loads a Qwen3-Reranker GGUF model and scores (query, document) pairs by
/// running a single forward pass and extracting Yes/No logit probabilities.
/// Unlike `LlamaOrchestrator`, this does NOT do autoregressive generation —
/// just one pass through the full input to get logits at the last position.
///
/// Uses llama.cpp's built-in tokenizer to look up Yes/No token IDs — no
/// external tokenizer.json required. The global `LlamaBackend` is used via
/// `llama_backend()`.
pub struct LlamaRerank {
    model: LlamaModel,
    yes_token_id: i32,
    no_token_id: i32,
    /// Resolved once at load — see [`resolve_n_threads`].
    n_threads: i32,
}

// Safety: LlamaModel is Send+Sync per llama-cpp-2 docs.
// LlamaContext borrows the model, so it lives inside a call and is never stored.
unsafe impl Send for LlamaRerank {}

impl std::fmt::Debug for LlamaRerank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaRerank")
            .field("yes_token_id", &self.yes_token_id)
            .field("no_token_id", &self.no_token_id)
            .finish()
    }
}

impl LlamaRerank {
    /// Load a Qwen3-Reranker GGUF model from `models_dir`.
    ///
    /// Steps:
    /// 1. Resolve model URI (from config override or `ModelDefaults::default().rerank_uri`)
    /// 2. `ensure_model()` to download if needed
    /// 3. Load GGUF model via llama.cpp
    /// 4. Look up Yes/No token IDs using the model's built-in tokenizer (no tokenizer.json needed)
    pub fn new(models_dir: &Path, config: &crate::config::Config) -> Result<Self> {
        let defaults = ModelDefaults::default();
        let uri_str = config
            .models
            .rerank
            .as_deref()
            .unwrap_or(&defaults.rerank_uri);
        let uri = HfModelUri::parse(uri_str)?;
        let model_path = ensure_model(&uri, models_dir)?;

        // Use global backend and llama.cpp's built-in tokenizer (no tokenizer.json required).
        let backend = llama_backend()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("loading reranker model {}: {e}", model_path.display()))?;

        // Look up Yes/No token IDs via the model's built-in tokenizer.
        // str_to_token returns Vec<LlamaToken>; we take the first token ID (skip BOS).
        let yes_tokens = model
            .str_to_token("Yes", AddBos::Never)
            .map_err(|e| anyhow::anyhow!("tokenizing 'Yes': {e}"))?;
        let yes_token_id = yes_tokens
            .first()
            .map(|t| t.0)
            .ok_or_else(|| anyhow::anyhow!("model tokenizer returned no tokens for 'Yes'"))?;

        let no_tokens = model
            .str_to_token("No", AddBos::Never)
            .map_err(|e| anyhow::anyhow!("tokenizing 'No': {e}"))?;
        let no_token_id = no_tokens
            .first()
            .map(|t| t.0)
            .ok_or_else(|| anyhow::anyhow!("model tokenizer returned no tokens for 'No'"))?;

        let n_threads = resolve_n_threads(config);
        tracing::info!(
            "loaded LlamaRerank from {}, yes_id={}, no_id={}, n_threads={}",
            uri_str,
            yes_token_id,
            no_token_id,
            n_threads
        );

        Ok(Self {
            model,
            yes_token_id,
            no_token_id,
            n_threads,
        })
    }
}

impl RerankModel for LlamaRerank {
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32> {
        // One code path: a single pair is a batch of one.
        self.rerank_batch(query, &[document])?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("reranker returned no score"))
    }

    /// Score every pair through **one** llama.cpp context (issue #13).
    ///
    /// A reranked search scores 30 candidates; before this, that was 30 context
    /// creations to do 30 single forward passes. The context is sized to the
    /// longest pair and each is decoded from a cleared cache, so nothing carries
    /// between candidates and the scores are the per-call ones.
    ///
    /// Measured, the saving is **not** why this is worth having: on the eval
    /// vault a reranked query is 8.10 s either way (n=20, medians 8.105 →
    /// 8.096). Context setup for a 0.6B model is milliseconds, and 30 of them
    /// sit under the noise floor. What this buys is the shape — one call with
    /// the whole candidate set, which is what issues #14 and #15 build on.
    fn rerank_batch(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize up front — the context is sized to the longest pair.
        let tokenized = documents
            .iter()
            .map(|document| {
                let input_text = format_reranker_input(query, document);
                let tokens = self
                    .model
                    .str_to_token(&input_text, AddBos::Always)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
                if tokens.is_empty() {
                    bail!("tokenizer returned empty token sequence");
                }
                Ok(tokens)
            })
            .collect::<Result<Vec<_>>>()?;

        let max_tokens = tokenized
            .iter()
            .map(|t| t.len())
            .max()
            .expect("documents is non-empty");
        let n_ctx = (max_tokens + 16) as u32;
        // `n_threads_batch` is the one that matters here: each candidate is a
        // single multi-token forward pass, with no generation at all (#20).
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);
        let mut ctx = self
            .model
            .new_context(llama_backend()?, ctx_params)
            .map_err(|e| anyhow::anyhow!("creating reranker context: {e}"))?;

        let mut batch = LlamaBatch::new(max_tokens + 16, 1);
        let mut scores = Vec::with_capacity(tokenized.len());

        for tokens in &tokenized {
            // Each pair is judged on its own, from position 0 with an empty
            // cache — a cross-encoder score must not depend on what was scored
            // before it.
            batch.clear();
            ctx.clear_kv_cache();

            // Add all tokens; mark last as logit-producing.
            for (i, token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                batch
                    .add(*token, i as i32, &[0], is_last)
                    .map_err(|e| anyhow::anyhow!("adding token to reranker batch: {e}"))?;
            }

            // Single forward pass through the full input.
            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("reranker decode failed: {e}"))?;

            // Get logits for the last token position.
            let logits = ctx.get_logits_ith(batch.n_tokens() - 1);

            // Extract Yes/No logits and compute softmax probability.
            let yes_logit = logits[self.yes_token_id as usize];
            let no_logit = logits[self.no_token_id as usize];

            let max_logit = yes_logit.max(no_logit);
            let yes_exp = (yes_logit - max_logit).exp();
            let no_exp = (no_logit - max_logit).exp();
            scores.push(yes_exp / (yes_exp + no_exp));
        }

        Ok(scores)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset means the machine, not llama.cpp's constant 4 (issue #20).
    ///
    /// The bug this guards is silent: inheriting `GGML_DEFAULT_N_THREADS` is not
    /// an error, produces correct output, and simply runs the models on four
    /// threads of whatever box they are on. The only visible symptom was a
    /// user/wall ratio pinned just under 4.
    #[test]
    fn n_threads_defaults_to_the_machine() {
        let config = crate::config::Config::default();
        assert_eq!(config.models.n_threads, None, "unset by default");

        let expected = physical_cores().unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(FALLBACK_N_THREADS as usize)
        }) as i32;
        assert_eq!(resolve_n_threads(&config), expected);
        assert!(resolve_n_threads(&config) >= 1);
    }

    /// The default counts cores, not SMT siblings (issue #20).
    ///
    /// Logical count is the tempting default and it is the *worst* setting
    /// measured on this box: 15 s a query against 12 s for the four threads it
    /// was replacing, because both siblings of a core end up in the same ggml
    /// barrier. Skipped where sysfs cannot answer.
    #[test]
    fn the_default_counts_cores_not_siblings() {
        let Some(physical) = physical_cores() else {
            return;
        };
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(physical);
        assert!(physical >= 1);
        assert!(
            physical <= logical,
            "physical cores ({physical}) cannot exceed logical CPUs ({logical})"
        );
        assert_eq!(
            resolve_n_threads(&crate::config::Config::default()),
            physical as i32
        );
    }

    /// An explicit setting wins, including one that asks for fewer threads than
    /// the machine has — leaving headroom is a legitimate reason to set this.
    #[test]
    fn n_threads_honours_an_explicit_setting() {
        let mut config = crate::config::Config::default();
        for n in [1, 2, 4, 8, 64] {
            config.models.n_threads = Some(n);
            assert_eq!(resolve_n_threads(&config), n as i32);
        }
    }

    /// Zero threads is not a request llama.cpp can serve; clamp rather than pass
    /// it through and fail at context creation.
    #[test]
    fn n_threads_never_resolves_below_one() {
        let mut config = crate::config::Config::default();
        config.models.n_threads = Some(0);
        assert_eq!(resolve_n_threads(&config), 1);

        config.models.n_threads = Some(usize::MAX);
        assert!(resolve_n_threads(&config) > 0, "must not overflow negative");
    }

    #[test]
    fn test_mock_embed_deterministic() {
        let mut mock = MockLlm::new(256);
        let v1 = mock.embed_one("hello").unwrap();
        let v2 = mock.embed_one("hello").unwrap();
        assert_eq!(v1.len(), 256);
        assert_eq!(v1, v2, "same input must produce same output");
    }

    #[test]
    fn test_mock_embed_different_inputs() {
        let mut mock = MockLlm::new(256);
        let v1 = mock.embed_one("hello").unwrap();
        let v2 = mock.embed_one("world").unwrap();
        assert_ne!(v1, v2, "different inputs should produce different vectors");
    }

    #[test]
    fn test_mock_embed_batch() {
        let mut mock = MockLlm::new(256);
        let vecs = mock.embed_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 256));
    }

    #[test]
    fn test_mock_embed_normalized() {
        let mut mock = MockLlm::new(256);
        let v = mock.embed_one("test").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "mock vectors should be L2-normalized"
        );
    }

    #[test]
    fn test_mock_rerank() {
        let mut mock = MockLlm::new(256);
        let score = mock.rerank_score("query", "document text").unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_mock_orchestrate() {
        let mut mock = MockLlm::new(256);
        let result = mock.orchestrate("how does auth work").unwrap();
        assert_eq!(result.intent, QueryIntent::Exploratory);
        assert!(!result.expansions.is_empty());
        assert_eq!(result.expansions[0], "how does auth work");
    }

    #[test]
    fn the_default_rerank_batch_agrees_with_scoring_each_pair() {
        // MockLlm does not override `rerank_batch`, so this exercises the
        // trait's default implementation — the one every backend without setup
        // to amortize keeps (issue #13).
        let documents = ["dragon lore", "banking regulations", "a temple at dusk"];
        let mut mock = MockLlm::new(256);
        let batched = mock.rerank_batch("dragon", &documents).unwrap();

        let one_at_a_time: Vec<f32> = documents
            .iter()
            .map(|d| mock.rerank_score("dragon", d).unwrap())
            .collect();

        assert_eq!(
            batched, one_at_a_time,
            "batching must not change the scores"
        );
    }

    #[test]
    fn rerank_batch_of_nothing_is_no_scores() {
        let mut mock = MockLlm::new(256);
        assert!(mock.rerank_batch("dragon", &[]).unwrap().is_empty());
    }

    #[test]
    fn test_mock_rerank_empty_query() {
        let mut mock = MockLlm::new(256);
        let score = mock.rerank_score("", "document text").unwrap();
        assert_eq!(score, 0.0, "empty query should score 0.0");
    }

    #[test]
    fn test_lane_weights_from_intent() {
        let exact = LaneWeights::from_intent(&QueryIntent::Exact);
        assert!(exact.fts > exact.semantic, "exact intent should favor FTS");

        let conceptual = LaneWeights::from_intent(&QueryIntent::Conceptual);
        assert!(
            conceptual.semantic > conceptual.fts,
            "conceptual should favor semantic"
        );

        let relationship = LaneWeights::from_intent(&QueryIntent::Relationship);
        assert!(
            relationship.graph >= LaneWeights::from_intent(&QueryIntent::Exact).graph,
            "relationship should still lean on the graph more than a lookup does"
        );
    }

    /// The graph lane must never outweigh a content lane, under any intent.
    ///
    /// Not a style rule — issue #9. `graph_expand` excludes seed files, so graph
    /// results share no documents with the semantic or FTS lanes and can never
    /// gain an agreement term in RRF. Their fused score is therefore just
    /// `weight/(60+rank)`, and once that constant is large enough for the lane's
    /// *worst* result to beat a content lane's *best*, the graph lane takes the
    /// entire ranking. With expansions capped at 20 and `k = 60`, the crossover
    /// is at `graph/80 > content/61`, i.e. a ratio of about 1.31.
    ///
    /// Holding graph at or below the content lanes keeps it inside the ranking
    /// instead of on top of it. Delete this test only along with the
    /// disjointness itself (#15).
    #[test]
    fn the_graph_lane_never_outweighs_a_content_lane() {
        let intents = [
            QueryIntent::Exact,
            QueryIntent::Conceptual,
            QueryIntent::Relationship,
            QueryIntent::Exploratory,
            QueryIntent::Temporal,
        ];
        for intent in &intents {
            let w = LaneWeights::from_intent(intent);
            let content = w.semantic.max(w.fts);
            assert!(
                w.graph <= content,
                "{intent:?}: graph {} exceeds the strongest content lane {content} — \
                 disjoint graph results would crowd the ranking out (#9)",
                w.graph,
            );
        }
        let w = LaneWeights::default_no_intelligence();
        assert!(w.graph <= w.semantic.max(w.fts));
    }

    #[test]
    fn test_parse_hf_uri() {
        let uri = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
        let parsed = HfModelUri::parse(uri).unwrap();
        assert_eq!(parsed.repo, "ggml-org/embeddinggemma-300M-GGUF");
        assert_eq!(parsed.filename, "embeddinggemma-300M-Q8_0.gguf");
        assert_eq!(
            parsed.download_url(),
            "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf"
        );
    }

    #[test]
    fn test_parse_hf_uri_invalid() {
        assert!(HfModelUri::parse("not-a-hf-uri").is_err());
        assert!(HfModelUri::parse("hf:only-repo").is_err());
    }

    #[test]
    fn test_model_cache_path() {
        let uri = HfModelUri::parse("hf:ggml-org/embeddinggemma-300M-GGUF/model.gguf").unwrap();
        let cache_dir = std::path::Path::new("/tmp/models");
        let path = uri.cache_path(cache_dir);
        assert_eq!(
            path,
            cache_dir.join("ggml-org--embeddinggemma-300M-GGUF--model.gguf")
        );
    }

    #[test]
    fn test_model_defaults() {
        let defaults = ModelDefaults::default();
        assert!(defaults.embed_uri.starts_with("hf:"));
        // No dimension here on purpose — it belongs to the model (issue #12).
        assert!(
            defaults.embed_uri.contains("embeddinggemma"),
            "default embed model should be embeddinggemma"
        );
    }

    // ── LlamaEmbed / PromptFormat tests ────────────────────────────────────

    #[test]
    fn test_llama_embed_struct_exists() {
        fn assert_embed_model<E: EmbedModel>(_e: &E) {}
        let mock = MockLlm::new(256);
        assert_embed_model(&mock);
        // LlamaEmbed also implements EmbedModel — verified at compile time.
        // We can't instantiate LlamaEmbed without a real GGUF model,
        // but the trait bound compiles.
    }

    #[test]
    fn test_prompt_format_embeddinggemma_query() {
        let fmt = PromptFormat::detect("embeddinggemma-300M-Q8_0.gguf");
        let formatted = fmt.format_query("how does auth work");
        assert!(formatted.contains("search_query"));
        assert!(formatted.contains("how does auth work"));
    }

    #[test]
    fn test_prompt_format_embeddinggemma_document() {
        let fmt = PromptFormat::detect("embeddinggemma-300M-Q8_0.gguf");
        let formatted = fmt.format_document("Note Title", "some content");
        assert!(formatted.contains("Note Title"));
        assert!(formatted.contains("some content"));
        assert!(formatted.contains("search_document"));
    }

    #[test]
    fn test_prompt_format_unknown_model() {
        let fmt = PromptFormat::detect("unknown-model.gguf");
        let formatted = fmt.format_query("test query");
        assert_eq!(formatted, "test query");
    }

    #[test]
    fn test_prompt_format_qwen_embedding() {
        let fmt = PromptFormat::detect("qwen-embed-v2.gguf");
        let formatted = fmt.format_query("find me something");
        assert!(formatted.contains("Instruct:"));
        assert!(formatted.contains("Query:"));
        assert!(formatted.contains("find me something"));
    }

    #[test]
    fn test_prompt_format_qwen_document() {
        let fmt = PromptFormat::detect("qwen-embed-v2.gguf");
        let formatted = fmt.format_document("Title", "Body text");
        assert_eq!(formatted, "Title\nBody text");
    }

    #[test]
    fn test_prompt_format_raw_document() {
        let fmt = PromptFormat::detect("random-model.gguf");
        let formatted = fmt.format_document("Title", "Body");
        assert_eq!(formatted, "Title\nBody");
    }

    // ── heuristic_orchestrate tests ──────────────────────────────────────────

    #[test]
    fn test_heuristic_orchestrate_single_word() {
        let result = heuristic_orchestrate("auth");
        assert_eq!(result.intent, QueryIntent::Exploratory);
        assert_eq!(result.expansions, vec!["auth"]);
    }

    #[test]
    fn test_heuristic_orchestrate_multi_word() {
        let result = heuristic_orchestrate("how does auth work");
        assert_eq!(result.intent, QueryIntent::Exploratory);
        assert!(
            result
                .expansions
                .contains(&"how does auth work".to_string())
        );
        assert!(result.expansions.len() > 1);
    }

    #[test]
    fn test_heuristic_orchestrate_docid() {
        let result = heuristic_orchestrate("#ab12cd");
        assert_eq!(result.intent, QueryIntent::Exact);
    }

    #[test]
    fn test_heuristic_orchestrate_ticket_id() {
        let result = heuristic_orchestrate("BRE-1234");
        assert_eq!(result.intent, QueryIntent::Exact);
    }

    #[test]
    fn test_heuristic_orchestrate_who_query() {
        let result = heuristic_orchestrate("who works on checkout");
        assert_eq!(result.intent, QueryIntent::Relationship);
    }

    // ── parse_orchestration_json tests ───────────────────────────────────────

    #[test]
    fn test_parse_orchestration_json_valid() {
        let json =
            r#"{"intent": "conceptual", "expansions": ["auth work", "authentication design"]}"#;
        let result = parse_orchestration_json(json).unwrap();
        assert_eq!(result.intent, QueryIntent::Conceptual);
        assert_eq!(result.expansions.len(), 2);
    }

    #[test]
    fn test_parse_orchestration_json_with_surrounding_text() {
        let text =
            "Here is the analysis:\n{\"intent\": \"exact\", \"expansions\": [\"BRE-1234\"]}\nDone.";
        let result = parse_orchestration_json(text).unwrap();
        assert_eq!(result.intent, QueryIntent::Exact);
    }

    #[test]
    fn test_parse_orchestration_json_invalid() {
        let bad = "not json at all";
        assert!(parse_orchestration_json(bad).is_err());
    }

    #[test]
    fn test_parse_orchestration_json_unknown_intent() {
        let json = r#"{"intent": "unknown_type", "expansions": ["query"]}"#;
        let result = parse_orchestration_json(json).unwrap();
        assert_eq!(result.intent, QueryIntent::Exploratory);
    }

    #[test]
    fn test_extract_json_object_nested() {
        let text = r#"prefix {"a": {"b": 1}} suffix"#;
        let extracted = extract_json_object(text).unwrap();
        assert_eq!(extracted, r#"{"a": {"b": 1}}"#);
    }

    #[test]
    fn test_extract_json_object_none() {
        assert!(extract_json_object("no braces here").is_none());
    }

    #[test]
    fn test_extract_json_object_unclosed() {
        assert!(extract_json_object("{ open but never closed").is_none());
    }

    /// #23: the model is asked for the original query first and routinely
    /// omits it, so the parser stops trusting and the caller repairs it.
    #[test]
    fn the_original_query_is_restored_when_the_model_drops_it() {
        let mut result = parse_orchestration_json(
            r#"{"intent": "conceptual", "expansions": ["Archdragon character", "Archdragon concept"]}"#,
        )
        .unwrap();
        result.ensure_original("Archdragon");
        assert_eq!(
            result.expansions,
            vec!["Archdragon", "Archdragon character", "Archdragon concept"]
        );
    }

    #[test]
    fn restoring_the_original_does_not_duplicate_one_already_present() {
        let mut result = parse_orchestration_json(
            r#"{"intent": "exact", "expansions": ["auth work", "authentication design"]}"#,
        )
        .unwrap();
        result.ensure_original("  AUTH WORK  ");
        assert_eq!(
            result.expansions,
            vec!["auth work", "authentication design"]
        );
    }

    #[test]
    fn restoring_the_original_ignores_a_blank_query() {
        let mut result = OrchestrationResult {
            intent: QueryIntent::Exploratory,
            expansions: vec!["something".to_string()],
            date_range: None,
        };
        result.ensure_original("   ");
        assert_eq!(result.expansions, vec!["something"]);
    }

    #[test]
    fn test_parse_orchestration_json_empty_expansions() {
        let json = r#"{"intent": "exact", "expansions": []}"#;
        assert!(parse_orchestration_json(json).is_err());
    }

    #[test]
    fn test_parse_orchestration_json_missing_expansions() {
        let json = r#"{"intent": "exact"}"#;
        assert!(parse_orchestration_json(json).is_err());
    }

    // ── LlamaOrchestrator tests ─────────────────────────────────────────────

    #[test]
    fn test_llama_orchestrator_format_prompt() {
        let prompt = LlamaOrchestrator::format_prompt("how does auth work");
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("how does auth work"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_llama_orchestrator_implements_trait() {
        // Compile-time check: LlamaOrchestrator implements OrchestratorModel.
        fn assert_orchestrator<O: OrchestratorModel>() {}
        assert_orchestrator::<LlamaOrchestrator>();
    }

    // ── LlamaRerank tests ──────────────────────────────────────────────────

    #[test]
    fn test_format_reranker_input() {
        let formatted = format_reranker_input("auth system", "The auth module handles OAuth");
        assert!(formatted.contains("auth system"));
        assert!(formatted.contains("The auth module handles OAuth"));
        assert!(formatted.contains("Respond only with"));
    }

    #[test]
    fn test_llama_rerank_trait_compliance() {
        // Verify MockLlm still satisfies RerankModel.
        fn assert_rerank<R: RerankModel>(_r: &R) {}
        let mock = MockLlm::new(256);
        assert_rerank(&mock);
    }

    // ── Temporal intent tests ────────────────────────────────────────────────

    #[test]
    fn test_temporal_intent_weights() {
        let weights = LaneWeights::from_intent(&QueryIntent::Temporal);
        assert!(weights.temporal > weights.semantic);
        assert!(weights.temporal > 1.0);
    }

    #[test]
    fn test_non_temporal_intent_has_zero_temporal() {
        let exact = LaneWeights::from_intent(&QueryIntent::Exact);
        assert!((exact.temporal - 0.0).abs() < f64::EPSILON);
        let conceptual = LaneWeights::from_intent(&QueryIntent::Conceptual);
        assert!((conceptual.temporal - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_heuristic_orchestrate_temporal() {
        let result = heuristic_orchestrate("what happened yesterday");
        assert_eq!(result.intent, QueryIntent::Temporal);
        assert!(result.date_range.is_some());
    }

    #[test]
    fn test_heuristic_orchestrate_non_temporal() {
        let result = heuristic_orchestrate("how does auth work");
        assert!(result.date_range.is_none());
        assert_ne!(result.intent, QueryIntent::Temporal);
    }

    #[test]
    fn test_parse_json_with_date_range() {
        let json = r#"{"intent":"temporal","expansions":["last week updates"],"date_range":{"start":"2026-03-19","end":"2026-03-25"}}"#;
        let result = parse_orchestration_json(json).unwrap();
        assert_eq!(result.intent, QueryIntent::Temporal);
        assert!(result.date_range.is_some());
    }

    #[test]
    fn test_parse_json_without_date_range_backward_compat() {
        let json = r#"{"intent":"exact","expansions":["BRE-1234"]}"#;
        let result = parse_orchestration_json(json).unwrap();
        assert!(result.date_range.is_none());
    }
}
