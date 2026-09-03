use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::{LlamaBackendDeviceType, list_llama_ggml_backend_devices};

/// Input wall used when a model does not report a training context length.
/// A conservative floor: every embedder in use reports at least this
/// (issue #75).
pub const FALLBACK_MAX_CONTEXT: usize = 1024;

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

/// The compute device llama.cpp resolved for this process, as a stable string.
///
/// Folded into [`LlamaEmbed`]'s and [`LlamaRerank`]'s fingerprints (issue #33).
/// CUDA and CPU kernels are not bitwise identical, so a store built on one and
/// extended on the other holds vectors from two devices while reporting itself
/// healthy — the exact silent staleness `fingerprint` exists to end.
///
/// Read at load rather than from `cfg!(feature = "cuda")`. A compile-time token
/// records the *intent* to offload; only a runtime one records what the process
/// actually got. VRAM on this box is shared with the Windows host, so one binary
/// can come up on either device depending on what was free at the time.
///
/// `memory_free` is deliberately not a component: it moves between two runs of
/// the same binary on the same device, and a fingerprint that changes when
/// nothing did is a full re-index nobody asked for.
pub fn device_identity() -> String {
    // The ggml device registry is populated by backend init, so asking before
    // that would answer `cpu` on a CUDA build — a wrong reading that stamps a
    // GPU-built index as CPU-built, which is the one outcome this must not have.
    // Every caller has already initialized it; the `OnceLock` makes this a get.
    if llama_backend().is_err() {
        return "unknown".to_string();
    }

    let mut accelerators: Vec<String> = list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|device| device.device_type != LlamaBackendDeviceType::Cpu)
        .map(|device| format!("{}/{}", device.backend, device.description))
        .collect();
    // The registry's order is the order backends registered themselves, which is
    // not a promise. Sorting makes the string depend on the set, not the walk.
    accelerators.sort();
    accelerators.dedup();

    if accelerators.is_empty() {
        "cpu".to_string()
    } else {
        accelerators.join("+")
    }
}

/// Compose `embedding_fingerprint`'s model half from its six components.
///
/// Separate from [`LlamaEmbed::new`] so the composition can be exercised without
/// a GGUF on disk or a GPU in the box: what issue #33 needs to hold is that two
/// devices give two fingerprints, and that is a property of this function alone.
///
/// `n_threads` is deliberately absent — threads change how the arithmetic is
/// scheduled, never its result. `device` is present for the opposite reason.
fn embed_fingerprint(
    artifact: &str,
    dim: usize,
    tokenizer_identity: &str,
    prompt_format: &PromptFormat,
    device: &str,
) -> String {
    crate::fingerprint::digest(&[
        artifact,
        &dim.to_string(),
        tokenizer_identity,
        &format!(
            "{}:{}",
            crate::fingerprint::PROMPT_TEMPLATE_VERSION,
            prompt_format.template_id()
        ),
        &crate::fingerprint::EMBEDDING_NORMALIZATION_VERSION.to_string(),
        device,
    ])
}

/// Compose `reranker_fingerprint`'s model half. Separate from
/// [`LlamaRerank::new`] for the same reason as [`embed_fingerprint`].
///
/// The Yes/No ids are components because they are the reranker's whole output
/// contract: a model whose vocabulary numbers them differently produces scores
/// on a different scale from the same weights.
fn rerank_fingerprint(artifact: &str, yes_token_id: i32, no_token_id: i32, device: &str) -> String {
    crate::fingerprint::digest(&[
        artifact,
        &yes_token_id.to_string(),
        &no_token_id.to_string(),
        device,
    ])
}

// ── Prompt format ────────────────────────────────────────────────────────────

/// Which document template the `EmbeddingGemma` prompt format writes (issue #10).
///
/// The choice decides what a stored vector means, so it is a component of
/// `embedding_fingerprint` and switching it re-indexes the vault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentTemplate {
    /// `<bos>search_document: {title} {text}` — nomic-embed-text's convention.
    /// The control the documented template was measured against.
    Legacy,
    /// `<bos>title: {title | none} | text: {text}` — the model card's.
    #[default]
    Documented,
}

/// What fills the document template's `title:` field (issues #36, #38).
///
/// The documented template has such a field, and the model card fills it with
/// the literal `none` when a document has no title. The vault knows what else
/// could go there, and design §5.4's breadcrumb is what #36 put there.
///
/// **The breadcrumb ships in the lexical lane and not in this field.** The
/// breadcrumb rule has three limbs that carry the same string. The keyword
/// index (#37) and the cross-encoder's input (#30) are both measured gains.
/// This limb is not: #38 ran it against `None` with the other two limbs on. No
/// tracked answer separates them and the six positive queries read as a draw
/// below the answers, so abstention decides it — the breadcrumb scores four of
/// the eleven negatives higher. It is the closest call in `eval/probes.md`.
///
/// Like [`DocumentTemplate`], the choice decides what a stored vector means, so
/// it is a component of `embedding_fingerprint` and switching it re-indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentTitle {
    /// The literal `none` — the model card's own value for an untitled
    /// document, and the default.
    ///
    /// It holds all seven tracked answers at the same rank and the same score
    /// as [`Self::Breadcrumb`], so the choice between the two rests entirely on
    /// the results below them, and there the two arms draw. `None` wins probe 6,
    /// the right repair spell where the breadcrumb puts a healing spell above
    /// the floor, and probe 7, the right species where the breadcrumb puts two
    /// wrong ones. The breadcrumb wins probe 3, where it returns four answering
    /// spells in the top seven against three, and probe 4, where it keeps three
    /// archdemon sections out of an archdragon query.
    ///
    /// Abstention is what decides it. `None` scores four of the eleven verified
    /// negatives lower — one of them at a quarter of the breadcrumb's score —
    /// and it is the only difference in the pool that is not a manual judgment
    /// about one passage.
    #[default]
    None,
    /// The note's effective title: frontmatter `name`, else the filename stem.
    ///
    /// **Do not use this.** It is a per-file constant, which is the mechanism
    /// issue #2 lost to, and it reproduces #2's failure exactly: the exact-name
    /// probe's answer leaves the top 20 (`eval/probes.md`).
    Note,
    /// `Note Title > H1 > H2 > H3` — the note's title and the chunk's ancestor
    /// headings. Design §5.4's breadcrumb.
    ///
    /// The one component of document identity that is **not** a per-file
    /// constant, which is what separates it from issue #2: the heading path
    /// differs between the sections of one document, so it cannot flatten them
    /// together the way #2's prefix did. That is why it holds the exact-name
    /// answer that [`Self::Note`] loses.
    ///
    /// It costs no tracked answer either, so #36 shipped it on the design's
    /// authority and left the reshuffle beneath the answers open. #38 closed it
    /// against this value, and narrowly — see [`Self::None`] and
    /// `eval/probes.md`. The setting stays because the reading is eighteen
    /// queries and six manual judgments, and #3's relevance labels are the
    /// evidence that would settle it properly.
    Breadcrumb,
}

impl DocumentTitle {
    /// The spelling `embedding_fingerprint` hashes.
    pub fn id(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Note => "note",
            Self::Breadcrumb => "breadcrumb",
        }
    }
}

/// Which query template the `EmbeddingGemma` prompt format writes (issue #10).
///
/// A query is embedded and discarded, so this is **not** a fingerprint
/// component: changing it needs no re-index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryTemplate {
    /// `<bos>search_query: {query}` — nomic-embed-text's convention.
    /// The control the documented template was measured against.
    Legacy,
    /// `<bos>task: search result | query: {query}`.
    #[default]
    Documented,
}

/// The task an EmbeddingGemma query prompt names.
///
/// The model card documents eight of them. `search result` is the one a
/// retrieval engine asks for: the others describe classification, clustering
/// and similarity, which knapper never wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedTask {
    /// No task field at all — the legacy `search_query:` prefix.
    Legacy,
    /// `task: search result` — retrieval.
    SearchResult,
}

impl EmbedTask {
    /// Resolve the task from the configured template.
    pub fn resolve(template: QueryTemplate) -> Self {
        match template {
            QueryTemplate::Legacy => Self::Legacy,
            QueryTemplate::Documented => Self::SearchResult,
        }
    }

    /// The task description the model card spells out.
    fn description(self) -> &'static str {
        match self {
            Self::Legacy | Self::SearchResult => "search result",
        }
    }
}

/// Which llama.cpp forward pass an embedding model's graph needs (issue #8).
///
/// The two are not interchangeable, and picking the wrong one is a segfault
/// rather than a bad number. `llama_context::encode` runs the graph with a
/// **null** memory context, because an encoder needs no KV cache; a model whose
/// graph opens with `build_attn_inp_kv()` dereferences that null. `decode`
/// supplies the cache.
///
/// llama.cpp's own `llama_model_has_encoder` answers a narrower question — it
/// is true for T5 alone — so it cannot be the predicate here: it would send
/// EmbeddingGemma down the decoder pass too, and that is the shipped default's
/// path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardPass {
    /// `llama_context::encode` — a non-causal graph that builds no cache.
    Encode,
    /// `llama_context::decode` — a causal graph that reads a KV cache.
    Decode,
}

/// The task a Qwen3-Embedding query names (issue #8).
///
/// The model card's own default, verbatim. Qwen documents the instruct as
/// something to write for your own task, and reports its retrieval numbers
/// against this string — so a vault-specific rewording is an unmeasured
/// deviation from the only phrasing the model was published with.
///
/// It is query-side. It reaches no fingerprint, so changing it re-indexes
/// nothing and a sweep costs one search each.
const QWEN_RETRIEVAL_TASK: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Model-family-specific prompt templates for embedding models.
#[derive(Debug, Clone)]
pub enum PromptFormat {
    /// Google embeddinggemma family: an asymmetric query/document pair, whose
    /// document half is selected by [`DocumentTemplate`].
    EmbeddingGemma { document: DocumentTemplate },
    /// Qwen embedding family: uses `Instruct:` / `Query:` format.
    QwenEmbedding,
    /// No special formatting — pass text as-is.
    Raw,
}

impl PromptFormat {
    /// Auto-detect prompt format from a GGUF filename.
    pub fn detect(filename: &str, document: DocumentTemplate) -> Self {
        let lower = filename.to_lowercase();
        if lower.contains("embeddinggemma") {
            Self::EmbeddingGemma { document }
        } else if lower.contains("qwen") && lower.contains("embed") {
            Self::QwenEmbedding
        } else {
            Self::Raw
        }
    }

    /// What `embedding_fingerprint` hashes to know which template wrote a
    /// vector (issue #31). The template choice is *data*, so it is hashed
    /// exactly rather than carried by
    /// [`crate::fingerprint::PROMPT_TEMPLATE_VERSION`], which covers edits to
    /// the template text itself.
    pub fn template_id(&self) -> String {
        match self {
            Self::EmbeddingGemma { document } => {
                format!("embeddinggemma/{}", document.id())
            }
            Self::QwenEmbedding => "qwen-embedding/text".to_string(),
            Self::Raw => "raw".to_string(),
        }
    }

    /// Which llama.cpp forward pass this family's graph needs (issue #8).
    ///
    /// - [`Self::EmbeddingGemma`] builds `build_attn_inp_no_cache()`
    ///   (llama.cpp `src/models/gemma-embedding.cpp`), so `encode` is right and
    ///   is what every stored vector was produced by.
    /// - [`Self::QwenEmbedding`] is decoder-only: `llm_build_qwen3` opens with
    ///   `build_attn_inp_kv()` (`src/models/qwen3.cpp`), and `encode` passes a
    ///   null memory context into it.
    /// - [`Self::Raw`] keeps the pass it has always had. An unknown model that
    ///   needs the other one is why this is a family decision and not a guess.
    fn forward_pass(&self) -> ForwardPass {
        match self {
            Self::EmbeddingGemma { .. } | Self::Raw => ForwardPass::Encode,
            Self::QwenEmbedding => ForwardPass::Decode,
        }
    }

    /// Which special tokens llama.cpp adds of its own when it tokenizes for
    /// this family (issue #8).
    ///
    /// The name of `str_to_token`'s argument is narrower than what it does:
    /// `AddBos` is llama.cpp's `add_special`, and that flag gates the model's
    /// declared *trailing EOS* as well as its leading BOS. Which one a family
    /// wants follows from how it pools.
    ///
    /// - [`Self::EmbeddingGemma`] pools the mean, and writes its own `<bos>`
    ///   into the template — `parse_special` turns the literal into the real
    ///   token. Asking llama.cpp for the GGUF's own would give it a second BOS
    ///   and change every stored vector, so this stays `Never`.
    /// - [`Self::QwenEmbedding`] pools the **last** token. Its GGUF declares
    ///   `add_bos_token = false, add_eos_token = true`, so asking for the
    ///   model's own tokens appends exactly the EOS that pooling reads and no
    ///   BOS. Without it llama.cpp pools the last token of the *content*.
    /// - [`Self::Raw`] is an unknown model with no template, so it gets no
    ///   tokens it did not ask for.
    fn add_special(&self) -> AddBos {
        match self {
            Self::EmbeddingGemma { .. } | Self::Raw => AddBos::Never,
            Self::QwenEmbedding => AddBos::Always,
        }
    }

    /// Format text for a search query.
    ///
    /// The documented strings carry no `<bos>`, and this one is written
    /// literally because `str_to_token` is called with `AddBos::Never` — see
    /// [`LlamaEmbed::embed_formatted`]. `parse_special` is on in llama-cpp-2, so
    /// the literal becomes the real BOS token. Dropping it here and switching
    /// that call to `AddBos::Always` would also add a BOS to `QwenEmbedding`
    /// and `Raw`, which currently have none.
    pub fn format_query(&self, query: &str, task: EmbedTask) -> String {
        match self {
            Self::EmbeddingGemma { .. } => match task {
                EmbedTask::Legacy => format!("<bos>search_query: {query}"),
                other => format!("<bos>task: {} | query: {query}", other.description()),
            },
            Self::QwenEmbedding => {
                format!("Instruct: {QWEN_RETRIEVAL_TASK}\nQuery:{query}")
            }
            Self::Raw => query.to_string(),
        }
    }

    /// Format text for a document to be indexed.
    ///
    /// An empty `title` is the documented literal `none` under
    /// [`DocumentTemplate::Documented`] — a supported input rather than a
    /// degenerate one. Issue #36 is whether the vault's own identity belongs in
    /// that field.
    pub fn format_document(&self, title: &str, text: &str) -> String {
        match self {
            Self::EmbeddingGemma {
                document: DocumentTemplate::Legacy,
            } => format!("<bos>search_document: {title} {text}"),
            Self::EmbeddingGemma {
                document: DocumentTemplate::Documented,
            } => {
                let title = if title.trim().is_empty() {
                    "none"
                } else {
                    title.trim()
                };
                format!("<bos>title: {title} | text: {text}")
            }
            // Qwen3-Embedding embeds a document as itself: the model card
            // gives the instruct to the *query* half alone, and there is no
            // `title:` field to put a breadcrumb in. Gluing one on invents a
            // format the card does not have, and at the shipped
            // `document_title = none` it wrote a leading blank line.
            Self::QwenEmbedding => text.to_string(),
            Self::Raw => format!("{title}\n{text}"),
        }
    }
}

impl DocumentTemplate {
    fn id(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Documented => "documented",
        }
    }
}

/// `[embedding_prompt]` — which EmbeddingGemma templates this instance writes
/// (issue #10), and what it puts in the document template's `title:` field
/// (issue #36). The templates default to the model card's own pair, and the
/// title field to the design's breadcrumb.
///
/// The keys have very different costs. `document` and `document_title` are
/// fingerprint components and changing either re-indexes the vault; `query` is
/// read per search and changing it costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EmbeddingPromptConfig {
    /// The template every stored vector is embedded through.
    pub document: DocumentTemplate,
    /// The template every query is embedded through.
    pub query: QueryTemplate,
    /// What fills the document template's `title:` field (issue #36). A
    /// fingerprint component, like `document`. `none` is the control, and it is
    /// what every store built before this key existed holds.
    pub document_title: DocumentTitle,
}

// ── Traits ───────────────────────────────────────────────────────────────────

/// One document as the embedder is shown it: the two fields the document half of
/// an asymmetric template has (issue #36).
///
/// The title is a *field*, not a prefix. It reaches
/// [`PromptFormat::format_document`] and nothing else — storage, snippets and
/// FTS keep the raw chunk — and what goes in it is
/// [`DocumentTitle`]'s decision, made once per chunk by
/// [`crate::prefix::embed_inputs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedDoc<'a> {
    pub title: &'a str,
    pub text: &'a str,
}

impl<'a> EmbedDoc<'a> {
    pub fn new(title: &'a str, text: &'a str) -> Self {
        Self { title, text }
    }

    /// A document with nothing for the title field, which the documented
    /// template spells as the literal `none`.
    pub fn untitled(text: &'a str) -> Self {
        Self { title: "", text }
    }
}

/// Embedding backend — converts text into dense float vectors.
pub trait EmbedModel: Send {
    /// Embed a batch of documents in one call.
    fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>>;

    /// Convenience wrapper for a single untitled text.
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[EmbedDoc::untitled(text)])?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embed_batch returned empty results"))
    }

    /// Embed a search query — the other half of an asymmetric model's pair.
    ///
    /// The default implementation ignores the asymmetry, which is correct for a
    /// symmetric model and for the test embedders.
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(text)
    }

    /// Approximate token count for `text` (used for chunk-size budgeting).
    fn token_count(&self, text: &str) -> usize;

    /// Dimensionality of vectors produced by this model.
    fn dim(&self) -> usize;

    /// The model's real-token input wall, read from the GGUF at load
    /// (`n_ctx_train`). A chunk longer than this is silently truncated by the
    /// embedder, so it is the ceiling `split_oversized_chunks` enforces
    /// (issue #75).
    fn max_context(&self) -> usize;

    /// Everything about this model that decides what a stored vector *means*:
    /// the artifact's bytes, its width, its tokenizer, the prompt template it is
    /// fed through, and what happens to the vector afterwards.
    ///
    /// Feeds `embedding_fingerprint` (issue #31), which **subsumes**
    /// `ensure_embedding_dim` — the width is one component here rather than the
    /// only thing checked anywhere. The dimension guard stays as it is: a width
    /// change is a shape error, not merely a stale index, and it fails harder.
    ///
    /// Computed once at load. A filename is not identity, so this hashes bytes,
    /// and hashing 640 MB is not something to do per call.
    fn fingerprint(&self) -> String;
}

// Blanket impl: `Box<dyn EmbedModel + Send>` itself implements `EmbedModel`.
// This lets `Arc<Mutex<Box<dyn EmbedModel + Send>>>` callers pass
// `&mut *guard` (which is `&mut Box<dyn EmbedModel + Send>`) to any
// function taking `&mut impl EmbedModel`.
//
// Every method with a default body (`embed_one`, `embed_query`) still needs
// an explicit forward here. Leaving one out does not fail to compile — it
// silently falls through to the trait's default, which calls back through
// `self` (a `Box`, not the inner type), so it never reaches an override the
// inner type wrote. That is exactly how `embed_query` went missing here
// before: every call through a `Box` ran the trait default (`embed_one`)
// instead of `ApiEmbedder`'s query-task-typed override, embedding every
// search query as a document.
impl EmbedModel for Box<dyn EmbedModel + Send> {
    fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(docs)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_one(text)
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text)
    }

    fn token_count(&self, text: &str) -> usize {
        (**self).token_count(text)
    }

    fn dim(&self) -> usize {
        (**self).dim()
    }

    fn max_context(&self) -> usize {
        (**self).max_context()
    }

    fn fingerprint(&self) -> String {
        (**self).fingerprint()
    }
}

/// Which embedder `config.models.embed` names: the local llama.cpp GGUF path
/// (`None`, or an `hf:` URI) or a hosted API provider (issue #84).
#[derive(Debug)]
pub enum EmbedScheme {
    Local,
    Gemini { model_id: String },
}

/// Parses `[models] embed` into a routable scheme. A `gemini:` id must be
/// pinned and versioned — ending in a digit — so `gemini:gemini-embedding`
/// and the moving alias `gemini:gemini-embedding-latest` are rejected rather
/// than silently re-pointing a store's vectors to a model that changed
/// underneath it.
pub fn parse_embed_scheme(uri: Option<&str>) -> Result<EmbedScheme> {
    match uri {
        None => Ok(EmbedScheme::Local),
        Some(u) if u.starts_with("hf:") => Ok(EmbedScheme::Local),
        Some(u) if u.starts_with("gemini:") => {
            let model_id = u.trim_start_matches("gemini:").to_string();
            let versioned = model_id.chars().last().is_some_and(|c| c.is_ascii_digit());
            anyhow::ensure!(
                versioned,
                "gemini model id must be pinned and versioned (ending in a version number), got: {model_id:?}"
            );
            Ok(EmbedScheme::Gemini { model_id })
        }
        Some(other) => anyhow::bail!("unknown embed model URI scheme: {other}"),
    }
}

/// Builds the embedder `config.models.embed` names: the local llama.cpp GGUF
/// or a hosted API provider, boxed to one trait object either way.
pub fn load_embedder(
    models_dir: &Path,
    config: &crate::config::Config,
) -> Result<Box<dyn EmbedModel + Send>> {
    match parse_embed_scheme(config.models.embed.as_deref())? {
        EmbedScheme::Local => {
            let e = LlamaEmbed::new(models_dir, config)?;
            Ok(Box::new(e) as Box<dyn EmbedModel + Send>)
        }
        EmbedScheme::Gemini { model_id } => {
            let api = &config.models.embed_api;
            let provider = crate::embed_api::Gemini {
                model_id,
                endpoint_override: api.endpoint.clone(),
            };
            let e = crate::embed_api::ApiEmbedder::new(
                provider,
                api.dim,
                api.timeout_secs,
                api.max_retries,
            )?;
            Ok(Box::new(e) as Box<dyn EmbedModel + Send>)
        }
    }
}

/// Cross-encoder reranker — scores a (query, document) pair.
pub trait RerankModel: Send {
    /// Return a relevance score in [0.0, 1.0].
    fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32>;

    /// Everything about this reranker that decides what a score *means*: the
    /// artifact's bytes and its Yes/No token identity.
    ///
    /// Feeds `reranker_fingerprint` (issue #31), the one key whose action is
    /// not a rebuild — nothing the reranker touches is stored. What it
    /// invalidates is any threshold calibrated against its scores, which is why
    /// it is recorded before there is a threshold to protect.
    fn fingerprint(&self) -> String;

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

    /// Count the tokens in `text` for the budget in `packaging` (#35).
    ///
    /// The default is the documented `chars / 3.33` estimate, which a model
    /// with no tokenizer of its own falls back to. `LlamaRerank` overrides it
    /// with its own vocabulary, so the count is the model's own.
    fn count_tokens(&self, text: &str) -> usize {
        (text.chars().count() * 100).div_ceil(333)
    }
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

    /// Produce a deterministic vector for a document, title field included.
    ///
    /// A document with no title hashes exactly its text, so the mock keeps
    /// writing the vectors it wrote before the title field existed and only a
    /// test that sets one sees any change.
    fn hash_doc(&self, doc: EmbedDoc<'_>) -> Vec<f32> {
        match doc.title.is_empty() {
            true => self.hash_to_vector(doc.text),
            false => self.hash_to_vector(&format!("title: {} | text: {}", doc.title, doc.text)),
        }
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
    fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
        Ok(docs.iter().map(|d| self.hash_doc(*d)).collect())
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

    fn max_context(&self) -> usize {
        2048
    }

    fn fingerprint(&self) -> String {
        // Width is the only thing that varies about the mock, and `hash_to_vector`
        // is its whole algorithm. Bump the version if that changes, or a store
        // built by an old test binary reads as current.
        format!("mock-embed-v1:{}", self.dim)
    }
}

impl RerankModel for MockLlm {
    fn fingerprint(&self) -> String {
        "mock-rerank-v1".to_string()
    }

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
///
/// Returns the tokenizer and its **identity** — a digest of the artifact it came
/// from, which `embedding_fingerprint` folds in (issue #31). A tokenizer swap
/// changes what every stored vector was built from and nothing else in the
/// database notices, so the identity is a digest of bytes rather than a repo
/// name. The GGUF-embedded case needs no digest of its own: it is part of the
/// model file, which is hashed separately.
fn load_tokenizer_for_model(
    uri: &HfModelUri,
    models_dir: &Path,
) -> Result<(FlexTokenizer, String)> {
    // First try: external tokenizer.json from candidate repos.
    if let Some((tok, tok_path)) = try_external_tokenizer(uri, models_dir) {
        let identity = crate::fingerprint::artifact_digest(&tok_path)
            .unwrap_or_else(|_| format!("path:{}", tok_path.display()));
        return Ok((
            FlexTokenizer::HuggingFace(Box::new(tok)),
            format!("hf:{identity}"),
        ));
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
        return Ok((
            FlexTokenizer::Gguf(Box::new(tok)),
            "gguf-embedded".to_string(),
        ));
    }

    bail!(
        "could not find tokenizer for model '{}': no external tokenizer.json \
         and GGUF file not yet downloaded",
        uri.repo
    )
}

/// Try downloading tokenizer.json from candidate HuggingFace repos.
///
/// Returns the tokenizer with the path it was read from, so the caller can
/// digest the artifact rather than trust the repo name it was fetched under.
fn try_external_tokenizer(
    uri: &HfModelUri,
    models_dir: &Path,
) -> Option<(tokenizers::Tokenizer, PathBuf)> {
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
            return Some((tok, tok_path));
        }

        if let Ok(p) = ensure_model(&tok_uri, models_dir)
            && let Ok(tok) = tokenizers::Tokenizer::from_file(&p)
        {
            return Some((tok, p));
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
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            embed_uri: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".into(),
            rerank_uri: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
                .into(),
        }
    }
}

/// One embedder `knapper models` offers (issue #8).
///
/// The catalogue is a menu, not a policy: every row is something
/// `models.embed` accepts, the first is the shipped default, and choosing
/// another is the user's call. Nothing here decides what loads — the width,
/// the context and the prompt format all come from the GGUF and its filename
/// at load time (issue #12), and the numbers below are what that model reports
/// so a user can choose before downloading 4 GB.
pub struct KnownEmbedder {
    /// What goes in `models.embed`.
    pub uri: &'static str,
    /// The model's native output width, as its GGUF declares it.
    pub dim: usize,
    /// The model's training context length, in tokens.
    pub context: usize,
    /// The GGUF's size on disk, for the first-use download.
    pub download: &'static str,
    /// One line on what taking this row costs.
    pub note: &'static str,
}

/// The embedders `models list` offers.
///
/// Taking a row that is not the first has two costs, and both fire at runtime
/// as well: the store re-indexes at the new width, because
/// `embedding_fingerprint` and the vec table both move; and `[calibrated]`'s
/// four numbers are EmbeddingGemma's fit, so the model-free ranking path runs
/// a floor read off a scale that is gone until you refit or configure a
/// cross-encoder.
pub fn known_embedders() -> &'static [KnownEmbedder] {
    &[
        KnownEmbedder {
            uri: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
            dim: 768,
            context: 2048,
            download: "334 MB",
            note: "Default. The embedder [calibrated] is fit against.",
        },
        KnownEmbedder {
            uri: "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            dim: 1024,
            context: 32768,
            download: "639 MB",
            note: "Runs on CPU. Re-indexes at 1024; refit [calibrated].",
        },
        KnownEmbedder {
            uri: "hf:Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
            dim: 2560,
            context: 40960,
            download: "4.28 GB",
            note: "Wants a GPU. Re-indexes at 2560; refit [calibrated].",
        },
    ]
}

/// The embed model name `status` reports: the configured `models.embed`, or
/// the shipped default when none is set. An `hf:` URI or a bare path reduces
/// to the model file's stem, because the rest of the URI names a download
/// location rather than a model; a `gemini:` id keeps its scheme, which is
/// what says the model is not local.
pub fn embed_model_display(config: &crate::config::Config) -> String {
    let defaults = ModelDefaults::default();
    let uri = config
        .models
        .embed
        .as_deref()
        .unwrap_or(&defaults.embed_uri);
    if uri.starts_with("gemini:") {
        return uri.to_string();
    }
    let file = uri.rsplit('/').next().unwrap_or(uri);
    file.strip_suffix(".gguf").unwrap_or(file).to_string()
}

/// llama.cpp's own thread default, used only when the machine will say nothing
/// at all about how many cores it has.
const FALLBACK_N_THREADS: i32 = 4;

/// Physical cores, or `None` if the platform will not say.
///
/// Counts distinct `thread_siblings` masks in sysfs — one entry per physical
/// core, with an SMT pair sharing a mask. This is llama.cpp's own rule, from
/// `common/common.cpp::cpu_get_num_physical_cores`, which is what its CLI feeds
/// to `n_threads`; only the *library* default is the flat 4 that knapper used to
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
/// passes, and knapper is all of the latter: nothing generates. The reranker
/// decodes a whole ~155-token pair and reads the final logits, and the embedder
/// encodes a chunk.
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
    max_context: usize,
    prompt_format: PromptFormat,
    /// Which query template to write (issue #10). Held apart from
    /// `prompt_format` because the document half is a fingerprint component and
    /// this half is not: a query is embedded and discarded.
    query_template: QueryTemplate,
    /// Resolved once at load — see [`resolve_n_threads`].
    n_threads: i32,
    /// Computed once at load, because it hashes hundreds of megabytes of GGUF.
    /// See [`EmbedModel::fingerprint`].
    fingerprint: String,
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
        let (tokenizer, tokenizer_identity) = load_tokenizer_for_model(&uri, models_dir)?;

        // Detect prompt format from filename; the document half of the
        // EmbeddingGemma pair is a configured choice (issue #10).
        let prompt_format = PromptFormat::detect(&uri.filename, config.embedding_prompt.document);

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

        let max_context = usize::try_from(model.n_ctx_train())
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                tracing::warn!(
                    "model {uri_str} reports no training context length; using {FALLBACK_MAX_CONTEXT}"
                );
                FALLBACK_MAX_CONTEXT
            });

        let n_threads = resolve_n_threads(config);
        let device = device_identity();
        tracing::info!(
            "loaded LlamaEmbed from {}, dim={}, max_context={}, n_threads={}, device={}",
            uri_str,
            dim,
            max_context,
            n_threads,
            device
        );

        let fingerprint = embed_fingerprint(
            &crate::fingerprint::artifact_digest(&model_path)?,
            dim,
            &tokenizer_identity,
            &prompt_format,
            &device,
        );

        Ok(Self {
            model,
            tokenizer,
            dim,
            max_context,
            prompt_format,
            query_template: config.embedding_prompt.query,
            n_threads,
            fingerprint,
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
        //
        // The flag is llama.cpp's `add_special`, and which tokens a family
        // wants is [`PromptFormat::add_special`]'s decision (#8): EmbeddingGemma
        // writes its own `<bos>` into the template, and Qwen3-Embedding needs
        // the GGUF's trailing EOS because it pools the last token.
        let add_special = self.prompt_format.add_special();
        let tokenized = texts
            .iter()
            .map(|text| {
                let tokens = self
                    .model
                    .str_to_token(text, add_special)
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

            // Which pass runs is the model family's, not a constant (#8).
            // `encode` hands the graph a null memory context, which a
            // decoder-only embedder dereferences — see [`ForwardPass`].
            match self.prompt_format.forward_pass() {
                ForwardPass::Encode => ctx
                    .encode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("embedding encode failed: {e}"))?,
                ForwardPass::Decode => ctx
                    .decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("embedding decode failed: {e}"))?,
            }

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
    fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
        // Apply document prompt format for indexing (asymmetric models need this),
        // then hand the whole batch to one context (issue #13).
        let formatted: Vec<String> = docs
            .iter()
            .map(|d| self.prompt_format.format_document(d.title, d.text))
            .collect();
        self.embed_formatted(&formatted)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_query(text)
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        // Apply query prompt format (asymmetric models like embeddinggemma need this).
        let task = EmbedTask::resolve(self.query_template);
        let formatted = self.prompt_format.format_query(text, task);
        self.embed_text(&formatted)
    }

    fn token_count(&self, text: &str) -> usize {
        self.tokenizer.token_count(text)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

// ── LlamaRerank — GGUF cross-encoder reranker via llama.cpp ─────────────────────

/// Format query+document for cross-encoder reranking.
pub fn format_reranker_input(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the \
         Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\
         <|im_end|>\n\
         <|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages \
         that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// The cross-encoder families knapper can score.
///
/// Only Qwen3-Reranker's generative yes/no judge is implemented (see
/// [`format_reranker_input`] and [`LlamaRerank`]). A GGUF of any other family —
/// a BGE, Jina or mxbai sequence-classification head — is a different llama.cpp
/// execution mode, so [`LlamaRerank::new`] refuses it at load rather than run it
/// through this template and return a meaningless score (#82). The enum is the
/// seam a second family extends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerFamily {
    /// Qwen3-Reranker: a causal LM read by the softmax of its yes/no logits.
    Qwen3,
}

impl RerankerFamily {
    /// Detect the family from the model URI, or `None` for one the scorer does
    /// not handle.
    ///
    /// Keys on a loose substring of the repo and filename, because a filename is
    /// not identity here — the reason [`crate::fingerprint::artifact_digest`]
    /// hashes bytes. Allowlisting the one family that scores correctly fails
    /// safe: a valid Qwen model with an odd name is refused and seen, where a
    /// non-Qwen family run anyway is a wrong score no one sees.
    pub fn detect(uri: &HfModelUri) -> Option<Self> {
        let hay = format!("{} {}", uri.repo, uri.filename).to_lowercase();
        if hay.contains("qwen3-reranker") {
            Some(Self::Qwen3)
        } else {
            None
        }
    }
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
    /// Computed once at load. See [`RerankModel::fingerprint`].
    fingerprint: String,
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
        if RerankerFamily::detect(&uri).is_none() {
            bail!(
                "unsupported reranker model '{uri_str}': knapper scores only \
                 Qwen3-Reranker GGUFs. Set [models] rerank to a Qwen3-Reranker model."
            );
        }
        let model_path = ensure_model(&uri, models_dir)?;

        // Use global backend and llama.cpp's built-in tokenizer (no tokenizer.json required).
        let backend = llama_backend()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
            .map_err(|e| anyhow::anyhow!("loading reranker model {}: {e}", model_path.display()))?;

        // Look up Yes/No token IDs via the model's built-in tokenizer.
        // str_to_token returns Vec<LlamaToken>; we take the first token ID (skip BOS).
        let yes_tokens = model
            .str_to_token("yes", AddBos::Never)
            .map_err(|e| anyhow::anyhow!("tokenizing 'Yes': {e}"))?;
        let yes_token_id = yes_tokens
            .first()
            .map(|t| t.0)
            .ok_or_else(|| anyhow::anyhow!("model tokenizer returned no tokens for 'Yes'"))?;

        let no_tokens = model
            .str_to_token("no", AddBos::Never)
            .map_err(|e| anyhow::anyhow!("tokenizing 'No': {e}"))?;
        let no_token_id = no_tokens
            .first()
            .map(|t| t.0)
            .ok_or_else(|| anyhow::anyhow!("model tokenizer returned no tokens for 'No'"))?;

        let n_threads = resolve_n_threads(config);
        let device = device_identity();
        tracing::info!(
            "loaded LlamaRerank from {}, yes_id={}, no_id={}, n_threads={}, device={}",
            uri_str,
            yes_token_id,
            no_token_id,
            n_threads,
            device
        );

        // The device is a component because the logits those ids index shift
        // with the kernels that produced them, and this fingerprint's declared
        // action — discarding calibrated thresholds — is exactly the right
        // response to a score scale that moved underneath them (issue #33).
        let fingerprint = rerank_fingerprint(
            &crate::fingerprint::artifact_digest(&model_path)?,
            yes_token_id,
            no_token_id,
            &device,
        );

        Ok(Self {
            model,
            yes_token_id,
            no_token_id,
            n_threads,
            fingerprint,
        })
    }
}

impl RerankModel for LlamaRerank {
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }

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

    fn count_tokens(&self, text: &str) -> usize {
        // The reranker's own vocabulary, not an estimate — AddBos::Never for
        // the same reason the yes/no ids are read that way.
        self.model
            .str_to_token(text, llama_cpp_2::model::AddBos::Never)
            .map(|t| t.len())
            .unwrap_or_else(|_| (text.chars().count() * 100).div_ceil(333))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Serializes tests that touch the process-global GEMINI_API_KEY env var,
    // since `cargo test --lib` runs tests in parallel by default. Mirrors the
    // ENV_LOCK pattern in `embed_api.rs`'s tests.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        let vecs = mock
            .embed_batch(&[
                EmbedDoc::untitled("a"),
                EmbedDoc::untitled("b"),
                EmbedDoc::untitled("c"),
            ])
            .unwrap();
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 256));
    }

    /// The mock has to see the title field, or no test of what fills it (#36)
    /// could tell a change from a no-op. An untitled document keeps the vector
    /// the mock wrote before the field existed.
    #[test]
    fn the_mock_hashes_the_title_field_and_an_untitled_document_is_unchanged() {
        let mut mock = MockLlm::new(256);

        let untitled = mock.embed_batch(&[EmbedDoc::untitled("body")]).unwrap();
        assert_eq!(untitled[0], mock.hash_to_vector("body"));

        let titled = mock
            .embed_batch(&[EmbedDoc::new("Archdragon > Definition", "body")])
            .unwrap();
        assert_ne!(titled[0], untitled[0]);
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
    fn mock_reports_a_max_context() {
        let m = MockLlm::new(8);
        assert_eq!(m.max_context(), 2048);
    }

    #[test]
    fn test_mock_rerank() {
        let mut mock = MockLlm::new(256);
        let score = mock.rerank_score("query", "document text").unwrap();
        assert!((0.0..=1.0).contains(&score));
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
    fn mock_rerank_counts_tokens_by_the_documented_ratio() {
        let mock = MockLlm::new(256);
        // 40 characters -> ceil(40 / 3.33) = 13 tokens on the fallback path.
        let text = "a".repeat(40);
        assert_eq!(mock.count_tokens(&text), 13);
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

    fn gemma(document: DocumentTemplate) -> PromptFormat {
        PromptFormat::detect("embeddinggemma-300M-Q8_0.gguf", document)
    }

    #[test]
    fn test_prompt_format_embeddinggemma_query() {
        let formatted =
            gemma(DocumentTemplate::Legacy).format_query("how does auth work", EmbedTask::Legacy);
        assert!(formatted.contains("search_query"));
        assert!(formatted.contains("how does auth work"));
    }

    #[test]
    fn test_prompt_format_embeddinggemma_document() {
        let formatted =
            gemma(DocumentTemplate::Legacy).format_document("Note Title", "some content");
        assert!(formatted.contains("Note Title"));
        assert!(formatted.contains("some content"));
        assert!(formatted.contains("search_document"));
    }

    #[test]
    fn test_prompt_format_unknown_model() {
        let fmt = PromptFormat::detect("unknown-model.gguf", DocumentTemplate::Documented);
        let formatted = fmt.format_query("test query", EmbedTask::SearchResult);
        assert_eq!(formatted, "test query");
    }

    #[test]
    fn test_prompt_format_raw_document() {
        let fmt = PromptFormat::detect("random-model.gguf", DocumentTemplate::Documented);
        let formatted = fmt.format_document("Title", "Body");
        assert_eq!(formatted, "Title\nBody");
    }

    // ── The Qwen3 embedding family (#8) ────────────────────────────────────
    //
    // The family is an *option*, not a default: `models.embed` points at it
    // and everything else — the width, the fingerprint, the re-index — falls
    // out of the model's own GGUF. What does not fall out is the special-token
    // policy. `str_to_token`'s `AddBos` is llama.cpp's `add_special`, which
    // gates the trailing EOS as well as the leading BOS, and Qwen3-Embedding
    // pools the **last** token: with no EOS appended, llama.cpp pools the last
    // token of the content instead. Its GGUF declares
    // `add_bos_token = false, add_eos_token = true`, so asking llama.cpp for
    // the model's own special tokens adds exactly that EOS and no BOS.

    fn qwen3(file: &str) -> PromptFormat {
        PromptFormat::detect(file, DocumentTemplate::Documented)
    }

    #[test]
    fn the_shipped_qwen3_embedding_files_detect_as_the_qwen_family() {
        for file in [
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
            "Qwen3-Embedding-4B-Q8_0.gguf",
            "Qwen3-Embedding-0.6B-f16.gguf",
        ] {
            assert!(
                matches!(qwen3(file), PromptFormat::QwenEmbedding),
                "{file} should detect as the Qwen embedding family"
            );
        }
    }

    #[test]
    fn the_qwen_reranker_is_not_an_embedding_model() {
        assert!(
            matches!(qwen3("qwen3-reranker-0.6b-q8_0.gguf"), PromptFormat::Raw),
            "the cross-encoder shares the vendor and not the format"
        );
    }

    #[test]
    fn qwen_asks_llama_cpp_for_the_models_own_special_tokens() {
        assert_eq!(
            qwen3("Qwen3-Embedding-0.6B-Q8_0.gguf").add_special(),
            AddBos::Always,
            "last-token pooling needs the GGUF's trailing EOS"
        );
    }

    #[test]
    fn qwen_runs_the_decoder_pass_because_its_graph_needs_a_kv_cache() {
        assert_eq!(
            qwen3("Qwen3-Embedding-0.6B-Q8_0.gguf").forward_pass(),
            ForwardPass::Decode,
            "encode() hands the graph a null memory context and qwen3 dereferences it"
        );
    }

    #[test]
    fn embeddinggemma_runs_the_encoder_pass_it_has_always_run() {
        assert_eq!(
            gemma(DocumentTemplate::Documented).forward_pass(),
            ForwardPass::Encode,
            "its graph builds no cache, and this is the shipped default's path"
        );
    }

    #[test]
    fn an_unrecognised_model_keeps_the_encoder_pass() {
        assert_eq!(
            qwen3("random-model.gguf").forward_pass(),
            ForwardPass::Encode
        );
    }

    #[test]
    fn embeddinggemma_writes_its_own_bos_and_asks_for_no_others() {
        assert_eq!(
            gemma(DocumentTemplate::Documented).add_special(),
            AddBos::Never,
            "the template writes <bos> literally; a second one would change every stored vector"
        );
    }

    #[test]
    fn an_unrecognised_model_asks_for_no_special_tokens() {
        assert_eq!(qwen3("random-model.gguf").add_special(), AddBos::Never);
    }

    #[test]
    fn the_qwen_query_template_is_the_model_cards() {
        assert_eq!(
            qwen3("Qwen3-Embedding-0.6B-Q8_0.gguf")
                .format_query("who guards the gate", EmbedTask::SearchResult),
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:who guards the gate"
        );
    }

    #[test]
    fn the_qwen_query_template_ignores_the_gemma_task_setting() {
        let fmt = qwen3("Qwen3-Embedding-0.6B-Q8_0.gguf");
        assert_eq!(
            fmt.format_query("who guards the gate", EmbedTask::Legacy),
            fmt.format_query("who guards the gate", EmbedTask::SearchResult),
            "[embedding_prompt] is EmbeddingGemma's and reaches no other family"
        );
    }

    #[test]
    fn a_qwen_document_is_its_text_and_nothing_else() {
        let fmt = qwen3("Qwen3-Embedding-0.6B-Q8_0.gguf");
        assert_eq!(fmt.format_document("", "Body text"), "Body text");
        assert_eq!(
            fmt.format_document("Note Title > H1", "Body text"),
            "Body text",
            "the title: field is a slot in Gemma's template and Qwen has none"
        );
    }

    // ── The embedder catalogue (#8) ────────────────────────────────────────

    #[test]
    fn the_catalogues_first_row_is_the_shipped_default() {
        assert_eq!(
            known_embedders()[0].uri,
            ModelDefaults::default().embed_uri,
            "one table names the default, or `models list` drifts from what loads"
        );
    }

    #[test]
    fn every_catalogued_embedder_is_a_uri_the_loader_can_parse() {
        for e in known_embedders() {
            HfModelUri::parse(e.uri).unwrap_or_else(|err| panic!("{}: {err}", e.uri));
        }
    }

    #[test]
    fn every_catalogued_embedder_detects_a_prompt_format_of_its_own() {
        for e in known_embedders() {
            let file = HfModelUri::parse(e.uri).unwrap().filename;
            assert!(
                !matches!(
                    PromptFormat::detect(&file, DocumentTemplate::Documented),
                    PromptFormat::Raw
                ),
                "{file} falls through to the raw format, which embeds it unprompted"
            );
        }
    }

    #[test]
    fn the_catalogue_offers_both_qwen3_embedders() {
        let dims: Vec<usize> = known_embedders()
            .iter()
            .filter(|e| e.uri.contains("Qwen3-Embedding"))
            .map(|e| e.dim)
            .collect();
        assert_eq!(
            dims,
            vec![1024, 2560],
            "the 0.6B and the 4B, at their native widths"
        );
    }

    #[test]
    fn the_qwen_template_id_is_not_the_one_the_title_glue_wrote() {
        let id = qwen3("Qwen3-Embedding-0.6B-Q8_0.gguf").template_id();
        assert_ne!(
            id, "qwen-embedding",
            "a store built under the old document template must re-index"
        );
        assert!(id.starts_with("qwen-embedding/"), "got {id}");
    }

    // ── The documented EmbeddingGemma templates (#10) ──────────────────────
    //
    // These assert the model card's strings exactly. A test that only checks
    // for a substring is what let nomic-embed-text's convention live under
    // this variant's name.

    #[test]
    fn documented_query_template_is_the_model_cards() {
        assert_eq!(
            gemma(DocumentTemplate::Documented)
                .format_query("who guards the gate", EmbedTask::SearchResult),
            "<bos>task: search result | query: who guards the gate"
        );
    }

    #[test]
    fn documented_document_template_is_the_model_cards() {
        assert_eq!(
            gemma(DocumentTemplate::Documented).format_document("Archdragon", "It flies."),
            "<bos>title: Archdragon | text: It flies."
        );
    }

    #[test]
    fn an_untitled_document_is_the_literal_none() {
        // The model card spells the empty case. The legacy template has no
        // spelling for it and emits a double space instead.
        assert_eq!(
            gemma(DocumentTemplate::Documented).format_document("", "It flies."),
            "<bos>title: none | text: It flies."
        );
        assert_eq!(
            gemma(DocumentTemplate::Documented).format_document("   ", "It flies."),
            "<bos>title: none | text: It flies."
        );
    }

    #[test]
    fn every_embeddinggemma_template_keeps_its_bos() {
        // `str_to_token` is called with `AddBos::Never`, so the literal here is
        // the only BOS the model gets. The documented strings do not carry one.
        for document in [DocumentTemplate::Legacy, DocumentTemplate::Documented] {
            let fmt = gemma(document);
            assert!(fmt.format_document("t", "x").starts_with("<bos>"));
            for task in [EmbedTask::Legacy, EmbedTask::SearchResult] {
                assert!(fmt.format_query("q", task).starts_with("<bos>"));
            }
        }
    }

    #[test]
    fn each_query_template_names_its_own_task() {
        assert_eq!(
            EmbedTask::resolve(QueryTemplate::Documented),
            EmbedTask::SearchResult
        );
        assert_eq!(EmbedTask::resolve(QueryTemplate::Legacy), EmbedTask::Legacy);
    }

    #[test]
    fn the_document_template_is_a_fingerprint_component_and_the_query_one_is_not() {
        // Which template wrote a vector decides what it means, so a store built
        // one way must not be read the other. A query is embedded and
        // discarded, so `QueryTemplate` reaches no fingerprint at all — it is
        // not an input to `embed_fingerprint` and cannot be.
        let legacy = embed_fingerprint(
            "artifact",
            768,
            "tok",
            &gemma(DocumentTemplate::Legacy),
            "cpu",
        );
        let documented = embed_fingerprint(
            "artifact",
            768,
            "tok",
            &gemma(DocumentTemplate::Documented),
            "cpu",
        );
        assert_ne!(legacy, documented);
    }

    // ── LlamaRerank tests ──────────────────────────────────────────────────

    /// The four pieces Qwen3-Reranker's model card specifies, each of which we
    /// got wrong until the ranking stage made the score matter (#30).
    ///
    /// The empty `<think></think>` block is the load-bearing one: without it
    /// the next token the model produces is the start of its reasoning, so the
    /// yes/no logits being read are from a distribution that was never about
    /// yes or no.
    #[test]
    fn the_reranker_is_asked_the_question_its_model_card_documents() {
        let formatted = format_reranker_input("auth system", "The auth module handles OAuth");

        assert!(formatted.contains("<Query>: auth system"));
        assert!(formatted.contains("<Document>: The auth module handles OAuth"));
        assert!(formatted.contains("<Instruct>: "));
        assert!(
            formatted.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "the answer is read from the token after this: {formatted:?}"
        );
        assert!(
            formatted.contains("can only be \"yes\" or \"no\""),
            "the scored tokens are lowercase and the system prompt has to say so"
        );
    }

    #[test]
    fn the_reranker_family_recognizes_qwen3_and_no_other() {
        let qwen =
            HfModelUri::parse("hf:Qwen/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B-Q8_0.gguf")
                .unwrap();
        assert_eq!(RerankerFamily::detect(&qwen), Some(RerankerFamily::Qwen3));

        for uri in [
            "hf:BAAI/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf",
            "hf:jinaai/jina-reranker-v2-base-multilingual-GGUF/model-Q8_0.gguf",
            "hf:mixedbread-ai/mxbai-rerank-large-v1-GGUF/model.gguf",
        ] {
            let parsed = HfModelUri::parse(uri).unwrap();
            assert_eq!(
                RerankerFamily::detect(&parsed),
                None,
                "a non-Qwen family must not be recognized: {uri}"
            );
        }
    }

    #[test]
    fn new_refuses_an_unsupported_reranker_before_it_downloads() {
        let mut config = crate::config::Config::default();
        config.models.rerank =
            Some("hf:BAAI/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf".to_string());
        // The path does not exist: the guard must return before `ensure_model`
        // reads it, so an unsupported model never spends a download.
        let err = LlamaRerank::new(Path::new("/knapper-nonexistent-models"), &config)
            .expect_err("a non-Qwen reranker must be refused at load");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported reranker model"),
            "the error must name the mismatch, got: {msg}"
        );
    }

    #[test]
    fn test_llama_rerank_trait_compliance() {
        // Verify MockLlm still satisfies RerankModel.
        fn assert_rerank<R: RerankModel>(_r: &R) {}
        let mock = MockLlm::new(256);
        assert_rerank(&mock);
    }

    /// The same weights on two devices are two fingerprints (issue #33).
    ///
    /// CUDA and CPU kernels are not bitwise identical, so a store built on one
    /// and extended on the other holds mixed-provenance vectors. Without the
    /// device as a component that store reports itself healthy, which is the
    /// silent staleness #31 exists to end. Needs no GPU: the property under
    /// test belongs to the composition, not to the hardware.
    #[test]
    fn the_device_changes_the_embedding_fingerprint() {
        let on_cpu = embed_fingerprint(
            "artifact",
            768,
            "tok",
            &gemma(DocumentTemplate::Legacy),
            "cpu",
        );
        let on_gpu = embed_fingerprint(
            "artifact",
            768,
            "tok",
            &gemma(DocumentTemplate::Legacy),
            "CUDA/NVIDIA GeForce RTX 4070 Ti",
        );
        assert_ne!(
            on_cpu, on_gpu,
            "same model, two devices — the fingerprints must differ or the reindex never fires"
        );

        // And it is only the device that moved: the same one agrees with itself.
        assert_eq!(
            on_cpu,
            embed_fingerprint(
                "artifact",
                768,
                "tok",
                &gemma(DocumentTemplate::Legacy),
                "cpu"
            ),
            "the composition must be stable for a fixed device"
        );
    }

    /// Same property for the reranker, whose declared action is discarding
    /// calibrated thresholds rather than a reindex — the right response to a
    /// score scale that moved underneath them.
    #[test]
    fn the_device_changes_the_reranker_fingerprint() {
        let on_cpu = rerank_fingerprint("artifact", 9693, 2152, "cpu");
        let on_gpu = rerank_fingerprint("artifact", 9693, 2152, "CUDA/NVIDIA GeForce RTX 4070 Ti");
        assert_ne!(on_cpu, on_gpu);
        assert_eq!(on_cpu, rerank_fingerprint("artifact", 9693, 2152, "cpu"));
    }

    /// A box with no accelerator resolves to `cpu`, and the answer does not
    /// depend on how much VRAM happened to be free — `memory_free` is excluded
    /// precisely so that two runs of one binary on one device agree.
    #[test]
    fn the_device_identity_is_stable_across_calls() {
        let first = device_identity();
        let second = device_identity();
        assert_eq!(
            first, second,
            "device identity must not drift between calls"
        );
        assert!(
            !first.is_empty(),
            "there is always a device, even if it is cpu"
        );
    }

    #[test]
    fn scheme_routes_local_and_gemini() {
        assert!(matches!(
            parse_embed_scheme(None).unwrap(),
            EmbedScheme::Local
        ));
        assert!(matches!(
            parse_embed_scheme(Some("hf:org/repo/x.gguf")).unwrap(),
            EmbedScheme::Local
        ));
        match parse_embed_scheme(Some("gemini:gemini-embedding-2")).unwrap() {
            EmbedScheme::Gemini { model_id } => assert_eq!(model_id, "gemini-embedding-2"),
            _ => panic!("expected Gemini"),
        }
    }

    #[test]
    fn scheme_rejects_moving_alias() {
        for bad in [
            "gemini:gemini-embedding-latest",
            "gemini:gemini-embedding",
            "gemini:",
        ] {
            let err = parse_embed_scheme(Some(bad)).unwrap_err();
            assert!(
                err.to_string().contains("versioned"),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn load_embedder_builds_gemini_from_config() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: ENV_LOCK serializes every test that touches process env vars.
        unsafe { std::env::set_var("GEMINI_API_KEY", "k") };
        let mut cfg = crate::config::Config::default();
        cfg.models.embed = Some("gemini:gemini-embedding-2".into());
        cfg.models.embed_api.dim = Some(1536);
        let e = load_embedder(std::path::Path::new("/nonexistent"), &cfg).unwrap();
        assert_eq!(e.dim(), 1536);
        assert!(e.fingerprint().starts_with("gemini/gemini-embedding-2"));
    }

    /// Overrides `embed_query` but not `embed_one` — the same shape as
    /// `ApiEmbedder`, which has a query-task-typed `embed_query` and relies on
    /// the trait's default `embed_one` (-> `embed_batch`). The two paths
    /// return distinguishable vectors so a call through a `Box<dyn EmbedModel
    /// + Send>` proves which one it actually took.
    struct SeamProbe;

    impl EmbedModel for SeamProbe {
        fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
            Ok(docs.iter().map(|_| vec![0.0]).collect())
        }

        fn embed_query(&mut self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0])
        }

        fn token_count(&self, _text: &str) -> usize {
            0
        }

        fn dim(&self) -> usize {
            1
        }

        fn max_context(&self) -> usize {
            0
        }

        fn fingerprint(&self) -> String {
            "seam-probe".to_string()
        }
    }

    /// Regression for the missing `embed_query` forward in the blanket `impl
    /// EmbedModel for Box<dyn EmbedModel + Send>`. Without that forward, a
    /// `Box`'s `embed_query` falls through to the trait's default, which
    /// calls back through the `Box` into `embed_one` — the document path —
    /// instead of the inner type's `embed_query` override. Every production
    /// query call (CLI, HTTP, MCP) goes through exactly this kind of `Box`.
    #[test]
    fn the_box_forwards_embed_query_not_the_document_default() {
        let mut boxed: Box<dyn EmbedModel + Send> = Box::new(SeamProbe);
        assert_eq!(
            boxed.embed_query("q").unwrap(),
            vec![1.0],
            "embed_query through the box must reach the inner override, not embed_one/embed_batch"
        );
    }

    /// `status` reports the embed model that is actually configured, not a
    /// hardcoded name (launch-day bug: it printed all-MiniLM-L6-v2 always).
    #[test]
    fn the_status_model_name_is_the_configured_model_not_a_constant() {
        let cfg = crate::config::Config::default();
        assert_eq!(embed_model_display(&cfg), "embeddinggemma-300M-Q8_0");
    }

    #[test]
    fn an_explicit_hf_uri_displays_as_its_model_file_stem() {
        let mut cfg = crate::config::Config::default();
        cfg.models.embed = Some("hf:org/some-model-GGUF/some-model-Q4_K_M.gguf".into());
        assert_eq!(embed_model_display(&cfg), "some-model-Q4_K_M");
    }

    #[test]
    fn a_gemini_model_displays_with_its_scheme() {
        let mut cfg = crate::config::Config::default();
        cfg.models.embed = Some("gemini:gemini-embedding-2".into());
        assert_eq!(embed_model_display(&cfg), "gemini:gemini-embedding-2");
    }
}
