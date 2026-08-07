use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::config::GroupBy;
use crate::fusion::{self, RankedResult};
use crate::graph;
use crate::llm::{self, EmbedModel, OrchestratorModel, RerankModel};
use crate::store::{Store, StoreStats};

/// Compute cache key for orchestration results (SHA256 of query).
fn orchestration_cache_key(query: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(query.as_bytes());
    format!("{:x}", hash)
}

/// A single search result with metadata.
pub struct SearchResult {
    pub score: f32,
    pub confidence: f64,
    pub file_path: String,
    /// Which section of the file this result is, 0-based.
    pub chunk_seq: i64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
}

/// Structured search result for internal use (no I/O).
#[derive(Debug, Clone, serde::Serialize)]
pub struct InternalSearchResult {
    pub file_path: String,
    pub file_id: i64,
    /// Which section of the file this result is, 0-based.
    pub chunk_seq: i64,
    pub score: f64,
    pub confidence: f64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
}

/// Output from `search_internal`: structured results plus raw fused data for --explain.
pub struct SearchOutput {
    pub results: Vec<InternalSearchResult>,
    pub fused: Vec<fusion::FusedResult>,
    pub intent: Option<crate::llm::QueryIntent>,
}

/// Configuration for the intelligence search pipeline.
pub struct SearchConfig<'a> {
    pub orchestrator: Option<&'a mut dyn OrchestratorModel>,
    pub reranker: Option<&'a mut dyn RerankModel>,
    pub store: &'a Store,
    pub rerank_candidates: usize,
    /// Ceiling on how many sections of one document may appear in the results.
    pub max_chunks_per_file: usize,
    /// Whether results address sections or whole documents.
    pub group_by: GroupBy,
}

impl<'a> SearchConfig<'a> {
    /// A model-free config carrying the caller's retrieval settings.
    pub fn new(store: &'a Store, config: &crate::config::Config) -> Self {
        Self {
            orchestrator: None,
            reranker: None,
            store,
            rerank_candidates: 30,
            max_chunks_per_file: config.max_chunks_per_file,
            group_by: config.group_by,
        }
    }
}

/// Run hybrid search and return structured results (no I/O).
/// Used by both `run_search` (CLI) and context engine.
///
/// Thin wrapper around `search_with_intelligence` with no intelligence models,
/// preserving the existing heuristic-only behavior.
pub fn search_internal(
    query: &str,
    top_n: usize,
    store: &Store,
    embedder: &mut impl EmbedModel,
    group_by: GroupBy,
) -> Result<SearchOutput> {
    let mut config = SearchConfig {
        orchestrator: None,
        reranker: None,
        store,
        rerank_candidates: 30,
        max_chunks_per_file: crate::config::default_max_chunks_per_file(),
        group_by,
    };
    search_with_intelligence(query, top_n, embedder, &mut config)
}

/// Full intelligence search pipeline.
///
/// 1. Orchestrate (intent + expansions + weights) — LLM if available, else heuristic.
/// 2. 3-lane retrieval per expanded query (semantic, FTS, graph).
/// 3. RRF Pass 1 with top candidates.
/// 4. Reranker scores each candidate (4th lane) if available.
/// 5. RRF Pass 2 with all 4 lanes for final ranking.
pub fn search_with_intelligence(
    query: &str,
    top_n: usize,
    embedder: &mut impl EmbedModel,
    config: &mut SearchConfig<'_>,
) -> Result<SearchOutput> {
    // --- Step 1: Orchestrate (with LLM cache when orchestrator is present) ---
    let orchestration = match &mut config.orchestrator {
        Some(orch) => {
            let cache_key = orchestration_cache_key(query);
            if let Some(cached_json) = config.store.get_llm_cache(&cache_key)? {
                serde_json::from_str(&cached_json).unwrap_or_else(|_| {
                    orch.orchestrate(query)
                        .unwrap_or_else(|_| llm::heuristic_orchestrate(query))
                })
            } else {
                let result = orch.orchestrate(query)?;
                if let Ok(json) = serde_json::to_string(&result) {
                    let _ = config
                        .store
                        .set_llm_cache(&cache_key, &json, "orchestrator");
                }
                result
            }
        }
        None => llm::heuristic_orchestrate(query),
    };
    tracing::debug!(
        intent = ?orchestration.intent,
        expansions = orchestration.expansions.len(),
        "orchestration complete"
    );
    let weights = llm::LaneWeights::from_intent(&orchestration.intent);

    // --- Step 2: Run 3-lane retrieval for EACH expanded query ---
    let mut all_semantic: Vec<RankedResult> = Vec::new();
    let mut all_fts: Vec<RankedResult> = Vec::new();

    for expanded_query in &orchestration.expansions {
        // Semantic lane
        let query_vec = embedder
            .embed_one(expanded_query)
            .context("embedding query")?;
        let tombstones = std::collections::HashSet::new();
        let raw_results = config
            .store
            .search_vec(&query_vec, top_n * 3, &tombstones)?;

        for (vector_id, distance) in raw_results {
            if let Some(chunk) = config.store.get_chunk_by_vector_id(vector_id)? {
                let (file_path, docid) = match config.store.get_file_by_id(chunk.file_id)? {
                    Some(f) => (f.path, f.docid),
                    None => ("<unknown>".to_string(), None),
                };
                let heading = if chunk.heading.is_empty() {
                    None
                } else {
                    Some(chunk.heading)
                };

                all_semantic.push(RankedResult {
                    file_path,
                    file_id: chunk.file_id,
                    chunk_seq: chunk.seq,
                    score: (1.0 - distance) as f64,
                    heading,
                    snippet: chunk.snippet,
                    docid,
                });
            }
        }

        // FTS lane
        let fts_raw = config
            .store
            .fts_search(expanded_query, top_n * 3)
            .unwrap_or_default();

        for fr in fts_raw {
            let (file_path, docid) = match config.store.get_file_by_id(fr.file_id)? {
                Some(f) => (f.path, f.docid),
                None => continue,
            };
            // The FTS index holds text only; the heading lives on the chunk row,
            // which `(file_id, chunk_seq)` now reaches.
            let heading = config
                .store
                .get_chunk_by_seq(fr.file_id, fr.chunk_seq)?
                .map(|c| c.heading)
                .filter(|h| !h.is_empty());

            all_fts.push(RankedResult {
                file_path,
                file_id: fr.file_id,
                chunk_seq: fr.chunk_seq,
                score: fr.score,
                heading,
                snippet: fr.snippet,
                docid,
            });
        }
    }

    // Deduplicate across expanded queries, then bound each file's share of the
    // lane. Without the bound a 33-chunk document would take 33 of the ranks
    // this lane hands to RRF, pushing every other document down.
    let cap = config.max_chunks_per_file;
    let semantic_results = collapse_lane(all_semantic, cap);
    let fts_results = collapse_lane(all_fts, cap);

    // --- Graph lane from combined seeds ---
    let mut combined_seeds = merge_seeds(&semantic_results, &fts_results);

    // Inject temporal candidates as graph seeds when date_range is present.
    // Seeds are consumed by `graph_expand`, which reads only the file, so these
    // never reach fusion and their chunk identity is not used.
    let temporal_seeds: Vec<RankedResult> = if let Some(range) = &orchestration.date_range {
        config
            .store
            .get_files_in_date_range(range.0, range.1)
            .unwrap_or_default()
            .iter()
            .map(|f| RankedResult {
                file_path: f.path.clone(),
                file_id: f.id,
                chunk_seq: 0,
                score: 1.0,
                heading: None,
                snippet: String::new(),
                docid: f.docid.clone(),
            })
            .collect()
    } else {
        vec![]
    };
    for ts in &temporal_seeds {
        let dominated = combined_seeds
            .iter()
            .any(|s| s.file_path == ts.file_path && s.score >= ts.score);
        if !dominated {
            combined_seeds.retain(|s| s.file_path != ts.file_path);
            combined_seeds.push(ts.clone());
        }
    }

    let graph_results =
        graph::graph_expand(config.store, &combined_seeds, query, 2, 20).unwrap_or_default();

    // --- Step 3: RRF Pass 1 (3-lane) ---
    const RRF_K: usize = 60;
    let fused_pass1 = fusion::rrf_fuse(
        &[
            ("semantic", &semantic_results, weights.semantic),
            ("fts", &fts_results, weights.fts),
            ("graph", &graph_results, weights.graph),
        ],
        RRF_K,
    );

    // --- Step 4: Reranker (4th lane) if available ---
    let mut rerank_results: Vec<RankedResult> = Vec::new();
    let reranker_used = if let Some(reranker) = &mut config.reranker {
        let candidates: Vec<_> = fused_pass1.iter().take(config.rerank_candidates).collect();
        let documents: Vec<&str> = candidates.iter().map(|c| c.snippet.as_str()).collect();

        // One call for all thirty pairs, so the reranker sets up once instead of
        // once per candidate (issue #13). A failure now costs the whole lane
        // rather than one candidate; the failures in reach are tokenizer and
        // decode errors, which would take every pair down anyway, and unlike the
        // old per-pair `unwrap_or(0.0)` this one says so.
        let scores = reranker
            .rerank_batch(query, &documents)
            .unwrap_or_else(|e| {
                tracing::warn!("rerank lane unavailable: {e:#}");
                vec![0.0; documents.len()]
            });

        for (candidate, score) in candidates.iter().zip(scores) {
            let score = score as f64;
            rerank_results.push(RankedResult {
                file_path: candidate.file_path.clone(),
                file_id: candidate.file_id,
                chunk_seq: candidate.chunk_seq,
                score,
                heading: candidate.heading.clone(),
                snippet: candidate.snippet.clone(),
                docid: candidate.docid.clone(),
            });
        }
        rerank_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        true
    } else {
        false
    };

    // --- Step 5: Temporal lane (5th lane) when date_range is present ---
    let final_fused = if let Some(range) = &orchestration.date_range {
        // Build temporal lane: score ALL candidates from pass1/reranked by date proximity
        let base_fused = if reranker_used {
            fusion::rrf_fuse(
                &[
                    ("semantic", &semantic_results, weights.semantic),
                    ("fts", &fts_results, weights.fts),
                    ("graph", &graph_results, weights.graph),
                    ("rerank", &rerank_results, weights.rerank),
                ],
                RRF_K,
            )
        } else {
            // Use pass1 as the candidate source; avoid clone by re-referencing
            fused_pass1
        };
        let mut temporal_results: Vec<RankedResult> = base_fused
            .iter()
            .filter_map(|c| {
                let file = config.store.get_file(&c.file_path).ok()??;
                let nd = file.note_date?;
                let score = crate::temporal::temporal_score(nd, range.0, range.1);
                Some(RankedResult {
                    file_path: c.file_path.clone(),
                    file_id: c.file_id,
                    chunk_seq: c.chunk_seq,
                    score,
                    heading: c.heading.clone(),
                    snippet: c.snippet.clone(),
                    docid: c.docid.clone(),
                })
            })
            .collect();
        temporal_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 5-lane RRF (rerank_results is empty when reranker absent, weight 0)
        fusion::rrf_fuse(
            &[
                ("semantic", &semantic_results, weights.semantic),
                ("fts", &fts_results, weights.fts),
                ("graph", &graph_results, weights.graph),
                ("rerank", &rerank_results, weights.rerank),
                ("temporal", &temporal_results, weights.temporal),
            ],
            RRF_K,
        )
    } else if reranker_used {
        // Non-temporal with reranker: 4-lane (existing behavior)
        fusion::rrf_fuse(
            &[
                ("semantic", &semantic_results, weights.semantic),
                ("fts", &fts_results, weights.fts),
                ("graph", &graph_results, weights.graph),
                ("rerank", &rerank_results, weights.rerank),
            ],
            RRF_K,
        )
    } else {
        // Non-temporal without reranker: 3-lane (existing behavior)
        fused_pass1
    };

    // Bound each document's share of the result set. `GroupBy::File` is the same
    // operation with a cap of one, which is what engraph did before chunks were
    // addressable at all.
    let final_fused = fusion::cap_per_file(
        final_fused,
        match config.group_by {
            GroupBy::File => 1,
            GroupBy::Chunk => cap,
        },
    );

    // Convert fused results to InternalSearchResult, taking top_n.
    let results: Vec<InternalSearchResult> = final_fused
        .iter()
        .take(top_n)
        .map(|f| InternalSearchResult {
            file_path: f.file_path.clone(),
            file_id: f.file_id,
            chunk_seq: f.chunk_seq,
            score: f.rrf_score,
            confidence: f.confidence,
            heading: f.heading.clone(),
            snippet: f.snippet.clone(),
            docid: f.docid.clone(),
        })
        .collect();

    Ok(SearchOutput {
        results,
        fused: final_fused,
        intent: Some(orchestration.intent),
    })
}

/// Prepare one lane's raw hits for fusion: one entry per chunk, best score wins,
/// sorted best-first, and at most `cap` chunks from any single file.
///
/// The same chunk can arrive several times — once per query expansion — and the
/// cap is what keeps a long document from owning the lane's top ranks.
fn collapse_lane(results: Vec<RankedResult>, cap: usize) -> Vec<RankedResult> {
    let mut by_chunk: HashMap<(i64, i64), RankedResult> = HashMap::new();
    for r in results {
        let dominated = by_chunk
            .get(&r.chunk_key())
            .is_some_and(|existing| existing.score >= r.score);
        if !dominated {
            by_chunk.insert(r.chunk_key(), r);
        }
    }
    let mut deduped: Vec<RankedResult> = by_chunk.into_values().collect();
    deduped.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Ties are common in FTS; order by chunk so results are stable.
            .then_with(|| a.chunk_key().cmp(&b.chunk_key()))
    });

    if cap == 0 {
        return deduped;
    }
    let mut per_file: HashMap<i64, usize> = HashMap::new();
    deduped.retain(|r| {
        let count = per_file.entry(r.file_id).or_insert(0);
        *count += 1;
        *count <= cap
    });
    deduped
}

/// Merge semantic and FTS seed results, keeping the highest score per file.
fn merge_seeds(semantic: &[RankedResult], fts: &[RankedResult]) -> Vec<RankedResult> {
    let mut by_file: HashMap<String, RankedResult> = HashMap::new();
    for r in semantic.iter().chain(fts.iter()) {
        let dominated = by_file
            .get(&r.file_path)
            .is_some_and(|existing| existing.score >= r.score);
        if !dominated {
            by_file.insert(r.file_path.clone(), r.clone());
        }
    }
    // Seeds drive graph expansion; a stable order keeps that reproducible.
    let mut seeds: Vec<RankedResult> = by_file.into_values().collect();
    seeds.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_key().cmp(&b.chunk_key()))
    });
    seeds
}

/// Run a search query and print results.
///
/// Performs both semantic (sqlite-vec) and keyword (FTS5) search, then fuses
/// results using Reciprocal Rank Fusion. When `explain` is true, each
/// result includes per-lane score breakdown.
pub fn run_search(
    query: &str,
    top_n: usize,
    json: bool,
    explain: bool,
    group_by: GroupBy,
    data_dir: &Path,
    config: &crate::config::Config,
) -> Result<()> {
    let models_dir = data_dir.join("models");
    let mut embedder =
        crate::llm::LlamaEmbed::new(&models_dir, config).context("loading embedder")?;

    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path).context("opening store")?;
    store.verify_embedding_dim(embedder.dim())?;

    // Load intelligence models if enabled.
    let mut orchestrator_model: Option<Box<dyn llm::OrchestratorModel>> =
        if config.intelligence_enabled() {
            match crate::llm::LlamaOrchestrator::new(&models_dir, config) {
                Ok(o) => Some(Box::new(o)),
                Err(e) => {
                    tracing::warn!("failed to load orchestrator: {e}");
                    None
                }
            }
        } else {
            None
        };
    let mut reranker_model: Option<Box<dyn llm::RerankModel>> = if config.intelligence_enabled() {
        match crate::llm::LlamaRerank::new(&models_dir, config) {
            Ok(r) => Some(Box::new(r)),
            Err(e) => {
                tracing::warn!("failed to load reranker: {e}");
                None
            }
        }
    } else {
        None
    };

    let output = {
        let mut search_config = SearchConfig {
            orchestrator: orchestrator_model
                .as_mut()
                .map(|o| o.as_mut() as &mut dyn llm::OrchestratorModel),
            reranker: reranker_model
                .as_mut()
                .map(|r| r.as_mut() as &mut dyn llm::RerankModel),
            group_by,
            ..SearchConfig::new(&store, config)
        };
        search_with_intelligence(query, top_n, &mut embedder, &mut search_config)?
    };

    let results: Vec<SearchResult> = output
        .results
        .iter()
        .map(|r| SearchResult {
            score: r.score as f32,
            confidence: r.confidence,
            file_path: r.file_path.clone(),
            chunk_seq: r.chunk_seq,
            heading: r.heading.clone(),
            snippet: r.snippet.clone(),
            docid: r.docid.clone(),
        })
        .collect();

    let mut out = format_results(&results, json);

    if explain && !json {
        let mut explain_out = String::new();
        if let Some(ref intent) = output.intent {
            explain_out.push_str(&format!("Intent: {:?}\n\n", intent));
        }
        explain_out.push_str("--- Explain ---\n");
        for f in output.fused.iter().take(top_n) {
            explain_out.push_str(&format!("{}\n", f.file_path));
            explain_out.push_str(&fusion::format_explain(f));
        }
        out.push_str(&explain_out);
    }

    print!("{out}");
    Ok(())
}

/// Run the status command and print index information.
pub fn run_status(json: bool, data_dir: &Path) -> Result<()> {
    let db_path = data_dir.join("engraph.db");
    let store = Store::open(&db_path).context("opening store")?;
    let stats = store.stats()?;
    let date_count = store.count_files_with_dates().unwrap_or(0);

    // Compute index size on disk (sqlite db file).
    let index_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    let model_name = "all-MiniLM-L6-v2";

    let config = crate::config::Config::load().unwrap_or_default();
    let intelligence = if config.intelligence_enabled() {
        "enabled"
    } else {
        "disabled"
    };

    let output = format_status(
        &stats,
        index_size,
        model_name,
        intelligence,
        date_count,
        json,
    );
    print!("{output}");
    Ok(())
}

/// Format search results for display (pure function, no I/O).
pub fn format_results(results: &[SearchResult], json: bool) -> String {
    if results.is_empty() {
        return if json {
            "[]\n".to_string()
        } else {
            "No results found.\n".to_string()
        };
    }

    if json {
        let items: Vec<serde_json::Value> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                // Round score to 2 decimal places via f64 to avoid f32 precision artifacts.
                let score_rounded = ((r.score as f64) * 100.0).round() / 100.0;
                json!({
                    "rank": i + 1,
                    "score": score_rounded,
                    "confidence": r.confidence,
                    "file": r.file_path,
                    "section": r.chunk_seq,
                    "heading": r.heading,
                    "snippet": r.snippet,
                    "docid": r.docid,
                })
            })
            .collect();
        format!("{}\n", serde_json::to_string_pretty(&items).unwrap())
    } else {
        let mut out = String::new();
        for (i, r) in results.iter().enumerate() {
            let heading_part = match &r.heading {
                Some(h) => format!(" > {h}"),
                None => String::new(),
            };
            let docid_part = match &r.docid {
                Some(d) => format!(" #{d}"),
                None => String::new(),
            };
            let snippet = truncate_snippet(&r.snippet, 200);
            out.push_str(&format!(
                "{:>2}. [{:>3.0}%] {}{}{}\n    {}\n",
                i + 1,
                r.confidence,
                r.file_path,
                heading_part,
                docid_part,
                snippet,
            ));
        }
        out
    }
}

/// Format status information for display (pure function, no I/O).
pub fn format_status(
    stats: &StoreStats,
    index_size: u64,
    model_name: &str,
    intelligence: &str,
    date_count: usize,
    json: bool,
) -> String {
    let vault = stats.vault_path.as_deref().unwrap_or("<not set>");
    let last_indexed = stats.last_indexed_at.as_deref().unwrap_or("never");

    if json {
        let mut obj = json!({
            "vault": vault,
            "files": stats.file_count,
            "chunks": stats.chunk_count,
            "tombstones": stats.tombstone_count,
            "last_indexed": last_indexed,
            "index_size": index_size,
            "model": model_name,
            "intelligence": intelligence,
            "files_with_dates": date_count,
        });
        if let (Some(edges), Some(wl), Some(mn)) =
            (stats.edge_count, stats.wikilink_count, stats.mention_count)
        {
            obj["edges"] = json!(edges);
            obj["wikilink_edges"] = json!(wl);
            obj["mention_edges"] = json!(mn);
        }
        format!("{}\n", serde_json::to_string_pretty(&obj).unwrap())
    } else {
        let mut out = format!(
            "Vault:      {}\n\
             Files:      {}\n\
             Chunks:     {}\n",
            vault, stats.file_count, stats.chunk_count,
        );
        if let (Some(edges), Some(wl), Some(mn)) =
            (stats.edge_count, stats.wikilink_count, stats.mention_count)
        {
            out.push_str(&format!(
                "Edges:      {} ({} wikilinks, {} mentions)\n",
                edges, wl, mn
            ));
        }
        out.push_str(&format!(
            "Dates:      {}/{} files\n\
             Tombstones: {} (pending cleanup)\n\
             Last index: {}\n\
             Index size: {}\n\
             Model:      {}\n\
             Intelligence: {}\n",
            date_count,
            stats.file_count,
            stats.tombstone_count,
            last_indexed,
            format_bytes(index_size),
            model_name,
            intelligence,
        ));
        out
    }
}

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
fn truncate_snippet(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find a char boundary near max_len.
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_human_result() {
        let results = vec![SearchResult {
            score: 0.87,
            confidence: 100.0,
            file_path: "foo.md".to_string(),
            chunk_seq: 0,
            heading: Some("## Bar".to_string()),
            snippet: "Some text...".to_string(),
            docid: Some("ab12cd".to_string()),
        }];
        let output = format_results(&results, false);
        assert_eq!(
            output,
            " 1. [100%] foo.md > ## Bar #ab12cd\n    Some text...\n"
        );
    }

    #[test]
    fn test_format_human_result_no_docid() {
        let results = vec![SearchResult {
            score: 0.87,
            confidence: 100.0,
            file_path: "foo.md".to_string(),
            chunk_seq: 0,
            heading: Some("## Bar".to_string()),
            snippet: "Some text...".to_string(),
            docid: None,
        }];
        let output = format_results(&results, false);
        assert_eq!(output, " 1. [100%] foo.md > ## Bar\n    Some text...\n");
    }

    #[test]
    fn test_format_json_result() {
        let results = vec![SearchResult {
            score: 0.87,
            confidence: 100.0,
            file_path: "foo.md".to_string(),
            chunk_seq: 0,
            heading: Some("## Bar".to_string()),
            snippet: "Some text...".to_string(),
            docid: Some("ab12cd".to_string()),
        }];
        let output = format_results(&results, true);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["rank"], 1);
        assert_eq!(parsed[0]["score"], 0.87);
        assert_eq!(parsed[0]["confidence"], 100.0);
        assert_eq!(parsed[0]["file"], "foo.md");
        assert_eq!(
            parsed[0]["section"], 0,
            "results address a section, not a file"
        );
        assert_eq!(parsed[0]["heading"], "## Bar");
        assert_eq!(parsed[0]["snippet"], "Some text...");
        assert_eq!(parsed[0]["docid"], "ab12cd");
    }

    #[test]
    fn test_no_results_message() {
        let output = format_results(&[], false);
        assert_eq!(output, "No results found.\n");

        let json_output = format_results(&[], true);
        assert_eq!(json_output, "[]\n");
    }

    #[test]
    fn test_format_status_human() {
        let stats = StoreStats {
            file_count: 42,
            chunk_count: 187,
            tombstone_count: 3,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
            edge_count: None,
            wikilink_count: None,
            mention_count: None,
        };
        let output = format_status(&stats, 2_516_582, "all-MiniLM-L6-v2", "disabled", 30, false);

        assert!(output.contains("/path/to/vault"), "missing vault path");
        assert!(output.contains("42"), "missing file count");
        assert!(output.contains("187"), "missing chunk count");
        assert!(output.contains("30/42 files"), "missing date coverage");
        assert!(output.contains("3"), "missing tombstone count");
        assert!(output.contains("2026-03-19 14:30:00"), "missing last index");
        assert!(output.contains("2.4 MB"), "missing index size");
        assert!(output.contains("all-MiniLM-L6-v2"), "missing model");
        assert!(output.contains("disabled"), "missing intelligence");
    }

    #[test]
    fn test_format_status_json() {
        let stats = StoreStats {
            file_count: 42,
            chunk_count: 187,
            tombstone_count: 3,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
            edge_count: None,
            wikilink_count: None,
            mention_count: None,
        };
        let output = format_status(&stats, 2_516_582, "all-MiniLM-L6-v2", "enabled", 30, true);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(parsed["vault"], "/path/to/vault");
        assert_eq!(parsed["files"], 42);
        assert_eq!(parsed["chunks"], 187);
        assert_eq!(parsed["tombstones"], 3);
        assert_eq!(parsed["last_indexed"], "2026-03-19 14:30:00");
        assert_eq!(parsed["index_size"], 2_516_582);
        assert_eq!(parsed["model"], "all-MiniLM-L6-v2");
        assert_eq!(parsed["intelligence"], "enabled");
        assert_eq!(parsed["files_with_dates"], 30);
    }

    #[test]
    fn test_truncate_snippet() {
        let short = "hello";
        assert_eq!(truncate_snippet(short, 200), "hello");

        let long = "a".repeat(300);
        let truncated = truncate_snippet(&long, 200);
        assert!(truncated.ends_with("..."));
        assert_eq!(truncated.len(), 203); // 200 + "..."
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(2_516_582), "2.4 MB");
    }

    #[test]
    fn test_cache_key_deterministic() {
        let key1 = super::orchestration_cache_key("how does auth work");
        let key2 = super::orchestration_cache_key("how does auth work");
        assert_eq!(key1, key2);

        let key3 = super::orchestration_cache_key("different query");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_search_output_has_intent() {
        let output = SearchOutput {
            results: vec![],
            fused: vec![],
            intent: Some(crate::llm::QueryIntent::Conceptual),
        };
        assert_eq!(output.intent, Some(crate::llm::QueryIntent::Conceptual));
    }

    #[test]
    fn test_search_output_intent_none() {
        let output = SearchOutput {
            results: vec![],
            fused: vec![],
            intent: None,
        };
        assert!(output.intent.is_none());
    }

    fn ranked(file_path: &str, file_id: i64, chunk_seq: i64, score: f64) -> RankedResult {
        RankedResult {
            file_path: file_path.to_string(),
            file_id,
            chunk_seq,
            score,
            heading: None,
            snippet: format!("{file_path}#{chunk_seq}@{score}"),
            docid: None,
        }
    }

    #[test]
    fn collapse_lane_keeps_best_score_per_chunk() {
        // The same chunk arrives twice — once per query expansion.
        let results = vec![
            ranked("a.md", 1, 0, 0.5),
            ranked("a.md", 1, 0, 0.9),
            ranked("b.md", 2, 0, 0.7),
        ];
        let collapsed = collapse_lane(results, 3);
        assert_eq!(collapsed.len(), 2);
        // Sorted by score descending
        assert_eq!(collapsed[0].file_path, "a.md");
        assert!((collapsed[0].score - 0.9).abs() < 1e-10);
        assert_eq!(collapsed[0].snippet, "a.md#0@0.9");
        assert_eq!(collapsed[1].file_path, "b.md");
    }

    #[test]
    fn collapse_lane_keeps_distinct_chunks_of_one_file() {
        // This is the whole point of #6: two sections of one document are two
        // results, not one. Before, the second was discarded.
        let results = vec![
            ranked("a.md", 1, 0, 0.5),
            ranked("a.md", 1, 4, 0.9),
            ranked("a.md", 1, 7, 0.7),
        ];
        let collapsed = collapse_lane(results, 3);
        assert_eq!(collapsed.len(), 3);
        let seqs: Vec<i64> = collapsed.iter().map(|r| r.chunk_seq).collect();
        assert_eq!(seqs, vec![4, 7, 0], "ordered by score, not by position");
    }

    #[test]
    fn collapse_lane_caps_one_files_share_of_the_lane() {
        // A 6-section document must not own the lane's top six ranks.
        let mut results: Vec<RankedResult> = (0..6)
            .map(|seq| ranked("long.md", 1, seq, 0.9 - seq as f64 * 0.01))
            .collect();
        results.push(ranked("other.md", 2, 0, 0.5));

        let collapsed = collapse_lane(results, 2);
        assert_eq!(collapsed.len(), 3);
        assert_eq!(
            collapsed
                .iter()
                .filter(|r| r.file_path == "long.md")
                .count(),
            2
        );
        assert!(
            collapsed.iter().any(|r| r.file_path == "other.md"),
            "the capped file must not crowd out other documents"
        );
        // The two survivors are its best-scoring sections.
        assert_eq!(collapsed[0].chunk_seq, 0);
        assert_eq!(collapsed[1].chunk_seq, 1);
    }

    #[test]
    fn collapse_lane_cap_of_zero_is_unlimited() {
        let results: Vec<RankedResult> = (0..5).map(|seq| ranked("long.md", 1, seq, 0.5)).collect();
        assert_eq!(collapse_lane(results, 0).len(), 5);
    }

    #[test]
    fn collapse_lane_empty() {
        assert!(collapse_lane(vec![], 3).is_empty());
    }

    /// Index a small vault in memory. Embeddings are hash-based, so only the FTS
    /// lane carries meaning here — which is enough to pin retrieval granularity.
    fn indexed_vault() -> (tempfile::TempDir, Store, crate::llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("rules")).unwrap();
        std::fs::write(
            root.join("rules/abjuration-spells.md"),
            "# Abjuration\n\n\
             ## Level 3 Counterspell\n\nA warding effect that stops a spell mid-cast.\n\n\
             ## Level 5 Dispel Magic\n\nA warding effect that ends an ongoing spell.\n\n\
             ## Level 9 Dimensional Anchor\n\nA warding effect that pins a creature.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("rules/evocation-spells.md"),
            "# Evocation\n\n## Level 1 Firebolt\n\nA bolt of flame.\n",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(root, &config, &store, &mut embedder, false, None)
            .unwrap();
        (tmp, store, embedder)
    }

    #[test]
    fn one_document_can_contribute_several_sections() {
        // #6's acceptance criterion. Three sections of one file match "warding";
        // before chunk-level fusion only one of them could ever be returned.
        let (_tmp, store, mut embedder) = indexed_vault();

        let output = search_internal("warding", 10, &store, &mut embedder, GroupBy::Chunk).unwrap();

        let hits: Vec<&InternalSearchResult> = output
            .results
            .iter()
            .filter(|r| r.file_path == "rules/abjuration-spells.md")
            .collect();
        assert!(
            hits.len() > 1,
            "expected several sections of one document, got {}: {:#?}",
            hits.len(),
            output.results
        );

        let seqs: std::collections::HashSet<i64> = hits.iter().map(|r| r.chunk_seq).collect();
        assert_eq!(seqs.len(), hits.len(), "each result is a distinct section");

        // Every result names the section it came from.
        assert!(hits.iter().all(|r| r.heading.is_some()));
    }

    /// Records how the rerank lane is driven. Whether an implementation can
    /// amortize its setup depends on being handed the whole candidate set at
    /// once (issue #13), so "one call, not one per candidate" is a property to
    /// hold onto rather than an implementation detail.
    struct CountingReranker {
        inner: llm::MockLlm,
        batch_calls: usize,
        pairs_scored: usize,
    }

    impl RerankModel for CountingReranker {
        fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32> {
            self.inner.rerank_score(query, document)
        }

        fn rerank_batch(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            self.batch_calls += 1;
            self.pairs_scored += documents.len();
            documents
                .iter()
                .map(|d| self.inner.rerank_score(query, d))
                .collect()
        }
    }

    #[test]
    fn the_rerank_lane_scores_all_its_candidates_in_one_call() {
        let (_tmp, store, mut embedder) = indexed_vault();
        let mut reranker = CountingReranker {
            inner: llm::MockLlm::new(8),
            batch_calls: 0,
            pairs_scored: 0,
        };

        {
            let mut config = SearchConfig {
                orchestrator: None,
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
            };
            let output =
                search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();
            assert!(!output.results.is_empty(), "the lane produced no results");
        }

        assert_eq!(
            reranker.batch_calls, 1,
            "the lane must set up once, not once per candidate"
        );
        assert!(
            reranker.pairs_scored > 1,
            "expected several candidates, got {}",
            reranker.pairs_scored
        );
    }

    #[test]
    fn results_respect_the_per_file_cap() {
        let (_tmp, store, mut embedder) = indexed_vault();

        for cap in [1, 2] {
            let mut config = SearchConfig {
                orchestrator: None,
                reranker: None,
                store: &store,
                rerank_candidates: 30,
                max_chunks_per_file: cap,
                group_by: GroupBy::Chunk,
            };
            let output =
                search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();

            let count = output
                .results
                .iter()
                .filter(|r| r.file_path == "rules/abjuration-spells.md")
                .count();
            assert!(count <= cap, "cap {cap} exceeded: {count} results");
        }
    }

    #[test]
    fn group_by_file_returns_one_result_per_document() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let output = search_internal("warding", 10, &store, &mut embedder, GroupBy::File).unwrap();

        let mut seen = std::collections::HashSet::new();
        for r in &output.results {
            assert!(
                seen.insert(r.file_path.clone()),
                "{} appeared twice under file grouping",
                r.file_path
            );
        }
        assert!(!output.results.is_empty());
    }

    #[test]
    fn fts_results_carry_their_heading() {
        // The FTS lane used to return `heading: None` because the FTS index holds
        // no headings; `(file_id, chunk_seq)` is what recovers them.
        let (_tmp, store, mut embedder) = indexed_vault();

        let output =
            search_internal("mid-cast", 10, &store, &mut embedder, GroupBy::Chunk).unwrap();
        let hit = output
            .results
            .iter()
            .find(|r| r.file_path == "rules/abjuration-spells.md")
            .expect("the phrase appears in exactly one section");
        assert_eq!(hit.heading.as_deref(), Some("## Level 3 Counterspell"));
    }

    #[test]
    fn test_merge_seeds_deduplicates() {
        let semantic = vec![RankedResult {
            file_path: "shared.md".to_string(),
            file_id: 1,
            chunk_seq: 0,
            score: 0.8,
            heading: None,
            snippet: "sem".to_string(),
            docid: None,
        }];
        let fts = vec![
            RankedResult {
                file_path: "shared.md".to_string(),
                file_id: 1,
                chunk_seq: 0,
                score: 0.9,
                heading: None,
                snippet: "fts".to_string(),
                docid: None,
            },
            RankedResult {
                file_path: "fts_only.md".to_string(),
                file_id: 2,
                chunk_seq: 0,
                score: 0.6,
                heading: None,
                snippet: "fts only".to_string(),
                docid: None,
            },
        ];
        let merged = merge_seeds(&semantic, &fts);
        assert_eq!(merged.len(), 2);
        // "shared.md" should have the FTS score (0.9 > 0.8)
        let shared = merged.iter().find(|r| r.file_path == "shared.md").unwrap();
        assert!((shared.score - 0.9).abs() < 1e-10);
    }
}
