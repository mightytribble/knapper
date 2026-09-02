use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::config::{GroupBy, RankingMode, db_path};
use crate::fusion::{self, FusedResult, RankedResult};
use crate::graph;
use crate::llm::{self, EmbedModel, RerankModel};
use crate::packaging::est_tokens_fallback;
use crate::ranking;
use crate::store::{EdgeStats, Store, StoreStats};

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
    /// The scored window (#35): the whole chunk at the shipped default.
    pub text: String,
    /// The stored breadcrumb, `Note Title > H1 > H2 > H3`, path-rooted (#46).
    pub heading_path: String,
    /// Tokens in `text`: the reranker's own count, or `ceil(chars / 3.33)`.
    pub token_count: usize,
    /// Whether `[rerank] max_document_chars` cut `text` below its section.
    pub truncated: bool,
    /// Which lanes account for this result, in place of a numeric score (#35).
    pub provenance: crate::packaging::Provenance,
}

/// Output from the search pipeline: structured results plus raw fused data for --explain.
pub struct SearchOutput {
    pub results: Vec<InternalSearchResult>,
    pub fused: Vec<fusion::FusedResult>,
    /// The cross-encoder was meant to sort this query and could not, so the
    /// documented 3:1 interleave produced the order instead (#30).
    ///
    /// A fallback nobody is told about becomes the real ranking the moment the
    /// model is slow or missing, and no number in the result distinguishes it.
    pub degraded: bool,
    /// What the query actually retrieved — the `--explain` record of the
    /// retrieval step, as opposed to the fusion step.
    pub retrieval: RetrievalTrace,
    /// The keyword index's declared columns and their BM25 weights (#37).
    /// Printed by `--explain` because the weights are a measurement input that
    /// leaves no other trace in the output.
    pub fts_columns: Vec<(&'static str, f64)>,
}

/// The tag scope a query ran under (#60).
///
/// Both halves are what `--explain` needs and nothing else knows: the filter
/// as the caller wrote it, and how many notes it actually admitted. A thin
/// result under a scope is explained by the second number more often than by
/// anything the lanes report.
pub struct ScopeTrace {
    /// `all=type/undead none=status/draft`, folded, empty fields omitted.
    pub filter: String,
    /// How many notes the filter resolved to.
    pub notes: usize,
}

/// The query as run, and what it brought back, per lane.
///
/// This exists because three separate defects (#18, #22, #23) all lived in the
/// gap between what the pipeline was asked to search for and what it searched
/// for, and none of them were visible from the outside: `--explain` reported
/// how the lanes were fused while saying nothing about what went into them.
/// Reconstructing that needed direct SQL against the index.
pub struct RetrievalTrace {
    /// The user's own words, which is now the only thing searched for (#59).
    pub query: String,
    /// The FTS5 MATCH expression this became, or `None` if it had no searchable
    /// token and the lane was skipped.
    pub fts_expr: Option<String>,
    /// The tag scope, or `None` when the query ran over the whole vault (#60).
    pub scope: Option<ScopeTrace>,
    pub semantic_hits: usize,
    pub fts_hits: usize,
    /// What the calibrated scorer computed for the query itself, or `None` on
    /// every other path — the legacy stage, a configured cross-encoder, and
    /// the degraded interleave (spec 2026-08-30).
    pub calibrated: Option<CalibratedTrace>,
}

/// What the calibrated scorer computed for the query itself (spec 2026-08-30):
/// the BM25 upper bound and each term's idf. Per-candidate probabilities are
/// the `"calibrated"` lane contribution and print with the fusion detail.
#[derive(Debug, Clone)]
pub struct CalibratedTrace {
    pub bound: f64,
    pub idfs: Vec<(String, f64)>,
}

/// Configuration for the intelligence search pipeline.
pub struct SearchConfig<'a> {
    pub reranker: Option<&'a mut dyn RerankModel>,
    pub store: &'a Store,
    /// How many candidates the legacy stage shows the cross-encoder. The sorted
    /// stage reads `ranking.candidates` instead.
    pub rerank_candidates: usize,
    /// How candidates are presented to the reranker.
    pub rerank: crate::config::RerankConfig,
    /// Ceiling on how many sections of one document may appear in the results.
    pub max_chunks_per_file: usize,
    /// Whether results address sections or whole documents.
    pub group_by: GroupBy,
    /// Which ranking stage runs, and what reaches it (issue #30).
    pub ranking: crate::config::RankingConfig,
    /// What each lane's rank is worth to fusion (issue #59).
    pub lane_weights: crate::config::LaneWeights,
    /// What the keyword lane indexes, and how each column is weighted (#37).
    /// The flags have to be the ones the store was built with — the BM25
    /// weights are positional over the declared columns — and `fingerprint`
    /// blocks this path when they are not.
    pub fts: crate::config::FtsConfig,
    /// The notes this query may answer from (#60). Empty means the whole vault.
    pub scope: crate::tags::Scope,
    /// Calibrated score fusion for the model-free sorted stage
    /// (docs/specs/2026-08-30-calibrated-fusion-design.md).
    pub calibrated: crate::config::CalibratedConfig,
}

impl<'a> SearchConfig<'a> {
    /// A model-free config carrying the caller's retrieval settings.
    pub fn new(store: &'a Store, config: &crate::config::Config) -> Self {
        Self {
            reranker: None,
            store,
            rerank_candidates: 30,
            rerank: config.rerank,
            max_chunks_per_file: config.max_chunks_per_file,
            group_by: config.group_by,
            ranking: config.ranking,
            lane_weights: config.lane_weights,
            fts: config.fts,
            scope: crate::tags::Scope::default(),
            calibrated: config.calibrated.clone(),
        }
    }
}

/// The text the cross-encoder is asked to judge, one string per candidate.
///
/// Before issue #14 this was `candidate.snippet`, which is not a chunk and is
/// not even the same fraction of one from candidate to candidate: the semantic
/// and graph lanes attach the leading 200 characters, while the FTS lane
/// attaches SQLite's 64-token window centred on the match. Reading the whole
/// document jointly with the query is the only reason to pay for a
/// cross-encoder, so it now reads `chunks.text` — a primary-key lookup, since
/// #14 gave `chunks` its own copy.
///
/// A candidate whose text cannot be found falls back to its snippet. That is
/// the pre-#14 behaviour for that one candidate, which beats scoring it against
/// nothing.
/// One candidate as the rerank lane needs it: where its text lives, and what to
/// fall back to if the lookup misses.
///
/// A borrowed view rather than a trait, because the two ranking stages hold
/// their candidates in different types and neither should have to become the
/// other to be scored.
struct RerankTarget<'a> {
    file_id: i64,
    chunk_seq: i64,
    file_path: &'a str,
    snippet: &'a str,
}

/// One rerank candidate, as two strings and a flag (#35).
///
/// `document` is what the model scores: the body, with the title prepended
/// when `[rerank] document_title` is on. `body` is the emitted window — the
/// text `Candidate::emit_text` carries forward, so the result the caller reads
/// is the same text the model judged. `truncated` records whether
/// `max_document_chars` cut the window below its section.
struct RerankUnit {
    document: String,
    body: String,
    truncated: bool,
}

fn rerank_documents(
    store: &Store,
    candidates: &[RerankTarget<'_>],
    settings: crate::config::RerankConfig,
) -> Vec<RerankUnit> {
    let keys: Vec<(i64, i64)> = candidates
        .iter()
        .map(|c| (c.file_id, c.chunk_seq))
        .collect();
    let texts = store.get_chunk_texts(&keys).unwrap_or_else(|e| {
        tracing::warn!("reranker falling back to snippets: {e:#}");
        vec![None; keys.len()]
    });

    let units: Vec<RerankUnit> = candidates
        .iter()
        .zip(texts)
        .map(|(candidate, text)| {
            let text = text.unwrap_or_else(|| candidate.snippet.to_string());
            let raw_chars = text.chars().count();
            // Truncate first, then prepend the title, so a title can never be
            // the thing the cap cuts off.
            let body = truncate_chars(text, settings.max_document_chars);
            let truncated = settings.max_document_chars > 0 && body.chars().count() < raw_chars;
            let document = if settings.document_title {
                format!("{}\n\n{body}", document_title(candidate.file_path))
            } else {
                body.clone()
            };
            RerankUnit {
                document,
                body,
                truncated,
            }
        })
        .collect();

    // The one measurement that predicts this query's latency. Recorded because
    // every finding in #22 through #25 needed stage-level numbers that did not
    // exist, and end-to-end timings cannot separate "more candidates" from
    // "longer candidates".
    tracing::debug!(
        candidates = units.len(),
        chars = units.iter().map(|u| u.document.len()).sum::<usize>(),
        longest_chars = units.iter().map(|u| u.document.len()).max().unwrap_or(0),
        cap = settings.max_document_chars,
        "rerank input assembled"
    );

    units
}

/// Keep at most `max_chars` characters of `text`; 0 means keep all of it.
///
/// Characters, not bytes: cutting a UTF-8 string at an arbitrary byte offset
/// panics, and a vault that holds an em-dash should not be able to crash a
/// search. Nothing is appended to mark the cut — an ellipsis would be one more
/// token for the model to read and nothing downstream shows this text to a
/// person, since results carry their own snippet.
fn truncate_chars(mut text: String, max_chars: usize) -> String {
    if max_chars == 0 {
        return text;
    }
    // `nth` returns None when the string is already short enough, which is the
    // common case and costs one pass over at most `max_chars` characters.
    if let Some((byte_offset, _)) = text.char_indices().nth(max_chars) {
        text.truncate(byte_offset);
    }
    text
}

/// A document's title: its file stem, which is what an Obsidian vault names it by.
fn document_title(file_path: &str) -> &str {
    std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_path)
}

/// Full search pipeline.
///
/// 1. 3-lane retrieval for the query (semantic, FTS, graph).
/// 2. RRF Pass 1 with top candidates.
/// 3. Reranker scores each candidate (4th lane) if available.
/// 4. RRF Pass 2 with all 4 lanes for final ranking.
///
/// The query is the user's own words and nothing else. Query expansion is gone:
/// seventeen pool cells in the #57 section of `eval/probes.md` measure it, and
/// no expander — the word-splitter, Qwen3-0.6B, or a hand-written set — ever put
/// a tier-1 member in front of the cross-encoder that the query would not have.
/// Against a 22-slot shortlist it cost six of them by displacement (#59).
pub fn search_with_intelligence(
    query: &str,
    top_n: usize,
    embedder: &mut impl EmbedModel,
    config: &mut SearchConfig<'_>,
) -> Result<SearchOutput> {
    // The date range is a regex over the query and was never the model's, so it
    // outlives the orchestrator that used to carry it.
    let date_range = crate::temporal::parse_date_range_heuristic(query);
    let weights = config.lane_weights;

    // The scope resolves before anything is embedded, so a filter that admits
    // no note costs no model call, and a filter naming no tag fails with the
    // repair rather than with an empty result (#60).
    let scope_ids: Option<Vec<i64>> = match config.scope.is_empty() {
        true => None,
        false => Some(config.store.files_in_scope(&config.scope)?),
    };
    let scope_trace = scope_ids.as_ref().map(|ids| ScopeTrace {
        filter: config.scope.describe(),
        notes: ids.len(),
    });
    if scope_ids.as_ref().is_some_and(|ids| ids.is_empty()) {
        return Ok(SearchOutput {
            results: Vec::new(),
            fused: Vec::new(),
            degraded: false,
            retrieval: RetrievalTrace {
                query: query.to_string(),
                fts_expr: crate::fts::any_term_expr(query),
                semantic_hits: 0,
                fts_hits: 0,
                scope: scope_trace,
                // No lane ran, so there is nothing for the calibrated scorer
                // to have computed.
                calibrated: None,
            },
            fts_columns: config.fts.columns(),
        });
    }
    let scope_files: Option<&[i64]> = scope_ids.as_deref();
    let scope_set: Option<std::collections::HashSet<i64>> =
        scope_ids.as_ref().map(|ids| ids.iter().copied().collect());

    // --- Step 1: Run 3-lane retrieval ---
    //
    // Two pools per lane, holding the same hits with different score semantics:
    // the lane's own scores feed fusion, and a normalised copy feeds graph
    // expansion. See `normalise_lane_scores` for why one pool cannot serve both
    // (#26).
    //
    // How deep each content lane digs is `[ranking] retrieval_width` and not
    // `top_n`. The two were one number until #49, so asking for five more
    // results changed which candidates reached the model and rewrote the top of
    // the ranking. `top_n` now truncates the output and nothing else.
    let lane_width = config.ranking.retrieval_width;
    let mut semantic_hits: Vec<RankedResult> = Vec::new();
    let mut fts_hits: Vec<RankedResult> = Vec::new();

    // Semantic lane
    let query_vec = embedder.embed_query(query).context("embedding query")?;
    let tombstones = std::collections::HashSet::new();
    let raw_results = config
        .store
        .search_vec(&query_vec, lane_width, &tombstones, scope_files)?;

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

            semantic_hits.push(RankedResult {
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

    // FTS lane. `fts_search_any` and not `fts_search`: the latter phrase-
    // matches the whole string, which returned nothing for four of the five
    // seed probes and left this lane empty for every multi-word query (#22).
    let fts_raw = config
        .store
        .fts_search_any(query, lane_width, &config.fts.weights(), scope_files)
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

        fts_hits.push(RankedResult {
            file_path,
            file_id: fr.file_id,
            chunk_seq: fr.chunk_seq,
            score: fr.score,
            heading,
            snippet: fr.snippet,
            docid,
        });
    }

    // Mutable: the sorted stage fills `calibrated` in after it runs, below,
    // because `apply_calibrated_scores` has not been called yet at this point
    // in the function.
    let mut trace = RetrievalTrace {
        query: query.to_string(),
        fts_expr: crate::fts::any_term_expr(query),
        semantic_hits: semantic_hits.len(),
        fts_hits: fts_hits.len(),
        scope: scope_trace,
        calibrated: None,
    };

    // Rescale each lane against its own range on the way into the seed pool.
    // Before this, `1.0 - distance` (0.2-0.7) and negated BM25 (2-17) reached
    // `merge_seeds` raw, and every comparison between them was decided by which
    // lane's unit was bigger (#26).
    let semantic_seeds = normalised(&semantic_hits);
    let fts_seeds = normalised(&fts_hits);
    let all_semantic = semantic_hits;
    let all_fts = fts_hits;

    // Bound each file's share of the lane. Without the bound a 33-chunk document
    // would take 33 of the ranks this lane hands to RRF, pushing every other
    // document down.
    //
    // Under the sorted stage this is the *shortlist* cap and nothing else: what
    // the model is shown is bounded, what it returns is not (#30).
    //
    // Which stage runs is decided once, here, because retrieval is shaped for
    // it: the cap it applies and the size of the graph lane both belong to the
    // stage that will consume them.
    let sorts_by_model = config.ranking.mode == RankingMode::Sorted && config.reranker.is_some();
    // The model-free default (spec 2026-08-30): with no cross-encoder the
    // sorted stage still runs, and the calibrated logistic sorts the pool.
    // `[calibrated] enabled = false` is the control and restores the legacy
    // routing below, byte for byte.
    let sorts_calibrated = config.ranking.mode == RankingMode::Sorted
        && config.reranker.is_none()
        && config.calibrated.enabled;
    let enters_sorted = sorts_by_model || sorts_calibrated;
    let cap = if enters_sorted {
        config.ranking.shortlist_cap
    } else {
        config.max_chunks_per_file
    };
    let semantic_results = collapse_lane(all_semantic, cap);
    let fts_results = collapse_lane(all_fts, cap);

    // --- Graph lane from combined seeds ---
    //
    // Built from the normalised pools, not `semantic_results`/`fts_results`:
    // the seeds are the one place a lane's *magnitude* is read rather than its
    // rank, and magnitudes only mean something once each lane is on its own
    // scale (#26).
    let mut combined_seeds = merge_seeds(
        &collapse_lane(semantic_seeds, cap),
        &collapse_lane(fts_seeds, cap),
    );

    // Inject temporal candidates as graph seeds when date_range is present.
    // Seeds are consumed by `graph_expand`, which reads only the file, so these
    // never reach fusion and their chunk identity is not used.
    //
    // The flat 1.0 means "top of its lane" now that the content lanes are
    // normalised (#26). Before, it landed in the dead zone between them: always
    // above every semantic seed, never above any FTS one — a position that
    // described neither lane's confidence nor the date match's.
    let temporal_seeds: Vec<RankedResult> = if let Some(range) = &date_range {
        config
            .store
            .get_files_in_date_range(range.0, range.1)
            .unwrap_or_default()
            .iter()
            // A restart point asserts the note is a candidate answer, which is
            // not the same as being walked through, so the scope reaches these
            // where it does not reach the walk (#60).
            .filter(|f| scope_set.as_ref().is_none_or(|s| s.contains(&f.id)))
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

    // The lane does not read the query: its admission filter was a weaker second
    // copy of the FTS lane, scoped to one file (#29). What it was protecting
    // against — an unrankable score — is gone now that mass accumulates, so a
    // structurally implicated chunk simply sorts where its mass puts it.
    let graph_started = std::time::Instant::now();
    let default_expansions = graph::PprParams::default().max_expansions;
    let graph_results = graph::graph_expand(
        config.store,
        &combined_seeds,
        &graph::PprParams {
            cap_per_file: cap,
            // A reserve larger than the lane is allowed to produce would be a
            // quota nothing can fill — starvation dressed as a routing
            // guarantee. Below the default this changes nothing.
            max_expansions: if enters_sorted {
                config.ranking.graph_reserve.max(default_expansions)
            } else {
                default_expansions
            },
            ..graph::PprParams::default()
        },
        scope_set.as_ref(),
    )
    .unwrap_or_default();
    // The lane's cost was the second-largest stage in the query before #29 and
    // nothing reported it; `RUST_LOG=knapper::search=debug` is how the sweep
    // reads it back, as it does for the reranker's assembled size.
    tracing::debug!(
        seeds = combined_seeds.len(),
        expansions = graph_results.len(),
        elapsed_us = graph_started.elapsed().as_micros() as u64,
        "graph lane"
    );

    const RRF_K: usize = 60;

    // --- Steps 2-4: the ranking stage ---
    //
    // Two of them, chosen by `[ranking] mode`. The sorted stage is #30; the
    // legacy stage below is what it is measured against, and reproduces the
    // pre-#30 order byte for byte.
    //
    // **The sorted stage runs with a model or with the calibrated scorer.**
    // Its claim is that an absolute score is a better order than fused rank.
    // A cross-encoder gives one, and where there is no model the calibrated
    // logistic gives one too: each content lane is normalized onto the same
    // `[0, 1]` evidence scale and three fitted coefficients fuse the pair
    // (spec 2026-08-30). So a build with no cross-encoder is the default
    // install and it ranks with that probability.
    //
    // A build with no cross-encoder **and `[calibrated] enabled = false`**
    // takes the legacy stage below, which is the control: it reproduces the
    // pre-change order byte for byte and keeps the fusion that #9, #26, #28
    // and #29 each tuned against these probes.
    //
    // The interleave is unchanged and is neither of these two paths: it is the
    // fallback for a *configured* model that fails at call time (§7.3), and a
    // deliberate configuration is not that.
    if enters_sorted {
        let (final_fused, degraded, calibrated) = sorted_stage(
            query,
            &query_vec,
            config,
            &semantic_results,
            &fts_results,
            &graph_results,
            &weights,
            date_range,
            RRF_K,
        );
        trace.calibrated = calibrated;

        // No limit on the results by default. Limit what the model reads, not
        // what it returns: if one document holds the ten best sections, then ten
        // sections is the correct answer. #6 limited the results because lanes
        // voted, and the cross-encoder now sorts instead. `group_by = "file"`
        // stays, because it asks for a shape of answer and does not guard
        // against a lane mechanic. `per_note_cap` holds §9.1's opposite
        // position, and it ships with no limit so that a sweep needs only a
        // config edit (#34).
        let final_fused = match config.group_by {
            GroupBy::File => fusion::cap_per_file(final_fused, 1),
            GroupBy::Chunk => fusion::cap_per_file(final_fused, config.ranking.per_note_cap),
        };

        let results: Vec<InternalSearchResult> = final_fused
            .iter()
            .take(top_n)
            .map(|f| {
                // Whichever scorer sorted the pool — the cross-encoder or
                // the calibrated logistic — its own number, not a fused one.
                // This is the absolute score layer 2 thresholds on for
                // abstention.
                let score = model_score(f).unwrap_or(f.rrf_score);
                build_result(config.store, f, config.rerank, score)
            })
            .collect();

        return Ok(finalize_search_output(
            results,
            final_fused,
            degraded,
            trace,
            config.fts.columns(),
            config.ranking.coalesce_adjacent,
        ));
    }

    // --- Step 2: RRF Pass 1 (3-lane) ---
    let fused_pass1 = fusion::rrf_fuse(
        &[
            ("semantic", &semantic_results, weights.semantic),
            ("fts", &fts_results, weights.fts),
            ("graph", &graph_results, weights.graph),
        ],
        RRF_K,
    );

    // --- Step 3: Reranker (4th lane) if available ---
    let mut rerank_results: Vec<RankedResult> = Vec::new();
    let reranker_used = if let Some(reranker) = &mut config.reranker {
        let candidates: Vec<_> = fused_pass1.iter().take(config.rerank_candidates).collect();
        let targets: Vec<RerankTarget<'_>> = candidates.iter().map(|c| target_of(c)).collect();
        let units = rerank_documents(config.store, &targets, config.rerank);
        let documents: Vec<&str> = units.iter().map(|u| u.document.as_str()).collect();

        // One call for all thirty pairs, so the reranker sets up once instead of
        // once per candidate (issue #13). A failure now costs the whole lane
        // rather than one candidate; the failures in reach are tokenizer and
        // decode errors, which would take every pair down anyway, and unlike the
        // old per-pair `unwrap_or(0.0)` this one says so.
        // Timed the way the sorted stage times it, so a cost comparison
        // between the two is a comparison and not two different measurements.
        let started = std::time::Instant::now();
        let scores = reranker
            .rerank_batch(query, &documents)
            .unwrap_or_else(|e| {
                tracing::warn!("rerank lane unavailable: {e:#}");
                vec![0.0; documents.len()]
            });
        tracing::debug!(
            candidates = documents.len(),
            elapsed_us = started.elapsed().as_micros() as u64,
            "rerank lane voted"
        );

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

    // --- Step 4: Temporal lane (5th lane) when date_range is present ---
    let final_fused = if let Some(range) = &date_range {
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
    // operation with a cap of one, which is what knapper did before chunks were
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
        .map(|f| build_result(config.store, f, config.rerank, f.rrf_score))
        .collect();

    Ok(finalize_search_output(
        results,
        final_fused,
        false,
        trace,
        config.fts.columns(),
        config.ranking.coalesce_adjacent,
    ))
}

/// A `FusedResult` as the rerank lane needs it.
fn target_of(result: &FusedResult) -> RerankTarget<'_> {
    RerankTarget {
        file_id: result.file_id,
        chunk_seq: result.chunk_seq,
        file_path: &result.file_path,
        snippet: &result.snippet,
    }
}

/// The sort lane's own score for a result, if one ran — the cross-encoder's
/// `"rerank"` or the model-free `"calibrated"` logistic (spec 2026-08-30).
/// Whichever one sorted the pool, `build_result`'s score reads it here.
fn model_score(result: &FusedResult) -> Option<f64> {
    result
        .lane_contributions
        .iter()
        .find(|l| l.lane_name == "rerank" || l.lane_name == "calibrated")
        .map(|l| l.raw_score)
}

/// Build the search output. Apply the adjacent-chunk merge once, when the
/// key is on (#39). Both ranking stages converge here. So the pass runs
/// whether the cross-encoder is present or absent. `fused` stays per-chunk.
/// It is the `--explain` record of the fusion step. It is not the presented
/// result.
fn finalize_search_output(
    results: Vec<InternalSearchResult>,
    fused: Vec<FusedResult>,
    degraded: bool,
    trace: RetrievalTrace,
    fts_columns: Vec<(&'static str, f64)>,
    coalesce: bool,
) -> SearchOutput {
    let results = if coalesce {
        crate::coalesce::coalesce_adjacent(results)
    } else {
        results
    };
    SearchOutput {
        results,
        fused,
        degraded,
        retrieval: trace,
        fts_columns,
    }
}

/// Build the contract row for one fused result (#35).
///
/// The window is the one the sorted stage threaded onto the candidate, when it
/// threaded one. The legacy and degraded paths score no window, so the capped
/// chunk text is re-derived here instead and its count is the `3.33`
/// estimate — that is what those two paths are documented to emit.
/// `heading_path` is metadata, not part of the window invariant, so it is read
/// from the store here rather than carried on `FusedResult`.
///
/// `score` is supplied by the caller rather than derived here, because the two
/// ranking stages read it differently: the sorted stage reports the probability
/// its scorer produced — the cross-encoder's, or the calibrated logistic's when
/// no model is configured — and the legacy stage the fused RRF score it has
/// always been the byte-for-byte control for.
fn build_result(
    store: &Store,
    f: &FusedResult,
    rerank: crate::config::RerankConfig,
    score: f64,
) -> InternalSearchResult {
    let row = store
        .get_chunk_by_seq(f.file_id, f.chunk_seq)
        .ok()
        .flatten();
    let heading_path = row
        .as_ref()
        .map(|c| c.heading_path.clone())
        .unwrap_or_default();
    let (text, token_count, truncated) = match &f.emit_text {
        Some(body) => (
            body.clone(),
            f.emit_token_count
                .unwrap_or_else(|| est_tokens_fallback(body)),
            f.emit_truncated,
        ),
        None => {
            let raw = row.map(|c| c.text).unwrap_or_default();
            let raw_chars = raw.chars().count();
            let body = truncate_chars(raw, rerank.max_document_chars);
            let truncated = rerank.max_document_chars > 0 && body.chars().count() < raw_chars;
            let count = est_tokens_fallback(&body);
            (body, count, truncated)
        }
    };
    InternalSearchResult {
        file_path: f.file_path.clone(),
        file_id: f.file_id,
        chunk_seq: f.chunk_seq,
        score,
        confidence: f.confidence,
        heading: f.heading.clone(),
        snippet: f.snippet.clone(),
        docid: f.docid.clone(),
        text,
        heading_path,
        token_count,
        truncated,
        provenance: crate::packaging::Provenance::derive(&f.lane_contributions, f.graph_provenance),
    }
}

/// The ranking stage of issue #30: the graph reaches the scorer by reserved
/// quota, and the scorer sorts what reaches it.
///
/// Which scorer that is depends on the build. A configured cross-encoder sorts
/// the shortlist, as it always has. With no cross-encoder the calibrated
/// logistic sorts it instead, from the two content lanes' own scores
/// (spec 2026-08-30) — which is why `query_vec` comes in: the backfill needs
/// the query's embedding to give a candidate no content lane fetched a cosine.
///
/// Returns the ranked results, whether the order is the degraded one, and the
/// calibrated trace when the calibrated scorer is what ran (`None` for a
/// configured cross-encoder, and for the degraded interleave).
#[allow(clippy::too_many_arguments)]
fn sorted_stage(
    query: &str,
    query_vec: &[f32],
    config: &mut SearchConfig<'_>,
    semantic_results: &[RankedResult],
    fts_results: &[RankedResult],
    graph_results: &[RankedResult],
    weights: &crate::config::LaneWeights,
    date_range: Option<(i64, i64)>,
    rrf_k: usize,
) -> (Vec<FusedResult>, bool, Option<CalibratedTrace>) {
    // Two lanes, not three. The graph is a candidate generator and not a
    // scorer (§3), so it no longer votes here — `weights.graph` is read by the
    // legacy stage and by nothing else.
    let content_fused = fusion::rrf_fuse(
        &[
            ("semantic", semantic_results, weights.semantic),
            ("fts", fts_results, weights.fts),
        ],
        rrf_k,
    );

    let temporal_keys = match date_range {
        Some(range) => temporal_promotions(config.store, range, &content_fused, graph_results),
        None => Vec::new(),
    };
    let reserves = ranking::Reserves {
        budget: config.ranking.candidates,
        graph: config.ranking.graph_reserve,
        // A reserve for a source with nothing to say is a slot taken from the
        // content lanes for no reason.
        temporal: if temporal_keys.is_empty() {
            0
        } else {
            config.ranking.temporal_reserve
        },
    };

    let (mut pool, shape) =
        ranking::build_pool(content_fused, graph_results, &temporal_keys, reserves);
    // Pool starvation, weak generation and model rejection all end in "the
    // graph contributed nothing", and the result list cannot tell them apart
    // (#30, §13.5). These are the first two of the three counts; the third is
    // logged after the sort.
    tracing::debug!(
        budget = shape.budget,
        admitted = shape.admitted,
        rrf_available = shape.rrf_available,
        graph_available = shape.graph_available,
        from_rrf = shape.from_rrf,
        from_graph = shape.from_graph,
        from_temporal = shape.from_temporal,
        backfilled = shape.backfilled,
        graph_only = shape.graph_only,
        "shortlist assembled"
    );

    // Which scorer fills `rerank_score`. One slot, two writers: the sort, the
    // floor and the confidence all read the same field, whichever one ran.
    //
    // Set only by the calibrated arm below, so a configured cross-encoder and
    // the degraded interleave both report `None` — there is no bound or idf
    // to show for either.
    let mut calibrated_trace: Option<CalibratedTrace> = None;
    let scored = if let Some(reranker) = &mut config.reranker {
        let targets: Vec<RerankTarget<'_>> = pool
            .iter()
            .map(|c| RerankTarget {
                file_id: c.file_id,
                chunk_seq: c.chunk_seq,
                file_path: &c.file_path,
                snippet: &c.snippet,
            })
            .collect();
        let units = rerank_documents(config.store, &targets, config.rerank);
        let documents: Vec<&str> = units.iter().map(|u| u.document.as_str()).collect();
        drop(targets);

        let started = std::time::Instant::now();
        let scores = reranker.rerank_batch(query, &documents);
        let elapsed_us = started.elapsed().as_micros() as u64;
        let candidate_count = documents.len();
        drop(documents);
        match scores {
            Ok(scores) => {
                for (candidate, (unit, score)) in pool.iter_mut().zip(units.into_iter().zip(scores))
                {
                    candidate.rerank_score = Some(score as f64);
                    candidate.emit_token_count = Some(reranker.count_tokens(&unit.body));
                    candidate.emit_text = Some(unit.body);
                    candidate.emit_truncated = unit.truncated;
                }
                tracing::debug!(
                    candidates = candidate_count,
                    elapsed_us,
                    "cross-encoder sorted the shortlist"
                );
                true
            }
            Err(e) => {
                // The whole ordering, not one candidate. Scoring everything 0.0
                // would leave the tie-break deciding the entire ranking while
                // the output claimed a model had judged it.
                tracing::warn!("rerank lane unavailable, ordering degraded: {e:#}");
                false
            }
        }
    } else {
        // No model is configured here — either none was asked for, or one was
        // asked for and did not load. Routing admitted this build because
        // `[calibrated] enabled`, so the logistic sorts where the model would
        // (spec 2026-08-30). The Err arm above keeps the interleave: a model
        // that loaded and then failed at call time is a different finding from
        // no model at all.
        let (bound, idfs) = apply_calibrated_scores(
            config.store,
            &config.calibrated,
            query,
            query_vec,
            &config.fts.weights(),
            semantic_results,
            fts_results,
            &mut pool,
        );
        tracing::debug!(bound, terms = idfs.len(), "calibrated scores assembled");
        calibrated_trace = Some(CalibratedTrace { bound, idfs });
        true
    };

    if !scored {
        let ordered = ranking::degraded_interleave(pool);
        let mut results = ranking::into_fused(ordered, "rerank");
        ranking::degraded_confidence(&mut results);
        return (results, true, None);
    }

    ranking::sort_by_rerank(&mut pool, config.ranking.tiebreak);

    // The answer floor (#34). It runs here and not after `into_fused`, because
    // this is the last point where `rerank_score` is still an `Option`. The
    // floor must tell "the model gave this a low score" apart from "nothing
    // scored this", and `confidence` reduces both to one f64.
    //
    // Each scorer brings its own floor, because each was fit against its own
    // scores: `[ranking] answer_floor` for the cross-encoder, `[calibrated]
    // floor` for the logistic. One number cannot mean both.
    let calibrated_ran = config.reranker.is_none();
    let floor = if calibrated_ran {
        config.calibrated.floor
    } else {
        config.ranking.answer_floor
    };
    let supported = pool.len();
    let dropped = ranking::apply_answer_floor(&mut pool, floor);
    if dropped > 0 {
        // The cost per query. The fit uses the best score of each query, and
        // the floor applies to a whole list, so how much of the list it removes
        // is a separate measurement from whether it returns nothing correctly.
        tracing::debug!(
            floor,
            scored = supported,
            dropped,
            surviving = pool.len(),
            "answer floor"
        );
    }

    // The third count: of the candidates only the graph found, how many the
    // model kept near the top. Distinguishes "the reserve fed it junk" from
    // "the reserve never got the chance".
    tracing::debug!(
        graph_only_admitted = shape.graph_only,
        graph_only_in_top_10 = pool.iter().take(10).filter(|c| c.graph_only()).count(),
        best_graph_only = pool
            .iter()
            .filter(|c| c.graph_only())
            .filter_map(|c| c.rerank_score)
            .fold(f64::NAN, f64::max),
        best_overall = pool
            .first()
            .and_then(|c| c.rerank_score)
            .unwrap_or(f64::NAN),
        "graph reserve, after the sort"
    );

    // Name the lane that sorted, so `--explain` does not call the logistic a
    // cross-encoder.
    let sort_lane = if calibrated_ran {
        "calibrated"
    } else {
        "rerank"
    };
    (
        ranking::into_fused(pool, sort_lane),
        false,
        calibrated_trace,
    )
}

/// Compute the calibrated probability for every pool candidate and store it in
/// `rerank_score` — the slot the cross-encoder writes, so the sort, the floor
/// and the confidence all read one field (spec 2026-08-30).
///
/// Features come from the lanes when the lanes fetched the candidate, and from
/// the store when they did not (a graph or temporal admission): one vector
/// fetch and dot product for the cosine, one scoped FTS query for the BM25.
/// The dot product is the cosine because the embedders L2-normalize what they
/// return, which is the same assumption `search_vec`'s distance rests on.
///
/// Returns the query's BM25 upper bound and per-term idfs for the trace.
#[allow(clippy::too_many_arguments)]
fn apply_calibrated_scores(
    store: &Store,
    params: &crate::config::CalibratedConfig,
    query: &str,
    query_vec: &[f32],
    fts_weights: &[f64],
    semantic_results: &[RankedResult],
    fts_results: &[RankedResult],
    pool: &mut [ranking::Candidate],
) -> (f64, Vec<(String, f64)>) {
    let mut cos: HashMap<(i64, i64), f64> = semantic_results
        .iter()
        .map(|r| ((r.file_id, r.chunk_seq), r.score))
        .collect();
    let mut bm25: HashMap<(i64, i64), f64> = fts_results
        .iter()
        .map(|r| ((r.file_id, r.chunk_seq), r.score))
        .collect();

    // Backfill for candidates no content lane fetched. The pool is ~40 rows,
    // so both reads are noise beside the query embed that already ran.
    let missing_cos: Vec<i64> = pool
        .iter()
        .filter(|c| !cos.contains_key(&c.key()))
        .map(|c| c.file_id)
        .collect();
    if !missing_cos.is_empty() {
        match store.vectors_for_files(&missing_cos) {
            Ok(rows) => {
                for (file_id, seq, v) in rows {
                    let dot: f64 = v
                        .iter()
                        .zip(query_vec)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    cos.entry((file_id, seq)).or_insert(dot);
                }
            }
            // The candidates keep their slot and lose a whole feature: with no
            // vector the cosine reads as 0.0 and only the keyword lane can
            // carry them. A lost feature is a ranking change, so it is logged.
            Err(e) => tracing::warn!(
                "calibrated cosine backfill unavailable for {} candidates: {e:#}",
                missing_cos.len()
            ),
        }
    }
    let missing_bm25: Vec<i64> = pool
        .iter()
        .filter(|c| !bm25.contains_key(&c.key()))
        .map(|c| c.file_id)
        .collect();
    if !missing_bm25.is_empty() {
        match store.fts_search_any(
            query,
            pool.len().max(1) * 4,
            fts_weights,
            Some(&missing_bm25),
        ) {
            Ok(rows) => {
                for r in rows {
                    bm25.entry((r.file_id, r.chunk_seq)).or_insert(r.score);
                }
            }
            // The other half of the same loss: no keyword evidence for these
            // candidates, so the cosine alone decides where they sort.
            Err(e) => tracing::warn!(
                "calibrated keyword backfill unavailable for {} candidates: {e:#}",
                missing_bm25.len()
            ),
        }
    }

    // The query's own ceiling (spec, "The two normalizations"). The `[fts]`
    // column weights do not enter it: each term's contribution saturates at
    // `idf · (k1 + 1)`, and FTS5 folds a weight into the frequency before that
    // ratio, so a heavier weight moves the score toward the same ceiling
    // instead of raising it.
    //
    // A failed row count is **not** an empty corpus. `idf(0, df)` clamps to the
    // floor for every term, which collapses the bound and saturates every
    // keyword match at `bm25n = 1.0` — a failed `SELECT count(*)` would read as
    // maximal lexical evidence and take the floor's abstention with it. So the
    // failure reads as no bound instead: no idfs, a bound of 0.0, and
    // `normalize_bm25` already answers a non-positive bound with no evidence.
    let n_rows = match store.chunk_row_count() {
        Ok(n) => Some(n),
        Err(e) => {
            tracing::warn!("calibrated bound unavailable, keyword evidence reads as none: {e:#}");
            None
        }
    };
    let idfs: Vec<(String, f64)> = match n_rows {
        None => Vec::new(),
        Some(n_rows) => crate::fts::query_terms(query)
            .into_iter()
            .map(|t| {
                let df = match store.fts_doc_frequency(&crate::fts::phrase_expr(&t)) {
                    Ok(df) => df,
                    Err(e) => {
                        // Conservative in the safe direction: 0 is the rarest a
                        // term can be, which raises the bound and lowers every
                        // `bm25n` measured against it.
                        tracing::warn!(
                            "doc frequency unavailable for {t:?}, term reads as rare: {e:#}"
                        );
                        0
                    }
                };
                let idf = crate::calibrate::idf(n_rows, df);
                (t, idf)
            })
            .collect(),
    };
    let bound = crate::calibrate::upper_bound(&idfs.iter().map(|(_, i)| *i).collect::<Vec<_>>());

    for c in pool.iter_mut() {
        let key = c.key();
        let s = cos.get(&key).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let k = crate::calibrate::normalize_bm25(bm25.get(&key).copied().unwrap_or(0.0), bound);
        c.rerank_score = Some(crate::calibrate::probability(s, k, params));
    }
    (bound, idfs)
}

/// Chunk keys whose note falls in the query's date range, best match first.
///
/// Drawn from the candidates retrieval already produced rather than generated:
/// the temporal signal ranks *notes*, and choosing which passage of a note a
/// date match implicates would be putting a number on a guess. Under the sorted
/// stage temporal cannot stay a voter, and no probe covers it — so it becomes
/// the source that promotes date-matching candidates the content order cut, and
/// that choice is recorded as unmeasured rather than argued for.
fn temporal_promotions(
    store: &Store,
    range: (i64, i64),
    content: &[FusedResult],
    graph: &[RankedResult],
) -> Vec<ranking::ChunkKey> {
    let candidates = content
        .iter()
        .map(|c| ((c.file_id, c.chunk_seq), c.file_path.as_str()))
        .chain(
            graph
                .iter()
                .map(|r| ((r.file_id, r.chunk_seq), r.file_path.as_str())),
        );

    let mut dates: HashMap<&str, Option<i64>> = HashMap::new();
    let mut scored: Vec<(ranking::ChunkKey, f64)> = Vec::new();
    let mut seen: std::collections::HashSet<ranking::ChunkKey> = std::collections::HashSet::new();
    for (key, path) in candidates {
        if !seen.insert(key) {
            continue;
        }
        let note_date = *dates.entry(path).or_insert_with(|| {
            store
                .get_file(path)
                .ok()
                .flatten()
                .and_then(|f| f.note_date)
        });
        if let Some(note_date) = note_date {
            scored.push((
                key,
                crate::temporal::temporal_score(note_date, range.0, range.1),
            ));
        }
    }
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().map(|(key, _)| key).collect()
}

/// The bottom of the normalised seed range.
///
/// Not zero. `graph_expand` computes `seed.score * decay` and sorts the result
/// before truncating, so a seed scored 0 is deleted from the graph lane's
/// ordering rather than placed last in it — and the weakest hit of a lane is
/// still a hit.
const SEED_SCORE_FLOOR: f64 = 0.1;

/// Rescale one lane's hits into `[SEED_SCORE_FLOOR, 1.0]` against their own
/// range (issue #26).
///
/// The two lanes fill `RankedResult::score` from incommensurable units —
/// `1.0 - cosine_distance`, measured at 0.2–0.7 on the eval vault, against
/// negated BM25 at 2–17. The ranges do not overlap, so every comparison that
/// mixed them was decided by which lane's unit was larger: `merge_seeds` handed
/// every contested file to FTS, and the ordering that feeds `graph_expand`'s
/// `truncate` was BM25 magnitude rather than anything about the graph.
///
/// Each lane is scaled against *itself*. No raw unit crosses a lane boundary,
/// and the two lanes are never mapped onto a shared range against each other —
/// that would be the same defect with more arithmetic.
///
/// This is deliberately not a rank transform. Rank is outlier-immune, which is
/// why RRF is right at *final* fusion, but it pays for that by discarding
/// magnitude: a lane returning 0.68 and then a cliff to 0.31 becomes "1st, 2nd",
/// and the size of that cliff is exactly what should decide how hard a seed
/// expands.
///
/// A lane with one hit — or N identical ones — has no basis to rank them, so
/// they all map to the top of the range.
///
/// # Why this touches the seed pool and not the fusion pool
///
/// `score` has two consumers and they want opposite things. **Fusion consumes
/// rank** — `rrf_fuse` reads position and never the number — and the rank comes
/// from `collapse_lane` sorting the lane's own scores, which normalising would
/// not change but would obscure. **Seeding consumes magnitude**: `graph_expand`
/// computes
/// `seed.score * decay`, sorts, and truncates, so the two lanes have to be on
/// one scale or the bigger unit wins every time.
///
/// This was measured, not reasoned. Normalising the pool that feeds fusion —
/// the shape #26 originally specified — moved three of six probe targets down,
/// including `archdragon.md` 3 → 6 on probe 4, which exists precisely to catch
/// BM25 regressions. Normalising only the seed pool leaves all four real probes
/// byte-identical while the seed set still becomes a mix. See `eval/probes.md`.
fn normalise_lane_scores(results: &mut [RankedResult]) {
    let Some(max) = results.iter().map(|r| r.score).reduce(f64::max) else {
        return;
    };
    let min = results
        .iter()
        .map(|r| r.score)
        .reduce(f64::min)
        .unwrap_or(max);
    let span = max - min;
    for r in results.iter_mut() {
        r.score = if span > 0.0 {
            SEED_SCORE_FLOOR + (1.0 - SEED_SCORE_FLOOR) * (r.score - min) / span
        } else {
            1.0
        };
    }
}

/// `normalise_lane_scores` against a copy, for the pool that needs both.
fn normalised(hits: &[RankedResult]) -> Vec<RankedResult> {
    let mut out = hits.to_vec();
    normalise_lane_scores(&mut out);
    out
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

/// Render the retrieval step for `--explain`: the query that was actually run,
/// and what each lane returned for it.
///
/// Hit counts are pre-collapse — what the lane read, before the per-file cap.
/// That is the number that answers "was anything retrieved at all", which is
/// the question these three tickets kept turning out to hinge on.
fn format_retrieval(trace: &RetrievalTrace, columns: &[(&str, f64)]) -> String {
    let declared = columns
        .iter()
        .map(|(name, weight)| format!("{name}·{weight}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut out = format!("--- Query run ---\nkeyword index: {declared}\n");
    if let Some(scope) = &trace.scope {
        out.push_str(&format!(
            "scope: {} -> {} notes\n",
            scope.filter, scope.notes
        ));
    }
    out.push_str(&format!(
        "{:?}\n   semantic {} · fts {}",
        trace.query, trace.semantic_hits, trace.fts_hits
    ));
    match &trace.fts_expr {
        Some(expr) => out.push_str(&format!("  ← {expr}\n")),
        None => out.push_str("  ← (no searchable term; fts skipped)\n"),
    }
    // The query-level half of the calibrated scorer (spec 2026-08-30): the
    // per-candidate probability already prints as the `"calibrated"` lane
    // below, in the fusion detail. This line is the bound and idfs it was
    // computed from, absent on every other path.
    if let Some(cal) = &trace.calibrated {
        let idfs = cal
            .idfs
            .iter()
            .map(|(t, i)| format!("idf({t})={i:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&format!("calibrated: bound={:.2}  {}\n", cal.bound, idfs));
    }
    out.push('\n');
    out
}

/// Merge semantic and FTS seed results, keeping the highest score per file.
///
/// "Highest" only means something because both lanes arrive normalised to
/// `[SEED_SCORE_FLOOR, 1.0]` against their own ranges (#26). The comparison
/// below reads as "which lane placed this file higher within its own results",
/// which is a question about the file. Before #26 it read as "is negated BM25
/// bigger than a cosine similarity", which is a question about the units, and
/// the answer was always yes.
fn merge_seeds(semantic: &[RankedResult], fts: &[RankedResult]) -> Vec<RankedResult> {
    let labelled = semantic
        .iter()
        .map(|r| (r, "semantic"))
        .chain(fts.iter().map(|r| (r, "fts")));

    let mut by_file: HashMap<String, (RankedResult, &'static str)> = HashMap::new();
    for (r, lane) in labelled {
        let dominated = by_file
            .get(&r.file_path)
            .is_some_and(|(existing, _)| existing.score >= r.score);
        if !dominated {
            by_file.insert(r.file_path.clone(), (r.clone(), lane));
        }
    }
    // Seeds drive graph expansion; a stable order keeps that reproducible.
    let mut seeds: Vec<(RankedResult, &'static str)> = by_file.into_values().collect();
    seeds.sort_by(|(a, _), (b, _)| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_key().cmp(&b.chunk_key()))
    });

    // Which lane won, and won the top of the order. #26's whole case was a
    // measurement of exactly this — 10 of the top 10 seeds from FTS on three of
    // three real queries — and it took direct SQL to get, because nothing in
    // the pipeline reported it.
    tracing::debug!(
        seeds = seeds.len(),
        fts_won = seeds.iter().filter(|(_, l)| *l == "fts").count(),
        top10_fts = seeds.iter().take(10).filter(|(_, l)| *l == "fts").count(),
        "seed set assembled"
    );

    seeds.into_iter().map(|(r, _)| r).collect()
}

/// Run a search query and print the envelope (#35).
///
/// Performs both semantic (sqlite-vec) and keyword (FTS5) search, then fuses
/// results using Reciprocal Rank Fusion, packages the ranked results into a
/// `SearchEnvelope`, and renders it: JSON for `--json`, text otherwise. When
/// `explain` is true, the per-lane score breakdown prints after the text
/// rendering — clap's `conflicts_with` keeps it from ever running with `json`.
#[allow(clippy::too_many_arguments)]
pub fn run_search(
    query: &str,
    top_n: usize,
    json: bool,
    explain: bool,
    budget_tokens: Option<u32>,
    full: bool,
    summaries: bool,
    scores: bool,
    group_by: GroupBy,
    scope: &crate::tags::Scope,
    data_dir: &Path,
    config: &crate::config::Config,
) -> Result<()> {
    let models_dir = data_dir.join("models");
    let mut embedder =
        crate::llm::load_embedder(&models_dir, config).context("loading embedder")?;

    let db_path = db_path(data_dir);
    let store = Store::open(&db_path).context("opening store")?;
    store.verify_embedding_dim(embedder.dim())?;

    // Load the cross-encoder if enabled.
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

    // Refuse to answer from an index this build did not produce (issue #31).
    // A stale index and a real effect are indistinguishable in a result list,
    // which is the whole reason the check is here rather than in a log line.
    {
        let fingerprints = crate::fingerprint::Fingerprints::compute(
            config,
            &llm::EmbedModel::fingerprint(&embedder),
            reranker_model.as_ref().map(|r| r.fingerprint()).as_deref(),
        );
        crate::fingerprint::verify(&store, &fingerprints)?;
    }

    let output = {
        let mut search_config = SearchConfig {
            reranker: reranker_model
                .as_mut()
                .map(|r| r.as_mut() as &mut dyn llm::RerankModel),
            group_by,
            scope: scope.clone(),
            ..SearchConfig::new(&store, config)
        };
        search_with_intelligence(query, top_n, &mut embedder, &mut search_config)?
    };

    // `full` and `summaries` both name the whole result set and disagree on
    // its shape, so asking for both is a usage error rather than one flag
    // silently winning (#35).
    if full && summaries {
        anyhow::bail!("--full and --summaries are mutually exclusive");
    }

    // Per call, with the configured default behind it, the same pattern
    // `top_n` follows (#35, #62).
    let budget = budget_tokens.unwrap_or(config.output.budget_tokens);
    let mut env = crate::packaging::assemble(
        &output.results,
        crate::packaging::AssembleParams {
            budget_tokens: budget,
            full,
            summaries,
            degraded: output.degraded,
            per_note_cap: config.ranking.per_note_cap,
        },
    );
    // A number invites a caller to trust it as ground truth rather than as a
    // reranker's opinion, so it ships only when asked (#35).
    if scores {
        crate::packaging::apply_scores(&mut env, &output.results);
    }

    // JSON is the CLI's machine channel and carries the envelope alone; the
    // text rendering — including the degraded banner — is design §9.3's form
    // and is what `--explain` appends to (#35, #62).
    let mut out = if json {
        format!("{}\n", serde_json::to_string_pretty(&env)?)
    } else {
        crate::packaging::render_text(&env, scores)
    };

    if explain && !json {
        out.push_str(&explain_report(&output, top_n));
    }

    print!("{out}");
    Ok(())
}

/// The `--explain` record: the retrieval step, then the fusion step per result.
///
/// It reads the retrieval trace and the fused lanes, which no caller of the
/// pipeline holds by any other route, so the one composition serves all three
/// surfaces: the CLI prints it, MCP sends it as a second content block and the
/// HTTP envelope carries it in `explain` (#62). It is built only when the
/// caller asks for it, because an agent that did not ask must not read past it.
pub fn explain_report(output: &SearchOutput, top_n: usize) -> String {
    let mut out = format_retrieval(&output.retrieval, &output.fts_columns);
    out.push_str("--- Explain ---\n");
    for f in output.fused.iter().take(top_n) {
        out.push_str(&format!("{}\n", f.file_path));
        out.push_str(&fusion::format_explain(f));
    }
    out
}

/// What `status` reports, read from the store and from the config.
///
/// The CLI prints it and the two servers answer it as JSON, so all three read
/// one gatherer and one field list (#62).
struct StatusInputs {
    stats: StoreStats,
    edges: EdgeStats,
    index_size: u64,
    model_name: String,
    intelligence: &'static str,
    date_count: usize,
}

/// The status fields, read from a store the caller supplies.
///
/// The three reads are separate statements, so a writer that commits between
/// them would give an answer mixing two snapshots. A server passes the store
/// it already holds the lock on, which is what stops that (#62).
fn collect_status(store: &Store, data_dir: &Path) -> Result<StatusInputs> {
    let stats = store.stats()?;
    let edges = store.get_edge_stats()?;
    let date_count = store.count_files_with_dates().unwrap_or(0);

    // Compute index size on disk (sqlite db file).
    let index_size = std::fs::metadata(db_path(data_dir))
        .map(|m| m.len())
        .unwrap_or(0);

    let config = crate::config::Config::load().unwrap_or_default();
    Ok(StatusInputs {
        stats,
        edges,
        index_size,
        model_name: crate::llm::embed_model_display(&config),
        intelligence: if config.intelligence_enabled() {
            "enabled"
        } else {
            "disabled"
        },
        date_count,
    })
}

/// Run the status command and print index information. The CLI holds no
/// store, so this opens one.
pub fn run_status(json: bool, data_dir: &Path) -> Result<()> {
    let store = Store::open(&db_path(data_dir)).context("opening store")?;
    let s = collect_status(&store, data_dir)?;
    let output = format_status(
        &s.stats,
        &s.edges,
        s.index_size,
        &s.model_name,
        s.intelligence,
        s.date_count,
        json,
    );
    print!("{output}");
    Ok(())
}

/// What `status` reports, as the object the JSON channel names.
///
/// `run_status` prints; this is what the two servers answer with, and they
/// pass the store they already hold rather than opening a second connection:
/// `Store::open` runs the schema batch and the migrations again, which would
/// wait out the busy timeout against the server's own writer and then fail
/// the call. Both routes compose the fields through `status_object`, so the
/// three surfaces report the same ones and cannot drift apart (#62).
pub fn status_json(store: &Store, data_dir: &Path) -> Result<serde_json::Value> {
    let s = collect_status(store, data_dir)?;
    Ok(status_object(
        &s.stats,
        &s.edges,
        s.index_size,
        &s.model_name,
        s.intelligence,
        s.date_count,
    ))
}

/// The fields `status` reports, composed once (pure function, no I/O).
///
/// The CLI's `--json`, the MCP tool and the HTTP route all answer this
/// object, so a field added here reaches the three surfaces together (#62).
pub fn status_object(
    stats: &StoreStats,
    edges: &EdgeStats,
    index_size: u64,
    model_name: &str,
    intelligence: &str,
    date_count: usize,
) -> serde_json::Value {
    let vault = stats.vault_path.as_deref().unwrap_or("<not set>");
    let last_indexed = stats.last_indexed_at.as_deref().unwrap_or("never");

    // `edges` is the one source for every edge number. `files` above already
    // says how many files the index holds, so a second `total_files` derived
    // from the connectivity split would be the same number twice (#62).
    json!({
        "vault": vault,
        "files": stats.file_count,
        "chunks": stats.chunk_count,
        "tombstones": stats.tombstone_count,
        "last_indexed": last_indexed,
        "index_size": index_size,
        "model": model_name,
        "intelligence": intelligence,
        "files_with_dates": date_count,
        "edges": edges.total_edges,
        "wikilink_edges": edges.wikilink_count,
        "wikilink_pairs": edges.wikilink_count / 2,
        "connected_files": edges.connected_file_count,
        "isolated_files": edges.isolated_file_count,
    })
}

/// Format status information for display (pure function, no I/O).
///
/// `edges` folds in `graph stats` (#62): `status` is what answers "what is in
/// the index", so the connectivity counts belong beside the file and chunk
/// counts rather than behind a second command.
pub fn format_status(
    stats: &StoreStats,
    edges: &EdgeStats,
    index_size: u64,
    model_name: &str,
    intelligence: &str,
    date_count: usize,
    json: bool,
) -> String {
    let vault = stats.vault_path.as_deref().unwrap_or("<not set>");
    let last_indexed = stats.last_indexed_at.as_deref().unwrap_or("never");
    let wikilink_pairs = edges.wikilink_count / 2;
    let total_files = edges.connected_file_count + edges.isolated_file_count;
    let connected_pct = if total_files > 0 {
        edges.connected_file_count as f64 / total_files as f64 * 100.0
    } else {
        0.0
    };

    if json {
        let obj = status_object(
            stats,
            edges,
            index_size,
            model_name,
            intelligence,
            date_count,
        );
        format!("{}\n", serde_json::to_string_pretty(&obj).unwrap())
    } else {
        // The `Edges:` header and the four lines under it come from one
        // `EdgeStats`, so the header is always printed and the indented lines
        // always have one to sit under (#62).
        let mut out = format!(
            "Vault:      {}\n\
             Files:      {}\n\
             Chunks:     {}\n\
             Edges:      {}\n",
            vault, stats.file_count, stats.chunk_count, edges.total_edges,
        );
        out.push_str(&format!(
            "  Wikilink edges:  {} ({} bidirectional pairs)\n",
            edges.wikilink_count, wikilink_pairs
        ));
        out.push_str(&format!(
            "  Connected files: {} / {} ({:.1}%)\n",
            edges.connected_file_count, total_files, connected_pct
        ));
        out.push_str(&format!(
            "  Isolated files:  {}\n",
            edges.isolated_file_count
        ));
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

    fn sample_edge_stats() -> EdgeStats {
        EdgeStats {
            total_edges: 10,
            wikilink_count: 6,
            connected_file_count: 8,
            isolated_file_count: 2,
        }
    }

    #[test]
    fn test_format_status_human() {
        let stats = StoreStats {
            file_count: 42,
            chunk_count: 187,
            tombstone_count: 3,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
        };
        let output = format_status(
            &stats,
            &sample_edge_stats(),
            2_516_582,
            "all-MiniLM-L6-v2",
            "disabled",
            30,
            false,
        );

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

    /// `status` absorbs `graph stats` (#62): the connectivity counts that
    /// used to need a second command are printed here.
    #[test]
    fn status_reports_the_edge_counts_graph_stats_used_to_own() {
        let stats = StoreStats {
            file_count: 10,
            chunk_count: 20,
            tombstone_count: 0,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
        };
        let output = format_status(
            &stats,
            &sample_edge_stats(),
            2_516_582,
            "all-MiniLM-L6-v2",
            "disabled",
            10,
            false,
        );

        assert!(
            output.contains("Wikilink edges:"),
            "graph stats's wikilink line is missing: {output}"
        );
        assert!(output.contains("6 (3 bidirectional pairs)"));
        assert!(output.contains("Connected files: 8 / 10 (80.0%)"));
        assert!(output.contains("Isolated files:  2"));
    }

    #[test]
    fn test_format_status_json() {
        let stats = StoreStats {
            file_count: 42,
            chunk_count: 187,
            tombstone_count: 3,
            last_indexed_at: Some("2026-03-19 14:30:00".to_string()),
            vault_path: Some("/path/to/vault".to_string()),
        };
        let output = format_status(
            &stats,
            &sample_edge_stats(),
            2_516_582,
            "all-MiniLM-L6-v2",
            "enabled",
            30,
            true,
        );
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
        // The printed and JSON views report the same numbers.
        assert_eq!(parsed["wikilink_pairs"], 3);
        assert_eq!(parsed["connected_files"], 8);
        assert_eq!(parsed["isolated_files"], 2);
        // `files` is the file count; a second `total_files` derived from the
        // connectivity split said the same thing twice (#62).
        assert!(parsed.get("total_files").is_none(), "got {parsed}");
        // The edge numbers have one source now, and the JSON always carries
        // them.
        assert_eq!(parsed["edges"], 10);
        assert_eq!(parsed["wikilink_edges"], 6);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(2_516_582), "2.4 MB");
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
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    /// Run the pipeline with default settings and no intelligence models.
    fn heuristic_search(
        query: &str,
        top_n: usize,
        store: &Store,
        embedder: &mut impl EmbedModel,
        group_by: GroupBy,
    ) -> SearchOutput {
        let mut config = SearchConfig {
            reranker: None,
            store,
            rerank_candidates: 30,
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: crate::config::default_max_chunks_per_file(),
            group_by,
            fts: crate::config::FtsConfig::default(),
            // These helpers test the ranking pipeline. The ranking pipeline
            // is below coalescing. Coalescing has its own tests (#39).
            ranking: crate::config::RankingConfig {
                coalesce_adjacent: false,
                ..crate::config::RankingConfig::default()
            },
            lane_weights: crate::config::LaneWeights::default(),
            scope: crate::tags::Scope::default(),
            // These tests assert the pre-calibration paths; the calibrated
            // sort has its own tests (Task 7).
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        search_with_intelligence(query, top_n, embedder, &mut config).unwrap()
    }

    #[test]
    fn one_document_can_contribute_several_sections() {
        // #6's acceptance criterion. Three sections of one file match "warding";
        // before chunk-level fusion only one of them could ever be returned.
        let (_tmp, store, mut embedder) = indexed_vault();

        let output = heuristic_search("warding", 10, &store, &mut embedder, GroupBy::Chunk);

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

    /// A vault whose one file is a section and its subsections, which is the
    /// unit coalescing merges since #101: the sibling sections of
    /// `indexed_vault` are separate topics and no longer form one block.
    /// Every body clears `chunk_min_chars`, so each section is its own chunk.
    fn nested_vault() -> (tempfile::TempDir, Store, crate::llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("rules")).unwrap();
        std::fs::write(
            root.join("rules/warding-school.md"),
            "## Warding School

The school of warding covers every effect that blocks,              ends or pins magic, and the entries below inherit their rules from this              section rather than restating them each time.

             ### Counterspell

A warding effect that stops a spell mid-cast.              It interrupts the casting itself and does nothing to a spell already in effect,              which is what separates it from the entry below.

             ### Dispel Magic

A warding effect that ends an ongoing spell.              It reaches an effect already in place and cannot interrupt one              that is still being cast, which is the whole of the difference.
",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    #[test]
    fn coalescing_runs_with_no_cross_encoder() {
        // A section and its two subsections are one block, and the pass ran
        // with reranker: None — so it does not depend on the model.
        let (_tmp, store, mut embedder) = nested_vault();
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: None,
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: crate::config::default_max_chunks_per_file(),
            group_by: GroupBy::Chunk,
            ranking: crate::config::RankingConfig {
                coalesce_adjacent: true,
                ..crate::config::RankingConfig::default()
            },
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let out = search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();
        let warding: Vec<&InternalSearchResult> = out
            .results
            .iter()
            .filter(|r| r.file_path == "rules/warding-school.md")
            .collect();
        assert_eq!(
            warding.len(),
            1,
            "a section and its subsections are one block: {:#?}",
            out.results
        );
        assert_eq!(
            warding[0].heading_path, "rules/warding-school.md > Warding School",
            "headed by the section that contains the rest"
        );
        assert!(warding[0].text.contains("school of warding"));
        assert!(warding[0].text.contains("Counterspell"));
        assert!(warding[0].text.contains("Dispel Magic"));
    }

    #[test]
    fn coalescing_off_is_the_per_section_control() {
        let (_tmp, store, mut embedder) = indexed_vault();
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: None,
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: crate::config::default_max_chunks_per_file(),
            group_by: GroupBy::Chunk,
            ranking: crate::config::RankingConfig {
                coalesce_adjacent: false,
                ..crate::config::RankingConfig::default()
            },
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let out = search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();
        let count = out
            .results
            .iter()
            .filter(|r| r.file_path == "rules/abjuration-spells.md")
            .count();
        assert!(count > 1, "with coalescing off each section is its own row");
    }

    #[test]
    fn coalescing_preserves_the_top_score_with_a_reranker() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let top_score = |coalesce: bool, embedder: &mut llm::MockLlm| {
            let mut reranker = llm::MockLlm::new(8);
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: sorted_config(crate::config::RankingConfig {
                    coalesce_adjacent: coalesce,
                    ..crate::config::RankingConfig::default()
                }),
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            let out = search_with_intelligence("warding", 20, embedder, &mut config).unwrap();
            out.results.first().map(|r| r.score)
        };

        let on = top_score(true, &mut embedder);
        let off = top_score(false, &mut embedder);
        assert_eq!(
            on, off,
            "coalescing keeps the anchor's score, so the top score is unchanged"
        );
    }

    #[test]
    fn group_by_file_makes_coalescing_a_no_op() {
        // `group_by = "file"` collapses each file to one section before
        // coalescing runs. So coalescing finds nothing adjacent to merge.
        // The result must be the same with coalescing on or off (design
        // doc, "Composition and control").
        let (_tmp, store, mut embedder) = indexed_vault();

        let run = |coalesce_adjacent, embedder: &mut llm::MockLlm| {
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: None,
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: crate::config::default_max_chunks_per_file(),
                group_by: GroupBy::File,
                ranking: crate::config::RankingConfig {
                    coalesce_adjacent,
                    ..crate::config::RankingConfig::default()
                },
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            search_with_intelligence("warding", 10, embedder, &mut config)
                .unwrap()
                .results
        };

        let with_coalescing = run(true, &mut embedder);
        let without_coalescing = run(false, &mut embedder);

        let abjuration_with: Vec<&InternalSearchResult> = with_coalescing
            .iter()
            .filter(|r| r.file_path == "rules/abjuration-spells.md")
            .collect();
        let abjuration_without: Vec<&InternalSearchResult> = without_coalescing
            .iter()
            .filter(|r| r.file_path == "rules/abjuration-spells.md")
            .collect();

        assert_eq!(
            abjuration_with.len(),
            1,
            "group_by = file caps one file to one row: {with_coalescing:#?}"
        );
        assert_eq!(
            abjuration_without.len(),
            1,
            "group_by = file caps one file to one row: {without_coalescing:#?}"
        );
        assert_eq!(
            abjuration_with[0].text, abjuration_without[0].text,
            "group_by = file already collapsed the file to one section, \
             so coalescing must change nothing"
        );
    }

    #[test]
    fn per_note_cap_bounds_a_coalesced_block_span() {
        // `per_note_cap` limits how many sections of one file reach the
        // result, above coalescing. So it also limits how many sections a
        // coalesced block can span (design doc, "Composition and control").
        let (_tmp, store, mut embedder) = nested_vault();

        let warding_text = |per_note_cap, embedder: &mut llm::MockLlm| {
            let mut reranker = llm::MockLlm::new(8);
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: sorted_config(crate::config::RankingConfig {
                    coalesce_adjacent: true,
                    per_note_cap,
                    ..crate::config::RankingConfig::default()
                }),
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            let out = search_with_intelligence("warding", 20, embedder, &mut config).unwrap();
            out.results
                .into_iter()
                .find(|r| r.file_path == "rules/warding-school.md")
                .map(|r| r.text)
                .unwrap_or_default()
        };

        let markers = ["school of warding", "Counterspell", "Dispel Magic"];

        // Control: unbounded, the block spans all three sections.
        let unbounded = warding_text(0, &mut embedder);
        let unbounded_count = markers.iter().filter(|m| unbounded.contains(**m)).count();
        assert_eq!(
            unbounded_count, 3,
            "per_note_cap: 0 must let all three sections merge, got: {unbounded}"
        );

        // `per_note_cap: 2` bounds the sections available to merge to two, so
        // the block must span strictly fewer sections than the control.
        let bounded = warding_text(2, &mut embedder);
        let bounded_count = markers.iter().filter(|m| bounded.contains(**m)).count();
        assert!(
            bounded_count < unbounded_count,
            "per_note_cap: 2 must bound the block to fewer sections than the \
             unbounded case, got {bounded_count} markers: {bounded}"
        );
        assert!(
            bounded_count <= 2,
            "per_note_cap: 2 must not let more than 2 sections merge, got: {bounded}"
        );
    }

    /// A vault of one-section files, all carrying the same term, so a lane has
    /// more rows to fetch than any width under test and the per-file collapse
    /// cap never binds.
    fn vault_of_many_files(count: usize) -> (tempfile::TempDir, Store, llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        for i in 0..count {
            std::fs::write(
                root.join(format!("ward-{i:02}.md")),
                format!("# Ward {i}\n\n## Level {i} Ward\n\nA warding effect, number {i}.\n"),
            )
            .unwrap();
        }

        let store = Store::open_memory().unwrap();
        let mut embedder = llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    /// Run a search at one `top_n` and one lane width.
    fn search_at(
        query: &str,
        top_n: usize,
        retrieval_width: usize,
        store: &Store,
        embedder: &mut llm::MockLlm,
    ) -> SearchOutput {
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            reranker: None,
            store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: crate::config::default_max_chunks_per_file(),
            group_by: GroupBy::Chunk,
            ranking: crate::config::RankingConfig {
                retrieval_width,
                ..crate::config::RankingConfig::default()
            },
            scope: crate::tags::Scope::default(),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        search_with_intelligence(query, top_n, embedder, &mut config).unwrap()
    }

    /// Two notes on one subject, one tagged and one not, each with its own
    /// distinctive section, so a scope has something to include and exclude.
    fn tagged_vault() -> (tempfile::TempDir, Store, llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(
            root.join("wight.md"),
            "---\ntags: [type/undead]\n---\n\n\
             # Wight\n\n## Warding\n\nA warding effect that pins an undead creature \
             in the space it stands in, and does not care how it got there.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("wolf.md"),
            "---\ntags: [type/beast]\n---\n\n\
             # Wolf\n\n## Warding\n\nA warding effect that pins a beast in the space \
             it stands in, and does not care how it got there.\n",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    /// Counts the embed calls a search makes, so the short-circuit is asserted
    /// as "no model ran" and not merely as "no results".
    struct CountingEmbed {
        inner: llm::MockLlm,
        calls: usize,
    }

    impl llm::EmbedModel for CountingEmbed {
        fn embed_batch(&mut self, docs: &[llm::EmbedDoc<'_>]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.calls += 1;
            self.inner.embed_batch(docs)
        }
        fn token_count(&self, text: &str) -> usize {
            self.inner.token_count(text)
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn max_context(&self) -> usize {
            self.inner.max_context()
        }
        fn fingerprint(&self) -> String {
            // Disambiguated: `MockLlm` implements both model traits, and each
            // declares a `fingerprint`.
            llm::EmbedModel::fingerprint(&self.inner)
        }
    }

    /// Run a search under one tag scope, at the shipped defaults.
    fn search_scoped(
        query: &str,
        scope: crate::tags::Scope,
        store: &Store,
        embedder: &mut impl llm::EmbedModel,
    ) -> Result<SearchOutput> {
        let mut config = SearchConfig {
            scope,
            // `SearchConfig::new` takes the loaded config's calibrated
            // section, which defaults to enabled. Overridden here for the
            // same reason as `heuristic_search`: these tests assert the
            // pre-calibration paths, and the calibrated sort has its own
            // tests (Task 7).
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
            ..SearchConfig::new(store, &crate::config::Config::default())
        };
        search_with_intelligence(query, 20, embedder, &mut config)
    }

    #[test]
    fn a_scope_keeps_the_results_inside_the_tagged_notes() {
        let (_tmp, store, mut embedder) = tagged_vault();
        let filter = crate::tags::Scope::parse(&["type/undead".to_string()], &[], &[]).unwrap();

        let unscoped = search_scoped(
            "warding",
            crate::tags::Scope::default(),
            &store,
            &mut embedder,
        )
        .unwrap();
        let paths: Vec<&str> = unscoped
            .results
            .iter()
            .map(|r| r.file_path.as_str())
            .collect();
        assert!(paths.contains(&"wolf.md"), "unscoped, both notes answer");

        let scoped = search_scoped("warding", filter, &store, &mut embedder).unwrap();
        assert!(!scoped.results.is_empty(), "the tagged note still answers");
        for r in &scoped.results {
            assert_eq!(r.file_path, "wight.md", "an out-of-scope note answered");
        }

        // A hard scope, not a cut over the output: each content lane fetched
        // fewer rows, so the excluded note never occupied a lane's width.
        assert!(
            scoped.retrieval.semantic_hits < unscoped.retrieval.semantic_hits
                && scoped.retrieval.fts_hits < unscoped.retrieval.fts_hits,
            "the lanes read the out-of-scope note and it was removed afterwards"
        );
        let trace = scoped
            .retrieval
            .scope
            .as_ref()
            .expect("the trace carries the scope");
        assert_eq!(trace.filter, "all=type/undead");
        assert_eq!(trace.notes, 1);
    }

    #[test]
    fn an_empty_filter_reproduces_the_unscoped_search_exactly() {
        // The control for the whole change: with no scope, the pipeline runs
        // the queries it ran before #60 and returns the same rows in the same
        // order.
        let (_tmp, store, mut embedder) = tagged_vault();

        let a = search_scoped(
            "warding",
            crate::tags::Scope::default(),
            &store,
            &mut embedder,
        )
        .unwrap();
        let b = search_at("warding", 20, 60, &store, &mut embedder);

        let keys = |o: &SearchOutput| -> Vec<(String, i64, String)> {
            o.results
                .iter()
                .map(|r| (r.file_path.clone(), r.chunk_seq, format!("{:.9}", r.score)))
                .collect()
        };
        assert_eq!(keys(&a), keys(&b));
        assert!(a.retrieval.scope.is_none());
    }

    #[test]
    fn a_scope_no_note_satisfies_answers_nothing_without_running_a_model() {
        let (_tmp, store, embedder) = tagged_vault();
        let mut counting = CountingEmbed {
            inner: embedder,
            calls: 0,
        };
        let filter = crate::tags::Scope::parse(
            &["type/undead".to_string(), "type/beast".to_string()],
            &[],
            &[],
        )
        .unwrap();

        let out = search_scoped("warding", filter, &store, &mut counting).unwrap();

        assert!(out.results.is_empty());
        assert!(out.fused.is_empty());
        assert_eq!(
            counting.calls, 0,
            "the query was embedded for an empty scope"
        );
        let trace = out.retrieval.scope.expect("the trace carries the scope");
        assert_eq!(trace.filter, "all=type/undead,type/beast");
        assert_eq!(trace.notes, 0);
        assert_eq!(out.retrieval.fts_expr.as_deref(), Some("\"warding\""));
    }

    #[test]
    fn a_scope_naming_no_tag_fails_the_search() {
        let (_tmp, store, mut embedder) = tagged_vault();
        let filter = crate::tags::Scope::parse(&["type/undeed".to_string()], &[], &[]).unwrap();
        // Not `unwrap_err`: a `SearchOutput` is not `Debug`, and giving it one
        // for a test would be the tail wagging the dog.
        let err = match search_scoped("warding", filter, &store, &mut embedder) {
            Ok(_) => panic!("a scope naming no tag must fail the search"),
            Err(e) => e,
        };
        assert_eq!(
            err.to_string(),
            "no such tag 'type/undeed'; nearest: 'type/undead'"
        );
    }

    /// A dated note outside the scope, linking to a note inside it, so the
    /// only way the linked note can earn graph mass is through the dated
    /// note's restart point.
    fn dated_vault() -> (tempfile::TempDir, Store, llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(
            root.join("journal.md"),
            "---\ndate: 2024-03-15\ntags: [type/beast]\n---\n\n\
             # Journal\n\n## Entry\n\nThe day the pack was counted, written up in \
             [[Ledger]] and nowhere else.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("ledger.md"),
            "---\ntags: [type/undead]\n---\n\n\
             # Ledger\n\n## Tally\n\nA tally of the counts, kept in a column of \
             numbers with nothing else said about them.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("wight.md"),
            "---\ntags: [type/undead]\n---\n\n\
             # Wight\n\n## Warding\n\nA warding effect that pins an undead creature \
             in the space it stands in.\n",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    #[test]
    fn a_dated_note_outside_the_scope_is_not_a_restart_point() {
        // #60. A temporal seed asserts the note is a candidate answer, so the
        // scope reaches it. `journal.md` is the only note linking to
        // `ledger.md`, but unscoped it is already a graph seed through the
        // content lanes too, so the scoped assertion is the one that proves
        // the temporal path: out of scope, `journal.md` earns neither seed,
        // and `ledger.md` loses its graph credit only because of that.
        let (_tmp, store, mut embedder) = dated_vault();
        let query = "warding 2024-03-15";

        // The fixture drives the temporal path, or the test below proves
        // nothing: the query parses to a range, and the dated note is in it.
        let range = crate::temporal::parse_date_range_heuristic(query)
            .expect("the query names a date range");
        let dated: Vec<String> = store
            .get_files_in_date_range(range.0, range.1)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(dated, vec!["journal.md".to_string()]);

        let graph_credits = |out: &SearchOutput| -> Vec<String> {
            out.fused
                .iter()
                .filter(|f| f.lane_contributions.iter().any(|l| l.lane_name == "graph"))
                .map(|f| f.file_path.clone())
                .collect()
        };

        let unscoped =
            search_scoped(query, crate::tags::Scope::default(), &store, &mut embedder).unwrap();
        assert!(
            graph_credits(&unscoped).contains(&"ledger.md".to_string()),
            "unscoped, the dated note seeds the walk that credits its target"
        );

        let filter = crate::tags::Scope::parse(&["type/undead".to_string()], &[], &[]).unwrap();
        let scoped = search_scoped(query, filter, &store, &mut embedder).unwrap();
        assert!(
            graph_credits(&scoped).is_empty(),
            "an out-of-scope dated note restarted the walk: {:?}",
            graph_credits(&scoped)
        );
    }

    /// Two notes on one subject in two folders, so a directory scope has one
    /// to include and one to exclude.
    fn foldered_vault() -> (tempfile::TempDir, Store, llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("Locations")).unwrap();
        std::fs::create_dir_all(root.join("People")).unwrap();
        std::fs::write(
            root.join("Locations/wight.md"),
            "# Wight\n\n## Warding\n\nA warding effect that pins an undead creature \
             in the space it stands in.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("People/wolf.md"),
            "# Wolf\n\n## Warding\n\nA warding effect that pins a beast in the space \
             it stands in.\n",
        )
        .unwrap();
        let store = Store::open_memory().unwrap();
        let mut embedder = llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    #[test]
    fn a_directory_scope_keeps_the_search_results_inside_the_folder() {
        // #65. Reach test: `search_scoped` is #60's wiring, unchanged here.
        // This proves it already carries a directory term with no production
        // change.
        let (_tmp, store, mut embedder) = foldered_vault();
        let scope = crate::tags::Scope::parse(&["/Locations/".to_string()], &[], &[]).unwrap();
        let out = search_scoped("warding", scope, &store, &mut embedder).unwrap();
        assert!(!out.results.is_empty(), "the in-folder note answers");
        for r in &out.results {
            assert_eq!(
                r.file_path, "Locations/wight.md",
                "an out-of-folder note answered a directory-scoped search"
            );
        }
    }

    #[test]
    fn lane_width_is_the_setting_and_not_top_n() {
        // #49. Each content lane fetched `top_n * 3`, so the size of the
        // candidate pool moved with the length of the answer.
        let (_tmp, store, mut embedder) = vault_of_many_files(12);

        let narrow = search_at("warding", 1, 4, &store, &mut embedder);
        let wide = search_at("warding", 20, 4, &store, &mut embedder);

        for output in [&narrow, &wide] {
            let trace = &output.retrieval;
            assert_eq!(trace.fts_hits, 4, "the keyword lane fetches the width");
            assert_eq!(
                trace.semantic_hits, 4,
                "the semantic lane fetches the width"
            );
        }
    }

    #[test]
    fn a_longer_request_extends_the_ranking_rather_than_replacing_it() {
        // The output contract #49 is about: more results means more results,
        // and not other results.
        // More files than the pre-#49 widths for either `top_n` below, so the
        // two runs are not both saturated by a small corpus.
        let (_tmp, store, mut embedder) = vault_of_many_files(40);
        let width = crate::config::default_retrieval_width();

        let short = search_at("warding", 3, width, &store, &mut embedder);
        let long = search_at("warding", 10, width, &store, &mut embedder);

        assert_eq!(short.results.len(), 3);
        assert!(long.results.len() > short.results.len());

        let identity = |r: &InternalSearchResult| (r.file_path.clone(), r.chunk_seq);
        let short_ids: Vec<_> = short.results.iter().map(identity).collect();
        let long_prefix: Vec<_> = long.results.iter().take(3).map(identity).collect();
        assert_eq!(short_ids, long_prefix, "the shorter answer is a prefix");
    }

    /// Records how the rerank lane is driven. Whether an implementation can
    /// amortize its setup depends on being handed the whole candidate set at
    /// once (issue #13), so "one call, not one per candidate" is a property to
    /// hold onto rather than an implementation detail.
    struct CountingReranker {
        inner: llm::MockLlm,
        batch_calls: usize,
        pairs_scored: usize,
        /// Every document the lane handed over, in order — so a test can assert
        /// what the cross-encoder actually read (issue #14).
        documents: Vec<String>,
    }

    impl RerankModel for CountingReranker {
        fn fingerprint(&self) -> String {
            RerankModel::fingerprint(&self.inner)
        }

        fn rerank_score(&mut self, query: &str, document: &str) -> Result<f32> {
            self.inner.rerank_score(query, document)
        }

        fn rerank_batch(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            self.batch_calls += 1;
            self.pairs_scored += documents.len();
            self.documents
                .extend(documents.iter().map(|d| (*d).to_owned()));
            documents
                .iter()
                .map(|d| self.inner.rerank_score(query, d))
                .collect()
        }
    }

    impl CountingReranker {
        fn new() -> Self {
            Self {
                inner: llm::MockLlm::new(8),
                batch_calls: 0,
                pairs_scored: 0,
                documents: Vec::new(),
            }
        }
    }

    /// Run a rerank-lane search and hand back what the reranker was shown.
    fn documents_shown_to_reranker(
        query: &str,
        store: &Store,
        embedder: &mut llm::MockLlm,
        settings: crate::config::RerankConfig,
    ) -> CountingReranker {
        let mut reranker = CountingReranker::new();
        {
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: settings,
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: crate::config::RankingConfig::default(),
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            search_with_intelligence(query, 10, embedder, &mut config).unwrap();
        }
        reranker
    }

    /// A vault whose chunks run past the 200-character snippet boundary, with a
    /// term only on the far side of it.
    fn vault_with_text_past_the_snippet_boundary() -> (tempfile::TempDir, Store, llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("rules")).unwrap();
        let filler = "A warding effect that stops a spell mid-cast. ".repeat(8);
        std::fs::write(
            root.join("rules/counterspell.md"),
            format!("## Counterspell\n\n{filler}\n\nIt cannot stop a Wyrmsbane invocation.\n"),
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    /// Issue #14. The rerank lane used to be handed `candidate.snippet` — the
    /// leading 200 characters from the semantic and graph lanes, and a 64-token
    /// match window from FTS. Reading the whole document jointly with the query
    /// is the reason a cross-encoder exists, so it has to be given the chunk.
    #[test]
    fn the_reranker_is_shown_the_whole_chunk_not_the_snippet() {
        let (_tmp, store, mut embedder) = vault_with_text_past_the_snippet_boundary();

        let reranker = documents_shown_to_reranker(
            "warding",
            &store,
            &mut embedder,
            crate::config::RerankConfig::default(),
        );

        assert!(!reranker.documents.is_empty(), "the lane scored nothing");
        assert!(
            reranker
                .documents
                .iter()
                .any(|d| d.contains("Wyrmsbane invocation")),
            "the reranker never saw past the 200-character mark: {:?}",
            reranker
                .documents
                .iter()
                .map(|d| d.len())
                .collect::<Vec<_>>()
        );
    }

    /// The candidate the cross-encoder reads names the document it came from,
    /// and stops naming it when the switch is turned off.
    ///
    /// A chunk is a section, and a section of `archdragon.md` can go a thousand
    /// characters without saying "archdragon" — which did not matter while the
    /// score was a vote and decides the ranking now (#30).
    #[test]
    fn the_document_title_is_prepended_unless_it_is_switched_off() {
        let (_tmp, store, mut embedder) = vault_with_text_past_the_snippet_boundary();

        let without = documents_shown_to_reranker(
            "warding",
            &store,
            &mut embedder,
            crate::config::RerankConfig {
                document_title: false,
                ..Default::default()
            },
        );
        assert!(
            without
                .documents
                .iter()
                .all(|d| !d.starts_with("counterspell")),
            "the title was prepended with the switch off"
        );

        let with = documents_shown_to_reranker(
            "warding",
            &store,
            &mut embedder,
            crate::config::RerankConfig::default(),
        );
        assert!(
            with.documents
                .iter()
                .all(|d| d.starts_with("counterspell\n\n")),
            "expected every document to open with the file's title"
        );
        assert!(
            with.documents
                .iter()
                .any(|d| d.contains("Wyrmsbane invocation")),
            "prepending the title must not replace the chunk"
        );
    }

    /// A sorted-stage search with a live reranker and the answer floor off, so
    /// `MockLlm`'s hash-based scores cannot empty the result the way a floor
    /// fit for a real cross-encoder would (see `sorted_config`).
    fn search_with_reranker(
        query: &str,
        store: &Store,
        embedder: &mut llm::MockLlm,
    ) -> SearchOutput {
        let mut reranker = llm::MockLlm::new(8);
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: Some(&mut reranker),
            store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::Chunk,
            ranking: sorted_config(crate::config::RankingConfig::default()),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        search_with_intelligence(query, 20, embedder, &mut config).unwrap()
    }

    /// #35's window invariant: the text a result emits has to be a substring
    /// of the chunk the cross-encoder actually scored, not a re-truncated copy
    /// derived independently of it. `vault_with_text_past_the_snippet_boundary`
    /// answers "warding" past the 200-character snippet, which is what makes
    /// the invariant worth checking rather than trivially true.
    #[test]
    fn every_emitted_window_is_a_substring_of_what_was_scored() {
        let (_tmp, store, mut embedder) = vault_with_text_past_the_snippet_boundary();
        let out = search_with_reranker("warding", &store, &mut embedder);
        assert!(
            !out.results.is_empty(),
            "the fixture query answered nothing"
        );
        let mut saw_the_long_chunk = false;
        for r in &out.results {
            let scored = store
                .get_chunk_texts(&[(r.file_id, r.chunk_seq)])
                .unwrap()
                .remove(0)
                .expect("chunk text");
            assert!(
                scored.contains(&r.text),
                "emitted text for {} is not a substring of the scored chunk",
                r.file_path
            );
            assert!(!r.text.is_empty());
            // The substring check above is necessary but not sufficient: the
            // 200-character `snippet` is always a substring of the full
            // chunk too, so it alone cannot tell "the whole chunk" apart from
            // "a regression back to the snippet". "Wyrmsbane invocation"
            // sits past that 200-character mark, so a result whose emitted
            // text still carries it proves the window is the chunk the model
            // scored and not the truncated preview.
            if scored.contains("Wyrmsbane invocation") {
                saw_the_long_chunk = true;
                assert!(
                    r.text.contains("Wyrmsbane invocation"),
                    "emitted text for {} regressed to the 200-character snippet",
                    r.file_path
                );
            }
        }
        assert!(
            saw_the_long_chunk,
            "the fixture's past-boundary chunk never reached the results"
        );
    }

    /// #35: a result reports which lanes support it, plus the breadcrumb and
    /// the token count, in place of a bare number.
    #[test]
    fn results_carry_content_provenance_not_a_bare_score() {
        let (_tmp, store, mut embedder) = vault_with_text_past_the_snippet_boundary();
        let out = search_with_reranker("warding", &store, &mut embedder);
        assert!(
            !out.results.is_empty(),
            "the fixture query answered nothing"
        );
        let top = &out.results[0];
        assert!(
            top.provenance.keyword || top.provenance.semantic || top.provenance.graph,
            "a result with no supporting lane at all is not a real answer"
        );
        assert!(!top.heading_path.is_empty());
        assert!(top.token_count > 0);
    }

    /// Issue #25. The cross-encoder's cost is very nearly linear in the text it
    /// is handed, so this cap is what bounds query latency — and it has to hold
    /// per candidate, since `n_ctx` is sized to the longest pair in the batch.
    #[test]
    fn the_character_cap_bounds_every_document_the_reranker_sees() {
        let (_tmp, store, mut embedder) = vault_with_text_past_the_snippet_boundary();

        // Explicitly unlimited rather than `default()`, which now carries a cap
        // of its own — this arm has to be the thing the cap is measured
        // against. The title is off in both arms for the same reason: the cap
        // bounds the chunk, and `the_character_cap_never_eats_the_document_title`
        // is where that boundary is pinned.
        let uncapped = documents_shown_to_reranker(
            "warding",
            &store,
            &mut embedder,
            crate::config::RerankConfig {
                max_document_chars: 0,
                document_title: false,
            },
        );
        assert!(
            uncapped.documents.iter().any(|d| d.chars().count() > 120),
            "the fixture stopped producing a document long enough to cap"
        );

        let capped = documents_shown_to_reranker(
            "warding",
            &store,
            &mut embedder,
            crate::config::RerankConfig {
                max_document_chars: 120,
                document_title: false,
            },
        );
        assert!(!capped.documents.is_empty(), "the lane scored nothing");
        assert!(
            capped.documents.iter().all(|d| d.chars().count() <= 120),
            "a document ran past the cap: {:?}",
            capped
                .documents
                .iter()
                .map(|d| d.chars().count())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            capped.documents.len(),
            uncapped.documents.len(),
            "capping the text must not change which candidates are scored"
        );
    }

    /// The cap applies to the chunk, not to the assembled pair, so the switch in
    /// `document_title` cannot be the thing it cuts off.
    #[test]
    fn the_character_cap_never_eats_the_document_title() {
        let (_tmp, store, mut embedder) = vault_with_text_past_the_snippet_boundary();

        let capped = documents_shown_to_reranker(
            "warding",
            &store,
            &mut embedder,
            crate::config::RerankConfig {
                document_title: true,
                max_document_chars: 20,
            },
        );
        assert!(!capped.documents.is_empty(), "the lane scored nothing");
        assert!(
            capped
                .documents
                .iter()
                .all(|d| d.starts_with("counterspell\n\n")),
            "a title was truncated away: {:?}",
            capped.documents
        );
    }

    #[test]
    fn a_cap_of_zero_keeps_the_whole_document() {
        let text = "a".repeat(5_000);
        assert_eq!(truncate_chars(text.clone(), 0), text);
    }

    #[test]
    fn text_shorter_than_the_cap_is_returned_unchanged() {
        assert_eq!(truncate_chars("short".to_string(), 100), "short");
        assert_eq!(truncate_chars("exact".to_string(), 5), "exact");
        assert_eq!(truncate_chars(String::new(), 100), "");
    }

    /// Truncating on a byte offset would panic here rather than return a short
    /// string, and an em-dash in a vault is not an exotic input.
    #[test]
    fn the_cap_counts_characters_not_bytes() {
        // Four characters, eleven bytes.
        let text = "é—漢字".to_string();
        assert_eq!(text.len(), 11);
        assert_eq!(truncate_chars(text.clone(), 2), "é—");
        assert_eq!(truncate_chars(text.clone(), 4), text);
        assert_eq!(truncate_chars(text, 3).chars().count(), 3);
    }

    #[test]
    fn the_rerank_lane_scores_all_its_candidates_in_one_call() {
        let (_tmp, store, mut embedder) = indexed_vault();
        let mut reranker = CountingReranker::new();

        {
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: sorted_config(crate::config::RankingConfig::default()),
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
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
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: None,
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: cap,
                group_by: GroupBy::Chunk,
                // The results cap is the legacy stage's; #30 caps the
                // shortlist instead. See `the_sorted_stage_caps_the_shortlist_
                // and_not_the_results`.
                ranking: crate::config::RankingConfig {
                    mode: crate::config::RankingMode::Legacy,
                    ..Default::default()
                },
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
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

    /// A vault where one document answers the query in six sections, and a
    /// second document is reachable only by a wikilink out of it.
    fn vault_with_one_deep_document_and_a_linked_neighbour()
    -> (tempfile::TempDir, Store, llm::MockLlm) {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("rules")).unwrap();

        let mut deep = String::from("# Wards\n\n");
        for level in 1..=6 {
            deep.push_str(&format!(
                "## Level {level} Ward\n\nA warding effect at level {level}. \
                 See [[quiet-neighbour|the neighbour]]. The ward holds for as long as \
                 its caster keeps channelling, and it fails the moment they stop.\n\n"
            ));
        }
        std::fs::write(root.join("rules/wards.md"), deep).unwrap();
        // Nothing here matches the query lexically, and the mock embedder is a
        // hash — so the only route to this file is the link above.
        std::fs::write(
            root.join("rules/quiet-neighbour.md"),
            "# Quiet Neighbour\n\nUnrelated prose about pottery and rope.\n",
        )
        .unwrap();

        let store = Store::open_memory().unwrap();
        let mut embedder = llm::MockLlm::new(256);
        let config = crate::config::Config::default();
        crate::indexer::run_index_shared(
            root,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &store,
            &mut embedder,
            false,
            None,
        )
        .unwrap();
        (tmp, store, embedder)
    }

    /// The sorted stage with the answer floor turned off.
    ///
    /// `MockLlm::rerank_score` returns a hash of the pair and not a calibrated
    /// probability. A floor fit against a real cross-encoder rejects nearly all
    /// of those scores. With the floor on, every test of routing, limits and
    /// batches becomes a test of where the hash lands. The floor has its own
    /// tests, and they state the value they use.
    fn sorted_config(ranking: crate::config::RankingConfig) -> crate::config::RankingConfig {
        crate::config::RankingConfig {
            mode: crate::config::RankingMode::Sorted,
            answer_floor: 0.0,
            ..ranking
        }
    }

    /// #30's two caps, split by job: bound what the model is shown, because it
    /// cannot rank what it never saw; do not bound what it returns, because
    /// ranking is its job. Under the legacy stage the same query is cut to
    /// `max_chunks_per_file` on the way out.
    #[test]
    fn the_sorted_stage_caps_the_shortlist_and_not_the_results() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let count_for = |ranking, embedder: &mut llm::MockLlm| {
            let mut reranker = CountingReranker::new();
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking,
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            let output = search_with_intelligence("warding", 20, embedder, &mut config).unwrap();
            output
                .results
                .iter()
                .filter(|r| r.file_path == "rules/wards.md")
                .count()
        };

        // This test exercises the ranking pipeline. The ranking pipeline is
        // below coalescing. Coalescing has its own tests (#39).
        let legacy = count_for(
            crate::config::RankingConfig {
                mode: crate::config::RankingMode::Legacy,
                coalesce_adjacent: false,
                ..Default::default()
            },
            &mut embedder,
        );
        assert_eq!(legacy, 3, "the legacy stage caps the result set");

        let sorted = count_for(
            sorted_config(crate::config::RankingConfig {
                shortlist_cap: 6,
                coalesce_adjacent: false,
                ..Default::default()
            }),
            &mut embedder,
        );
        assert!(
            sorted > 3,
            "one document holding the best sections must be able to return them, got {sorted}"
        );
        assert!(
            sorted <= 6,
            "the shortlist cap still bounds what the model was shown, got {sorted}"
        );
    }

    /// The answer floor end to end (#34), with its control.
    ///
    /// A floor above every score the model can return must give an empty
    /// response. A floor of `0.0` must leave the same query as #30 left it.
    /// The two together test the floor in the pipeline and not only in
    /// `ranking::apply_answer_floor`. A floor that no code calls passes the unit
    /// tests and changes no output.
    #[test]
    fn the_answer_floor_empties_a_response_and_zero_is_inert() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let results_at = |floor: f64, embedder: &mut llm::MockLlm| {
            let mut reranker = CountingReranker::new();
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: crate::config::RankingConfig {
                    mode: crate::config::RankingMode::Sorted,
                    answer_floor: floor,
                    ..Default::default()
                },
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            search_with_intelligence("warding", 20, embedder, &mut config)
                .unwrap()
                .results
        };

        // Above 1.0, so the floor rejects every candidate whatever the mock's
        // hash returns.
        let gated = results_at(1.01, &mut embedder);
        assert!(
            gated.is_empty(),
            "no candidate was above the floor, but the engine returned {} results",
            gated.len()
        );
        // The floor's empty response reaches the envelope as `NoResults`, and
        // the CLI's text rendering is the same literal message #34 always
        // gave — through the real packaging path, not a hardcoded slice (#35).
        let env = crate::packaging::assemble(
            &gated,
            crate::packaging::AssembleParams {
                budget_tokens: 8192,
                full: false,
                summaries: false,
                degraded: false,
                per_note_cap: 0,
            },
        );
        assert_eq!(env.status, crate::packaging::SearchStatus::NoResults);
        assert_eq!(
            crate::packaging::render_text(&env, false),
            format!("{}\n", crate::ranking::NO_RELEVANT_CONTENT)
        );

        let ungated = results_at(0.0, &mut embedder);
        assert!(!ungated.is_empty(), "the control removed a result");
    }

    /// The results limit that §9.1 wants and #30 does not, shipped as a key with
    /// no limit. The default must not limit one document's share, and the key
    /// must limit it when it is set.
    #[test]
    fn per_note_cap_is_unbounded_by_default_and_binds_when_set() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let count_for = |per_note_cap, embedder: &mut llm::MockLlm| {
            let mut reranker = CountingReranker::new();
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                // This test exercises the ranking pipeline. The ranking
                // pipeline is below coalescing. Coalescing has its own
                // tests (#39).
                ranking: sorted_config(crate::config::RankingConfig {
                    shortlist_cap: 6,
                    per_note_cap,
                    coalesce_adjacent: false,
                    ..Default::default()
                }),
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            search_with_intelligence("warding", 20, embedder, &mut config)
                .unwrap()
                .results
                .iter()
                .filter(|r| r.file_path == "rules/wards.md")
                .count()
        };

        let unbounded = count_for(0, &mut embedder);
        assert!(
            unbounded > 2,
            "the default limited the results, got {unbounded}"
        );
        assert_eq!(count_for(2, &mut embedder), 2, "the key did not limit them");
    }

    /// The defect the reserve answers. With a budget of four and two slots
    /// spoken for by the content order, a candidate no content lane found still
    /// reaches the model — which is the whole difference between a routing
    /// guarantee and a fusion weight.
    #[test]
    fn a_graph_only_candidate_reaches_the_model_on_a_budget_that_would_have_cut_it() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let mut reranker = CountingReranker::new();
        {
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig {
                    max_document_chars: 0,
                    ..Default::default()
                },
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: sorted_config(crate::config::RankingConfig {
                    candidates: 4,
                    graph_reserve: 2,
                    ..Default::default()
                }),
                calibrated: crate::config::CalibratedConfig {
                    enabled: false,
                    ..Default::default()
                },
            };
            search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();
        }

        assert_eq!(reranker.batch_calls, 1);
        assert!(
            reranker.pairs_scored <= 4,
            "the budget is the only number that binds, got {}",
            reranker.pairs_scored
        );
        assert!(
            reranker
                .documents
                .iter()
                .any(|d| d.contains("pottery and rope")),
            "the linked neighbour never reached the model: {:?}",
            reranker.documents
        );
    }

    /// A cross-encoder that is present and fails.
    struct BrokenReranker;

    impl RerankModel for BrokenReranker {
        fn fingerprint(&self) -> String {
            "broken".to_string()
        }

        fn rerank_score(&mut self, _query: &str, _document: &str) -> Result<f32> {
            anyhow::bail!("decode failed")
        }

        fn rerank_batch(&mut self, _query: &str, _documents: &[&str]) -> Result<Vec<f32>> {
            anyhow::bail!("decode failed")
        }
    }

    /// When the model that should have sorted the shortlist cannot, the
    /// fallback is the documented interleave and it says so.
    ///
    /// The failure mode this guards is not the missing order — it is a missing
    /// order that looks exactly like a ranked answer. Scoring every candidate
    /// 0.0 instead would leave the tie-break deciding the whole ranking while
    /// the output claimed a model had judged it.
    #[test]
    fn a_failed_cross_encoder_degrades_the_order_and_labels_it() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let mut reranker = BrokenReranker;
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: Some(&mut reranker),
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::Chunk,
            ranking: sorted_config(crate::config::RankingConfig::default()),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let output = search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();

        assert!(output.degraded, "the fallback did not label itself");
        assert!(!output.results.is_empty(), "and it still answered");
    }

    /// No cross-encoder configured is not a degraded query — it is a different
    /// configuration, and it keeps the fusion those probes were tuned against.
    #[test]
    fn a_build_without_a_cross_encoder_keeps_the_legacy_stage() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: None,
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::Chunk,
            // This test exercises the ranking pipeline. The ranking
            // pipeline is below coalescing. Coalescing has its own
            // tests (#39).
            ranking: sorted_config(crate::config::RankingConfig {
                shortlist_cap: 6,
                coalesce_adjacent: false,
                ..Default::default()
            }),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let output = search_with_intelligence("warding", 20, &mut embedder, &mut config).unwrap();

        assert!(!output.degraded);
        assert_eq!(
            output
                .results
                .iter()
                .filter(|r| r.file_path == "rules/wards.md")
                .count(),
            3,
            "the legacy stage caps the results at max_chunks_per_file"
        );
    }
    // ── The calibrated sort: the model-free default (spec 2026-08-30) ──

    /// A no-model query over `indexed_vault`'s corpus, with the `[calibrated]`
    /// section the caller gives.
    ///
    /// `sorted_config` holds `[ranking] answer_floor` at 0.0, so a result the
    /// floor removes was removed by `[calibrated] floor` and by nothing else.
    fn calibrated_search(
        query: &str,
        store: &Store,
        embedder: &mut llm::MockLlm,
        calibrated: crate::config::CalibratedConfig,
    ) -> SearchOutput {
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: None,
            store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::Chunk,
            // This test exercises the ranking pipeline. The ranking pipeline is
            // below coalescing. Coalescing has its own tests (#39).
            ranking: sorted_config(crate::config::RankingConfig {
                coalesce_adjacent: false,
                ..Default::default()
            }),
            calibrated,
        };
        search_with_intelligence(query, 20, embedder, &mut config).unwrap()
    }

    /// The sort score of a result, if the calibrated logistic produced it.
    fn calibrated_score(result: &FusedResult) -> Option<f64> {
        result
            .lane_contributions
            .iter()
            .find(|l| l.lane_name == "calibrated")
            .map(|l| l.raw_score)
    }

    /// An output as a byte-exactness control compares it: the identity of every
    /// result and both of its numbers, to the bit. A control that compared
    /// rounded scores would pass a change it was written to catch.
    fn result_shape(out: &SearchOutput) -> Vec<(String, i64, u64, u64)> {
        out.results
            .iter()
            .map(|r| {
                (
                    r.file_path.clone(),
                    r.chunk_seq,
                    r.score.to_bits(),
                    r.confidence.to_bits(),
                )
            })
            .collect()
    }

    /// With no cross-encoder the sorted stage still runs, and the logistic
    /// sorts the pool where the model would. The probability is the confidence,
    /// so a query with no good answer no longer reports 100% for whatever it
    /// ranked first.
    #[test]
    fn the_model_free_default_sorts_by_calibrated_probability() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let out = calibrated_search(
            "warding",
            &store,
            &mut embedder,
            crate::config::CalibratedConfig {
                floor: 0.0,
                ..Default::default()
            },
        );

        assert!(
            !out.degraded,
            "the calibrated sort is not the degraded ordering"
        );
        assert!(!out.fused.is_empty(), "the query retrieved nothing to sort");

        let top = &out.fused[0];
        let p = calibrated_score(top).expect("the sort score is labeled calibrated");
        assert!(
            (top.confidence - p * 100.0).abs() < 1e-9,
            "confidence is the probability"
        );
        assert!(
            out.fused.iter().all(|f| calibrated_score(f).is_some()),
            "every candidate the pool held was scored, provenance regardless"
        );
        assert!(
            out.fused
                .windows(2)
                .all(|w| calibrated_score(&w[0]).unwrap_or(0.0)
                    >= calibrated_score(&w[1]).unwrap_or(0.0)),
            "sorted by the probability"
        );
    }

    /// The floor of this path is `[calibrated] floor` and not
    /// `[ranking] answer_floor`. A floor the logistic cannot reach empties the
    /// response; `0.0` is the floor's own control and removes nothing.
    #[test]
    fn the_floor_bounds_the_calibrated_result() {
        let (_tmp, store, mut embedder) = indexed_vault();

        // Sigma is open above 0 and below 1, so no candidate reaches 1.0.
        let floored = calibrated_search(
            "warding",
            &store,
            &mut embedder,
            crate::config::CalibratedConfig {
                floor: 1.0,
                ..Default::default()
            },
        );
        assert!(
            floored.results.is_empty(),
            "an unreachable floor empties the result"
        );

        let open = calibrated_search(
            "warding",
            &store,
            &mut embedder,
            crate::config::CalibratedConfig {
                floor: 0.0,
                ..Default::default()
            },
        );
        assert!(!open.results.is_empty(), "0.0 removes nothing");
    }

    /// The control (spec, "Where it sits in the pipeline"): `enabled = false`
    /// sends a no-model build to the legacy stage exactly as before, so its
    /// confidence is a share of the best score again and no lane is labeled
    /// `calibrated`.
    #[test]
    fn enabled_false_restores_the_legacy_routing() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let out = calibrated_search(
            "warding",
            &store,
            &mut embedder,
            crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        );

        assert!(!out.fused.is_empty(), "the query retrieved nothing");
        assert_eq!(
            out.fused[0].confidence, 100.0,
            "legacy confidence normalizes the top to 100%"
        );
        assert!(
            out.fused.iter().all(|f| calibrated_score(f).is_none()),
            "the legacy stage runs no calibrated scorer"
        );
    }

    /// With a cross-encoder configured the whole `[calibrated]` section is
    /// inert: the model sorts, its `answer_floor` applies, and the output does
    /// not move whatever the section holds.
    #[test]
    fn a_configured_cross_encoder_ignores_the_calibrated_section() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let run = |calibrated: crate::config::CalibratedConfig, embedder: &mut llm::MockLlm| {
            let mut reranker = llm::MockLlm::new(8);
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: Some(&mut reranker),
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: sorted_config(crate::config::RankingConfig {
                    coalesce_adjacent: false,
                    ..Default::default()
                }),
                calibrated,
            };
            search_with_intelligence("warding", 20, embedder, &mut config).unwrap()
        };

        // Every field is loud, not defaulted. Two arms that differ only in
        // `enabled` leave the coefficients and the floor equal, so they test
        // one field of five: the reranked path could read `[calibrated] floor`
        // or scale by `[calibrated] semantic` and both arms would still agree.
        // A section this wrong that changes nothing is the invariant.
        let on = run(
            crate::config::CalibratedConfig {
                enabled: true,
                semantic: 99.0,
                keyword: 99.0,
                intercept: 99.0,
                floor: 0.99,
            },
            &mut embedder,
        );
        let off = run(
            crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
            &mut embedder,
        );

        assert!(
            !result_shape(&on).is_empty(),
            "the two arms must compare something"
        );
        assert_eq!(result_shape(&on), result_shape(&off));
    }

    /// Byte-exactness invariant 3: `mode = "legacy"` is untouched, and
    /// `[calibrated]` is inert under it.
    ///
    /// Both routing flags read the mode first, so an explicit legacy mode
    /// reaches neither scorer however the section is set. This is the arm the
    /// other two controls do not cover: no cross-encoder *and* the calibrated
    /// section at its shipped default.
    #[test]
    fn the_legacy_mode_ignores_the_calibrated_section() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let run = |calibrated: crate::config::CalibratedConfig, embedder: &mut llm::MockLlm| {
            let mut config = SearchConfig {
                fts: crate::config::FtsConfig::default(),
                scope: crate::tags::Scope::default(),
                reranker: None,
                store: &store,
                rerank_candidates: 30,
                lane_weights: crate::config::LaneWeights::default(),
                rerank: crate::config::RerankConfig::default(),
                max_chunks_per_file: 3,
                group_by: GroupBy::Chunk,
                ranking: crate::config::RankingConfig {
                    mode: crate::config::RankingMode::Legacy,
                    coalesce_adjacent: false,
                    ..Default::default()
                },
                calibrated,
            };
            search_with_intelligence("warding", 20, embedder, &mut config).unwrap()
        };

        let on = run(crate::config::CalibratedConfig::default(), &mut embedder);
        let off = run(
            crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
            &mut embedder,
        );

        assert!(
            !result_shape(&on).is_empty(),
            "the two arms must compare something"
        );
        assert_eq!(result_shape(&on), result_shape(&off));
        assert!(
            on.fused.iter().all(|f| calibrated_score(f).is_none()),
            "the legacy stage runs no calibrated scorer"
        );
    }

    /// A graph or temporal admission that no content lane fetched still carries
    /// both features: the cosine from one vector fetch, the BM25 from one
    /// scoped keyword query. Provenance does not decide whether a candidate is
    /// scored.
    ///
    /// The same candidate is scored twice, against the chunk's own vector and
    /// against one orthogonal to it. Only the vector backfill can tell the two
    /// runs apart, so deleting it makes the two scores equal and fails this
    /// test — which is what stops it from passing on the keyword half alone.
    #[test]
    fn a_candidate_no_lane_fetched_still_gets_a_probability() {
        let store = Store::open_memory().unwrap();
        let path = "rules/linked.md";
        let file_id = store
            .insert_file(
                path,
                "hash",
                100,
                &crate::docid::generate_docid(path),
                None,
                None,
            )
            .unwrap();
        store
            .insert_chunk_with_vector(
                &crate::store::NewChunk {
                    file_id,
                    seq: 0,
                    text: "a warding effect that stops a spell mid-cast",
                    vector_id: 1,
                    token_count: 9,
                    ..Default::default()
                },
                // A unit vector, so its dot product with the aligned query
                // below is 1.0 and with the orthogonal one is 0.0.
                &[1.0, 0.0, 0.0, 0.0],
            )
            .unwrap();
        // Two more rows the term does not match, so `df < N` and the term's idf
        // is its own value rather than the clamp an all-matching corpus gives.
        for (seq, text) in [(1, "pottery and rope"), (2, "a rope bridge")] {
            store
                .insert_chunk_with_vector(
                    &crate::store::NewChunk {
                        file_id,
                        seq,
                        text,
                        vector_id: seq as u64 + 1,
                        token_count: 4,
                        ..Default::default()
                    },
                    &[0.0, 0.0, 1.0, 0.0],
                )
                .unwrap();
        }

        let params = crate::config::CalibratedConfig::default();
        let score_against = |query_vec: &[f32]| {
            let mut pool = vec![ranking::Candidate {
                file_path: path.to_string(),
                file_id,
                chunk_seq: 0,
                heading: None,
                snippet: String::new(),
                docid: None,
                rrf_rank: None,
                rrf_score: 0.0,
                graph_rank: Some(1),
                temporal: false,
                admitted_by: ranking::Source::Graph,
                lane_contributions: Vec::new(),
                rerank_score: None,
                emit_text: None,
                emit_token_count: None,
                emit_truncated: false,
            }];
            let (bound, idfs) = apply_calibrated_scores(
                &store,
                &params,
                "warding",
                query_vec,
                &crate::config::FtsConfig::default().weights(),
                &[],
                &[],
                &mut pool,
            );
            let p = pool[0]
                .rerank_score
                .expect("the backfill must score a candidate no lane fetched");
            (p, bound, idfs.len())
        };

        let (aligned, bound, terms) = score_against(&[1.0, 0.0, 0.0, 0.0]);
        let (orthogonal, _, _) = score_against(&[0.0, 1.0, 0.0, 0.0]);

        assert_eq!(terms, 1, "one query term, one doc-frequency count");
        assert!(bound > 0.0, "one term with a positive idf bounds the query");
        assert!(
            aligned > orthogonal,
            "the cosine backfill must reach the logistic: {aligned} against {orthogonal}"
        );
        assert!(
            orthogonal > crate::calibrate::probability(0.0, 0.0, &params),
            "and the keyword backfill must reach it as well: {orthogonal}"
        );
    }

    /// The query-level half of `apply_calibrated_scores`'s return (spec
    /// 2026-08-30): the BM25 upper bound and each term's idf, which reach
    /// `--explain` beside the per-candidate probability that already prints as
    /// the `"calibrated"` lane. A reranked run has no bound or idf to report,
    /// so the line does not print.
    #[test]
    fn explain_carries_the_calibrated_bound_and_idfs() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let calibrated_out = calibrated_search(
            "warding",
            &store,
            &mut embedder,
            crate::config::CalibratedConfig {
                floor: 0.0,
                ..Default::default()
            },
        );
        let report = explain_report(&calibrated_out, 10);
        assert!(report.contains("calibrated: bound="), "{report}");
        assert!(report.contains("idf("), "{report}");

        let mut reranker = CountingReranker::new();
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: Some(&mut reranker),
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::Chunk,
            ranking: sorted_config(crate::config::RankingConfig {
                coalesce_adjacent: false,
                ..Default::default()
            }),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let reranked_out =
            search_with_intelligence("warding", 20, &mut embedder, &mut config).unwrap();
        let reranked_report = explain_report(&reranked_out, 10);
        assert!(
            !reranked_report.contains("calibrated: bound="),
            "{reranked_report}"
        );
    }

    /// The reranker runs, so the order is the model's and nothing claims to be
    /// degraded.
    #[test]
    fn a_sorted_query_with_a_model_is_not_degraded() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let mut reranker = CountingReranker::new();
        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: Some(&mut reranker),
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::Chunk,
            ranking: sorted_config(crate::config::RankingConfig::default()),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let output = search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();

        assert!(!output.degraded);
        // Confidence is the model's probability, not a share of the best score,
        // so the top result is only 100% if the model actually said so.
        assert!(output.results.iter().all(|r| r.confidence <= 100.0));
    }

    /// `group_by = "file"` is a request about the shape of the answer, not a
    /// vote-counting guard, so it survives the removal of the results cap.
    #[test]
    fn file_grouping_survives_the_sorted_stage() {
        let (_tmp, store, mut embedder) = vault_with_one_deep_document_and_a_linked_neighbour();

        let mut config = SearchConfig {
            fts: crate::config::FtsConfig::default(),
            scope: crate::tags::Scope::default(),
            reranker: None,
            store: &store,
            rerank_candidates: 30,
            lane_weights: crate::config::LaneWeights::default(),
            rerank: crate::config::RerankConfig::default(),
            max_chunks_per_file: 3,
            group_by: GroupBy::File,
            ranking: sorted_config(crate::config::RankingConfig {
                shortlist_cap: 6,
                ..Default::default()
            }),
            calibrated: crate::config::CalibratedConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let output = search_with_intelligence("warding", 10, &mut embedder, &mut config).unwrap();

        let mut seen = std::collections::HashSet::new();
        for r in &output.results {
            assert!(
                seen.insert(r.file_path.clone()),
                "{} appeared twice",
                r.file_path
            );
        }
    }

    #[test]
    fn group_by_file_returns_one_result_per_document() {
        let (_tmp, store, mut embedder) = indexed_vault();

        let output = heuristic_search("warding", 10, &store, &mut embedder, GroupBy::File);

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

        let output = heuristic_search("mid-cast", 10, &store, &mut embedder, GroupBy::Chunk);
        let hit = output
            .results
            .iter()
            .find(|r| r.file_path == "rules/abjuration-spells.md")
            .expect("the phrase appears in exactly one section");
        assert_eq!(hit.heading.as_deref(), Some("## Level 3 Counterspell"));
    }

    fn scored(scores: &[f64]) -> Vec<RankedResult> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &score)| RankedResult {
                file_path: format!("f{i}.md"),
                file_id: i as i64,
                chunk_seq: 0,
                score,
                heading: None,
                snippet: String::new(),
                docid: None,
            })
            .collect()
    }

    fn order_of(lane: &[RankedResult]) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..lane.len()).collect();
        idx.sort_by(|&a, &b| lane[b].score.partial_cmp(&lane[a].score).unwrap());
        idx
    }

    #[test]
    fn a_lane_is_rescaled_onto_its_own_range() {
        // Negated BM25, of the magnitude the eval vault actually produces.
        let mut lane = scored(&[16.961, 9.693, 2.503]);
        normalise_lane_scores(&mut lane);

        assert_eq!(lane[0].score, 1.0);
        assert_eq!(lane[2].score, SEED_SCORE_FLOOR);
        // The middle keeps its position within the span — this is the point of
        // normalising rather than ranking. 9.693 sits 49.7% of the way up.
        assert!(
            (lane[1].score - 0.5473).abs() < 1e-3,
            "got {}",
            lane[1].score
        );
    }

    #[test]
    fn the_two_lanes_become_comparable() {
        // The defect, in one assertion. Semantic tops out at 0.686 on this vault
        // and FTS bottoms out at 2.162, so before #26 the *worst* keyword hit
        // outscored the *best* semantic one and `merge_seeds` handed every
        // contested file to FTS regardless of relevance.
        let mut semantic = scored(&[0.686, 0.426]);
        let mut fts = scored(&[9.693, 2.162]);
        assert!(fts[1].score > semantic[0].score, "premise");

        normalise_lane_scores(&mut semantic);
        normalise_lane_scores(&mut fts);

        assert_eq!(semantic[0].score, fts[0].score, "each lane's best is 1.0");
        assert_eq!(semantic[1].score, fts[1].score, "each lane's worst is 0.1");
    }

    #[test]
    fn normalisation_never_scores_a_hit_at_zero() {
        // `graph_expand` multiplies the seed score by a hop decay and sorts
        // before truncating, so 0 would delete the weakest seed of each lane
        // from the expansion ordering rather than putting it last.
        let mut lane = scored(&[5.0, 4.0, 0.001]);
        normalise_lane_scores(&mut lane);
        assert!(lane.iter().all(|r| r.score >= SEED_SCORE_FLOOR));
    }

    #[test]
    fn a_lane_with_nothing_to_rank_puts_everything_at_the_top() {
        // Probe 5's FTS lane returned exactly one hit. One hit — or N identical
        // ones — means the lane has no basis for an ordering, and the floor
        // would understate every one of them.
        let mut single = scored(&[4.184]);
        normalise_lane_scores(&mut single);
        assert_eq!(single[0].score, 1.0);

        let mut tied = scored(&[3.0, 3.0, 3.0]);
        normalise_lane_scores(&mut tied);
        assert!(tied.iter().all(|r| r.score == 1.0));

        let mut empty: Vec<RankedResult> = vec![];
        normalise_lane_scores(&mut empty);
    }

    #[test]
    fn the_fusion_pool_keeps_the_lanes_own_scores() {
        // The split that makes #26 safe: `normalised` hands back a rescaled
        // copy and leaves the caller's hits alone, so the pool that feeds
        // `collapse_lane` — and through it the rank `rrf_fuse` reads — still
        // carries what the lane actually said.
        let lane = scored(&[16.961, 9.693, 2.503]);
        let seeds = normalised(&lane);

        assert_eq!(
            lane.iter().map(|r| r.score).collect::<Vec<_>>(),
            vec![16.961, 9.693, 2.503],
            "the fusion pool must not be rescaled"
        );
        assert_eq!(seeds[0].score, 1.0);
        assert_eq!(seeds[2].score, SEED_SCORE_FLOOR);
    }

    #[test]
    fn normalisation_preserves_the_lanes_own_ordering() {
        let mut lane = scored(&[2.5, 16.9, 9.6, 4.4]);
        let before: Vec<usize> = order_of(&lane);
        normalise_lane_scores(&mut lane);
        assert_eq!(before, order_of(&lane));
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
