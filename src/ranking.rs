//! The ranking stage: the cross-encoder sorts, and the graph reaches it by
//! reserved quota (issue #30).
//!
//! Before this, the pipeline fused five lanes by weighted RRF and the
//! cross-encoder was one of them. Two things follow from that arrangement, and
//! both are defects:
//!
//! - **The one absolute score in the pipeline was converted to a rank.** A
//!   cross-encoder reads the query and the passage together and returns a
//!   calibrated probability; `weight/(k + rank)` throws that away and averages
//!   what is left with the lanes it exists to correct.
//! - **A lane could be shut out of the shortlist.** Candidates were taken from
//!   the *fused* order, so a lane that lost the fusion arithmetic never reached
//!   the model at all. The model cannot rescue a candidate it was never shown.
//!
//! So the pool is built by **reserved quota** and the cross-encoder **sorts**
//! it. The reserve is a routing guarantee and not a score bonus: graph
//! candidates are guaranteed to be *seen*, and then live or die by the model.
//!
//! Nothing reblends. RRF rank re-enters only as a tie-break between candidates
//! the model scored identically — see [`Tiebreak`].

use std::collections::{HashMap, HashSet};

use crate::config::Tiebreak;
use crate::fusion::{FusedResult, LaneContribution, RankedResult};

/// The key a candidate is identified by, everywhere: `(file_id, chunk_seq)`.
pub type ChunkKey = (i64, i64);

/// Which source's reserve admitted a candidate to the pool.
///
/// This is a routing fact, not a score. It answers "why was the model shown
/// this", which is the question the acceptance criteria in #30 are three
/// separate counts of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The fused semantic + keyword order.
    Rrf,
    /// The graph lane's reach order — the reserved 16 of 64.
    Graph,
    /// A note whose date matches the query's range.
    Temporal,
    /// Admitted after every reserve was satisfied and slots remained.
    Backfill,
}

/// One candidate in the shortlist the cross-encoder is shown.
///
/// It carries where it came from as well as what it is, because "the graph lane
/// contributed nothing" and "the graph lane was never in the pool" are
/// different findings and the result list cannot tell them apart.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub file_path: String,
    pub file_id: i64,
    pub chunk_seq: i64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
    /// Its 1-based position in the fused content order, if it appeared there.
    pub rrf_rank: Option<usize>,
    /// Its fused score, 0.0 when no content lane found it.
    pub rrf_score: f64,
    /// Its 1-based position in the graph lane's reach order, if the lane
    /// reached it.
    pub graph_rank: Option<usize>,
    /// Whether the query carried a date range this candidate's note matches.
    pub temporal: bool,
    /// Which reserve gave it its slot.
    pub admitted_by: Source,
    /// The per-lane detail `--explain` prints, carried over from fusion.
    pub lane_contributions: Vec<LaneContribution>,
    /// The cross-encoder's probability, once it has run.
    pub rerank_score: Option<f64>,
}

impl Candidate {
    pub fn key(&self) -> ChunkKey {
        (self.file_id, self.chunk_seq)
    }

    /// Reached only by the graph: no content lane found it.
    ///
    /// This is the count that makes the reserve measurable. A candidate both
    /// lanes found would have entered the pool anyway.
    pub fn graph_only(&self) -> bool {
        self.rrf_rank.is_none() && self.graph_rank.is_some()
    }
}

/// How many slots the pool holds, and how many of them are spoken for.
///
/// The quotas are ceilings, not allocations: a source that has less than its
/// quota gives the remainder back, and the leftovers of every source compete
/// for it. So `budget` is the only number that binds.
#[derive(Debug, Clone, Copy)]
pub struct Reserves {
    /// How many candidates the cross-encoder is shown. §8.6's 64.
    pub budget: usize,
    /// Slots reserved for graph candidates, in reach order. §8.6's 16.
    pub graph: usize,
    /// Slots reserved for date-matching candidates the content order cut.
    ///
    /// **Unmeasured.** No probe covers the temporal lane, so this is the
    /// conservative reading of §8: temporal stops being a voter, becomes a
    /// candidate source, and its effect on ranking is recorded as unknown
    /// rather than claimed.
    pub temporal: usize,
}

/// What the pool was made of. Logged on every query.
///
/// Three separate questions, per #30's acceptance criteria: how many graph
/// candidates existed, how many entered the pool, and how many survived to the
/// output. Logging only the last cannot distinguish weak generation from pool
/// starvation from model rejection.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolShape {
    pub budget: usize,
    pub rrf_available: usize,
    pub graph_available: usize,
    pub admitted: usize,
    pub from_rrf: usize,
    pub from_graph: usize,
    pub from_temporal: usize,
    pub backfilled: usize,
    /// Admitted candidates no content lane found.
    pub graph_only: usize,
}

/// Assemble the shortlist: content anchors in fused order, graph anchors in
/// reach order, then backfill either way to the budget.
///
/// `temporal` is the chunk keys whose note matches the query's date range, best
/// match first. They are promoted out of the leftovers rather than generated:
/// the temporal signal ranks *notes*, and a note is not a passage, so inventing
/// a chunk for it would be putting a number on a guess.
pub fn build_pool(
    rrf: Vec<FusedResult>,
    graph: &[RankedResult],
    temporal: &[ChunkKey],
    reserves: Reserves,
) -> (Vec<Candidate>, PoolShape) {
    let budget = reserves.budget;
    let graph_quota = reserves.graph.min(budget);
    let temporal_quota = reserves.temporal.min(budget - graph_quota);
    let rrf_allowance = budget - graph_quota - temporal_quota;

    let graph_rank: HashMap<ChunkKey, usize> = graph
        .iter()
        .enumerate()
        .map(|(i, r)| (r.chunk_key(), i + 1))
        .collect();
    let dated: HashSet<ChunkKey> = temporal.iter().copied().collect();

    let mut shape = PoolShape {
        budget,
        rrf_available: rrf.len(),
        graph_available: graph.len(),
        ..PoolShape::default()
    };

    let mut pool: Vec<Candidate> = Vec::with_capacity(budget);
    let mut admitted: HashSet<ChunkKey> = HashSet::new();
    let mut leftovers: Vec<Option<Candidate>> = Vec::new();

    // Content anchors, in fused order.
    for (i, fused) in rrf.into_iter().enumerate() {
        let mut candidate = from_fused(fused, i + 1);
        candidate.graph_rank = graph_rank.get(&candidate.key()).copied();
        candidate.temporal = dated.contains(&candidate.key());
        if pool.len() < rrf_allowance {
            admitted.insert(candidate.key());
            pool.push(candidate);
            shape.from_rrf += 1;
        } else {
            leftovers.push(Some(candidate));
        }
    }

    // Graph anchors, in reach order. The routing guarantee.
    let mut graph_taken = 0usize;
    for (i, result) in graph.iter().enumerate() {
        if admitted.contains(&result.chunk_key()) {
            continue;
        }
        let mut candidate = from_ranked(result, Source::Graph);
        candidate.graph_rank = Some(i + 1);
        candidate.temporal = dated.contains(&candidate.key());
        if graph_taken < graph_quota {
            graph_taken += 1;
            admitted.insert(candidate.key());
            pool.push(candidate);
            shape.from_graph += 1;
        } else {
            leftovers.push(Some(candidate));
        }
    }

    // Date-matching candidates the content order cut, best date match first.
    let by_key: HashMap<ChunkKey, usize> = leftovers
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().map(|c| (c.key(), i)))
        .collect();
    let mut temporal_taken = 0usize;
    for key in temporal {
        if temporal_taken >= temporal_quota || pool.len() >= budget {
            break;
        }
        let Some(&index) = by_key.get(key) else {
            continue;
        };
        let Some(mut candidate) = leftovers[index].take() else {
            continue;
        };
        candidate.admitted_by = Source::Temporal;
        temporal_taken += 1;
        admitted.insert(candidate.key());
        pool.push(candidate);
        shape.from_temporal += 1;
    }

    // Backfill either way, in the order the leftovers were produced: content
    // first, then graph. A reserve nobody filled is not a slot left empty.
    for slot in leftovers.iter_mut() {
        if pool.len() >= budget {
            break;
        }
        let Some(mut candidate) = slot.take() else {
            continue;
        };
        candidate.admitted_by = Source::Backfill;
        pool.push(candidate);
        shape.backfilled += 1;
    }

    shape.admitted = pool.len();
    shape.graph_only = pool.iter().filter(|c| c.graph_only()).count();
    (pool, shape)
}

/// Order the pool by what the cross-encoder said.
///
/// Descending by score; unscored candidates sort last. The tie-break is what
/// makes the order reproducible — see [`Tiebreak`].
pub fn sort_by_rerank(pool: &mut [Candidate], tiebreak: Tiebreak) {
    pool.sort_by(|a, b| {
        b.rerank_score
            .partial_cmp(&a.rerank_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| match tiebreak {
                Tiebreak::Rrf => rrf_position(a).cmp(&rrf_position(b)),
                Tiebreak::Identity => std::cmp::Ordering::Equal,
            })
            .then_with(|| a.file_path.cmp(&b.file_path))
            .then_with(|| a.chunk_seq.cmp(&b.chunk_seq))
    });
}

fn rrf_position(candidate: &Candidate) -> usize {
    candidate.rrf_rank.unwrap_or(usize::MAX)
}

/// What a query with no supported answer says (issue #34).
///
/// One string, used by every channel that can carry prose, so "the engine found
/// nothing" reads the same whether it came from the CLI or from a tool call.
pub const NO_RELEVANT_CONTENT: &str = "No relevant content found for this query in the vault.";

/// Drop the candidates the cross-encoder did not support (issue #34).
///
/// The gate #30 made possible: the sort key is a calibrated probability, so
/// there is finally a quantity that means *this passage does not answer the
/// question* rather than *this passage answered it less well than that one*.
/// Before it, `confidence` was `rrf_score / max_score * 100` and the top hit was
/// 100% by construction, whatever it was.
///
/// **Per candidate, not per response.** Gating the top score alone would deliver
/// abstention and nothing else; gating every candidate also stops five
/// confident-looking rows padding the bottom of an answer that has one real
/// row — most of what #4 was filed for. It subsumes the response-level rule,
/// since a pool with nothing above the floor empties, so there is one mechanism
/// rather than two.
///
/// **An unscored candidate is kept.** `rerank_score` is `None` under the legacy
/// stage and under [`degraded_interleave`], where confidence is a *position*.
/// Abstaining on a position is abstaining on nothing, and a consumer can discard
/// a weak block it received where it cannot recover one that was withheld.
///
/// `floor <= 0.0` disables the gate and is the inert control the change is
/// measured against.
///
/// Returns how many candidates were dropped, which is the tail cost the fit is
/// judged on: the floor is fit on best-score-per-query and applied to a whole
/// list, and those are different distributions.
pub fn apply_answer_floor(pool: &mut Vec<Candidate>, floor: f64) -> usize {
    if floor <= 0.0 {
        return 0;
    }
    let before = pool.len();
    pool.retain(|c| c.rerank_score.is_none_or(|score| score >= floor));
    before - pool.len()
}

/// The documented fallback when no cross-encoder is available: three content
/// candidates, then one from the other sources, repeating.
///
/// With no model there is no sort, so the fallback has to be *defined* rather
/// than left as whatever the fused order happened to be — otherwise the
/// reserved candidates, which exist precisely because fusion under-ranks them,
/// are ranked by the thing they were routed around. Each stream keeps its own
/// internal order.
pub fn degraded_interleave(pool: Vec<Candidate>) -> Vec<Candidate> {
    let (mut content, mut reserved): (Vec<Candidate>, Vec<Candidate>) =
        pool.into_iter().partition(|c| c.rrf_rank.is_some());
    content.sort_by_key(rrf_position);
    reserved.sort_by(|a, b| {
        a.graph_rank
            .unwrap_or(usize::MAX)
            .cmp(&b.graph_rank.unwrap_or(usize::MAX))
            .then_with(|| a.file_path.cmp(&b.file_path))
            .then_with(|| a.chunk_seq.cmp(&b.chunk_seq))
    });

    let mut out = Vec::with_capacity(content.len() + reserved.len());
    let mut content = content.into_iter();
    let mut reserved = reserved.into_iter();
    loop {
        let mut moved = false;
        for _ in 0..3 {
            if let Some(c) = content.next() {
                out.push(c);
                moved = true;
            }
        }
        if let Some(r) = reserved.next() {
            out.push(r);
            moved = true;
        }
        if !moved {
            break;
        }
    }
    out
}

/// Render the ranked pool for `--explain` and for the result list.
///
/// `rrf_score` keeps meaning the fused score it always meant — the sort key is
/// the cross-encoder's, and it is reported as its own lane contribution rather
/// than written over a field named after something else. `confidence` is the
/// cross-encoder's probability as a percentage, which is an absolute number for
/// the first time: a query with no good answer no longer reports 100% for
/// whatever it ranked first.
pub fn into_fused(pool: Vec<Candidate>) -> Vec<FusedResult> {
    pool.into_iter()
        .map(|c| {
            let mut lane_contributions = c.lane_contributions;
            if let Some(score) = c.rerank_score {
                lane_contributions.push(LaneContribution {
                    lane_name: "rerank".to_string(),
                    rank: 0,
                    raw_score: score,
                    weighted_contribution: score,
                    detail: Some(provenance(&c.admitted_by, c.graph_rank)),
                });
            }
            FusedResult {
                file_path: c.file_path,
                file_id: c.file_id,
                chunk_seq: c.chunk_seq,
                rrf_score: c.rrf_score,
                heading: c.heading,
                snippet: c.snippet,
                docid: c.docid,
                lane_contributions,
                confidence: c.rerank_score.map(|s| s * 100.0).unwrap_or(0.0),
            }
        })
        .collect()
}

/// Confidence for the degraded order: a **position**, not a probability.
///
/// Nothing calibrated this order, so there is no number to report — and the two
/// streams are not on one scale anyway, since a reserved candidate has no fused
/// score at all and would read as 0% beside the content candidates it was
/// deliberately interleaved with. Position is the only thing the degraded order
/// actually knows, so position is what it says.
///
/// Layer 2 of the convergence plan replaces the percentage with provenance and
/// a status field, which is where this stops needing a caveat.
pub fn degraded_confidence(results: &mut [FusedResult]) {
    let total = results.len() as f64;
    for (i, r) in results.iter_mut().enumerate() {
        r.confidence = if total > 0.0 {
            ((total - i as f64) / total) * 100.0
        } else {
            0.0
        };
    }
}

fn provenance(source: &Source, graph_rank: Option<usize>) -> String {
    match (source, graph_rank) {
        (Source::Graph, Some(rank)) => format!("graph reserve, reach #{rank}"),
        (Source::Temporal, _) => "temporal reserve".to_string(),
        (Source::Backfill, Some(rank)) => format!("backfill, reach #{rank}"),
        (Source::Backfill, None) => "backfill".to_string(),
        (Source::Rrf, Some(rank)) => format!("content, also reach #{rank}"),
        (Source::Rrf, None) => "content".to_string(),
        (Source::Graph, None) => "graph reserve".to_string(),
    }
}

fn from_fused(fused: FusedResult, rank: usize) -> Candidate {
    Candidate {
        file_path: fused.file_path,
        file_id: fused.file_id,
        chunk_seq: fused.chunk_seq,
        heading: fused.heading,
        snippet: fused.snippet,
        docid: fused.docid,
        rrf_rank: Some(rank),
        rrf_score: fused.rrf_score,
        graph_rank: None,
        temporal: false,
        admitted_by: Source::Rrf,
        lane_contributions: fused.lane_contributions,
        rerank_score: None,
    }
}

fn from_ranked(result: &RankedResult, source: Source) -> Candidate {
    Candidate {
        file_path: result.file_path.clone(),
        file_id: result.file_id,
        chunk_seq: result.chunk_seq,
        heading: result.heading.clone(),
        snippet: result.snippet.clone(),
        docid: result.docid.clone(),
        rrf_rank: None,
        rrf_score: 0.0,
        graph_rank: None,
        temporal: false,
        admitted_by: source,
        lane_contributions: Vec::new(),
        rerank_score: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct paths need distinct ids: the pool dedups on
    /// `(file_id, chunk_seq)`, so a fixture that hashes paths into collisions
    /// would merge candidates and read as a routing bug.
    const CONTENT_IDS: i64 = 1_000;
    const GRAPH_IDS: i64 = 2_000;

    fn fused_at(file_id: i64, path: &str, chunk_seq: i64, rrf_score: f64) -> FusedResult {
        FusedResult {
            file_path: path.to_string(),
            file_id,
            chunk_seq,
            rrf_score,
            heading: None,
            snippet: format!("{path}#{chunk_seq}"),
            docid: None,
            lane_contributions: Vec::new(),
            confidence: 0.0,
        }
    }

    fn ranked_at(file_id: i64, path: &str, chunk_seq: i64, score: f64) -> RankedResult {
        RankedResult {
            file_path: path.to_string(),
            file_id,
            chunk_seq,
            score,
            heading: None,
            snippet: format!("{path}#{chunk_seq}"),
            docid: None,
        }
    }

    fn fused(path: &str, chunk_seq: i64, rrf_score: f64) -> FusedResult {
        fused_at(9_000 + path.len() as i64, path, chunk_seq, rrf_score)
    }

    fn ranked(path: &str, chunk_seq: i64, score: f64) -> RankedResult {
        ranked_at(9_000 + path.len() as i64, path, chunk_seq, score)
    }

    fn content(n: usize) -> Vec<FusedResult> {
        (0..n)
            .map(|i| {
                let i = i as i64;
                fused_at(
                    CONTENT_IDS + i,
                    &format!("c{i}.md"),
                    0,
                    1.0 - i as f64 / 1000.0,
                )
            })
            .collect()
    }

    fn graph(n: usize) -> Vec<RankedResult> {
        (0..n)
            .map(|i| {
                let i = i as i64;
                ranked_at(
                    GRAPH_IDS + i,
                    &format!("g{i}.md"),
                    0,
                    1.0 - i as f64 / 1000.0,
                )
            })
            .collect()
    }

    fn reserves() -> Reserves {
        Reserves {
            budget: 64,
            graph: 16,
            temporal: 0,
        }
    }

    /// The defect #30 exists for. A lane that loses the fusion arithmetic used
    /// to reach the model on none of its results; now sixteen slots are its
    /// regardless of what fusion thought.
    #[test]
    fn the_graph_lane_reaches_the_model_however_fusion_ranked_it() {
        let (pool, shape) = build_pool(content(200), &graph(30), &[], reserves());

        assert_eq!(pool.len(), 64, "the budget is the only number that binds");
        assert_eq!(shape.from_graph, 16, "the reserve was not filled");
        assert_eq!(shape.from_rrf, 48);
        assert_eq!(shape.graph_only, 16);
    }

    /// A reserve is a ceiling, not an allocation: a lane with three results
    /// takes three slots and gives thirteen back.
    #[test]
    fn an_unfilled_reserve_is_given_back_rather_than_left_empty() {
        let (pool, shape) = build_pool(content(200), &graph(3), &[], reserves());

        assert_eq!(pool.len(), 64);
        assert_eq!(shape.from_graph, 3);
        assert_eq!(shape.from_rrf, 48);
        assert_eq!(shape.backfilled, 13, "the unused reserve went to content");
        assert!(pool.iter().all(|c| c.rrf_rank.is_some() || c.graph_only()));
    }

    /// The mirror case: thin content, and the graph backfills past its quota.
    #[test]
    fn backfill_runs_either_way() {
        let (pool, shape) = build_pool(content(10), &graph(100), &[], reserves());

        assert_eq!(pool.len(), 64);
        assert_eq!(shape.from_rrf, 10);
        assert_eq!(shape.from_graph, 16);
        assert_eq!(shape.backfilled, 38);
    }

    /// A candidate both sources found is one candidate, and it is not
    /// graph-only — which is the distinction the reserve is measured by.
    #[test]
    fn a_candidate_both_sources_found_takes_one_slot_and_keeps_both_ranks() {
        let mut rrf = vec![fused_at(7, "shared.md", 2, 0.9)];
        rrf.extend(content(5));
        let graph = vec![
            ranked_at(7, "shared.md", 2, 0.8),
            ranked_at(GRAPH_IDS, "g0.md", 0, 0.7),
        ];

        let (pool, shape) = build_pool(rrf, &graph, &[], reserves());

        let shared: Vec<&Candidate> = pool.iter().filter(|c| c.file_path == "shared.md").collect();
        assert_eq!(shared.len(), 1, "admitted twice");
        assert_eq!(shared[0].rrf_rank, Some(1));
        assert_eq!(shared[0].graph_rank, Some(1), "its reach was not recorded");
        assert!(!shared[0].graph_only());
        assert_eq!(shape.graph_only, 1, "only g0.md is graph-only");
    }

    /// The temporal reserve promotes a note the content order cut, and takes
    /// its slots out of the content share rather than the graph's.
    #[test]
    fn the_temporal_reserve_promotes_a_dated_note_the_content_order_cut() {
        let mut rrf = content(20);
        rrf.push(fused_at(42, "dated.md", 0, 0.001));
        let key = (42i64, 0i64);

        let reserves = Reserves {
            budget: 8,
            graph: 2,
            temporal: 2,
        };
        let (pool, shape) = build_pool(rrf, &graph(4), &[key], reserves);

        assert_eq!(shape.from_rrf, 4, "budget 8 less two reserves of two");
        assert_eq!(shape.from_temporal, 1);
        assert_eq!(shape.from_graph, 2);
        let promoted = pool.iter().find(|c| c.file_path == "dated.md").unwrap();
        assert_eq!(promoted.admitted_by, Source::Temporal);
        assert!(promoted.temporal);
    }

    /// Ranking is the model's job. Nothing about fusion survives into the order
    /// except as a tie-break.
    #[test]
    fn the_cross_encoder_sorts_and_fusion_does_not_reblend() {
        let mut pool = vec![
            Candidate {
                rerank_score: Some(0.2),
                ..from_fused(fused("top-of-rrf.md", 0, 9.9), 1)
            },
            Candidate {
                rerank_score: Some(0.9),
                ..from_fused(fused("bottom-of-rrf.md", 0, 0.1), 40)
            },
        ];
        sort_by_rerank(&mut pool, Tiebreak::Rrf);

        assert_eq!(pool[0].file_path, "bottom-of-rrf.md");
        assert_eq!(pool[1].file_path, "top-of-rrf.md");
    }

    /// The tie-break decides only exact ties, and it is the one place fused
    /// rank is still read.
    #[test]
    fn ties_break_on_fused_rank_and_then_on_identity() {
        let tied = |path: &str, rank: usize| Candidate {
            rerank_score: Some(0.5),
            ..from_fused(fused(path, 0, 1.0), rank)
        };
        let mut pool = vec![tied("b.md", 3), tied("a.md", 7), tied("c.md", 1)];

        sort_by_rerank(&mut pool, Tiebreak::Rrf);
        let paths: Vec<&str> = pool.iter().map(|c| c.file_path.as_str()).collect();
        assert_eq!(paths, vec!["c.md", "b.md", "a.md"], "fused rank decides");

        sort_by_rerank(&mut pool, Tiebreak::Identity);
        let paths: Vec<&str> = pool.iter().map(|c| c.file_path.as_str()).collect();
        assert_eq!(paths, vec!["a.md", "b.md", "c.md"], "identity decides");
    }

    /// An unscored candidate sorts last rather than at 0.0 among the scored —
    /// the two are not the same claim.
    #[test]
    fn an_unscored_candidate_sorts_last() {
        let mut pool = vec![
            Candidate {
                rerank_score: None,
                ..from_fused(fused("unscored.md", 0, 9.9), 1)
            },
            Candidate {
                rerank_score: Some(0.01),
                ..from_fused(fused("scored.md", 0, 0.1), 2)
            },
        ];
        sort_by_rerank(&mut pool, Tiebreak::Rrf);
        assert_eq!(pool[0].file_path, "scored.md");
    }

    #[test]
    fn the_degraded_order_is_three_content_then_one_reserved() {
        let mut pool: Vec<Candidate> = (0..7)
            .map(|i| from_fused(fused(&format!("c{i}.md"), 0, 1.0), i + 1))
            .collect();
        for i in 0..3 {
            let mut c = from_ranked(&ranked(&format!("g{i}.md"), 0, 1.0), Source::Graph);
            c.graph_rank = Some(i as usize + 1);
            pool.push(c);
        }

        let order: Vec<String> = degraded_interleave(pool)
            .into_iter()
            .map(|c| c.file_path)
            .collect();
        assert_eq!(
            order,
            vec![
                "c0.md", "c1.md", "c2.md", "g0.md", "c3.md", "c4.md", "c5.md", "g1.md", "c6.md",
                "g2.md",
            ]
        );
    }

    /// One stream running out must not stop the other.
    #[test]
    fn the_degraded_order_drains_both_streams() {
        let pool: Vec<Candidate> = (0..2)
            .map(|i| {
                let mut c = from_ranked(&ranked(&format!("g{i}.md"), 0, 1.0), Source::Graph);
                c.graph_rank = Some(i as usize + 1);
                c
            })
            .collect();
        assert_eq!(degraded_interleave(pool).len(), 2);

        let pool: Vec<Candidate> = (0..5)
            .map(|i| from_fused(fused(&format!("c{i}.md"), 0, 1.0), i + 1))
            .collect();
        assert_eq!(degraded_interleave(pool).len(), 5);

        assert!(degraded_interleave(vec![]).is_empty());
    }

    /// Probe 5's defect, in one assertion: the top result of a query with no
    /// good answer used to report 100% because confidence was renormalised per
    /// query. The cross-encoder's probability is absolute.
    #[test]
    fn confidence_is_the_models_own_number_not_a_share_of_the_best_one() {
        let pool = vec![
            Candidate {
                rerank_score: Some(0.04),
                ..from_fused(fused("nothing-relevant.md", 0, 0.5), 1)
            },
            Candidate {
                rerank_score: Some(0.02),
                ..from_fused(fused("also-nothing.md", 0, 0.4), 2)
            },
        ];
        let results = into_fused(pool);
        assert!((results[0].confidence - 4.0).abs() < 1e-9);
        assert!((results[1].confidence - 2.0).abs() < 1e-9);
    }

    /// `--explain` has to keep showing what fusion said; the cross-encoder is
    /// reported beside it, not written over it.
    #[test]
    fn the_fused_score_survives_into_explain_beside_the_models() {
        let pool = vec![Candidate {
            rerank_score: Some(0.75),
            ..from_fused(fused("a.md", 3, 0.0312), 4)
        }];
        let results = into_fused(pool);

        assert!((results[0].rrf_score - 0.0312).abs() < 1e-9);
        let rerank = results[0]
            .lane_contributions
            .iter()
            .find(|l| l.lane_name == "rerank")
            .expect("the sort key is not in the explain output");
        assert!((rerank.raw_score - 0.75).abs() < 1e-9);
    }

    /// A budget smaller than the reserves is a configuration, not a panic.
    #[test]
    fn a_budget_below_the_reserves_still_produces_a_pool() {
        let reserves = Reserves {
            budget: 4,
            graph: 16,
            temporal: 8,
        };
        let (pool, shape) = build_pool(content(50), &graph(50), &[], reserves);
        assert_eq!(pool.len(), 4);
        assert_eq!(shape.from_graph, 4, "the graph takes the whole budget");
        assert_eq!(shape.from_rrf, 0);
    }

    fn scored(path: &str, rank: usize, score: Option<f64>) -> Candidate {
        Candidate {
            rerank_score: score,
            ..from_fused(fused(path, 0, 1.0), rank)
        }
    }

    /// Probe 5's defect, as behaviour rather than as a number. `quantum banking
    /// regulations` scores 0.29% on its best candidate against 91.7% for the
    /// weakest positive in the pool; below the floor nothing is supported and
    /// the engine has to be able to say so.
    #[test]
    fn a_query_with_nothing_above_the_floor_supports_nothing() {
        let mut pool = vec![
            scored("nothing-relevant.md", 1, Some(0.0029)),
            scored("also-nothing.md", 2, Some(0.0010)),
        ];
        let dropped = apply_answer_floor(&mut pool, 0.89);

        assert_eq!(dropped, 2);
        assert!(pool.is_empty(), "the engine answered anyway");
    }

    /// The gate is per candidate, so a query with one real answer keeps the
    /// answer and loses the tail. P6 is the worked example: its answer scores
    /// 91.7% and its second result scores 9.1%.
    #[test]
    fn the_gate_truncates_the_tail_of_a_query_that_does_have_an_answer() {
        let mut pool = vec![
            scored("mend-object.md", 1, Some(0.9174)),
            scored("waterproof.md", 2, Some(0.0905)),
            scored("harden.md", 3, Some(0.0803)),
        ];
        let dropped = apply_answer_floor(&mut pool, 0.89);

        assert_eq!(dropped, 2);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].file_path, "mend-object.md");
    }

    /// The inert control. `answer_floor = 0.0` must leave the ranking exactly as
    /// #30 left it — that is what makes the probe tables either side comparable.
    #[test]
    fn a_floor_of_zero_is_a_no_op() {
        let candidates = || {
            vec![
                scored("a.md", 1, Some(0.99)),
                scored("b.md", 2, Some(0.0001)),
                scored("c.md", 3, Some(0.0)),
            ]
        };
        let mut pool = candidates();
        let dropped = apply_answer_floor(&mut pool, 0.0);

        assert_eq!(dropped, 0);
        let paths: Vec<&str> = pool.iter().map(|c| c.file_path.as_str()).collect();
        assert_eq!(paths, vec!["a.md", "b.md", "c.md"]);
    }

    /// Confidence is a *position* under the degraded interleave and under the
    /// legacy stage, and a position is not a probability. Thresholding one is
    /// thresholding nothing, so the gate stands down rather than guessing.
    #[test]
    fn an_unscored_candidate_is_not_gated() {
        let mut pool = vec![
            scored("degraded-1.md", 1, None),
            scored("degraded-2.md", 2, None),
        ];
        let dropped = apply_answer_floor(&mut pool, 0.89);

        assert_eq!(dropped, 0);
        assert_eq!(pool.len(), 2, "abstained on an order nothing calibrated");
    }

    /// The floor is a floor, not a strict bound: a candidate exactly at it is
    /// supported. Fitting a threshold and then applying a different one is the
    /// off-by-one that would make the pool table describe a build that does not
    /// exist.
    #[test]
    fn a_candidate_exactly_at_the_floor_survives() {
        let mut pool = vec![scored("exact.md", 1, Some(0.89))];
        assert_eq!(apply_answer_floor(&mut pool, 0.89), 0);
        assert_eq!(pool.len(), 1);
    }

    /// Order is the model's, and the gate is a filter — it removes rows without
    /// touching the arrangement of the ones it keeps.
    #[test]
    fn the_gate_preserves_the_models_order() {
        let mut pool = vec![
            scored("first.md", 9, Some(0.99)),
            scored("cut.md", 1, Some(0.10)),
            scored("second.md", 4, Some(0.95)),
        ];
        sort_by_rerank(&mut pool, Tiebreak::Rrf);
        apply_answer_floor(&mut pool, 0.89);

        let paths: Vec<&str> = pool.iter().map(|c| c.file_path.as_str()).collect();
        assert_eq!(paths, vec!["first.md", "second.md"]);
    }

    #[test]
    fn an_empty_retrieval_produces_an_empty_pool() {
        let (pool, shape) = build_pool(vec![], &[], &[], reserves());
        assert!(pool.is_empty());
        assert_eq!(shape.admitted, 0);
    }
}
