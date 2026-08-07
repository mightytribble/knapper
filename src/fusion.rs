/// Reciprocal Rank Fusion (RRF) engine.
///
/// Merges ranked results from multiple search lanes (e.g. semantic vector
/// and FTS5 keyword search) into a single ranked list using the RRF formula:
///
///   rrf_score = sum( weight_i / (k + rank_i) )
///
/// A ranked result from a single search lane.
#[derive(Clone)]
pub struct RankedResult {
    pub file_path: String,
    pub file_id: i64,
    /// Which chunk of the file this result is about, 0-based.
    ///
    /// Fusion keys on `(file_id, chunk_seq)`, so a lane that leaves this at 0 for
    /// results it cannot attribute is claiming they are all the file's first
    /// chunk. Lanes that rank whole files rather than chunks — graph, temporal —
    /// have to name a chunk before fusing.
    pub chunk_seq: i64,
    pub score: f64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
}

impl RankedResult {
    /// The key fusion and dedup group on.
    pub fn chunk_key(&self) -> (i64, i64) {
        (self.file_id, self.chunk_seq)
    }
}

/// A fused result after RRF merging across lanes.
pub struct FusedResult {
    pub file_path: String,
    pub file_id: i64,
    pub chunk_seq: i64,
    pub rrf_score: f64,
    pub heading: Option<String>,
    pub snippet: String,
    pub docid: Option<String>,
    pub lane_contributions: Vec<LaneContribution>,
    pub confidence: f64, // 0-100% normalized score
}

/// Per-lane contribution details for --explain output.
pub struct LaneContribution {
    pub lane_name: String,
    pub rank: usize,
    pub raw_score: f64,
    pub weighted_contribution: f64,
    pub detail: Option<String>, // e.g., "1-hop from BRE-2579"
}

use std::collections::HashMap;

/// Fuse ranked results from multiple search lanes using Reciprocal Rank Fusion.
///
/// Each lane is a tuple of `(lane_name, results, weight)`.
/// Results are grouped by `(file_id, chunk_seq)` — one entry per chunk, so a
/// document with several relevant sections can occupy several result slots.
/// The best snippet/heading per chunk is kept from the highest-ranked lane.
///
/// `k` is the RRF constant (typically 60).
pub fn rrf_fuse(lanes: &[(&str, &[RankedResult], f64)], k: usize) -> Vec<FusedResult> {
    // Track per-chunk: rrf_score, best snippet info, lane contributions
    struct Accumulator {
        file_path: String,
        file_id: i64,
        chunk_seq: i64,
        rrf_score: f64,
        heading: Option<String>,
        snippet: String,
        docid: Option<String>,
        best_rank: usize, // lowest rank seen (for picking best snippet)
        lane_contributions: Vec<LaneContribution>,
    }

    let mut acc_map: HashMap<(i64, i64), Accumulator> = HashMap::new();

    for &(lane_name, results, weight) in lanes {
        for (idx, r) in results.iter().enumerate() {
            let rank = idx + 1; // 1-based
            let contribution = weight / (k as f64 + rank as f64);

            let acc = acc_map.entry(r.chunk_key()).or_insert_with(|| Accumulator {
                file_path: r.file_path.clone(),
                file_id: r.file_id,
                chunk_seq: r.chunk_seq,
                rrf_score: 0.0,
                heading: r.heading.clone(),
                snippet: r.snippet.clone(),
                docid: r.docid.clone(),
                best_rank: rank,
                lane_contributions: Vec::new(),
            });

            acc.rrf_score += contribution;

            // Keep snippet from the best-ranked appearance
            if rank < acc.best_rank {
                acc.best_rank = rank;
                acc.heading = r.heading.clone();
                acc.snippet = r.snippet.clone();
                if r.docid.is_some() {
                    acc.docid = r.docid.clone();
                }
            }

            acc.lane_contributions.push(LaneContribution {
                lane_name: lane_name.to_string(),
                rank,
                raw_score: r.score,
                weighted_contribution: contribution,
                detail: None,
            });
        }
    }

    let mut results: Vec<FusedResult> = acc_map
        .into_values()
        .map(|a| FusedResult {
            file_path: a.file_path,
            file_id: a.file_id,
            chunk_seq: a.chunk_seq,
            rrf_score: a.rrf_score,
            heading: a.heading,
            snippet: a.snippet,
            docid: a.docid,
            lane_contributions: a.lane_contributions,
            confidence: 0.0,
        })
        .collect();

    // Sort by rrf_score descending. Ties break on the chunk key, not on hash
    // order: RRF scores collide constantly (every lane hands out the same
    // `weight/(k+rank)` values), and without this the same query against the
    // same index returns a different top 20 on each run.
    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file_path.cmp(&b.file_path))
            .then_with(|| a.chunk_seq.cmp(&b.chunk_seq))
    });

    // Normalize confidence as percentage of max score
    let max_score = results.first().map(|r| r.rrf_score).unwrap_or(1.0);
    for r in &mut results {
        r.confidence = if max_score > 0.0 {
            (r.rrf_score / max_score) * 100.0
        } else {
            0.0
        };
    }

    results
}

/// Keep at most `cap` results per file, preserving rank order.
///
/// Chunk-level fusion is what lets one document contribute several sections; this
/// is the guard rail that stops a long document from contributing all of them.
/// `cap_per_file(results, 1)` is exactly the file-level grouping engraph did
/// before chunks were addressable, which is how `group_by = "file"` is served.
///
/// Expects `results` sorted best-first, as [`rrf_fuse`] returns them. A cap of 0
/// is treated as unlimited.
pub fn cap_per_file(results: Vec<FusedResult>, cap: usize) -> Vec<FusedResult> {
    if cap == 0 {
        return results;
    }
    let mut seen: HashMap<i64, usize> = HashMap::new();
    results
        .into_iter()
        .filter(|r| {
            let count = seen.entry(r.file_id).or_insert(0);
            *count += 1;
            *count <= cap
        })
        .collect()
}

/// Format explain output for a single fused result.
pub fn format_explain(result: &FusedResult) -> String {
    let mut out = format!("  RRF: {:.4}\n", result.rrf_score);
    for lc in &result.lane_contributions {
        let detail_str = lc
            .detail
            .as_deref()
            .map(|d| format!(" ({})", d))
            .unwrap_or_default();
        out += &format!(
            "    {}: rank #{}, raw {:.2}{}, +{:.4}\n",
            lc.lane_name, lc.rank, lc.raw_score, detail_str, lc.weighted_contribution
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable pseudo-id per path, so two lanes naming the same file agree on the
    /// fusion key without the tests having to thread ids around.
    fn fake_file_id(file_path: &str) -> i64 {
        file_path.bytes().map(|b| b as i64).sum()
    }

    fn make_result(file_path: &str, score: f64) -> RankedResult {
        make_chunk_result(file_path, 0, score)
    }

    fn make_chunk_result(file_path: &str, chunk_seq: i64, score: f64) -> RankedResult {
        RankedResult {
            file_path: file_path.to_string(),
            file_id: fake_file_id(file_path),
            chunk_seq,
            score,
            heading: Some(format!("heading {chunk_seq} for {file_path}")),
            snippet: format!("snippet {chunk_seq} for {file_path}"),
            docid: None,
        }
    }

    #[test]
    fn test_rrf_basic() {
        // Item appearing in both lanes should rank highest
        let semantic = vec![
            make_result("both.md", 0.87),
            make_result("sem_only.md", 0.75),
        ];
        let fts = vec![make_result("fts_only.md", 5.0), make_result("both.md", 3.2)];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0), ("fts", &fts, 1.0)], 60);

        assert_eq!(fused.len(), 3);
        // "both.md" should be first because it appears in both lanes
        assert_eq!(fused[0].file_path, "both.md");

        // Verify the RRF score for "both.md":
        // semantic rank 1: 1.0 / (60 + 1) = 0.01639...
        // fts rank 2: 1.0 / (60 + 2) = 0.01613...
        // total = 0.03252...
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].rrf_score - expected).abs() < 1e-10);

        // Both single-lane items should have lower scores
        assert!(fused[0].rrf_score > fused[1].rrf_score);
        assert!(fused[0].rrf_score > fused[2].rrf_score);

        // "both.md" should have 2 lane contributions
        assert_eq!(fused[0].lane_contributions.len(), 2);
    }

    #[test]
    fn test_rrf_weighted() {
        // FTS weighted 3x should make FTS-only item win over semantic-only item
        let semantic = vec![make_result("sem.md", 0.95)];
        let fts = vec![make_result("fts.md", 8.0)];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0), ("fts", &fts, 3.0)], 60);

        assert_eq!(fused.len(), 2);
        // FTS item at rank 1 with weight 3.0: 3.0 / 61 = 0.04918...
        // Semantic item at rank 1 with weight 1.0: 1.0 / 61 = 0.01639...
        assert_eq!(fused[0].file_path, "fts.md");
        assert_eq!(fused[1].file_path, "sem.md");

        let fts_expected = 3.0 / 61.0;
        let sem_expected = 1.0 / 61.0;
        assert!((fused[0].rrf_score - fts_expected).abs() < 1e-10);
        assert!((fused[1].rrf_score - sem_expected).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_single_lane() {
        let semantic = vec![
            make_result("a.md", 0.9),
            make_result("b.md", 0.8),
            make_result("c.md", 0.7),
        ];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0)], 60);

        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].file_path, "a.md");
        assert_eq!(fused[1].file_path, "b.md");
        assert_eq!(fused[2].file_path, "c.md");

        // Each should have exactly 1 lane contribution
        for f in &fused {
            assert_eq!(f.lane_contributions.len(), 1);
            assert_eq!(f.lane_contributions[0].lane_name, "semantic");
        }
    }

    #[test]
    fn test_format_explain() {
        let result = FusedResult {
            file_path: "test.md".to_string(),
            file_id: 1,
            chunk_seq: 0,
            rrf_score: 0.0328,
            heading: None,
            snippet: "test".to_string(),
            docid: None,
            confidence: 100.0,
            lane_contributions: vec![
                LaneContribution {
                    lane_name: "semantic".to_string(),
                    rank: 1,
                    raw_score: 0.87,
                    weighted_contribution: 0.0164,
                    detail: None,
                },
                LaneContribution {
                    lane_name: "fts".to_string(),
                    rank: 3,
                    raw_score: 5.23,
                    weighted_contribution: 0.0159,
                    detail: None,
                },
            ],
        };

        let output = format_explain(&result);
        assert!(output.contains("RRF: 0.0328"));
        assert!(output.contains("semantic: rank #1, raw 0.87, +0.0164"));
        assert!(output.contains("fts: rank #3, raw 5.23, +0.0159"));
    }

    #[test]
    fn two_sections_of_one_file_are_two_results() {
        // The behaviour #6 exists to change: previously the second chunk of
        // `spells.md` was discarded and only one could reach the results.
        let semantic = vec![
            make_chunk_result("spells.md", 9, 0.81),
            make_chunk_result("spells.md", 3, 0.79),
            make_chunk_result("other.md", 0, 0.60),
        ];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0)], 60);

        assert_eq!(fused.len(), 3);
        assert_eq!(
            (fused[0].file_path.as_str(), fused[0].chunk_seq),
            ("spells.md", 9)
        );
        assert_eq!(
            (fused[1].file_path.as_str(), fused[1].chunk_seq),
            ("spells.md", 3)
        );
        // Each keeps its own snippet — they are different sections.
        assert_ne!(fused[0].snippet, fused[1].snippet);
    }

    #[test]
    fn lanes_agreeing_on_a_chunk_reinforce_it() {
        // Two lanes naming the same section must add up on one entry; two lanes
        // naming different sections of one file must not.
        let semantic = vec![make_chunk_result("spells.md", 3, 0.8)];
        let fts = vec![make_chunk_result("spells.md", 3, 4.0)];

        let fused = rrf_fuse(&[("semantic", &semantic, 1.0), ("fts", &fts, 1.0)], 60);

        assert_eq!(fused.len(), 1, "same chunk, one result");
        assert_eq!(fused[0].lane_contributions.len(), 2);
        assert!((fused[0].rrf_score - (1.0 / 61.0 + 1.0 / 61.0)).abs() < 1e-10);
    }

    #[test]
    fn cap_per_file_bounds_one_documents_share() {
        let results = rrf_fuse(
            &[(
                "semantic",
                &[
                    make_chunk_result("long.md", 0, 0.9),
                    make_chunk_result("long.md", 1, 0.8),
                    make_chunk_result("long.md", 2, 0.7),
                    make_chunk_result("other.md", 0, 0.6),
                ][..],
                1.0,
            )],
            60,
        );

        let capped = cap_per_file(results, 2);
        assert_eq!(capped.len(), 3);
        assert_eq!(
            capped.iter().filter(|r| r.file_path == "long.md").count(),
            2
        );
        assert_eq!(capped[2].file_path, "other.md");
        // Rank order within the surviving results is untouched.
        assert!(capped[0].rrf_score >= capped[1].rrf_score);
    }

    #[test]
    fn cap_of_one_is_file_level_grouping() {
        let results = rrf_fuse(
            &[(
                "semantic",
                &[
                    make_chunk_result("a.md", 5, 0.9),
                    make_chunk_result("a.md", 1, 0.8),
                    make_chunk_result("b.md", 0, 0.7),
                ][..],
                1.0,
            )],
            60,
        );

        let grouped = cap_per_file(results, 1);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].file_path, "a.md");
        assert_eq!(
            grouped[0].chunk_seq, 5,
            "the file is represented by its best section"
        );
        assert_eq!(grouped[1].file_path, "b.md");
    }

    #[test]
    fn cap_of_zero_is_unlimited() {
        let results = rrf_fuse(
            &[(
                "semantic",
                &[
                    make_chunk_result("a.md", 0, 0.9),
                    make_chunk_result("a.md", 1, 0.8),
                    make_chunk_result("a.md", 2, 0.7),
                ][..],
                1.0,
            )],
            60,
        );
        assert_eq!(cap_per_file(results, 0).len(), 3);
    }

    #[test]
    fn test_rrf_empty_lanes() {
        let fused = rrf_fuse(&[], 60);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_empty_results() {
        let empty: Vec<RankedResult> = vec![];
        let fused = rrf_fuse(&[("semantic", &empty, 1.0), ("fts", &empty, 1.0)], 60);
        assert!(fused.is_empty());
    }
}
