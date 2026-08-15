//! The output contract (#35): one `SearchEnvelope` that every surface renders.
//!
//! `search` returns the passage the cross-encoder scored, bounded by a token
//! budget, with provenance in place of numeric scores on the machine channels.

use crate::fusion::LaneContribution;
use crate::search::InternalSearchResult;

const PER_BLOCK_OVERHEAD: usize = 50; // §9.1: the framing a block costs beyond its text.

/// Whether a search found anything to return (#35).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStatus {
    Ok,
    NoResults,
}

/// A result included in full, with its scored text (#35).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Block {
    pub id: String,
    pub path: String,
    pub heading_path: String,
    pub provenance: Provenance,
    pub text: String,
    pub untrusted_content: bool,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// A result named but not included, because the budget ran out (#35).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Summary {
    pub id: String,
    pub path: String,
    pub heading_path: String,
    pub provenance: Provenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// The one shape every surface renders a search through (#35).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchEnvelope {
    pub status: SearchStatus,
    pub degraded: bool,
    pub warnings: Vec<String>,
    pub blocks: Vec<Block>,
    pub overflow: Vec<Summary>,
}

/// The knobs `assemble` reads; everything else about a search stays in
/// `InternalSearchResult` (#35).
pub struct AssembleParams {
    pub budget_tokens: u32,
    pub full: bool,
    pub summaries: bool,
    pub degraded: bool,
    pub per_note_cap: usize,
}

/// Which lanes account for a result, the machine channels' answer in place of
/// a number.
///
/// `keyword` and `semantic` come from the content lanes' contributions;
/// `graph` is set when the graph lane introduced the candidate. `linked_from`
/// is the seed paths that reached it and ships empty — populating it needs the
/// graph lane to attribute seeds per candidate (#74).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Provenance {
    pub keyword: bool,
    pub semantic: bool,
    pub graph: bool,
    pub linked_from: Vec<String>,
}

impl Provenance {
    /// Derive provenance from the fused lane contributions and a graph flag the
    /// caller computed from `admitted_by` / `graph_rank`.
    pub fn derive(lanes: &[LaneContribution], graph: bool) -> Provenance {
        let has = |name: &str| lanes.iter().any(|l| l.lane_name == name);
        Provenance {
            keyword: has("fts"),
            semantic: has("semantic"),
            graph: graph || has("graph"),
            linked_from: Vec::new(),
        }
    }
}

/// `ceil(chars / 3.33)`, the documented estimate for a result with no scored
/// window to read a reranker's own count from (#35). Matches
/// `RerankModel::count_tokens`'s default; the two are kept separate because
/// `llm.rs` must not depend on `packaging`.
pub fn est_tokens_fallback(text: &str) -> usize {
    (text.chars().count() * 100).div_ceil(333)
}

/// `<docid>#<seq>`, the stable handle for a result (#35).
fn result_id(r: &InternalSearchResult) -> String {
    let docid = r.docid.clone().unwrap_or_else(|| "000000".to_string());
    format!("{docid}#{}", r.chunk_seq)
}

fn summary_of(r: &InternalSearchResult) -> Summary {
    Summary {
        id: result_id(r),
        path: r.file_path.clone(),
        heading_path: r.heading_path.clone(),
        provenance: r.provenance.clone(),
        score: None,
    }
}

fn block_of(r: &InternalSearchResult) -> Block {
    Block {
        id: result_id(r),
        path: r.file_path.clone(),
        heading_path: r.heading_path.clone(),
        provenance: r.provenance.clone(),
        text: r.text.clone(),
        untrusted_content: true,
        truncated: r.truncated,
        score: None,
    }
}

/// Assemble the ranked results into the envelope (#35).
///
/// The included set is a prefix: fill stops at the first result that would
/// break the budget, and that result and every one after it become overflow.
/// The first result is always included. `full` skips the budget; `summaries`
/// emits every rank as a text-less row. `per_note_cap` is inert at 0.
pub fn assemble(results: &[InternalSearchResult], p: AssembleParams) -> SearchEnvelope {
    if results.is_empty() {
        return SearchEnvelope {
            status: SearchStatus::NoResults,
            degraded: p.degraded,
            warnings: Vec::new(),
            blocks: Vec::new(),
            overflow: Vec::new(),
        };
    }

    // Results cap on the assembled set; 0 is unbounded (#30, #34).
    let capped: Vec<&InternalSearchResult> = if p.per_note_cap == 0 {
        results.iter().collect()
    } else {
        let mut per_note: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        results
            .iter()
            .filter(|r| {
                let n = per_note.entry(r.file_id).or_insert(0);
                *n += 1;
                *n <= p.per_note_cap
            })
            .collect()
    };

    let mut blocks = Vec::new();
    let mut overflow = Vec::new();

    if p.summaries {
        overflow = capped.iter().map(|r| summary_of(r)).collect();
        return SearchEnvelope {
            status: SearchStatus::Ok,
            degraded: p.degraded,
            warnings: Vec::new(),
            blocks,
            overflow,
        };
    }

    let mut used = 0usize;
    let mut stopped = false;
    for r in &capped {
        let cost = r.token_count + PER_BLOCK_OVERHEAD;
        if !p.full && !blocks.is_empty() && used + cost > p.budget_tokens as usize {
            stopped = true;
        }
        if stopped {
            overflow.push(summary_of(r));
        } else {
            blocks.push(block_of(r));
            used += cost;
        }
    }

    SearchEnvelope {
        status: SearchStatus::Ok,
        degraded: p.degraded,
        warnings: Vec::new(),
        blocks,
        overflow,
    }
}

fn provenance_label(p: &Provenance) -> String {
    let mut parts = Vec::new();
    if p.semantic {
        parts.push("semantic");
    }
    if p.keyword {
        parts.push("keyword");
    }
    if p.graph {
        parts.push("linked");
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("+")
    }
}

/// Fill each row's `score` from the matching result's confidence — `--scores`
/// only (#35). Degraded rows have no probability, so `score` stays `None`.
pub fn apply_scores(env: &mut SearchEnvelope, results: &[InternalSearchResult]) {
    let by_id: std::collections::HashMap<String, f64> = results
        .iter()
        .map(|r| (result_id(r), r.confidence))
        .collect();
    if env.degraded {
        return;
    }
    for b in &mut env.blocks {
        b.score = by_id.get(&b.id).copied();
    }
    for s in &mut env.overflow {
        s.score = by_id.get(&s.id).copied();
    }
}

/// The convenience text rendering of the envelope (design §9.3).
pub fn render_text(env: &SearchEnvelope, scores: bool) -> String {
    if matches!(env.status, SearchStatus::NoResults) {
        return format!("{}\n", crate::ranking::NO_RELEVANT_CONTENT);
    }
    let mut out = String::new();
    if env.degraded {
        out.push_str("(degraded ordering: no cross-encoder available)\n\n");
    }
    for b in &env.blocks {
        let pct = match (scores, b.score) {
            (true, Some(s)) => format!(" [{s:.0}%]"),
            _ => String::new(),
        };
        out.push_str(&format!(
            "--- [{}]{pct} {} (matched: {})\n",
            b.id,
            b.heading_path,
            provenance_label(&b.provenance)
        ));
        if b.truncated {
            out.push_str("(truncated)\n");
        }
        out.push_str(&b.text);
        out.push_str("\n\n");
    }
    for s in &env.overflow {
        let pct = match (scores, s.score) {
            (true, Some(v)) => format!(" [{v:.0}%]"),
            _ => String::new(),
        };
        out.push_str(&format!(
            "Not included (lower relevance): [{}]{pct} {} (matched: {})\n",
            s.id,
            s.heading_path,
            provenance_label(&s.provenance)
        ));
    }
    out
}

#[cfg(test)]
mod assemble_tests {
    use super::*;
    use crate::search::InternalSearchResult;

    fn result(seq: i64, tokens: usize) -> InternalSearchResult {
        InternalSearchResult {
            file_path: format!("n{seq}.md"),
            file_id: seq,
            chunk_seq: seq,
            score: 0.9,
            confidence: 90.0,
            heading: None,
            snippet: String::new(),
            docid: Some(format!("{seq:06x}")),
            text: "x".repeat(tokens * 3),
            heading_path: format!("n{seq}.md > H"),
            token_count: tokens,
            truncated: false,
            provenance: Provenance {
                keyword: true,
                semantic: false,
                graph: false,
                linked_from: vec![],
            },
        }
    }
    fn params(budget: u32) -> AssembleParams {
        AssembleParams {
            budget_tokens: budget,
            full: false,
            summaries: false,
            degraded: false,
            per_note_cap: 0,
        }
    }

    #[test]
    fn fills_greedily_and_overflows_the_rest() {
        // Each block costs tokens + 50. Budget 260 fits two 80s (130+130=260), third overflows.
        let rs = vec![result(1, 80), result(2, 80), result(3, 80)];
        let env = assemble(&rs, params(260));
        assert_eq!(env.blocks.len(), 2);
        assert_eq!(env.overflow.len(), 1);
        assert_eq!(env.overflow[0].id, "000003#3");
    }

    #[test]
    fn drop_is_a_suffix_not_a_skip() {
        // A big block at rank 2 stops the fill; rank 3 does not sneak in.
        let rs = vec![result(1, 80), result(2, 10_000), result(3, 10)];
        let env = assemble(&rs, params(260));
        assert_eq!(env.blocks.len(), 1);
        assert_eq!(
            env.overflow
                .iter()
                .map(|s| s.id.clone())
                .collect::<Vec<_>>(),
            vec!["000002#2".to_string(), "000003#3".to_string()]
        );
    }

    #[test]
    fn the_first_result_is_always_included() {
        let rs = vec![result(1, 10_000), result(2, 10)];
        let env = assemble(&rs, params(1));
        assert_eq!(env.blocks.len(), 1);
        assert_eq!(env.blocks[0].id, "000001#1");
    }

    #[test]
    fn full_ignores_the_budget() {
        let rs = vec![result(1, 10_000), result(2, 10_000)];
        let mut p = params(100);
        p.full = true;
        let env = assemble(&rs, p);
        assert_eq!(env.blocks.len(), 2);
        assert!(env.overflow.is_empty());
    }

    #[test]
    fn summaries_emits_no_text() {
        let rs = vec![result(1, 10), result(2, 10)];
        let mut p = params(10_000);
        p.summaries = true;
        let env = assemble(&rs, p);
        assert!(env.blocks.is_empty());
        assert_eq!(env.overflow.len(), 2);
    }

    #[test]
    fn no_results_is_its_own_status() {
        let env = assemble(&[], params(8192));
        assert_eq!(env.status, SearchStatus::NoResults);
        assert!(env.blocks.is_empty() && env.overflow.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane(name: &str) -> LaneContribution {
        LaneContribution {
            lane_name: name.to_string(),
            rank: 1,
            raw_score: 0.0,
            weighted_contribution: 0.0,
            detail: None,
        }
    }

    #[test]
    fn content_lanes_map_to_keyword_and_semantic() {
        let p = Provenance::derive(&[lane("semantic"), lane("fts")], false);
        assert_eq!(
            p,
            Provenance {
                keyword: true,
                semantic: true,
                graph: false,
                linked_from: vec![]
            }
        );
    }

    #[test]
    fn a_graph_only_candidate_still_carries_a_provenance() {
        // No lane contributions (sorted-stage graph reserve), graph flag on.
        let p = Provenance::derive(&[], true);
        assert!(p.graph && !p.keyword && !p.semantic);
    }

    #[test]
    fn a_legacy_graph_lane_sets_graph_from_its_contribution() {
        let p = Provenance::derive(&[lane("graph")], false);
        assert!(p.graph);
    }

    /// Pins the rounding: `ceil`, not `floor`, and 3.33 chars per token, not 3.
    #[test]
    fn the_fallback_estimate_rounds_up_at_3_33_chars_per_token() {
        assert_eq!(est_tokens_fallback(&"x".repeat(40)), 13);
        assert_eq!(est_tokens_fallback(""), 0);
        assert_eq!(est_tokens_fallback("x"), 1);
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;

    #[test]
    fn text_render_marks_provenance_and_omits_score_by_default() {
        let env = SearchEnvelope {
            status: SearchStatus::Ok,
            degraded: false,
            warnings: vec![],
            blocks: vec![Block {
                id: "abc#0".into(),
                path: "n.md".into(),
                heading_path: "n.md > H".into(),
                provenance: Provenance {
                    keyword: true,
                    semantic: true,
                    graph: false,
                    linked_from: vec![],
                },
                text: "body".into(),
                untrusted_content: true,
                truncated: false,
                score: None,
            }],
            overflow: vec![],
        };
        let out = render_text(&env, false);
        assert!(out.contains("[abc#0]"));
        assert!(out.contains("semantic+keyword"));
        assert!(!out.contains('%'));
    }

    #[test]
    fn no_results_text_is_the_literal_message() {
        let env = SearchEnvelope {
            status: SearchStatus::NoResults,
            degraded: false,
            warnings: vec![],
            blocks: vec![],
            overflow: vec![],
        };
        assert_eq!(
            render_text(&env, false).trim(),
            crate::ranking::NO_RELEVANT_CONTENT
        );
    }
}
