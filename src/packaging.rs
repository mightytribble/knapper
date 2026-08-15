//! The output contract (#35): one `SearchEnvelope` that every surface renders.
//!
//! `search` returns the passage the cross-encoder scored, bounded by a token
//! budget, with provenance in place of numeric scores on the machine channels.

use crate::fusion::LaneContribution;

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
}
