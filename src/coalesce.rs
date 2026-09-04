//! Coalescing adjacent chunks of one document into a single result block (#39).
//!
//! This is a query-time transform over the ranked result. It uses no model.
//! It runs after the ranking stage. It does not depend on the cross-encoder.
//! It does not depend on the embedder. It reads only each result's file,
//! section ordinal, heading path, score and text.

use crate::packaging::{Provenance, est_tokens_fallback};
use crate::search::InternalSearchResult;

/// Present a document's abutting chunks of one section as one block. It
/// works in place. It works over the ranked result. The block sits at its
/// anchor's rank. The anchor is the first member found. The block carries
/// the strongest member's score. The block is headed by its leading
/// section, the one with the lowest `seq`. The block holds the members'
/// text in document order. The block carries no length limit.
///
/// The block grows to a contiguous `seq` run **within one section and its
/// descendants** (#101). The merge exists because a section is one topic,
/// so a weaker follow-on chunk still carries that topic's context; a
/// subsection subdivides the same topic and keeps the premise, while a
/// sibling section starts a new one and breaks it. So the run holds a
/// `root` heading path, the shallowest its members reach, and admits a
/// member whose path is the root or descends from it. A member whose path
/// the root descends from widens the root instead, and the loop runs to a
/// fixed point, so which member was encountered first cannot decide the
/// block. Before #101 the run was `seq` alone, which merged across a
/// heading and then labelled the block with a section that had not
/// matched.
pub fn coalesce_adjacent(results: Vec<InternalSearchResult>) -> Vec<InternalSearchResult> {
    blocks(&results)
        .into_iter()
        .map(|members| {
            if members.len() == 1 {
                results[members[0]].clone()
            } else {
                merge_block(&results, &members)
            }
        })
        .collect()
}

/// The groups `coalesce_adjacent` folds into blocks, as member indices into
/// `results`, in output order. A group of one is a result that merges with
/// nothing.
///
/// It is public because `search.rs` counts blocks to honour `top_n` (#102),
/// and counting them any other way would be a second copy of the run rule
/// that could drift from this one.
pub fn blocks(results: &[InternalSearchResult]) -> Vec<Vec<usize>> {
    let mut taken = vec![false; results.len()];
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(results.len());

    for i in 0..results.len() {
        if taken[i] {
            continue;
        }
        taken[i] = true;
        let file_id = results[i].file_id;
        let mut members = vec![i];
        let (mut lo, mut hi) = (results[i].chunk_seq, results[i].chunk_seq);
        let mut root = results[i].heading_path.clone();

        loop {
            let mut grew = false;
            for j in (i + 1)..results.len() {
                if taken[j] || results[j].file_id != file_id {
                    continue;
                }
                let seq = results[j].chunk_seq;
                if seq != hi + 1 && seq != lo - 1 {
                    continue;
                }
                let path = &results[j].heading_path;
                // The candidate is inside the run's section, or it is the
                // section the run is inside — in which case it becomes the
                // root, and a later pass re-offers what the narrower root
                // refused.
                if within(&root, path) {
                    root = path.clone();
                } else if !within(path, &root) {
                    continue;
                }
                taken[j] = true;
                members.push(j);
                lo = lo.min(seq);
                hi = hi.max(seq);
                grew = true;
            }
            if !grew {
                break;
            }
        }

        out.push(members);
    }

    out
}

/// Whether `path` is `ancestor` or a section below it. The comparison is
/// over heading-path segments and never over characters, because
/// `prefix::breadcrumb` joins segments with `" > "` and `Foo` is a
/// character prefix of the unrelated sibling `Foo Bar`.
fn within(path: &str, ancestor: &str) -> bool {
    path == ancestor
        || path
            .strip_prefix(ancestor)
            .is_some_and(|r| r.starts_with(" > "))
}

/// Fold a group of members into one block. A member is an index into
/// `results`. Identity comes from the leading section, the one with the
/// lowest `seq` — which is also the shallowest, since a section precedes
/// the subsections it contains, so the block is named by whichever section
/// holds all of it. The score comes from the strongest member. The text
/// comes from the members in `seq` order.
fn merge_block(results: &[InternalSearchResult], members: &[usize]) -> InternalSearchResult {
    let mut ordered: Vec<&InternalSearchResult> = members.iter().map(|&k| &results[k]).collect();
    ordered.sort_by_key(|m| m.chunk_seq);
    let leading = ordered[0];

    let strongest = ordered
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("members is non-empty");

    let text = ordered
        .iter()
        .map(|m| m.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let token_count = est_tokens_fallback(&text);
    let truncated = ordered.iter().any(|m| m.truncated);

    let mut provenance = Provenance {
        keyword: false,
        semantic: false,
        graph: false,
        linked_from: Vec::new(),
    };
    for m in &ordered {
        provenance.keyword |= m.provenance.keyword;
        provenance.semantic |= m.provenance.semantic;
        provenance.graph |= m.provenance.graph;
        provenance
            .linked_from
            .extend(m.provenance.linked_from.iter().cloned());
    }
    provenance.linked_from.sort();
    provenance.linked_from.dedup();

    InternalSearchResult {
        file_path: leading.file_path.clone(),
        file_id: leading.file_id,
        chunk_seq: leading.chunk_seq,
        score: strongest.score,
        confidence: strongest.confidence,
        heading: leading.heading.clone(),
        snippet: leading.snippet.clone(),
        docid: leading.docid.clone(),
        text,
        heading_path: leading.heading_path.clone(),
        token_count,
        truncated,
        provenance,
        properties: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A result in one file's `S` section. The abutting-merge tests vary
    /// `seq` and score and never the section, so this is what they read.
    fn r(file_id: i64, seq: i64, score: f64, text: &str) -> InternalSearchResult {
        r_at(file_id, seq, score, text, "S")
    }

    /// A result at a named heading path below the file, so a test can place
    /// its members in different sections of one document.
    fn r_at(
        file_id: i64,
        seq: i64,
        score: f64,
        text: &str,
        heading_path: &str,
    ) -> InternalSearchResult {
        InternalSearchResult {
            file_path: format!("f{file_id}.md"),
            file_id,
            chunk_seq: seq,
            score,
            confidence: score,
            heading: heading_path.rsplit(" > ").next().map(str::to_string),
            snippet: text.chars().take(200).collect(),
            docid: Some(format!("{file_id:06x}")),
            text: text.to_string(),
            heading_path: format!("f{file_id}.md > {heading_path}"),
            token_count: est_tokens_fallback(text),
            truncated: false,
            provenance: Provenance {
                keyword: false,
                semantic: true,
                graph: false,
                linked_from: Vec::new(),
            },
            properties: Vec::new(),
        }
    }

    /// A result for a chunk above the note's first heading, whose heading
    /// path is the note itself.
    fn r_at_root(file_id: i64, seq: i64, score: f64, text: &str) -> InternalSearchResult {
        let mut result = r_at(file_id, seq, score, text, "unused");
        result.heading = None;
        result.heading_path = format!("f{file_id}.md");
        result
    }

    #[test]
    fn blocks_names_the_groups_the_merge_makes() {
        // The counter and the merge read one rule, so what `blocks` groups is
        // exactly what `coalesce_adjacent` folds.
        let rows = vec![r(1, 3, 90.0, "A"), r(2, 0, 80.0, "B"), r(1, 4, 70.0, "C")];
        assert_eq!(blocks(&rows), vec![vec![0, 2], vec![1]]);
        assert_eq!(coalesce_adjacent(rows).len(), 2);
    }

    #[test]
    fn abutting_sections_of_one_file_coalesce() {
        let out = coalesce_adjacent(vec![r(1, 3, 90.0, "A"), r(1, 4, 50.0, "B")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].chunk_seq, 3, "headed by the leading section");
        assert_eq!(out[0].score, 90.0, "scored by the strongest member");
        assert_eq!(out[0].text, "A\n\nB", "members in document order");
    }

    #[test]
    fn the_block_is_headed_by_the_leading_section_and_scored_by_the_strongest() {
        // The strongest member (seq 4, 90) sits below the leading one, which
        // is the section that contains it.
        let out = coalesce_adjacent(vec![
            r_at(1, 4, 90.0, "B", "Foo > Bar"),
            r_at(1, 3, 50.0, "A", "Foo"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].chunk_seq, 3);
        assert_eq!(out[0].heading.as_deref(), Some("Foo"));
        assert_eq!(out[0].heading_path, "f1.md > Foo");
        assert_eq!(out[0].score, 90.0);
        assert_eq!(out[0].text, "A\n\nB");
    }

    #[test]
    fn sibling_sections_do_not_coalesce() {
        // #101. The tail of one section abuts the head of the next, so the
        // seq run is contiguous, but the two sections are different topics.
        let out = coalesce_adjacent(vec![
            r_at(1, 9, 50.0, "blood", "Blood Magic"),
            r_at(1, 10, 90.0, "thrall", "Enthrallment Magic"),
        ]);
        assert_eq!(out.len(), 2, "a sibling section is a different topic");
        assert_eq!(out[1].heading_path, "f1.md > Enthrallment Magic");
        assert_eq!(out[1].score, 90.0);
    }

    #[test]
    fn a_section_coalesces_with_its_subsection() {
        // A subsection subdivides the section's topic, so the follow-on
        // chunk is still the context the merge exists to keep.
        let out = coalesce_adjacent(vec![
            r_at(1, 3, 50.0, "A", "Foo"),
            r_at(1, 4, 90.0, "B", "Foo > Bar"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].heading_path, "f1.md > Foo",
            "headed by the container"
        );
        assert_eq!(out[0].text, "A\n\nB");
    }

    #[test]
    fn sibling_subsections_do_not_coalesce_without_their_parent() {
        let out = coalesce_adjacent(vec![
            r_at(1, 4, 90.0, "B", "Foo > Bar"),
            r_at(1, 5, 80.0, "C", "Foo > Baz"),
        ]);
        assert_eq!(out.len(), 2, "neither contains the other");
    }

    #[test]
    fn a_parent_gathers_its_subsections_into_one_block() {
        // The parent is encountered last, so the run must widen its root
        // and re-admit what a narrower root had refused.
        let out = coalesce_adjacent(vec![
            r_at(1, 4, 90.0, "B", "Foo > Bar"),
            r_at(1, 5, 80.0, "C", "Foo > Baz"),
            r_at(1, 3, 70.0, "A", "Foo"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].heading_path, "f1.md > Foo");
        assert_eq!(out[0].chunk_seq, 3);
        assert_eq!(out[0].text, "A\n\nB\n\nC", "in document order");
        assert_eq!(out[0].score, 90.0);
    }

    #[test]
    fn a_shallower_unrelated_heading_does_not_coalesce() {
        let out = coalesce_adjacent(vec![
            r_at(1, 3, 90.0, "A", "Foo > Bar"),
            r_at(1, 4, 50.0, "B", "Qux"),
        ]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_shared_text_prefix_is_not_a_shared_path() {
        // `Foo` is a string prefix of `Foo Bar` and an ancestor of neither.
        // The containment test reads path segments, not characters.
        let out = coalesce_adjacent(vec![
            r_at(1, 3, 90.0, "A", "Foo"),
            r_at(1, 4, 50.0, "B", "Foo Bar"),
        ]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_notes_preamble_gathers_the_sections_below_it() {
        // A chunk before the first heading carries the note's own path, which
        // every section of the note descends from. So when the preamble is
        // itself a result, the run it roots reaches the sections abutting it
        // — and the block is headed by the note, which does contain them all.
        let out = coalesce_adjacent(vec![
            r_at_root(1, 0, 70.0, "lead"),
            r_at(1, 1, 90.0, "A", "Blood Magic"),
            r_at(1, 2, 80.0, "B", "Enthrallment Magic"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].heading_path, "f1.md");
        assert_eq!(out[0].text, "lead\n\nA\n\nB");
    }

    #[test]
    fn non_abutting_sections_do_not_coalesce() {
        let out = coalesce_adjacent(vec![r(1, 3, 90.0, "A"), r(1, 5, 50.0, "C")]);
        assert_eq!(out.len(), 2, "a gap at seq 4 keeps them apart");
    }

    #[test]
    fn an_out_of_order_run_becomes_one_block() {
        let out = coalesce_adjacent(vec![
            r(1, 3, 90.0, "C"),
            r(1, 2, 80.0, "B"),
            r(1, 1, 70.0, "A"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].chunk_seq, 1);
        assert_eq!(out[0].score, 90.0);
        assert_eq!(out[0].text, "A\n\nB\n\nC");
    }

    #[test]
    fn a_gap_splits_a_file_into_two_blocks() {
        let out = coalesce_adjacent(vec![
            r(1, 0, 90.0, "S0"),
            r(1, 1, 80.0, "S1"),
            r(1, 2, 70.0, "S2"),
            r(1, 4, 60.0, "S4"),
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk_seq, 0);
        assert_eq!(out[0].text, "S0\n\nS1\n\nS2");
        assert_eq!(out[1].chunk_seq, 4);
        assert_eq!(out[1].text, "S4");
    }

    #[test]
    fn lower_rows_settle_up() {
        // seq 4 of file 1 is absorbed, so file 4's row rises from rank 4 to 3.
        let out = coalesce_adjacent(vec![
            r(1, 3, 90.0, "a"),
            r(3, 0, 80.0, "b"),
            r(1, 4, 70.0, "c"),
            r(4, 0, 60.0, "d"),
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].file_id, 1);
        assert_eq!(out[1].file_id, 3);
        assert_eq!(out[2].file_id, 4, "settled up from rank 4 to rank 3");
    }

    #[test]
    fn score_is_the_maximum_over_members() {
        let out = coalesce_adjacent(vec![
            r(1, 1, 20.0, "A"),
            r(1, 2, 95.0, "B"),
            r(1, 3, 10.0, "C"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].score, 95.0);
    }

    #[test]
    fn provenance_is_the_union() {
        let mut kw = r(1, 1, 90.0, "A");
        kw.provenance = Provenance {
            keyword: true,
            semantic: false,
            graph: false,
            linked_from: Vec::new(),
        };
        let mut sem = r(1, 2, 50.0, "B");
        sem.provenance = Provenance {
            keyword: false,
            semantic: true,
            graph: true,
            linked_from: Vec::new(),
        };
        let out = coalesce_adjacent(vec![kw, sem]);
        assert_eq!(out.len(), 1);
        assert!(out[0].provenance.keyword);
        assert!(out[0].provenance.semantic);
        assert!(out[0].provenance.graph);
    }

    #[test]
    fn different_files_never_merge() {
        let out = coalesce_adjacent(vec![r(1, 0, 90.0, "A"), r(2, 1, 80.0, "B")]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_single_section_is_unchanged() {
        let out = coalesce_adjacent(vec![r(1, 0, 90.0, "A")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "A");
        assert_eq!(out[0].score, 90.0);
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(coalesce_adjacent(Vec::new()).is_empty());
    }

    #[test]
    fn rank_placement_is_independent_of_score_selection() {
        // File 1's first-encountered row (seq 3, score 50) is not its
        // strongest member (seq 4, score 80). File 1's block still lands
        // first, because its first row was encountered first. Its score is
        // still the max over its members, not the first row's score.
        let out = coalesce_adjacent(vec![
            r(1, 3, 50.0, "a1"),
            r(2, 0, 90.0, "b0"),
            r(1, 4, 80.0, "a2"),
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].file_id, 1, "file 1's row was encountered first");
        assert_eq!(
            out[0].score, 80.0,
            "scored by the strongest member, not the first row"
        );
        assert_eq!(out[1].file_id, 2);
        assert_eq!(out[1].score, 90.0);
    }
}
