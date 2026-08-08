use std::collections::{HashMap, HashSet, hash_map::Entry};

use anyhow::Result;

use crate::fusion::RankedResult;
use crate::store::Store;

/// A wikilink as written, with the heading it named still attached.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Wikilink {
    /// The note, with any `#Heading` and `|Display` stripped.
    pub target: String,
    /// The `#Heading` part, if the link named one — the deep-link case (#28).
    pub heading: Option<String>,
}

/// Extract unique wikilinks from text, heading and all.
/// Handles [[Target]], [[Target|Display]], [[Target#Heading]].
/// Skips embeds (![[...]]).
///
/// Deduplicated on `(target, heading)`, so `[[Note#A]]` and `[[Note#B]]` are two
/// links and a repeat of either is one. Callers that only want the note use
/// [`extract_wikilink_targets`].
pub fn extract_wikilinks(text: &str) -> Vec<Wikilink> {
    let bytes = text.as_bytes();
    let mut links = Vec::new();
    let mut seen = HashSet::new();
    let mut i = 0;

    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            // Check for embed prefix (! before [[)
            let is_embed = i > 0 && bytes[i - 1] == b'!';
            if let Some(rest) = text.get(i + 2..)
                && let Some(close) = rest.find("]]")
            {
                let inner = &rest[..close];
                if !is_embed && !inner.is_empty() && !inner.contains('\n') {
                    // Obsidian escapes the alias pipe as `\|` inside tables;
                    // unescape it so the `|` separator is recognized.
                    let inner = inner.replace("\\|", "|");
                    // Strip display first: the alias comes last and may itself
                    // contain a `#`. [[Note#Section|Display]] → "Note#Section"
                    let addressed = inner.split('|').next().unwrap_or(inner.as_str());
                    let (target, heading) = match addressed.split_once('#') {
                        Some((t, h)) => (t.trim(), Some(h.trim())),
                        None => (addressed.trim(), None),
                    };
                    let link = Wikilink {
                        target: target.to_string(),
                        heading: heading.filter(|h| !h.is_empty()).map(str::to_string),
                    };
                    if !link.target.is_empty() && seen.insert(link.clone()) {
                        links.push(link);
                    }
                }
                i += 2 + close + 2;
                continue;
            }
        }
        i += 1;
    }
    links
}

/// Extract unique wikilink targets from text, discarding any `#Heading`.
pub fn extract_wikilink_targets(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    extract_wikilinks(text)
        .into_iter()
        .map(|l| l.target)
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Extract query terms for relevance filtering.
/// Splits on whitespace, lowercases, drops terms shorter than 3 chars.
pub fn extract_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 3)
        .collect()
}

/// How much of its weight an edge keeps when the seed *passage* did not contain
/// it, only some other part of the seed's file (issue #28).
///
/// `1.0` is the pre-#28 behaviour — every link in the document counted as
/// though the matched passage had written it. `0.0` scopes hard: a passage that
/// points nowhere expands nowhere. In between is the two-tier reading, which is
/// what shipped: the document-level relationship stays reachable, at a discount
/// that stops it outbidding the passage's own links for the
/// `truncate(max_expansions)` slots.
pub const OFF_CHUNK_LINK_WEIGHT: f64 = 0.5;

/// Expand search results by following graph connections.
/// Seeds are the top results from semantic + FTS lanes.
/// Returns expanded results suitable for RRF fusion.
///
/// Expansion follows the links of the seed's **chunk**, not its file — see
/// [`Store::get_neighbors_from_chunk`] and [`OFF_CHUNK_LINK_WEIGHT`].
pub fn graph_expand(
    store: &Store,
    seeds: &[RankedResult],
    query: &str,
    max_hops: usize,
    max_expansions: usize,
    off_chunk_weight: f64,
) -> Result<Vec<RankedResult>> {
    let query_terms = extract_query_terms(query);
    let seed_ids: HashSet<i64> = seeds.iter().map(|s| s.file_id).collect();

    // Track best score per expanded file (multi-parent merge: take highest)
    // (file_id) → (best_score, hop_depth, seed_file_path, matched chunk seq)
    let mut expansions: HashMap<i64, (f64, usize, String, Option<i64>)> = HashMap::new();

    for seed in seeds {
        let neighbors = store.get_neighbors_from_chunk(
            seed.file_id,
            seed.chunk_seq,
            max_hops,
            off_chunk_weight,
        )?;

        for (neighbor_id, hop, scope) in neighbors {
            if seed_ids.contains(&neighbor_id) {
                continue;
            }

            let decay = match hop {
                1 => 0.8,
                2 => 0.5,
                _ => 0.3,
            };
            let mut expansion_score = seed.score * decay * scope;

            // Relevance filter: must match a query term via FTS or share tags.
            // The matching chunk doubles as this result's section — a file-level
            // lane still has to say which part of the file it means.
            let matched_seq = store
                .best_matching_chunk_seq(neighbor_id, &query_terms)
                .unwrap_or(None);

            if matched_seq.is_none() {
                let shared = store
                    .get_shared_tags_files(neighbor_id, 100)
                    .unwrap_or_default();
                if shared.contains(&seed.file_id) {
                    expansion_score *= 0.7;
                } else {
                    continue; // tangential — skip
                }
            }

            // Multi-parent merge: keep highest score
            match expansions.entry(neighbor_id) {
                Entry::Occupied(mut e) => {
                    if expansion_score > e.get().0 {
                        e.insert((expansion_score, hop, seed.file_path.clone(), matched_seq));
                    }
                }
                Entry::Vacant(e) => {
                    e.insert((expansion_score, hop, seed.file_path.clone(), matched_seq));
                }
            }
        }
    }

    // Sort by score descending, cap at max_expansions
    let mut results: Vec<(i64, f64, usize, String, Option<i64>)> = expansions
        .into_iter()
        .map(|(fid, (score, hop, seed, seq))| (fid, score, hop, seed, seq))
        .collect();
    // Hop decay quantises scores hard (seed × 0.8, × 0.5), so ties are the norm
    // here and the truncation below decides which files reach fusion at all.
    // Break them on file id rather than on hash order.
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    results.truncate(max_expansions);

    // Convert to RankedResult
    let mut ranked = Vec::new();
    for (file_id, score, _hop, _seed, matched_seq) in results {
        let file = store.get_file_by_id(file_id)?;
        let (file_path, docid) = match file {
            Some(f) => (f.path, f.docid),
            None => continue,
        };
        // Prefer the chunk that matched the query. Files admitted on shared tags
        // alone matched nothing, so they fall back to the file's largest chunk.
        let chunk = match matched_seq {
            Some(seq) => store.get_chunk_by_seq(file_id, seq)?,
            None => store.get_best_chunk_for_file(file_id)?,
        };
        let (chunk_seq, heading, snippet) = match chunk {
            Some(c) => (c.seq, c.heading, c.snippet),
            None => (0, String::new(), String::new()),
        };
        let heading = if heading.is_empty() {
            None
        } else {
            Some(heading)
        };

        ranked.push(RankedResult {
            file_path,
            file_id,
            chunk_seq,
            score,
            heading,
            snippet,
            docid,
        });
    }

    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::fusion::RankedResult;
    use crate::store::{DOC_LEVEL, Store};

    #[test]
    fn test_extract_wikilink_targets() {
        let text =
            "See [[Note One]] and [[Note Two|display]] for details. Also [[Note One]] again.";
        let targets = extract_wikilink_targets(text);
        assert!(targets.contains(&"Note One".to_string()));
        assert!(targets.contains(&"Note Two".to_string()));
        assert_eq!(targets.len(), 2); // deduplicated
    }

    #[test]
    fn test_extract_wikilinks_with_headings() {
        let text = "Link to [[Note#Section]] here.";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Note"]);
    }

    #[test]
    fn test_extract_wikilinks_empty() {
        assert!(extract_wikilink_targets("no links here").is_empty());
        assert!(extract_wikilink_targets("").is_empty());
    }

    #[test]
    fn test_extract_wikilinks_skip_embeds() {
        let text = "![[embedded image.png]] and [[real link]]";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["real link"]);
    }

    #[test]
    fn test_extract_wikilinks_heading_and_display() {
        let text = "[[Note#Section|Custom Display]]";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Note"]); // strip both heading and display
    }

    #[test]
    fn test_extract_wikilinks_escaped_pipe_in_table() {
        // Obsidian escapes the alias pipe as `\|` inside tables; the target
        // must still resolve to the note name, not "Name\".
        let text = "| [[Page Name\\|Page]] | done |";
        let targets = extract_wikilink_targets(text);
        assert_eq!(targets, vec!["Page Name"]);
    }

    #[test]
    fn a_deep_link_keeps_the_heading_it_named() {
        // #28's target side: the heading is the whole point, and
        // `extract_wikilink_targets` throws it away by design.
        assert_eq!(
            extract_wikilinks("[[Note#Section]] and [[Note]] and [[Other#A|Display]]"),
            vec![
                Wikilink {
                    target: "Note".into(),
                    heading: Some("Section".into())
                },
                Wikilink {
                    target: "Note".into(),
                    heading: None
                },
                Wikilink {
                    target: "Other".into(),
                    heading: Some("A".into())
                },
            ]
        );
    }

    #[test]
    fn two_headings_of_one_note_are_two_links() {
        // Deduplication is on `(target, heading)`, so a note linked at two
        // sections produces two edges and a repeat produces one.
        let links = extract_wikilinks("[[Note#A]] … [[Note#B]] … [[Note#A]]");
        assert_eq!(links.len(), 2);
        assert_eq!(
            extract_wikilink_targets("[[Note#A]] … [[Note#B]]"),
            vec!["Note"]
        );
    }

    #[test]
    fn an_alias_containing_a_hash_is_not_a_heading() {
        // The alias comes last and may hold anything; splitting on `#` first
        // would read `C#` as a section of `Language`.
        assert_eq!(
            extract_wikilinks("[[Language|C# notes]]"),
            vec![Wikilink {
                target: "Language".into(),
                heading: None
            }]
        );
    }

    #[test]
    fn test_extract_query_terms() {
        let terms = extract_query_terms("BRE-2579 delivery date");
        assert_eq!(terms, vec!["bre-2579", "delivery", "date"]);
    }

    #[test]
    fn test_extract_query_terms_short_words_dropped() {
        let terms = extract_query_terms("a is the big query");
        assert_eq!(terms, vec!["the", "big", "query"]);
    }

    #[test]
    fn test_graph_expand_basic() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "seed.md",
                "h1",
                100,
                &["rust".into()],
                &generate_docid("seed.md"),
                None,
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "linked.md",
                "h2",
                100,
                &["rust".into()],
                &generate_docid("linked.md"),
                None,
                None,
            )
            .unwrap();
        let _f3 = store
            .insert_file(
                "unlinked.md",
                "h3",
                100,
                &[],
                &generate_docid("unlinked.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_chunk(f2, 0, "## Linked", "Linked content about delivery", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f2, 0, "Linked content about delivery")
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: f1,
            chunk_seq: 0,
            score: 0.85,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        let expanded =
            graph_expand(&store, &seeds, "delivery", 2, 20, OFF_CHUNK_LINK_WEIGHT).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].file_path, "linked.md");
        assert!(expanded[0].score > 0.0 && expanded[0].score < 0.85);
    }

    #[test]
    fn the_matched_passage_decides_what_expansion_reaches() {
        // #28 end to end. `seed.md` links to `near` from chunk 0 and to `far`
        // from chunk 1; a seed on chunk 0 must not drag `far` in at full weight
        // as though the matched passage had pointed at it.
        let store = Store::open_memory().unwrap();
        let mk = |path: &str| {
            store
                .insert_file(path, "h", 100, &[], &generate_docid(path), None, None)
                .unwrap()
        };
        let seed = mk("seed.md");
        let near = mk("near.md");
        let far = mk("far.md");
        store
            .insert_edge(seed, 0, near, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(seed, 1, far, DOC_LEVEL, "wikilink")
            .unwrap();
        for (id, name) in [(near, "near"), (far, "far")] {
            let text = format!("{name} content about delivery");
            store
                .insert_chunk(id, 0, "## Role", &text, id as u64, 20)
                .unwrap();
            store.insert_fts_chunk(id, 0, &text).unwrap();
        }

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: seed,
            chunk_seq: 0,
            score: 1.0,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        let scoped = graph_expand(&store, &seeds, "delivery", 2, 20, 0.0).unwrap();
        assert_eq!(
            scoped
                .iter()
                .map(|r| r.file_path.as_str())
                .collect::<Vec<_>>(),
            vec!["near.md"],
            "hard scope: the passage points at near.md and nowhere else"
        );

        let tiered = graph_expand(&store, &seeds, "delivery", 2, 20, 0.5).unwrap();
        assert_eq!(
            tiered
                .iter()
                .map(|r| r.file_path.as_str())
                .collect::<Vec<_>>(),
            vec!["near.md", "far.md"],
            "two-tier: far.md is still reachable, but ranked below"
        );
        assert!(tiered[0].score > tiered[1].score);
    }

    #[test]
    fn test_graph_expand_names_the_matching_section() {
        // The graph lane ranks whole files, but fusion keys on chunks, so it has
        // to say which section it means. The one containing the query beats the
        // file's longest section.
        let store = Store::open_memory().unwrap();
        let seed = store
            .insert_file(
                "seed.md",
                "h1",
                100,
                &[],
                &generate_docid("seed.md"),
                None,
                None,
            )
            .unwrap();
        let neighbor = store
            .insert_file(
                "linked.md",
                "h2",
                100,
                &[],
                &generate_docid("linked.md"),
                None,
                None,
            )
            .unwrap();
        store
            .insert_edge(seed, DOC_LEVEL, neighbor, DOC_LEVEL, "wikilink")
            .unwrap();

        // Section 0 is much longer, so it wins `get_best_chunk_for_file`.
        store
            .insert_chunk(
                neighbor,
                0,
                "## Overview",
                "## Overview\nA long introduction with plenty of words in it.",
                10,
                200,
            )
            .unwrap();
        store
            .insert_fts_chunk(
                neighbor,
                0,
                "## Overview\nA long introduction with plenty of words in it.",
            )
            .unwrap();
        store
            .insert_chunk(
                neighbor,
                1,
                "## Delivery",
                "## Delivery\nThe delivery date slipped.",
                11,
                20,
            )
            .unwrap();
        store
            .insert_fts_chunk(neighbor, 1, "## Delivery\nThe delivery date slipped.")
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: seed,
            chunk_seq: 0,
            score: 0.85,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        let expanded =
            graph_expand(&store, &seeds, "delivery", 2, 20, OFF_CHUNK_LINK_WEIGHT).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].chunk_seq, 1);
        assert_eq!(expanded[0].heading.as_deref(), Some("## Delivery"));
    }

    #[test]
    fn test_graph_expand_skips_seeds() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], &generate_docid("a.md"), None, None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, &[], &generate_docid("b.md"), None, None)
            .unwrap();

        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_chunk(f2, 0, "## B", "Content B", 10, 20)
            .unwrap();
        store.insert_fts_chunk(f2, 0, "Content B").unwrap();

        let seeds = vec![
            RankedResult {
                file_path: "a.md".into(),
                file_id: f1,
                chunk_seq: 0,
                score: 0.9,
                heading: None,
                snippet: "A".into(),
                docid: None,
            },
            RankedResult {
                file_path: "b.md".into(),
                file_id: f2,
                chunk_seq: 0,
                score: 0.8,
                heading: None,
                snippet: "B".into(),
                docid: None,
            },
        ];

        let expanded =
            graph_expand(&store, &seeds, "content", 2, 20, OFF_CHUNK_LINK_WEIGHT).unwrap();
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_graph_expand_multi_parent_takes_highest() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], &generate_docid("a.md"), None, None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, &[], &generate_docid("b.md"), None, None)
            .unwrap();
        let f3 = store
            .insert_file(
                "shared.md",
                "h3",
                100,
                &[],
                &generate_docid("shared.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_edge(f1, DOC_LEVEL, f3, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f2, DOC_LEVEL, f3, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_chunk(f3, 0, "## Shared", "Shared topic content", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f3, 0, "Shared topic content")
            .unwrap();

        let seeds = vec![
            RankedResult {
                file_path: "a.md".into(),
                file_id: f1,
                chunk_seq: 0,
                score: 0.9,
                heading: None,
                snippet: "A".into(),
                docid: None,
            },
            RankedResult {
                file_path: "b.md".into(),
                file_id: f2,
                chunk_seq: 0,
                score: 0.5,
                heading: None,
                snippet: "B".into(),
                docid: None,
            },
        ];

        let expanded = graph_expand(&store, &seeds, "topic", 1, 20, OFF_CHUNK_LINK_WEIGHT).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].file_path, "shared.md");
        // Should use highest parent: 0.9 * 0.8 = 0.72
        assert!((expanded[0].score - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_graph_expand_empty_graph() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], "aaa111", None, None)
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "a.md".into(),
            file_id: f1,
            chunk_seq: 0,
            score: 0.9,
            heading: None,
            snippet: "A".into(),
            docid: None,
        }];

        let expanded = graph_expand(&store, &seeds, "query", 2, 20, OFF_CHUNK_LINK_WEIGHT).unwrap();
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_graph_expand_tag_fallback() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "seed.md",
                "h1",
                100,
                &["rust".into(), "cli".into()],
                &generate_docid("seed.md"),
                None,
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "linked.md",
                "h2",
                100,
                &["rust".into()],
                &generate_docid("linked.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_chunk(f2, 0, "## Linked", "Unrelated content", 10, 20)
            .unwrap();
        store
            .insert_fts_chunk(f2, 0, "Unrelated content here")
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: f1,
            chunk_seq: 0,
            score: 0.85,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        // Query doesn't match FTS, but shared tag "rust" should keep it (with 0.7x penalty)
        let expanded = graph_expand(
            &store,
            &seeds,
            "nonexistent query term",
            2,
            20,
            OFF_CHUNK_LINK_WEIGHT,
        )
        .unwrap();
        assert_eq!(expanded.len(), 1);
        // Score: 0.85 * 0.8 * 0.7 = 0.476
        assert!((expanded[0].score - 0.476).abs() < 0.01);
    }

    #[test]
    fn test_graph_expand_follows_backlinks() {
        let store = Store::open_memory().unwrap();
        let seed = store
            .insert_file(
                "seed.md",
                "h1",
                100,
                &[],
                &generate_docid("seed.md"),
                None,
                None,
            )
            .unwrap();
        let backlinker = store
            .insert_file(
                "backlink.md",
                "h2",
                100,
                &[],
                &generate_docid("backlink.md"),
                None,
                None,
            )
            .unwrap();

        // backlink.md links TO seed.md; seed.md has no outgoing links.
        store
            .insert_edge(backlinker, DOC_LEVEL, seed, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_chunk(
                backlinker,
                0,
                "## Backlink",
                "Backlink content about delivery",
                10,
                20,
            )
            .unwrap();
        store
            .insert_fts_chunk(backlinker, 0, "Backlink content about delivery")
            .unwrap();

        let seeds = vec![RankedResult {
            file_path: "seed.md".into(),
            file_id: seed,
            chunk_seq: 0,
            score: 0.85,
            heading: None,
            snippet: "Seed".into(),
            docid: None,
        }];

        // Graph expansion is undirected: a note that links INTO the seed and
        // matches the query is surfaced as an expansion.
        let expanded =
            graph_expand(&store, &seeds, "delivery", 2, 20, OFF_CHUNK_LINK_WEIGHT).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].file_path, "backlink.md");
    }
}
