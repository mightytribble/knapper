use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::fusion::RankedResult;
use crate::store::{DOC_LEVEL, FileRecord, Store};

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

/// Resolve a wikilink target name to a file ID in the store.
///
/// Lives beside the link vocabulary rather than in the indexer, because
/// resolving what a link names is what every reader of a link has to do —
/// `build_edges_for_file` writes edges from it and [`deep_links_from`] reports
/// on it (#99).
pub(crate) fn resolve_link_target(store: &Store, target: &str) -> Result<Option<i64>> {
    let with_ext = if target.ends_with(".md") {
        target.to_string()
    } else {
        format!("{}.md", target)
    };

    // Try exact path match
    if let Some(f) = store.get_file(&with_ext)? {
        return Ok(Some(f.id));
    }

    // Try basename match (case-insensitive)
    let all_files = store.get_all_files()?;
    let target_lower = with_ext.to_lowercase();
    let mut matches: Vec<&FileRecord> = all_files
        .iter()
        .filter(|f| {
            let path_lower = f.path.to_lowercase();
            path_lower == target_lower || path_lower.ends_with(&format!("/{}", target_lower))
        })
        .collect();

    matches.sort_by_key(|f| f.path.len());
    Ok(matches.first().map(|f| f.id))
}

/// A deep wikilink as one note wrote it, resolved to the note it names.
///
/// The heading is the **source note's** text and not the target's. That is
/// what a report quotes and what a rewrite would have to edit, and the edge
/// table keeps neither: an edge names a `to_chunk_seq`, and a heading that no
/// longer resolves is already flattened to [`DOC_LEVEL`] there (#99).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DeepLink {
    /// The linking note's path.
    pub source: String,
    /// The note the link resolves to.
    pub target_id: i64,
    /// The `#Heading` the link named, as written.
    pub heading: String,
}

/// Every deep wikilink written in `sources`, resolved to the note it names.
///
/// Reads `chunks.text` and never the vault, so the answer is what each note
/// held at its last index. A link whose note does not resolve is left out:
/// that is the `unresolved_links` table's fact, and repeating it here would
/// report one broken link twice.
///
/// Two callers, one enumeration (#99). A rename passes the notes that link to
/// the note being renamed — `get_incoming` gives them, since a note writing
/// `[[Target#Anything]]` always holds a wikilink edge to `Target` — and keeps
/// the headings that match the old name. `health` passes every note and keeps
/// the headings the target no longer holds.
///
/// A link inside a code fence counts, because [`extract_wikilinks`] does not
/// track fences. A note that documents the syntax therefore reports a link it
/// does not really hold. That costs a reader one glance, which is the reason
/// both callers report and neither writes.
pub fn deep_links_from(store: &Store, sources: &[i64]) -> Result<Vec<DeepLink>> {
    let mut out = Vec::new();
    for &source_id in sources {
        let Some(source) = store.get_file_by_id(source_id)? else {
            continue; // the file went away mid-read
        };
        // One note naming one heading is one link however many passages of the
        // note write it; `extract_wikilinks` dedups within a passage only.
        let mut seen = HashSet::new();
        for chunk in store.get_chunks_by_file(source_id)? {
            for link in extract_wikilinks(&chunk.text) {
                let Some(heading) = link.heading else {
                    continue; // a plain link names the document and cannot go stale
                };
                if !seen.insert((link.target.clone(), heading.clone())) {
                    continue;
                }
                if let Some(target_id) = resolve_link_target(store, &link.target)? {
                    out.push(DeepLink {
                        source: source.path.clone(),
                        target_id,
                        heading,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// How much of its weight an edge keeps when the seed *passage* did not contain
/// it, only some other part of the seed's file (issue #28).
///
/// `1.0` is the pre-#28 behaviour — every link in the document counted as
/// though the matched passage had written it. `0.0` scopes hard: a passage that
/// points nowhere expands nowhere. In between is the two-tier reading, which is
/// what shipped: the document-level relationship stays reachable, at a discount
/// that stops it outbidding the passage's own links for the output slots.
///
/// **#28 could not measure this and #29 can.** Under the old `max` merge every
/// value from 0.0 to 0.9 produced byte-identical probe output, because a file
/// was nearly always reachable chunk-locally from *some* one of 60–120 seeds and
/// the max took that path. Summing makes the discount bite: it changes both what
/// arrives and the degree it is divided by. Hard scoping now costs two tracked
/// targets, so 0.5 ships on evidence rather than on argument.
pub const OFF_CHUNK_LINK_WEIGHT: f64 = 0.5;

/// Personalized PageRank over the chunk graph (issue #29).
///
/// The lane this replaced scored an expansion `seed.score × hop_decay`, kept the
/// **max** across parents, and admitted it only if a keyword-or-tag filter let
/// it through. That is legible as a one-iteration PPR with the accumulation
/// operator wrong: hop decay *is* damping, two hops *are* two iterations, and
/// the seed scores *are* the restart distribution. What was missing:
///
/// ```text
/// score(c) = Σ over seeds s:  seed_score(s) × 1/L(s) × target_share(link, c)
///                             ─────────────   ──────   ─────────────────────
///                             #26: normalised  source   1 for a deep link,
///                             to [0.1, 1]      out-deg  1/√N for a document one
/// ```
///
/// Three normalisations doing three separate jobs: seed strength across lanes,
/// hub suppression on the source, specificity on the target. The two exponents
/// were swept and landed in opposite places — see the fields.
#[derive(Debug, Clone, Copy)]
pub struct PprParams {
    /// How many times the walk operator is applied. `1` is one hop.
    ///
    /// Swept: `2` costs two tracked targets and gains none. Two-hop
    /// neighbourhoods in a densely wikilinked vault are enormous, and what is
    /// wanted here is a recall boost from immediate structural neighbours, not
    /// PageRank's globally-important-node-several-hops-out.
    pub iterations: usize,
    /// Damping per iteration. **Inert at `iterations = 1`**: it scales every
    /// chunk's mass equally, and both the lane's own sort and RRF read order.
    /// It is a live parameter only in the two-iteration sweep, which lost.
    pub alpha: f64,
    /// Exponent on the source chunk's weighted degree. `1.0` is the standard
    /// `1/L` and conserves mass exactly; `0.5` is the usual softening.
    ///
    /// The softening was worth trying because the link-dense chunks in a world
    /// vault are `### Connections` sections that exist *specifically* to encode
    /// relationships, and dividing them by twenty may gut the right signal. It
    /// lost: `1/√L` costs two tracked targets and gains none. Full normalisation
    /// on the source it is.
    pub out_degree_exp: f64,
    /// Exponent on the target document's chunk count for a [`DOC_LEVEL`] link.
    /// `1.0` divides one link's mass across the passages it could mean; `0.0`
    /// gives each of them the whole of it, which would make a link's value
    /// proportional to how long the note it points at happens to be.
    ///
    /// **`0.5` won, which is the opposite of where `out_degree_exp` landed.**
    /// The ticket specified `1/N` and it is the mass-conserving choice; measured,
    /// `1/√N` moves two tracked targets up and none down, and `1/N` — full
    /// spreading — is a strong bias *against* long notes, which on this vault are
    /// the substantial ones. The residual bias is real and bounded: an N-passage
    /// note collects √N times a one-passage note's endorsement from the same
    /// link, where #28's rejected materialisation would have given it N times.
    pub target_spread_exp: f64,
    /// See [`OFF_CHUNK_LINK_WEIGHT`].
    pub off_chunk_weight: f64,
    /// How many chunks reach fusion. A cut on accumulated mass, unlike the
    /// `truncate(20)` it replaces, which cut on `seed.score × decay`.
    pub max_expansions: usize,
    /// At most this many chunks of any one document, as the content lanes do.
    /// A document-level link spreads across *every* passage of its target at
    /// exactly equal mass, so without this one note can take the whole lane.
    pub cap_per_file: usize,
}

impl Default for PprParams {
    fn default() -> Self {
        Self {
            iterations: 1,
            alpha: 0.6,
            out_degree_exp: 1.0,
            target_spread_exp: 0.5,
            off_chunk_weight: OFF_CHUNK_LINK_WEIGHT,
            max_expansions: 20,
            cap_per_file: 3,
        }
    }
}

/// The largest frontier carried into another iteration.
///
/// Two-hop neighbourhoods in a densely wikilinked vault are enormous and the
/// frontier is the input to the next fetch, so this bounds the walk's cost. It
/// is a cut on mass, and it logs when it bites — a silent one would read as
/// "the graph ends here".
const MAX_FRONTIER: usize = 2000;

/// Expand search results by walking the chunk graph out of the seed passages.
///
/// Seeds are the top results from the semantic + FTS lanes, normalised to a
/// shared scale (#26); the walk is over chunk-to-chunk edges (#28); the output
/// is chunks, so nothing downstream has to guess which passage was meant.
///
/// `scope` is the tag scope's file ids, or `None` for the whole vault (#60).
/// It filters admission and not traversal: mass still flows through an
/// untagged note to reach an in-scope one, because that link is the
/// co-citation signal this lane exists to read. Confining the walk itself
/// would strip most of the lane's reach on a partly tagged vault.
pub fn graph_expand(
    store: &Store,
    seeds: &[RankedResult],
    params: &PprParams,
    scope: Option<&std::collections::HashSet<i64>>,
) -> Result<Vec<RankedResult>> {
    // The restart distribution. Two seeds on the same chunk are one mass — the
    // dedup that replaced the disjointness skip.
    let mut frontier: HashMap<(i64, i64), f64> = HashMap::new();
    for seed in seeds {
        *frontier
            .entry((seed.file_id, seed.chunk_seq))
            .or_insert(0.0) += seed.score;
    }

    // Σ over iterations of αⁱ · Wⁱ · d₀. The restart distribution itself is not
    // in the sum: a seed chunk earns its place here only by being *walked to*,
    // which is the co-citation signal. It is no longer excluded, though — a
    // chunk that is both a strong content hit and heavily pointed-at is the best
    // candidate in the pool, and the old skip made that unrepresentable.
    let mut accumulated: HashMap<(i64, i64), f64> = HashMap::new();
    let mut damping = 1.0;

    for _ in 0..params.iterations {
        if frontier.is_empty() {
            break;
        }
        damping *= params.alpha;
        frontier = walk(store, &frontier, params)?;
        for (chunk, mass) in &frontier {
            *accumulated.entry(*chunk).or_insert(0.0) += mass * damping;
        }
        if frontier.len() > MAX_FRONTIER {
            let dropped = frontier.len() - MAX_FRONTIER;
            let mut by_mass: Vec<((i64, i64), f64)> = frontier.into_iter().collect();
            by_mass.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            by_mass.truncate(MAX_FRONTIER);
            frontier = by_mass.into_iter().collect();
            tracing::debug!(dropped, "frontier truncated before the next iteration");
        }
    }

    // A cut on a meaningful number, for the first time: mass, not `seed.score`
    // times a constant. Ties are exact and common — a document-level link gives
    // every passage of its target the same share — so break them on identity.
    let mut ranked: Vec<((i64, i64), f64)> = accumulated.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    // Before the cap and before the truncate: a candidate that cannot be an
    // answer must not spend one of `max_expansions`, or the reserve is a quota
    // filled with rows the caller asked to exclude (#60).
    if let Some(scope) = scope {
        ranked.retain(|((file_id, _), _)| scope.contains(file_id));
    }
    if params.cap_per_file > 0 {
        let mut per_file: HashMap<i64, usize> = HashMap::new();
        ranked.retain(|((file_id, _), _)| {
            let count = per_file.entry(*file_id).or_insert(0);
            *count += 1;
            *count <= params.cap_per_file
        });
    }
    ranked.truncate(params.max_expansions);

    let mut results = Vec::new();
    for ((file_id, chunk_seq), score) in ranked {
        let Some(file) = store.get_file_by_id(file_id)? else {
            continue;
        };
        let Some(chunk) = store.get_chunk_by_seq(file_id, chunk_seq)? else {
            continue;
        };
        results.push(RankedResult {
            file_path: file.path,
            file_id,
            chunk_seq: chunk.seq,
            score,
            heading: Some(chunk.heading).filter(|h| !h.is_empty()),
            snippet: chunk.snippet,
            docid: file.docid,
        });
    }

    Ok(results)
}

/// One application of the walk operator: mass on chunks in, mass on chunks out.
///
/// The whole frontier's edges come back in **one** indexed fetch, which is where
/// the cost went: this replaced 55–91 separate breadth-first traversals, each
/// carrying its own `visited` set and issuing two queries per node visited.
fn walk(
    store: &Store,
    frontier: &HashMap<(i64, i64), f64>,
    params: &PprParams,
) -> Result<HashMap<(i64, i64), f64>> {
    let files: Vec<i64> = frontier
        .keys()
        .map(|(f, _)| *f)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let edges = store.incident_wikilink_edges(&files)?;

    let mut by_file: HashMap<i64, Vec<(i64, i64, i64)>> = HashMap::new();
    for (near_file, near_seq, far_file, far_seq) in edges {
        by_file
            .entry(near_file)
            .or_default()
            .push((near_seq, far_file, far_seq));
    }

    // What a document-level target resolves to. Fetched for the files that
    // actually need it, once for the whole frontier.
    let doc_targets: Vec<i64> = by_file
        .values()
        .flatten()
        .filter(|(_, _, far_seq)| *far_seq == DOC_LEVEL)
        .map(|(_, far_file, _)| *far_file)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let spread = store.chunk_seqs_for_files(&doc_targets)?;

    let mut next: HashMap<(i64, i64), f64> = HashMap::new();
    for (&(file_id, chunk_seq), &mass) in frontier {
        let Some(incident) = by_file.get(&file_id) else {
            continue;
        };

        // Scope each incident edge to the passage in hand (#28), then take the
        // degree over what survived: an edge that carries no mass must not
        // dilute the ones that do.
        let outgoing: Vec<(f64, i64, i64)> = incident
            .iter()
            .filter_map(|&(near_seq, far_file, far_seq)| {
                // `build_edges_for_file` never writes a self-edge, so this only
                // guards a future one — a walk that stayed on its own file would
                // be counting a document against itself.
                if far_file == file_id {
                    return None;
                }
                let local = near_seq == chunk_seq || near_seq == DOC_LEVEL;
                let weight = if local { 1.0 } else { params.off_chunk_weight };
                (weight > 0.0).then_some((weight, far_file, far_seq))
            })
            .collect();

        let degree: f64 = outgoing.iter().map(|(w, _, _)| w).sum();
        if degree <= 0.0 {
            continue;
        }
        let divisor = degree.powf(params.out_degree_exp);

        for (weight, far_file, far_seq) in outgoing {
            let sent = mass * weight / divisor;
            if far_seq != DOC_LEVEL {
                *next.entry((far_file, far_seq)).or_insert(0.0) += sent;
                continue;
            }
            // The link named a note, not a passage of it. Divide its mass across
            // the passages it could mean rather than handing each of them the
            // whole of it — #28 declined to materialise those rows precisely so
            // that a 37-chunk note would not read as 37× the endorsement of a
            // one-chunk note under this sum.
            let Some(seqs) = spread.get(&far_file) else {
                continue; // an unchunked note is nothing to land on
            };
            let share = sent / (seqs.len() as f64).powf(params.target_spread_exp);
            for &seq in seqs {
                *next.entry((far_file, seq)).or_insert(0.0) += share;
            }
        }
    }

    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::fusion::RankedResult;
    use crate::store::{DOC_LEVEL, NewChunk, Store};

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

    // ── Deep links (#99) ─────────────────────────────────────────

    /// A note whose passages are `(heading, text)`, so a link one note wrote
    /// can be found in the text the store holds.
    fn note(store: &Store, path: &str, passages: &[(&str, &str)]) -> i64 {
        let id = store
            .insert_file(path, "h", 100, &generate_docid(path), None, None)
            .unwrap();
        for (seq, (heading, text)) in passages.iter().enumerate() {
            store
                .insert_chunk(&NewChunk {
                    file_id: id,
                    seq: seq as i64,
                    heading,
                    text,
                    vector_id: (id * 100 + seq as i64) as u64,
                    token_count: 20,
                    ..Default::default()
                })
                .unwrap();
        }
        id
    }

    /// The heading is kept as the source note wrote it, not as the target
    /// holds it: it is the text a report quotes and a rewrite would edit, and
    /// the edge table throws it away (#99).
    #[test]
    fn a_deep_link_carries_the_heading_the_source_note_wrote() {
        let store = Store::open_memory().unwrap();
        let target = note(&store, "Roads.md", &[("## Norlund to Westport", "body")]);
        let source = note(
            &store,
            "Trade.md",
            &[("## Legs", "See [[Roads#Norlund to Westport]].")],
        );
        assert_eq!(
            deep_links_from(&store, &[source]).unwrap(),
            vec![DeepLink {
                source: "Trade.md".into(),
                target_id: target,
                heading: "Norlund to Westport".into(),
            }]
        );
    }

    /// A plain `[[Note]]` names the document and cannot go stale (#99).
    #[test]
    fn a_plain_wikilink_is_not_a_deep_link() {
        let store = Store::open_memory().unwrap();
        note(&store, "Roads.md", &[("## Norlund to Westport", "body")]);
        let source = note(&store, "Trade.md", &[("## Legs", "See [[Roads]].")]);
        assert!(deep_links_from(&store, &[source]).unwrap().is_empty());
    }

    /// The alias is display text; the heading behind it is what resolves (#99).
    #[test]
    fn an_aliased_deep_link_keeps_its_heading() {
        let store = Store::open_memory().unwrap();
        let target = note(&store, "Roads.md", &[("## Norlund to Westport", "body")]);
        let source = note(
            &store,
            "Trade.md",
            &[(
                "## Legs",
                "See [[Roads#Norlund to Westport|the north leg]].",
            )],
        );
        assert_eq!(
            deep_links_from(&store, &[source]).unwrap(),
            vec![DeepLink {
                source: "Trade.md".into(),
                target_id: target,
                heading: "Norlund to Westport".into(),
            }]
        );
    }

    /// A link to a note that does not exist is the `unresolved_links` table's
    /// fact, and reporting it here would name it twice (#99).
    #[test]
    fn a_deep_link_whose_note_does_not_resolve_is_not_one() {
        let store = Store::open_memory().unwrap();
        let source = note(&store, "Trade.md", &[("## Legs", "See [[Nowhere#Bend]].")]);
        assert!(deep_links_from(&store, &[source]).unwrap().is_empty());
    }

    /// A link names its note the way a link may: by basename, folded, without
    /// the folder it sits in — the rule `build_edges_for_file` already
    /// resolves by (#99).
    #[test]
    fn a_deep_link_resolves_its_note_by_basename() {
        let store = Store::open_memory().unwrap();
        let target = note(
            &store,
            "01-Places/Roads.md",
            &[("## Norlund to Westport", "body")],
        );
        let source = note(
            &store,
            "Trade.md",
            &[("## Legs", "See [[roads#Norlund to Westport]].")],
        );
        assert_eq!(
            deep_links_from(&store, &[source]).unwrap(),
            vec![DeepLink {
                source: "Trade.md".into(),
                target_id: target,
                heading: "Norlund to Westport".into(),
            }]
        );
    }

    // ── Personalized PageRank (#29) ──────────────────────────────

    fn file(store: &Store, path: &str) -> i64 {
        store
            .insert_file(path, "h", 100, &generate_docid(path), None, None)
            .unwrap()
    }

    /// A note with `n` passages, so a document-level link into it has something
    /// to divide across.
    fn chunked(store: &Store, path: &str, n: i64) -> i64 {
        let id = file(store, path);
        for seq in 0..n {
            let text = format!("{path} passage {seq}");
            store
                .insert_chunk(&NewChunk {
                    file_id: id,
                    seq,
                    heading: &format!("## S{seq}"),
                    text: &text,
                    vector_id: (id * 100 + seq) as u64,
                    token_count: 20,
                    ..Default::default()
                })
                .unwrap();
        }
        id
    }

    fn seed(store: &Store, file_id: i64, chunk_seq: i64, score: f64) -> RankedResult {
        RankedResult {
            file_path: store.get_file_by_id(file_id).unwrap().unwrap().path,
            file_id,
            chunk_seq,
            score,
            heading: None,
            snippet: String::new(),
            docid: None,
        }
    }

    /// The shipping parameters with damping switched off.
    ///
    /// α is a uniform scale on a single iteration — pinned by
    /// `damping_cannot_change_a_single_iterations_order` — so setting it to 1
    /// keeps the assertions below about the walk rather than about α.
    fn ppr() -> PprParams {
        PprParams {
            alpha: 1.0,
            ..PprParams::default()
        }
    }

    /// `(path, seq)` per result, in rank order — the lane's whole output.
    fn reached(results: &[RankedResult]) -> Vec<(String, i64)> {
        results
            .iter()
            .map(|r| (r.file_path.clone(), r.chunk_seq))
            .collect()
    }

    #[test]
    fn the_walk_returns_the_passage_it_reached() {
        // No `get_best_chunk_for_file`, no `best_matching_chunk_seq`: a deep link
        // names a passage and that passage is the result, heading and all.
        let store = Store::open_memory().unwrap();
        let a = chunked(&store, "a.md", 1);
        let b = chunked(&store, "b.md", 3);
        store.insert_edge(a, 0, b, 1, "wikilink").unwrap();

        let out = graph_expand(
            &store,
            &[seed(&store, a, 0, 1.0)],
            &PprParams::default(),
            None,
        )
        .unwrap();
        assert_eq!(reached(&out), vec![("b.md".to_string(), 1)]);
        assert_eq!(out[0].heading.as_deref(), Some("## S1"));
        assert_eq!(out[0].snippet, "b.md passage 1");
    }

    #[test]
    fn many_weak_endorsements_beat_one_strong_one() {
        // The co-citation signal, and the reason `sum` replaced `max`. Under the
        // old merge `shared` scored 0.4 — its best single parent — and lost to
        // `solo`; nothing could test that, because `max` made it unrepresentable.
        let store = Store::open_memory().unwrap();
        let shared = chunked(&store, "shared.md", 1);
        let solo = chunked(&store, "solo.md", 1);
        let mut seeds = Vec::new();
        for i in 0..3 {
            let s = file(&store, &format!("cite{i}.md"));
            store
                .insert_edge(s, 0, shared, DOC_LEVEL, "wikilink")
                .unwrap();
            seeds.push(seed(&store, s, 0, 0.4));
        }
        let strong = file(&store, "strong.md");
        store
            .insert_edge(strong, 0, solo, DOC_LEVEL, "wikilink")
            .unwrap();
        seeds.push(seed(&store, strong, 0, 0.9));

        let out = graph_expand(&store, &seeds, &ppr(), None).unwrap();
        assert_eq!(
            reached(&out),
            vec![("shared.md".to_string(), 0), ("solo.md".to_string(), 0)]
        );
        assert!((out[0].score - 1.2).abs() < 1e-9, "0.4 three times over");
        assert!((out[1].score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn a_hub_divides_its_endorsement_among_everything_it_points_at() {
        // Out-degree normalisation, and the reason the sum does not simply
        // re-elect the hubs: a link from a nine-link section is worth a ninth of
        // one from a section that points at a single note.
        let store = Store::open_memory().unwrap();
        let hub = file(&store, "hub.md");
        let sparse = file(&store, "sparse.md");
        let from_hub = chunked(&store, "from-hub.md", 1);
        let from_sparse = chunked(&store, "from-sparse.md", 1);
        store
            .insert_edge(hub, 0, from_hub, DOC_LEVEL, "wikilink")
            .unwrap();
        for i in 0..8 {
            let filler = chunked(&store, &format!("filler{i}.md"), 1);
            store
                .insert_edge(hub, 0, filler, DOC_LEVEL, "wikilink")
                .unwrap();
        }
        store
            .insert_edge(sparse, 0, from_sparse, DOC_LEVEL, "wikilink")
            .unwrap();

        let seeds = [seed(&store, hub, 0, 1.0), seed(&store, sparse, 0, 0.2)];

        // 1/L: the hub sends 1/9 = 0.111 down each of its nine links, which is
        // less than the sparse seed's whole 0.2.
        let full = graph_expand(&store, &seeds, &ppr(), None).unwrap();
        assert_eq!(full[0].file_path, "from-sparse.md");
        assert!((full[0].score - 0.2).abs() < 1e-9);
        assert!((full[1].score - 1.0 / 9.0).abs() < 1e-9);

        // 1/√L: the same link is now worth 1/3, and the dense section outranks
        // the sparse one. This is the sweep the ticket asked for, and the whole
        // of what it moves.
        let softened = PprParams {
            out_degree_exp: 0.5,
            ..ppr()
        };
        let soft = graph_expand(&store, &seeds, &softened, None).unwrap();
        assert_eq!(soft[0].file_path, "from-hub.md");
        assert!((soft[0].score - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_document_level_link_divides_across_the_passages_it_could_mean() {
        // #28 declined to store one row per target chunk; this is where that
        // decision is paid for. `long.md` gets the same *total* endorsement as
        // `short.md`, spread thin, rather than four times as much.
        let store = Store::open_memory().unwrap();
        let src = file(&store, "src.md");
        let long = chunked(&store, "long.md", 4);
        let short = chunked(&store, "short.md", 1);
        store
            .insert_edge(src, 0, long, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(src, 0, short, DOC_LEVEL, "wikilink")
            .unwrap();

        let seeds = [seed(&store, src, 0, 1.0)];

        // Full spreading, which is what the ticket specified: `long.md` collects
        // the same *total* as `short.md`, divided four ways.
        let conserving = PprParams {
            target_spread_exp: 1.0,
            ..ppr()
        };
        let out = graph_expand(&store, &seeds, &conserving, None).unwrap();
        assert_eq!(out[0].file_path, "short.md");
        assert!((out[0].score - 0.5).abs() < 1e-9);
        assert_eq!(
            reached(&out)[1..],
            [
                ("long.md".to_string(), 0),
                ("long.md".to_string(), 1),
                ("long.md".to_string(), 2)
            ],
            "the per-file cap stops one note taking the lane; ties break on seq"
        );
        for r in &out[1..] {
            assert!((r.score - 0.125).abs() < 1e-9, "0.5 split four ways");
        }

        // What ships: `1/√N`, so a four-passage note collects twice a
        // one-passage note's total from the same link rather than the same — and
        // rather than the four times #28's rejected materialisation would have
        // given it. It is still each passage individually that ranks below.
        let shipped = graph_expand(&store, &seeds, &ppr(), None).unwrap();
        assert_eq!(shipped[0].file_path, "short.md");
        for r in &shipped[1..] {
            assert!((r.score - 0.25).abs() < 1e-9, "0.5 split by √4");
        }
    }

    #[test]
    fn a_passage_expands_along_its_own_links_only() {
        // #28's scoping, now enforced here rather than in the store's traversal.
        // `hub` links to `near` from seq 0 and to `far` from seq 1.
        let store = Store::open_memory().unwrap();
        let hub = file(&store, "hub.md");
        let near = chunked(&store, "near.md", 1);
        let far = chunked(&store, "far.md", 1);
        store
            .insert_edge(hub, 0, near, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(hub, 1, far, DOC_LEVEL, "wikilink")
            .unwrap();

        let hard = PprParams {
            off_chunk_weight: 0.0,
            ..ppr()
        };
        assert_eq!(
            reached(&graph_expand(&store, &[seed(&store, hub, 0, 1.0)], &hard, None).unwrap()),
            vec![("near.md".to_string(), 0)]
        );
        assert_eq!(
            reached(&graph_expand(&store, &[seed(&store, hub, 1, 1.0)], &hard, None).unwrap()),
            vec![("far.md".to_string(), 0)]
        );

        // Two-tier: the document's other link is still reachable, at a discount,
        // and the degree is taken over the weighted edges — 1.0 + 0.5 = 1.5.
        let tiered = graph_expand(&store, &[seed(&store, hub, 0, 1.0)], &ppr(), None).unwrap();
        assert_eq!(
            reached(&tiered),
            vec![("near.md".to_string(), 0), ("far.md".to_string(), 0)]
        );
        assert!((tiered[0].score - 1.0 / 1.5).abs() < 1e-9);
        assert!((tiered[1].score - 0.5 / 1.5).abs() < 1e-9);
    }

    #[test]
    fn a_link_aimed_at_the_whole_document_leaves_from_every_passage() {
        // `DOC_LEVEL` on the near end means "any passage of this file", so a
        // backlink into `hub` is walkable from whichever passage seeded.
        let store = Store::open_memory().unwrap();
        let hub = chunked(&store, "hub.md", 8);
        let other = chunked(&store, "other.md", 4);
        store
            .insert_edge(other, 3, hub, DOC_LEVEL, "wikilink")
            .unwrap();

        let hard = PprParams {
            off_chunk_weight: 0.0,
            ..PprParams::default()
        };
        for seq in [0, 7] {
            let out = graph_expand(&store, &[seed(&store, hub, seq, 1.0)], &hard, None).unwrap();
            assert_eq!(
                reached(&out),
                vec![("other.md".to_string(), 3)],
                "from hub.md#{seq}"
            );
        }
    }

    #[test]
    fn a_seed_can_be_its_own_expansion() {
        // The disjointness skip is gone. A chunk that is both a content hit and
        // pointed at by other seeds is the best candidate in the pool; the old
        // `if seed_ids.contains(..) { continue }` made that structurally
        // impossible, which is what #9 found and could only work around.
        let store = Store::open_memory().unwrap();
        let a = chunked(&store, "a.md", 1);
        let b = chunked(&store, "b.md", 1);
        store.insert_edge(a, 0, b, DOC_LEVEL, "wikilink").unwrap();

        let seeds = [seed(&store, a, 0, 0.9), seed(&store, b, 0, 0.8)];
        let out = graph_expand(&store, &seeds, &PprParams::default(), None).unwrap();
        assert_eq!(
            reached(&out),
            vec![("b.md".to_string(), 0), ("a.md".to_string(), 0)],
            "each end walks to the other; b is endorsed harder because a scored higher"
        );
    }

    #[test]
    fn damping_cannot_change_a_single_iterations_order() {
        // α is a uniform scale on one iteration, and both the sort here and RRF
        // downstream read order. It is only a parameter once there are two.
        let store = Store::open_memory().unwrap();
        let src = file(&store, "src.md");
        for i in 0..3 {
            let t = chunked(&store, &format!("t{i}.md"), 1);
            store.insert_edge(src, 0, t, DOC_LEVEL, "wikilink").unwrap();
        }
        let seeds = [seed(&store, src, 0, 1.0)];
        let low = graph_expand(
            &store,
            &seeds,
            &PprParams {
                alpha: 0.1,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let high = graph_expand(
            &store,
            &seeds,
            &PprParams {
                alpha: 0.9,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(reached(&low), reached(&high));
        assert!((high[0].score / low[0].score - 9.0).abs() < 1e-9);
    }

    #[test]
    fn a_second_iteration_is_the_same_operator_applied_again() {
        // `a#0 → b#0 → c`. One iteration reaches b; two reach c, damped by α
        // twice and divided by b's degree — which counts the backlink to a.
        let store = Store::open_memory().unwrap();
        let a = chunked(&store, "a.md", 1);
        let b = chunked(&store, "b.md", 1);
        let c = chunked(&store, "c.md", 1);
        store.insert_edge(a, 0, b, 0, "wikilink").unwrap();
        store.insert_edge(b, 0, c, DOC_LEVEL, "wikilink").unwrap();

        let seeds = [seed(&store, a, 0, 1.0)];
        let one = graph_expand(&store, &seeds, &PprParams::default(), None).unwrap();
        assert_eq!(reached(&one), vec![("b.md".to_string(), 0)]);

        let two = PprParams {
            iterations: 2,
            alpha: 0.5,
            ..PprParams::default()
        };
        let out = graph_expand(&store, &seeds, &two, None).unwrap();
        let scores: HashMap<String, f64> =
            out.iter().map(|r| (r.file_path.clone(), r.score)).collect();
        assert!((scores["b.md"] - 0.5).abs() < 1e-9);
        // b splits its mass between c and the backlink to a, then α² = 0.25.
        assert!((scores["c.md"] - 0.125).abs() < 1e-9);
        assert!((scores["a.md"] - 0.125).abs() < 1e-9);
        assert!(scores["b.md"] > scores["c.md"]);
    }

    #[test]
    fn expansion_follows_backlinks() {
        // Undirected: a note that links *into* the seed is a neighbour of it.
        let store = Store::open_memory().unwrap();
        let seed_file = chunked(&store, "seed.md", 1);
        let backlinker = chunked(&store, "backlink.md", 1);
        store
            .insert_edge(backlinker, 0, seed_file, DOC_LEVEL, "wikilink")
            .unwrap();

        let out = graph_expand(
            &store,
            &[seed(&store, seed_file, 0, 0.85)],
            &PprParams::default(),
            None,
        )
        .unwrap();
        assert_eq!(reached(&out), vec![("backlink.md".to_string(), 0)]);
    }

    #[test]
    fn an_unconnected_seed_expands_to_nothing() {
        let store = Store::open_memory().unwrap();
        let a = chunked(&store, "a.md", 1);
        let out = graph_expand(
            &store,
            &[seed(&store, a, 0, 0.9)],
            &PprParams::default(),
            None,
        )
        .unwrap();
        assert!(out.is_empty());
        assert!(
            graph_expand(&store, &[], &PprParams::default(), None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn an_unchunked_note_is_nothing_to_land_on() {
        // A document-level link into a note with no passages has no target to
        // divide across. It must drop out rather than arrive at a phantom seq 0.
        let store = Store::open_memory().unwrap();
        let src = file(&store, "src.md");
        let empty = file(&store, "empty.md");
        store
            .insert_edge(src, 0, empty, DOC_LEVEL, "wikilink")
            .unwrap();
        assert!(
            graph_expand(
                &store,
                &[seed(&store, src, 0, 1.0)],
                &PprParams::default(),
                None
            )
            .unwrap()
            .is_empty()
        );
    }

    // ── Tag scope (#60) ──────────────────────────────────────────

    /// Tag `file_id` with one path, so a scope can select it.
    fn tagged(store: &Store, file_id: i64, path: &str) {
        store
            .reconcile_file_tags(
                file_id,
                &[crate::tags::Tag {
                    path: path.into(),
                    display: path.into(),
                }],
            )
            .unwrap();
    }

    #[test]
    fn a_scope_drops_an_expansion_outside_it_and_the_next_one_takes_the_slot() {
        // #60. Two targets, one in scope and one out. Unscoped the lane is
        // allowed one expansion and takes the heavier target; scoped, the
        // filter runs before the truncate, so the quota goes to the in-scope
        // one instead of being spent and then emptied.
        let store = Store::open_memory().unwrap();
        let a = chunked(&store, "a.md", 1);
        let heavy = chunked(&store, "heavy.md", 1);
        let light = chunked(&store, "light.md", 1);
        tagged(&store, light, "type/undead");

        // Two edges into `heavy` and one into `light`, so `heavy` ranks first.
        store.insert_edge(a, 0, heavy, 0, "wikilink").unwrap();
        store.insert_edge(a, 0, light, 0, "wikilink").unwrap();
        let b = chunked(&store, "b.md", 1);
        store.insert_edge(b, 0, heavy, 0, "wikilink").unwrap();

        let one = PprParams {
            max_expansions: 1,
            ..ppr()
        };
        let seeds = vec![seed(&store, a, 0, 1.0), seed(&store, b, 0, 1.0)];

        let unscoped = graph_expand(&store, &seeds, &one, None).unwrap();
        assert_eq!(reached(&unscoped), vec![("heavy.md".to_string(), 0)]);

        let scope = std::collections::HashSet::from([light]);
        let scoped = graph_expand(&store, &seeds, &one, Some(&scope)).unwrap();
        assert_eq!(reached(&scoped), vec![("light.md".to_string(), 0)]);
    }

    #[test]
    fn a_walk_through_an_untagged_note_still_credits_its_in_scope_target() {
        // #60. The scope filters answers, not the vault. Two iterations, so
        // the mass reaches `target` only by passing through `middle`, which
        // carries no tag at all.
        let store = Store::open_memory().unwrap();
        let start = chunked(&store, "start.md", 1);
        let middle = chunked(&store, "middle.md", 1);
        let target = chunked(&store, "target.md", 1);
        tagged(&store, target, "type/undead");

        store.insert_edge(start, 0, middle, 0, "wikilink").unwrap();
        store.insert_edge(middle, 0, target, 0, "wikilink").unwrap();

        let two = PprParams {
            iterations: 2,
            ..ppr()
        };
        let scope = std::collections::HashSet::from([target]);
        let out = graph_expand(&store, &[seed(&store, start, 0, 1.0)], &two, Some(&scope)).unwrap();

        assert_eq!(reached(&out), vec![("target.md".to_string(), 0)]);
    }

    #[test]
    fn an_empty_scope_empties_the_graph_lane() {
        let store = Store::open_memory().unwrap();
        let a = chunked(&store, "a.md", 1);
        let b = chunked(&store, "b.md", 1);
        store.insert_edge(a, 0, b, 0, "wikilink").unwrap();

        let empty = std::collections::HashSet::new();
        let out = graph_expand(&store, &[seed(&store, a, 0, 1.0)], &ppr(), Some(&empty)).unwrap();
        assert!(out.is_empty());
    }
}
