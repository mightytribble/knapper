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
    /// Which lanes account for this result (#119).
    pub lanes: Vec<Lane>,
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
    /// Which lanes account for this result (#119).
    pub lanes: Vec<Lane>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// The one shape every surface renders a search through (#35).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchEnvelope {
    pub status: SearchStatus,
    pub degraded: bool,
    pub warnings: Vec<String>,
    /// Each note a block came from, keyed by path, with that note's
    /// frontmatter properties (#119).
    ///
    /// The rows sat on every block before, so a note that answered with four
    /// of its sections carried four identical copies. Omitted when no
    /// included note carries a property, so a vault with no custom properties
    /// renders as it did before (#66).
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub notes: std::collections::BTreeMap<String, NoteProperties>,
    pub blocks: Vec<Block>,
    pub overflow: Vec<Summary>,
    /// The candidates the answer floor rejected, ranked, filling the slots
    /// the answers did not use (#133).
    ///
    /// `no_results` alone cannot tell "the vault holds nothing like this"
    /// from "several notes nearly answered and the floor took them", and the
    /// two call for opposite next moves. These rows are the evidence behind
    /// the status and not a softening of it: `status` does not move, and a
    /// row carries no text, so acting on one is a deliberate `read`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub less_relevant: Vec<Summary>,
    /// The floor those candidates failed to clear, on the envelope's own
    /// 0-100 scale (#133).
    ///
    /// Which scorer ran decides the number — `[ranking] answer_floor` for
    /// the cross-encoder, `[calibrated] floor` for the logistic — and the
    /// one reported is the one that applied. A score shown against the other
    /// floor would be worse than showing nothing. Absent when no row is
    /// reported, so a floor never travels without the rows it explains.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer_floor: Option<f64>,
    /// The per-lane score breakdown, when the caller asked for it (#126).
    ///
    /// A client that reads `structuredContent` discards the text content
    /// blocks beside it, so a report sent as one reaches nothing. It travels
    /// in the envelope for that reason, and is absent for a caller that did
    /// not ask.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<String>,
}

/// One note's frontmatter properties, by name (#119).
pub type NoteProperties = std::collections::BTreeMap<String, PropertyValue>;

/// One property value, under its own JSON type (#119).
///
/// `store::PropertyRow` carries every value as a `String` with a separate
/// `kind`, so the row's text alone cannot tell the number `5` from the text
/// `"5"`, or a link from the words it is written with. The kind decides the
/// type here instead, which is what lets the rows collapse to a name-keyed
/// map without losing what they hold.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum PropertyValue {
    /// A key the note declares with no value.
    Empty,
    Checkbox(bool),
    Number(serde_json::Number),
    /// A wikilink, and the note it resolves to when it resolves at all.
    Link {
        /// The target as the note writes it, with `#Heading` and `|Display`
        /// dropped.
        link: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    Text(String),
    /// Every value of a name the note carries more than once. A YAML list
    /// writes one row per element, so the name is not unique in the rows.
    List(Vec<PropertyValue>),
}

impl PropertyValue {
    /// The row under its kind's own type. A `number` that does not parse
    /// stays text, so a malformed row is reported rather than dropped.
    fn from_row(r: &crate::store::PropertyRow) -> PropertyValue {
        use crate::properties::Kind;
        match r.kind {
            Kind::Empty => PropertyValue::Empty,
            Kind::Checkbox => PropertyValue::Checkbox(r.value == "true"),
            Kind::Number => match r.value.parse::<serde_json::Number>() {
                Ok(n) => PropertyValue::Number(n),
                Err(_) => PropertyValue::Text(r.value.clone()),
            },
            Kind::Link => PropertyValue::Link {
                link: r.value.clone(),
                path: r.target_path.clone(),
            },
            Kind::Text => PropertyValue::Text(r.value.clone()),
        }
    }
}

/// The knobs `assemble` reads; everything else about a search stays in
/// `InternalSearchResult` (#35).
pub struct AssembleParams<'a> {
    pub budget_tokens: u32,
    pub full: bool,
    pub summaries: bool,
    pub degraded: bool,
    pub per_note_cap: usize,
    /// How many rows the reply carries, of any tier (#133).
    ///
    /// It counted the answers before `less_relevant` existed, and the answers
    /// still take their slots first — what they leave is what the rejected
    /// candidates may fill. That is the whole budget for the new field: a
    /// search that answered in full reports none, which is what keeps the
    /// common query exactly as it was.
    pub top_n: usize,
    /// The candidates the answer floor rejected, ranked (#133).
    ///
    /// Empty on a degraded search, which ran no scorer and so has no
    /// probability and no floor, and empty when `[output]
    /// show_less_relevant` is off. The pipeline bounds the hydration; this
    /// function bounds what is reported.
    pub less_relevant: &'a [InternalSearchResult],
    /// The floor those candidates failed to clear, on the **scorer's** own
    /// 0-1 scale; `assemble` puts it on the envelope's 0-100 scale (#133).
    pub answer_floor: Option<f64>,
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

/// One retrieval lane, named on the wire the way the text rendering names it
/// (#119).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    Semantic,
    Keyword,
    /// The graph lane. `linked` is the word the text rendering has always
    /// used for it.
    Linked,
}

impl Lane {
    fn label(self) -> &'static str {
        match self {
            Lane::Semantic => "semantic",
            Lane::Keyword => "keyword",
            Lane::Linked => "linked",
        }
    }
}

impl Provenance {
    /// The lanes that fired, in the text rendering's order (#119).
    ///
    /// Three booleans and an empty `linked_from` cost a caller more than they
    /// tell it, so the wire carries the lanes that fired and nothing about
    /// the ones that did not. `linked_from` reaches no surface until the
    /// graph lane attributes its seeds (#74); `Provenance` keeps the field so
    /// that `coalesce` still merges it.
    pub fn lanes(&self) -> Vec<Lane> {
        let mut out = Vec::new();
        if self.semantic {
            out.push(Lane::Semantic);
        }
        if self.keyword {
            out.push(Lane::Keyword);
        }
        if self.graph {
            out.push(Lane::Linked);
        }
        out
    }

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

/// A rejected candidate's row: a summary that always carries its score (#133).
///
/// `overflow` reports a score only when the caller asked for one, because
/// there the row is an answer and the rank is the claim. Here the score
/// against the floor *is* the claim, and without it "less relevant" is an
/// assertion the caller cannot weigh.
fn rejected_of(r: &InternalSearchResult) -> Summary {
    Summary {
        score: Some(r.confidence),
        ..summary_of(r)
    }
}

fn summary_of(r: &InternalSearchResult) -> Summary {
    Summary {
        id: result_id(r),
        path: r.file_path.clone(),
        heading_path: r.heading_path.clone(),
        lanes: r.provenance.lanes(),
        score: None,
    }
}

fn block_of(r: &InternalSearchResult) -> Block {
    Block {
        id: result_id(r),
        path: r.file_path.clone(),
        heading_path: r.heading_path.clone(),
        lanes: r.provenance.lanes(),
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
        // The case #133 was raised for: nothing cleared the floor, and the
        // whole reply is what the floor rejected. Every slot is free.
        let less_relevant = rejected_rows(&p, 0);
        return SearchEnvelope {
            status: SearchStatus::NoResults,
            degraded: p.degraded,
            warnings: Vec::new(),
            notes: Default::default(),
            blocks: Vec::new(),
            overflow: Vec::new(),
            answer_floor: floor_of(&p, &less_relevant),
            less_relevant,
            explain: None,
        };
    }

    // Results cap on the assembled set; 0 is unbounded (#30, #34). This is
    // the spec-mandated cap read here, inert at its shipped 0. The pipeline
    // already applies `cap_per_file` before `take(top_n)`, so on the normal
    // call path this cap is redundant and idempotent; it stays as the guard
    // for a direct or future `assemble` caller that skips the pipeline cap
    // (#35).
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
        let less_relevant = rejected_rows(&p, overflow.len());
        return SearchEnvelope {
            status: SearchStatus::Ok,
            degraded: p.degraded,
            warnings: Vec::new(),
            notes: Default::default(),
            blocks,
            overflow,
            answer_floor: floor_of(&p, &less_relevant),
            less_relevant,
            explain: None,
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

    // Say that the budget is what shortened the answer (#102). `overflow`
    // names each held-back result, but a caller reading `blocks` alone sees a
    // short list and no reason for it — and the reason is a number they can
    // raise. It names no flag, because the CLI spells the budget `--tokens`
    // and MCP and HTTP spell it `budget_tokens`. The `summaries` return above
    // carries no warning, because there the empty `blocks` is what was asked
    // for.
    let mut warnings = Vec::new();
    if !overflow.is_empty() {
        let n = overflow.len();
        let s = if n == 1 { "result" } else { "results" };
        warnings.push(format!(
            "{n} {s} held back by the token budget; raise it to see {}",
            if n == 1 { "it" } else { "them" }
        ));
    }

    // One entry per note the included blocks came from, however many of its
    // sections answered (#119).
    let mut notes: std::collections::BTreeMap<String, NoteProperties> = Default::default();
    for r in &capped {
        if r.properties.is_empty() || notes.contains_key(&r.file_path) {
            continue;
        }
        notes.insert(r.file_path.clone(), note_properties(&r.properties));
    }

    // An `overflow` row cleared the floor and the budget held it back, so it
    // is an answer and keeps its slot. Counting blocks alone would serve
    // rejected candidates to a search that found plenty (#133).
    let less_relevant = rejected_rows(&p, blocks.len() + overflow.len());
    SearchEnvelope {
        status: SearchStatus::Ok,
        degraded: p.degraded,
        warnings,
        notes,
        blocks,
        overflow,
        answer_floor: floor_of(&p, &less_relevant),
        less_relevant,
        explain: None,
    }
}

/// The rejected candidates that fit the slots the answers did not use (#133).
///
/// `top_n` is the reply's whole row budget and the answers are served first,
/// so this needs no cap of its own and no tuned lower bound: a search that
/// answered in full reports nothing here, and one that answered nothing
/// reports up to `top_n`.
fn rejected_rows(p: &AssembleParams, answers: usize) -> Vec<Summary> {
    p.less_relevant
        .iter()
        .take(p.top_n.saturating_sub(answers))
        .map(rejected_of)
        .collect()
}

/// The floor on the envelope's 0-100 scale, and only beside the rows it
/// explains (#133).
///
/// `Summary::score` is the confidence — `rerank_score * 100` — so a floor
/// reported on the scorer's own 0-1 scale would sit in the same reply as the
/// scores it is meant to be read against and be a hundredfold out.
fn floor_of(p: &AssembleParams, rows: &[Summary]) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    p.answer_floor.map(|f| f * 100.0)
}

/// One note's property rows as a map from name to value (#119).
fn note_properties(rows: &[crate::store::PropertyRow]) -> NoteProperties {
    let mut out = NoteProperties::new();
    for r in rows {
        let v = PropertyValue::from_row(r);
        match out.entry(r.name.clone()) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(v);
            }
            // A YAML list writes one row per element, so a second row under
            // one name widens the value rather than replacing it (#119).
            std::collections::btree_map::Entry::Occupied(mut e) => match e.get_mut() {
                PropertyValue::List(items) => items.push(v),
                held => {
                    let first = std::mem::replace(held, PropertyValue::Empty);
                    *held = PropertyValue::List(vec![first, v]);
                }
            },
        }
    }
    out
}

fn provenance_label(lanes: &[Lane]) -> String {
    if lanes.is_empty() {
        return "none".to_string();
    }
    lanes
        .iter()
        .map(|l| l.label())
        .collect::<Vec<_>>()
        .join("+")
}

/// Carry the per-lane report in the envelope — `--explain` only (#126).
pub fn apply_explain(env: &mut SearchEnvelope, report: String) {
    env.explain = Some(report);
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
        // The message #34 always gave, and then the evidence behind it. The
        // status has not moved: these are what the floor rejected, and
        // saying so is what lets a caller tell a vault that covers nothing
        // from a query that missed (#133).
        let mut out = format!("{}\n", crate::ranking::NO_RELEVANT_CONTENT);
        out.push_str(&render_less_relevant(env));
        return out;
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
            provenance_label(&b.lanes)
        ));
        if b.truncated {
            out.push_str("(truncated)\n");
        }
        out.push_str(&b.text);
        out.push_str("\n\n");
    }
    // The reason, above the list it explains: "lower relevance" is the rank
    // order and not why these were cut, and the budget is a number the caller
    // can raise (#102).
    for w in &env.warnings {
        out.push_str(&format!("({w})\n"));
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
            provenance_label(&s.lanes)
        ));
    }
    out.push_str(&render_less_relevant(env));
    out
}

/// The rejected candidates, under a line that says what the claim is (#133).
///
/// The label carries the numbers rather than only the words: "less relevant"
/// alone reads as a weaker ranking of relevant things, which invites a caller
/// to treat the rows as answers. The score beside the floor it missed says
/// what the claim actually is, and the scorer is known to be wrong in a
/// measured way on short notes — so the rows are worth handing over, and
/// worth labelling honestly.
fn render_less_relevant(env: &SearchEnvelope) -> String {
    if env.less_relevant.is_empty() {
        return String::new();
    }
    let n = env.less_relevant.len();
    let mut out = match env.answer_floor {
        Some(floor) => format!(
            "\n({n} scored below the answer floor of {floor:.0}%, so {} not answers)\n",
            if n == 1 { "it is" } else { "they are" }
        ),
        None => format!("\n({n} scored below the answer floor)\n"),
    };
    for s in &env.less_relevant {
        let pct = match s.score {
            Some(v) => format!(" [{v:.0}%]"),
            None => String::new(),
        };
        out.push_str(&format!(
            "Below the floor: [{}]{pct} {} (matched: {})\n",
            s.id,
            s.heading_path,
            provenance_label(&s.lanes)
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
            properties: Vec::new(),
        }
    }
    fn params(budget: u32) -> AssembleParams<'static> {
        AssembleParams {
            budget_tokens: budget,
            full: false,
            summaries: false,
            degraded: false,
            per_note_cap: 0,
            top_n: 5,
            less_relevant: &[],
            answer_floor: None,
        }
    }

    /// #133: a search that cleared nothing still knows what it rejected, and
    /// the rows fill the slots the answers did not use.
    #[test]
    fn a_search_that_answered_nothing_reports_what_the_floor_rejected() {
        let rejected = vec![result(7, 10), result(8, 10)];
        let mut p = params(100_000);
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&[], p);

        assert_eq!(env.less_relevant.len(), 2);
        assert_eq!(env.less_relevant[0].id, "000007#7");
    }

    /// The status is the server's claim and the field is the evidence behind
    /// it. Handing back the rejects does not make them answers (#133).
    #[test]
    fn rejected_candidates_do_not_move_the_status() {
        let rejected = vec![result(7, 10)];
        let mut p = params(100_000);
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&[], p);

        assert_eq!(env.status, SearchStatus::NoResults);
    }

    /// The answers get the slots first. A search that filled `top_n` reports
    /// no rejects at all, which is what keeps the common query unchanged.
    #[test]
    fn a_full_answer_leaves_no_room_for_the_rejected() {
        let rs = vec![result(1, 10), result(2, 10), result(3, 10)];
        let rejected = vec![result(7, 10), result(8, 10)];
        let mut p = params(100_000);
        p.top_n = 3;
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&rs, p);

        assert_eq!(env.blocks.len(), 3);
        assert!(env.less_relevant.is_empty());
    }

    /// Two answers out of three slots leaves one, and one reject fills it.
    #[test]
    fn a_half_answer_fills_only_the_slots_it_did_not_use() {
        let rs = vec![result(1, 10), result(2, 10)];
        let rejected = vec![result(7, 10), result(8, 10)];
        let mut p = params(100_000);
        p.top_n = 3;
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&rs, p);

        assert_eq!(env.less_relevant.len(), 1);
        assert_eq!(env.less_relevant[0].id, "000007#7");
    }

    /// An overflow row cleared the floor — the budget held it back. It is an
    /// answer, so it keeps a slot the rejects cannot have.
    #[test]
    fn an_overflow_row_is_an_answer_and_keeps_its_slot() {
        // Budget 130 admits one 80-token block; the second overflows.
        let rs = vec![result(1, 80), result(2, 80)];
        let rejected = vec![result(7, 10), result(8, 10)];
        let mut p = params(130);
        p.top_n = 2;
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&rs, p);

        assert_eq!(env.blocks.len(), 1);
        assert_eq!(env.overflow.len(), 1);
        assert!(env.less_relevant.is_empty());
    }

    /// The score is the whole signal, so it is present whether or not the
    /// caller asked for scores, and the floor it missed rides beside it.
    #[test]
    fn a_rejected_row_carries_its_score_and_the_floor_it_missed() {
        let rejected = vec![result(7, 10)];
        let mut p = params(100_000);
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&[], p);

        // `result` scores 90.0 on the envelope's 0-100 scale; the floor is
        // reported on that same scale rather than the scorer's own 0-1.
        assert_eq!(env.less_relevant[0].score, Some(90.0));
        assert_eq!(env.answer_floor, Some(75.0));
    }

    /// The empty search says what it rejected, under the message it always
    /// gave. The numbers carry the claim: "less relevant" alone reads as a
    /// weaker ranking of relevant things (#133).
    #[test]
    fn the_text_rendering_names_the_floor_on_an_empty_search() {
        let rejected = vec![result(7, 10)];
        let mut p = params(100_000);
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&[], p);
        let text = render_text(&env, false);

        assert!(
            text.starts_with(crate::ranking::NO_RELEVANT_CONTENT),
            "the floor's own message went missing: {text}"
        );
        assert!(text.contains("75%"), "{text}");
        assert!(text.contains("[90%]"), "{text}");
        assert!(text.contains("000007#7"), "{text}");
    }

    /// The score rides on a rejected row whether or not `--scores` was asked
    /// for, because there the score against the floor is the whole claim.
    #[test]
    fn a_rejected_row_shows_its_score_without_being_asked() {
        let rejected = vec![result(7, 10)];
        let mut p = params(100_000);
        p.less_relevant = &rejected;
        p.answer_floor = Some(0.75);
        let env = assemble(&[], p);

        assert!(render_text(&env, false).contains("[90%]"));
    }

    /// A vault and query that reject nothing render exactly as they did
    /// before the field existed.
    #[test]
    fn rejecting_nothing_adds_no_json() {
        let rs = vec![result(1, 10)];
        let env = assemble(&rs, params(100_000));
        let json = serde_json::to_value(&env).unwrap();

        assert!(json.get("less_relevant").is_none(), "{json}");
        assert!(json.get("answer_floor").is_none(), "{json}");
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
    fn the_budget_says_so_when_it_holds_results_back() {
        let rs = vec![result(1, 80), result(2, 80), result(3, 80)];
        let env = assemble(&rs, params(260));
        assert_eq!(
            env.warnings,
            vec!["1 result held back by the token budget; raise it to see it".to_string()]
        );
    }

    #[test]
    fn a_budget_that_holds_nothing_back_warns_about_nothing() {
        let rs = vec![result(1, 80), result(2, 80)];
        let env = assemble(&rs, params(100_000));
        assert!(env.warnings.is_empty());
        assert!(env.overflow.is_empty());
    }

    #[test]
    fn summaries_overflow_is_not_a_budget_warning() {
        // `summaries` puts every rank in overflow by request, not because the
        // budget stopped anything.
        let rs = vec![result(1, 80), result(2, 80)];
        let mut p = params(100_000);
        p.summaries = true;
        assert!(assemble(&rs, p).warnings.is_empty());
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

    #[test]
    fn apply_scores_fills_matching_confidence_when_not_degraded() {
        // Budget 200 admits one 80-token block (130) and overflows the next.
        let mut r1 = result(1, 80);
        r1.confidence = 77.0;
        let mut r2 = result(2, 80);
        r2.confidence = 42.0;
        let rs = vec![r1, r2];
        let mut env = assemble(&rs, params(200));
        assert_eq!(env.blocks.len(), 1);
        assert_eq!(env.overflow.len(), 1);
        apply_scores(&mut env, &rs);
        assert_eq!(env.blocks[0].score, Some(77.0));
        assert_eq!(env.overflow[0].score, Some(42.0));
    }

    #[test]
    fn apply_scores_leaves_scores_none_when_degraded() {
        let mut r1 = result(1, 80);
        r1.confidence = 77.0;
        let mut r2 = result(2, 80);
        r2.confidence = 42.0;
        let rs = vec![r1, r2];
        let mut p = params(200);
        p.degraded = true;
        let mut env = assemble(&rs, p);
        assert!(env.degraded);
        assert_eq!(env.blocks.len(), 1);
        assert_eq!(env.overflow.len(), 1);
        apply_scores(&mut env, &rs);
        assert!(env.blocks.iter().all(|b| b.score.is_none()));
        assert!(env.overflow.iter().all(|s| s.score.is_none()));
    }

    #[test]
    fn block_and_summary_wire_shape_matches_the_contract() {
        // This locks the wire contract of #35: what serde emits on the JSON
        // wire, not just what the Rust struct holds. Budget 260 admits the
        // first two 80-token blocks (130+130); rank 3 overflows.
        let rs = vec![result(1, 80), result(2, 80), result(3, 80)];
        let mut env = assemble(&rs, params(260));

        let block_json = serde_json::to_value(&env.blocks[0]).unwrap();
        assert_eq!(block_json["id"], "000001#1");
        assert_eq!(block_json["path"], "n1.md");
        assert_eq!(block_json["heading_path"], "n1.md > H");
        assert_eq!(block_json["text"], "x".repeat(80 * 3));
        assert_eq!(block_json["untrusted_content"], true);
        assert_eq!(block_json["truncated"], false);
        assert_eq!(block_json["lanes"], serde_json::json!(["keyword"]));
        // No score requested: the field is absent from the wire, not null.
        assert!(!block_json.as_object().unwrap().contains_key("score"));

        // A Summary carries no text at all.
        let summary_json = serde_json::to_value(&env.overflow[0]).unwrap();
        assert!(!summary_json.as_object().unwrap().contains_key("text"));

        // Once scores are asked for, the field appears and carries a number.
        apply_scores(&mut env, &rs);
        let scored_block_json = serde_json::to_value(&env.blocks[0]).unwrap();
        assert!(scored_block_json["score"].is_number());
        assert_eq!(scored_block_json["score"], 90.0);
    }

    #[test]
    fn a_lanes_wire_name_and_its_text_label_are_the_same_word() {
        // The JSON name comes from serde and the text rendering's from
        // `label`, so the two can drift apart on a rename.
        for l in [Lane::Semantic, Lane::Keyword, Lane::Linked] {
            assert_eq!(serde_json::to_value(l).unwrap(), l.label());
        }
    }

    #[test]
    fn a_row_names_the_lanes_that_fired_and_no_others() {
        // `linked_from` has shipped permanently empty since #35, and three
        // false flags say nothing a caller can act on (#119).
        let mut r = result(1, 10);
        r.provenance = Provenance {
            keyword: true,
            semantic: true,
            graph: false,
            linked_from: vec![],
        };
        let env = assemble(&[r], params(10_000));
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(
            json["blocks"][0]["lanes"],
            serde_json::json!(["semantic", "keyword"])
        );
        let block = json["blocks"][0].as_object().unwrap();
        assert!(!block.contains_key("provenance"));
        assert!(!block.contains_key("linked_from"));
    }

    #[test]
    fn an_overflow_row_names_its_lanes_the_same_way() {
        let mut r = result(1, 10);
        r.provenance = Provenance {
            keyword: false,
            semantic: false,
            graph: true,
            linked_from: vec![],
        };
        let env = assemble(
            &[r],
            AssembleParams {
                budget_tokens: 10_000,
                full: false,
                summaries: true,
                degraded: false,
                per_note_cap: 0,
                top_n: 5,
                less_relevant: &[],
                answer_floor: None,
            },
        );
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["overflow"][0]["lanes"], serde_json::json!(["linked"]));
    }

    fn prop(name: &str, value: &str, kind: crate::properties::Kind) -> crate::store::PropertyRow {
        crate::store::PropertyRow {
            chunk_seq: crate::store::DOC_LEVEL,
            heading_path: None,
            name: name.into(),
            value: value.into(),
            kind,
            target_path: None,
        }
    }

    #[test]
    fn two_blocks_of_one_note_share_one_notes_entry() {
        // Item 1 of #119: the duplication this removes is the whole point —
        // four chunks of one note carried four identical property copies.
        use crate::properties::Kind;
        let mut a = result(1, 10);
        let mut b = result(2, 10);
        b.file_path = a.file_path.clone();
        let rows = vec![prop("status", "draft", Kind::Text)];
        a.properties = rows.clone();
        b.properties = rows;
        let env = assemble(&[a, b], params(10_000));
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["notes"]["n1.md"]["status"], "draft");
        assert_eq!(json["notes"].as_object().unwrap().len(), 1);
        // The rows no longer ride on the block.
        assert!(
            !json["blocks"][0]
                .as_object()
                .unwrap()
                .contains_key("properties")
        );
    }

    fn link_prop(name: &str, value: &str, target: Option<&str>) -> crate::store::PropertyRow {
        let mut r = prop(name, value, crate::properties::Kind::Link);
        r.target_path = target.map(str::to_string);
        r
    }

    #[test]
    fn a_link_carries_the_note_it_resolves_to() {
        // The resolved path is the addressable half: it is what a caller can
        // hand to `read` without another lookup (#119).
        let mut r = result(1, 10);
        r.properties = vec![link_prop(
            "part_of",
            "Volgard",
            Some("locations/Volgard.md"),
        )];
        let env = assemble(&[r], params(10_000));
        let n = &serde_json::to_value(&env).unwrap()["notes"]["n1.md"];
        assert_eq!(n["part_of"]["link"], "Volgard");
        assert_eq!(n["part_of"]["path"], "locations/Volgard.md");
    }

    #[test]
    fn a_link_that_resolves_to_nothing_carries_no_path() {
        let mut r = result(1, 10);
        r.properties = vec![link_prop("part_of", "Nowhere", None)];
        let env = assemble(&[r], params(10_000));
        let n = &serde_json::to_value(&env).unwrap()["notes"]["n1.md"];
        assert_eq!(n["part_of"]["link"], "Nowhere");
        assert!(!n["part_of"].as_object().unwrap().contains_key("path"));
    }

    #[test]
    fn a_name_the_note_carries_twice_becomes_an_array() {
        // A YAML list writes one row per element (`properties.rs`), so a
        // name-keyed map has to hold both or silently drop one (#119).
        use crate::properties::Kind;
        let mut r = result(1, 10);
        r.properties = vec![
            prop("realm", "Skaldi", Kind::Text),
            prop("realm", "Volgard", Kind::Text),
            prop("status", "draft", Kind::Text),
        ];
        let env = assemble(&[r], params(10_000));
        let n = &serde_json::to_value(&env).unwrap()["notes"]["n1.md"];
        assert_eq!(n["realm"], serde_json::json!(["Skaldi", "Volgard"]));
        // A name carried once stays a scalar.
        assert_eq!(n["status"], "draft");
    }

    #[test]
    fn a_scalar_property_keeps_its_own_json_type() {
        // `PropertyRow.value` is always a String, so `kind` is the only thing
        // that separates the number 5 from the text "5" (#119).
        use crate::properties::Kind;
        let mut r = result(1, 10);
        r.properties = vec![
            prop("level", "5", Kind::Number),
            prop("rating", "4.5", Kind::Number),
            prop("done", "true", Kind::Checkbox),
            prop("note", "5", Kind::Text),
            prop("blank", "", Kind::Empty),
        ];
        let env = assemble(&[r], params(10_000));
        let n = &serde_json::to_value(&env).unwrap()["notes"]["n1.md"];
        assert_eq!(n["level"], 5);
        assert_eq!(n["rating"], 4.5);
        assert_eq!(n["done"], true);
        assert_eq!(n["note"], "5");
        assert!(n["blank"].is_null());
    }

    #[test]
    fn the_sidecar_carries_a_notes_properties_and_is_absent_when_none_does() {
        use crate::properties::Kind;
        let mut r = result(1, 10);
        r.properties = vec![prop("status", "draft", Kind::Text)];
        let env = assemble(&[r.clone()], params(10_000));
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["notes"]["n1.md"]["status"], "draft");
        r.properties.clear();
        let env = assemble(&[r], params(10_000));
        let json = serde_json::to_string(&env).unwrap();
        assert!(!json.contains("\"notes\""), "{json}");
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
            notes: Default::default(),
            blocks: vec![Block {
                id: "abc#0".into(),
                path: "n.md".into(),
                heading_path: "n.md > H".into(),
                lanes: Provenance {
                    keyword: true,
                    semantic: true,
                    graph: false,
                    linked_from: vec![],
                }
                .lanes(),
                text: "body".into(),
                untrusted_content: true,
                truncated: false,
                score: None,
            }],
            overflow: vec![],
            less_relevant: vec![],
            answer_floor: None,
            explain: None,
        };
        let out = render_text(&env, false);
        assert!(out.contains("[abc#0]"));
        assert!(out.contains("semantic+keyword"));
        assert!(!out.contains('%'));
    }

    #[test]
    fn text_render_gives_the_reason_above_the_excluded_list() {
        let env = SearchEnvelope {
            status: SearchStatus::Ok,
            degraded: false,
            warnings: vec!["1 result held back by the token budget".to_string()],
            notes: Default::default(),
            blocks: vec![],
            overflow: vec![Summary {
                id: "000002#2".into(),
                path: "b.md".into(),
                heading_path: "b.md > B".into(),
                lanes: Provenance {
                    keyword: true,
                    semantic: false,
                    graph: false,
                    linked_from: vec![],
                }
                .lanes(),
                score: None,
            }],
            less_relevant: vec![],
            answer_floor: None,
            explain: None,
        };
        let text = render_text(&env, false);
        let reason = text
            .find("1 result held back")
            .expect("the warning renders");
        let listed = text
            .find("Not included")
            .expect("the excluded list renders");
        assert!(reason < listed, "the reason comes before the list: {text}");
    }

    #[test]
    fn no_results_text_is_the_literal_message() {
        let env = SearchEnvelope {
            status: SearchStatus::NoResults,
            degraded: false,
            warnings: vec![],
            notes: Default::default(),
            blocks: vec![],
            overflow: vec![],
            less_relevant: vec![],
            answer_floor: None,
            explain: None,
        };
        assert_eq!(
            render_text(&env, false).trim(),
            crate::ranking::NO_RELEVANT_CONTENT
        );
    }

    #[test]
    fn text_render_includes_the_percentage_only_when_scores_is_requested() {
        let env = SearchEnvelope {
            status: SearchStatus::Ok,
            degraded: false,
            warnings: vec![],
            notes: Default::default(),
            blocks: vec![Block {
                id: "abc#0".into(),
                path: "n.md".into(),
                heading_path: "n.md > H".into(),
                lanes: Provenance {
                    keyword: true,
                    semantic: true,
                    graph: false,
                    linked_from: vec![],
                }
                .lanes(),
                text: "body".into(),
                untrusted_content: true,
                truncated: false,
                score: Some(83.0),
            }],
            overflow: vec![],
            less_relevant: vec![],
            answer_floor: None,
            explain: None,
        };
        assert!(render_text(&env, true).contains("[83%]"));
        assert!(!render_text(&env, false).contains('%'));
    }

    #[test]
    fn text_render_shows_the_degraded_banner() {
        let env = SearchEnvelope {
            status: SearchStatus::Ok,
            degraded: true,
            warnings: vec![],
            notes: Default::default(),
            blocks: vec![],
            overflow: vec![],
            less_relevant: vec![],
            answer_floor: None,
            explain: None,
        };
        assert!(
            render_text(&env, false).contains("(degraded ordering: no cross-encoder available)")
        );
    }
}
