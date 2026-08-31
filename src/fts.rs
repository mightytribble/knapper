// FTS5 search support.
//
// The `FtsResult` struct and `fts_search` method live on `Store` (in store.rs)
// since the store owns the database connection. We re-export `FtsResult` here
// so downstream code can import it from either location.
//
// The MATCH expressions themselves are built here, because which one a caller
// wants is a retrieval decision rather than a storage one — see the two
// builders below.

pub use crate::store::FtsResult;

/// Quote one token so FTS5 reads every character in it literally.
///
/// Without this, `-`, `*`, `:`, `^`, `(` and the bare words `AND`/`OR`/`NOT`
/// are query syntax, so a docid (`#a1b2c3`), a ticket ID (`BRE-1234`) or a
/// language name (`C++`) would be parsed instead of searched for.
fn quote_token(token: &str) -> String {
    format!("\"{}\"", token.replace('"', "\"\""))
}

/// A MATCH expression requiring the query to appear **verbatim and contiguous**.
///
/// This is the right shape for identity work — resolving `Archivist Lenne` to a
/// person's note should not match every chunk that says "Archivist" — and the
/// wrong shape for a keyword lane, which is what issue #22 was about.
pub fn phrase_expr(query: &str) -> String {
    quote_token(query)
}

/// The query's searchable tokens: whitespace-split, alphanumeric-bearing,
/// case-insensitively deduplicated, first spelling kept. This is the token
/// list behind [`any_term_expr`], exposed on its own because the calibrated
/// bound needs the terms themselves — one doc-frequency count each — and not
/// the joined MATCH expression (spec 2026-08-30).
pub fn query_terms(query: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    query
        .split_whitespace()
        // A token with no alphanumerics contributes no searchable term. FTS5
        // tolerates one (`"-"` matches nothing rather than erroring), but it
        // would still be noise in `--explain`.
        .filter(|t| t.chars().any(|c| c.is_alphanumeric()))
        // Repeats change nothing about which rows match, and reading
        // `"the" OR "the"` in an explain trace invites a bug hunt that ends
        // nowhere.
        .filter(|t| seen.insert(t.to_lowercase()))
        .map(str::to_string)
        .collect()
}

/// A MATCH expression satisfied by **any** token of the query, each still
/// literal: `"dragon" OR "human" OR "form"`.
///
/// This is what the keyword lane wants. Quoting the whole query instead, as the
/// lane used to, makes it a phrase query — and a phrase query matches only where
/// the user has already guessed the corpus's exact wording. Measured on the
/// 1598-chunk eval vault, four of the five seed probes returned zero rows, and
/// the keyword lane's only source of hits was the single-word fragments the
/// word-splitter happened to produce (#22, and why #18 became a deletion).
///
/// One OR expression also scores better than the same tokens run as separate
/// queries: BM25 weights each term by IDF *within* the expression, where
/// per-token queries produce scores from different queries that `collapse_lane`
/// then pools as though they shared a scale.
///
/// Returns `None` when nothing searchable survives — a query of pure
/// punctuation — so the caller can skip the round trip rather than issue a
/// MATCH that can only return nothing.
pub fn any_term_expr(query: &str) -> Option<String> {
    let terms: Vec<String> = query_terms(query).iter().map(|t| quote_token(t)).collect();
    if terms.is_empty() {
        return None;
    }
    Some(terms.join(" OR "))
}

#[cfg(test)]
mod expr_tests {
    use super::{any_term_expr, phrase_expr, query_terms};

    #[test]
    fn a_phrase_expression_quotes_the_whole_query() {
        assert_eq!(phrase_expr("human form"), r#""human form""#);
    }

    #[test]
    fn query_terms_is_the_token_list_behind_any_term_expr() {
        assert_eq!(query_terms("storm wolf"), vec!["storm", "wolf"]);
        assert_eq!(
            query_terms("The the THE wolf"),
            vec!["The", "wolf"],
            "case-insensitive dedup, first spelling kept"
        );
        assert_eq!(
            query_terms("- -- ---"),
            Vec::<String>::new(),
            "no alphanumerics, no terms"
        );
    }

    #[test]
    fn any_term_splits_on_whitespace() {
        assert_eq!(
            any_term_expr("dragon human form").unwrap(),
            r#""dragon" OR "human" OR "form""#
        );
    }

    /// The non-regression guarantee for #22: a single-token query is the same
    /// string under both builders, so probe 4 (`Archdragon`, the BM25
    /// exact-name probe) cannot move as a result of this change.
    #[test]
    fn a_single_token_is_identical_under_both_builders() {
        for q in ["Archdragon", "BRE-1234", "#a1b2c3", "C++"] {
            assert_eq!(any_term_expr(q).unwrap(), phrase_expr(q), "query {q:?}");
        }
    }

    /// Quoting is why the lane can search for a ticket ID at all — unquoted,
    /// FTS5 reads `-`, `*` and `^` as operators.
    #[test]
    fn every_term_stays_literal() {
        assert_eq!(
            any_term_expr("BRE-1234 C++ *wild*").unwrap(),
            r#""BRE-1234" OR "C++" OR "*wild*""#
        );
    }

    #[test]
    fn an_embedded_quote_is_escaped_not_dropped() {
        assert_eq!(
            any_term_expr(r#"say "hi""#).unwrap(),
            r#""say" OR """hi""""#
        );
    }

    #[test]
    fn repeated_terms_appear_once_regardless_of_case() {
        assert_eq!(
            any_term_expr("Dragon dragon DRAGON form").unwrap(),
            r#""Dragon" OR "form""#
        );
    }

    #[test]
    fn terms_with_no_alphanumerics_are_dropped() {
        assert_eq!(
            any_term_expr("human -- form").unwrap(),
            r#""human" OR "form""#
        );
    }

    /// No searchable term means no round trip, rather than a MATCH that can
    /// only return nothing.
    #[test]
    fn a_query_with_nothing_to_search_for_yields_no_expression() {
        assert!(any_term_expr("").is_none());
        assert!(any_term_expr("   ").is_none());
        assert!(any_term_expr("-- ... ***").is_none());
    }
}

#[cfg(test)]
mod tests {
    use crate::docid::generate_docid;
    use crate::store::{NewChunk, Store};

    fn setup_store() -> Store {
        let store = Store::open_memory().unwrap();
        store.ensure_fts_table().unwrap();
        store
    }

    /// Index one chunk of `file_id`, the only way there is: `chunks_fts` is
    /// external content over `chunks`, so a keyword index entry exists exactly
    /// where a chunk row does (issue #37). `vector_id` is unique per store and
    /// otherwise unused here.
    fn index_chunk(store: &Store, file_id: i64, seq: i64, text: &str) {
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq,
                text,
                vector_id: (file_id * 100 + seq) as u64,
                token_count: text.split_whitespace().count() as i64,
                ..Default::default()
            })
            .unwrap();
    }

    #[test]
    fn test_fts_exact_match() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/ticket.md",
                "hash1",
                100,
                &generate_docid("notes/ticket.md"),
                None,
                None,
            )
            .unwrap();

        index_chunk(
            &store,
            file_id,
            0,
            "BRE-2579 delivery date extension for checkout",
        );

        let results = store.fts_search("BRE-2579", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_id, file_id);
        assert_eq!(results[0].chunk_seq, 0);
        assert!(
            results[0].score > 0.0,
            "score should be positive (negated BM25)"
        );
    }

    #[test]
    fn test_fts_no_match() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/note.md",
                "hash1",
                100,
                &generate_docid("notes/note.md"),
                None,
                None,
            )
            .unwrap();

        index_chunk(&store, file_id, 0, "Rust programming language guide");

        let results = store.fts_search("kubernetes", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    /// The defect in #22, at the store boundary: a multi-word query is a phrase
    /// query, so it finds nothing unless the caller already knew the wording.
    /// `fts_search_any` is the search lane's answer; `fts_search` keeps the
    /// phrase behaviour that identity resolution depends on.
    #[test]
    fn a_multi_word_query_matches_terms_only_via_fts_search_any() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/note.md",
                "hash1",
                100,
                &generate_docid("notes/note.md"),
                None,
                None,
            )
            .unwrap();

        index_chunk(&store, file_id, 0, "Rust programming language guide");

        // The words are all present, but not contiguous and not in this order.
        assert!(store.fts_search("guide to Rust", 10).unwrap().is_empty());
        assert_eq!(
            store
                .fts_search_any("guide to Rust", 10, &[], None)
                .unwrap()
                .len(),
            1
        );

        // Any one term is enough, which is what makes the lane a recall lane.
        assert_eq!(
            store
                .fts_search_any("kubernetes Rust", 10, &[], None)
                .unwrap()
                .len(),
            1
        );
        assert!(
            store
                .fts_search_any("kubernetes helm", 10, &[], None)
                .unwrap()
                .is_empty()
        );

        // Nothing searchable: no rows, no error.
        assert!(
            store
                .fts_search_any("-- ...", 10, &[], None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_fts_multiple_results() {
        let store = setup_store();

        let file_id1 = store
            .insert_file(
                "notes/a.md",
                "h1",
                100,
                &generate_docid("notes/a.md"),
                None,
                None,
            )
            .unwrap();
        let file_id2 = store
            .insert_file(
                "notes/b.md",
                "h2",
                100,
                &generate_docid("notes/b.md"),
                None,
                None,
            )
            .unwrap();
        let file_id3 = store
            .insert_file(
                "notes/c.md",
                "h3",
                100,
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        // Chunk with "delivery" appearing multiple times should rank higher.
        index_chunk(
            &store,
            file_id1,
            0,
            "delivery date delivery schedule delivery tracking",
        );
        index_chunk(&store, file_id2, 0, "delivery date for the checkout page");
        index_chunk(
            &store,
            file_id3,
            0,
            "unrelated content about Rust and WebAssembly",
        );

        let results = store.fts_search("delivery", 10).unwrap();
        assert_eq!(results.len(), 2, "only 2 chunks mention 'delivery'");

        // Results should be sorted by score descending.
        assert!(
            results[0].score >= results[1].score,
            "results should be ranked by relevance"
        );
    }

    #[test]
    fn deleting_a_files_chunks_removes_its_keyword_index_entries() {
        let store = setup_store();
        let file_id = store
            .insert_file(
                "notes/del.md",
                "hash1",
                100,
                &generate_docid("notes/del.md"),
                None,
                None,
            )
            .unwrap();

        index_chunk(&store, file_id, 0, "first chunk content");
        index_chunk(&store, file_id, 1, "second chunk content");

        // Verify they exist.
        let results = store.fts_search("chunk", 10).unwrap();
        assert_eq!(results.len(), 2);

        // Deleting the chunks deletes the index entries: the triggers are the
        // only writer, so there is no second delete to forget (issue #37).
        store.delete_chunks_for_file(file_id).unwrap();
        let results = store.fts_search("chunk", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn a_scope_keeps_the_keyword_lane_inside_the_tagged_notes() {
        // #60. The out-of-scope note is the better BM25 match, so a scoped
        // query returning it would mean the filter ran after the ranking.
        let store = crate::store::Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };

        let wight = store
            .insert_file("wight.md", "h", 1, "d000001", None, None)
            .unwrap();
        let wolf = store
            .insert_file("wolf.md", "h", 2, "d000002", None, None)
            .unwrap();
        store
            .reconcile_file_tags(wight, &[tag("type/undead")])
            .unwrap();
        store
            .reconcile_file_tags(wolf, &[tag("type/beast")])
            .unwrap();

        store
            .insert_chunk(&crate::store::NewChunk {
                file_id: wight,
                seq: 0,
                heading: "## Wight",
                text: "A wight lurks.",
                vector_id: 1,
                token_count: 5,
                ..Default::default()
            })
            .unwrap();
        store
            .insert_chunk(&crate::store::NewChunk {
                file_id: wolf,
                seq: 0,
                heading: "## Wolf",
                text: "A wight, a wight, a wight and a wolf.",
                vector_id: 2,
                token_count: 12,
                ..Default::default()
            })
            .unwrap();

        // limit = 1, not some larger number: at a limit both rows fit under,
        // a post-filter over the unscoped top-N would smuggle `wight` back in
        // and this test would pass whether the SQL scoped the query or not.
        // At limit 1 the unscoped top-1 is `wolf` alone, so a post-filter
        // implementation has nothing left to filter `wight` into (#60).
        let unscoped: Vec<i64> = store
            .fts_search_any("wight", 1, &[], None)
            .unwrap()
            .iter()
            .map(|r| r.file_id)
            .collect();
        assert_eq!(
            unscoped.first(),
            Some(&wolf),
            "the fixture only tests anything if the out-of-scope note wins unscoped"
        );

        let scope = [wight];
        let scoped: Vec<i64> = store
            .fts_search_any("wight", 1, &[], Some(&scope))
            .unwrap()
            .iter()
            .map(|r| r.file_id)
            .collect();
        assert_eq!(scoped, vec![wight]);
    }
}
