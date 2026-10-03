//! The `chunks_fts` keyword index: its declaration, its triggers and its queries.

use super::Store;
use anyhow::{Context, Result};

/// A single result from an FTS5 full-text search.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub file_id: i64,
    pub chunk_seq: i64,
    pub score: f64,
    pub snippet: String,
}

/// The keyword index's declaration and its three sync triggers, as one batch of
/// SQL, because `fts_fingerprint` hashes the text (issue #31).
///
/// This is the one fingerprint input that needs no version constant beside it:
/// the schema *is* the text, so any change to the column list or to a trigger
/// body changes the digest exactly and nothing else does. `[fts]` reaches the
/// fingerprint through here too, since the flags decide the column list.
///
/// `chunks_fts` is **external content** over `chunks` (issue #37). It stores an
/// index and no text of its own, and it reads every column value back out of
/// the chunk row. Two consequences, both of them the point:
///
/// - the keyword index cannot hold a different string from `chunks.text`, which
///   is the state issue #11 existed to repair. There is no second copy to
///   disagree.
/// - the triggers are the only writer. A chunk row inserted, updated or deleted
///   by any path updates the index, including the delete SQLite performs itself
///   when `files` cascades. Measured: after `DELETE FROM files`, the index is
///   empty and `integrity-check` passes.
///
/// The column order is body, breadcrumb, tags, and `bm25()` takes its weights
/// in that order. A disabled column is *absent* from the declaration rather
/// than present at weight zero: BM25 normalises over every token in the row, so
/// a populated column at weight 0.0 still moves every score, while a column the
/// table does not declare is exactly inert.
pub fn fts_objects_sql(cfg: &crate::config::FtsConfig) -> String {
    let mut columns = vec!["text"];
    if cfg.heading_path {
        columns.push("heading_path");
    }
    if cfg.tags {
        columns.push("tags_text");
    }
    let column_list = columns.join(", ");
    // `new.`/`old.` qualified, for the trigger bodies.
    let new_values = columns
        .iter()
        .map(|c| format!("new.{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    let old_values = columns
        .iter()
        .map(|c| format!("old.{c}"))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                {column_list},
                content='chunks',
                content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, {column_list})
                    VALUES (new.id, {new_values});
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, {column_list})
                    VALUES ('delete', old.id, {old_values});
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, {column_list})
                    VALUES ('delete', old.id, {old_values});
                INSERT INTO chunks_fts(rowid, {column_list})
                    VALUES (new.id, {new_values});
            END;"
    )
}

impl Store {
    /// The columns `chunks_fts` is declared over, or `None` if it does not
    /// exist. The shape the store is *in*, as against the one `[fts]` asks for.
    pub fn fts_columns(&self) -> Result<Option<Vec<String>>> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'chunks_fts'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        if !exists {
            return Ok(None);
        }
        let mut stmt = self.conn.prepare("PRAGMA table_info(chunks_fts)")?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Some(names))
    }

    /// Whether the keyword index in the store is the one `cfg` describes.
    fn fts_shape_matches(&self, cfg: &crate::config::FtsConfig) -> Result<bool> {
        let mut wanted = vec!["text".to_string()];
        if cfg.heading_path {
            wanted.push("heading_path".to_string());
        }
        if cfg.tags {
            wanted.push("tags_text".to_string());
        }
        Ok(self.fts_columns()? == Some(wanted))
    }

    /// Create the keyword index and its triggers if the store has none.
    ///
    /// Called during init, where no [`Config`](crate::config::Config) has been
    /// read yet, so it builds the default shape. A store whose index is already
    /// declared some other way is left exactly as it is: the triggers name the
    /// table's own columns, so creating a set that disagrees with the table
    /// would break the next chunk insert. Reconciling the two is a write-path
    /// job — see [`sync_fts_objects`](Self::sync_fts_objects) — and until it
    /// runs, `fts_fingerprint` blocks the read paths anyway.
    pub fn ensure_fts_table(&self) -> Result<()> {
        let cfg = crate::config::FtsConfig::default();
        match self.fts_columns()? {
            None => self
                .conn
                .execute_batch(&fts_objects_sql(&cfg))
                .context("failed to create FTS5 virtual table")?,
            // `IF NOT EXISTS` throughout, so this only fills in a trigger that
            // an interrupted earlier run left uncreated.
            Some(_) if self.fts_shape_matches(&cfg)? => self
                .conn
                .execute_batch(&fts_objects_sql(&cfg))
                .context("failed to create FTS5 triggers")?,
            Some(_) => {}
        }
        Ok(())
    }

    /// Make the keyword index the shape `cfg` describes, rebuilding it if it is
    /// not. Returns the number of rows indexed, or `None` if nothing was done.
    ///
    /// A write path calls this, because it is the path that has a config in
    /// hand. On a fresh store this is what turns the default shape built by
    /// `init` into the configured one, at a cost of nothing, since there are no
    /// chunks yet.
    pub fn sync_fts_objects(&self, cfg: &crate::config::FtsConfig) -> Result<Option<usize>> {
        if self.fts_shape_matches(cfg)? {
            return Ok(None);
        }
        Ok(Some(self.rebuild_fts(cfg)?))
    }

    /// Discard `chunks_fts` and re-derive it from the `chunks` table.
    ///
    /// The action `fts_fingerprint` declares (issue #31). It reads no files and
    /// runs no model: every column the index is declared over is a column of
    /// `chunks`, so the keyword index is derivable from what is already stored.
    /// That is the only reason an FTS schema change is cheap rather than a
    /// reindex.
    ///
    /// The triggers go with the table. They name the table's columns, so a set
    /// left behind from an earlier declaration would fail on the next chunk
    /// insert, and `DROP TABLE` does not take them with it.
    pub fn rebuild_fts(&self, cfg: &crate::config::FtsConfig) -> Result<usize> {
        self.conn.execute_batch(
            "DROP TRIGGER IF EXISTS chunks_fts_insert;
             DROP TRIGGER IF EXISTS chunks_fts_delete;
             DROP TRIGGER IF EXISTS chunks_fts_update;
             DROP TABLE IF EXISTS chunks_fts;",
        )?;
        self.conn.execute_batch(&fts_objects_sql(cfg))?;
        // The external-content rebuild command. It reads the content table
        // directly, which is why it reproduces a trigger-built index exactly
        // rather than approximately.
        self.conn
            .execute_batch("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")?;
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM chunks_fts", [], |row| {
                row.get::<_, i64>(0)
            })? as usize)
    }

    /// Search the FTS5 index. Returns results ranked by BM25 score.
    /// BM25 in SQLite returns negative values (more negative = better match),
    /// so we negate them to get positive scores where higher = better.
    ///
    /// The query is wrapped in double quotes so that FTS5 treats it as a
    /// phrase/literal rather than interpreting operators like `-`.
    ///
    /// Unweighted, and that is a decision rather than an omission: this is the
    /// identity-resolution query, which asks whether a name appears verbatim.
    /// Weighting a column changes the order among rows that already match, and
    /// no caller of this function ranks by that order.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>> {
        self.fts_search_expr(&crate::fts::phrase_expr(query), limit, &[], None)
    }

    /// Keyword search matching **any** token of `query`, each taken literally.
    ///
    /// What the search lane wants, and what [`Self::fts_search`] cannot give it:
    /// a phrase query only fires where the caller already guessed the corpus's
    /// wording. See [`crate::fts::any_term_expr`] for the measurements (#22).
    ///
    /// A query with no searchable token returns no rows rather than an error.
    ///
    /// `weights` are the BM25 column weights, in the order `chunks_fts` declares
    /// its columns — [`FtsConfig::weights`](crate::config::FtsConfig::weights)
    /// builds them from the same config the declaration came from. An empty
    /// slice is plain `bm25()`, every column at 1.0.
    ///
    /// `scope` is the tag scope's file ids, or `None` for the whole vault
    /// (#60). See [`fts_search_expr`](Self::fts_search_expr) for how it binds.
    pub fn fts_search_any(
        &self,
        query: &str,
        limit: usize,
        weights: &[f64],
        scope: Option<&[i64]>,
    ) -> Result<Vec<FtsResult>> {
        match crate::fts::any_term_expr(query) {
            Some(expr) => self.fts_search_expr(&expr, limit, weights, scope),
            None => Ok(Vec::new()),
        }
    }

    /// How many chunks match one FTS5 expression — `df(t)` for the calibrated
    /// bound. Callers quote a single term with [`crate::fts::phrase_expr`]; the
    /// count is rows matching in any indexed column, which is the population
    /// the scorer's idf is computed over.
    pub fn fts_doc_frequency(&self, term_expr: &str) -> Result<u64> {
        Ok(self.conn.query_row(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?1",
            [term_expr],
            |row| row.get::<_, i64>(0),
        )? as u64)
    }

    /// Run a prepared FTS5 MATCH expression. Callers build the expression with
    /// `crate::fts`, which is where the quoting rules and their reasons live.
    ///
    /// `file_id` and `chunk_seq` come from a join and not from the index: since
    /// issue #37 `chunks_fts` is external content over `chunks`, and it is
    /// keyed on the chunk's rowid rather than carrying a copy of the pair.
    ///
    /// `scope` narrows the match to a set of file ids, pre-filtering the FTS5
    /// query rather than cutting its answer — the same property `vecstore::
    /// search_vec` holds for the semantic lane (#60).
    fn fts_search_expr(
        &self,
        fts_query: &str,
        limit: usize,
        weights: &[f64],
        scope: Option<&[i64]>,
    ) -> Result<Vec<FtsResult>> {
        // More weights than the table has columns is an error in SQLite, and a
        // caller that holds a different `[fts]` from the one the store was built
        // with would hit it. `fingerprint::verify` already refuses that state on
        // the paths that read a config, so the ones that reach here with a
        // mismatch are the ones carrying defaults; they get a weight per column
        // rather than a failed query that reads as an empty keyword lane.
        let declared = self.fts_columns()?.map(|c| c.len()).unwrap_or(0);
        let weights = &weights[..weights.len().min(declared)];

        // Interpolated rather than bound: `bm25()`'s weights are arguments to a
        // function in the select list, and SQLite has no way to bind a variadic
        // argument list. They are `f64` and formatted here, so no caller can put
        // anything else in the string.
        let bm25 = match weights.is_empty() {
            true => "bm25(chunks_fts)".to_string(),
            false => format!(
                "bm25(chunks_fts, {})",
                weights
                    .iter()
                    .map(|w| format!("{w:.6}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        // The scope binds as `?3`, so the two parameters that were here keep
        // their positions and the `LIMIT` stays where it reads (#60).
        let mut binds: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(fts_query.to_string()), Box::new(limit as i64)];
        let scope_clause = match scope {
            None => "",
            Some(ids) => {
                let array: rusqlite::vtab::array::Array = std::rc::Rc::new(
                    ids.iter()
                        .copied()
                        .map(rusqlite::types::Value::from)
                        .collect::<Vec<_>>(),
                );
                binds.push(Box::new(array));
                " AND c.file_id IN rarray(?3)"
            }
        };
        let mut stmt = self.conn.prepare(&format!(
            "SELECT c.file_id, c.seq, {bm25} as score,
                    snippet(chunks_fts, 0, '<b>', '</b>', '...', 64)
             FROM chunks_fts
             JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?1{scope_clause}
             ORDER BY score
             LIMIT ?2",
        ))?;

        let rows = stmt.query_map(rusqlite::params_from_iter(binds.iter()), |row| {
            Ok(FtsResult {
                file_id: row.get(0)?,
                chunk_seq: row.get(1)?,
                score: {
                    let raw: f64 = row.get(2)?;
                    -raw // negate: SQLite BM25 returns negative, more negative = better
                },
                snippet: row.get(3)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::*;

    use crate::docid::generate_docid;

    /// A file with `n` chunks, each carrying a breadcrumb and the file's tags.
    fn indexed_file(store: &Store, path: &str, sections: &[(&str, &str)]) -> i64 {
        let file_id = store
            .insert_file(path, "h", 0, &generate_docid(path), None, None)
            .unwrap();
        for (seq, (heading, text)) in sections.iter().enumerate() {
            store
                .insert_chunk(&NewChunk {
                    file_id,
                    seq: seq as i64,
                    heading,
                    heading_path: &format!("Doc > {heading}"),
                    tags_text: "grimoire",
                    text,
                    vector_id: (file_id * 100 + seq as i64) as u64,
                    token_count: 10,
                })
                .unwrap();
        }
        file_id
    }

    /// Every posting in the index: which term, in which row, in which column,
    /// at which offset. This is the index's content, read through `fts5vocab`.
    ///
    /// Not the bytes of `chunks_fts_data`. Those differ, and legitimately: an
    /// incremental write leaves one segment per batch where a rebuild writes a
    /// single merged one. Segmentation is a storage layout that every query
    /// reads through, so the postings are what "the same index" has to mean.
    fn fts_postings(store: &Store) -> Vec<(String, i64, String, i64)> {
        store
            .conn
            .execute_batch(
                "DROP TABLE IF EXISTS fts_vocab;
                 CREATE VIRTUAL TABLE fts_vocab USING fts5vocab(chunks_fts, 'instance');",
            )
            .unwrap();
        let mut stmt = store
            .conn
            .prepare("SELECT term, doc, col, offset FROM fts_vocab ORDER BY 1, 2, 3, 4")
            .unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .unwrap();
        rows.map(|r| r.unwrap()).collect()
    }

    /// The invariant the issue names: `'rebuild'` reproduces what the triggers
    /// built. If it did not, `rebuild_fts` — the action `fts_fingerprint`
    /// declares — would return a *different* index from an incremental write,
    /// and a store's keyword results would depend on how it got there.
    #[test]
    fn a_rebuild_reproduces_the_trigger_built_index_exactly() {
        let store = Store::open_memory().unwrap();
        indexed_file(
            &store,
            "rules/spells.md",
            &[
                ("Abjuration", "Counterspell stops a caster."),
                ("Restoration", "Mend Object repairs torn cloth."),
            ],
        );
        indexed_file(&store, "lore/dragon.md", &[("Definition", "Rank SS.")]);

        let scores = |store: &Store| -> Vec<(i64, i64, String)> {
            store
                .fts_search_any(
                    "counterspell cloth grimoire Abjuration",
                    10,
                    &[1.0, 3.0, 4.0],
                    None,
                )
                .unwrap()
                .iter()
                .map(|r| (r.file_id, r.chunk_seq, format!("{:.9}", r.score)))
                .collect()
        };
        let by_trigger = fts_postings(&store);
        let by_trigger_scores = scores(&store);
        assert!(!by_trigger.is_empty() && !by_trigger_scores.is_empty());

        let rows = store
            .rebuild_fts(&crate::config::FtsConfig::default())
            .unwrap();

        assert_eq!(rows, 3, "one indexed row per chunk");
        assert_eq!(by_trigger, fts_postings(&store), "postings differ");
        assert_eq!(by_trigger_scores, scores(&store), "BM25 differs");
    }

    /// Insert, update and delete round-trips agree between the two tables. The
    /// triggers are the only writer, so this is the whole contract — and it is
    /// what makes #11's bug class, a keyword index holding a different string
    /// from the chunk, unreachable rather than fixed.
    #[test]
    fn every_write_to_chunks_reaches_the_keyword_index() {
        let store = Store::open_memory().unwrap();
        let file_id = indexed_file(&store, "n.md", &[("One", "alpha bravo")]);
        let hit = |term: &str| store.fts_search(term, 10).unwrap().len();

        assert_eq!(hit("alpha"), 1);
        assert_eq!(hit("One"), 1, "the breadcrumb column is indexed");

        store
            .conn
            .execute(
                "UPDATE chunks SET text = 'charlie delta' WHERE file_id = ?1",
                params![file_id],
            )
            .unwrap();
        assert_eq!(hit("alpha"), 0, "the old text is still indexed");
        assert_eq!(hit("charlie"), 1);

        store.delete_chunks_for_file(file_id).unwrap();
        assert_eq!(hit("charlie"), 0);
        assert_eq!(hit("One"), 0);
    }

    /// The delete SQLite performs itself, on the cascade from `files`, fires
    /// the trigger too. Nothing in the write paths has to remember the keyword
    /// index, which is the reason the explicit deletes could be removed.
    #[test]
    fn a_cascade_from_files_takes_the_keyword_index_with_it() {
        let store = Store::open_memory().unwrap();
        let file_id = indexed_file(&store, "n.md", &[("One", "alpha bravo")]);

        store.delete_file(file_id).unwrap();

        assert_eq!(store.fts_search("alpha", 10).unwrap().len(), 0);
        // A desynced external-content index is exactly what this reports.
        store
            .conn
            .execute_batch("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1);")
            .unwrap();
    }

    /// The control is declared over the body alone, so a heading term and a tag
    /// stop being reachable. That is what makes it a control and not a setting
    /// with a smaller weight.
    #[test]
    fn the_control_declares_the_body_column_only() {
        let store = Store::open_memory().unwrap();
        indexed_file(&store, "n.md", &[("Abjuration", "alpha bravo")]);
        store
            .rebuild_fts(&crate::config::FtsConfig::CONTROL)
            .unwrap();

        assert_eq!(store.fts_columns().unwrap(), Some(vec!["text".to_string()]));
        assert_eq!(store.fts_search("alpha", 10).unwrap().len(), 1);
        assert_eq!(store.fts_search("Abjuration", 10).unwrap().len(), 0);
        assert_eq!(store.fts_search("grimoire", 10).unwrap().len(), 0);
    }

    /// A store whose index is declared some other way is left alone until a
    /// path holding a config reconciles it. Creating triggers that name columns
    /// the table does not have would break the next chunk insert, and the store
    /// has no config to know better with.
    #[test]
    fn init_leaves_an_index_it_did_not_declare_alone() {
        let store = Store::open_memory().unwrap();
        store
            .rebuild_fts(&crate::config::FtsConfig::CONTROL)
            .unwrap();
        store.ensure_fts_table().unwrap();
        assert_eq!(store.fts_columns().unwrap(), Some(vec!["text".to_string()]));

        // And the write path is what fixes it.
        let rebuilt = store
            .sync_fts_objects(&crate::config::FtsConfig::default())
            .unwrap();
        assert_eq!(rebuilt, Some(0), "an empty store, rebuilt");
        assert_eq!(
            store.fts_columns().unwrap(),
            Some(vec!["text".to_string(), "heading_path".to_string()])
        );
        assert_eq!(
            store
                .sync_fts_objects(&crate::config::FtsConfig::default())
                .unwrap(),
            None,
            "a matching declaration is not rebuilt"
        );
    }
}
