//! The `chunks` table: a note's sections, their text and their stored vectors.

use super::Store;
use anyhow::Result;
use rusqlite::{OptionalExtension, params};

/// A record representing a chunk of a file.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub id: i64,
    pub file_id: i64,
    /// Ordinal position within the file, 0-based. `(file_id, seq)` is the chunk's
    /// retrieval identity: it is the only key the semantic and FTS lanes can both
    /// produce, so it is what search dedups and fuses on.
    pub seq: i64,
    pub heading: String,
    /// Leading 200 characters of `text`, for display. Derived on insert, never
    /// supplied — see [`Store::insert_chunk`].
    pub snippet: String,
    /// The whole chunk, as chunked and as embedded.
    ///
    /// Empty only on a database written before the column existed whose FTS row
    /// could not be found to backfill from. Nothing should read this without
    /// deciding what an empty one means.
    pub text: String,
    /// The breadcrumb this chunk is indexed under, `Note Title > H1 > H2`
    /// (issue #37). Empty on a database written before the column existed, and
    /// on a chunk of a file with no headings.
    pub heading_path: String,
    /// The file's frontmatter tags, sorted and space separated (issue #37).
    pub tags_text: String,
    pub vector_id: u64,
    pub token_count: i64,
}

/// Columns selected for every [`ChunkRecord`], in the order [`chunk_from_row`] expects.
const CHUNK_COLUMNS: &str =
    "id, file_id, seq, heading, snippet, text, heading_path, tags_text, vector_id, token_count";

/// The `chunk_seq` standing for "the document as a whole" on either end of an edge.
///
/// Edges are chunk-to-chunk (issue #28), but not every link names a chunk. On the
/// source end this is a link the indexer could not attribute to a passage; on the
/// target end it is a plain `[[Note]]`, or a `[[Note#Section]]` whose heading no
/// longer resolves. Reading it as "every chunk of that file" is what keeps the
/// document-level view — `SELECT DISTINCT from_file, to_file` — complete.
///
/// A sentinel rather than `NULL` because SQLite counts two NULLs as *distinct*
/// in a `UNIQUE` constraint, which would quietly stop `INSERT OR IGNORE` from
/// deduplicating the commonest edge there is.
pub const DOC_LEVEL: i64 = -1;

/// Reduce a heading to the form two spellings of the same section share.
///
/// Strips the leading `#`s a stored heading carries and a link's does not,
/// case-folds, and drops every trailing `(cont.)` — a `[[Note#Events]]` means
/// `## Events (cont.)` too, whether the suffix is on the row because an index
/// written before #139 labelled a split piece that way, or because the note
/// wrote the heading itself.
///
/// The loop is what folds a compounded `## Events (cont.) (cont.)`: after one
/// suffix comes off, the remainder ends in a space, and a single
/// `trim_end_matches` stops there.
pub(crate) fn normalise_heading(heading: &str) -> String {
    let mut base = heading.trim_start_matches('#').trim();
    while let Some(shorter) = base.strip_suffix("(cont.)") {
        base = shorter.trim_end();
    }
    base.to_lowercase()
}

/// Build a [`ChunkRecord`] from a row selecting [`CHUNK_COLUMNS`].
fn chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRecord> {
    Ok(ChunkRecord {
        id: row.get(0)?,
        file_id: row.get(1)?,
        seq: row.get(2)?,
        heading: row.get(3)?,
        snippet: row.get(4)?,
        text: row.get(5)?,
        heading_path: row.get(6)?,
        tags_text: row.get(7)?,
        vector_id: row.get::<_, i64>(8)? as u64,
        token_count: row.get(9)?,
    })
}

/// One chunk row, as it is written.
///
/// A struct rather than nine positional arguments, and named after the thing
/// `fingerprint::CHUNK_RECORD_VERSION` versions: what a chunk row records is now
/// something a reader depends on, so it is worth having one place that says what
/// that is. `Default` gives every text field the empty string, which is what a
/// test that cares about two of them wants.
#[derive(Debug, Clone, Copy, Default)]
pub struct NewChunk<'a> {
    pub file_id: i64,
    /// The chunk's 0-based position in its file.
    pub seq: i64,
    /// The chunk's own heading line, as the chunker found it.
    pub heading: &'a str,
    /// The breadcrumb — `crate::prefix::breadcrumb` (issue #37).
    pub heading_path: &'a str,
    /// The file's frontmatter tags, sorted and space separated (issue #37).
    pub tags_text: &'a str,
    /// The whole chunk. `snippet` is derived from it.
    pub text: &'a str,
    pub vector_id: u64,
    pub token_count: i64,
}

impl Store {
    /// Delete a file's chunks without touching its `files` row.
    ///
    /// The re-index counterpart of [`delete_file`](Self::delete_file): keeping
    /// the row keeps the file's id, and keeping the id keeps the edges other
    /// files point at it.
    pub fn delete_chunks_for_file(&self, file_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
        Ok(())
    }

    // See [`NewChunk`] for what one row holds and why it is one argument.

    /// Insert a chunk.
    ///
    /// `text` is the **whole chunk**. The `snippet` column is derived from it
    /// here rather than passed in: a chunk row that holds a preview but not the
    /// text it previews is the state issue #14 exists to remove, and taking one
    /// argument makes it unreachable.
    ///
    /// The keyword index needs no separate write. `chunks_fts` is external
    /// content over this table, so the insert trigger indexes the row (#37).
    pub fn insert_chunk(&self, chunk: &NewChunk<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chunks
                (file_id, seq, heading, heading_path, tags_text, snippet, text, vector_id, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                chunk.file_id,
                chunk.seq,
                chunk.heading,
                chunk.heading_path,
                chunk.tags_text,
                crate::chunker::make_snippet(chunk.text),
                chunk.text,
                chunk.vector_id as i64,
                chunk.token_count
            ],
        )?;
        Ok(())
    }

    /// Insert a chunk with its embedding vector stored as a BLOB.
    pub fn insert_chunk_with_vector(&self, chunk: &NewChunk<'_>, vector: &[f32]) -> Result<()> {
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO chunks
                (file_id, seq, heading, heading_path, tags_text, snippet, text, vector_id, token_count, vector)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                chunk.file_id,
                chunk.seq,
                chunk.heading,
                chunk.heading_path,
                chunk.tags_text,
                crate::chunker::make_snippet(chunk.text),
                chunk.text,
                chunk.vector_id as i64,
                chunk.token_count,
                vector_bytes
            ],
        )?;
        Ok(())
    }

    /// How many rows the keyword index covers — BM25's N, for the calibrated
    /// bound (spec 2026-08-30).
    pub fn chunk_row_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |row| {
                row.get::<_, i64>(0)
            })? as u64)
    }

    /// Every chunk vector of the named files, for pool candidates no content
    /// lane fetched (a graph or temporal admission). Decoded the way
    /// [`Self::get_all_vectors`] decodes; rows with no vector are skipped.
    ///
    /// It reads a whole file's vectors rather than the pool's own rows. The
    /// pool holds `(file_id, seq)` pairs, but `rarray` binds one flat list of
    /// one type, so narrowing this needs the signature to take pairs and a
    /// `WHERE (file_id, seq) IN (VALUES ...)` built at call time — not a
    /// filter that can be added to the clause below. Filtering the result in
    /// Rust saves nothing, because avoiding the fetch is the point. The
    /// over-fetch is bounded: it fires only for candidates no content lane
    /// found, so the file count is at most `graph_reserve + temporal_reserve`,
    /// and the waste per file is that note's chunk count. At the measured
    /// corpus the whole calibrated path costs 1.7 ms a query, so this is
    /// recorded rather than fixed.
    pub fn vectors_for_files(&self, file_ids: &[i64]) -> Result<Vec<(i64, i64, Vec<f32>)>> {
        if file_ids.is_empty() {
            return Ok(Vec::new());
        }
        let array: rusqlite::vtab::array::Array = std::rc::Rc::new(
            file_ids
                .iter()
                .copied()
                .map(rusqlite::types::Value::from)
                .collect::<Vec<_>>(),
        );
        let mut stmt = self.conn.prepare(
            "SELECT file_id, seq, vector FROM chunks
             WHERE file_id IN rarray(?1) AND vector IS NOT NULL",
        )?;
        let rows = stmt.query_map([array], |row| {
            let file_id: i64 = row.get(0)?;
            let seq: i64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let vector = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((file_id, seq, vector))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Get all stored vectors with their IDs.
    /// Returns (vector_id, vector) pairs.
    pub fn get_all_vectors(&self) -> Result<Vec<(u64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id, vector FROM chunks WHERE vector IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            let vid: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let vector: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((vid as u64, vector))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn get_chunks_by_file(&self, file_id: i64) -> Result<Vec<ChunkRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE file_id = ?1 ORDER BY seq"
        ))?;
        let rows = stmt.query_map(params![file_id], chunk_from_row)?;
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row?);
        }
        Ok(chunks)
    }

    /// The seqs of a file's chunks sitting under `heading`.
    ///
    /// A deep link's target end (issue #28). Plural because `(file, heading)` is
    /// not unique: an oversized section is split across several chunks, each
    /// labelled with the section's own heading, and a link to `#Events` means
    /// every one of them (#139). A store written before that carries the later
    /// pieces as `## Events (cont.)`, which [`normalise_heading`] folds.
    ///
    /// Empty when nothing matches — a renamed heading. The caller degrades that
    /// to [`DOC_LEVEL`] rather than dropping the link, because a deep link is
    /// more fragile than a plain one and the graph must not lose recall over a
    /// retitled section.
    pub fn chunk_seqs_with_heading(&self, file_id: i64, heading: &str) -> Result<Vec<i64>> {
        let wanted = normalise_heading(heading);
        let mut stmt = self
            .conn
            .prepare("SELECT seq, heading FROM chunks WHERE file_id = ?1 ORDER BY seq")?;
        let rows = stmt.query_map(params![file_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut seqs = Vec::new();
        for row in rows {
            let (seq, stored) = row?;
            if normalise_heading(&stored) == wanted {
                seqs.push(seq);
            }
        }
        Ok(seqs)
    }

    pub fn get_chunk_by_vector_id(&self, vector_id: u64) -> Result<Option<ChunkRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE vector_id = ?1"
        ))?;
        let mut rows = stmt.query_map(params![vector_id as i64], chunk_from_row)?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    /// Look up a chunk by its retrieval identity.
    ///
    /// The FTS lane returns `(file_id, chunk_seq)` and nothing else; this is how
    /// it recovers the heading and full snippet the semantic lane gets for free.
    pub fn get_chunk_by_seq(&self, file_id: i64, seq: i64) -> Result<Option<ChunkRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE file_id = ?1 AND seq = ?2"
        ))?;
        let mut rows = stmt.query_map(params![file_id, seq], chunk_from_row)?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    /// Fetch the full text of each `(file_id, seq)` in one pass.
    ///
    /// This is the reranker's read (issue #14): a cross-encoder has to see the
    /// chunk, not the preview a lane happened to attach to it. An entry is
    /// `None` when the chunk is gone or predates `chunks.text` and could not be
    /// backfilled; the caller decides what to fall back to.
    pub fn get_chunk_texts(&self, keys: &[(i64, i64)]) -> Result<Vec<Option<String>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT text FROM chunks WHERE file_id = ?1 AND seq = ?2")?;
        keys.iter()
            .map(|(file_id, seq)| {
                let text: Option<String> = stmt
                    .query_row(params![file_id, seq], |row| row.get(0))
                    .optional()?;
                Ok(text.filter(|t| !t.is_empty()))
            })
            .collect()
    }

    /// Return vector_ids for all chunks belonging to a file.
    /// Read before a changed file's chunks are replaced, so their vectors can be deleted.
    pub fn get_vector_ids_for_file(&self, file_id: i64) -> Result<Vec<u64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id FROM chunks WHERE file_id = ?1")?;
        let rows = stmt.query_map(params![file_id], |row| Ok(row.get::<_, i64>(0)? as u64))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }

    /// The chunk seqs each of `file_ids` actually has, in order.
    ///
    /// What a [`DOC_LEVEL`] link resolves to at *walk* time. #28 stores such a
    /// link as one row rather than one row per target chunk; this is the other
    /// half of that decision — the set is materialised only when a walk needs to
    /// divide mass across it, and never in the table.
    pub fn chunk_seqs_for_files(
        &self,
        file_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, Vec<i64>>> {
        let mut map: std::collections::HashMap<i64, Vec<i64>> = std::collections::HashMap::new();
        if file_ids.is_empty() {
            return Ok(map);
        }
        let ph = vec!["?"; file_ids.len()].join(",");
        let sql = format!(
            "SELECT file_id, seq FROM chunks WHERE file_id IN ({ph}) ORDER BY file_id, seq"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(file_ids.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (file_id, seq) = row?;
            map.entry(file_id).or_default().push(seq);
        }
        Ok(map)
    }

    /// Check if a file's FTS5 content contains a term. Escapes for FTS5.
    pub fn file_contains_term(&self, file_id: i64, term: &str) -> Result<bool> {
        let escaped = term.replace('"', "\"\"");
        let query = format!("\"{}\"", escaped);
        let result: Result<i64, _> = self.conn.query_row(
            "SELECT 1 FROM chunks_fts
             JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?1 AND c.file_id = ?2 LIMIT 1",
            params![query, file_id],
            |row| row.get(0),
        );
        Ok(result.is_ok())
    }

    /// Which chunk of `file_id` best matches any of `terms`, by BM25.
    ///
    /// Returns `None` when no chunk of the file matches — which is also the
    /// relevance signal `file_contains_term` used to give.
    ///
    /// No longer on any retrieval path: the graph lane used this to name a
    /// section for a file it had ranked, and since #29 the walk is over chunks
    /// and returns the chunk it reached (issue #29). Kept as the primitive that
    /// answers "which passage of this note holds this term", which is what the
    /// indexer's and writer's chunk-identity tests assert against.
    pub fn best_matching_chunk_seq(&self, file_id: i64, terms: &[String]) -> Result<Option<i64>> {
        if terms.is_empty() {
            return Ok(None);
        }
        let disjunction = terms
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");

        let result: rusqlite::Result<i64> = self.conn.query_row(
            "SELECT c.seq FROM chunks_fts
             JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?1 AND c.file_id = ?2
             ORDER BY bm25(chunks_fts) LIMIT 1",
            params![disjunction, file_id],
            |row| row.get(0),
        );
        match result {
            Ok(seq) => Ok(Some(seq)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            // A malformed FTS expression means no match, not a failed search.
            Err(_) => Ok(None),
        }
    }

    pub fn next_vector_id(&self) -> Result<u64> {
        let max: Option<i64> = self
            .conn
            .query_row("SELECT MAX(vector_id) FROM chunks", [], |row| row.get(0))
            .ok()
            .flatten();
        Ok(max.map_or(0, |m| m as u64 + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::docid::generate_docid;
    use crate::store::fixtures::*;

    #[test]
    fn test_insert_and_get_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/chunk_test.md",
                "hash1",
                100,
                &generate_docid("notes/chunk_test.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 0,
                heading: "Heading 1",
                text: "Some text here",
                vector_id: 1,
                token_count: 42,
                ..Default::default()
            })
            .unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 1,
                heading: "Heading 2",
                text: "More text",
                vector_id: 2,
                token_count: 30,
                ..Default::default()
            })
            .unwrap();

        let chunks = store.get_chunks_by_file(file_id).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "Heading 1");
        assert_eq!(chunks[0].vector_id, 1);
        assert_eq!(chunks[0].token_count, 42);
        assert_eq!(chunks[1].snippet, "More text");

        let chunk = store.get_chunk_by_vector_id(2).unwrap().unwrap();
        assert_eq!(chunk.heading, "Heading 2");
    }

    #[test]
    fn test_file_contains_term() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/fts.md",
                "h1",
                100,
                &generate_docid("n/fts.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_chunk(&NewChunk {
                file_id: f1,
                seq: 0,
                text: "BRE-2579 delivery date extension",
                vector_id: 1,
                token_count: 4,
                ..Default::default()
            })
            .unwrap();

        assert!(store.file_contains_term(f1, "delivery").unwrap());
        assert!(store.file_contains_term(f1, "extension").unwrap());
        assert!(!store.file_contains_term(f1, "checkout").unwrap());
    }

    /// Insert a file with one chunk per (heading, text) pair, numbered in order.
    fn seed_sections(store: &Store, path: &str, sections: &[(&str, &str)]) -> i64 {
        let docid = generate_docid(path);
        store
            .insert_file(path, "hash", 100, &docid, None, None)
            .unwrap();
        let file_id = store.get_file(path).unwrap().unwrap().id;
        for (seq, (heading, text)) in sections.iter().enumerate() {
            store
                .insert_chunk(&NewChunk {
                    file_id,
                    seq: seq as i64,
                    heading,
                    text,
                    vector_id: (file_id * 100 + seq as i64) as u64,
                    token_count: 10,
                    ..Default::default()
                })
                .unwrap();
        }
        file_id
    }

    #[test]
    fn test_get_chunk_by_seq() {
        let store = Store::open_memory().unwrap();
        let file_id = seed_sections(
            &store,
            "rules/abjuration.md",
            &[
                ("## Level 3 Counterspell", "stops a spell being cast"),
                ("## Level 9 Dimensional Anchor", "pins a creature in place"),
            ],
        );

        let chunk = store.get_chunk_by_seq(file_id, 1).unwrap().unwrap();
        assert_eq!(chunk.heading, "## Level 9 Dimensional Anchor");
        assert_eq!(chunk.seq, 1);

        assert!(store.get_chunk_by_seq(file_id, 9).unwrap().is_none());
    }

    #[test]
    fn test_best_matching_chunk_seq_picks_the_matching_section() {
        let store = Store::open_memory().unwrap();
        // Snippets carry their heading line, as the chunker emits them.
        let file_id = seed_sections(
            &store,
            "rules/abjuration.md",
            &[
                ("## Overview", "## Overview\nan introduction to wards"),
                (
                    "## Counterspell",
                    "## Counterspell\nstops a spell being cast",
                ),
                (
                    "## Dimensional Anchor",
                    "## Dimensional Anchor\npins a creature in place",
                ),
            ],
        );

        let seq = store
            .best_matching_chunk_seq(file_id, &["counterspell".to_string()])
            .unwrap();
        assert_eq!(seq, Some(1), "must name the section that matched");

        // No match is the relevance signal the graph lane filters on.
        let none = store
            .best_matching_chunk_seq(file_id, &["quantum".to_string()])
            .unwrap();
        assert!(none.is_none());

        // No terms cannot mean "everything matches".
        assert!(
            store
                .best_matching_chunk_seq(file_id, &[])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_best_matching_chunk_seq_scores_across_all_terms() {
        let store = Store::open_memory().unwrap();
        let file_id = seed_sections(
            &store,
            "notes/temple.md",
            &[
                ("## Description", "the temple stands at the crossroads"),
                ("## Noises", "the archivist investigates strange noises"),
            ],
        );

        // "temple" matches section 0 and "noises" matches section 1; the section
        // matching more of the query has to win, or a stopword picks the answer.
        let terms = vec![
            "investigates".to_string(),
            "strange".to_string(),
            "noises".to_string(),
            "temple".to_string(),
        ];
        assert_eq!(
            store.best_matching_chunk_seq(file_id, &terms).unwrap(),
            Some(1)
        );
    }

    /// The reranker's read. A missing chunk is `None` rather than an error, so
    /// one stale candidate cannot take the whole lane down.
    #[test]
    fn get_chunk_texts_reports_misses_without_failing() {
        let store = Store::open_memory().unwrap();
        let file_id = store.insert_file("a.md", "h", 0, "d", None, None).unwrap();
        let long = "x".repeat(500);
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 0,
                heading: "H",
                text: &long,
                vector_id: 1,
                token_count: 10,
                ..Default::default()
            })
            .unwrap();

        let texts = store
            .get_chunk_texts(&[(file_id, 0), (file_id, 9), (999, 0)])
            .unwrap();

        assert_eq!(texts[0].as_deref(), Some(long.as_str()));
        assert_eq!(texts[1], None, "no such seq");
        assert_eq!(texts[2], None, "no such file");
    }

    #[test]
    fn test_next_vector_id_empty() {
        let store = Store::open_memory().unwrap();
        assert_eq!(store.next_vector_id().unwrap(), 0);
    }

    #[test]
    fn chunk_seqs_with_heading_finds_every_piece_of_a_split_section() {
        // What the chunker writes since #139: each piece of a split section
        // carries the section's own heading, so `(file, heading)` is not
        // unique and a link to `#Events` means every piece.
        let store = Store::open_memory().unwrap();
        let f = file(&store, "session.md");
        for (seq, heading) in [(0, "## Summary"), (1, "## Events"), (2, "## Events")] {
            store
                .insert_chunk_with_vector(
                    &NewChunk {
                        file_id: f,
                        seq,
                        heading,
                        text: "text",
                        vector_id: seq as u64,
                        token_count: 1,
                        ..Default::default()
                    },
                    &[0.0],
                )
                .unwrap();
        }
        assert_eq!(
            store.chunk_seqs_with_heading(f, "Events").unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn chunk_seqs_with_heading_finds_a_split_section_in_a_store_written_before_139() {
        // A store an earlier binary built labelled the later pieces
        // `## Events (cont.)`. `normalise_heading` folds the suffix, so a deep
        // link still resolves to both without a re-index.
        let store = Store::open_memory().unwrap();
        let f = file(&store, "session.md");
        for (seq, heading) in [
            (0, "## Summary"),
            (1, "## Events"),
            (2, "## Events (cont.)"),
        ] {
            store
                .insert_chunk_with_vector(
                    &NewChunk {
                        file_id: f,
                        seq,
                        heading,
                        text: "text",
                        vector_id: seq as u64,
                        token_count: 1,
                        ..Default::default()
                    },
                    &[0.0],
                )
                .unwrap();
        }
        assert_eq!(
            store.chunk_seqs_with_heading(f, "Events").unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            store.chunk_seqs_with_heading(f, "summary").unwrap(),
            vec![0]
        );
        assert!(
            store
                .chunk_seqs_with_heading(f, "Aftermath")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn normalise_heading_folds_a_compounded_continuation() {
        // `split_oversized_chunks` runs over what the packing budget already
        // cut, so a twice-split section was labelled `## X (cont.) (cont.)`.
        // `trim_end_matches` stopped after one suffix — the remainder ends in
        // `"(cont.) "`, and the space defeats the match — so `[[Note#X]]` did
        // not resolve to that piece (#139).
        assert_eq!(normalise_heading("## Events (cont.) (cont.)"), "events");
        assert_eq!(normalise_heading("## Events (cont.)"), "events");
        assert_eq!(normalise_heading("## Events"), "events");
    }

    /// Two files, three chunks: "storm wolf" is in two of them, "basilisk"
    /// in the third alone. Each chunk carries a vector, for
    /// `vectors_for_files` to read back.
    fn seed_calibrated_fixture(store: &Store) -> (i64, i64) {
        let file_a = store
            .insert_file(
                "notes/a.md",
                "ha",
                100,
                &generate_docid("notes/a.md"),
                None,
                None,
            )
            .unwrap();
        let file_b = store
            .insert_file(
                "notes/b.md",
                "hb",
                100,
                &generate_docid("notes/b.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_chunk_with_vector(
                &NewChunk {
                    file_id: file_a,
                    seq: 0,
                    text: "a storm wolf howls at the storm",
                    vector_id: 1,
                    token_count: 7,
                    ..Default::default()
                },
                &[0.1, 0.2, 0.3, 0.4],
            )
            .unwrap();
        store
            .insert_chunk_with_vector(
                &NewChunk {
                    file_id: file_a,
                    seq: 1,
                    text: "a basilisk turns prey to stone",
                    vector_id: 2,
                    token_count: 6,
                    ..Default::default()
                },
                &[0.5, 0.6, 0.7, 0.8],
            )
            .unwrap();
        store
            .insert_chunk_with_vector(
                &NewChunk {
                    file_id: file_b,
                    seq: 0,
                    text: "a storm wolf pack hunts by night",
                    vector_id: 3,
                    token_count: 7,
                    ..Default::default()
                },
                &[0.9, 1.0, 1.1, 1.2],
            )
            .unwrap();

        (file_a, file_b)
    }

    #[test]
    fn chunk_row_count_counts_the_chunks_table() {
        let store = Store::open_memory().unwrap();
        seed_calibrated_fixture(&store);
        assert_eq!(store.chunk_row_count().unwrap(), 3);
    }

    #[test]
    fn fts_doc_frequency_counts_rows_matching_one_term() {
        let store = Store::open_memory().unwrap();
        seed_calibrated_fixture(&store);
        assert_eq!(
            store
                .fts_doc_frequency(&crate::fts::phrase_expr("storm"))
                .unwrap(),
            2
        );
        assert_eq!(
            store
                .fts_doc_frequency(&crate::fts::phrase_expr("basilisk"))
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .fts_doc_frequency(&crate::fts::phrase_expr("absent"))
                .unwrap(),
            0
        );
    }

    #[test]
    fn vectors_for_files_returns_the_named_files_vectors_and_no_others() {
        let store = Store::open_memory().unwrap();
        let (file_a, _file_b) = seed_calibrated_fixture(&store);

        let rows = store.vectors_for_files(&[file_a]).unwrap();
        assert!(rows.iter().all(|(f, _, _)| *f == file_a));
        assert!(!rows.is_empty());
        let (_, _, v) = &rows[0];
        assert_eq!(v.len(), 4, "decoded to the width it was written at");
        assert!(store.vectors_for_files(&[]).unwrap().is_empty());
    }
}
