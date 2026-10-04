//! The `chunks_vec` table and its width.

use super::Store;
use crate::fault::Fault;
use anyhow::{Result, bail};
use rusqlite::OptionalExtension;

impl Store {
    /// The embedding width this database was built at, if it has been indexed.
    ///
    /// Three sources, in order of directness: the dimension recorded at the last
    /// index, the width `chunks_vec` was declared with, and the length of a
    /// stored `chunks.vector` BLOB. The last two cover databases written by
    /// versions that predate the meta key or the vec0 table respectively.
    pub(super) fn recorded_embedding_dim(&self) -> Result<Option<usize>> {
        if let Some(dim) = self
            .get_meta("embedding_dim")?
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|d| *d > 0)
        {
            return Ok(Some(dim));
        }
        if let Some(dim) = self.vec_table_dim()? {
            return Ok(Some(dim));
        }
        // A BLOB is a packed `[f32]`, so its length divided by four is the width.
        let blob_len: Option<i64> = self
            .conn
            .query_row(
                "SELECT length(vector) FROM chunks WHERE vector IS NOT NULL LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(blob_len
            .map(|n| n as usize / std::mem::size_of::<f32>())
            .filter(|d| *d > 0))
    }

    /// The dimensionality `chunks_vec` was declared with, or `None` if the
    /// table does not exist.
    pub fn vec_table_dim(&self) -> Result<Option<usize>> {
        let sql: Option<String> = self
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name = 'chunks_vec'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        // The declaration is `embedding float[N] distance_metric=cosine`.
        Ok(sql.and_then(|s| {
            let start = s.find("float[")? + "float[".len();
            let rest = &s[start..];
            let end = rest.find(']')?;
            rest[..end].trim().parse::<usize>().ok()
        }))
    }

    /// Copy the vectors a store written before `chunks_vec` existed holds in
    /// `chunks.vector`, sizing the table from them when it is absent. Run by
    /// `migrate_to_2` and frozen with it.
    pub(super) fn copy_blob_vectors_into_vec0(&self) -> Result<()> {
        let blob_count: i64 = self.conn.query_row(
            "SELECT count(*) FROM chunks WHERE vector IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        if blob_count == 0 {
            return Ok(());
        }
        if self.vec_table_dim()?.is_none() {
            let Some(dim) = self.recorded_embedding_dim()? else {
                return Ok(());
            };
            crate::vecstore::init_vec_table(&self.conn, dim)?;
        }
        let vec_count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks_vec", [], |row| row.get(0))?;
        if vec_count > 0 {
            return Ok(());
        }
        tracing::info!(blob_count, "copying BLOB vectors into chunks_vec");
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id, vector FROM chunks WHERE vector IS NOT NULL")?;
        let rows: Vec<(i64, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (vid, blob) in &rows {
            self.conn.execute(
                "INSERT OR IGNORE INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![vid, blob],
            )?;
        }
        Ok(())
    }

    /// Store a vector, sizing the table to it on a database that has never
    /// been indexed.
    ///
    /// The first vector written decides the width, because it is the first
    /// evidence of what the embedding model produces — `Store::init` has none
    /// and must not guess (issue #12). A width *change* is a different matter
    /// and goes through [`Store::ensure_embedding_dim`], which discards the
    /// index rather than mixing two shapes.
    pub fn insert_vec(&self, vector_id: u64, embedding: &[f32]) -> Result<()> {
        if self.vec_table_dim()?.is_none() {
            crate::vecstore::init_vec_table(&self.conn, embedding.len())?;
            self.set_meta("embedding_dim", &embedding.len().to_string())?;
        }
        crate::vecstore::insert_vec(&self.conn, vector_id, embedding)
    }

    pub fn delete_vec(&self, vector_id: u64) -> Result<()> {
        if self.vec_table_dim()?.is_none() {
            return Ok(());
        }
        crate::vecstore::delete_vec(&self.conn, vector_id)
    }

    pub fn search_vec(
        &self,
        query: &[f32],
        k: usize,
        scope: Option<&[i64]>,
    ) -> Result<Vec<(u64, f32)>> {
        // A database that has never been indexed has no vec table at all, and
        // an empty semantic lane is the honest answer there.
        if self.vec_table_dim()?.is_none() {
            return Ok(Vec::new());
        }
        crate::vecstore::search_vec(&self.conn, query, k, scope)
    }

    pub fn clear_vec(&self) -> Result<()> {
        if self.vec_table_dim()?.is_none() {
            return Ok(());
        }
        crate::vecstore::clear_vec(&self.conn)
    }

    /// Bring vector storage into line with the embedding model's width.
    ///
    /// Creates `chunks_vec` at `model_dim` if the database has never been
    /// indexed, and rebuilds it if the model's width has changed since. Records
    /// `model_dim` in meta either way, so the single dimension decided when the
    /// model was chosen is the one the whole pipeline uses (issue #12).
    ///
    /// Returns the width the database previously held, and only then — that
    /// case discards every chunk, so the caller must force a full rebuild. A
    /// fresh database returns `None`: nothing was thrown away.
    pub fn ensure_embedding_dim(&self, model_dim: usize) -> Result<Option<usize>> {
        let previous = self.vec_table_dim()?;
        let outcome = match previous {
            Some(dim) if dim == model_dim => None,
            Some(dim) => {
                self.reset_for_reindex(model_dim)?;
                Some(dim)
            }
            None => {
                crate::vecstore::init_vec_table(&self.conn, model_dim)?;
                None
            }
        };
        self.set_meta("embedding_dim", &model_dim.to_string())?;
        Ok(outcome)
    }

    /// Fail if the index was built at a different width than `model_dim`.
    ///
    /// For read and write paths that do not reindex. Searching a 256-wide table
    /// with a 768-wide query vector is not a soft failure, so say what to do
    /// about it rather than letting sqlite-vec raise a shape error.
    pub fn verify_embedding_dim(&self, model_dim: usize) -> Result<()> {
        if let Some(dim) = self.vec_table_dim()?
            && dim != model_dim
        {
            bail!(Fault::StaleIndex(format!(
                "index was built with {dim}-dimensional embeddings but the model \
                 produces {model_dim}. Run 'knapper index' to rebuild it."
            )));
        }
        Ok(())
    }

    /// Drop the vec table and all chunk/FTS records. Used during dimension migration.
    pub fn reset_for_reindex(&self, new_dim: usize) -> Result<()> {
        self.conn.execute("DROP TABLE IF EXISTS chunks_vec", [])?;
        crate::vecstore::init_vec_table(&self.conn, new_dim)?;
        // The keyword index follows the chunks (issue #37).
        self.conn.execute("DELETE FROM chunks", [])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::*;

    #[test]
    fn test_store_has_vec_table() {
        let store = Store::open_memory().unwrap();
        // The table appears once something establishes its width — not before,
        // because `init` would have to guess one (issue #12).
        store.insert_vec(0, &[0.5_f32; 256]).unwrap();
        let count: i64 = store
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(store.vec_table_dim().unwrap(), Some(256));
    }

    #[test]
    fn test_store_vec_roundtrip() {
        let store = Store::open_memory().unwrap();
        let vector: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        store.insert_vec(0, &vector).unwrap();

        let results = store.search_vec(&vector, 1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01);
    }

    /// A store written before `chunks_vec` existed holds its vectors in
    /// `chunks.vector` alone. Step 2 sizes the table from them and copies
    /// them in, once.
    #[test]
    fn a_version_1_store_with_blob_vectors_only_gets_them_copied_into_vec0() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = tmp.path().join("knapper.db");
        let vector: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        {
            let store = Store::open(&db).unwrap();
            let file_id = store
                .insert_file("test.md", "hash123", 0, "abc123", None, None)
                .unwrap();
            // `insert_chunk_with_vector` writes the BLOB and nothing else, so
            // this store has no `chunks_vec` at all.
            store
                .insert_chunk_with_vector(
                    &NewChunk {
                        file_id,
                        seq: 0,
                        heading: "heading",
                        text: "snippet",
                        vector_id: 0,
                        token_count: 100,
                        ..Default::default()
                    },
                    &vector,
                )
                .unwrap();
            assert_eq!(store.vec_table_dim().unwrap(), None);
            store.set_user_version(1).unwrap();
        }
        let store = Store::open(&db).unwrap();
        assert_eq!(store.vec_table_dim().unwrap(), Some(256));
        let results = store.search_vec(&vector, 1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_embedding_dim_meta() {
        let store = Store::open_memory().unwrap();
        assert!(store.get_meta("embedding_dim").unwrap().is_none());
        store.set_meta("embedding_dim", "256").unwrap();
        assert_eq!(
            store.get_meta("embedding_dim").unwrap(),
            Some("256".to_string())
        );
    }

    #[test]
    fn fresh_database_has_no_vec_table_until_a_model_sizes_it() {
        // `Store::init` cannot know the embedding width — no model is loaded —
        // so it must not invent one (issue #12).
        let store = Store::open_memory().unwrap();
        assert_eq!(store.vec_table_dim().unwrap(), None);
        assert!(store.get_meta("embedding_dim").unwrap().is_none());

        assert_eq!(store.ensure_embedding_dim(768).unwrap(), None);
        assert_eq!(store.vec_table_dim().unwrap(), Some(768));
        assert_eq!(
            store.get_meta("embedding_dim").unwrap(),
            Some("768".to_string())
        );
    }

    #[test]
    fn searching_a_never_indexed_database_returns_nothing() {
        let store = Store::open_memory().unwrap();
        let hits = store.search_vec(&[0.1_f32; 768], 5, None).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn ensure_embedding_dim_is_idempotent_at_the_same_width() {
        let store = Store::open_memory().unwrap();
        store.ensure_embedding_dim(768).unwrap();
        assert_eq!(store.ensure_embedding_dim(768).unwrap(), None);
        assert_eq!(store.vec_table_dim().unwrap(), Some(768));
    }

    #[test]
    fn ensure_embedding_dim_rebuilds_storage_when_the_width_increases() {
        // The transition that ships with issue #12: an existing 256-wide index
        // meeting a model that produces 768. Nothing about it may be silent.
        let store = Store::open_memory().unwrap();
        store.ensure_embedding_dim(256).unwrap();
        let file_id = store
            .insert_file("note.md", "hash", 100, "abc123", None, None)
            .unwrap();
        let vid = store.next_vector_id().unwrap();
        store
            .insert_chunk_with_vector(
                &NewChunk {
                    file_id,
                    seq: 0,
                    heading: "H",
                    text: "snippet",
                    vector_id: vid,
                    token_count: 10,
                    ..Default::default()
                },
                &[0.1_f32; 256],
            )
            .unwrap();
        store.insert_vec(vid, &[0.1_f32; 256]).unwrap();

        assert_eq!(store.ensure_embedding_dim(768).unwrap(), Some(256));

        // The table is the new width, and every chunk indexed at the old one is
        // gone — which is why the caller must force a full rebuild.
        assert_eq!(store.vec_table_dim().unwrap(), Some(768));
        assert_eq!(
            store.get_meta("embedding_dim").unwrap(),
            Some("768".to_string())
        );
        assert!(store.get_chunks_by_file(file_id).unwrap().is_empty());
        assert!(store.fts_search("chunk text", 10).unwrap().is_empty());
        // A 768-wide vector now stores without a shape error.
        store.insert_vec(vid, &[0.1_f32; 768]).unwrap();
    }

    #[test]
    fn ensure_embedding_dim_rebuilds_storage_when_the_width_decreases() {
        let store = Store::open_memory().unwrap();
        store.ensure_embedding_dim(768).unwrap();
        assert_eq!(store.ensure_embedding_dim(256).unwrap(), Some(768));
        assert_eq!(store.vec_table_dim().unwrap(), Some(256));
    }

    #[test]
    fn verify_embedding_dim_rejects_a_model_that_disagrees_with_the_index() {
        let store = Store::open_memory().unwrap();
        store.ensure_embedding_dim(256).unwrap();

        assert!(store.verify_embedding_dim(256).is_ok());
        let err = store.verify_embedding_dim(768).unwrap_err();
        assert_eq!(
            crate::fault::Fault::of(&err).map(|f| f.kind()),
            Some("stale_index")
        );
        let err = err.to_string();
        assert!(err.contains("256"), "{err}");
        assert!(err.contains("768"), "{err}");
        assert!(err.contains("knapper index"), "{err}");
    }

    #[test]
    fn verify_embedding_dim_accepts_a_never_indexed_database() {
        // Nothing to disagree with yet — the first index will size it.
        let store = Store::open_memory().unwrap();
        assert!(store.verify_embedding_dim(768).is_ok());
    }

    #[test]
    fn reopening_a_database_recovers_its_width_without_guessing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        {
            let store = Store::open(&path).unwrap();
            store.ensure_embedding_dim(768).unwrap();
        }
        let store = Store::open(&path).unwrap();
        assert_eq!(store.vec_table_dim().unwrap(), Some(768));
    }
}
