use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashSet;
use std::path::Path;

/// A record representing an indexed file.
#[derive(Debug, Clone)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub content_hash: String,
    pub mtime: i64,
    pub tags: Vec<String>,
    pub indexed_at: String,
    pub docid: Option<String>,
    pub created_by: Option<String>,
    pub note_date: Option<i64>,
}

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
    pub vector_id: u64,
    pub token_count: i64,
}

/// Columns selected for every [`ChunkRecord`], in the order [`chunk_from_row`] expects.
const CHUNK_COLUMNS: &str = "id, file_id, seq, heading, snippet, text, vector_id, token_count";

/// Build a [`ChunkRecord`] from a row selecting [`CHUNK_COLUMNS`].
fn chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRecord> {
    Ok(ChunkRecord {
        id: row.get(0)?,
        file_id: row.get(1)?,
        seq: row.get(2)?,
        heading: row.get(3)?,
        snippet: row.get(4)?,
        text: row.get(5)?,
        vector_id: row.get::<_, i64>(6)? as u64,
        token_count: row.get(7)?,
    })
}

/// A single result from an FTS5 full-text search.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub file_id: i64,
    pub chunk_seq: i64,
    pub score: f64,
    pub snippet: String,
}

/// Statistics about edges in the graph.
#[derive(Debug)]
pub struct EdgeStats {
    pub total_edges: usize,
    pub wikilink_count: usize,
    pub mention_count: usize,
    pub connected_file_count: usize,
    pub isolated_file_count: usize,
}

/// A record of a PARA migration operation (batch file moves).
#[derive(Debug, Clone)]
pub struct MigrationEntry {
    pub id: i64,
    pub migration_id: String,
    pub old_path: String,
    pub new_path: String,
    pub category: String,
    pub confidence: f64,
    pub migrated_at: String,
}

/// A record representing a CLI event (for observability/analytics).
#[derive(Debug, Clone)]
pub struct CliEvent {
    pub id: i64,
    pub timestamp: String,
    pub operation: String,
    pub outcome: String,
    pub detail: Option<String>,
}

/// A record of a placement correction (user moved a note from suggested folder).
#[derive(Debug, Clone)]
pub struct PlacementCorrection {
    pub id: i64,
    pub file_path: String,
    pub suggested_folder: String,
    pub actual_folder: String,
    pub corrected_at: String,
}

/// A fact about the user's identity, inferred or stated (v1.6).
#[derive(Debug, Clone, serde::Serialize)]
pub struct IdentityFact {
    pub id: i64,
    pub tier: i64,
    pub key: String,
    pub value: String,
    pub source: Option<String>,
    pub updated_at: String,
}

/// Summary statistics for the store.
#[derive(Debug)]
pub struct StoreStats {
    pub file_count: usize,
    pub chunk_count: usize,
    pub tombstone_count: usize,
    pub last_indexed_at: Option<String>,
    pub vault_path: Option<String>,
    pub edge_count: Option<usize>,
    pub wikilink_count: Option<usize>,
    pub mention_count: Option<usize>,
}

const SCHEMA: &str = r#"
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    mtime        INTEGER NOT NULL,
    tags         TEXT NOT NULL DEFAULT '[]',
    indexed_at   TEXT NOT NULL,
    docid        TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL DEFAULT 0,
    heading     TEXT NOT NULL,
    snippet     TEXT NOT NULL,
    -- The whole chunk. Added by issue #14: the reranker has to read what it
    -- scores, and `chunks_fts` — the only other copy — cannot be keyed into
    -- without a MATCH, because its `file_id`/`chunk_seq` are UNINDEXED.
    text        TEXT NOT NULL DEFAULT '',
    vector_id   INTEGER UNIQUE NOT NULL,
    token_count INTEGER NOT NULL,
    vector      BLOB
);
-- idx_chunks_file_seq is created in `migrate`, not here: on a database written
-- before `seq` existed the CREATE TABLE above is a no-op, and indexing a column
-- the table does not have yet fails the whole schema batch.

CREATE TABLE IF NOT EXISTS tombstones (
    id         INTEGER PRIMARY KEY,
    vector_id  INTEGER UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    query_hash TEXT PRIMARY KEY,
    result     TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"#;

pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open a store backed by a file on disk.
    pub fn open(path: &Path) -> Result<Self> {
        crate::vecstore::init_sqlite_vec();
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_memory() -> Result<Self> {
        crate::vecstore::init_sqlite_vec();
        let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        // Enable WAL mode for concurrent reads during writes (fixes "database is locked"
        // errors with rapid MCP calls and parallel CLI + server access).
        // busy_timeout makes SQLite retry for up to 5 seconds instead of failing immediately.
        self.conn
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA busy_timeout = 5000;",
            )
            .context("failed to set WAL pragmas")?;
        self.conn
            .execute_batch(SCHEMA)
            .context("failed to initialize schema")?;
        self.migrate()?;
        self.ensure_fts_table()?;
        // The vector table's width is the embedding model's, and no model is
        // loaded here — so this must not guess (issue #12). A database that has
        // been indexed tells us its width; one that has not gets no vec table
        // until [`Store::ensure_embedding_dim`] reconciles it against the model.
        if let Some(dim) = self.recorded_embedding_dim()? {
            crate::vecstore::init_vec_table(&self.conn, dim)?;
            self.migrate_vectors_to_vec0()?;
        }
        Ok(())
    }

    /// The embedding width this database was built at, if it has been indexed.
    ///
    /// Three sources, in order of directness: the dimension recorded at the last
    /// index, the width `chunks_vec` was declared with, and the length of a
    /// stored `chunks.vector` BLOB. The last two cover databases written by
    /// versions that predate the meta key or the vec0 table respectively.
    fn recorded_embedding_dim(&self) -> Result<Option<usize>> {
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

    /// One-time migration: copy BLOB vectors from `chunks.vector` into the vec0 virtual table.
    /// Safe to call on every startup — skips if vec0 is already populated or no BLOBs exist.
    pub fn migrate_vectors_to_vec0(&self) -> Result<()> {
        // Nowhere to migrate to on a database that has never been indexed.
        if self.vec_table_dim()?.is_none() {
            return Ok(());
        }
        let vec_count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM chunks_vec", [], |row| row.get(0))
            .unwrap_or(0);
        let blob_count: i64 = self
            .conn
            .query_row(
                "SELECT count(*) FROM chunks WHERE vector IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if vec_count == 0 && blob_count > 0 {
            tracing::info!(blob_count, "migrating BLOB vectors to vec0");
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
            tracing::info!(migrated = rows.len(), "BLOB vector migration complete");
        }

        Ok(())
    }

    /// Whether `table` already has a column named `column`.
    fn column_exists(&self, table: &str, column: &str) -> Result<bool> {
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let mut rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        Ok(rows.any(|name| name.as_deref() == Ok(column)))
    }

    /// Copy each chunk's text out of the FTS index and into `chunks.text`.
    ///
    /// Runs once, when the column is added. `chunks_fts.file_id`/`chunk_seq` are
    /// UNINDEXED, so joining against them directly would rescan the FTS content
    /// for every chunk; the temp table exists to make that one scan instead of
    /// N. A chunk whose FTS row is missing keeps its snippet, which is a
    /// truncation of the right text rather than the wrong text.
    fn backfill_chunk_text(&self) -> Result<()> {
        let has_fts: bool = self
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'chunks_fts'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;
        if !has_fts {
            return Ok(());
        }
        tracing::info!("backfilling chunks.text from the FTS index");
        self.conn.execute_batch(
            "CREATE TEMP TABLE _fts_text AS
                 SELECT file_id, chunk_seq, content FROM chunks_fts;
             CREATE INDEX _fts_text_key ON _fts_text(file_id, chunk_seq);
             UPDATE chunks SET text = COALESCE(
                 (SELECT content FROM _fts_text t
                  WHERE t.file_id = chunks.file_id AND t.chunk_seq = chunks.seq),
                 snippet
             );
             DROP TABLE _fts_text;",
        )?;
        Ok(())
    }

    /// Run migrations for existing databases that may be missing newer columns.
    fn migrate(&self) -> Result<()> {
        if !self.column_exists("files", "docid")? {
            self.conn
                .execute_batch("ALTER TABLE files ADD COLUMN docid TEXT;")?;
        }
        // Always ensure the index exists (safe for both fresh and migrated DBs).
        self.conn
            .execute_batch("CREATE INDEX IF NOT EXISTS idx_files_docid ON files(docid);")?;

        // Add created_by column (idempotent — ignores error if column already exists).
        let _ = self
            .conn
            .execute_batch("ALTER TABLE files ADD COLUMN created_by TEXT;");

        // Add note_date column (idempotent — ignores error if column already exists).
        let _ = self
            .conn
            .execute_batch("ALTER TABLE files ADD COLUMN note_date INTEGER;");

        // Add chunks.seq, and backfill it for databases indexed before chunk
        // identity existed. Chunks were always inserted in document order, so the
        // ordinal of a chunk's rowid within its file is the seq the FTS index was
        // built with — that is what makes the two lanes joinable.
        if !self.column_exists("chunks", "seq")? {
            self.conn.execute_batch(
                "ALTER TABLE chunks ADD COLUMN seq INTEGER NOT NULL DEFAULT 0;
                 UPDATE chunks SET seq = (
                     SELECT COUNT(*) FROM chunks older
                     WHERE older.file_id = chunks.file_id AND older.id < chunks.id
                 );",
            )?;
        }
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_chunks_file_seq ON chunks(file_id, seq);",
        )?;

        // Add chunks.text, and backfill it from the FTS copy for databases
        // indexed before the column existed (issue #14).
        if !self.column_exists("chunks", "text")? {
            self.conn
                .execute_batch("ALTER TABLE chunks ADD COLUMN text TEXT NOT NULL DEFAULT '';")?;
            self.backfill_chunk_text()?;
        }

        // Check if edges table exists.
        let has_edges: bool = {
            let mut stmt = self
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")?;
            let mut rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.next().is_some()
        };
        if !has_edges {
            self.conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS edges (
                    id         INTEGER PRIMARY KEY,
                    from_file  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    to_file    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    edge_type  TEXT NOT NULL,
                    UNIQUE(from_file, to_file, edge_type)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_file);
                CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_file);
                CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);",
            )?;
        }

        // Folder centroids table
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS folder_centroids (
                folder     TEXT PRIMARY KEY,
                centroid   BLOB NOT NULL,
                file_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )?;

        // Tag registry table
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tag_registry (
                name        TEXT PRIMARY KEY,
                usage_count INTEGER NOT NULL DEFAULT 0,
                last_used   TEXT,
                created_by  TEXT NOT NULL DEFAULT 'indexer'
            );",
        )?;

        // Placement corrections table
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS placement_corrections (
                id              INTEGER PRIMARY KEY,
                file_path       TEXT NOT NULL,
                suggested_folder TEXT NOT NULL,
                actual_folder   TEXT NOT NULL,
                corrected_at    TEXT NOT NULL
            );",
        )?;

        // Link skiplist table (reserved for future use)
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS link_skiplist (
                id INTEGER PRIMARY KEY,
                pattern TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );",
        )?;

        // CLI events table (observability/analytics)
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cli_events (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                operation TEXT NOT NULL,
                outcome TEXT NOT NULL,
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cli_events_ts ON cli_events(timestamp);",
        )?;

        // Unresolved links table — tracks wikilink targets that couldn't be
        // resolved to a file during indexing. Used by health analysis.
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS unresolved_links (
                id          INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL,
                target      TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(source_file, target)
            );
            CREATE INDEX IF NOT EXISTS idx_unresolved_source ON unresolved_links(source_file);",
        )?;

        // Migration log table — records PARA migration batch operations.
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS migration_log (
                id           INTEGER PRIMARY KEY,
                migration_id TEXT NOT NULL,
                old_path     TEXT NOT NULL,
                new_path     TEXT NOT NULL,
                category     TEXT NOT NULL,
                confidence   REAL NOT NULL,
                migrated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_migration_id ON migration_log(migration_id);",
        )?;

        // Identity facts table (v1.6)
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS identity_facts (
                id         INTEGER PRIMARY KEY,
                tier       INTEGER NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                source     TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(tier, key, value)
            );",
        )?;

        Ok(())
    }

    // ── Meta ────────────────────────────────────────────────────

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query_map(params![key], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(val) => Ok(Some(val?)),
            None => Ok(None),
        }
    }

    // ── LLM Cache ───────────────────────────────────────────────

    /// Cache an LLM orchestration result by query hash.
    pub fn set_llm_cache(&self, query_hash: &str, result: &str, model: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache (query_hash, result, model, created_at)
             VALUES (?1, ?2, ?3, datetime('now'))",
            params![query_hash, result, model],
        )?;
        Ok(())
    }

    /// Retrieve a cached LLM result by query hash.
    pub fn get_llm_cache(&self, query_hash: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT result FROM llm_cache WHERE query_hash = ?1")?;
        let result = stmt
            .query_row(params![query_hash], |row| row.get::<_, String>(0))
            .optional()?;
        Ok(result)
    }

    // ── Files ───────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn insert_file(
        &self,
        path: &str,
        hash: &str,
        mtime: i64,
        tags: &[String],
        docid: &str,
        created_by: Option<&str>,
        note_date: Option<i64>,
    ) -> Result<i64> {
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let now = chrono_now();
        self.conn.execute(
            "INSERT INTO files (path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(path) DO UPDATE SET
                content_hash = excluded.content_hash,
                mtime        = excluded.mtime,
                tags         = excluded.tags,
                indexed_at   = excluded.indexed_at,
                docid        = excluded.docid,
                created_by   = excluded.created_by,
                note_date    = excluded.note_date",
            params![path, hash, mtime, tags_json, now, docid, created_by, note_date],
        )?;
        let file_id: i64 = self.conn.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![path],
            |row| row.get(0),
        )?;
        Ok(file_id)
    }

    pub fn get_file(&self, path: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date FROM files WHERE path = ?1",
        )?;
        let mut rows = stmt.query_map(params![path], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    pub fn get_all_files(&self) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date FROM files",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        let mut files = Vec::new();
        for row in rows {
            files.push(row?);
        }
        Ok(files)
    }

    /// Delete a file's row.
    ///
    /// `chunks` and `edges` both reference `files(id)` `ON DELETE CASCADE`, so
    /// this takes the file's chunks *and every edge touching it in either
    /// direction* with it — including edges other files own. That is right when
    /// the file is going away and wrong when it is being re-indexed, which is
    /// why `index_file` uses [`delete_chunks_for_file`](Self::delete_chunks_for_file)
    /// and lets `insert_file`'s upsert keep the row (issue #27).
    pub fn delete_file(&self, file_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        Ok(())
    }

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

    // ── Chunks ──────────────────────────────────────────────────

    /// Insert a chunk. `seq` is its 0-based position in the file and must match
    /// the `chunk_seq` given to [`insert_fts_chunk`](Self::insert_fts_chunk) for
    /// the same chunk, or the two lanes will disagree about what they retrieved.
    ///
    /// `text` is the **whole chunk**. The `snippet` column is derived from it
    /// here rather than passed in: a chunk row that holds a preview but not the
    /// text it previews is the state issue #14 exists to remove, and taking one
    /// argument makes it unreachable.
    pub fn insert_chunk(
        &self,
        file_id: i64,
        seq: i64,
        heading: &str,
        text: &str,
        vector_id: u64,
        token_count: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chunks (file_id, seq, heading, snippet, text, vector_id, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                file_id,
                seq,
                heading,
                crate::chunker::make_snippet(text),
                text,
                vector_id as i64,
                token_count
            ],
        )?;
        Ok(())
    }

    /// Insert a chunk with its embedding vector stored as a BLOB.
    ///
    /// `text` is the whole chunk, for the reason given on
    /// [`insert_chunk`](Self::insert_chunk).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_chunk_with_vector(
        &self,
        file_id: i64,
        seq: i64,
        heading: &str,
        text: &str,
        vector_id: u64,
        token_count: i64,
        vector: &[f32],
    ) -> Result<()> {
        let vector_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO chunks (file_id, seq, heading, snippet, text, vector_id, token_count, vector)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                file_id,
                seq,
                heading,
                crate::chunker::make_snippet(text),
                text,
                vector_id as i64,
                token_count,
                vector_bytes
            ],
        )?;
        Ok(())
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

    // ── Tombstones ──────────────────────────────────────────────

    pub fn add_tombstones(&self, vector_ids: &[u64]) -> Result<()> {
        let now = chrono_now();
        let mut stmt = self
            .conn
            .prepare("INSERT OR IGNORE INTO tombstones (vector_id, created_at) VALUES (?1, ?2)")?;
        for &vid in vector_ids {
            stmt.execute(params![vid as i64, now])?;
        }
        Ok(())
    }

    pub fn get_tombstones(&self) -> Result<HashSet<u64>> {
        let mut stmt = self.conn.prepare("SELECT vector_id FROM tombstones")?;
        let rows = stmt.query_map([], |row| Ok(row.get::<_, i64>(0)? as u64))?;
        let mut set = HashSet::new();
        for row in rows {
            set.insert(row?);
        }
        Ok(set)
    }

    pub fn tombstone_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM tombstones", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn clear_tombstones(&self) -> Result<()> {
        self.conn.execute("DELETE FROM tombstones", [])?;
        Ok(())
    }

    // ── Edges ──────────────────────────────────────────────────

    /// Insert an edge. Uses INSERT OR IGNORE for the UNIQUE constraint.
    pub fn insert_edge(&self, from_file: i64, to_file: i64, edge_type: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (from_file, to_file, edge_type) VALUES (?1, ?2, ?3)",
            params![from_file, to_file, edge_type],
        )?;
        Ok(())
    }

    /// Delete all edges involving a file (both directions: from_file OR to_file).
    ///
    /// Only correct when the file itself is going away. An edge is owned by its
    /// **source** file's content, so deleting the incoming half throws away
    /// other files' links and nothing re-creates them — those files are not
    /// being re-indexed. Re-index paths want
    /// [`delete_outgoing_edges_for_file`](Self::delete_outgoing_edges_for_file).
    pub fn delete_edges_for_file(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM edges WHERE from_file = ?1 OR to_file = ?1",
            params![file_id],
        )?;
        Ok(())
    }

    /// Delete the edges a file owns — the ones its own content created.
    ///
    /// The partner of `indexer::build_edges_for_file`: together they recompute
    /// exactly the set of edges this file is the author of, and leave every
    /// backlink into it alone (issue #27).
    pub fn delete_outgoing_edges_for_file(&self, file_id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM edges WHERE from_file = ?1", params![file_id])?;
        Ok(())
    }

    /// Clear all edges (used during --rebuild).
    pub fn clear_edges(&self) -> Result<()> {
        self.conn.execute("DELETE FROM edges", [])?;
        Ok(())
    }

    /// Get outgoing edges, optionally filtered by type.
    pub fn get_outgoing(
        &self,
        file_id: i64,
        edge_type: Option<&str>,
    ) -> Result<Vec<(i64, String)>> {
        let mut results = Vec::new();
        match edge_type {
            Some(et) => {
                let mut stmt = self.conn.prepare(
                    "SELECT to_file, edge_type FROM edges WHERE from_file = ?1 AND edge_type = ?2",
                )?;
                let rows = stmt.query_map(params![file_id, et], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
            None => {
                let mut stmt = self
                    .conn
                    .prepare("SELECT to_file, edge_type FROM edges WHERE from_file = ?1")?;
                let rows = stmt.query_map(params![file_id], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
        }
        Ok(results)
    }

    /// Get incoming edges, optionally filtered by type.
    pub fn get_incoming(
        &self,
        file_id: i64,
        edge_type: Option<&str>,
    ) -> Result<Vec<(i64, String)>> {
        let mut results = Vec::new();
        match edge_type {
            Some(et) => {
                let mut stmt = self.conn.prepare(
                    "SELECT from_file, edge_type FROM edges WHERE to_file = ?1 AND edge_type = ?2",
                )?;
                let rows = stmt.query_map(params![file_id, et], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
            None => {
                let mut stmt = self
                    .conn
                    .prepare("SELECT from_file, edge_type FROM edges WHERE to_file = ?1")?;
                let rows = stmt.query_map(params![file_id], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
        }
        Ok(results)
    }

    // ── Stats ───────────────────────────────────────────────────

    pub fn stats(&self) -> Result<StoreStats> {
        let file_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;
        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        let tombstone_count = self.tombstone_count()?;
        let last_indexed_at = self.get_meta("last_indexed_at")?;
        let vault_path = self.get_meta("vault_path")?;
        let (edge_count, wikilink_count, mention_count) = match self.get_edge_stats() {
            Ok(es) => (
                Some(es.total_edges),
                Some(es.wikilink_count),
                Some(es.mention_count),
            ),
            Err(_) => (None, None, None),
        };
        Ok(StoreStats {
            file_count: file_count as usize,
            chunk_count: chunk_count as usize,
            tombstone_count,
            last_indexed_at,
            vault_path,
            edge_count,
            wikilink_count,
            mention_count,
        })
    }

    /// Look up a file's path by its row ID.
    pub fn get_file_path_by_id(&self, file_id: i64) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files WHERE id = ?1")?;
        let mut rows = stmt.query_map(params![file_id], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(val) => Ok(Some(val?)),
            None => Ok(None),
        }
    }

    /// Look up a file record by its row ID.
    pub fn get_file_by_id(&self, file_id: i64) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date FROM files WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![file_id], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    /// Look up a file by its 6-character docid.
    pub fn get_file_by_docid(&self, docid: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date FROM files WHERE docid = ?1",
        )?;
        let mut rows = stmt.query_map(params![docid], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        match rows.next() {
            Some(rec) => Ok(Some(rec?)),
            None => Ok(None),
        }
    }

    // ── FTS5 ──────────────────────────────────────────────────

    /// Ensure the FTS5 virtual table exists. Called during init.
    pub fn ensure_fts_table(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                file_id UNINDEXED,
                chunk_seq UNINDEXED
            );",
            )
            .context("failed to create FTS5 virtual table")?;
        Ok(())
    }

    /// Insert a chunk's text into the FTS5 table.
    ///
    /// `text` is the **whole chunk**, not `chunks.snippet` — this table is the
    /// only place a chunk's full text is retained, so anything not passed here
    /// is unreachable by keyword search (issue #11). `chunks_fts` is a standalone
    /// FTS5 table rather than external-content, so it keeps its own copy.
    pub fn insert_fts_chunk(&self, file_id: i64, chunk_seq: i64, text: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chunks_fts (content, file_id, chunk_seq) VALUES (?1, ?2, ?3)",
            params![text, file_id, chunk_seq],
        )?;
        Ok(())
    }

    /// Delete all FTS5 entries for a file.
    pub fn delete_fts_chunks_for_file(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE file_id = ?1",
            params![file_id],
        )?;
        Ok(())
    }

    /// Search the FTS5 index. Returns results ranked by BM25 score.
    /// BM25 in SQLite returns negative values (more negative = better match),
    /// so we negate them to get positive scores where higher = better.
    ///
    /// The query is wrapped in double quotes so that FTS5 treats it as a
    /// phrase/literal rather than interpreting operators like `-`.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>> {
        self.fts_search_expr(&crate::fts::phrase_expr(query), limit)
    }

    /// Keyword search matching **any** token of `query`, each taken literally.
    ///
    /// What the search lane wants, and what [`Self::fts_search`] cannot give it:
    /// a phrase query only fires where the caller already guessed the corpus's
    /// wording. See [`crate::fts::any_term_expr`] for the measurements (#22).
    ///
    /// A query with no searchable token returns no rows rather than an error.
    pub fn fts_search_any(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>> {
        match crate::fts::any_term_expr(query) {
            Some(expr) => self.fts_search_expr(&expr, limit),
            None => Ok(Vec::new()),
        }
    }

    /// Run a prepared FTS5 MATCH expression. Callers build the expression with
    /// `crate::fts`, which is where the quoting rules and their reasons live.
    fn fts_search_expr(&self, fts_query: &str, limit: usize) -> Result<Vec<FtsResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT file_id, chunk_seq, bm25(chunks_fts) as score,
                    snippet(chunks_fts, 0, '<b>', '</b>', '...', 64)
             FROM chunks_fts
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
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

    /// Return vector_ids for all chunks belonging to a file.
    /// Useful for tombstoning before re-indexing a changed file.
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

    // ── Graph helpers ────────────────────────────────────────────

    /// Get neighbor file IDs within N hops via wikilinks, in either direction
    /// (outgoing links and backlinks). Uses Rust-side BFS, not recursive SQL CTE.
    pub fn get_neighbors(&self, file_id: i64, depth: usize) -> Result<Vec<(i64, usize)>> {
        use std::collections::VecDeque;
        let mut visited = HashSet::new();
        visited.insert(file_id);
        let mut queue = VecDeque::new();
        let mut results = Vec::new();
        queue.push_back((file_id, 0usize));
        while let Some((current, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }
            // Treat wikilinks as undirected for neighbor discovery: follow
            // links out of `current` and backlinks into it. Edges are stored
            // directionally (one per [[link]]), but a knowledge-graph neighbor
            // is related in either direction, so search and context expansion
            // should surface backlinks too.
            let outgoing = self.get_outgoing(current, Some("wikilink"))?;
            let incoming = self.get_incoming(current, Some("wikilink"))?;
            for (neighbor_id, _) in outgoing.into_iter().chain(incoming) {
                if visited.insert(neighbor_id) {
                    let hop = current_depth + 1;
                    results.push((neighbor_id, hop));
                    queue.push_back((neighbor_id, hop));
                }
            }
        }
        Ok(results)
    }

    /// Find files that share at least one tag with the given file.
    pub fn get_shared_tags_files(&self, file_id: i64, limit: usize) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT f2.id
             FROM files f1
             JOIN files f2 ON f2.id != f1.id
             WHERE f1.id = ?1
             AND EXISTS (
                 SELECT 1 FROM json_each(f1.tags) t1
                 JOIN json_each(f2.tags) t2 ON t1.value = t2.value
             )
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![file_id, limit as i64], |row| row.get::<_, i64>(0))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Check if a file's FTS5 content contains a term. Escapes for FTS5.
    pub fn file_contains_term(&self, file_id: i64, term: &str) -> Result<bool> {
        let escaped = term.replace('"', "\"\"");
        let query = format!("\"{}\"", escaped);
        let result: Result<i64, _> = self.conn.query_row(
            "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ?1 AND file_id = ?2 LIMIT 1",
            params![query, file_id],
            |row| row.get(0),
        );
        Ok(result.is_ok())
    }

    /// Which chunk of `file_id` best matches any of `terms`, by BM25.
    ///
    /// The graph lane ranks whole files but has to name a section; this is how it
    /// names the one that actually contains the query, rather than the longest.
    /// Returns `None` when no chunk of the file matches — which is also the
    /// relevance signal `file_contains_term` used to give.
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
            "SELECT chunk_seq FROM chunks_fts
             WHERE chunks_fts MATCH ?1 AND file_id = ?2
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

    /// Get the best (highest token_count) chunk for a file.
    pub fn get_best_chunk_for_file(&self, file_id: i64) -> Result<Option<ChunkRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE file_id = ?1 ORDER BY token_count DESC LIMIT 1"
        ))?;
        let mut rows = stmt.query_map(params![file_id], chunk_from_row)?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Get statistics about edges in the graph.
    pub fn get_edge_stats(&self) -> Result<EdgeStats> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        let wikilinks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'wikilink'",
            [],
            |r| r.get(0),
        )?;
        let mentions: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'mention'",
            [],
            |r| r.get(0),
        )?;
        let connected: i64 = self.conn.query_row(
            "SELECT COUNT(DISTINCT id) FROM files WHERE id IN \
             (SELECT from_file FROM edges UNION SELECT to_file FROM edges)",
            [],
            |r| r.get(0),
        )?;
        let total_files: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        Ok(EdgeStats {
            total_edges: total as usize,
            wikilink_count: wikilinks as usize,
            mention_count: mentions as usize,
            connected_file_count: connected as usize,
            isolated_file_count: (total_files - connected) as usize,
        })
    }

    /// List files filtered by folder prefix and/or tags (AND logic).
    pub fn list_files(
        &self,
        folder: Option<&str>,
        tags: &[String],
        created_by: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FileRecord>> {
        let mut sql = String::from(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date FROM files WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(f) = folder {
            sql.push_str(" AND path LIKE ?");
            param_values.push(Box::new(format!("{}%", f)));
        }
        for tag in tags {
            sql.push_str(" AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)");
            param_values.push(Box::new(tag.clone()));
        }
        if let Some(cb) = created_by {
            sql.push_str(" AND created_by = ?");
            param_values.push(Box::new(cb.to_string()));
        }
        sql.push_str(" ORDER BY indexed_at DESC LIMIT ?");
        param_values.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(param_values.iter()), |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Top-level folder grouping with note counts.
    pub fn folder_note_counts(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT CASE WHEN instr(path, '/') > 0
                    THEN substr(path, 1, instr(path, '/') - 1)
                    ELSE '(root)'
                    END AS folder,
                    COUNT(*) as cnt
             FROM files GROUP BY folder ORDER BY cnt DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Tag frequency aggregation via json_each.
    pub fn top_tags(&self, limit: usize) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT value, COUNT(*) as cnt
             FROM files, json_each(files.tags)
             GROUP BY value ORDER BY cnt DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Most recently indexed files.
    pub fn recent_files(&self, limit: usize) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date
             FROM files ORDER BY indexed_at DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Total edges (both directions) for a given file.
    pub fn edge_count_for_file(&self, file_id: i64) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE from_file = ?1 OR to_file = ?1",
            params![file_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get edge counts for multiple files in a single query.
    pub fn edge_counts_for_files(
        &self,
        file_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, usize>> {
        use std::collections::HashMap;
        if file_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: Vec<String> = file_ids.iter().map(|_| "?".to_string()).collect();
        let ph = placeholders.join(",");
        let sql = format!(
            "SELECT fid, COUNT(*) FROM (
                SELECT from_file AS fid FROM edges WHERE from_file IN ({ph})
                UNION ALL
                SELECT to_file AS fid FROM edges WHERE to_file IN ({ph})
            ) GROUP BY fid"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = file_ids
            .iter()
            .chain(file_ids.iter())
            .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut map = HashMap::new();
        for row in rows {
            let (id, count) = row?;
            map.insert(id, count);
        }
        Ok(map)
    }

    /// Find all files whose path matches a LIKE pattern (e.g., "03-Resources/People/%").
    pub fn find_files_by_prefix(&self, pattern: &str) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date
             FROM files WHERE path LIKE ?1",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| anyhow::anyhow!("find_files_by_prefix: {e}"))
    }

    /// Find a file by case-insensitive basename match. Returns first match (shortest path).
    pub fn find_file_by_basename(&self, basename: &str) -> Result<Option<FileRecord>> {
        let base = if basename.ends_with(".md") {
            basename.to_string()
        } else {
            format!("{basename}.md")
        };

        // Try exact path first.
        if let Some(f) = self.get_file(&base)? {
            return Ok(Some(f));
        }

        // Build candidate names: exact, spaces→hyphens, hyphens→spaces, spaces→underscores.
        let normalized = basename.replace(['-', '_'], " ");
        let hyphenated = basename.replace(' ', "-");
        let underscored = basename.replace(' ', "_");
        let mut candidates = vec![base];
        for v in [normalized, hyphenated, underscored] {
            let c = if v.ends_with(".md") {
                v
            } else {
                format!("{v}.md")
            };
            if !candidates.contains(&c) {
                candidates.push(c);
            }
        }

        // Try each candidate as a case-insensitive basename match.
        for candidate in &candidates {
            let mut stmt = self.conn.prepare(
                "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date
                 FROM files
                 WHERE lower(path) LIKE '%/' || lower(?1) OR lower(path) = lower(?1)
                 ORDER BY length(path) ASC LIMIT 1",
            )?;
            let mut rows = stmt.query_map(params![candidate], |row| {
                Ok(FileRecord {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    content_hash: row.get(2)?,
                    mtime: row.get(3)?,
                    tags: parse_tags(&row.get::<_, String>(4)?),
                    indexed_at: row.get(5)?,
                    docid: row.get(6)?,
                    created_by: row.get(7)?,
                    note_date: row.get(8)?,
                })
            })?;
            if let Some(row) = rows.next() {
                return Ok(Some(row?));
            }
        }

        Ok(None)
    }

    /// Query files whose note_date falls within a given range (inclusive).
    pub fn get_files_in_date_range(&self, start: i64, end: i64) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, content_hash, mtime, tags, indexed_at, docid, created_by, note_date
             FROM files WHERE note_date BETWEEN ?1 AND ?2
             ORDER BY note_date ASC",
        )?;
        let rows = stmt.query_map(params![start, end], |row| {
            Ok(FileRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                content_hash: row.get(2)?,
                mtime: row.get(3)?,
                tags: parse_tags(&row.get::<_, String>(4)?),
                indexed_at: row.get(5)?,
                docid: row.get(6)?,
                created_by: row.get(7)?,
                note_date: row.get(8)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Count files that have a non-NULL note_date.
    pub fn count_files_with_dates(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM files WHERE note_date IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Rename a file's path in the store, preserving its row ID (and thus edge integrity).
    pub fn update_file_path(&self, old_path: &str, new_path: &str, new_docid: &str) -> Result<()> {
        if self.get_file(new_path)?.is_some() {
            anyhow::bail!("target path already exists: {}", new_path);
        }
        let rows_affected = self.conn.execute(
            "UPDATE files SET path = ?1, docid = ?2 WHERE path = ?3",
            params![new_path, new_docid, old_path],
        )?;
        if rows_affected == 0 {
            anyhow::bail!("file not found: {}", old_path);
        }
        Ok(())
    }

    /// Update only the mtime (and optionally content_hash) for a file in the store.
    /// Used after write operations to keep the stored mtime in sync with disk.
    pub fn update_file_mtime(&self, path: &str, mtime: i64) -> Result<()> {
        let rows_affected = self.conn.execute(
            "UPDATE files SET mtime = ?1 WHERE path = ?2",
            params![mtime, path],
        )?;
        if rows_affected == 0 {
            anyhow::bail!("file not found in store: {}", path);
        }
        Ok(())
    }

    // ── Vec (sqlite-vec) ────────────────────────────────────────

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
        tombstones: &std::collections::HashSet<u64>,
    ) -> Result<Vec<(u64, f32)>> {
        // A database that has never been indexed has no vec table at all, and
        // an empty semantic lane is the honest answer there.
        if self.vec_table_dim()?.is_none() {
            return Ok(Vec::new());
        }
        crate::vecstore::search_vec(&self.conn, query, k, tombstones)
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
            bail!(
                "index was built with {dim}-dimensional embeddings but the model \
                 produces {model_dim}. Run 'engraph index' to rebuild it."
            );
        }
        Ok(())
    }

    /// Drop the vec table and all chunk/FTS records. Used during dimension migration.
    pub fn reset_for_reindex(&self, new_dim: usize) -> Result<()> {
        self.conn.execute("DROP TABLE IF EXISTS chunks_vec", [])?;
        crate::vecstore::init_vec_table(&self.conn, new_dim)?;
        self.conn.execute("DELETE FROM chunks", [])?;
        self.conn.execute("DELETE FROM chunks_fts", [])?;
        Ok(())
    }

    // ── Transactions ────────────────────────────────────────────

    pub fn begin_transaction(&self) -> Result<()> {
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        Ok(())
    }

    pub fn commit(&self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    pub fn rollback(&self) -> Result<()> {
        self.conn.execute_batch("ROLLBACK")?;
        Ok(())
    }

    // ── Folder centroids ─────────────────────────────────────────

    pub fn upsert_folder_centroid(
        &self,
        folder: &str,
        centroid: &[f32],
        file_count: usize,
    ) -> Result<()> {
        let blob: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO folder_centroids (folder, centroid, file_count, updated_at)
             VALUES (?1, ?2, ?3, datetime('now'))
             ON CONFLICT(folder) DO UPDATE SET centroid = ?2, file_count = ?3, updated_at = datetime('now')",
            params![folder, blob, file_count as i64],
        )?;
        Ok(())
    }

    pub fn get_folder_centroids(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT folder, centroid FROM folder_centroids")?;
        let rows = stmt.query_map([], |row| {
            let folder: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let centroid: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((folder, centroid))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get a single folder's centroid and file count.
    pub fn get_folder_centroid(&self, folder: &str) -> Result<Option<(Vec<f32>, usize)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT centroid, file_count FROM folder_centroids WHERE folder = ?1")?;
        let mut rows = stmt.query_map(params![folder], |row| {
            let blob: Vec<u8> = row.get(0)?;
            let count: i64 = row.get(1)?;
            let centroid: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((centroid, count as usize))
        })?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Incrementally adjust a folder centroid using online mean math.
    /// If `increment` is true, adds a file vector; if false, removes one.
    pub fn adjust_folder_centroid(
        &self,
        folder: &str,
        file_vec: &[f32],
        increment: bool,
    ) -> Result<()> {
        let existing = self.get_folder_centroid(folder)?;
        match (existing, increment) {
            (None, true) => {
                // New folder — centroid is just this vector
                self.upsert_folder_centroid(folder, file_vec, 1)?;
            }
            (None, false) => {
                // Nothing to remove from — no-op
            }
            (Some((old, n)), true) => {
                // online mean addition: new = (old * n + vec) / (n + 1)
                let nf = n as f32;
                let new_n = n + 1;
                let updated: Vec<f32> = old
                    .iter()
                    .zip(file_vec.iter())
                    .map(|(o, v)| (o * nf + v) / new_n as f32)
                    .collect();
                self.upsert_folder_centroid(folder, &updated, new_n)?;
            }
            (Some((_old, n)), false) if n <= 1 => {
                // Last file — delete centroid row
                self.conn.execute(
                    "DELETE FROM folder_centroids WHERE folder = ?1",
                    params![folder],
                )?;
            }
            (Some((old, n)), false) => {
                // online mean subtraction: new = (old * n - vec) / (n - 1)
                let nf = n as f32;
                let new_n = n - 1;
                let updated: Vec<f32> = old
                    .iter()
                    .zip(file_vec.iter())
                    .map(|(o, v)| (o * nf - v) / new_n as f32)
                    .collect();
                self.upsert_folder_centroid(folder, &updated, new_n)?;
            }
        }
        Ok(())
    }

    // ── Chunk vectors ──────────────────────────────────────────

    /// Retrieve all chunk vectors for a given file, ordered by chunk id.
    pub fn get_chunk_vectors_for_file(&self, file_id: i64) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT vector FROM chunks WHERE file_id = ?1 AND vector IS NOT NULL ORDER BY id",
        )?;
        let rows = stmt.query_map(params![file_id], |row| {
            let blob: Vec<u8> = row.get(0)?;
            let vector: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(vector)
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Placement corrections ────────────────────────────────────

    /// Record a placement correction (user moved a note from suggested folder).
    pub fn insert_placement_correction(
        &self,
        file_path: &str,
        suggested_folder: &str,
        actual_folder: &str,
    ) -> Result<()> {
        let dt = time::OffsetDateTime::now_utc();
        let now = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            dt.year(),
            dt.month() as u8,
            dt.day(),
            dt.hour(),
            dt.minute(),
            dt.second(),
        );
        self.conn.execute(
            "INSERT INTO placement_corrections (file_path, suggested_folder, actual_folder, corrected_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![file_path, suggested_folder, actual_folder, now],
        )?;
        Ok(())
    }

    /// Get recent placement corrections, latest first.
    pub fn get_placement_corrections(&self, limit: usize) -> Result<Vec<PlacementCorrection>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, suggested_folder, actual_folder, corrected_at
             FROM placement_corrections ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(PlacementCorrection {
                id: row.get(0)?,
                file_path: row.get(1)?,
                suggested_folder: row.get(2)?,
                actual_folder: row.get(3)?,
                corrected_at: row.get(4)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Migration Log ────────────────────────────────────────────

    /// Record a single file move as part of a named migration batch.
    pub fn log_migration(
        &self,
        migration_id: &str,
        old_path: &str,
        new_path: &str,
        category: &str,
        confidence: f64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO migration_log (migration_id, old_path, new_path, category, confidence)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![migration_id, old_path, new_path, category, confidence],
        )?;
        Ok(())
    }

    /// Retrieve all entries for a migration, ordered by insertion order.
    pub fn get_migration(&self, migration_id: &str) -> Result<Vec<MigrationEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, migration_id, old_path, new_path, category, confidence, migrated_at
             FROM migration_log WHERE migration_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![migration_id], |row| {
            Ok(MigrationEntry {
                id: row.get(0)?,
                migration_id: row.get(1)?,
                old_path: row.get(2)?,
                new_path: row.get(3)?,
                category: row.get(4)?,
                confidence: row.get(5)?,
                migrated_at: row.get(6)?,
            })
        })?;
        let results: Result<Vec<_>, _> = rows.collect();
        Ok(results?)
    }

    /// Return the migration_id of the most recently created migration, if any.
    pub fn get_last_migration_id(&self) -> Result<Option<String>> {
        let result = self
            .conn
            .query_row(
                "SELECT migration_id FROM migration_log ORDER BY migrated_at DESC, id DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(result)
    }

    /// Delete all entries for a migration (for undo / rollback support).
    pub fn delete_migration(&self, migration_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM migration_log WHERE migration_id = ?1",
            params![migration_id],
        )?;
        Ok(())
    }

    // ── Identity Facts ───────────────────────────────────────────

    pub fn upsert_identity_fact(
        &self,
        tier: i64,
        key: &str,
        value: &str,
        source: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO identity_facts (tier, key, value, source, updated_at)
             VALUES (?1, ?2, ?3, ?4, datetime('now'))
             ON CONFLICT(tier, key, value) DO UPDATE SET
               source = excluded.source,
               updated_at = datetime('now')",
            rusqlite::params![tier, key, value, source],
        )?;
        Ok(())
    }

    pub fn get_identity_facts(&self, tier: i64) -> Result<Vec<IdentityFact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tier, key, value, source, updated_at
             FROM identity_facts WHERE tier = ?1 ORDER BY key, value",
        )?;
        let rows = stmt.query_map(rusqlite::params![tier], |row| {
            Ok(IdentityFact {
                id: row.get(0)?,
                tier: row.get(1)?,
                key: row.get(2)?,
                value: row.get(3)?,
                source: row.get(4)?,
                updated_at: row.get(5)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn clear_identity_facts(&self, tier: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM identity_facts WHERE tier = ?1",
            rusqlite::params![tier],
        )?;
        Ok(())
    }

    // ── Helpers ─────────────────────────────────────────────────

    pub fn next_vector_id(&self) -> Result<u64> {
        let max: Option<i64> = self
            .conn
            .query_row("SELECT MAX(vector_id) FROM chunks", [], |row| row.get(0))
            .ok()
            .flatten();
        Ok(max.map_or(0, |m| m as u64 + 1))
    }

    // ── Tags ────────────────────────────────────────────────────

    /// Tags created by agents (not by indexer).
    pub fn agent_created_tags(&self) -> Result<Vec<(String, String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, created_by, usage_count FROM tag_registry WHERE created_by != 'indexer' ORDER BY usage_count DESC",
        )?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.into())
    }

    /// Tags used fewer than N times (cleanup candidates).
    pub fn low_usage_tags(&self, max_count: i64) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, usage_count FROM tag_registry WHERE usage_count < ?1 ORDER BY usage_count",
        )?;
        let rows = stmt.query_map(params![max_count], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.into())
    }

    /// Tags unused for more than N days.
    pub fn stale_tags(&self, days: i64) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, last_used FROM tag_registry WHERE last_used IS NOT NULL AND julianday('now') - julianday(last_used) > ?1 ORDER BY last_used",
        )?;
        let rows = stmt.query_map(params![days], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.into())
    }

    /// Borrow the underlying connection (for modules that need direct access).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Resolve a file reference (path, basename, or #docid) to a FileRecord.
    ///
    /// Resolution order:
    /// 1. `#docid` — 6-char hex prefixed with `#`
    /// 2. Exact path match
    /// 3. Basename match (case-insensitive, with separator normalization)
    /// 4. Fuzzy match — Levenshtein distance ≤ 2 on basenames (stripped of `.md`)
    ///    - If exactly one candidate: return it
    ///    - If multiple equidistant candidates: error with candidate list
    ///    - If none within threshold: return None
    pub fn resolve_file(&self, file_or_docid: &str) -> Result<Option<FileRecord>> {
        if file_or_docid.starts_with('#') && file_or_docid.len() == 7 {
            return self.get_file_by_docid(&file_or_docid[1..]);
        }
        if let Some(f) = self.get_file(file_or_docid)? {
            return Ok(Some(f));
        }
        if let Some(f) = self.find_file_by_basename(file_or_docid)? {
            return Ok(Some(f));
        }
        self.find_file_by_fuzzy(file_or_docid)
    }

    /// Fuzzy-match a query against all stored file basenames using Levenshtein distance.
    /// Returns the unique closest match within distance ≤ 2, or an error if ambiguous.
    fn find_file_by_fuzzy(&self, query: &str) -> Result<Option<FileRecord>> {
        use strsim::levenshtein;

        // Normalize query: strip .md, lowercase.
        let query_stem = query.strip_suffix(".md").unwrap_or(query).to_lowercase();

        // Collect all (path, basename_stem) pairs from the store.
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut best_distance = usize::MAX;
        let mut best_paths: Vec<String> = Vec::new();

        for path in &paths {
            // Extract basename and strip .md extension for comparison.
            let basename = std::path::Path::new(path)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or(path);
            let stem = basename
                .strip_suffix(".md")
                .unwrap_or(basename)
                .to_lowercase();

            let dist = levenshtein(&query_stem, &stem);
            if dist > 2 {
                continue;
            }
            if dist < best_distance {
                best_distance = dist;
                best_paths.clear();
                best_paths.push(path.clone());
            } else if dist == best_distance {
                best_paths.push(path.clone());
            }
        }

        match best_paths.len() {
            0 => Ok(None),
            1 => self.get_file(&best_paths[0]),
            _ => Err(anyhow::anyhow!(
                "ambiguous fuzzy match for '{}': [{}]",
                query,
                best_paths.join(", ")
            )),
        }
    }

    pub fn resolve_tag(&self, proposed: &str) -> Result<crate::tags::TagResolution> {
        crate::tags::resolve_tag(&self.conn, proposed)
    }

    pub fn resolve_tags(&self, proposed: &[String]) -> Result<Vec<String>> {
        crate::tags::resolve_tags(&self.conn, proposed)
    }

    pub fn register_tag(&self, name: &str, created_by: &str) -> Result<()> {
        crate::tags::register_tag(&self.conn, name, created_by)
    }

    // ── CLI Events ──────────────────────────────────────────────

    /// Log a CLI event for observability/analytics.
    pub fn log_cli_event(
        &self,
        operation: &str,
        outcome: &str,
        detail: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO cli_events (timestamp, operation, outcome, detail)
             VALUES (datetime('now'), ?1, ?2, ?3)",
            params![operation, outcome, detail],
        )?;
        Ok(())
    }

    /// Get CLI events since a given ISO-8601 date string (e.g., "2020-01-01").
    pub fn get_cli_events_since(&self, since: &str) -> Result<Vec<CliEvent>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, operation, outcome, detail
             FROM cli_events WHERE timestamp >= ?1 ORDER BY timestamp DESC",
        )?;
        let rows = stmt.query_map(params![since], |row| {
            Ok(CliEvent {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                operation: row.get(2)?,
                outcome: row.get(3)?,
                detail: row.get(4)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Prune CLI events older than the given number of days.
    pub fn prune_cli_events(&self, days: u32) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM cli_events WHERE julianday('now') - julianday(timestamp) > ?1",
            params![days],
        )?;
        Ok(deleted)
    }

    // ── Hard delete ──────────────────────────────────────────────

    /// Completely remove a file and all associated data from the store.
    ///
    /// Deletion order:
    /// 1. Collect chunk vector_ids for the file
    /// 2. Delete from `chunks_vec` (virtual table, no CASCADE)
    /// 3. Delete from `chunks_fts` (virtual table, no CASCADE)
    /// 4. Delete from `edges` where from_file or to_file matches
    /// 5. Delete from `files` (CASCADE handles chunks table)
    pub fn delete_file_hard(&self, path: &str) -> Result<()> {
        let file = self
            .get_file(path)?
            .ok_or_else(|| anyhow::anyhow!("file not found: {}", path))?;
        let file_id = file.id;

        // 1. Collect chunk vector_ids
        let vector_ids = self.get_vector_ids_for_file(file_id)?;

        // 2. Delete from chunks_vec (virtual table — no CASCADE)
        for vid in &vector_ids {
            self.delete_vec(*vid)?;
        }

        // 3. Delete from chunks_fts (virtual table — no CASCADE)
        self.delete_fts_chunks_for_file(file_id)?;

        // 4. Delete from edges (both directions)
        self.delete_edges_for_file(file_id)?;

        // 5. Delete from files (CASCADE handles chunks table)
        self.delete_file(file_id)?;

        Ok(())
    }

    // ── Unresolved Links ─────────────────────────────────────────

    /// Record a wikilink target that could not be resolved during indexing.
    pub fn insert_unresolved_link(&self, source_file: &str, target: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO unresolved_links (source_file, target) VALUES (?1, ?2)",
            params![source_file, target],
        )?;
        Ok(())
    }

    /// Remove all unresolved links originating from the given source file.
    pub fn clear_unresolved_links_for_file(&self, source_file: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM unresolved_links WHERE source_file = ?1",
            params![source_file],
        )?;
        Ok(())
    }

    /// Return all unresolved links (source_file, target) pairs.
    pub fn get_unresolved_links(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT source_file, target FROM unresolved_links ORDER BY source_file")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Health Queries ───────────────────────────────────────────

    /// Find files that have no edges (neither incoming nor outgoing).
    /// Optionally exclude files whose path starts with any of the given prefixes.
    pub fn find_isolated_files(&self, exclude_prefixes: &[&str]) -> Result<Vec<FileRecord>> {
        let all_files = self.get_all_files()?;
        let connected: HashSet<i64> = {
            let mut stmt = self.conn.prepare(
                "SELECT DISTINCT id FROM files WHERE id IN \
                 (SELECT from_file FROM edges UNION SELECT to_file FROM edges)",
            )?;
            let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
            let mut set = HashSet::new();
            for row in rows {
                set.insert(row?);
            }
            set
        };
        let isolated = all_files
            .into_iter()
            .filter(|f| !connected.contains(&f.id))
            .filter(|f| {
                !exclude_prefixes
                    .iter()
                    .any(|prefix| f.path.starts_with(prefix))
            })
            .collect();
        Ok(isolated)
    }
}

fn parse_tags(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

fn chrono_now() -> String {
    // Simple ISO-8601-ish timestamp without pulling in chrono crate.
    // Uses the system time formatted via std.
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    // Return seconds as a string; good enough for ordering.
    // A later task can swap in proper chrono formatting.
    format!("{}", duration.as_secs())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;

    #[test]
    fn test_create_schema() {
        let store = Store::open_memory().unwrap();
        // Verify all four tables exist by querying sqlite_master.
        let tables: Vec<String> = {
            let mut stmt = store
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .unwrap();
            let rows = stmt.query_map([], |row| row.get(0)).unwrap();
            rows.filter_map(|r| r.ok()).collect()
        };
        assert!(tables.contains(&"meta".to_string()));
        assert!(tables.contains(&"files".to_string()));
        assert!(tables.contains(&"chunks".to_string()));
        assert!(tables.contains(&"tombstones".to_string()));
    }

    #[test]
    fn test_insert_and_get_file() {
        let store = Store::open_memory().unwrap();
        let tags = vec!["rust".to_string(), "programming".to_string()];
        let docid = generate_docid("notes/test.md");
        let file_id = store
            .insert_file(
                "notes/test.md",
                "abc123",
                1700000000,
                &tags,
                &docid,
                None,
                None,
            )
            .unwrap();
        assert!(file_id > 0);

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.path, "notes/test.md");
        assert_eq!(rec.content_hash, "abc123");
        assert_eq!(rec.mtime, 1700000000);
        assert_eq!(rec.tags, tags);
        assert_eq!(rec.docid.unwrap(), docid);
    }

    #[test]
    fn test_insert_and_get_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/chunk_test.md",
                "hash1",
                100,
                &[],
                &generate_docid("notes/chunk_test.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_chunk(file_id, 0, "Heading 1", "Some text here", 1, 42)
            .unwrap();
        store
            .insert_chunk(file_id, 1, "Heading 2", "More text", 2, 30)
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
    fn test_delete_file_cascades_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/del.md",
                "hash",
                100,
                &[],
                &generate_docid("notes/del.md"),
                None,
                None,
            )
            .unwrap();
        store
            .insert_chunk(file_id, 0, "H", "snippet", 10, 5)
            .unwrap();
        store
            .insert_chunk(file_id, 1, "H2", "snippet2", 11, 6)
            .unwrap();

        assert_eq!(store.get_chunks_by_file(file_id).unwrap().len(), 2);

        store.delete_file(file_id).unwrap();

        assert!(store.get_file("notes/del.md").unwrap().is_none());
        assert_eq!(store.get_chunks_by_file(file_id).unwrap().len(), 0);
    }

    #[test]
    fn test_tombstone_lifecycle() {
        let store = Store::open_memory().unwrap();

        assert_eq!(store.tombstone_count().unwrap(), 0);
        assert!(store.get_tombstones().unwrap().is_empty());

        store.add_tombstones(&[100, 200, 300]).unwrap();
        assert_eq!(store.tombstone_count().unwrap(), 3);

        let ts = store.get_tombstones().unwrap();
        assert!(ts.contains(&100));
        assert!(ts.contains(&200));
        assert!(ts.contains(&300));

        // Duplicate insert should be ignored.
        store.add_tombstones(&[200, 400]).unwrap();
        assert_eq!(store.tombstone_count().unwrap(), 4);

        store.clear_tombstones().unwrap();
        assert_eq!(store.tombstone_count().unwrap(), 0);
    }

    #[test]
    fn test_file_hash_changed() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/change.md");
        let file_id = store
            .insert_file(
                "notes/change.md",
                "old_hash",
                100,
                &["tag1".to_string()],
                &docid,
                None,
                None,
            )
            .unwrap();
        store.insert_chunk(file_id, 0, "H", "text", 50, 10).unwrap();
        store
            .insert_chunk(file_id, 1, "H2", "text2", 51, 12)
            .unwrap();

        // Simulate detecting hash change: collect old vector_ids for tombstoning.
        let old_vector_ids = store.get_vector_ids_for_file(file_id).unwrap();
        assert_eq!(old_vector_ids.len(), 2);
        assert!(old_vector_ids.contains(&50));
        assert!(old_vector_ids.contains(&51));

        // Tombstone old vectors, delete file (cascades chunks), re-insert.
        store.add_tombstones(&old_vector_ids).unwrap();
        store.delete_file(file_id).unwrap();

        let new_file_id = store
            .insert_file(
                "notes/change.md",
                "new_hash",
                200,
                &["tag1".to_string()],
                &docid,
                None,
                None,
            )
            .unwrap();
        store
            .insert_chunk(new_file_id, 0, "H", "new text", 60, 15)
            .unwrap();

        let rec = store.get_file("notes/change.md").unwrap().unwrap();
        assert_eq!(rec.content_hash, "new_hash");

        let chunks = store.get_chunks_by_file(new_file_id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].vector_id, 60);

        // Old vectors are tombstoned.
        let ts = store.get_tombstones().unwrap();
        assert!(ts.contains(&50));
        assert!(ts.contains(&51));
    }

    #[test]
    fn test_vault_path_storage() {
        let store = Store::open_memory().unwrap();

        assert!(store.get_meta("vault_path").unwrap().is_none());

        store.set_meta("vault_path", "/home/user/vault").unwrap();
        assert_eq!(
            store.get_meta("vault_path").unwrap().unwrap(),
            "/home/user/vault"
        );

        // Update the value.
        store.set_meta("vault_path", "/other/vault").unwrap();
        assert_eq!(
            store.get_meta("vault_path").unwrap().unwrap(),
            "/other/vault"
        );

        // Verify stats reflects it.
        let st = store.stats().unwrap();
        assert_eq!(st.vault_path.unwrap(), "/other/vault");
    }

    #[test]
    fn test_get_file_by_docid() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/findme.md");
        store
            .insert_file("notes/findme.md", "hash", 100, &[], &docid, None, None)
            .unwrap();

        let rec = store.get_file_by_docid(&docid).unwrap().unwrap();
        assert_eq!(rec.path, "notes/findme.md");
        assert_eq!(rec.docid.unwrap(), docid);

        // Non-existent docid returns None.
        assert!(store.get_file_by_docid("ffffff").unwrap().is_none());
    }

    // ── Edge tests ─────────────────────────────────────────────

    /// Helper: create two files and return their IDs.
    fn setup_two_files(store: &Store) -> (i64, i64) {
        let a = store
            .insert_file(
                "notes/a.md",
                "ha",
                100,
                &[],
                &generate_docid("notes/a.md"),
                None,
                None,
            )
            .unwrap();
        let b = store
            .insert_file(
                "notes/b.md",
                "hb",
                100,
                &[],
                &generate_docid("notes/b.md"),
                None,
                None,
            )
            .unwrap();
        (a, b)
    }

    #[test]
    fn test_insert_and_get_edges() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);

        store.insert_edge(a, b, "wikilink").unwrap();

        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (b, "wikilink".to_string()));

        let inc = store.get_incoming(b, None).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0], (a, "wikilink".to_string()));

        // No edges in the other direction.
        assert!(store.get_outgoing(b, None).unwrap().is_empty());
        assert!(store.get_incoming(a, None).unwrap().is_empty());
    }

    #[test]
    fn deleting_a_files_own_edges_leaves_the_backlinks_into_it() {
        // The re-index deletion (issue #27). `b` is being re-indexed: the edges
        // it authored go, the edge `a` authored into it stays — `a`'s content
        // has not changed and nothing else is going to put that edge back.
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(b, a, "wikilink").unwrap();

        store.delete_outgoing_edges_for_file(b).unwrap();

        assert!(store.get_outgoing(b, None).unwrap().is_empty());
        assert_eq!(
            store.get_incoming(b, None).unwrap(),
            vec![(a, "wikilink".to_string())],
            "a's link into b is a's to delete, not b's"
        );
    }

    #[test]
    fn deleting_a_files_chunks_keeps_its_row_and_its_backlinks() {
        // `delete_file` cascades `edges` in both directions, which is what made
        // every edit destroy backlinks. The re-index path clears chunks instead
        // and lets `insert_file` upsert the row (issue #27).
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        store.insert_edge(a, b, "wikilink").unwrap();

        store.delete_chunks_for_file(b).unwrap();

        assert!(store.get_chunks_by_file(b).unwrap().is_empty());
        assert!(store.get_file("notes/b.md").unwrap().is_some());
        assert_eq!(store.get_incoming(b, None).unwrap().len(), 1);

        // And the upsert returns the same id, so the edge still points at it.
        let reborn = store
            .insert_file(
                "notes/b.md",
                "hb2",
                101,
                &[],
                &generate_docid("notes/b.md"),
                None,
                None,
            )
            .unwrap();
        assert_eq!(reborn, b);
        assert_eq!(store.get_incoming(reborn, None).unwrap().len(), 1);
    }

    #[test]
    fn test_delete_edges_for_file_both_directions() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        let c = store
            .insert_file(
                "notes/c.md",
                "hc",
                100,
                &[],
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        // a -> b, c -> a
        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(c, a, "mention").unwrap();

        // Delete edges for file a — should remove both.
        store.delete_edges_for_file(a).unwrap();

        assert!(store.get_outgoing(a, None).unwrap().is_empty());
        assert!(store.get_incoming(a, None).unwrap().is_empty());
        assert!(store.get_incoming(b, None).unwrap().is_empty());
        assert!(store.get_outgoing(c, None).unwrap().is_empty());
    }

    #[test]
    fn test_edge_cascade_on_file_delete() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        let c = store
            .insert_file(
                "notes/c.md",
                "hc",
                100,
                &[],
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        // a -> b, b -> c
        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(b, c, "mention").unwrap();

        // Delete file b — CASCADE should remove both edges.
        store.delete_file(b).unwrap();

        assert!(store.get_outgoing(a, None).unwrap().is_empty());
        assert!(store.get_incoming(c, None).unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_edge_ignored() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(a, b, "wikilink").unwrap(); // duplicate

        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 1);

        // Same pair with different type is NOT a duplicate.
        store.insert_edge(a, b, "mention").unwrap();
        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_get_outgoing_filtered_by_type() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);
        let c = store
            .insert_file(
                "notes/c.md",
                "hc",
                100,
                &[],
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(a, c, "mention").unwrap();

        let wikilinks = store.get_outgoing(a, Some("wikilink")).unwrap();
        assert_eq!(wikilinks.len(), 1);
        assert_eq!(wikilinks[0].0, b);

        let mentions = store.get_outgoing(a, Some("mention")).unwrap();
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].0, c);

        // Incoming filtered.
        let inc = store.get_incoming(b, Some("wikilink")).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].0, a);

        let inc = store.get_incoming(b, Some("mention")).unwrap();
        assert!(inc.is_empty());
    }

    // ── Graph helper tests ─────────────────────────────────────

    #[test]
    fn test_get_neighbors_depth_1() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/f1.md",
                "h1",
                100,
                &[],
                &generate_docid("n/f1.md"),
                None,
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "n/f2.md",
                "h2",
                100,
                &[],
                &generate_docid("n/f2.md"),
                None,
                None,
            )
            .unwrap();
        let f3 = store
            .insert_file(
                "n/f3.md",
                "h3",
                100,
                &[],
                &generate_docid("n/f3.md"),
                None,
                None,
            )
            .unwrap();

        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f1, f3, "wikilink").unwrap();

        let neighbors = store.get_neighbors(f1, 1).unwrap();
        assert_eq!(neighbors.len(), 2);

        let ids: Vec<i64> = neighbors.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&f2));
        assert!(ids.contains(&f3));

        // All at depth 1.
        for (_, d) in &neighbors {
            assert_eq!(*d, 1);
        }
    }

    #[test]
    fn test_get_neighbors_depth_2() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/f1.md",
                "h1",
                100,
                &[],
                &generate_docid("n/f1.md"),
                None,
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "n/f2.md",
                "h2",
                100,
                &[],
                &generate_docid("n/f2.md"),
                None,
                None,
            )
            .unwrap();
        let f3 = store
            .insert_file(
                "n/f3.md",
                "h3",
                100,
                &[],
                &generate_docid("n/f3.md"),
                None,
                None,
            )
            .unwrap();
        let f4 = store
            .insert_file(
                "n/f4.md",
                "h4",
                100,
                &[],
                &generate_docid("n/f4.md"),
                None,
                None,
            )
            .unwrap();

        // f1 -> f2 -> f3 -> f4
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f3, "wikilink").unwrap();
        store.insert_edge(f3, f4, "wikilink").unwrap();

        let neighbors = store.get_neighbors(f1, 2).unwrap();
        assert_eq!(neighbors.len(), 2);

        // f2 at depth 1, f3 at depth 2, f4 NOT included.
        let map: std::collections::HashMap<i64, usize> = neighbors.into_iter().collect();
        assert_eq!(map[&f2], 1);
        assert_eq!(map[&f3], 2);
        assert!(!map.contains_key(&f4));
    }

    #[test]
    fn test_get_neighbors_includes_backlinks() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/f1.md",
                "h1",
                100,
                &[],
                &generate_docid("n/f1.md"),
                None,
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "n/f2.md",
                "h2",
                100,
                &[],
                &generate_docid("n/f2.md"),
                None,
                None,
            )
            .unwrap();

        // f2 links to f1; f1 has no outgoing links of its own.
        store.insert_edge(f2, f1, "wikilink").unwrap();

        // Neighbor discovery is undirected: f1's neighbors include its
        // backlink f2 even though f1 has no outgoing edge.
        let neighbors = store.get_neighbors(f1, 1).unwrap();
        let ids: Vec<i64> = neighbors.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![f2]);
        assert_eq!(neighbors[0].1, 1);
    }

    #[test]
    fn test_get_shared_tags_files() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/f1.md",
                "h1",
                100,
                &["rust".to_string(), "cli".to_string()],
                &generate_docid("n/f1.md"),
                None,
                None,
            )
            .unwrap();
        let f2 = store
            .insert_file(
                "n/f2.md",
                "h2",
                100,
                &["rust".to_string(), "web".to_string()],
                &generate_docid("n/f2.md"),
                None,
                None,
            )
            .unwrap();
        let _f3 = store
            .insert_file(
                "n/f3.md",
                "h3",
                100,
                &["python".to_string()],
                &generate_docid("n/f3.md"),
                None,
                None,
            )
            .unwrap();

        let shared = store.get_shared_tags_files(f1, 10).unwrap();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0], f2);
    }

    #[test]
    fn test_file_contains_term() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/fts.md",
                "h1",
                100,
                &[],
                &generate_docid("n/fts.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_fts_chunk(f1, 0, "BRE-2579 delivery date extension")
            .unwrap();

        assert!(store.file_contains_term(f1, "delivery").unwrap());
        assert!(store.file_contains_term(f1, "extension").unwrap());
        assert!(!store.file_contains_term(f1, "checkout").unwrap());
    }

    #[test]
    fn test_get_best_chunk_for_file() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file(
                "n/best.md",
                "h1",
                100,
                &[],
                &generate_docid("n/best.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_chunk(f1, 0, "Small heading", "small snippet", 1, 10)
            .unwrap();
        store
            .insert_chunk(f1, 1, "Big heading", "big snippet", 2, 100)
            .unwrap();

        let best = store.get_best_chunk_for_file(f1).unwrap().unwrap();
        assert_eq!(best.heading, "Big heading");
        assert_eq!(best.snippet, "big snippet");
        assert_eq!(best.seq, 1, "identity travels with the chunk");
    }

    /// Insert a file with one chunk per (heading, text) pair, numbered in order.
    fn seed_sections(store: &Store, path: &str, sections: &[(&str, &str)]) -> i64 {
        let docid = generate_docid(path);
        store
            .insert_file(path, "hash", 100, &[], &docid, None, None)
            .unwrap();
        let file_id = store.get_file(path).unwrap().unwrap().id;
        for (seq, (heading, text)) in sections.iter().enumerate() {
            store
                .insert_chunk(
                    file_id,
                    seq as i64,
                    heading,
                    text,
                    (file_id * 100 + seq as i64) as u64,
                    10,
                )
                .unwrap();
            store.insert_fts_chunk(file_id, seq as i64, text).unwrap();
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

    #[test]
    fn test_migration_backfills_chunk_seq_from_insertion_order() {
        // Databases indexed before chunk identity existed have no seq column, but
        // their FTS rows were written with 0,1,2… — the backfill has to land on
        // the same numbers or the two lanes silently retrieve different chunks.
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("legacy.db");

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE files (
                     id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                     content_hash TEXT NOT NULL, mtime INTEGER NOT NULL,
                     tags TEXT NOT NULL DEFAULT '[]', indexed_at TEXT NOT NULL, docid TEXT);
                 CREATE TABLE chunks (
                     id INTEGER PRIMARY KEY,
                     file_id INTEGER NOT NULL,
                     heading TEXT NOT NULL, snippet TEXT NOT NULL,
                     vector_id INTEGER UNIQUE NOT NULL, token_count INTEGER NOT NULL,
                     vector BLOB);
                 INSERT INTO files (id, path, content_hash, mtime, indexed_at)
                     VALUES (1, 'a.md', 'h', 1, 'now'), (2, 'b.md', 'h', 1, 'now');
                 INSERT INTO chunks (file_id, heading, snippet, vector_id, token_count)
                     VALUES (1, 'A0', 's', 10, 1), (1, 'A1', 's', 11, 1),
                            (2, 'B0', 's', 12, 1), (1, 'A2', 's', 13, 1);",
            )
            .unwrap();
        }

        let store = Store::open(&db_path).unwrap();

        let seq_of = |heading: &str| -> i64 {
            store
                .conn
                .query_row(
                    "SELECT seq FROM chunks WHERE heading = ?1",
                    [heading],
                    |row| row.get(0),
                )
                .unwrap()
        };
        assert_eq!(seq_of("A0"), 0);
        assert_eq!(seq_of("A1"), 1);
        assert_eq!(seq_of("A2"), 2, "numbering is per file, by insertion order");
        assert_eq!(seq_of("B0"), 0, "a second file restarts at zero");

        // Re-opening must not renumber anything.
        drop(store);
        let store = Store::open(&db_path).unwrap();
        assert_eq!(
            store
                .conn
                .query_row("SELECT seq FROM chunks WHERE heading = 'A2'", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
    }

    /// Issue #14. Before `chunks.text` existed, a chunk's full text lived only
    /// in the FTS index, which cannot be keyed into. Adding the column has to
    /// recover it for databases already on disk, or the reranker on an
    /// un-reindexed vault silently keeps reading previews.
    #[test]
    fn the_text_column_backfills_from_the_fts_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("legacy.db");

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE files (
                     id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                     content_hash TEXT NOT NULL, mtime INTEGER NOT NULL,
                     tags TEXT NOT NULL DEFAULT '[]', indexed_at TEXT NOT NULL, docid TEXT);
                 CREATE TABLE chunks (
                     id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, seq INTEGER NOT NULL,
                     heading TEXT NOT NULL, snippet TEXT NOT NULL,
                     vector_id INTEGER UNIQUE NOT NULL, token_count INTEGER NOT NULL,
                     vector BLOB);
                 CREATE VIRTUAL TABLE chunks_fts USING fts5(
                     content, file_id UNINDEXED, chunk_seq UNINDEXED);
                 INSERT INTO files (id, path, content_hash, mtime, indexed_at)
                     VALUES (1, 'a.md', 'h', 1, 'now');
                 INSERT INTO chunks (file_id, seq, heading, snippet, vector_id, token_count)
                     VALUES (1, 0, 'A0', 'the preview', 10, 1),
                            (1, 1, 'A1', 'orphan preview', 11, 1);
                 INSERT INTO chunks_fts (content, file_id, chunk_seq)
                     VALUES ('the preview and everything after it', 1, 0);",
            )
            .unwrap();
        }

        let store = Store::open(&db_path).unwrap();

        assert_eq!(
            store.get_chunk_by_seq(1, 0).unwrap().unwrap().text,
            "the preview and everything after it",
            "the FTS copy should have been recovered"
        );
        assert_eq!(
            store.get_chunk_by_seq(1, 1).unwrap().unwrap().text,
            "orphan preview",
            "a chunk with no FTS row keeps its snippet — a truncation of the \
             right text beats the wrong text"
        );

        // Re-opening must not re-run the backfill over text already written.
        drop(store);
        let store = Store::open(&db_path).unwrap();
        assert_eq!(
            store.get_chunk_by_seq(1, 0).unwrap().unwrap().text,
            "the preview and everything after it"
        );
    }

    /// The reranker's read. A missing chunk is `None` rather than an error, so
    /// one stale candidate cannot take the whole lane down.
    #[test]
    fn get_chunk_texts_reports_misses_without_failing() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("a.md", "h", 0, &[], "d", None, None)
            .unwrap();
        let long = "x".repeat(500);
        store.insert_chunk(file_id, 0, "H", &long, 1, 10).unwrap();

        let texts = store
            .get_chunk_texts(&[(file_id, 0), (file_id, 9), (999, 0)])
            .unwrap();

        assert_eq!(texts[0].as_deref(), Some(long.as_str()));
        assert_eq!(texts[1], None, "no such seq");
        assert_eq!(texts[2], None, "no such file");
    }

    #[test]
    fn test_get_edge_stats() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file(
                "n/a.md",
                "ha",
                100,
                &[],
                &generate_docid("n/a.md"),
                None,
                None,
            )
            .unwrap();
        let b = store
            .insert_file(
                "n/b.md",
                "hb",
                100,
                &[],
                &generate_docid("n/b.md"),
                None,
                None,
            )
            .unwrap();
        let c = store
            .insert_file(
                "n/c.md",
                "hc",
                100,
                &[],
                &generate_docid("n/c.md"),
                None,
                None,
            )
            .unwrap();
        // d is isolated (no edges).
        let _d = store
            .insert_file(
                "n/d.md",
                "hd",
                100,
                &[],
                &generate_docid("n/d.md"),
                None,
                None,
            )
            .unwrap();

        store.insert_edge(a, b, "wikilink").unwrap();
        store.insert_edge(a, c, "wikilink").unwrap();
        store.insert_edge(b, c, "mention").unwrap();

        let stats = store.get_edge_stats().unwrap();
        assert_eq!(stats.total_edges, 3);
        assert_eq!(stats.wikilink_count, 2);
        assert_eq!(stats.mention_count, 1);
        assert_eq!(stats.connected_file_count, 3); // a, b, c
        assert_eq!(stats.isolated_file_count, 1); // d
    }

    #[test]
    fn test_list_files_no_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file(
                "01-Projects/a.md",
                "h1",
                100,
                &["rust".into()],
                "aaa111",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "02-Areas/b.md",
                "h2",
                200,
                &["health".into()],
                "bbb222",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "01-Projects/c.md",
                "h3",
                300,
                &["rust".into(), "cli".into()],
                "ccc333",
                None,
                None,
            )
            .unwrap();
        let files = store.list_files(None, &[], None, 20).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_list_files_folder_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, &[], "aaa111", None, None)
            .unwrap();
        store
            .insert_file("02-Areas/b.md", "h2", 200, &[], "bbb222", None, None)
            .unwrap();
        let files = store
            .list_files(Some("01-Projects"), &[], None, 20)
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "01-Projects/a.md");
    }

    #[test]
    fn test_list_files_tag_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file(
                "a.md",
                "h1",
                100,
                &["rust".into(), "cli".into()],
                "aaa111",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file("b.md", "h2", 200, &["rust".into()], "bbb222", None, None)
            .unwrap();
        store
            .insert_file("c.md", "h3", 300, &["python".into()], "ccc333", None, None)
            .unwrap();
        let files = store.list_files(None, &["rust".into()], None, 20).unwrap();
        assert_eq!(files.len(), 2);
        let files = store
            .list_files(None, &["rust".into(), "cli".into()], None, 20)
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "a.md");
    }

    #[test]
    fn test_list_files_created_by_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("a.md", "h1", 100, &[], "aaa111", Some("cli"), None)
            .unwrap();
        store
            .insert_file("b.md", "h2", 200, &[], "bbb222", Some("mcp"), None)
            .unwrap();
        store
            .insert_file("c.md", "h3", 300, &[], "ccc333", None, None)
            .unwrap();

        // Filter by "cli" → only the cli-created file
        let files = store.list_files(None, &[], Some("cli"), 20).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "a.md");
        assert_eq!(files[0].created_by, Some("cli".to_string()));

        // Filter by "mcp" → only the mcp-created file
        let files = store.list_files(None, &[], Some("mcp"), 20).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "b.md");

        // Filter by None → all 3
        let files = store.list_files(None, &[], None, 20).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_folder_note_counts() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, &[], "a1", None, None)
            .unwrap();
        store
            .insert_file("01-Projects/b.md", "h2", 100, &[], "b2", None, None)
            .unwrap();
        store
            .insert_file("02-Areas/c.md", "h3", 100, &[], "c3", None, None)
            .unwrap();
        store
            .insert_file("root.md", "h4", 100, &[], "d4", None, None)
            .unwrap();
        let counts = store.folder_note_counts().unwrap();
        assert!(counts.iter().any(|(f, c)| f == "01-Projects" && *c == 2));
        assert!(counts.iter().any(|(f, c)| f == "02-Areas" && *c == 1));
        assert!(counts.iter().any(|(f, c)| f == "(root)" && *c == 1));
    }

    #[test]
    fn test_top_tags() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file(
                "a.md",
                "h1",
                100,
                &["rust".into(), "cli".into()],
                "a1",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "b.md",
                "h2",
                100,
                &["rust".into(), "web".into()],
                "b2",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file("c.md", "h3", 100, &["rust".into()], "c3", None, None)
            .unwrap();
        let tags = store.top_tags(10).unwrap();
        assert_eq!(tags[0].0, "rust");
        assert_eq!(tags[0].1, 3);
    }

    #[test]
    fn test_recent_files() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("old.md", "h1", 100, &[], "a1", None, None)
            .unwrap();
        store
            .insert_file("new.md", "h2", 200, &[], "b2", None, None)
            .unwrap();
        let recent = store.recent_files(1).unwrap();
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_edge_count_for_file() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], "a1", None, None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, &[], "b2", None, None)
            .unwrap();
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f1, "wikilink").unwrap();
        assert_eq!(store.edge_count_for_file(f1).unwrap(), 2);
        assert_eq!(store.edge_count_for_file(f2).unwrap(), 2);
    }

    #[test]
    fn test_find_file_by_basename() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file(
                "01-Projects/Work/note.md",
                "h1",
                100,
                &[],
                "aaa111",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file("root.md", "h2", 100, &[], "bbb222", None, None)
            .unwrap();

        let found = store.find_file_by_basename("note").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().path, "01-Projects/Work/note.md");

        let found = store.find_file_by_basename("note.md").unwrap();
        assert!(found.is_some());

        let found = store.find_file_by_basename("nonexistent").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_edge_counts_for_files() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, &[], "a1", None, None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, &[], "b2", None, None)
            .unwrap();
        let f3 = store
            .insert_file("c.md", "h3", 100, &[], "c3", None, None)
            .unwrap();
        store.insert_edge(f1, f2, "wikilink").unwrap();
        store.insert_edge(f2, f1, "wikilink").unwrap();
        store.insert_edge(f1, f3, "wikilink").unwrap();
        let counts = store.edge_counts_for_files(&[f1, f2, f3]).unwrap();
        assert_eq!(*counts.get(&f1).unwrap(), 3);
        assert_eq!(*counts.get(&f2).unwrap(), 2);
        assert_eq!(*counts.get(&f3).unwrap(), 1);
        // Empty input returns empty map
        let empty = store.edge_counts_for_files(&[]).unwrap();
        assert!(empty.is_empty());
    }

    // ── Vec integration tests ───────────────────────────────────

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

        let results = store
            .search_vec(&vector, 1, &std::collections::HashSet::new())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_migrate_vectors_to_vec0() {
        let store = Store::open_memory().unwrap();
        store.ensure_embedding_dim(256).unwrap();
        // Insert a file + chunk with a vector BLOB.
        let file_id = store
            .insert_file("test.md", "hash123", 0, &[], "abc123", None, None)
            .unwrap();
        let vector: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        store
            .insert_chunk_with_vector(file_id, 0, "heading", "snippet", 0, 100, &vector)
            .unwrap();

        // Clear vec0 to simulate a pre-migration state, then re-run the migration.
        store.clear_vec().unwrap();
        store.migrate_vectors_to_vec0().unwrap();

        // Verify vec0 is now populated.
        let results = store
            .search_vec(&vector, 1, &std::collections::HashSet::new())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_store_transaction() {
        let store = Store::open_memory().unwrap();
        store.begin_transaction().unwrap();
        store.set_meta("test_key", "test_value").unwrap();
        store.commit().unwrap();
        assert_eq!(
            store.get_meta("test_key").unwrap(),
            Some("test_value".into())
        );
    }

    #[test]
    fn test_next_vector_id_empty() {
        let store = Store::open_memory().unwrap();
        assert_eq!(store.next_vector_id().unwrap(), 0);
    }

    // ── Tag query tests ──────────────────────────────────────────

    #[test]
    fn test_tag_query_functions() {
        let store = Store::open_memory().unwrap();

        // Register tags with different creators
        store.register_tag("rust", "indexer").unwrap();
        store.register_tag("work", "indexer").unwrap();
        store.register_tag("engraph", "claude-code").unwrap();
        store.register_tag("decision", "claude-code").unwrap();

        // Bump usage counts
        store.register_tag("rust", "indexer").unwrap();
        store.register_tag("rust", "indexer").unwrap();

        // agent_created_tags: should return only non-indexer tags
        let agent_tags = store.agent_created_tags().unwrap();
        assert_eq!(agent_tags.len(), 2);
        assert!(agent_tags.iter().all(|(_, by, _)| by != "indexer"));
        let names: Vec<&str> = agent_tags.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"engraph"));
        assert!(names.contains(&"decision"));

        // low_usage_tags: tags with usage_count < 2
        let low = store.low_usage_tags(2).unwrap();
        // engraph and decision have count 1, work has count 1, rust has count 3
        assert!(low.iter().any(|(n, _)| n == "engraph"));
        assert!(low.iter().any(|(n, _)| n == "work"));
        assert!(!low.iter().any(|(n, _)| n == "rust"));

        // stale_tags: no tags should be stale since they were just created
        let stale = store.stale_tags(1).unwrap();
        assert!(stale.is_empty());
    }

    #[test]
    fn test_adjust_folder_centroid_increment() {
        let store = Store::open_memory().unwrap();
        // Seed centroid [1.0, 0.0, 0.0] with n=2
        store
            .upsert_folder_centroid("01-Projects", &[1.0, 0.0, 0.0], 2)
            .unwrap();
        // Add [0.0, 1.0, 0.0] → new = (old*2 + new) / 3 = [2/3, 1/3, 0]
        store
            .adjust_folder_centroid("01-Projects", &[0.0, 1.0, 0.0], true)
            .unwrap();
        let (centroid, count) = store
            .get_folder_centroid("01-Projects")
            .unwrap()
            .expect("centroid should exist");
        assert_eq!(count, 3);
        assert!((centroid[0] - 0.6667).abs() < 0.01);
        assert!((centroid[1] - 0.3333).abs() < 0.01);
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_folder_centroid_decrement() {
        let store = Store::open_memory().unwrap();
        // Seed centroid [0.667, 0.333, 0.0] with n=3
        store
            .upsert_folder_centroid("01-Projects", &[0.667, 0.333, 0.0], 3)
            .unwrap();
        // Remove [0.0, 1.0, 0.0] → new = (old*3 - vec) / 2 = [1.0005, ~0.0, 0.0]
        store
            .adjust_folder_centroid("01-Projects", &[0.0, 1.0, 0.0], false)
            .unwrap();
        let (centroid, count) = store
            .get_folder_centroid("01-Projects")
            .unwrap()
            .expect("centroid should exist");
        assert_eq!(count, 2);
        assert!((centroid[0] - 1.0).abs() < 0.01);
        assert!((centroid[1] - 0.0).abs() < 0.02); // (0.333*3 - 1.0)/2 = ~0.0
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_folder_centroid_decrement_last_file() {
        let store = Store::open_memory().unwrap();
        // Seed with n=1
        store
            .upsert_folder_centroid("01-Projects", &[1.0, 0.0, 0.0], 1)
            .unwrap();
        // Remove last file → centroid deleted
        store
            .adjust_folder_centroid("01-Projects", &[1.0, 0.0, 0.0], false)
            .unwrap();
        let result = store.get_folder_centroid("01-Projects").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_adjust_folder_centroid_new_folder() {
        let store = Store::open_memory().unwrap();
        // No existing centroid, increment → creates centroid
        store
            .adjust_folder_centroid("02-Areas", &[0.5, 0.5, 0.0], true)
            .unwrap();
        let (centroid, count) = store
            .get_folder_centroid("02-Areas")
            .unwrap()
            .expect("centroid should exist");
        assert_eq!(count, 1);
        assert!((centroid[0] - 0.5).abs() < 0.01);
        assert!((centroid[1] - 0.5).abs() < 0.01);
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_insert_file_with_created_by() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/test.md");
        store
            .insert_file(
                "notes/test.md",
                "hash1",
                100,
                &[],
                &docid,
                Some("cli"),
                None,
            )
            .unwrap();
        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.created_by, Some("cli".to_string()));
    }

    #[test]
    fn test_insert_file_without_created_by() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/test.md");
        store
            .insert_file("notes/test.md", "hash1", 100, &[], &docid, None, None)
            .unwrap();
        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.created_by, None);
    }

    #[test]
    fn test_update_file_path() {
        let store = Store::open_memory().unwrap();
        let old_docid = generate_docid("notes/old.md");
        let file_id = store
            .insert_file("notes/old.md", "hash1", 100, &[], &old_docid, None, None)
            .unwrap();

        let new_docid = generate_docid("notes/new.md");
        store
            .update_file_path("notes/old.md", "notes/new.md", &new_docid)
            .unwrap();

        // Old path should be gone
        assert!(store.get_file("notes/old.md").unwrap().is_none());
        // New path should exist with same file_id
        let rec = store.get_file("notes/new.md").unwrap().unwrap();
        assert_eq!(rec.id, file_id);
        assert_eq!(rec.docid.unwrap(), new_docid);
    }

    #[test]
    fn test_update_file_path_collision() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file(
                "notes/a.md",
                "h1",
                100,
                &[],
                &generate_docid("notes/a.md"),
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "notes/b.md",
                "h2",
                100,
                &[],
                &generate_docid("notes/b.md"),
                None,
                None,
            )
            .unwrap();

        // Renaming a→b should fail because b already exists
        let result =
            store.update_file_path("notes/a.md", "notes/b.md", &generate_docid("notes/b.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_get_chunk_vectors_for_file() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/vec.md",
                "h1",
                100,
                &[],
                &generate_docid("notes/vec.md"),
                None,
                None,
            )
            .unwrap();

        let v1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v2: Vec<f32> = vec![4.0, 5.0, 6.0];
        store
            .insert_chunk_with_vector(file_id, 0, "H1", "text1", 100, 10, &v1)
            .unwrap();
        store
            .insert_chunk_with_vector(file_id, 0, "H2", "text2", 101, 10, &v2)
            .unwrap();

        let vectors = store.get_chunk_vectors_for_file(file_id).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], v1);
        assert_eq!(vectors[1], v2);
    }

    #[test]
    fn test_get_chunk_vectors_empty() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/empty.md",
                "h1",
                100,
                &[],
                &generate_docid("notes/empty.md"),
                None,
                None,
            )
            .unwrap();

        let vectors = store.get_chunk_vectors_for_file(file_id).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_insert_placement_correction() {
        let store = Store::open_memory().unwrap();
        store
            .insert_placement_correction("notes/test.md", "00-Inbox", "01-Projects/Work")
            .unwrap();

        let corrections = store.get_placement_corrections(10).unwrap();
        assert_eq!(corrections.len(), 1);
        assert_eq!(corrections[0].file_path, "notes/test.md");
        assert_eq!(corrections[0].suggested_folder, "00-Inbox");
        assert_eq!(corrections[0].actual_folder, "01-Projects/Work");
        assert!(!corrections[0].corrected_at.is_empty());
    }

    #[test]
    fn test_get_placement_corrections_ordering() {
        let store = Store::open_memory().unwrap();
        store
            .insert_placement_correction("notes/first.md", "00-Inbox", "01-Projects")
            .unwrap();
        store
            .insert_placement_correction("notes/second.md", "02-Areas", "03-Resources")
            .unwrap();

        let corrections = store.get_placement_corrections(10).unwrap();
        assert_eq!(corrections.len(), 2);
        // Latest first (ORDER BY id DESC)
        assert_eq!(corrections[0].file_path, "notes/second.md");
        assert_eq!(corrections[1].file_path, "notes/first.md");
    }

    // ── LLM cache tests ────────────────────────────────────────

    #[test]
    fn test_llm_cache_roundtrip() {
        let store = Store::open_memory().unwrap();
        store
            .set_llm_cache("abc123", r#"{"intent":"exact"}"#, "qwen3-0.6B")
            .unwrap();
        let result = store.get_llm_cache("abc123").unwrap();
        assert_eq!(result, Some(r#"{"intent":"exact"}"#.to_string()));
    }

    #[test]
    fn test_llm_cache_miss() {
        let store = Store::open_memory().unwrap();
        let result = store.get_llm_cache("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_llm_cache_overwrite() {
        let store = Store::open_memory().unwrap();
        store.set_llm_cache("key1", "old", "model1").unwrap();
        store.set_llm_cache("key1", "new", "model1").unwrap();
        let result = store.get_llm_cache("key1").unwrap();
        assert_eq!(result, Some("new".to_string()));
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
        let hits = store
            .search_vec(&[0.1_f32; 768], 5, &std::collections::HashSet::new())
            .unwrap();
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
            .insert_file("note.md", "hash", 100, &[], "abc123", None, None)
            .unwrap();
        let vid = store.next_vector_id().unwrap();
        store
            .insert_chunk_with_vector(file_id, 0, "H", "snippet", vid, 10, &[0.1_f32; 256])
            .unwrap();
        store.insert_vec(vid, &[0.1_f32; 256]).unwrap();
        store.insert_fts_chunk(file_id, 0, "chunk text").unwrap();

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
        let err = store.verify_embedding_dim(768).unwrap_err().to_string();
        assert!(err.contains("256"), "{err}");
        assert!(err.contains("768"), "{err}");
        assert!(err.contains("engraph index"), "{err}");
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
        let path = dir.path().join("engraph.db");
        {
            let store = Store::open(&path).unwrap();
            store.ensure_embedding_dim(768).unwrap();
        }
        let store = Store::open(&path).unwrap();
        assert_eq!(store.vec_table_dim().unwrap(), Some(768));
    }

    // ── Fuzzy resolve tests ───────────────────────────────────

    #[test]
    fn test_resolve_file_fuzzy_match() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("Steve Barbera.md", "hash1", 100, &[], "ab1234", None, None)
            .unwrap();
        // "Steve Barbara" is within Levenshtein 2 of "Steve Barbera"
        let result = store.resolve_file("Steve Barbara").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().path, "Steve Barbera.md");
    }

    #[test]
    fn test_resolve_file_fuzzy_ambiguous() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("test-a.md", "h1", 100, &[], "aaa111", None, None)
            .unwrap();
        store
            .insert_file("test-b.md", "h2", 100, &[], "bbb222", None, None)
            .unwrap();
        // "test-c" is equidistant from both — should error, not pick arbitrarily
        let result = store.resolve_file("test-c");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_file_existing_docid() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("note.md", "hash", 100, &[], "abc123", None, None)
            .unwrap();
        let result = store.resolve_file("#abc123").unwrap();
        assert!(result.is_some());
    }

    // ── CLI events tests ────────────────────────────────────────

    #[test]
    fn test_cli_events_insert_and_query() {
        let store = Store::open_memory().unwrap();
        store.log_cli_event("edit", "success", None).unwrap();
        store
            .log_cli_event("edit", "fallback", Some("timeout"))
            .unwrap();
        let events = store.get_cli_events_since("2020-01-01").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].operation, "edit");
        assert_eq!(events[1].operation, "edit");
        // Most recent first
        assert_eq!(events[0].outcome, "fallback");
        assert_eq!(events[0].detail.as_deref(), Some("timeout"));
        assert_eq!(events[1].outcome, "success");
        assert!(events[1].detail.is_none());
    }

    #[test]
    fn test_cli_events_prune() {
        let store = Store::open_memory().unwrap();
        store.log_cli_event("search", "success", None).unwrap();
        // Events inserted just now should NOT be pruned with days=0 (julianday diff ~0)
        let pruned = store.prune_cli_events(1).unwrap();
        assert_eq!(pruned, 0);
        let events = store.get_cli_events_since("2020-01-01").unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_cli_events_table_exists() {
        let store = Store::open_memory().unwrap();
        let tables: Vec<String> = {
            let mut stmt = store
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='cli_events'")
                .unwrap();
            let rows = stmt.query_map([], |row| row.get(0)).unwrap();
            rows.filter_map(|r| r.ok()).collect()
        };
        assert!(tables.contains(&"cli_events".to_string()));
    }

    // ── delete_file_hard tests ──────────────────────────────────

    #[test]
    fn test_delete_file_hard() {
        let store = Store::open_memory().unwrap();
        let tags = vec!["tag".to_string()];
        let file_id = store
            .insert_file("delete-me.md", "hash", 100, &tags, "del123", None, None)
            .unwrap();

        // Insert a chunk + FTS entry + vec entry for the file
        let vid = store.next_vector_id().unwrap();
        store
            .insert_chunk(file_id, 0, "## Heading", "chunk text", vid, 10)
            .unwrap();
        store.insert_fts_chunk(file_id, 0, "chunk text").unwrap();

        // Insert an embedding vector into chunks_vec
        let embedding = vec![0.1_f32; 256];
        store.insert_vec(vid, &embedding).unwrap();

        // Insert an edge from this file to itself (just to test edge cleanup)
        let file_id2 = store
            .insert_file("other.md", "hash2", 100, &[], "oth123", None, None)
            .unwrap();
        store.insert_edge(file_id, file_id2, "wikilink").unwrap();
        store.insert_edge(file_id2, file_id, "wikilink").unwrap();

        // Verify data exists
        assert!(store.get_file("delete-me.md").unwrap().is_some());
        assert_eq!(store.get_chunks_by_file(file_id).unwrap().len(), 1);

        // Hard delete
        store.delete_file_hard("delete-me.md").unwrap();

        // File is gone
        assert!(store.get_file("delete-me.md").unwrap().is_none());
        // Chunks are gone (CASCADE)
        assert_eq!(store.get_chunks_by_file(file_id).unwrap().len(), 0);
        // FTS entries are gone
        let fts_results = store.fts_search("chunk text", 10).unwrap();
        assert!(fts_results.is_empty());
        // Edges are gone
        assert_eq!(store.edge_count_for_file(file_id).unwrap(), 0);
        // Only the edge from file_id2 to file_id was deleted, not file_id2's other edges
        // (file_id2 has no remaining edges since both directions involved file_id)
        assert_eq!(store.edge_count_for_file(file_id2).unwrap(), 0);
    }

    #[test]
    fn test_delete_file_hard_not_found() {
        let store = Store::open_memory().unwrap();
        let result = store.delete_file_hard("nonexistent.md");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("file not found"));
    }

    #[test]
    fn test_insert_file_with_note_date() {
        let store = Store::open_memory().unwrap();
        let note_date = Some(1774000000i64);
        store
            .insert_file("dated.md", "hash", 100, &[], "dat123", None, note_date)
            .unwrap();
        let file = store.get_file("dated.md").unwrap().unwrap();
        assert_eq!(file.note_date, note_date);
    }

    #[test]
    fn test_insert_file_without_note_date() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("undated.md", "hash", 100, &[], "und123", None, None)
            .unwrap();
        let file = store.get_file("undated.md").unwrap().unwrap();
        assert!(file.note_date.is_none());
    }

    #[test]
    fn test_get_files_in_date_range() {
        let store = Store::open_memory().unwrap();
        let day1 = 1774000000i64;
        let day2 = day1 + 86400;
        let day3 = day1 + 2 * 86400;
        store
            .insert_file("a.md", "h1", 100, &[], "aaa111", None, Some(day1))
            .unwrap();
        store
            .insert_file("b.md", "h2", 100, &[], "bbb222", None, Some(day2))
            .unwrap();
        store
            .insert_file("c.md", "h3", 100, &[], "ccc333", None, Some(day3))
            .unwrap();
        store
            .insert_file("d.md", "h4", 100, &[], "ddd444", None, None)
            .unwrap();
        let results = store.get_files_in_date_range(day1, day2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_count_files_with_dates() {
        let store = Store::open_memory().unwrap();
        let day1 = 1774000000i64;
        store
            .insert_file("a.md", "h1", 100, &[], "aaa111", None, Some(day1))
            .unwrap();
        store
            .insert_file("b.md", "h2", 100, &[], "bbb222", None, None)
            .unwrap();
        store
            .insert_file("c.md", "h3", 100, &[], "ccc333", None, Some(day1 + 86400))
            .unwrap();
        assert_eq!(store.count_files_with_dates().unwrap(), 2);
    }

    #[test]
    fn test_migration_log_insert_and_query() {
        let store = Store::open_memory().unwrap();
        store
            .log_migration(
                "mig-001",
                "old/note.md",
                "01-Projects/note.md",
                "project",
                0.9,
            )
            .unwrap();
        store
            .log_migration(
                "mig-001",
                "old/ref.md",
                "03-Resources/ref.md",
                "resource",
                0.85,
            )
            .unwrap();
        let entries = store.get_migration("mig-001").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].old_path, "old/note.md");
    }

    #[test]
    fn test_migration_log_get_last() {
        let store = Store::open_memory().unwrap();
        store
            .log_migration("mig-001", "a.md", "01-Projects/a.md", "project", 0.9)
            .unwrap();
        store
            .log_migration("mig-002", "b.md", "02-Areas/b.md", "area", 0.8)
            .unwrap();
        let last_id = store.get_last_migration_id().unwrap();
        assert_eq!(last_id.as_deref(), Some("mig-002"));
    }

    #[test]
    fn test_migration_log_delete() {
        let store = Store::open_memory().unwrap();
        store
            .log_migration("mig-001", "a.md", "01-Projects/a.md", "project", 0.9)
            .unwrap();
        store.delete_migration("mig-001").unwrap();
        assert!(store.get_migration("mig-001").unwrap().is_empty());
    }

    #[test]
    fn test_wal_mode_enabled() {
        // In-memory databases report "memory" for journal_mode, but busy_timeout should still apply.
        let store = Store::open_memory().unwrap();
        let mode: String = store
            .conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert!(
            mode == "wal" || mode == "memory",
            "expected 'wal' or 'memory', got '{mode}'"
        );
        let timeout: i64 = store
            .conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[test]
    fn test_wal_mode_file_backed() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_wal.db");
        let store = Store::open(&db_path).unwrap();
        let mode: String = store
            .conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        let timeout: i64 = store
            .conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[test]
    fn test_concurrent_file_backed_access() {
        // Two Store instances can open the same DB file simultaneously with WAL mode.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_concurrent.db");

        let store1 = Store::open(&db_path).unwrap();
        let store2 = Store::open(&db_path).unwrap();

        // Write with store1
        store1
            .insert_file("concurrent.md", "hash1", 1000, &[], "doc-1", None, None)
            .unwrap();

        // Read with store2 while store1 has been writing
        let record = store2.get_file("concurrent.md").unwrap();
        assert!(record.is_some());
        assert_eq!(record.unwrap().content_hash, "hash1");
    }

    #[test]
    fn test_insert_and_get_identity_facts() {
        let store = Store::open_memory().unwrap();
        store
            .upsert_identity_fact(0, "name", "Test User", None)
            .unwrap();
        store
            .upsert_identity_fact(1, "active_project", "Project A", Some("01-Projects/a.md"))
            .unwrap();
        store
            .upsert_identity_fact(1, "active_project", "Project B", Some("01-Projects/b.md"))
            .unwrap();

        let l0 = store.get_identity_facts(0).unwrap();
        assert_eq!(l0.len(), 1);
        assert_eq!(l0[0].key, "name");
        assert_eq!(l0[0].value, "Test User");

        let l1 = store.get_identity_facts(1).unwrap();
        assert_eq!(l1.len(), 2);
    }

    #[test]
    fn test_upsert_identity_fact_replaces() {
        let store = Store::open_memory().unwrap();
        store
            .upsert_identity_fact(0, "name", "Old Name", None)
            .unwrap();
        store
            .upsert_identity_fact(0, "name", "New Name", None)
            .unwrap();

        let facts = store.get_identity_facts(0).unwrap();
        assert_eq!(facts.len(), 2); // Different values = different rows
    }

    #[test]
    fn test_clear_identity_facts_by_tier() {
        let store = Store::open_memory().unwrap();
        store.upsert_identity_fact(0, "name", "User", None).unwrap();
        store
            .upsert_identity_fact(1, "active_project", "P1", None)
            .unwrap();
        store.clear_identity_facts(1).unwrap();

        assert_eq!(store.get_identity_facts(0).unwrap().len(), 1);
        assert_eq!(store.get_identity_facts(1).unwrap().len(), 0);
    }
}
