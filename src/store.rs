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

/// Columns selected for every [`FileRecord`], in the order [`file_from_row`]
/// expects. Every query using it must alias the table `f`.
///
/// `tags` is not a column of `files` (#60): the display forms come from the
/// join, ordered by path and separated by 0x1f, which a tag cannot hold — a tag
/// holds letters, digits, `_`, `-` and `/`. The inner SELECT carries the ORDER
/// BY, because the order of rows an aggregate reads is otherwise undefined.
const FILE_COLUMNS: &str = "f.id, f.path, f.content_hash, f.mtime, \
     (SELECT group_concat(display, char(31)) FROM \
        (SELECT t.display AS display FROM file_tags ft JOIN tags t ON t.id = ft.tag_id \
          WHERE ft.file_id = f.id ORDER BY t.path)), \
     f.indexed_at, f.docid, f.created_by, f.note_date";

fn file_from_row(row: &rusqlite::Row) -> rusqlite::Result<FileRecord> {
    Ok(FileRecord {
        id: row.get(0)?,
        path: row.get(1)?,
        content_hash: row.get(2)?,
        mtime: row.get(3)?,
        tags: row
            .get::<_, Option<String>>(4)?
            .map(|joined| joined.split('\u{1f}').map(str::to_string).collect())
            .unwrap_or_default(),
        indexed_at: row.get(5)?,
        docid: row.get(6)?,
        created_by: row.get(7)?,
        note_date: row.get(8)?,
    })
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

/// The `edges` table, as created fresh and as rebuilt by the #28 migration.
///
/// The unique key is the full chunk-to-chunk identity: one row per
/// (source passage, target passage, kind). A document's link set is exactly the
/// union of its chunks', so only the fine grain is stored and the coarse view is
/// derived — a stored copy of a derivable fact is a copy that can drift.
const EDGES_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS edges (
    id             INTEGER PRIMARY KEY,
    from_file      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    from_chunk_seq INTEGER NOT NULL DEFAULT -1,
    to_file        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    to_chunk_seq   INTEGER NOT NULL DEFAULT -1,
    edge_type      TEXT NOT NULL,
    UNIQUE(from_file, from_chunk_seq, to_file, to_chunk_seq, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_file, from_chunk_seq);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_file, to_chunk_seq);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);";

/// The tag store (#60). A tag is an attribute of a note, so `file_tags` is the
/// fact and every count over it is derived.
///
/// No `parent_id` and no `depth`: the path text holds the ancestors, a leaf row
/// has no parent to orphan, and a materialised ancestor would need a recursive
/// delete that leaves rows behind when it stops early.
const TAGS_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS tags (
    id      INTEGER PRIMARY KEY,
    path    TEXT NOT NULL UNIQUE,
    display TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS file_tags (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (file_id, tag_id)
);
CREATE INDEX IF NOT EXISTS file_tags_tag ON file_tags(tag_id);";

/// Reduce a heading to the form two spellings of the same section share.
///
/// Strips the leading `#`s a stored heading carries and a link's does not,
/// case-folds, and drops the `(cont.)` suffix the chunker appends when it splits
/// an oversized section — a `[[Note#Events]]` means `## Events (cont.)` too.
fn normalise_heading(heading: &str) -> String {
    heading
        .trim_start_matches('#')
        .trim()
        .trim_end_matches("(cont.)")
        .trim()
        .to_lowercase()
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

/// A single result from an FTS5 full-text search.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub file_id: i64,
    pub chunk_seq: i64,
    pub score: f64,
    pub snippet: String,
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
    -- scores. Since issue #37 it is also what `chunks_fts` indexes: the keyword
    -- index is external-content over this table and keeps no copy of its own.
    text        TEXT NOT NULL DEFAULT '',
    -- The two columns the keyword index reads beside the body (issue #37).
    -- `heading_path` is the breadcrumb, `Note Title > H1 > H2 > H3`; `tags_text`
    -- is the file's frontmatter tags, sorted and space separated. Both are
    -- written on every chunk whatever `[fts]` says, because the config decides
    -- which columns the index is declared over and not what a chunk records.
    heading_path TEXT NOT NULL DEFAULT '',
    tags_text    TEXT NOT NULL DEFAULT '',
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
        // The orchestrator's result cache. Nothing reads it since #59, and a
        // cache row has no expiry, so a store carried across the upgrade would
        // hold rows forever that describe a pipeline that no longer exists.
        self.conn.execute_batch("DROP TABLE IF EXISTS llm_cache;")?;
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

        // Add the two lexical columns (issue #37). They stay empty here on
        // purpose. `tags_text` could be derived from the tag store, but the
        // breadcrumb cannot be derived from anything this table holds — only
        // the leaf heading is stored, and the ancestors live in the vault. A
        // half-populated pair would index one limb of the rule and not the
        // other, so both wait for the re-index that `chunk_record` declares.
        if !self.column_exists("chunks", "heading_path")? {
            self.conn.execute_batch(
                "ALTER TABLE chunks ADD COLUMN heading_path TEXT NOT NULL DEFAULT '';
                 ALTER TABLE chunks ADD COLUMN tags_text TEXT NOT NULL DEFAULT '';",
            )?;
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
            self.conn.execute_batch(EDGES_SCHEMA)?;
        } else if !self.column_exists("edges", "from_chunk_seq")? {
            // Widen edges to chunk granularity (issue #28). The unique key gains
            // two columns, which `ALTER TABLE` cannot do, so the table is rebuilt.
            //
            // Existing rows carry across at [`DOC_LEVEL`] on both ends — the
            // grain they were written at, and a truthful statement of what the
            // old schema knew. That leaves the store behaving exactly as it did
            // before until something re-derives the fine grain from `chunks.text`;
            // `indexer::backfill_edges_from_chunks` is that something, and the
            // `edges_backfill_pending` flag is how it learns it has work.
            self.conn.execute_batch(&format!(
                "ALTER TABLE edges RENAME TO edges_pre28;
                 -- The old indexes followed the rename and still own their names,
                 -- so `CREATE INDEX IF NOT EXISTS` below would silently no-op and
                 -- leave the new table unindexed once `edges_pre28` is dropped.
                 DROP INDEX IF EXISTS idx_edges_from;
                 DROP INDEX IF EXISTS idx_edges_to;
                 DROP INDEX IF EXISTS idx_edges_type;
                 {EDGES_SCHEMA}
                 INSERT INTO edges (from_file, from_chunk_seq, to_file, to_chunk_seq, edge_type)
                     SELECT from_file, {DOC_LEVEL}, to_file, {DOC_LEVEL}, edge_type
                     FROM edges_pre28;
                 DROP TABLE edges_pre28;"
            ))?;
            self.set_meta("edges_backfill_pending", "1")?;
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

        // The tag store (#60). A tag is an attribute of a note, so `file_tags`
        // is the fact table and every count over it is derived: usage is
        // `COUNT(*)` and last use is `MAX(files.mtime)`. Both numbers come
        // from `file_tags`, so neither can drift from the vault.
        self.conn.execute_batch(TAGS_SCHEMA)?;

        // `tag_registry` held a flat vocabulary with no join to `files`. Its
        // `usage_count` counted index events, not files; `remove_file` never
        // touched it; and nothing reported the drift. Both numbers now come
        // from `file_tags`. Dropping the table needs no backfill: the
        // re-index that `PARSER_VERSION` declares rebuilds `tags` and
        // `file_tags` from the vault.
        self.conn
            .execute_batch("DROP TABLE IF EXISTS tag_registry;")?;

        // `files.tags` was a JSON copy of the same fact, and nothing kept the
        // two in step (#60). The display path joins `file_tags` and `tags`.
        if self.column_exists("files", "tags")? {
            self.conn
                .execute_batch("ALTER TABLE files DROP COLUMN tags;")?;
        }

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

    // ── Files ───────────────────────────────────────────────────

    pub fn insert_file(
        &self,
        path: &str,
        hash: &str,
        mtime: i64,
        docid: &str,
        created_by: Option<&str>,
        note_date: Option<i64>,
    ) -> Result<i64> {
        let now = chrono_now();
        self.conn.execute(
            "INSERT INTO files (path, content_hash, mtime, indexed_at, docid, created_by, note_date)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(path) DO UPDATE SET
                content_hash = excluded.content_hash,
                mtime        = excluded.mtime,
                indexed_at   = excluded.indexed_at,
                docid        = excluded.docid,
                created_by   = excluded.created_by,
                note_date    = excluded.note_date",
            params![path, hash, mtime, now, docid, created_by, note_date],
        )?;
        let file_id: i64 = self.conn.query_row(
            "SELECT id FROM files WHERE path = ?1",
            params![path],
            |row| row.get(0),
        )?;
        Ok(file_id)
    }

    pub fn get_file(&self, path: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f WHERE f.path = ?1"
        ))?;
        let record = stmt.query_row(params![path], file_from_row).optional()?;
        Ok(record)
    }

    pub fn get_all_files(&self) -> Result<Vec<FileRecord>> {
        let mut stmt = self
            .conn
            .prepare(&format!("SELECT {FILE_COLUMNS} FROM files f"))?;
        let rows = stmt.query_map([], file_from_row)?;
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
    /// not unique: the chunker splits an oversized section into `## Events` and
    /// `## Events (cont.)`, and a link to `#Events` means both.
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

    /// Insert a chunk-to-chunk edge. Uses INSERT OR IGNORE for the UNIQUE constraint.
    ///
    /// Pass [`DOC_LEVEL`] for an end that names no passage. The source end is
    /// the chunk whose text contained the link; the target end is the chunk a
    /// `#Heading` resolved to.
    pub fn insert_edge(
        &self,
        from_file: i64,
        from_chunk_seq: i64,
        to_file: i64,
        to_chunk_seq: i64,
        edge_type: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (from_file, from_chunk_seq, to_file, to_chunk_seq, edge_type)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![from_file, from_chunk_seq, to_file, to_chunk_seq, edge_type],
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

    /// The document-level view of the wikilink graph: distinct `(from, to)` pairs.
    ///
    /// A document's link set is the union of its chunks', so this is derived
    /// rather than stored (issue #28) — a stored copy could drift from the rows
    /// it summarises, and this cannot.
    pub fn wikilink_pairs(&self) -> Result<Vec<(i64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT from_file, to_file FROM edges WHERE edge_type = 'wikilink'",
        )?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut pairs = Vec::new();
        for row in rows {
            pairs.push(row?);
        }
        Ok(pairs)
    }

    /// Whether this store's edges are still at the pre-#28 document grain.
    ///
    /// Set by the migration that widened the table and cleared by
    /// `indexer::backfill_edges_from_chunks`. Until then the store is correct,
    /// just coarse: every edge reads as document-to-document, which is what it
    /// meant when it was written.
    pub fn needs_edge_backfill(&self) -> Result<bool> {
        Ok(self.get_meta("edges_backfill_pending")?.as_deref() == Some("1"))
    }

    /// Clear all edges (used during --rebuild).
    pub fn clear_edges(&self) -> Result<()> {
        self.conn.execute("DELETE FROM edges", [])?;
        Ok(())
    }

    /// Get outgoing edges at document granularity, optionally filtered by type.
    ///
    /// `DISTINCT` because the stored grain is chunk-to-chunk (issue #28): a note
    /// linked from four passages is four rows and one relationship, and every
    /// caller of this wants the relationship.
    pub fn get_outgoing(
        &self,
        file_id: i64,
        edge_type: Option<&str>,
    ) -> Result<Vec<(i64, String)>> {
        let mut results = Vec::new();
        match edge_type {
            Some(et) => {
                let mut stmt = self.conn.prepare(
                    "SELECT DISTINCT to_file, edge_type FROM edges WHERE from_file = ?1 AND edge_type = ?2",
                )?;
                let rows = stmt.query_map(params![file_id, et], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT DISTINCT to_file, edge_type FROM edges WHERE from_file = ?1",
                )?;
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

    /// Get incoming edges at document granularity, optionally filtered by type.
    ///
    /// `DISTINCT` for the reason given on [`get_outgoing`](Self::get_outgoing).
    pub fn get_incoming(
        &self,
        file_id: i64,
        edge_type: Option<&str>,
    ) -> Result<Vec<(i64, String)>> {
        let mut results = Vec::new();
        match edge_type {
            Some(et) => {
                let mut stmt = self.conn.prepare(
                    "SELECT DISTINCT from_file, edge_type FROM edges WHERE to_file = ?1 AND edge_type = ?2",
                )?;
                let rows = stmt.query_map(params![file_id, et], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })?;
                for row in rows {
                    results.push(row?);
                }
            }
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT DISTINCT from_file, edge_type FROM edges WHERE to_file = ?1",
                )?;
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
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f WHERE f.id = ?1"
        ))?;
        let record = stmt.query_row(params![file_id], file_from_row).optional()?;
        Ok(record)
    }

    /// Look up a file by its 6-character docid.
    pub fn get_file_by_docid(&self, docid: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f WHERE f.docid = ?1"
        ))?;
        let record = stmt.query_row(params![docid], file_from_row).optional()?;
        Ok(record)
    }

    // ── FTS5 ──────────────────────────────────────────────────

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
        self.fts_search_expr(&crate::fts::phrase_expr(query), limit, &[])
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
    pub fn fts_search_any(
        &self,
        query: &str,
        limit: usize,
        weights: &[f64],
    ) -> Result<Vec<FtsResult>> {
        match crate::fts::any_term_expr(query) {
            Some(expr) => self.fts_search_expr(&expr, limit, weights),
            None => Ok(Vec::new()),
        }
    }

    /// Run a prepared FTS5 MATCH expression. Callers build the expression with
    /// `crate::fts`, which is where the quoting rules and their reasons live.
    ///
    /// `file_id` and `chunk_seq` come from a join and not from the index: since
    /// issue #37 `chunks_fts` is external content over `chunks`, and it is
    /// keyed on the chunk's rowid rather than carrying a copy of the pair.
    fn fts_search_expr(
        &self,
        fts_query: &str,
        limit: usize,
        weights: &[f64],
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
        let mut stmt = self.conn.prepare(&format!(
            "SELECT c.file_id, c.seq, {bm25} as score,
                    snippet(chunks_fts, 0, '<b>', '</b>', '...', 64)
             FROM chunks_fts
             JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?1
             ORDER BY score
             LIMIT ?2",
        ))?;

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

    /// Every wikilink edge touching any of `file_ids`, oriented near-end-first.
    ///
    /// Returns `(near_file, near_seq, far_file, far_seq)`. Wikilinks are walked
    /// in both directions — a knowledge-graph neighbour is related whichever way
    /// the link runs — so each arm of the union puts the end *nearest* the file
    /// asked about first. That is the end which has to match the passage in
    /// hand; the far end is where the walk lands.
    ///
    /// One indexed fetch for a whole frontier, which is what replaced the
    /// per-seed BFS in issue #29: the walk is arithmetic over this list, done in
    /// Rust, rather than two queries per node visited.
    pub fn incident_wikilink_edges(&self, file_ids: &[i64]) -> Result<Vec<(i64, i64, i64, i64)>> {
        if file_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ph = vec!["?"; file_ids.len()].join(",");
        let sql = format!(
            "SELECT from_file, from_chunk_seq, to_file, to_chunk_seq FROM edges
             WHERE edge_type = 'wikilink' AND from_file IN ({ph})
             UNION ALL
             SELECT to_file, to_chunk_seq, from_file, from_chunk_seq FROM edges
             WHERE edge_type = 'wikilink' AND to_file IN ({ph})"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let bound: Vec<Box<dyn rusqlite::types::ToSql>> = file_ids
            .iter()
            .chain(file_ids.iter())
            .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let rows = stmt.query_map(rusqlite::params_from_iter(bound.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        let mut edges = Vec::new();
        for row in rows {
            edges.push(row?);
        }
        Ok(edges)
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

    /// List files filtered by folder prefix, tag operators and creator.
    pub fn list_files(
        &self,
        folder: Option<&str>,
        tags: &crate::tags::TagFilter,
        created_by: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FileRecord>> {
        // `none` is not checked: excluding a tag no note carries is a no-op.
        let checked: Vec<&crate::tags::TagTerm> = tags.all.iter().chain(tags.any.iter()).collect();
        crate::tags::check_terms(&self.conn, &checked)?;

        let mut sql = format!("SELECT {FILE_COLUMNS} FROM files f WHERE 1=1");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(folder) = folder {
            sql.push_str(" AND f.path LIKE ?");
            param_values.push(Box::new(format!("{}%", folder)));
        }
        // The junction, not `json_each` over a JSON column: the old test
        // scanned `files` and parsed JSON for each row (#60). A term folds its
        // own path, so a folded query side meets a folded column.
        for term in &tags.all {
            let (pred, args) = crate::tags::predicate(term);
            sql.push_str(&format!(
                " AND EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id
                                WHERE ft.file_id = f.id AND {pred})"
            ));
            for arg in args {
                param_values.push(Box::new(arg));
            }
        }
        for (field, keyword) in [(&tags.any, "EXISTS"), (&tags.none, "NOT EXISTS")] {
            if field.is_empty() {
                continue;
            }
            let mut ors: Vec<String> = Vec::new();
            for term in field {
                let (pred, args) = crate::tags::predicate(term);
                ors.push(pred);
                for arg in args {
                    param_values.push(Box::new(arg));
                }
            }
            sql.push_str(&format!(
                " AND {keyword} (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id
                                   WHERE ft.file_id = f.id AND ({}))",
                ors.join(" OR ")
            ));
        }
        if let Some(cb) = created_by {
            sql.push_str(" AND f.created_by = ?");
            param_values.push(Box::new(cb.to_string()));
        }
        sql.push_str(" ORDER BY f.indexed_at DESC LIMIT ?");
        param_values.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(
            rusqlite::params_from_iter(param_values.iter()),
            file_from_row,
        )?;
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

    /// Tag frequency: how many notes carry each tag (#60).
    pub fn top_tags(&self, limit: usize) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.display, COUNT(*) AS cnt
               FROM tags t JOIN file_tags ft ON ft.tag_id = t.id
              GROUP BY t.id ORDER BY cnt DESC, t.path LIMIT ?",
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
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f ORDER BY f.indexed_at DESC LIMIT ?"
        ))?;
        let rows = stmt.query_map(params![limit as i64], file_from_row)?;
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
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f WHERE f.path LIKE ?1"
        ))?;
        let rows = stmt.query_map(params![pattern], file_from_row)?;
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
            let mut stmt = self.conn.prepare(&format!(
                "SELECT {FILE_COLUMNS}
                 FROM files f
                 WHERE lower(f.path) LIKE '%/' || lower(?1) OR lower(f.path) = lower(?1)
                 ORDER BY length(f.path) ASC LIMIT 1"
            ))?;
            let record = stmt
                .query_row(params![candidate], file_from_row)
                .optional()?;
            if let Some(record) = record {
                return Ok(Some(record));
            }
        }

        Ok(None)
    }

    /// Query files whose note_date falls within a given range (inclusive).
    pub fn get_files_in_date_range(&self, start: i64, end: i64) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS}
             FROM files f WHERE f.note_date BETWEEN ?1 AND ?2
             ORDER BY f.note_date ASC"
        ))?;
        let rows = stmt.query_map(params![start, end], file_from_row)?;
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
        // The keyword index follows the chunks (issue #37).
        self.conn.execute("DELETE FROM chunks", [])?;
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

    /// The tag ids a file currently holds.
    pub fn file_tag_ids(&self, file_id: i64) -> Result<Vec<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT tag_id FROM file_tags WHERE file_id = ?1")?;
        let rows = stmt.query_map(params![file_id], |row| row.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Make `tags` the tags of this file, in three steps (#60).
    ///
    /// Read the ids the file holds, replace its rows, then delete each id it
    /// released that no other file holds. Step three reads only the released
    /// ids: a scan of the whole table costs a full pass for each file and finds
    /// the same rows.
    ///
    /// The caller owns the file's tag rows the way `index_file` owns its
    /// chunks, its vectors and its outgoing edges.
    ///
    /// The store folds what it is given: `tags.path` is the identity and
    /// Obsidian matches a tag without regard to case, so the path is written
    /// folded here rather than trusted from the caller. `Type/Undead` and
    /// `type/undead` are one row whichever spelling arrives, and every query
    /// that folds its own argument — `files_with_tag`, `files_under_tag`, the
    /// `list_files` tag filter — meets a folded column.
    pub fn reconcile_file_tags(&self, file_id: i64, tags: &[crate::tags::Tag]) -> Result<()> {
        let released = self.file_tag_ids(file_id)?;
        self.conn
            .execute("DELETE FROM file_tags WHERE file_id = ?1", params![file_id])?;
        for tag in tags {
            let path = tag.path.to_lowercase();
            // The first spelling indexed supplies `display`.
            self.conn.execute(
                "INSERT INTO tags (path, display) VALUES (?1, ?2) ON CONFLICT(path) DO NOTHING",
                params![path, tag.display],
            )?;
            let tag_id: i64 = self.conn.query_row(
                "SELECT id FROM tags WHERE path = ?1",
                params![path],
                |row| row.get(0),
            )?;
            // A tag written in both the property and the body writes one row.
            self.conn.execute(
                "INSERT OR IGNORE INTO file_tags (file_id, tag_id) VALUES (?1, ?2)",
                params![file_id, tag_id],
            )?;
        }
        self.prune_unused_tags(&released)
    }

    /// Delete each released id that now has no row in `file_tags`.
    ///
    /// The counterpart of [`reconcile_file_tags`](Self::reconcile_file_tags)
    /// for a path that removes the links itself — `remove_file` cascades them
    /// off `files(id)` and then calls this with the ids the file held.
    pub fn prune_unused_tags(&self, released: &[i64]) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "DELETE FROM tags WHERE id = ?1
               AND NOT EXISTS (SELECT 1 FROM file_tags WHERE tag_id = ?1)",
        )?;
        for id in released {
            stmt.execute(params![id])?;
        }
        Ok(())
    }

    /// A file's tags as the vault spelled them, ordered by path.
    pub fn file_tags(&self, file_id: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.display FROM file_tags ft
               JOIN tags t ON t.id = ft.tag_id
              WHERE ft.file_id = ?1 ORDER BY t.path",
        )?;
        let rows = stmt.query_map(params![file_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// The notes tagged exactly `path`.
    ///
    /// The joins are aliased `ftag` and `tg`, because `FILE_COLUMNS` carries a
    /// subquery of its own that uses `ft` and `t`.
    pub fn files_with_tag(&self, path: &str) -> Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f
               JOIN file_tags ftag ON ftag.file_id = f.id
               JOIN tags tg ON tg.id = ftag.tag_id
              WHERE tg.path = ?1 ORDER BY f.path"
        ))?;
        let rows = stmt.query_map(params![path.to_lowercase()], file_from_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// The notes Obsidian's `tag:<path>` returns: the tag and every descendant.
    ///
    /// Left-anchored, so the unique index on `path` serves it. The descendant
    /// arm is a range, not a `LIKE` pattern: `_` is a legal tag-path character
    /// and is also `LIKE`'s single-character wildcard, and an `ESCAPE` clause
    /// would turn off SQLite's `LIKE` optimisation. `?3` is `?2` with the
    /// slash's next ASCII character in place of the slash, so the range holds
    /// every path that starts with `<path>/` and nothing else. `DISTINCT`
    /// because one note may carry several descendants of one tag.
    ///
    /// The joins are aliased `ftag` and `tg`, because `FILE_COLUMNS` carries a
    /// subquery of its own that uses `ft` and `t`.
    pub fn files_under_tag(&self, path: &str) -> Result<Vec<FileRecord>> {
        let folded = path.to_lowercase();
        let mut stmt = self.conn.prepare(&format!(
            "SELECT DISTINCT {FILE_COLUMNS} FROM files f
               JOIN file_tags ftag ON ftag.file_id = f.id
               JOIN tags tg ON tg.id = ftag.tag_id
              WHERE tg.path = ?1 OR (tg.path >= ?2 AND tg.path < ?3) ORDER BY f.path"
        ))?;
        let rows = stmt.query_map(
            params![folded, format!("{folded}/"), format!("{folded}0")],
            file_from_row,
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// The axes this vault holds, and how many notes each covers (#60).
    ///
    /// engraph names no axis. This reports what the vault wrote: the first
    /// segment of every path, counting each note once however many tags of that
    /// axis it carries.
    pub fn tag_axes(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self.conn.prepare(
            "SELECT substr(t.path, 1, COALESCE(NULLIF(instr(t.path, '/'), 0) - 1, length(t.path))) AS axis,
                    COUNT(DISTINCT ft.file_id) AS notes
               FROM tags t JOIN file_tags ft ON ft.tag_id = t.id
              GROUP BY axis ORDER BY notes DESC, axis",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
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
    /// 3. Delete from `edges` where from_file or to_file matches
    /// 4. Delete from `files` (CASCADE handles chunks, and the chunks carry
    ///    the keyword index with them — see [`fts_objects_sql`])
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

        // 3. Delete from edges (both directions)
        self.delete_edges_for_file(file_id)?;

        // 4. Delete from files (CASCADE handles chunks table)
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

    // ── The keyword index as external content (#37) ──────────────

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
        let docid = generate_docid("notes/test.md");
        let file_id = store
            .insert_file("notes/test.md", "abc123", 1700000000, &docid, None, None)
            .unwrap();
        assert!(file_id > 0);
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        store
            .reconcile_file_tags(file_id, &[tag("programming"), tag("rust")])
            .unwrap();

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.path, "notes/test.md");
        assert_eq!(rec.content_hash, "abc123");
        assert_eq!(rec.mtime, 1700000000);
        assert_eq!(rec.tags, store.file_tags(file_id).unwrap());
        assert_eq!(rec.tags, vec!["programming", "rust"]);
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
    fn test_delete_file_cascades_chunks() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file(
                "notes/del.md",
                "hash",
                100,
                &generate_docid("notes/del.md"),
                None,
                None,
            )
            .unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 0,
                heading: "H",
                text: "snippet",
                vector_id: 10,
                token_count: 5,
                ..Default::default()
            })
            .unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 1,
                heading: "H2",
                text: "snippet2",
                vector_id: 11,
                token_count: 6,
                ..Default::default()
            })
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
            .insert_file("notes/change.md", "old_hash", 100, &docid, None, None)
            .unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 0,
                heading: "H",
                text: "text",
                vector_id: 50,
                token_count: 10,
                ..Default::default()
            })
            .unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 1,
                heading: "H2",
                text: "text2",
                vector_id: 51,
                token_count: 12,
                ..Default::default()
            })
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
            .insert_file("notes/change.md", "new_hash", 200, &docid, None, None)
            .unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id: new_file_id,
                seq: 0,
                heading: "H",
                text: "new text",
                vector_id: 60,
                token_count: 15,
                ..Default::default()
            })
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
            .insert_file("notes/findme.md", "hash", 100, &docid, None, None)
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

        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();

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

        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(b, DOC_LEVEL, a, DOC_LEVEL, "wikilink")
            .unwrap();

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
        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();

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
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        // a -> b, c -> a
        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(c, DOC_LEVEL, a, DOC_LEVEL, "mention")
            .unwrap();

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
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        // a -> b, b -> c
        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(b, DOC_LEVEL, c, DOC_LEVEL, "mention")
            .unwrap();

        // Delete file b — CASCADE should remove both edges.
        store.delete_file(b).unwrap();

        assert!(store.get_outgoing(a, None).unwrap().is_empty());
        assert!(store.get_incoming(c, None).unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_edge_ignored() {
        let store = Store::open_memory().unwrap();
        let (a, b) = setup_two_files(&store);

        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap(); // duplicate

        let out = store.get_outgoing(a, None).unwrap();
        assert_eq!(out.len(), 1);

        // Same pair with different type is NOT a duplicate.
        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "mention")
            .unwrap();
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
                &generate_docid("notes/c.md"),
                None,
                None,
            )
            .unwrap();

        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(a, DOC_LEVEL, c, DOC_LEVEL, "mention")
            .unwrap();

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
            .insert_file("n/f1.md", "h1", 100, &generate_docid("n/f1.md"), None, None)
            .unwrap();
        let f2 = store
            .insert_file("n/f2.md", "h2", 100, &generate_docid("n/f2.md"), None, None)
            .unwrap();
        let f3 = store
            .insert_file("n/f3.md", "h3", 100, &generate_docid("n/f3.md"), None, None)
            .unwrap();

        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f1, DOC_LEVEL, f3, DOC_LEVEL, "wikilink")
            .unwrap();

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
            .insert_file("n/f1.md", "h1", 100, &generate_docid("n/f1.md"), None, None)
            .unwrap();
        let f2 = store
            .insert_file("n/f2.md", "h2", 100, &generate_docid("n/f2.md"), None, None)
            .unwrap();
        let f3 = store
            .insert_file("n/f3.md", "h3", 100, &generate_docid("n/f3.md"), None, None)
            .unwrap();
        let f4 = store
            .insert_file("n/f4.md", "h4", 100, &generate_docid("n/f4.md"), None, None)
            .unwrap();

        // f1 -> f2 -> f3 -> f4
        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f2, DOC_LEVEL, f3, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f3, DOC_LEVEL, f4, DOC_LEVEL, "wikilink")
            .unwrap();

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
            .insert_file("n/f1.md", "h1", 100, &generate_docid("n/f1.md"), None, None)
            .unwrap();
        let f2 = store
            .insert_file("n/f2.md", "h2", 100, &generate_docid("n/f2.md"), None, None)
            .unwrap();

        // f2 links to f1; f1 has no outgoing links of its own.
        store
            .insert_edge(f2, DOC_LEVEL, f1, DOC_LEVEL, "wikilink")
            .unwrap();

        // Neighbor discovery is undirected: f1's neighbors include its
        // backlink f2 even though f1 has no outgoing edge.
        let neighbors = store.get_neighbors(f1, 1).unwrap();
        let ids: Vec<i64> = neighbors.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![f2]);
        assert_eq!(neighbors[0].1, 1);
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
    fn test_get_edge_stats() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("n/a.md", "ha", 100, &generate_docid("n/a.md"), None, None)
            .unwrap();
        let b = store
            .insert_file("n/b.md", "hb", 100, &generate_docid("n/b.md"), None, None)
            .unwrap();
        let c = store
            .insert_file("n/c.md", "hc", 100, &generate_docid("n/c.md"), None, None)
            .unwrap();
        // d is isolated (no edges).
        let _d = store
            .insert_file("n/d.md", "hd", 100, &generate_docid("n/d.md"), None, None)
            .unwrap();

        store
            .insert_edge(a, DOC_LEVEL, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(a, DOC_LEVEL, c, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(b, DOC_LEVEL, c, DOC_LEVEL, "mention")
            .unwrap();

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
            .insert_file("01-Projects/a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        store
            .insert_file("02-Areas/b.md", "h2", 200, "bbb222", None, None)
            .unwrap();
        store
            .insert_file("01-Projects/c.md", "h3", 300, "ccc333", None, None)
            .unwrap();
        let files = store
            .list_files(None, &crate::tags::TagFilter::default(), None, 20)
            .unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_list_files_folder_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        store
            .insert_file("02-Areas/b.md", "h2", 200, "bbb222", None, None)
            .unwrap();
        let files = store
            .list_files(
                Some("01-Projects"),
                &crate::tags::TagFilter::default(),
                None,
                20,
            )
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "01-Projects/a.md");
    }

    #[test]
    fn test_list_files_tag_filter() {
        let store = Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        let a = store
            .insert_file("a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h2", 200, "bbb222", None, None)
            .unwrap();
        let c = store
            .insert_file("c.md", "h3", 300, "ccc333", None, None)
            .unwrap();
        store
            .reconcile_file_tags(a, &[tag("cli"), tag("rust")])
            .unwrap();
        store.reconcile_file_tags(b, &[tag("rust")]).unwrap();
        store.reconcile_file_tags(c, &[tag("python")]).unwrap();
        let files = store
            .list_files(
                None,
                &crate::tags::TagFilter::parse(&["rust".to_string()], &[], &[]),
                None,
                20,
            )
            .unwrap();
        assert_eq!(files.len(), 2);
        let files = store
            .list_files(
                None,
                &crate::tags::TagFilter::parse(&["rust".to_string(), "cli".to_string()], &[], &[]),
                None,
                20,
            )
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "a.md");
    }

    #[test]
    fn test_list_files_created_by_filter() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("a.md", "h1", 100, "aaa111", Some("cli"), None)
            .unwrap();
        store
            .insert_file("b.md", "h2", 200, "bbb222", Some("mcp"), None)
            .unwrap();
        store
            .insert_file("c.md", "h3", 300, "ccc333", None, None)
            .unwrap();

        // Filter by "cli" → only the cli-created file
        let files = store
            .list_files(None, &crate::tags::TagFilter::default(), Some("cli"), 20)
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "a.md");
        assert_eq!(files[0].created_by, Some("cli".to_string()));

        // Filter by "mcp" → only the mcp-created file
        let files = store
            .list_files(None, &crate::tags::TagFilter::default(), Some("mcp"), 20)
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "b.md");

        // Filter by None → all 3
        let files = store
            .list_files(None, &crate::tags::TagFilter::default(), None, 20)
            .unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_folder_note_counts() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/a.md", "h1", 100, "a1", None, None)
            .unwrap();
        store
            .insert_file("01-Projects/b.md", "h2", 100, "b2", None, None)
            .unwrap();
        store
            .insert_file("02-Areas/c.md", "h3", 100, "c3", None, None)
            .unwrap();
        store
            .insert_file("root.md", "h4", 100, "d4", None, None)
            .unwrap();
        let counts = store.folder_note_counts().unwrap();
        assert!(counts.iter().any(|(f, c)| f == "01-Projects" && *c == 2));
        assert!(counts.iter().any(|(f, c)| f == "02-Areas" && *c == 1));
        assert!(counts.iter().any(|(f, c)| f == "(root)" && *c == 1));
    }

    #[test]
    fn test_top_tags() {
        let store = Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        let a = store
            .insert_file("a.md", "h1", 100, "a1", None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h2", 100, "b2", None, None)
            .unwrap();
        let c = store
            .insert_file("c.md", "h3", 100, "c3", None, None)
            .unwrap();
        store
            .reconcile_file_tags(a, &[tag("cli"), tag("rust")])
            .unwrap();
        store
            .reconcile_file_tags(b, &[tag("rust"), tag("web")])
            .unwrap();
        store.reconcile_file_tags(c, &[tag("rust")]).unwrap();
        let tags = store.top_tags(10).unwrap();
        assert_eq!(tags[0].0, "rust");
        assert_eq!(tags[0].1, 3);
    }

    #[test]
    fn test_recent_files() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("old.md", "h1", 100, "a1", None, None)
            .unwrap();
        store
            .insert_file("new.md", "h2", 200, "b2", None, None)
            .unwrap();
        let recent = store.recent_files(1).unwrap();
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_edge_count_for_file() {
        let store = Store::open_memory().unwrap();
        let f1 = store
            .insert_file("a.md", "h1", 100, "a1", None, None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, "b2", None, None)
            .unwrap();
        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f2, DOC_LEVEL, f1, DOC_LEVEL, "wikilink")
            .unwrap();
        assert_eq!(store.edge_count_for_file(f1).unwrap(), 2);
        assert_eq!(store.edge_count_for_file(f2).unwrap(), 2);
    }

    #[test]
    fn test_find_file_by_basename() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("01-Projects/Work/note.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        store
            .insert_file("root.md", "h2", 100, "bbb222", None, None)
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
            .insert_file("a.md", "h1", 100, "a1", None, None)
            .unwrap();
        let f2 = store
            .insert_file("b.md", "h2", 100, "b2", None, None)
            .unwrap();
        let f3 = store
            .insert_file("c.md", "h3", 100, "c3", None, None)
            .unwrap();
        store
            .insert_edge(f1, DOC_LEVEL, f2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f2, DOC_LEVEL, f1, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(f1, DOC_LEVEL, f3, DOC_LEVEL, "wikilink")
            .unwrap();
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
            .insert_file("test.md", "hash123", 0, "abc123", None, None)
            .unwrap();
        let vector: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
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
            .insert_file("notes/test.md", "hash1", 100, &docid, Some("cli"), None)
            .unwrap();
        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.created_by, Some("cli".to_string()));
    }

    #[test]
    fn test_insert_file_without_created_by() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/test.md");
        store
            .insert_file("notes/test.md", "hash1", 100, &docid, None, None)
            .unwrap();
        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.created_by, None);
    }

    #[test]
    fn test_update_file_path() {
        let store = Store::open_memory().unwrap();
        let old_docid = generate_docid("notes/old.md");
        let file_id = store
            .insert_file("notes/old.md", "hash1", 100, &old_docid, None, None)
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
                &generate_docid("notes/vec.md"),
                None,
                None,
            )
            .unwrap();

        let v1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v2: Vec<f32> = vec![4.0, 5.0, 6.0];
        store
            .insert_chunk_with_vector(
                &NewChunk {
                    file_id,
                    seq: 0,
                    heading: "H1",
                    text: "text1",
                    vector_id: 100,
                    token_count: 10,
                    ..Default::default()
                },
                &v1,
            )
            .unwrap();
        store
            .insert_chunk_with_vector(
                &NewChunk {
                    file_id,
                    seq: 0,
                    heading: "H2",
                    text: "text2",
                    vector_id: 101,
                    token_count: 10,
                    ..Default::default()
                },
                &v2,
            )
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

    /// A store carried across #59 loses the orchestrator's cache table.
    ///
    /// The rows have no expiry and describe a pipeline that no longer exists,
    /// so leaving them would be dead weight that outlives every binary.
    #[test]
    fn migrating_drops_the_orchestrator_cache() {
        let store = Store::open_memory().unwrap();
        store
            .conn
            .execute_batch("CREATE TABLE llm_cache (query_hash TEXT PRIMARY KEY);")
            .unwrap();
        store.migrate().unwrap();
        let present: i64 = store
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name = 'llm_cache'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(present, 0);
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
            .insert_file("Steve Barbera.md", "hash1", 100, "ab1234", None, None)
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
            .insert_file("test-a.md", "h1", 100, "aaa111", None, None)
            .unwrap();
        store
            .insert_file("test-b.md", "h2", 100, "bbb222", None, None)
            .unwrap();
        // "test-c" is equidistant from both — should error, not pick arbitrarily
        let result = store.resolve_file("test-c");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_file_existing_docid() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("note.md", "hash", 100, "abc123", None, None)
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
        let file_id = store
            .insert_file("delete-me.md", "hash", 100, "del123", None, None)
            .unwrap();

        // Insert a chunk + vec entry for the file. The keyword index follows
        // the chunk row (issue #37), so there is no third insert.
        let vid = store.next_vector_id().unwrap();
        store
            .insert_chunk(&NewChunk {
                file_id,
                seq: 0,
                heading: "## Heading",
                text: "chunk text",
                vector_id: vid,
                token_count: 10,
                ..Default::default()
            })
            .unwrap();

        // Insert an embedding vector into chunks_vec
        let embedding = vec![0.1_f32; 256];
        store.insert_vec(vid, &embedding).unwrap();

        // Insert an edge from this file to itself (just to test edge cleanup)
        let file_id2 = store
            .insert_file("other.md", "hash2", 100, "oth123", None, None)
            .unwrap();
        store
            .insert_edge(file_id, DOC_LEVEL, file_id2, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(file_id2, DOC_LEVEL, file_id, DOC_LEVEL, "wikilink")
            .unwrap();

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
            .insert_file("dated.md", "hash", 100, "dat123", None, note_date)
            .unwrap();
        let file = store.get_file("dated.md").unwrap().unwrap();
        assert_eq!(file.note_date, note_date);
    }

    #[test]
    fn test_insert_file_without_note_date() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("undated.md", "hash", 100, "und123", None, None)
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
            .insert_file("a.md", "h1", 100, "aaa111", None, Some(day1))
            .unwrap();
        store
            .insert_file("b.md", "h2", 100, "bbb222", None, Some(day2))
            .unwrap();
        store
            .insert_file("c.md", "h3", 100, "ccc333", None, Some(day3))
            .unwrap();
        store
            .insert_file("d.md", "h4", 100, "ddd444", None, None)
            .unwrap();
        let results = store.get_files_in_date_range(day1, day2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_count_files_with_dates() {
        let store = Store::open_memory().unwrap();
        let day1 = 1774000000i64;
        store
            .insert_file("a.md", "h1", 100, "aaa111", None, Some(day1))
            .unwrap();
        store
            .insert_file("b.md", "h2", 100, "bbb222", None, None)
            .unwrap();
        store
            .insert_file("c.md", "h3", 100, "ccc333", None, Some(day1 + 86400))
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
            .insert_file("concurrent.md", "hash1", 1000, "doc-1", None, None)
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

    // ── Chunk-granular edges (#28) ───────────────────────────────

    fn file(store: &Store, path: &str) -> i64 {
        store
            .insert_file(path, "h", 100, &generate_docid(path), None, None)
            .unwrap()
    }

    #[test]
    fn an_incident_edge_is_returned_from_whichever_end_was_asked_for() {
        // Both arms of the union orient the *near* end first, so one edge is two
        // rows when both its files are in the frontier — that is what makes the
        // walk undirected without storing a reverse edge.
        let store = Store::open_memory().unwrap();
        let a = file(&store, "a.md");
        let b = file(&store, "b.md");
        let c = file(&store, "c.md");
        store.insert_edge(a, 0, b, 4, "wikilink").unwrap();
        store.insert_edge(a, 1, c, DOC_LEVEL, "mention").unwrap();

        assert_eq!(
            store.incident_wikilink_edges(&[a]).unwrap(),
            vec![(a, 0, b, 4)],
            "the mention edge is not part of the walk"
        );
        assert_eq!(
            store.incident_wikilink_edges(&[b]).unwrap(),
            vec![(b, 4, a, 0)],
            "asked from b, the near end is b's own passage"
        );
        let mut both = store.incident_wikilink_edges(&[a, b]).unwrap();
        both.sort();
        assert_eq!(both, vec![(a, 0, b, 4), (b, 4, a, 0)]);
        assert!(store.incident_wikilink_edges(&[]).unwrap().is_empty());
    }

    #[test]
    fn a_documents_chunk_seqs_are_what_a_doc_level_link_resolves_to() {
        let store = Store::open_memory().unwrap();
        let a = file(&store, "a.md");
        let b = file(&store, "b.md");
        for seq in [0, 1, 2] {
            store
                .insert_chunk(&NewChunk {
                    file_id: a,
                    seq,
                    heading: "## H",
                    text: "text",
                    vector_id: seq as u64,
                    token_count: 10,
                    ..Default::default()
                })
                .unwrap();
        }
        store
            .insert_chunk(&NewChunk {
                file_id: b,
                seq: 0,
                heading: "## H",
                text: "text",
                vector_id: 9,
                token_count: 10,
                ..Default::default()
            })
            .unwrap();

        let seqs = store.chunk_seqs_for_files(&[a, b]).unwrap();
        assert_eq!(seqs[&a], vec![0, 1, 2]);
        assert_eq!(seqs[&b], vec![0]);
        // A file with no chunks is absent from the map rather than present and
        // empty — the caller has to decide what a link into it means.
        let unchunked = file(&store, "unchunked.md");
        assert!(
            store
                .chunk_seqs_for_files(&[unchunked])
                .unwrap()
                .get(&unchunked)
                .is_none()
        );
    }

    #[test]
    fn the_two_ends_of_an_edge_are_independent() {
        // The unique key is the full chunk-to-chunk identity, so the same pair
        // of files can be joined by several distinct passages.
        let store = Store::open_memory().unwrap();
        let a = file(&store, "a.md");
        let b = file(&store, "b.md");
        for (from, to) in [(0, 2), (0, 5), (1, 2)] {
            store.insert_edge(a, from, b, to, "wikilink").unwrap();
        }
        store.insert_edge(a, 0, b, 2, "wikilink").unwrap(); // duplicate

        let count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 3, "INSERT OR IGNORE must still dedupe");
        assert_eq!(store.wikilink_pairs().unwrap(), vec![(a, b)]);
        assert_eq!(
            store.get_outgoing(a, Some("wikilink")).unwrap().len(),
            1,
            "the document-level view collapses them back to one relationship"
        );
    }

    #[test]
    fn chunk_seqs_with_heading_finds_a_split_section() {
        // `(file, heading)` is not unique: an oversized section becomes
        // `## Events` and `## Events (cont.)`, and a link to `#Events` means both.
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
    fn a_pre_28_store_keeps_its_edges_at_the_document_level() {
        // The migration rebuilds the table to widen the unique key, which is the
        // one operation that could silently lose the whole graph.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("engraph.db");
        let (a, b) = {
            let store = Store::open(&path).unwrap();
            let a = file(&store, "a.md");
            let b = file(&store, "b.md");
            (a, b)
        };

        // Put the old schema back, rows and all.
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(&format!(
                "DROP TABLE edges;
                 CREATE TABLE edges (
                     id INTEGER PRIMARY KEY,
                     from_file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                     to_file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                     edge_type TEXT NOT NULL,
                     UNIQUE(from_file, to_file, edge_type)
                 );
                 CREATE INDEX idx_edges_from ON edges(from_file);
                 CREATE INDEX idx_edges_to ON edges(to_file);
                 CREATE INDEX idx_edges_type ON edges(edge_type);
                 INSERT INTO edges (from_file, to_file, edge_type)
                     VALUES ({a}, {b}, 'wikilink'), ({b}, {a}, 'mention');"
            ))
            .unwrap();
        }

        let store = Store::open(&path).unwrap();
        assert!(
            store.needs_edge_backfill().unwrap(),
            "the store should know its edges are still coarse"
        );
        assert_eq!(store.wikilink_pairs().unwrap(), vec![(a, b)]);
        assert_eq!(
            store.get_incoming(a, Some("mention")).unwrap(),
            vec![(b, "mention".to_string())]
        );
        let seqs: Vec<(i64, i64)> = store
            .conn()
            .prepare("SELECT from_chunk_seq, to_chunk_seq FROM edges")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(seqs, vec![(DOC_LEVEL, DOC_LEVEL); 2]);

        // The rebuilt table has to keep its indexes: the old ones followed the
        // rename and would have been dropped along with the old table.
        let indexes: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='edges'
                 AND name LIKE 'idx_edges_%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(indexes, 3);
    }

    // ── The tag store ────────────────────────────────────────────

    fn tag_fixture() -> (Store, i64, i64) {
        let store = Store::open_memory().unwrap();
        let one = store
            .insert_file("one.md", "h1", 1, "d000001", None, None)
            .unwrap();
        let two = store
            .insert_file("two.md", "h2", 2, "d000002", None, None)
            .unwrap();
        (store, one, two)
    }

    fn tag_row_count(store: &Store) -> i64 {
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM tags", [], |row| row.get(0))
            .unwrap()
    }

    fn link_count(store: &Store) -> i64 {
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM file_tags", [], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn two_spellings_of_one_tag_are_one_row() {
        let (store, one, two) = tag_fixture();
        store
            .reconcile_file_tags(
                one,
                &[crate::tags::Tag {
                    path: "type/undead".into(),
                    display: "Type/Undead".into(),
                }],
            )
            .unwrap();
        store
            .reconcile_file_tags(
                two,
                &[crate::tags::Tag {
                    path: "type/undead".into(),
                    display: "type/undead".into(),
                }],
            )
            .unwrap();

        assert_eq!(tag_row_count(&store), 1);
        assert_eq!(link_count(&store), 2);
        let display: String = store
            .conn()
            .query_row("SELECT display FROM tags", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            display, "Type/Undead",
            "the first spelling supplies display"
        );
    }

    #[test]
    fn a_note_carrying_one_tag_twice_holds_one_link() {
        let (store, one, _) = tag_fixture();
        let tag = crate::tags::Tag {
            path: "habitat/swamp".into(),
            display: "habitat/swamp".into(),
        };
        store.reconcile_file_tags(one, &[tag.clone(), tag]).unwrap();
        assert_eq!(link_count(&store), 1);
    }

    #[test]
    fn dropping_the_last_use_of_a_tag_deletes_its_row() {
        let (store, one, _) = tag_fixture();
        let swamp = crate::tags::Tag {
            path: "habitat/swamp".into(),
            display: "habitat/swamp".into(),
        };
        store.reconcile_file_tags(one, &[swamp]).unwrap();
        assert_eq!(tag_row_count(&store), 1);

        store.reconcile_file_tags(one, &[]).unwrap();
        assert_eq!(tag_row_count(&store), 0);
        assert_eq!(link_count(&store), 0);
    }

    #[test]
    fn a_tag_two_notes_carry_survives_one_of_them_dropping_it() {
        let (store, one, two) = tag_fixture();
        let swamp = crate::tags::Tag {
            path: "habitat/swamp".into(),
            display: "habitat/swamp".into(),
        };
        store.reconcile_file_tags(one, &[swamp.clone()]).unwrap();
        store.reconcile_file_tags(two, &[swamp]).unwrap();

        store.reconcile_file_tags(one, &[]).unwrap();
        assert_eq!(tag_row_count(&store), 1);
        assert_eq!(link_count(&store), 1);
    }

    #[test]
    fn deleting_a_file_cascades_its_links_away() {
        let (store, one, _) = tag_fixture();
        let swamp = crate::tags::Tag {
            path: "habitat/swamp".into(),
            display: "habitat/swamp".into(),
        };
        store.reconcile_file_tags(one, &[swamp]).unwrap();
        let released = store.file_tag_ids(one).unwrap();
        assert_eq!(released.len(), 1);

        store.delete_file(one).unwrap();
        assert_eq!(link_count(&store), 0);
        // The junction cascades; the vocabulary row is the caller's step 3.
        assert_eq!(tag_row_count(&store), 1);
        store.prune_unused_tags(&released).unwrap();
        assert_eq!(tag_row_count(&store), 0);
    }

    /// The reconciler folds the path it is given, so the folding contract is
    /// an invariant of the store and not of the caller (#60).
    #[test]
    fn the_reconciler_folds_the_path_it_is_given() {
        let (store, one, two) = tag_fixture();
        store
            .reconcile_file_tags(
                one,
                &[crate::tags::Tag {
                    path: "Type/Undead".into(),
                    display: "Type/Undead".into(),
                }],
            )
            .unwrap();
        store
            .reconcile_file_tags(
                two,
                &[crate::tags::Tag {
                    path: "type/undead".into(),
                    display: "type/undead".into(),
                }],
            )
            .unwrap();

        // One row, so the two spellings are one tag and one axis value.
        assert_eq!(tag_row_count(&store), 1);
        assert_eq!(store.tag_axes().unwrap(), vec![("type".to_string(), 2)]);
        // And the folded query side meets a folded column.
        assert_eq!(store.files_with_tag("Type/Undead").unwrap().len(), 2);
        assert_eq!(store.files_under_tag("type").unwrap().len(), 2);
        assert_eq!(
            store
                .list_files(
                    None,
                    &crate::tags::TagFilter::parse(&["TYPE/UNDEAD".to_string()], &[], &[]),
                    None,
                    10,
                )
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn a_files_display_tags_come_back_in_path_order() {
        let (store, one, _) = tag_fixture();
        store
            .reconcile_file_tags(
                one,
                &[
                    crate::tags::Tag {
                        path: "zebra".into(),
                        display: "Zebra".into(),
                    },
                    crate::tags::Tag {
                        path: "apex".into(),
                        display: "Apex".into(),
                    },
                ],
            )
            .unwrap();
        assert_eq!(store.file_tags(one).unwrap(), vec!["Apex", "Zebra"]);
    }

    #[test]
    fn a_file_record_reads_its_tags_from_the_join() {
        let store = Store::open_memory().unwrap();
        let id = store
            .insert_file("n.md", "h", 1, "d000001", None, None)
            .unwrap();
        store
            .reconcile_file_tags(
                id,
                &[
                    crate::tags::Tag {
                        path: "zebra".into(),
                        display: "Zebra".into(),
                    },
                    crate::tags::Tag {
                        path: "apex".into(),
                        display: "Apex".into(),
                    },
                ],
            )
            .unwrap();

        let record = store.get_file("n.md").unwrap().unwrap();
        assert_eq!(record.tags, vec!["Apex", "Zebra"]);
        assert_eq!(
            store.get_all_files().unwrap()[0].tags,
            vec!["Apex", "Zebra"]
        );
        assert!(
            store
                .get_file_by_docid("d000001")
                .unwrap()
                .unwrap()
                .tags
                .len()
                == 2
        );
    }

    #[test]
    fn the_tag_filter_keeps_and_semantics() {
        let store = Store::open_memory().unwrap();
        let both = store
            .insert_file("both.md", "h", 1, "d000001", None, None)
            .unwrap();
        let one = store
            .insert_file("one.md", "h", 2, "d000002", None, None)
            .unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        store
            .reconcile_file_tags(both, &[tag("alpha"), tag("beta")])
            .unwrap();
        store.reconcile_file_tags(one, &[tag("alpha")]).unwrap();

        let hits = store
            .list_files(
                None,
                &crate::tags::TagFilter::parse(
                    &["alpha".to_string(), "beta".to_string()],
                    &[],
                    &[],
                ),
                None,
                10,
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "both.md");

        // Obsidian matches a tag without regard to case.
        let folded = store
            .list_files(
                None,
                &crate::tags::TagFilter::parse(&["ALPHA".to_string()], &[], &[]),
                None,
                10,
            )
            .unwrap();
        assert_eq!(folded.len(), 2);
    }

    /// Three notes: a wight under `type/undead`, a wolf under `type/beast`,
    /// and a draft that is also `type/beast`.
    fn operator_fixture() -> Store {
        let store = Store::open_memory().unwrap();
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
        let draft = store
            .insert_file("draft.md", "h", 3, "d000003", None, None)
            .unwrap();
        store
            .reconcile_file_tags(wight, &[tag("type/undead"), tag("habitat/swamp")])
            .unwrap();
        store
            .reconcile_file_tags(wolf, &[tag("type/beast")])
            .unwrap();
        store
            .reconcile_file_tags(draft, &[tag("type/beast"), tag("status/draft")])
            .unwrap();
        store
    }

    fn listed_paths(store: &Store, filter: &crate::tags::TagFilter) -> Vec<String> {
        let mut paths: Vec<String> = store
            .list_files(None, filter, None, 20)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn an_unknown_all_term_errors_and_names_the_nearest_tag() {
        let store = operator_fixture();
        let filter = crate::tags::TagFilter::parse(&["type/undeed".to_string()], &[], &[]);
        let err = store.list_files(None, &filter, None, 20).unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such tag 'type/undeed'; nearest: 'type/undead'"
        );
    }

    #[test]
    fn an_unknown_term_with_no_near_neighbour_errors_without_a_suggestion() {
        let store = operator_fixture();
        let filter = crate::tags::TagFilter::parse(&[], &["zzzzz".to_string()], &[]);
        let err = store.list_files(None, &filter, None, 20).unwrap_err();
        assert_eq!(err.to_string(), "no such tag 'zzzzz'");
    }

    #[test]
    fn an_unknown_subtree_term_prints_its_marker() {
        let store = operator_fixture();
        let filter = crate::tags::TagFilter::parse(&["nowhere/".to_string()], &[], &[]);
        let err = store.list_files(None, &filter, None, 20).unwrap_err();
        assert_eq!(err.to_string(), "no such tag 'nowhere/'");
    }

    #[test]
    fn an_unknown_none_term_is_not_an_error() {
        let store = operator_fixture();
        let filter =
            crate::tags::TagFilter::parse(&["type/".to_string()], &[], &["nowhere".to_string()]);
        assert_eq!(
            listed_paths(&store, &filter),
            vec!["draft.md", "wight.md", "wolf.md"]
        );
    }

    #[test]
    fn an_exact_term_errors_on_a_bare_axis_and_a_subtree_term_matches_below_it() {
        let store = operator_fixture();
        // The fixture holds `type/undead` and `type/beast`, never bare `type`.
        let exact = crate::tags::TagFilter::parse(&["type".to_string()], &[], &[]);
        let err = store.list_files(None, &exact, None, 20).unwrap_err();
        assert_eq!(err.to_string(), "no such tag 'type'");

        let subtree = crate::tags::TagFilter::parse(&["type/".to_string()], &[], &[]);
        assert_eq!(
            listed_paths(&store, &subtree),
            vec!["draft.md", "wight.md", "wolf.md"]
        );
    }

    #[test]
    fn all_terms_intersect_and_any_terms_union() {
        let store = operator_fixture();
        let all = crate::tags::TagFilter::parse(
            &["type/undead".to_string(), "habitat/swamp".to_string()],
            &[],
            &[],
        );
        assert_eq!(listed_paths(&store, &all), vec!["wight.md"]);

        let any = crate::tags::TagFilter::parse(
            &[],
            &["type/undead".to_string(), "status/draft".to_string()],
            &[],
        );
        assert_eq!(listed_paths(&store, &any), vec!["draft.md", "wight.md"]);
    }

    #[test]
    fn a_none_term_removes_a_note_the_other_fields_returned() {
        let store = operator_fixture();
        let filter = crate::tags::TagFilter::parse(
            &["type/".to_string()],
            &[],
            &["status/draft".to_string()],
        );
        assert_eq!(listed_paths(&store, &filter), vec!["wight.md", "wolf.md"]);
    }

    #[test]
    fn the_three_fields_combine_in_one_query() {
        let store = operator_fixture();
        let filter = crate::tags::TagFilter::parse(
            &["type/".to_string()],
            &["habitat/swamp".to_string(), "status/draft".to_string()],
            &["status/draft".to_string()],
        );
        assert_eq!(listed_paths(&store, &filter), vec!["wight.md"]);
    }

    #[test]
    fn a_subtree_term_stops_at_the_segment_boundary() {
        let store = Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        let inside = store
            .insert_file("inside.md", "h", 1, "d000001", None, None)
            .unwrap();
        let beside = store
            .insert_file("beside.md", "h", 2, "d000002", None, None)
            .unwrap();
        store
            .reconcile_file_tags(inside, &[tag("type/undead")])
            .unwrap();
        // `type_a` sorts after `type/` and must not fall inside the range.
        store.reconcile_file_tags(beside, &[tag("type_a")]).unwrap();
        let filter = crate::tags::TagFilter::parse(&["type/".to_string()], &[], &[]);
        assert_eq!(listed_paths(&store, &filter), vec!["inside.md"]);
    }

    #[test]
    fn an_empty_filter_returns_every_note() {
        let store = operator_fixture();
        let filter = crate::tags::TagFilter::default();
        assert_eq!(listed_paths(&store, &filter).len(), 3);
    }

    #[test]
    fn top_tags_counts_notes() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 1, "d000001", None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 2, "d000002", None, None)
            .unwrap();
        let tag = |p: &str, d: &str| crate::tags::Tag {
            path: p.into(),
            display: d.into(),
        };
        store
            .reconcile_file_tags(a, &[tag("shared", "Shared"), tag("solo", "solo")])
            .unwrap();
        store
            .reconcile_file_tags(b, &[tag("shared", "shared")])
            .unwrap();

        let top = store.top_tags(10).unwrap();
        assert_eq!(top[0], ("Shared".to_string(), 2));
    }

    fn axis_fixture() -> Store {
        let store = Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        let undead = store
            .insert_file("undead.md", "h", 1, "d000001", None, None)
            .unwrap();
        let beast = store
            .insert_file("beast.md", "h", 2, "d000002", None, None)
            .unwrap();
        let plain = store
            .insert_file("plain.md", "h", 3, "d000003", None, None)
            .unwrap();
        // A note carrying two tags of one axis, to prove the axis counts notes.
        store
            .reconcile_file_tags(
                undead,
                &[tag("type/undead"), tag("type/wight"), tag("habitat/swamp")],
            )
            .unwrap();
        store
            .reconcile_file_tags(beast, &[tag("type/beast")])
            .unwrap();
        store.reconcile_file_tags(plain, &[tag("type")]).unwrap();
        store
    }

    #[test]
    fn the_descendant_query_returns_what_obsidians_tag_search_returns() {
        let store = axis_fixture();
        // `tag:type` matches `type` and every descendant of it.
        let mut paths: Vec<String> = store
            .files_under_tag("type")
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        paths.sort();
        assert_eq!(paths, ["beast.md", "plain.md", "undead.md"]);

        // The exact query is the tag a note carries and no descendant.
        let exact: Vec<String> = store
            .files_with_tag("type")
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(exact, ["plain.md"]);
    }

    #[test]
    fn an_underscore_in_a_tag_path_is_not_a_wildcard() {
        let store = Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        let one = store
            .insert_file("one.md", "h", 1, "d000001", None, None)
            .unwrap();
        let two = store
            .insert_file("two.md", "h", 2, "d000002", None, None)
            .unwrap();
        let exact_note = store
            .insert_file("exact.md", "h", 3, "d000003", None, None)
            .unwrap();
        // `_` is a legal tag-path character and also `LIKE`'s single-character
        // wildcard. A `LIKE` pattern `type_a/%` would also match `typeXa/two`.
        store
            .reconcile_file_tags(one, &[tag("type_a/one")])
            .unwrap();
        store
            .reconcile_file_tags(two, &[tag("typeXa/two")])
            .unwrap();
        store
            .reconcile_file_tags(exact_note, &[tag("type_a")])
            .unwrap();

        let mut paths: Vec<String> = store
            .files_under_tag("type_a")
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        paths.sort();
        assert_eq!(paths, ["exact.md", "one.md"]);

        // The exact arm still answers for the tag itself.
        let exact: Vec<String> = store
            .files_with_tag("type_a")
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(exact, ["exact.md"]);
    }

    #[test]
    fn an_axis_counts_each_note_once() {
        let store = axis_fixture();
        let axes = store.tag_axes().unwrap();
        assert_eq!(
            axes[0],
            ("type".to_string(), 3),
            "undead.md carries two of them"
        );
        assert!(axes.contains(&("habitat".to_string(), 1)));
    }
}
