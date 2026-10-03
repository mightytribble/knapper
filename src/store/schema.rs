//! Every table the store holds, and how a store from an earlier knapper is brought up to them.

use super::DOC_LEVEL;
use super::Store;
use anyhow::Result;

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

/// The wikilink targets that resolved to no file, per source note (#98).
///
/// Keyed on `files(id)` and not on the source path, so it rides the same
/// cascade every other per-file table rides. A path key made this the one such
/// table `DELETE FROM files` could not reach, which left every removal path
/// owing a manual cleanup — one of six paid it, so a deleted note went on
/// reporting broken links from a file that was no longer there, and nothing
/// short of deleting the store could clear the row.
const UNRESOLVED_LINKS_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS unresolved_links (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    target     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(file_id, target)
);
CREATE INDEX IF NOT EXISTS idx_unresolved_file ON unresolved_links(file_id);";

/// Custom properties (#66). One row per value: a frontmatter row sits at
/// [`DOC_LEVEL`], a Dataview inline field at the chunk that holds it.
///
/// `target_file` is the note a `link` row resolves to. It is `SET NULL` and
/// not `CASCADE`: the property is a fact about the source note, and it
/// outlives the note it named. The source's own rows cascade off `files(id)`
/// the way `chunks`, `edges` and `file_tags` do, so no removal path owes
/// this table a cleanup.
const PROPERTIES_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS properties (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_seq   INTEGER NOT NULL DEFAULT -1,
    name        TEXT NOT NULL,
    value       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    target_file INTEGER REFERENCES files(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_properties_file   ON properties(file_id);
CREATE INDEX IF NOT EXISTS idx_properties_name   ON properties(name, value);
CREATE INDEX IF NOT EXISTS idx_properties_target ON properties(target_file);";

/// A note's aliases (#142): one row per alias, as `aliases::extract` reads
/// them from the note's frontmatter.
///
/// One table and not a vocabulary table beside a junction, as `tags` has. A
/// tag is a value many notes share. An alias names one note, and an alias two
/// notes carry is the case a lookup refuses. `folded` is the identity and
/// `display` is what the note wrote. `id` keeps the note's own order, because
/// the edge pass inserts a note's rows in the order the note lists them.
const ALIASES_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS aliases (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    folded  TEXT NOT NULL,
    display TEXT NOT NULL,
    UNIQUE(file_id, folded)
);
CREATE INDEX IF NOT EXISTS idx_aliases_folded ON aliases(folded);";

pub(super) const SCHEMA: &str = r#"
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
    docid        TEXT,
    -- The note's YAML block, as the note wrote it and with no `---` fences.
    -- The chunker strips it before it cuts a file into sections, so no chunk
    -- row holds it and `match` had no way to read it (#137).
    --
    -- Nullable, and with no default, because the two empty cases are not the
    -- same fact: `''` is a note that carries no frontmatter, NULL is a row
    -- written before this column existed. Reading them alike would answer a
    -- confident count over a half-read vault, which is the failure the
    -- capability exists to prevent.
    frontmatter  TEXT
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

impl Store {
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
    pub(super) fn migrate(&self) -> Result<()> {
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

        // Add files.frontmatter (#137). It stays NULL here on purpose: the
        // YAML block is in the vault and in no column this table holds, so
        // nothing can derive it without a file read. `LINK_RESOLVER_VERSION`
        // declares the `RebuildEdges` that fills it — a vault read and no
        // model — and `matching::run` refuses a frontmatter scan until it has.
        if !self.column_exists("files", "frontmatter")? {
            self.conn
                .execute_batch("ALTER TABLE files ADD COLUMN frontmatter TEXT;")?;
        }

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

        // Unresolved links table — the wikilink targets that resolved to no
        // file, per source note. Read by health analysis.
        if self.column_exists("unresolved_links", "source_file")? {
            // Rekey on `files(id)` (#98). The unique key changes, which
            // `ALTER TABLE` cannot do, so the table is rebuilt.
            //
            // A row carries across only where its `source_file` still names a
            // file the store holds. That is the whole repair: a row whose path
            // names nothing is a note that has already gone, and it is
            // precisely the row no code path could reach — so the rows dropped
            // here are exactly the ghosts a carried store was reporting, and
            // the rows kept are exactly the ones a re-index would rewrite.
            self.conn.execute_batch(&format!(
                "ALTER TABLE unresolved_links RENAME TO unresolved_links_pre98;
                 -- The old index followed the rename and still owns its name.
                 DROP INDEX IF EXISTS idx_unresolved_source;
                 {UNRESOLVED_LINKS_SCHEMA}
                 INSERT OR IGNORE INTO unresolved_links (file_id, target, created_at)
                     SELECT f.id, u.target, u.created_at
                     FROM unresolved_links_pre98 u
                     JOIN files f ON f.path = u.source_file;
                 DROP TABLE unresolved_links_pre98;"
            ))?;
        }
        self.conn.execute_batch(UNRESOLVED_LINKS_SCHEMA)?;

        // Custom properties (#66). Created empty; the edge pass fills it, and
        // the `LINK_RESOLVER_VERSION` bump that shipped with it declares the
        // rebuild that fills a store an earlier binary built.
        self.conn.execute_batch(PROPERTIES_SCHEMA)?;

        // Aliases (#142). Created empty; the edge pass fills it, and the
        // `LINK_RESOLVER_VERSION` bump that shipped with it declares the
        // rebuild that fills a store an earlier binary built.
        self.conn.execute_batch(ALIASES_SCHEMA)?;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A store written before #98 keys `unresolved_links` on the source
    /// **path**, so it holds rows for files that are no longer indexed and
    /// nothing can reach them. The migration rekeys the table on `files(id)`
    /// and carries across only the rows whose path still names a file — which
    /// leaves exactly the reachable ones and drops exactly the ghosts, so a
    /// carried store reports a clean `health` without being deleted (#98).
    #[test]
    fn the_migration_drops_the_unresolved_links_of_files_the_store_no_longer_holds() {
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("live.md", "hash", 100, "abc123", None, None)
            .unwrap();

        // Put the pre-#98 table back and fill it the way that build did.
        store
            .conn
            .execute_batch(
                "DROP TABLE unresolved_links;
                 CREATE TABLE unresolved_links (
                     id          INTEGER PRIMARY KEY,
                     source_file TEXT NOT NULL,
                     target      TEXT NOT NULL,
                     created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                     UNIQUE(source_file, target)
                 );
                 CREATE INDEX idx_unresolved_source ON unresolved_links(source_file);
                 INSERT INTO unresolved_links (source_file, target)
                     VALUES ('live.md', 'Nowhere'), ('deleted.md', 'Nowhere');",
            )
            .unwrap();

        store.migrate().unwrap();

        assert_eq!(
            store.get_unresolved_links().unwrap(),
            vec![("live.md".to_string(), "Nowhere".to_string())],
            "the ghost row goes and the reachable one carries across"
        );
        // And the carried row is keyed on the file, so the cascade reaches it.
        store.delete_file(file_id).unwrap();
        assert!(store.get_unresolved_links().unwrap().is_empty());
    }
}
