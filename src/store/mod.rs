//! SQLite persistence, one `Store` over one connection.
//!
//! Each file under this module is one table family: the types that describe
//! its rows and the `impl Store` block that reads and writes them. `schema.rs`
//! holds the versioned schema and the ladder that brings a store up to it.
//! Two declarations live outside it: `fts.rs` builds the keyword index from
//! `[fts]` and `vectors.rs` sizes the vector table to the model, and both are
//! reconciled on every open rather than by a step. Everything a caller names is re-exported here, so `crate::store::X`
//! is the path to every public item whichever file holds it.

mod aliases;
mod chunks;
mod edges;
mod files;
mod fts;
mod properties;
mod schema;
mod scope;
mod tags;
mod vectors;

pub(crate) use chunks::normalise_heading;
pub use chunks::{ChunkRecord, DOC_LEVEL, NewChunk};
pub use edges::EdgeStats;
pub use files::FileRecord;
pub use fts::{FtsResult, fts_objects_sql};
pub use properties::{NewProperty, PropertyCount, PropertyRow, ValueCount};
pub use schema::SCHEMA_VERSION;
pub use scope::{LinkIds, ListOrder, ListRow};
pub use tags::TagCount;

use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::path::Path;

/// Summary statistics for the store.
///
/// The edge counts are not here: [`EdgeStats`] is the one source for them, and
/// a copy on this struct made `status` run `get_edge_stats` twice and carry two
/// sources for the same numbers (#62).
#[derive(Debug)]
pub struct StoreStats {
    pub file_count: usize,
    pub chunk_count: usize,
    pub tombstone_count: usize,
    pub last_indexed_at: Option<String>,
    pub vault_path: Option<String>,
}

pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open a store backed by a file on disk.
    pub fn open(path: &Path) -> Result<Self> {
        crate::vecstore::init_sqlite_vec();
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;
        // The `rarray` table-valued function, which is how the search lanes
        // bind a tag scope: one pointer to a Vec<Value> rather than one bound
        // parameter per file id (#60). It is per-connection, so it is
        // registered where connections are made.
        rusqlite::vtab::array::load_module(&conn).context("registering rarray")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    /// Open an in-memory store (useful for tests).
    pub fn open_memory() -> Result<Self> {
        crate::vecstore::init_sqlite_vec();
        let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
        rusqlite::vtab::array::load_module(&conn).context("registering rarray")?;
        let store = Self { conn };
        store.init()?;
        Ok(store)
    }

    /// A second connection that reads and never writes.
    ///
    /// `serve` holds one of these beside its writer so a read does not wait
    /// for a search, a write or a re-index. It opens read-only, so SQLite
    /// refuses a write rather than convention. It runs no schema, migration or
    /// keyword-index reconciliation: the writer opened first and did that. WAL
    /// mode is persisted in the file, so this connection reads a committed
    /// snapshot while the writer is inside a transaction. On a filesystem
    /// where WAL is unavailable it is a rollback-journal reader, and a read
    /// can wait up to `busy_timeout` while a long write holds the file.
    pub fn open_reader(path: &Path) -> Result<Self> {
        use rusqlite::OpenFlags;
        crate::vecstore::init_sqlite_vec();
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI;
        let conn = Connection::open_with_flags(path, flags)
            .with_context(|| format!("failed to open reader at {}", path.display()))?;
        rusqlite::vtab::array::load_module(&conn).context("registering rarray")?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;")
            .context("failed to set the reader's busy_timeout")?;
        Ok(Self { conn })
    }

    fn init(&self) -> Result<()> {
        // Per-connection settings, so they are set on every open and not by a
        // schema step. WAL lets a reader read while the writer writes, and
        // `busy_timeout` makes a write wait for the lock rather than fail.
        // `foreign_keys` is what makes every ON DELETE CASCADE fire, and
        // SQLite ignores it inside a transaction, so it cannot sit in a step.
        self.conn
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA busy_timeout = 5000;
                 PRAGMA foreign_keys = ON;",
            )
            .context("failed to set the connection pragmas")?;
        self.upgrade_to_current()?;
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

    /// How many files the index holds.
    ///
    /// Disk truth, and the one question a `meta` key cannot answer for a store
    /// old enough to predate the key: an index built before a fingerprint
    /// existed still has its file rows (issue #141).
    pub fn file_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let file_count = self.file_count()?;
        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        let tombstone_count = self.tombstone_count()?;
        let last_indexed_at = self.get_meta("last_indexed_at")?;
        let vault_path = self.get_meta("vault_path")?;
        Ok(StoreStats {
            file_count,
            chunk_count: chunk_count as usize,
            tombstone_count,
            last_indexed_at,
            vault_path,
        })
    }

    /// Run `f` and keep what it wrote, or roll it back if `f` fails.
    ///
    /// The outermost call begins an immediate transaction and commits it. A
    /// call inside one opens a savepoint and releases it, so a failure inside
    /// rolls back only what the inner closure wrote and the outer transaction
    /// goes on. Every savepoint carries one name; SQLite resolves `RELEASE`
    /// and `ROLLBACK TO` to the innermost savepoint of that name, so nesting
    /// is a stack with no counter.
    ///
    /// The closure's error is the one returned. A rollback that itself fails
    /// is logged and does not replace it. A panic inside the closure rolls
    /// back on the way out, so the connection is never left inside a
    /// transaction nothing will commit.
    ///
    /// A caller that changes the disk after the call returns must not itself
    /// be inside a transaction: an inner release is final only when the
    /// outermost commit is, and no rollback can undo a rename.
    pub fn transaction<T>(&self, f: impl FnOnce(&Store) -> Result<T>) -> Result<T> {
        let nested = !self.conn.is_autocommit();
        let (begin, commit, rollback) = if nested {
            (
                "SAVEPOINT knapper",
                "RELEASE knapper",
                "ROLLBACK TO knapper; RELEASE knapper",
            )
        } else {
            ("BEGIN IMMEDIATE", "COMMIT", "ROLLBACK")
        };
        self.conn.execute_batch(begin)?;
        let mut open = OpenTransaction {
            conn: &self.conn,
            rollback,
            done: false,
        };
        match f(self) {
            Ok(value) => match self.conn.execute_batch(commit) {
                Ok(()) => {
                    open.done = true;
                    Ok(value)
                }
                Err(e) => {
                    open.finish();
                    Err(e.into())
                }
            },
            Err(e) => {
                open.finish();
                Err(e)
            }
        }
    }

    /// The connection, for a test that asserts on rows no method reads.
    #[cfg(test)]
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

/// The map the batched readers answer, grouped in the query's own order.
fn group_by_file<T>(
    rows: impl Iterator<Item = rusqlite::Result<(i64, T)>>,
) -> Result<std::collections::HashMap<i64, Vec<T>>> {
    let mut out: std::collections::HashMap<i64, Vec<T>> = std::collections::HashMap::new();
    for row in rows {
        let (file_id, item) = row?;
        out.entry(file_id).or_default().push(item);
    }
    Ok(out)
}

/// The file ids as one `rarray` argument, so a batched query binds one
/// pointer rather than one parameter per id.
fn id_array(file_ids: &[i64]) -> rusqlite::vtab::array::Array {
    std::rc::Rc::new(
        file_ids
            .iter()
            .copied()
            .map(rusqlite::types::Value::from)
            .collect(),
    )
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

/// A transaction or savepoint `Store::transaction` has begun and not yet
/// ended. Dropped before `done` is set, it rolls back: that is the path a
/// panic inside the closure takes.
struct OpenTransaction<'a> {
    conn: &'a Connection,
    rollback: &'static str,
    done: bool,
}

impl OpenTransaction<'_> {
    /// Roll back, unless SQLite already ended the transaction on its own.
    fn finish(&mut self) {
        if !self.conn.is_autocommit()
            && let Err(e) = self.conn.execute_batch(self.rollback)
        {
            tracing::warn!(error = %e, "rolling back a transaction");
        }
        self.done = true;
    }
}

impl Drop for OpenTransaction<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_panic_inside_the_closure_leaves_no_transaction_open() {
        let store = Store::open_memory().unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.transaction(|s| -> Result<()> {
                s.set_meta("k", "v")?;
                panic!("the closure panicked")
            })
        }));
        assert!(outcome.is_err());
        assert!(
            store.conn.is_autocommit(),
            "the guard rolled back on the way out"
        );
        assert_eq!(store.get_meta("k").unwrap(), None);
    }

    #[test]
    fn a_transaction_commits_what_its_closure_wrote() {
        let store = Store::open_memory().unwrap();
        store.transaction(|s| s.set_meta("k", "v")).unwrap();
        assert_eq!(store.get_meta("k").unwrap(), Some("v".into()));
        assert!(store.conn.is_autocommit(), "nothing is left open");
    }

    #[test]
    fn a_failing_closure_rolls_back_its_writes() {
        let store = Store::open_memory().unwrap();
        let err = store
            .transaction(|s| -> Result<()> {
                s.set_meta("k", "v")?;
                anyhow::bail!("no")
            })
            .unwrap_err();
        assert_eq!(err.to_string(), "no");
        assert_eq!(store.get_meta("k").unwrap(), None);
        assert!(store.conn.is_autocommit(), "nothing is left open");
    }

    #[test]
    fn a_nested_failure_rolls_back_only_the_inner_writes() {
        let store = Store::open_memory().unwrap();
        store
            .transaction(|s| {
                s.set_meta("outer", "1")?;
                let inner = s.transaction(|s| -> Result<()> {
                    s.set_meta("inner", "1")?;
                    anyhow::bail!("inner failed")
                });
                assert!(inner.is_err());
                assert!(
                    !s.conn.is_autocommit(),
                    "the outer transaction is still open after the inner one failed"
                );
                s.set_meta("after", "1")
            })
            .unwrap();
        assert_eq!(store.get_meta("outer").unwrap(), Some("1".into()));
        assert_eq!(store.get_meta("inner").unwrap(), None);
        assert_eq!(store.get_meta("after").unwrap(), Some("1".into()));
    }

    #[test]
    fn the_closures_error_is_the_one_returned() {
        use crate::fault::Fault;
        let store = Store::open_memory().unwrap();
        let err = store
            .transaction(|_| -> Result<()> {
                anyhow::bail!(Fault::Conflict("already there".into()))
            })
            .unwrap_err();
        assert!(
            matches!(Fault::of(&err), Some(Fault::Conflict(_))),
            "the kind survives the rollback: {err:#}"
        );
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

    /// A second connection reads what the first commits and cannot write
    /// (serve-core spec). WAL is a property of the file, so the reader sees a
    /// committed snapshot while the writer is inside a transaction.
    #[test]
    fn a_reader_sees_the_writers_commits_and_cannot_write() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = tmp.path().join("knapper.db");
        let writer = Store::open(&db).unwrap();
        let reader = Store::open_reader(&db).unwrap();

        writer
            .insert_file("a.md", "h", 1, "aaa111", None, None)
            .unwrap();
        assert_eq!(reader.file_count().unwrap(), 1);

        writer
            .transaction(|w| {
                w.insert_file("b.md", "h", 1, "bbb222", None, None)?;
                assert_eq!(
                    reader.file_count().unwrap(),
                    1,
                    "an uncommitted row is not visible"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(reader.file_count().unwrap(), 2);

        assert!(
            reader
                .insert_file("c.md", "h", 1, "ccc333", None, None)
                .is_err(),
            "SQLite enforces the read-only contract"
        );
    }

    #[test]
    fn foreign_keys_are_enforced_on_a_reopened_store() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = tmp.path().join("knapper.db");
        let file_id = {
            let store = Store::open(&db).unwrap();
            let id = store
                .insert_file("a.md", "h", 1, "aaa111", None, None)
                .unwrap();
            store
                .insert_chunk(&NewChunk {
                    file_id: id,
                    seq: 0,
                    heading: "H",
                    text: "text",
                    vector_id: 1,
                    token_count: 1,
                    ..Default::default()
                })
                .unwrap();
            id
        };
        let store = Store::open(&db).unwrap();
        store.delete_file(file_id).unwrap();
        assert_eq!(
            store.chunk_row_count().unwrap(),
            0,
            "the cascade fires on a connection that ran no schema step"
        );
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::{DOC_LEVEL, NewProperty, Store};
    use crate::docid::generate_docid;

    pub(super) fn file(store: &Store, path: &str) -> i64 {
        store
            .insert_file(path, "h", 100, &generate_docid(path), None, None)
            .unwrap()
    }

    pub(super) fn aliases(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    pub(super) fn prop<'a>(
        chunk_seq: i64,
        name: &'a str,
        value: &'a str,
        kind: crate::properties::Kind,
        target_file: Option<i64>,
    ) -> NewProperty<'a> {
        NewProperty {
            chunk_seq,
            name,
            value,
            kind,
            target_file,
        }
    }

    /// ada links to acme under `employer` (frontmatter) and to bob under
    /// `mentor` (body); bob links to acme with a plain body wikilink; acme
    /// carries `status: active`.
    pub(super) fn property_vault() -> (Store, i64, i64, i64) {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let ada = store
            .insert_file("ada.md", "h", 0, &generate_docid("ada.md"), None, None)
            .unwrap();
        let acme = store
            .insert_file("acme.md", "h", 0, &generate_docid("acme.md"), None, None)
            .unwrap();
        let bob = store
            .insert_file("bob.md", "h", 0, &generate_docid("bob.md"), None, None)
            .unwrap();
        store
            .replace_file_properties(
                ada,
                &[
                    prop(DOC_LEVEL, "employer", "acme", Kind::Link, Some(acme)),
                    prop(DOC_LEVEL, "status", "draft", Kind::Text, None),
                    prop(0, "mentor", "bob", Kind::Link, Some(bob)),
                ],
            )
            .unwrap();
        store
            .replace_file_properties(
                acme,
                &[prop(DOC_LEVEL, "status", "active", Kind::Text, None)],
            )
            .unwrap();
        store
            .insert_edge(ada, DOC_LEVEL, acme, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(ada, 0, bob, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(bob, 0, acme, DOC_LEVEL, "wikilink")
            .unwrap();
        (store, ada, acme, bob)
    }

    /// Three notes: a wight under `type/undead`, a wolf under `type/beast`,
    /// and a draft that is also `type/beast`.
    pub(super) fn operator_fixture() -> Store {
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

    pub(super) fn listed_paths(store: &Store, filter: &crate::tags::Scope) -> Vec<String> {
        let mut paths: Vec<String> = store
            .list_files(filter, None, Some(20))
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        paths.sort();
        paths
    }
}
