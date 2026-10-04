//! The `files` table: one row per note the index holds.

use super::Store;
use super::chrono_now;
use super::scope::scope_clauses;
use crate::fault::Fault;
use anyhow::Result;
use rusqlite::{OptionalExtension, params};

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
pub(super) const FILE_COLUMNS: &str = "f.id, f.path, f.content_hash, f.mtime, \
     (SELECT group_concat(display, char(31)) FROM \
        (SELECT t.display AS display FROM file_tags ft JOIN tags t ON t.id = ft.tag_id \
          WHERE ft.file_id = f.id ORDER BY t.path)), \
     f.indexed_at, f.docid, f.created_by, f.note_date";

pub(super) fn file_from_row(row: &rusqlite::Row) -> rusqlite::Result<FileRecord> {
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

impl Store {
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
                created_by   = COALESCE(excluded.created_by, files.created_by),
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

    /// Whether any note the scope admits predates `files.frontmatter` (#137).
    ///
    /// NULL is the pre-column state, and it is not `''`: a scan over it would
    /// answer a count of the wrong population without saying so, which is the
    /// one thing `match` must not do. Per-scope rather than per-store, so a
    /// scope of freshly indexed notes answers while the rest of the vault
    /// waits for its re-index.
    pub fn frontmatter_unindexed(&self, scope: &crate::tags::Scope) -> Result<bool> {
        let checked: Vec<&crate::tags::ScopeTerm> =
            scope.all.iter().chain(scope.any.iter()).collect();
        crate::tags::check_terms(&self.conn, &checked)?;
        let links = self.resolve_scope_links(scope)?;

        let (scope_sql, args) = scope_clauses(scope, &links);
        let mut stmt = self.conn.prepare(&format!(
            "SELECT EXISTS(SELECT 1 FROM files f WHERE f.frontmatter IS NULL{scope_sql})"
        ))?;
        let pending: bool =
            stmt.query_row(rusqlite::params_from_iter(args.iter()), |row| row.get(0))?;
        Ok(pending)
    }

    /// Record a note's YAML block, the text `match` scans beside its prose.
    ///
    /// Written by the pass that derives the rows read *out* of the block, so
    /// the raw text and the parsed properties are filled by one vault read
    /// and cannot disagree about which revision of the note they describe.
    pub fn set_file_frontmatter(&self, file_id: i64, frontmatter: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET frontmatter = ?2 WHERE id = ?1",
            params![file_id, frontmatter],
        )?;
        Ok(())
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

    /// Most recently indexed files.
    pub fn recent_files(&self, limit: usize) -> Result<Vec<FileRecord>> {
        // `mtime` and not `indexed_at` (#138): `indexed_at` is stamped when a
        // row is inserted, so `index --rebuild` reinserts every file in walk
        // order and the column collapses into that order. Presenting it as
        // recency answers the walk. `mtime` is the note's own, so it answers
        // what changed — as of the last index, which is all any column here
        // can answer. `f.path` is the tie-break, since a bulk-written vault
        // gives whole folders one mtime.
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS} FROM files f ORDER BY f.mtime DESC, f.path LIMIT ?"
        ))?;
        let rows = stmt.query_map(params![limit as i64], file_from_row)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
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
            anyhow::bail!(Fault::Conflict(format!(
                "target path already exists: {}",
                new_path
            )));
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

    /// `resolve_file`, with a miss as the caller's fault.
    ///
    /// The write tools address one note and refuse when it is absent; this
    /// is the one text and the one kind they answer with.
    pub fn require_file(&self, file_or_docid: &str) -> Result<FileRecord> {
        self.resolve_file(file_or_docid)?.ok_or_else(|| {
            anyhow::anyhow!(Fault::NotFound(format!("file not found: {file_or_docid}")))
        })
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

        // 4. Delete from files. The CASCADE off `files(id)` takes the chunks,
        //    the keyword index behind them, the `file_tags` rows and the
        //    file's `unresolved_links` — the last of those only since #98,
        //    when the table stopped keying on the source path. Nothing here
        //    needs a manual cleanup for them.
        self.delete_file(file_id)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::store::fixtures::*;
    use crate::store::*;

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
    fn recent_files_ranks_on_the_note_s_own_mtime() {
        // Insert order is walk order, which is what `index --rebuild` writes
        // into `indexed_at`. The edited note is the one inserted first, so a
        // sort on `indexed_at` puts it last and a sort on `mtime` puts it
        // first (#138).
        let store = Store::open_memory().unwrap();
        store
            .insert_file("edited.md", "h1", 900, "a1", None, None)
            .unwrap();
        store
            .insert_file("untouched.md", "h2", 100, "b2", None, None)
            .unwrap();
        // `insert_file` stamps `indexed_at` from the clock, and a rebuild
        // walks both files inside one second, so the two rows tie there and
        // the tie hides which column the sort reads. Oppose them outright.
        store
            .conn
            .execute(
                "UPDATE files SET indexed_at = '100' WHERE path = 'edited.md'",
                [],
            )
            .unwrap();
        store
            .conn
            .execute(
                "UPDATE files SET indexed_at = '900' WHERE path = 'untouched.md'",
                [],
            )
            .unwrap();

        let recent = store.recent_files(2).unwrap();
        assert_eq!(
            recent.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["edited.md", "untouched.md"]
        );
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

    //
    // `created_by` is provenance set once at creation. A reindex re-derives
    // it from the note's frontmatter, which is `None` for a note the write
    // pipeline made on this branch — `create` stamps no key of its own beyond
    // its resolved tags. The upsert must not let that `None` clear a value
    // the row already holds.

    /// The bug: a row that holds `created_by` keeps it when a later
    /// `insert_file` for the same path — a reindex reading no key from the
    /// note's frontmatter — passes `None`.
    #[test]
    fn a_stored_created_by_survives_a_later_insert_with_no_value() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/test.md");
        store
            .insert_file("notes/test.md", "hash1", 100, &docid, Some("cli"), None)
            .unwrap();

        // A reindex: same path, no `created_by` read from the frontmatter.
        store
            .insert_file("notes/test.md", "hash2", 200, &docid, None, None)
            .unwrap();

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.created_by, Some("cli".to_string()));
    }

    /// A later `insert_file` that does carry a value still overwrites the
    /// stored one, so a genuine change of agent is recorded rather than
    /// pinned forever by the fix above.
    #[test]
    fn a_later_insert_with_a_value_still_overwrites_the_stored_created_by() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/test.md");
        store
            .insert_file("notes/test.md", "hash1", 100, &docid, Some("cli"), None)
            .unwrap();

        store
            .insert_file(
                "notes/test.md",
                "hash2",
                200,
                &docid,
                Some("mcp-server"),
                None,
            )
            .unwrap();

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.created_by, Some("mcp-server".to_string()));
    }

    /// The `COALESCE` on `created_by` must not spread to the other upserted
    /// columns: `content_hash` and `mtime` still follow the second insert
    /// exactly, the way they did before this fix.
    #[test]
    fn the_other_upserted_columns_still_follow_a_later_insert() {
        let store = Store::open_memory().unwrap();
        let docid = generate_docid("notes/test.md");
        store
            .insert_file("notes/test.md", "hash1", 100, &docid, Some("cli"), None)
            .unwrap();

        store
            .insert_file("notes/test.md", "hash2", 200, &docid, None, None)
            .unwrap();

        let rec = store.get_file("notes/test.md").unwrap().unwrap();
        assert_eq!(rec.content_hash, "hash2");
        assert_eq!(rec.mtime, 200);
    }

    /// End to end: a note made through the write pipeline carries
    /// `created_by` immediately, and a later reindex — a watcher event or
    /// `knapper index` finding the file unchanged-but-rescanned — must not
    /// clear it. The note still answers a `list_files` query filtered on
    /// that `created_by`, which is the user-facing symptom (#92 follow-up).
    #[test]
    fn a_reindexed_note_still_matches_its_created_by_filter() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let store = Store::open_memory().unwrap();
        let mut embedder = crate::llm::MockLlm::new(32);
        let config = crate::config::Config::default();

        let written = crate::writer::create_note(
            crate::writer::CreateNoteInput {
                content: "Some content.\n".to_string(),
                filename: "provenance-check".into(),
                tags: vec![],
                folder: Some("notes".into()),
                created_by: "cli".into(),
                auto_link: Some(false),
            },
            &store,
            &mut embedder,
            crate::prefix::EmbedComposition::from_config(&config),
            config.chunk_options(),
            root,
            None,
        )
        .unwrap();

        let path = written.path.clone();
        let on_disk = std::fs::read_to_string(root.join(&path)).unwrap();

        // The note carries no `created_by:` key of its own (#92), so a
        // reindex over it — the watcher, or `knapper index` — extracts
        // `None` from its frontmatter.
        crate::indexer::index_file(
            &path,
            &on_disk,
            "a-different-hash",
            &store,
            &mut embedder,
            root,
            &config,
        )
        .unwrap();

        let rec = store.get_file(&path).unwrap().unwrap();
        assert_eq!(rec.created_by, Some("cli".to_string()));

        let scope = crate::tags::Scope::default();
        let filtered = store.list_files(&scope, Some("cli"), None).unwrap();
        assert!(
            filtered.iter().any(|f| f.path == path),
            "the created_by filter dropped the reindexed note: {filtered:#?}"
        );
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

    #[test]
    fn require_file_answers_not_found_with_the_name_the_caller_gave() {
        let store = Store::open_memory().unwrap();
        let err = store.require_file("nowhere.md").unwrap_err();
        assert_eq!(err.to_string(), "file not found: nowhere.md");
        assert_eq!(
            crate::fault::Fault::of(&err).map(|f| f.kind()),
            Some("not_found")
        );
    }

    #[test]
    fn the_write_resolver_does_not_read_aliases() {
        // `update`, `move`, `archive` and `delete` resolve through
        // `resolve_file`. A destructive call reached through a name the caller
        // did not know was an alias fails worse than `file not found`, so the
        // alias lookup belongs to the read side alone (#142).
        let store = Store::open_memory().unwrap();
        let id = store
            .insert_file("npcs/samantha-hoyle.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store
            .replace_file_aliases(id, &aliases(&["Empress"]))
            .unwrap();

        assert!(store.resolve_file("Empress").unwrap().is_none());
    }

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
}
