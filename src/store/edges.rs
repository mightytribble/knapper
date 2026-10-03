//! The `edges` and `unresolved_links` tables: the links the vault's notes write.

use super::FileRecord;
use super::Store;
use anyhow::Result;
use rusqlite::params;
use std::collections::HashSet;

/// Statistics about edges in the graph.
#[derive(Debug)]
pub struct EdgeStats {
    pub total_edges: usize,
    pub wikilink_count: usize,
    pub connected_file_count: usize,
    pub isolated_file_count: usize,
}

impl Store {
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
            connected_file_count: connected as usize,
            isolated_file_count: (total_files - connected) as usize,
        })
    }

    /// The notes the vault points at most, each with how many distinct notes
    /// link to it (#138).
    ///
    /// One grouped pass over `idx_edges_to`, not `list_files_with_links_in`'s
    /// per-file subquery: that call ranks on a computed column, so its `LIMIT`
    /// stops no work — every row in `files` is counted before ten survive, and
    /// `chunk_count` and `token_count` are counted beside it. `vault_map` asks
    /// this of the whole vault on every call, so it takes the plan that reads
    /// the edge index once.
    ///
    /// It counts what `list --sort links_in` counts, distinct linking notes,
    /// so a note that links to another four times is one link in and the two
    /// surfaces cannot disagree about the same vault.
    ///
    /// A note nothing links to is absent rather than zero: `edges` holds no
    /// row for it, and it is not an answer to which notes the vault points at.
    pub fn top_linked_files(&self, limit: usize) -> Result<Vec<(String, usize)>> {
        // `f.path` is the tie-break, so two equally linked notes come back in
        // the same order every call.
        let mut stmt = self.conn.prepare(
            "SELECT f.path, COUNT(DISTINCT e.from_file) AS links_in
               FROM edges e JOIN files f ON f.id = e.to_file
              GROUP BY e.to_file ORDER BY links_in DESC, f.path LIMIT ?",
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

    /// Record a wikilink target that could not be resolved during indexing.
    ///
    /// The source is the file's id, so the row goes when the file's row goes.
    pub fn insert_unresolved_link(&self, file_id: i64, target: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO unresolved_links (file_id, target) VALUES (?1, ?2)",
            params![file_id, target],
        )?;
        Ok(())
    }

    /// Remove all unresolved links originating from the given file.
    ///
    /// For a re-index, which rewrites what the file's own text says. A removal
    /// needs no call: the cascade off `files(id)` is what takes those rows.
    pub fn clear_unresolved_links_for_file(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM unresolved_links WHERE file_id = ?1",
            params![file_id],
        )?;
        Ok(())
    }

    /// Return all unresolved links as (source path, target) pairs.
    ///
    /// The path comes from the join rather than from the row, so a reported
    /// source is by construction a path the index holds — and a note that
    /// moved is reported where it is now, not where it was (#98).
    pub fn get_unresolved_links(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path, u.target FROM unresolved_links u \
             JOIN files f ON f.id = u.file_id ORDER BY f.path, u.target",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Every unresolved link as (source file id, target).
    ///
    /// The re-resolution form of the rows [`get_unresolved_links`] reports
    /// (#108). The caller re-derives the source's links, which is keyed on the
    /// id, and it resolves the target itself rather than reading a path.
    ///
    /// [`get_unresolved_links`]: Self::get_unresolved_links
    pub fn unresolved_link_sources(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT file_id, target FROM unresolved_links ORDER BY file_id, target")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docid::generate_docid;
    use crate::store::fixtures::*;
    use crate::store::*;

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
        assert_eq!(stats.connected_file_count, 3); // a, b, c
        assert_eq!(stats.isolated_file_count, 1); // d
    }

    #[test]
    fn top_linked_files_ranks_the_notes_the_vault_points_at() {
        let store = Store::open_memory().unwrap();
        let hub = store
            .insert_file("hub.md", "h", 1, "d000001", None, None)
            .unwrap();
        let mid = store
            .insert_file("mid.md", "h", 1, "d000002", None, None)
            .unwrap();
        let leaf = store
            .insert_file("leaf.md", "h", 1, "d000003", None, None)
            .unwrap();
        store.insert_edge(leaf, 0, hub, -1, "wikilink").unwrap();
        store.insert_edge(mid, 0, hub, -1, "wikilink").unwrap();
        store.insert_edge(leaf, 0, mid, -1, "wikilink").unwrap();

        let top = store.top_linked_files(10).unwrap();
        assert_eq!(
            top,
            vec![("hub.md".to_string(), 2), ("mid.md".to_string(), 1)]
        );
    }

    #[test]
    fn a_note_nothing_links_to_is_not_a_hub() {
        // The ranking answers which notes the vault points at. A note with no
        // inbound edge is not an answer to that, however short the list runs.
        let store = Store::open_memory().unwrap();
        let hub = store
            .insert_file("hub.md", "h", 1, "d000001", None, None)
            .unwrap();
        let leaf = store
            .insert_file("leaf.md", "h", 1, "d000002", None, None)
            .unwrap();
        store.insert_edge(leaf, 0, hub, -1, "wikilink").unwrap();

        let top = store.top_linked_files(10).unwrap();
        assert_eq!(top, vec![("hub.md".to_string(), 1)]);
    }

    #[test]
    fn one_note_linking_twice_is_one_link_in() {
        // `list --sort links_in` counts distinct linking notes, and the map's
        // ranking has to agree with it or the two disagree on the same vault.
        let store = Store::open_memory().unwrap();
        let hub = store
            .insert_file("hub.md", "h", 1, "d000001", None, None)
            .unwrap();
        let leaf = store
            .insert_file("leaf.md", "h", 1, "d000002", None, None)
            .unwrap();
        store.insert_edge(leaf, 0, hub, -1, "wikilink").unwrap();
        store.insert_edge(leaf, 1, hub, -1, "wikilink").unwrap();

        assert_eq!(
            store.top_linked_files(10).unwrap(),
            vec![("hub.md".to_string(), 1)]
        );
    }

    #[test]
    fn tied_hubs_come_back_in_path_order() {
        let store = Store::open_memory().unwrap();
        let b = store
            .insert_file("b.md", "h", 1, "d000001", None, None)
            .unwrap();
        let a = store
            .insert_file("a.md", "h", 1, "d000002", None, None)
            .unwrap();
        let leaf = store
            .insert_file("leaf.md", "h", 1, "d000003", None, None)
            .unwrap();
        store.insert_edge(leaf, 0, b, -1, "wikilink").unwrap();
        store.insert_edge(leaf, 0, a, -1, "wikilink").unwrap();

        let top = store.top_linked_files(10).unwrap();
        assert_eq!(
            top.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
            vec!["a.md", "b.md"]
        );
    }

    #[test]
    fn top_linked_files_honours_its_limit() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 1, "d000001", None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 1, "d000002", None, None)
            .unwrap();
        let leaf = store
            .insert_file("leaf.md", "h", 1, "d000003", None, None)
            .unwrap();
        store.insert_edge(leaf, 0, a, -1, "wikilink").unwrap();
        store.insert_edge(leaf, 0, b, -1, "wikilink").unwrap();

        assert_eq!(store.top_linked_files(1).unwrap().len(), 1);
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
            !store
                .chunk_seqs_for_files(&[unchunked])
                .unwrap()
                .contains_key(&unchunked)
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
    fn a_pre_28_store_keeps_its_edges_at_the_document_level() {
        // The migration rebuilds the table to widen the unique key, which is the
        // one operation that could silently lose the whole graph.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.db");
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
}
