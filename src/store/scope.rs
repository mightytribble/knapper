//! A scope over notes, and the listings and scans built on it.

use super::Store;
use super::files::{FILE_COLUMNS, file_from_row};
use super::{DOC_LEVEL, FileRecord};
use crate::fault::Fault;
use anyhow::Result;
use rusqlite::OptionalExtension;

/// How a listing is ordered (#121).
///
/// `Path` is the vault's own order, which is what a caller reading one folder
/// wants. The two link orders answer the question a caller with no note name
/// has — what this vault is built around — and its opposite.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ListOrder {
    #[default]
    Path,
    LinksInDesc,
    LinksInAsc,
}

/// One row of a listing: the note, and the three numbers the index can answer
/// about it without opening the file (#121, #131).
///
/// The sizes ride beside `links_in` because the question they answer is the
/// pair: a note fifteen others point at, holding almost nothing, is a promise
/// the vault cannot pay. Neither number finds that alone — hubs are usually
/// fine and most stubs are proportionate. Only the mismatch is a finding, and
/// the ratio is the caller's arithmetic over the row.
#[derive(Debug, Clone)]
pub struct ListRow {
    pub file: FileRecord,
    /// How many distinct notes link to this note, over the whole vault.
    pub links_in: usize,
    /// How many chunks the note is indexed as — how many distinct units
    /// `search` can return it as. Not a size: chunk token counts run from
    /// tens to hundreds, so four chunks says nothing about how much is there.
    pub chunk_count: usize,
    /// The note's indexed size, as the sum of its chunks' token counts. It is
    /// what the embedder counted over `chunks.text`, so it measures what the
    /// index holds rather than what the file holds: where a section ran past
    /// the model's input wall, `split_oversized_chunks` repeats
    /// `chunker::OVERLAP_TOKENS` at the head of each piece and the sum counts
    /// those tokens twice. Zero for a note with no chunks.
    pub token_count: usize,
}

/// The note ids a scope's link terms resolve to (#66). Built by
/// [`Store::resolve_scope_links`] and handed to `scope_clauses` beside the
/// scope, so a clause cannot be built from an unresolved name.
#[derive(Debug, Default, Clone, Copy)]
pub struct LinkIds {
    pub links_to: Option<i64>,
    pub linked_from: Option<i64>,
}

/// A scope over a note query: SQL over a `files` row aliased `f`, and its
/// arguments in the order the SQL binds them (#65).
///
/// One author for the three operators over both kinds of term, because
/// `list_files` and `files_in_scope` ask the same question and a second copy is
/// a second thing to keep right. A tag term is an `EXISTS` over the junction; a
/// directory term is a range predicate on `files.path`, which needs no join.
///
/// `archive` is the store's archive folder, which the clauses leave out unless
/// the scope asks for it (#151).
pub(super) fn scope_clauses(
    scope: &crate::tags::Scope,
    links: &LinkIds,
    archive: Option<&str>,
) -> (String, Vec<Box<dyn rusqlite::types::ToSql>>) {
    use crate::tags::ScopeTerm;

    fn fragment(term: &ScopeTerm, args: &mut Vec<Box<dyn rusqlite::types::ToSql>>) -> String {
        match term {
            ScopeTerm::Tag(tag) => {
                let (pred, values) = crate::tags::predicate(tag);
                for value in values {
                    args.push(Box::new(value));
                }
                format!(
                    "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id
                             WHERE ft.file_id = f.id AND {pred})"
                )
            }
            ScopeTerm::Folder(folder) => {
                let (pred, values) = crate::tags::folder_sql(folder);
                for value in values {
                    args.push(Box::new(value));
                }
                format!("({pred})")
            }
            ScopeTerm::File(path) => {
                args.push(Box::new(path.clone()));
                "(f.path = ?)".to_string()
            }
        }
    }

    let mut sql = String::new();
    let mut args: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    for term in &scope.all {
        let frag = fragment(term, &mut args);
        sql.push_str(&format!(" AND {frag}"));
    }
    for (field, negate) in [(&scope.any, false), (&scope.none, true)] {
        if field.is_empty() {
            continue;
        }
        let ors: Vec<String> = field.iter().map(|t| fragment(t, &mut args)).collect();
        let group = ors.join(" OR ");
        if negate {
            sql.push_str(&format!(" AND NOT ({group})"));
        } else {
            sql.push_str(&format!(" AND ({group})"));
        }
    }

    // The property and link filters (#66). A link term alone reads `edges`,
    // so a plain `[[X]]` counts; with a property beside it, only links filed
    // under that name count, and the plain property clause is not added:
    // the property names the link, not the note being selected.
    let property = scope.property.as_ref();
    let push_value = |sql: &mut String, args: &mut Vec<Box<dyn rusqlite::types::ToSql>>| {
        if let Some(p) = property
            && let Some(v) = &p.value
        {
            sql.push_str(" AND p.value = ?");
            args.push(Box::new(v.clone()));
        }
    };
    if let Some(to) = links.links_to {
        match property {
            Some(p) => {
                sql.push_str(
                    " AND EXISTS (SELECT 1 FROM properties p
                                   WHERE p.file_id = f.id AND p.name = ? AND p.target_file = ?",
                );
                args.push(Box::new(p.name.clone()));
                args.push(Box::new(to));
                push_value(&mut sql, &mut args);
                sql.push(')');
            }
            None => {
                sql.push_str(
                    " AND EXISTS (SELECT 1 FROM edges e
                                   WHERE e.from_file = f.id AND e.to_file = ? AND e.edge_type = 'wikilink')",
                );
                args.push(Box::new(to));
            }
        }
    }
    if let Some(from) = links.linked_from {
        match property {
            Some(p) => {
                sql.push_str(
                    " AND EXISTS (SELECT 1 FROM properties p
                                   WHERE p.file_id = ? AND p.name = ? AND p.target_file = f.id",
                );
                args.push(Box::new(from));
                args.push(Box::new(p.name.clone()));
                push_value(&mut sql, &mut args);
                sql.push(')');
            }
            None => {
                sql.push_str(
                    " AND EXISTS (SELECT 1 FROM edges e
                                   WHERE e.from_file = ? AND e.to_file = f.id AND e.edge_type = 'wikilink')",
                );
                args.push(Box::new(from));
            }
        }
    }
    if let Some(p) = property
        && links.links_to.is_none()
        && links.linked_from.is_none()
    {
        sql.push_str(
            " AND EXISTS (SELECT 1 FROM properties p WHERE p.file_id = f.id AND p.name = ?",
        );
        args.push(Box::new(p.name.clone()));
        push_value(&mut sql, &mut args);
        sql.push(')');
    }
    // The archive (#151): the store's folder, left out unless the scope asks
    // for it.
    if let Some(folder) = archive
        && !scope.admits_archive(folder)
    {
        let (pred, values) = crate::tags::outside_folder_sql(folder);
        sql.push_str(&format!(" AND {pred}"));
        for value in values {
            args.push(Box::new(value));
        }
    }
    (sql, args)
}

impl Store {
    /// Every note the scope admits, in path order.
    ///
    /// The order is `files.path` under SQLite's BINARY collation, which is
    /// the byte order of the stored relative path, so a folder's notes
    /// arrive together and `Lore/` sorts before `lore/`. The UNIQUE index
    /// on `files.path` carries no COLLATE clause, so that index serves the
    /// ordering and the listing needs no sort step (#68).
    ///
    /// An absent `limit` emits no LIMIT clause, so a bare listing answers
    /// every note the scope admits; `Some(0)` reaches SQL as `LIMIT 0` and
    /// answers none, which is what the number says (#68).
    pub fn list_files(
        &self,
        tags: &crate::tags::Scope,
        created_by: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<FileRecord>> {
        Ok(self
            .list_files_with_links_in(tags, created_by, limit, None, ListOrder::Path)?
            .into_iter()
            .map(|row| row.file)
            .collect())
    }

    /// The notes a scope admits, each with the number of distinct notes that
    /// link to it, in the order `order` names (#121).
    ///
    /// Two things about the count, both of them the point:
    ///
    /// - It is `COUNT(DISTINCT from_file)` and not `COUNT(*)`. `edges` holds
    ///   one row per (source chunk, target chunk) pair, so a note named from
    ///   eight sections of one note is one link in and not eight. Ranking on
    ///   the row count ranks by how wordy a note's neighbours are.
    /// - It is over the whole vault and not over the scope. A note listed
    ///   from one folder still counts the notes that name it from anywhere,
    ///   so a folder listing does not report every note in it as isolated.
    ///
    /// The ordering runs in SQL, before the limit: a limit applied first
    /// would rank the first N notes by path rather than the vault.
    ///
    /// `after` is the path of the last row a caller received, and the listing
    /// starts at the row after it, so `limit` and `after` read a listing in
    /// pages (#143). The page is found by the row's sort key and not by a
    /// count of rows from the start: a note created or deleted before the
    /// boundary between two pages repeats or skips no other note. In path
    /// order the key is the path alone, so `after` need not name a note the
    /// vault still holds. Under a ranking the key is the note's `links_in`
    /// and its path, the count is read from the note as it is now, and a
    /// path no note holds is refused, because a guessed count would move the
    /// boundary with no sign of it. A link edit between two pages changes
    /// the count the ranking sorts on, so it can still move a note across
    /// the boundary.
    pub fn list_files_with_links_in(
        &self,
        tags: &crate::tags::Scope,
        created_by: Option<&str>,
        limit: Option<usize>,
        after: Option<&str>,
        order: ListOrder,
    ) -> Result<Vec<ListRow>> {
        // `none` is not checked: excluding a tag no note carries is a no-op.
        let checked: Vec<&crate::tags::ScopeTerm> =
            tags.all.iter().chain(tags.any.iter()).collect();
        crate::tags::check_terms(&self.conn, &checked)?;
        let links = self.resolve_scope_links(tags)?;

        const LINKS_IN: &str =
            "(SELECT COUNT(DISTINCT e.from_file) FROM edges e WHERE e.to_file = f.id)";
        let ranked_after = match (after, order) {
            (Some(path), ListOrder::LinksInDesc | ListOrder::LinksInAsc) => {
                let count: i64 = self
                    .conn
                    .query_row(
                        &format!("SELECT {LINKS_IN} FROM files f WHERE f.path = ?1"),
                        [path],
                        |row| row.get(0),
                    )
                    .optional()?
                    .ok_or_else(|| {
                        anyhow::anyhow!(Fault::InvalidInput(format!(
                            "no such note '{path}' for 'after'; a links_in ranking \
                             starts a page from that note's count, so start the \
                             listing again or list in path order"
                        )))
                    })?;
                Some((path, count))
            }
            _ => None,
        };

        let mut sql = format!(
            "SELECT {FILE_COLUMNS}, \
             {LINKS_IN} AS links_in, \
             (SELECT COUNT(*) FROM chunks c WHERE c.file_id = f.id) \
               AS chunk_count, \
             (SELECT COALESCE(SUM(c.token_count), 0) FROM chunks c WHERE c.file_id = f.id) \
               AS token_count \
             FROM files f WHERE 1=1"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let (tag_sql, tag_args) = scope_clauses(tags, &links, self.archive_folder());
        sql.push_str(&tag_sql);
        param_values.extend(tag_args);
        if let Some(cb) = created_by {
            sql.push_str(" AND f.created_by = ?");
            param_values.push(Box::new(cb.to_string()));
        }
        if let (Some(path), ListOrder::Path) = (after, order) {
            // The UNIQUE index on `files.path` serves this range as well as
            // the ordering.
            sql.push_str(" AND f.path > ?");
            param_values.push(Box::new(path.to_string()));
        }
        if let Some((path, count)) = ranked_after {
            // `links_in` is a column of the result and not of `files`, so the
            // keyset condition reads it from a subselect.
            let past = if order == ListOrder::LinksInDesc {
                "<"
            } else {
                ">"
            };
            sql = format!(
                "SELECT * FROM ({sql}) \
                 WHERE links_in {past} ? OR (links_in = ? AND path > ?)"
            );
            param_values.push(Box::new(count));
            param_values.push(Box::new(count));
            param_values.push(Box::new(path.to_string()));
        }
        // The path is the tie-break under either ranking, so two notes with
        // the same number of links in come back in the same order every call,
        // and a page boundary inside a tie means one thing.
        let wrapped = ranked_after.is_some();
        sql.push_str(match (order, wrapped) {
            (ListOrder::Path, _) => " ORDER BY f.path",
            (ListOrder::LinksInDesc, false) => " ORDER BY links_in DESC, f.path",
            (ListOrder::LinksInAsc, false) => " ORDER BY links_in ASC, f.path",
            (ListOrder::LinksInDesc, true) => " ORDER BY links_in DESC, path",
            (ListOrder::LinksInAsc, true) => " ORDER BY links_in ASC, path",
        });
        if let Some(limit) = limit {
            sql.push_str(" LIMIT ?");
            param_values.push(Box::new(limit as i64));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(param_values.iter()), |row| {
            Ok(ListRow {
                file: file_from_row(row)?,
                links_in: row.get::<_, i64>(9)? as usize,
                chunk_count: row.get::<_, i64>(10)? as usize,
                token_count: row.get::<_, i64>(11)? as usize,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Stream every chunk a scope admits, in path and then chunk order (#106).
    ///
    /// A visitor rather than a returned list: the literal scan behind `match`
    /// is exhaustive by contract, so it reads every row in scope whatever its
    /// limit, and materializing the vault's text to do that would be waste.
    ///
    /// `none` is not checked, for the reason `list_files` does not check it:
    /// excluding a tag no note carries is a no-op.
    pub fn for_each_scan_row_in_scope(
        &self,
        scope: &crate::tags::Scope,
        scan: crate::params::Scan,
        mut visit: impl FnMut(crate::matching::ScanRow),
    ) -> Result<()> {
        use crate::params::Scan;

        let checked: Vec<&crate::tags::ScopeTerm> =
            scope.all.iter().chain(scope.any.iter()).collect();
        crate::tags::check_terms(&self.conn, &checked)?;
        let links = self.resolve_scope_links(scope)?;

        // One statement over both halves, not two passes, so a note's YAML
        // and its prose arrive together and `limit` still cuts in vault order
        // (#137). The frontmatter arm sits at `DOC_LEVEL`, the sentinel
        // `properties` already gives a frontmatter row, and -1 sorts before
        // chunk 0, so a note's YAML leads its own chunks. A note whose block
        // is empty contributes no row: there is nothing in it to match.
        //
        // The scope clauses are built twice because each arm binds its own
        // copy of the arguments; `scope_clauses` is the one builder, so the
        // two arms cannot select different notes.
        let (scope_sql, mut args) = scope_clauses(scope, &links, self.archive_folder());
        let frontmatter_arm = format!(
            "SELECT f.path AS path, {DOC_LEVEL} AS seq, '' AS heading_path, f.frontmatter AS text
               FROM files f
              WHERE f.frontmatter IS NOT NULL AND f.frontmatter <> ''{scope_sql}"
        );
        let body_arm = format!(
            "SELECT f.path AS path, c.seq AS seq, c.heading_path AS heading_path, c.text AS text
               FROM chunks c JOIN files f ON f.id = c.file_id
              WHERE 1=1{scope_sql}"
        );
        let (sql, bound) = match scan {
            Scan::All => {
                // Each arm binds its own copy: a bound value is a boxed
                // `dyn ToSql` and cannot be cloned, so the builder runs again.
                let (_, second) = scope_clauses(scope, &links, self.archive_folder());
                args.extend(second);
                (
                    format!("{frontmatter_arm} UNION ALL {body_arm} ORDER BY path, seq"),
                    args,
                )
            }
            Scan::Frontmatter => (format!("{frontmatter_arm} ORDER BY path, seq"), args),
            Scan::Body => (format!("{body_arm} ORDER BY path, seq"), args),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(bound.iter()))?;
        while let Some(row) = rows.next()? {
            let seq: i64 = row.get(1)?;
            let heading_path: String = row.get(2)?;
            visit(crate::matching::ScanRow {
                file: row.get(0)?,
                part: if seq == DOC_LEVEL {
                    crate::matching::Part::Frontmatter
                } else {
                    crate::matching::Part::Body
                },
                // NOT NULL with an empty default: no breadcrumb, rather than a
                // breadcrumb of no characters. A frontmatter row never has one.
                heading_path: (!heading_path.is_empty()).then_some(heading_path),
                text: row.get(3)?,
            });
        }
        Ok(())
    }

    /// The ids of the notes a tag filter admits (#60).
    ///
    /// The scope a search runs under, resolved once so that all three lanes
    /// filter against the same set and `--explain` can report a count that is
    /// the one the lanes saw. `none` is not checked, for the reason
    /// `list_files` does not check it: excluding a tag no note carries is a
    /// no-op, and erroring on it would refuse a correct query.
    pub fn files_in_scope(&self, filter: &crate::tags::Scope) -> Result<Vec<i64>> {
        let checked: Vec<&crate::tags::ScopeTerm> =
            filter.all.iter().chain(filter.any.iter()).collect();
        crate::tags::check_terms(&self.conn, &checked)?;
        let links = self.resolve_scope_links(filter)?;

        let (tag_sql, args) = scope_clauses(filter, &links, self.archive_folder());
        let mut stmt = self
            .conn
            .prepare(&format!("SELECT f.id FROM files f WHERE 1=1{tag_sql}"))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), |row| row.get(0))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }

    /// Resolve a scope's link terms to note ids (#66), the way a wikilink
    /// target resolves, and then by alias (#142). An alias more than one note
    /// carries errors, naming the candidates. An unresolvable name errors,
    /// naming the nearest note the fuzzy file resolver finds.
    pub fn resolve_scope_links(&self, scope: &crate::tags::Scope) -> Result<LinkIds> {
        let resolve = |field: &str, term: &Option<crate::tags::LinkTerm>| -> Result<Option<i64>> {
            let Some(term) = term else { return Ok(None) };
            if let Some(id) = crate::graph::resolve_link_target(self, &term.written)? {
                return Ok(Some(id));
            }
            // A name the caller passed in also resolves by alias (#142). Not
            // inside `resolve_link_target`, which every edge resolves through:
            // Obsidian leaves `[[Sam]]` unresolved when only an alias matches.
            if let Some(f) = self.find_file_by_alias(&term.written)? {
                return Ok(Some(f.id));
            }
            match self.resolve_file(&term.written).ok().flatten() {
                Some(near) => anyhow::bail!(Fault::InvalidInput(format!(
                    "no such note '{}' for '{field}'; nearest: '{}'",
                    term.written, near.path
                ))),
                None => anyhow::bail!(Fault::InvalidInput(format!(
                    "no such note '{}' for '{field}'",
                    term.written
                ))),
            }
        };
        Ok(LinkIds {
            links_to: resolve("links_to", &scope.links_to)?,
            linked_from: resolve("linked_from", &scope.linked_from)?,
        })
    }
}

/// What a scope holds, counted: notes, their chunks, and the edges they
/// write. An edge is its source note's, so it is counted where its source
/// is (#151).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeTotals {
    pub files: usize,
    pub chunks: usize,
    pub edges: usize,
}

impl Store {
    /// The clauses `scope` compiles to on this store: its terms, its
    /// resolved link filters, and the archive rule. For a query over a whole
    /// vault that a read can still narrow (#151).
    pub(super) fn scope_sql(
        &self,
        scope: &crate::tags::Scope,
    ) -> Result<(String, Vec<Box<dyn rusqlite::types::ToSql>>)> {
        let links = self.resolve_scope_links(scope)?;
        Ok(scope_clauses(scope, &links, self.archive_folder()))
    }

    /// The notes, chunks and edges a scope holds (#151).
    pub fn totals_in_scope(&self, scope: &crate::tags::Scope) -> Result<ScopeTotals> {
        let (files_sql, mut args) = self.scope_sql(scope)?;
        let (chunks_sql, chunk_args) = self.scope_sql(scope)?;
        let (edges_sql, edge_args) = self.scope_sql(scope)?;
        args.extend(chunk_args);
        args.extend(edge_args);
        let sql = format!(
            "SELECT (SELECT COUNT(*) FROM files f WHERE 1=1{files_sql}),
                    (SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id
                      WHERE 1=1{chunks_sql}),
                    (SELECT COUNT(*) FROM edges e JOIN files f ON f.id = e.from_file
                      WHERE 1=1{edges_sql})"
        );
        Ok(self
            .conn
            .query_row(&sql, rusqlite::params_from_iter(args.iter()), |row| {
                Ok(ScopeTotals {
                    files: row.get::<_, i64>(0)? as usize,
                    chunks: row.get::<_, i64>(1)? as usize,
                    edges: row.get::<_, i64>(2)? as usize,
                })
            })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::fixtures::*;
    use crate::store::*;

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
            .list_files(&crate::tags::Scope::default(), None, Some(20))
            .unwrap();
        assert_eq!(files.len(), 3);
    }

    /// The listing is ordered by path and not by when a note was indexed:
    /// a folder's notes arrive together, and a subtree reads as one block
    /// (#68). The three rows below are written so that `indexed_at DESC`
    /// is the exact reverse of the path order, which is what the old
    /// ordering would answer.
    #[test]
    fn list_files_answers_in_path_order_not_index_order() {
        let store = Store::open_memory().unwrap();
        for (path, docid, stamp) in [
            ("zeta.md", "zzz111", "2026-01-03T00:00:00Z"),
            ("locations/aurelian.md", "aaa222", "2026-01-02T00:00:00Z"),
            ("bestiary/wight.md", "bbb333", "2026-01-01T00:00:00Z"),
        ] {
            store
                .insert_file(path, "h", 100, docid, None, None)
                .unwrap();
            store
                .conn
                .execute(
                    "UPDATE files SET indexed_at = ?1 WHERE path = ?2",
                    rusqlite::params![stamp, path],
                )
                .unwrap();
        }
        let paths: Vec<String> = store
            .list_files(&crate::tags::Scope::default(), None, None)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(
            paths,
            vec![
                "bestiary/wight.md".to_string(),
                "locations/aurelian.md".to_string(),
                "zeta.md".to_string(),
            ]
        );
    }

    /// The order is SQLite's BINARY collation over the stored relative
    /// path, which is byte order, so a capital folder sorts before its
    /// lowercase twin (#68).
    #[test]
    fn list_files_sorts_a_capital_folder_before_its_lowercase_twin() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("lore/a.md", "h", 100, "low111", None, None)
            .unwrap();
        store
            .insert_file("Lore/a.md", "h", 100, "cap111", None, None)
            .unwrap();
        let paths: Vec<String> = store
            .list_files(&crate::tags::Scope::default(), None, None)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(
            paths,
            vec!["Lore/a.md".to_string(), "lore/a.md".to_string()]
        );
    }

    /// No limit answers every note the scope admits; a limit keeps the
    /// first n of the path order; `Some(0)` is the SQL reading of the
    /// number the caller wrote and answers none (#68).
    #[test]
    fn list_files_reads_an_absent_limit_as_no_limit() {
        let store = Store::open_memory().unwrap();
        for i in 0..5 {
            store
                .insert_file(
                    &format!("n{i}.md"),
                    "h",
                    100,
                    &format!("ddd{i:03}"),
                    None,
                    None,
                )
                .unwrap();
        }
        let all = store
            .list_files(&crate::tags::Scope::default(), None, None)
            .unwrap();
        assert_eq!(all.len(), 5);

        let first_two: Vec<String> = store
            .list_files(&crate::tags::Scope::default(), None, Some(2))
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(first_two, vec!["n0.md".to_string(), "n1.md".to_string()]);

        assert!(
            store
                .list_files(&crate::tags::Scope::default(), None, Some(0))
                .unwrap()
                .is_empty()
        );
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
                &crate::tags::Scope::parse(&["rust".to_string()], &[], &[]).unwrap(),
                None,
                Some(20),
            )
            .unwrap();
        assert_eq!(files.len(), 2);
        let files = store
            .list_files(
                &crate::tags::Scope::parse(&["rust".to_string(), "cli".to_string()], &[], &[])
                    .unwrap(),
                None,
                Some(20),
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
            .list_files(&crate::tags::Scope::default(), Some("cli"), Some(20))
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "a.md");
        assert_eq!(files[0].created_by, Some("cli".to_string()));

        // Filter by "mcp" → only the mcp-created file
        let files = store
            .list_files(&crate::tags::Scope::default(), Some("mcp"), Some(20))
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "b.md");

        // Filter by None → all 3
        let files = store
            .list_files(&crate::tags::Scope::default(), None, Some(20))
            .unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn links_in_counts_notes_and_not_the_sections_they_link_from() {
        let store = Store::open_memory().unwrap();
        let hub = store
            .insert_file("hub.md", "h", 100, "hub1", None, None)
            .unwrap();
        let wordy = store
            .insert_file("wordy.md", "h", 100, "wo1", None, None)
            .unwrap();
        let terse = store
            .insert_file("terse.md", "h", 100, "te1", None, None)
            .unwrap();
        // `wordy` names the hub from three of its sections; `terse` from one.
        // That is four edge rows and two notes linking in.
        for seq in 0..3 {
            store
                .insert_edge(wordy, seq, hub, DOC_LEVEL, "wikilink")
                .unwrap();
        }
        store
            .insert_edge(terse, 0, hub, DOC_LEVEL, "wikilink")
            .unwrap();
        assert_eq!(store.edge_count_for_file(hub).unwrap(), 4);

        let rows = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                None,
                None,
                ListOrder::LinksInDesc,
            )
            .unwrap();
        let hub_row = rows.iter().find(|r| r.file.path == "hub.md").unwrap();
        assert_eq!(hub_row.links_in, 2, "two notes link in, from four sections");
    }

    #[test]
    fn a_links_in_ranking_ranks_the_vault_and_not_the_first_page_of_it() {
        let store = Store::open_memory().unwrap();
        // The hub sorts last by path, so a limit applied before the ranking
        // would never reach it.
        let hub = store
            .insert_file("zeta.md", "h", 100, "z1", None, None)
            .unwrap();
        for (i, name) in ["alpha.md", "beta.md", "gamma.md"].iter().enumerate() {
            let f = store
                .insert_file(name, "h", 100, &format!("f{i}"), None, None)
                .unwrap();
            store.insert_edge(f, 0, hub, DOC_LEVEL, "wikilink").unwrap();
        }

        let top = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                Some(1),
                None,
                ListOrder::LinksInDesc,
            )
            .unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].file.path, "zeta.md");
        assert_eq!(top[0].links_in, 3);
    }

    #[test]
    fn links_in_counts_the_notes_that_name_it_from_outside_the_scope() {
        let store = Store::open_memory().unwrap();
        let inside = store
            .insert_file("lore/hub.md", "h", 100, "in1", None, None)
            .unwrap();
        let outside = store
            .insert_file("npcs/caller.md", "h", 100, "ou1", None, None)
            .unwrap();
        store
            .insert_edge(outside, 0, inside, DOC_LEVEL, "wikilink")
            .unwrap();

        let rows = store
            .list_files_with_links_in(
                &crate::tags::Scope::parse(&["/lore/".to_string()], &[], &[]).unwrap(),
                None,
                None,
                None,
                ListOrder::LinksInDesc,
            )
            .unwrap();
        assert_eq!(rows.len(), 1, "the scope admits the one note");
        assert_eq!(
            rows[0].links_in, 1,
            "a note the scope excludes still counts as a link in"
        );
    }

    #[test]
    fn notes_with_the_same_links_in_come_back_in_path_order() {
        let store = Store::open_memory().unwrap();
        let caller = store
            .insert_file("caller.md", "h", 100, "c1", None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 100, "b1", None, None)
            .unwrap();
        let a = store
            .insert_file("a.md", "h", 100, "a1", None, None)
            .unwrap();
        store
            .insert_edge(caller, 0, b, DOC_LEVEL, "wikilink")
            .unwrap();
        store
            .insert_edge(caller, 0, a, DOC_LEVEL, "wikilink")
            .unwrap();

        let rows = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                None,
                None,
                ListOrder::LinksInDesc,
            )
            .unwrap();
        let ranked: Vec<&str> = rows.iter().map(|r| r.file.path.as_str()).collect();
        assert_eq!(ranked, vec!["a.md", "b.md", "caller.md"]);
    }

    #[test]
    fn the_ascending_ranking_starts_at_the_note_fewest_others_name() {
        let store = Store::open_memory().unwrap();
        let hub = store
            .insert_file("hub.md", "h", 100, "h1", None, None)
            .unwrap();
        let lonely = store
            .insert_file("lonely.md", "h", 100, "l1", None, None)
            .unwrap();
        store
            .insert_edge(lonely, 0, hub, DOC_LEVEL, "wikilink")
            .unwrap();

        let rows = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                None,
                None,
                ListOrder::LinksInAsc,
            )
            .unwrap();
        assert_eq!(rows[0].file.path, "lonely.md");
        assert_eq!(rows[0].links_in, 0);
        assert_eq!(rows[1].file.path, "hub.md");
        assert_eq!(rows[1].links_in, 1);
    }

    #[test]
    fn a_listing_answers_each_notes_own_chunk_count_and_token_total() {
        let store = Store::open_memory().unwrap();
        let long = store
            .insert_file("long.md", "h", 100, "lo1", None, None)
            .unwrap();
        let short = store
            .insert_file("short.md", "h", 100, "sh1", None, None)
            .unwrap();
        for (seq, tokens) in [(0i64, 100i64), (1, 200), (2, 300)] {
            store
                .insert_chunk(&NewChunk {
                    file_id: long,
                    seq,
                    text: "body",
                    vector_id: seq as u64 + 1,
                    token_count: tokens,
                    ..Default::default()
                })
                .unwrap();
        }
        store
            .insert_chunk(&NewChunk {
                file_id: short,
                seq: 0,
                text: "body",
                vector_id: 10,
                token_count: 40,
                ..Default::default()
            })
            .unwrap();

        let rows = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                None,
                None,
                ListOrder::Path,
            )
            .unwrap();
        let long_row = rows.iter().find(|r| r.file.path == "long.md").unwrap();
        assert_eq!(long_row.chunk_count, 3);
        assert_eq!(
            long_row.token_count, 600,
            "the note's own chunks, summed, and not the vault's"
        );
        let short_row = rows.iter().find(|r| r.file.path == "short.md").unwrap();
        assert_eq!(short_row.chunk_count, 1);
        assert_eq!(short_row.token_count, 40);
    }

    /// `SUM` over no rows is NULL, so a note the chunker produced nothing for
    /// — frontmatter and no body — would fail the row read rather than report
    /// its size. `index_file` inserts the `files` row before it loops over the
    /// chunks, so such a note is reachable (#131).
    #[test]
    fn a_note_with_no_chunks_answers_a_zero_size_and_not_a_null() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("empty.md", "h", 100, "em1", None, None)
            .unwrap();

        let rows = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                None,
                None,
                ListOrder::Path,
            )
            .unwrap();
        assert_eq!(rows[0].chunk_count, 0);
        assert_eq!(rows[0].token_count, 0);
    }

    /// Six notes in two folders. The `links_in` counts are a 2, b 1, c 2,
    /// d 0, e 1, f 2, so a tie falls across a page boundary at most page
    /// sizes, in both rankings and under a `/lore/` scope (#143).
    fn paging_fixture() -> Store {
        let store = Store::open_memory().unwrap();
        let mut id = std::collections::HashMap::new();
        for name in ["a", "b", "c", "d", "e", "f"] {
            let folder = if name < "d" { "lore" } else { "npcs" };
            let path = format!("{folder}/{name}.md");
            id.insert(
                name,
                store
                    .insert_file(&path, "h", 100, &format!("{name}1"), None, None)
                    .unwrap(),
            );
        }
        for (from, to) in [
            ("d", "a"),
            ("e", "a"),
            ("d", "b"),
            ("e", "c"),
            ("f", "c"),
            ("a", "e"),
            ("a", "f"),
            ("b", "f"),
        ] {
            store
                .insert_edge(id[from], 0, id[to], DOC_LEVEL, "wikilink")
                .unwrap();
        }
        store
    }

    /// Read a listing page by page, each page starting after the last path
    /// the one before it answered, and stop at the first short page.
    fn page_through(
        store: &Store,
        scope: &crate::tags::Scope,
        order: ListOrder,
        size: usize,
    ) -> Vec<String> {
        let mut paths: Vec<String> = Vec::new();
        // A page that ignores `after` repeats itself for ever; the bound
        // turns that into a failure rather than a hang.
        for _ in 0..16 {
            let page = store
                .list_files_with_links_in(
                    scope,
                    None,
                    Some(size),
                    paths.last().map(String::as_str),
                    order,
                )
                .unwrap();
            let full = page.len() == size;
            paths.extend(page.into_iter().map(|r| r.file.path));
            if !full {
                return paths;
            }
        }
        panic!("no short page after 16 pages: {paths:?}");
    }

    #[test]
    fn pages_read_after_the_last_path_join_into_the_whole_listing_in_every_order() {
        let store = paging_fixture();
        let lore = crate::tags::Scope::parse(&["/lore/".to_string()], &[], &[]).unwrap();
        for scope in [crate::tags::Scope::default(), lore] {
            for order in [
                ListOrder::Path,
                ListOrder::LinksInDesc,
                ListOrder::LinksInAsc,
            ] {
                let whole: Vec<String> = store
                    .list_files_with_links_in(&scope, None, None, None, order)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.file.path)
                    .collect();
                for size in 1..=whole.len() + 1 {
                    assert_eq!(
                        page_through(&store, &scope, order, size),
                        whole,
                        "{order:?}, pages of {size}, scope {scope:?}"
                    );
                }
            }
        }
    }

    /// The case `after` exists for: a caller that writes notes while it
    /// reads a listing. A position counted from the start would shift by one
    /// for the new note and answer the boundary row a second time (#143).
    #[test]
    fn a_note_created_before_the_boundary_repeats_no_row_on_the_next_page() {
        let store = Store::open_memory().unwrap();
        for name in ["b.md", "d.md", "f.md", "h.md"] {
            store.insert_file(name, "h", 100, name, None, None).unwrap();
        }
        let scope = crate::tags::Scope::default();
        let first = store
            .list_files_with_links_in(&scope, None, Some(2), None, ListOrder::Path)
            .unwrap();
        assert_eq!(first.last().unwrap().file.path, "d.md");

        store
            .insert_file("a.md", "h", 100, "a.md", None, None)
            .unwrap();

        let second: Vec<String> = store
            .list_files_with_links_in(&scope, None, Some(2), Some("d.md"), ListOrder::Path)
            .unwrap()
            .into_iter()
            .map(|r| r.file.path)
            .collect();
        assert_eq!(second, vec!["f.md", "h.md"]);
    }

    /// In path order the boundary is a comparison of text, so the note that
    /// ended the last page may have gone since.
    #[test]
    fn a_path_listing_starts_after_a_path_no_note_holds() {
        let store = Store::open_memory().unwrap();
        for name in ["b.md", "d.md", "f.md"] {
            store.insert_file(name, "h", 100, name, None, None).unwrap();
        }
        let rows: Vec<String> = store
            .list_files_with_links_in(
                &crate::tags::Scope::default(),
                None,
                None,
                Some("c.md"),
                ListOrder::Path,
            )
            .unwrap()
            .into_iter()
            .map(|r| r.file.path)
            .collect();
        assert_eq!(rows, vec!["d.md", "f.md"]);
    }

    /// A ranking places the next page by the boundary note's count, and a
    /// note that has gone has none. Guessing one would skip or repeat rows
    /// with no sign of it, so the call is refused (#143).
    #[test]
    fn a_links_in_ranking_refuses_to_start_after_a_note_the_vault_does_not_hold() {
        let store = paging_fixture();
        for order in [ListOrder::LinksInDesc, ListOrder::LinksInAsc] {
            let err = store
                .list_files_with_links_in(
                    &crate::tags::Scope::default(),
                    None,
                    Some(2),
                    Some("lore/gone.md"),
                    order,
                )
                .unwrap_err()
                .to_string();
            assert!(
                err.starts_with("no such note 'lore/gone.md' for 'after'"),
                "{order:?}: {err}"
            );
        }
    }

    /// The scoped file ids as paths, sorted, so an assertion reads as notes.
    fn scoped_paths(store: &Store, filter: &crate::tags::Scope) -> Vec<String> {
        let mut paths: Vec<String> = store
            .files_in_scope(filter)
            .unwrap()
            .into_iter()
            .map(|id| store.get_file_by_id(id).unwrap().unwrap().path)
            .collect();
        paths.sort();
        paths
    }

    fn folder_fixture() -> Store {
        let store = Store::open_memory().unwrap();
        for (i, path) in [
            "Locations.md",
            "Locations/aurelian-empire.md",
            "Locations/cities/varenholt.md",
            "People/marcus.md",
        ]
        .iter()
        .enumerate()
        {
            store
                .insert_file(path, "h", i as i64, &format!("d00000{i}"), None, None)
                .unwrap();
        }
        store
    }

    fn folder_scope(all: &[&str], any: &[&str], none: &[&str]) -> crate::tags::Scope {
        crate::tags::Scope::parse(
            &all.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &any.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &none.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Two archived notes, two live ones, in a store that knows the folder.
    fn archive_fixture() -> Store {
        let store = Store::open_memory()
            .unwrap()
            .with_archive_folder("04-Archive");
        for (i, path) in [
            "04-Archive/lore/old.md",
            "04-Archive/top.md",
            "lore/live.md",
            "notes.md",
        ]
        .iter()
        .enumerate()
        {
            store
                .insert_file(path, "h", i as i64, &format!("a00000{i}"), None, None)
                .unwrap();
        }
        store
    }

    /// The archive is set aside: a read leaves it out unless it asks (#151).
    #[test]
    fn a_scope_leaves_the_archive_out_by_default() {
        let store = archive_fixture();
        let empty = crate::tags::Scope::default();
        assert_eq!(scoped_paths(&store, &empty), ["lore/live.md", "notes.md"]);
        let listed: Vec<String> = store
            .list_files(&empty, None, None)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(
            listed,
            ["lore/live.md", "notes.md"],
            "a listing inherits it"
        );
    }

    #[test]
    fn include_archive_brings_the_archive_back() {
        let store = archive_fixture();
        let scope = crate::tags::Scope::default().including_archive(true);
        assert_eq!(
            scoped_paths(&store, &scope),
            [
                "04-Archive/lore/old.md",
                "04-Archive/top.md",
                "lore/live.md",
                "notes.md"
            ]
        );
    }

    /// Naming the archive is asking for it, so the exclusion would answer an
    /// empty scope (#151).
    #[test]
    fn naming_the_archive_in_all_or_any_brings_it_back() {
        let store = archive_fixture();
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/04-Archive/"], &[], &[])),
            ["04-Archive/lore/old.md", "04-Archive/top.md"]
        );
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/04-Archive/lore/"], &[], &[])),
            ["04-Archive/lore/old.md"]
        );
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/04-Archive"], &[], &[])),
            ["04-Archive/top.md"],
            "the folder's own notes"
        );
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/04-Archive/top.md"], &[], &[])),
            ["04-Archive/top.md"],
            "one note inside it"
        );
        assert_eq!(
            scoped_paths(
                &store,
                &folder_scope(&[], &["/04-Archive/lore/", "/lore/"], &[])
            ),
            ["04-Archive/lore/old.md", "lore/live.md"]
        );
    }

    #[test]
    fn a_none_term_or_a_tag_term_leaves_the_archive_out() {
        let store = archive_fixture();
        let old = store
            .get_file("04-Archive/lore/old.md")
            .unwrap()
            .unwrap()
            .id;
        store
            .reconcile_file_tags(
                old,
                &[crate::tags::Tag {
                    path: "type/old".into(),
                    display: "type/old".into(),
                }],
            )
            .unwrap();
        assert!(
            scoped_paths(&store, &folder_scope(&["type/old"], &[], &[])).is_empty(),
            "a tag names no folder, so the archive stays out"
        );
        assert_eq!(
            scoped_paths(&store, &folder_scope(&[], &[], &["/lore/"])),
            ["notes.md"]
        );
    }

    /// A range and not a `LIKE`: `_` is literal and the case is kept, as a
    /// directory term's is (#65).
    #[test]
    fn the_archive_folder_is_matched_literally_and_in_its_case() {
        let store = Store::open_memory()
            .unwrap()
            .with_archive_folder("_archive");
        for (i, path) in [
            "_archive/n.md",
            "xarchive/n.md",
            "_ARCHIVE/n.md",
            "_archive.md",
        ]
        .iter()
        .enumerate()
        {
            store
                .insert_file(path, "h", i as i64, &format!("b00000{i}"), None, None)
                .unwrap();
        }
        assert_eq!(
            scoped_paths(&store, &crate::tags::Scope::default()),
            ["_ARCHIVE/n.md", "_archive.md", "xarchive/n.md"]
        );
    }

    #[test]
    fn a_store_with_no_archive_folder_leaves_nothing_out() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("04-Archive/n.md", "h", 1, "c00000", None, None)
            .unwrap();
        assert_eq!(
            scoped_paths(&store, &crate::tags::Scope::default()),
            ["04-Archive/n.md"]
        );
        assert!(
            !store
                .excludes_archive(&crate::tags::Scope::default())
                .unwrap()
        );
    }

    /// Search keeps its unscoped path when nothing is archived, so the
    /// exclusion counts only when the archive holds a note (#151).
    #[test]
    fn the_archive_counts_as_excluded_only_when_it_holds_a_note() {
        let store = Store::open_memory()
            .unwrap()
            .with_archive_folder("04-Archive");
        store
            .insert_file("lore/live.md", "h", 1, "e00000", None, None)
            .unwrap();
        let empty = crate::tags::Scope::default();
        assert!(
            !store.excludes_archive(&empty).unwrap(),
            "nothing is archived"
        );

        store
            .insert_file("04-Archive/old.md", "h", 2, "e00001", None, None)
            .unwrap();
        assert!(store.excludes_archive(&empty).unwrap());
        assert!(
            !store
                .excludes_archive(&empty.clone().including_archive(true))
                .unwrap()
        );
        assert!(
            !store
                .excludes_archive(&folder_scope(&["/04-Archive/"], &[], &[]))
                .unwrap()
        );
    }

    /// A file with two chunks, so a scan has an order to answer in.
    fn chunk_fixture() -> Store {
        let store = Store::open_memory().unwrap();
        for (i, path) in ["People/marcus.md", "Places/varenholt.md"]
            .iter()
            .enumerate()
        {
            let file_id = store
                .insert_file(path, "h", i as i64, &format!("d00000{i}"), None, None)
                .unwrap();
            store
                .set_file_frontmatter(file_id, &format!("{path} yaml"))
                .unwrap();
            for seq in 0..2 {
                store
                    .insert_chunk(&NewChunk {
                        file_id,
                        seq,
                        heading: "Biography",
                        heading_path: if seq == 0 { "" } else { "Marcus > Biography" },
                        tags_text: "",
                        text: &format!("{path} chunk {seq}"),
                        vector_id: (i * 2 + seq as usize) as u64,
                        token_count: 4,
                    })
                    .unwrap();
            }
        }
        store
    }

    fn scanned(store: &Store, scope: &crate::tags::Scope) -> Vec<crate::matching::ScanRow> {
        scanned_with(store, scope, crate::params::Scan::All)
    }

    /// The body arm alone, which is what the chunk-order tests below assert.
    fn scanned_body(store: &Store, scope: &crate::tags::Scope) -> Vec<crate::matching::ScanRow> {
        scanned_with(store, scope, crate::params::Scan::Body)
    }

    fn scanned_with(
        store: &Store,
        scope: &crate::tags::Scope,
        scan: crate::params::Scan,
    ) -> Vec<crate::matching::ScanRow> {
        let mut rows = Vec::new();
        store
            .for_each_scan_row_in_scope(scope, scan, |row| rows.push(row))
            .unwrap();
        rows
    }

    #[test]
    fn a_chunk_scan_answers_every_chunk_in_path_then_chunk_order() {
        let rows = scanned_body(&chunk_fixture(), &crate::tags::Scope::default());
        assert_eq!(
            rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
            vec![
                "People/marcus.md chunk 0",
                "People/marcus.md chunk 1",
                "Places/varenholt.md chunk 0",
                "Places/varenholt.md chunk 1",
            ]
        );
        assert_eq!(rows[0].file, "People/marcus.md");
    }

    #[test]
    fn a_scan_puts_a_notes_yaml_before_its_own_chunks() {
        // The frontmatter arm sits at `DOC_LEVEL`, and -1 sorts before chunk
        // 0, so the two halves interleave per note rather than arriving as
        // two passes — which is what keeps `limit` cutting in vault order.
        let rows = scanned(&chunk_fixture(), &crate::tags::Scope::default());
        assert_eq!(
            rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
            vec![
                "People/marcus.md yaml",
                "People/marcus.md chunk 0",
                "People/marcus.md chunk 1",
                "Places/varenholt.md yaml",
                "Places/varenholt.md chunk 0",
                "Places/varenholt.md chunk 1",
            ]
        );
        assert_eq!(rows[0].part, crate::matching::Part::Frontmatter);
        assert_eq!(rows[1].part, crate::matching::Part::Body);
    }

    #[test]
    fn a_scan_narrowed_to_one_half_reads_only_that_half() {
        let store = chunk_fixture();
        let read = |scan| {
            let mut rows = Vec::new();
            store
                .for_each_scan_row_in_scope(&crate::tags::Scope::default(), scan, |row| {
                    rows.push(row)
                })
                .unwrap();
            rows
        };

        let yaml = read(crate::params::Scan::Frontmatter);
        assert_eq!(yaml.len(), 2);
        assert!(
            yaml.iter()
                .all(|r| r.part == crate::matching::Part::Frontmatter)
        );

        let body = read(crate::params::Scan::Body);
        assert_eq!(body.len(), 4);
        assert!(body.iter().all(|r| r.part == crate::matching::Part::Body));
    }

    #[test]
    fn a_scans_two_halves_answer_one_scope() {
        // The arms are built from one `scope_clauses`, so a filter cannot
        // admit a note's prose and drop its frontmatter.
        let rows = scanned(&chunk_fixture(), &folder_scope(&["/Places/"], &[], &[]));
        assert_eq!(
            rows.iter().map(|r| r.file.as_str()).collect::<Vec<_>>(),
            vec![
                "Places/varenholt.md",
                "Places/varenholt.md",
                "Places/varenholt.md"
            ]
        );
    }

    #[test]
    fn a_note_with_an_empty_yaml_block_contributes_no_frontmatter_row() {
        // `''` is a note that carries no block. There is nothing in it to
        // match, so it is not a row — and it is still not the NULL that means
        // the index cannot say.
        let store = Store::open_memory().unwrap();
        let file_id = store
            .insert_file("People/marcus.md", "h", 1, "d000001", None, None)
            .unwrap();
        store.set_file_frontmatter(file_id, "").unwrap();

        assert!(scanned(&store, &crate::tags::Scope::default()).is_empty());
        assert!(
            !store
                .frontmatter_unindexed(&crate::tags::Scope::default())
                .unwrap()
        );
    }

    #[test]
    fn a_row_written_before_the_column_reads_as_unindexed_only_in_its_own_scope() {
        // Per-scope rather than per-store, so a scope of freshly indexed
        // notes answers while the rest of the vault waits for its re-index.
        let store = chunk_fixture();
        let marcus = store.get_file("People/marcus.md").unwrap().unwrap().id;
        store
            .conn
            .execute(
                "UPDATE files SET frontmatter = NULL WHERE id = ?1",
                [marcus],
            )
            .unwrap();

        assert!(
            store
                .frontmatter_unindexed(&crate::tags::Scope::default())
                .unwrap()
        );
        assert!(
            !store
                .frontmatter_unindexed(&folder_scope(&["/Places/"], &[], &[]))
                .unwrap()
        );
    }

    #[test]
    fn a_chunk_scan_answers_only_what_the_scope_admits() {
        let rows = scanned_body(&chunk_fixture(), &folder_scope(&["/Places/"], &[], &[]));
        assert_eq!(
            rows.iter().map(|r| r.file.as_str()).collect::<Vec<_>>(),
            vec!["Places/varenholt.md", "Places/varenholt.md"]
        );
    }

    #[test]
    fn a_chunk_with_no_breadcrumb_scans_as_no_heading_path() {
        // `chunks.heading_path` is NOT NULL and defaults to the empty string,
        // which is the absence of a breadcrumb rather than a breadcrumb of no
        // characters.
        let rows = scanned_body(&chunk_fixture(), &crate::tags::Scope::default());
        assert_eq!(rows[0].heading_path, None);
        assert_eq!(rows[1].heading_path.as_deref(), Some("Marcus > Biography"));
    }

    #[test]
    fn a_directory_subtree_takes_everything_beneath_it() {
        let store = folder_fixture();
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/Locations/"], &[], &[])),
            vec![
                "Locations/aurelian-empire.md",
                "Locations/cities/varenholt.md"
            ]
        );
    }

    #[test]
    fn a_directory_exact_takes_only_direct_children() {
        let store = folder_fixture();
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/Locations"], &[], &[])),
            vec!["Locations/aurelian-empire.md"]
        );
    }

    #[test]
    fn a_directory_subtree_excludes_a_root_note_of_the_same_name() {
        let store = folder_fixture();
        let paths = scoped_paths(&store, &folder_scope(&["/Locations/"], &[], &[]));
        assert!(
            !paths.contains(&"Locations.md".to_string()),
            "a root note is not under the folder that shares its stem"
        );
    }

    #[test]
    fn list_files_filters_by_a_directory_scope() {
        // #65. Reach test: `list_files` already takes a `Scope` (Task 3), so
        // this proves the wiring carries a directory term with no
        // production change.
        let store = folder_fixture();
        assert_eq!(
            listed_paths(&store, &folder_scope(&["/People/"], &[], &[])),
            vec!["People/marcus.md"]
        );
    }

    #[test]
    fn a_directory_scope_is_case_sensitive() {
        // Two folders differ only in case, so both validate; the range on
        // files.path keeps case, so /Locations/ resolves to the capitalized
        // folder alone, and a LOWER() comparison would fail this (#65).
        let store = Store::open_memory().unwrap();
        store
            .insert_file("Locations/a.md", "h", 1, "d000001", None, None)
            .unwrap();
        store
            .insert_file("locations/b.md", "h", 2, "d000002", None, None)
            .unwrap();
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/Locations/"], &[], &[])),
            vec!["Locations/a.md"]
        );
    }

    #[test]
    fn a_scope_mixes_a_tag_and_a_directory_term() {
        let store = Store::open_memory().unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        let empire = store
            .insert_file(
                "Locations/aurelian-empire.md",
                "h",
                1,
                "d000001",
                None,
                None,
            )
            .unwrap();
        store
            .insert_file(
                "Locations/cities/varenholt.md",
                "h",
                2,
                "d000002",
                None,
                None,
            )
            .unwrap();
        let marcus = store
            .insert_file("People/marcus.md", "h", 3, "d000003", None, None)
            .unwrap();
        store
            .reconcile_file_tags(empire, &[tag("type/place")])
            .unwrap();
        store
            .reconcile_file_tags(marcus, &[tag("type/person")])
            .unwrap();

        // all: under Locations/ AND tagged type/place -> the empire alone.
        assert_eq!(
            scoped_paths(
                &store,
                &folder_scope(&["/Locations/", "type/place"], &[], &[])
            ),
            vec!["Locations/aurelian-empire.md"]
        );
        // any: under People/ OR tagged type/place -> the empire and marcus.
        assert_eq!(
            scoped_paths(&store, &folder_scope(&[], &["/People/", "type/place"], &[])),
            vec!["Locations/aurelian-empire.md", "People/marcus.md"]
        );
    }

    #[test]
    fn a_directory_naming_no_note_errors_and_names_the_nearest_ancestor() {
        let store = folder_fixture();
        let err = store
            .files_in_scope(&folder_scope(&["/Locations/atlantis/"], &[], &[]))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such folder '/Locations/atlantis/'; nearest: '/Locations/'"
        );
    }

    #[test]
    fn an_exact_directory_with_only_deeper_notes_suggests_the_subtree() {
        let store = Store::open_memory().unwrap();
        store
            .insert_file("Deep/inner/note.md", "h", 1, "d000001", None, None)
            .unwrap();
        let err = store
            .files_in_scope(&folder_scope(&["/Deep"], &[], &[]))
            .unwrap_err();
        assert_eq!(err.to_string(), "no such folder '/Deep'; nearest: '/Deep/'");
    }

    #[test]
    fn a_wholly_unknown_directory_errors_without_a_nearest() {
        let store = folder_fixture();
        let err = store
            .files_in_scope(&folder_scope(&["/Nowhere/"], &[], &[]))
            .unwrap_err();
        assert_eq!(err.to_string(), "no such folder '/Nowhere/'");
    }

    #[test]
    fn a_file_term_admits_that_note_alone() {
        let store = folder_fixture();
        assert_eq!(
            scoped_paths(
                &store,
                &folder_scope(&["/Locations/aurelian-empire.md"], &[], &[])
            ),
            vec!["Locations/aurelian-empire.md"]
        );
        // `Locations.md` sits beside the `Locations/` folder, and a file term
        // names it without taking the folder's notes along.
        assert_eq!(
            scoped_paths(&store, &folder_scope(&["/Locations.md"], &[], &[])),
            vec!["Locations.md"]
        );
    }

    #[test]
    fn a_file_term_excludes_one_note_and_mixes_with_other_terms() {
        let store = folder_fixture();
        assert_eq!(
            scoped_paths(
                &store,
                &folder_scope(&["/Locations/"], &[], &["/Locations/aurelian-empire.md"])
            ),
            vec!["Locations/cities/varenholt.md"]
        );
        assert_eq!(
            scoped_paths(
                &store,
                &folder_scope(&[], &["/People/", "/Locations.md"], &[])
            ),
            vec!["Locations.md", "People/marcus.md"]
        );
    }

    #[test]
    fn a_file_term_differing_only_in_case_names_the_vaults_spelling() {
        let store = folder_fixture();
        let err = store
            .files_in_scope(&folder_scope(&["/locations/Aurelian-Empire.md"], &[], &[]))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such note '/locations/Aurelian-Empire.md'; nearest: '/Locations/aurelian-empire.md'"
        );
    }

    #[test]
    fn a_file_term_in_the_wrong_folder_names_the_note_of_that_name() {
        let store = folder_fixture();
        let err = store
            .files_in_scope(&folder_scope(&["/People/varenholt.md"], &[], &[]))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such note '/People/varenholt.md'; nearest: '/Locations/cities/varenholt.md'"
        );
    }

    #[test]
    fn an_unknown_file_term_falls_back_to_its_nearest_folder() {
        let store = folder_fixture();
        let err = store
            .files_in_scope(&folder_scope(&["/Locations/cities/atlantis.md"], &[], &[]))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such note '/Locations/cities/atlantis.md'; nearest: '/Locations/cities/'"
        );

        let err = store
            .files_in_scope(&folder_scope(&["/Nowhere.md"], &[], &[]))
            .unwrap_err();
        assert_eq!(err.to_string(), "no such note '/Nowhere.md'");
    }

    #[test]
    fn excluding_an_unknown_directory_is_not_an_error() {
        let store = folder_fixture();
        let paths = scoped_paths(&store, &folder_scope(&[], &[], &["/Nowhere/"]));
        assert!(!paths.is_empty(), "none is not checked, so this resolves");
    }

    #[test]
    fn a_scope_resolves_each_operator_the_way_list_does() {
        let store = operator_fixture();
        let parse = |all: &[&str], any: &[&str], none: &[&str]| {
            crate::tags::Scope::parse(
                &all.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                &any.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                &none.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            )
            .unwrap()
        };

        assert_eq!(
            scoped_paths(&store, &parse(&["type/beast"], &[], &[])),
            vec!["draft.md", "wolf.md"]
        );
        assert_eq!(
            scoped_paths(&store, &parse(&[], &["type/undead", "status/draft"], &[])),
            vec!["draft.md", "wight.md"]
        );
        assert_eq!(
            scoped_paths(&store, &parse(&["type/"], &[], &["status/draft"])),
            vec!["wight.md", "wolf.md"]
        );
        assert_eq!(
            scoped_paths(&store, &parse(&["type/beast", "status/draft"], &[], &[])),
            vec!["draft.md"]
        );
    }

    #[test]
    fn a_scope_reads_a_subtree_and_a_case_folded_term() {
        let store = operator_fixture();
        // `type/` takes the descendants; the bare `type` is a tag no note
        // carries, and its own subtree is what the error names instead.
        let subtree = crate::tags::Scope::parse(&["TYPE/".to_string()], &[], &[]).unwrap();
        assert_eq!(
            scoped_paths(&store, &subtree),
            vec!["draft.md", "wight.md", "wolf.md"]
        );
        let folded = crate::tags::Scope::parse(&["Habitat/Swamp".to_string()], &[], &[]).unwrap();
        assert_eq!(scoped_paths(&store, &folded), vec!["wight.md"]);
    }

    #[test]
    fn a_scope_naming_no_tag_errors_and_names_the_nearest() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::parse(&["type/undeed".to_string()], &[], &[]).unwrap();
        let err = store.files_in_scope(&filter).unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such tag 'type/undeed'; nearest: 'type/undead'"
        );
    }

    #[test]
    fn a_scope_excluding_an_unknown_tag_is_not_an_error() {
        let store = operator_fixture();
        let filter =
            crate::tags::Scope::parse(&["type/".to_string()], &[], &["nowhere".to_string()])
                .unwrap();
        assert_eq!(
            scoped_paths(&store, &filter),
            vec!["draft.md", "wight.md", "wolf.md"]
        );
    }

    #[test]
    fn a_scope_that_no_note_satisfies_is_empty_and_not_an_error() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::parse(
            &["type/undead".to_string(), "status/draft".to_string()],
            &[],
            &[],
        )
        .unwrap();
        assert!(store.files_in_scope(&filter).unwrap().is_empty());
    }

    #[test]
    fn an_unknown_all_term_errors_and_names_the_nearest_tag() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::parse(&["type/undeed".to_string()], &[], &[]).unwrap();
        let err = store.list_files(&filter, None, Some(20)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such tag 'type/undeed'; nearest: 'type/undead'"
        );
    }

    #[test]
    fn a_term_that_runs_past_an_existing_tag_names_that_ancestor() {
        let store = operator_fixture();
        // Two segments past `type/undead`, which the fixture holds — too far
        // for the fuzzy tier, which compares one segment against tags that
        // share a parent, to reach first.
        let filter =
            crate::tags::Scope::parse(&["type/undead/wight/banshee".to_string()], &[], &[])
                .unwrap();
        let err = store.list_files(&filter, None, Some(20)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such tag 'type/undead/wight/banshee'; nearest: 'type/undead'"
        );
    }

    #[test]
    fn an_unknown_term_with_no_near_neighbour_errors_without_a_suggestion() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::parse(&[], &["zzzzz".to_string()], &[]).unwrap();
        let err = store.list_files(&filter, None, Some(20)).unwrap_err();
        assert_eq!(err.to_string(), "no such tag 'zzzzz'");
    }

    #[test]
    fn an_unknown_subtree_term_prints_its_marker() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::parse(&["nowhere/".to_string()], &[], &[]).unwrap();
        let err = store.list_files(&filter, None, Some(20)).unwrap_err();
        assert_eq!(err.to_string(), "no such tag 'nowhere/'");
    }

    #[test]
    fn an_unknown_none_term_is_not_an_error() {
        let store = operator_fixture();
        let filter =
            crate::tags::Scope::parse(&["type/".to_string()], &[], &["nowhere".to_string()])
                .unwrap();
        assert_eq!(
            listed_paths(&store, &filter),
            vec!["draft.md", "wight.md", "wolf.md"]
        );
    }

    #[test]
    fn an_exact_term_errors_on_a_bare_axis_and_a_subtree_term_matches_below_it() {
        let store = operator_fixture();
        // The fixture holds `type/undead` and `type/beast`, never bare `type`.
        // The likeliest mistake is forgetting the subtree marker, so the
        // suggestion is the term with it added, not an unrelated fuzzy match.
        let exact = crate::tags::Scope::parse(&["type".to_string()], &[], &[]).unwrap();
        let err = store.list_files(&exact, None, Some(20)).unwrap_err();
        assert_eq!(err.to_string(), "no such tag 'type'; nearest: 'type/'");

        let subtree = crate::tags::Scope::parse(&["type/".to_string()], &[], &[]).unwrap();
        assert_eq!(
            listed_paths(&store, &subtree),
            vec!["draft.md", "wight.md", "wolf.md"]
        );
    }

    #[test]
    fn a_bare_axis_with_one_child_still_gets_the_subtree_suggestion() {
        let store = operator_fixture();
        // `habitat/swamp` is the fixture's only `habitat` tag, one segment
        // under an axis no fuzzy match would reach.
        let filter = crate::tags::Scope::parse(&["habitat".to_string()], &[], &[]).unwrap();
        let err = store.list_files(&filter, None, Some(20)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "no such tag 'habitat'; nearest: 'habitat/'"
        );
    }

    #[test]
    fn all_terms_intersect_and_any_terms_union() {
        let store = operator_fixture();
        let all = crate::tags::Scope::parse(
            &["type/undead".to_string(), "habitat/swamp".to_string()],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(listed_paths(&store, &all), vec!["wight.md"]);

        let any = crate::tags::Scope::parse(
            &[],
            &["type/undead".to_string(), "status/draft".to_string()],
            &[],
        )
        .unwrap();
        assert_eq!(listed_paths(&store, &any), vec!["draft.md", "wight.md"]);
    }

    #[test]
    fn a_none_term_removes_a_note_the_other_fields_returned() {
        let store = operator_fixture();
        let filter =
            crate::tags::Scope::parse(&["type/".to_string()], &[], &["status/draft".to_string()])
                .unwrap();
        assert_eq!(listed_paths(&store, &filter), vec!["wight.md", "wolf.md"]);
    }

    #[test]
    fn the_three_fields_combine_in_one_query() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::parse(
            &["type/".to_string()],
            &["habitat/swamp".to_string(), "status/draft".to_string()],
            &["status/draft".to_string()],
        )
        .unwrap();
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
        let filter = crate::tags::Scope::parse(&["type/".to_string()], &[], &[]).unwrap();
        assert_eq!(listed_paths(&store, &filter), vec!["inside.md"]);
    }

    #[test]
    fn an_empty_filter_returns_every_note() {
        let store = operator_fixture();
        let filter = crate::tags::Scope::default();
        assert_eq!(listed_paths(&store, &filter).len(), 3);
    }

    fn scoped(
        store: &Store,
        property: Option<&str>,
        to: Option<&str>,
        from: Option<&str>,
    ) -> Vec<String> {
        let scope = crate::tags::Scope::default()
            .with_filters(property, to, from)
            .unwrap();
        store
            .list_files(&scope, None, None)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect()
    }

    #[test]
    fn a_property_term_selects_by_name_and_by_value() {
        let (store, ..) = property_vault();
        assert_eq!(
            scoped(&store, Some("status"), None, None),
            ["acme.md", "ada.md"]
        );
        assert_eq!(
            scoped(&store, Some("status=active"), None, None),
            ["acme.md"]
        );
        assert_eq!(
            scoped(&store, Some("mentor"), None, None),
            ["ada.md"],
            "a body row counts"
        );
        assert!(scoped(&store, Some("status=gone"), None, None).is_empty());
    }

    #[test]
    fn links_to_alone_reads_edges_and_with_a_property_reads_the_name() {
        let (store, ..) = property_vault();
        assert_eq!(
            scoped(&store, None, Some("acme"), None),
            ["ada.md", "bob.md"]
        );
        assert_eq!(
            scoped(&store, Some("employer"), Some("acme"), None),
            ["ada.md"]
        );
        assert!(scoped(&store, Some("mentor"), Some("acme"), None).is_empty());
        assert_eq!(
            scoped(&store, Some("employer=acme"), Some("acme"), None),
            ["ada.md"]
        );
    }

    #[test]
    fn linked_from_alone_reads_edges_and_with_a_property_reads_the_name() {
        let (store, ..) = property_vault();
        assert_eq!(
            scoped(&store, None, None, Some("ada")),
            ["acme.md", "bob.md"]
        );
        assert_eq!(
            scoped(&store, Some("mentor"), None, Some("ada")),
            ["bob.md"]
        );
        assert_eq!(
            scoped(&store, Some("employer"), None, Some("ada")),
            ["acme.md"]
        );
    }

    #[test]
    fn the_filters_and_together_with_a_tag_term() {
        let (store, ada, _acme, bob) = property_vault();
        let tag = |p: &str| crate::tags::Tag {
            path: p.to_string(),
            display: p.to_string(),
        };
        // ada and bob are the people; acme is the firm. So the tag term
        // alone answers ada and bob, and `status` alone answers ada and
        // acme: each term admits a note the other refuses, and only the
        // AND of the two answers ada by herself.
        store
            .reconcile_file_tags(ada, &[tag("type/person")])
            .unwrap();
        store
            .reconcile_file_tags(bob, &[tag("type/person")])
            .unwrap();
        let scope = crate::tags::Scope::parse(&["type/person".into()], &[], &[])
            .unwrap()
            .with_filters(Some("status"), None, None)
            .unwrap();
        let got: Vec<String> = store
            .list_files(&scope, None, None)
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        assert_eq!(got, ["ada.md"]);
        assert_eq!(store.files_in_scope(&scope).unwrap(), vec![ada]);

        // Two link terms AND together the same way: bob is the note that
        // links to acme and that ada links to. `links_to` alone answers ada
        // and bob, `linked_from` alone answers acme and bob.
        assert_eq!(scoped(&store, None, Some("acme"), Some("ada")), ["bob.md"]);
    }

    #[test]
    fn an_unknown_link_note_errors_with_the_nearest_one() {
        let (store, ..) = property_vault();
        let scope = crate::tags::Scope::default()
            .with_filters(None, Some("acmee"), None)
            .unwrap();
        let err = store
            .list_files(&scope, None, None)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("no such note 'acmee'"), "{err}");
        assert!(err.contains("nearest: 'acme.md'"), "{err}");
        let scope = crate::tags::Scope::default()
            .with_filters(None, None, Some("zzzzzz"))
            .unwrap();
        let err = store.files_in_scope(&scope).unwrap_err().to_string();
        assert_eq!(err, "no such note 'zzzzzz' for 'linked_from'");
    }
}
