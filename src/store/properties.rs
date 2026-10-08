//! The `properties` table: one row per custom-property value a note carries (#66).

use super::Store;
use super::{DOC_LEVEL, group_by_file, id_array};
use anyhow::Result;
use rusqlite::params;

/// A property row as it is written (#66). One argument rather than five.
pub struct NewProperty<'a> {
    /// [`DOC_LEVEL`] for a frontmatter property, else the chunk's `seq`.
    pub chunk_seq: i64,
    pub name: &'a str,
    pub value: &'a str,
    pub kind: crate::properties::Kind,
    /// The note a `link` row resolves to, or none when it does not resolve.
    pub target_file: Option<i64>,
}

/// A property row as it is read (#66).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct PropertyRow {
    pub chunk_seq: i64,
    /// The breadcrumb of the chunk that holds a body row. Absent on a
    /// frontmatter row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading_path: Option<String>,
    pub name: String,
    pub value: String,
    pub kind: crate::properties::Kind,
    /// The path of the note a `link` row resolves to. Absent on a row of
    /// another kind, and on a link that resolves to nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,
}

/// One row of the property registry (#66): a name, how many notes carry it,
/// the kinds seen, and Obsidian's declared type when `types.json` names it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct PropertyCount {
    pub name: String,
    pub note_count: usize,
    pub kinds: Vec<crate::properties::Kind>,
    /// Filled by `properties::registry`, which reads the vault. The store
    /// does not.
    pub declared_type: Option<String>,
}

/// One distinct value of one property, and how many notes carry it (#66).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ValueCount {
    pub value: String,
    pub kind: crate::properties::Kind,
    pub note_count: usize,
}

/// The join every property reader runs, from `properties p` to the chunk
/// that holds a body row and the note a link row names.
///
/// Both [`Store::file_properties`] and [`Store::doc_properties_for_files`]
/// compose their `SELECT` and `WHERE` around this one copy, so a column or
/// join added here reaches both readers at once.
const PROPERTIES_JOIN_SQL: &str = "FROM properties p
       LEFT JOIN chunks c ON c.file_id = p.file_id AND c.seq = p.chunk_seq
       LEFT JOIN files t ON t.id = p.target_file";

/// One row of that join, read from the six columns starting at `base`:
/// `chunk_seq, heading_path, name, value, kind, target_path`.
fn property_row_at(row: &rusqlite::Row<'_>, base: usize) -> rusqlite::Result<PropertyRow> {
    let kind: String = row.get(base + 4)?;
    Ok(PropertyRow {
        chunk_seq: row.get(base)?,
        heading_path: row.get(base + 1)?,
        name: row.get(base + 2)?,
        value: row.get(base + 3)?,
        // A kind this build does not know reads as text: it is still a value.
        kind: crate::properties::Kind::parse(&kind).unwrap_or(crate::properties::Kind::Text),
        target_path: row.get(base + 5)?,
    })
}

/// One note's row: the six columns alone.
fn property_row_from(row: &rusqlite::Row<'_>) -> rusqlite::Result<PropertyRow> {
    property_row_at(row, 0)
}

/// A batched reader's row: `p.file_id` and then the six.
fn keyed_property_row_from(row: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, PropertyRow)> {
    Ok((row.get(0)?, property_row_at(row, 1)?))
}

impl Store {
    /// Replace one file's property rows (#66).
    ///
    /// Owns the file's rows the way `reconcile_file_tags` owns its tag rows:
    /// delete, then insert. The edge pass calls this once per file.
    pub fn replace_file_properties(&self, file_id: i64, rows: &[NewProperty<'_>]) -> Result<()> {
        self.conn.execute(
            "DELETE FROM properties WHERE file_id = ?1",
            params![file_id],
        )?;
        let mut stmt = self.conn.prepare(
            "INSERT INTO properties (file_id, chunk_seq, name, value, kind, target_file)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for row in rows {
            stmt.execute(params![
                file_id,
                row.chunk_seq,
                row.name,
                row.value,
                row.kind.as_str(),
                row.target_file
            ])?;
        }
        Ok(())
    }

    /// Every property row one note holds, frontmatter first, then by chunk.
    pub fn file_properties(&self, file_id: i64) -> Result<Vec<PropertyRow>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT p.chunk_seq, c.heading_path, p.name, p.value, p.kind, t.path
               {PROPERTIES_JOIN_SQL}
              WHERE p.file_id = ?1 ORDER BY p.chunk_seq, p.name, p.id"
        ))?;
        let rows = stmt.query_map(params![file_id], property_row_from)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// The frontmatter rows of each listed note, keyed by file id (#66).
    ///
    /// One query for a whole result set. A note with no frontmatter row has
    /// no entry.
    pub fn doc_properties_for_files(
        &self,
        file_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, Vec<PropertyRow>>> {
        let array = id_array(file_ids);
        let mut stmt = self.conn.prepare(&format!(
            "SELECT p.file_id, p.chunk_seq, c.heading_path, p.name, p.value, p.kind, t.path
               {PROPERTIES_JOIN_SQL}
              WHERE p.chunk_seq = {DOC_LEVEL} AND p.file_id IN rarray(?1)
              ORDER BY p.file_id, p.name, p.id"
        ))?;
        group_by_file(stmt.query_map(params![array], keyed_property_row_from)?)
    }

    /// The rows of each listed note that a scope's `property` term matched,
    /// keyed by file id (#66).
    ///
    /// One query for a whole listing, the way `doc_properties_for_files` is
    /// one query for a whole result set. The predicate is `scope_clauses`'s
    /// own: the name, the value when the term carries one, and the link
    /// target when a `links_to` term narrowed it. So the rows answered are
    /// the rows that admitted the note, and no other row the note holds.
    ///
    /// A note with no matching row has no entry.
    pub fn matched_properties_for_files(
        &self,
        file_ids: &[i64],
        term: &crate::tags::PropertyTerm,
        target_file: Option<i64>,
    ) -> Result<std::collections::HashMap<i64, Vec<PropertyRow>>> {
        let array = id_array(file_ids);
        let mut args: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(array), Box::new(term.name.clone())];
        let mut sql = format!(
            "SELECT p.file_id, p.chunk_seq, c.heading_path, p.name, p.value, p.kind, t.path
               {PROPERTIES_JOIN_SQL}
              WHERE p.file_id IN rarray(?) AND p.name = ?"
        );
        if let Some(value) = &term.value {
            sql.push_str(" AND p.value = ?");
            args.push(Box::new(value.clone()));
        }
        if let Some(target) = target_file {
            sql.push_str(" AND p.target_file = ?");
            args.push(Box::new(target));
        }
        sql.push_str(" ORDER BY p.file_id, p.chunk_seq, p.name, p.id");
        let mut stmt = self.conn.prepare(&sql)?;
        group_by_file(stmt.query_map(
            rusqlite::params_from_iter(args.iter()),
            keyed_property_row_from,
        )?)
    }

    /// One row per property name (#66): how many notes carry it and the
    /// kinds seen, by note count and then name. `declared_type` is left
    /// empty; `properties::registry` fills it from the vault.
    ///
    /// Only the notes `scope` admits count, which by default leaves the archive out (#151).
    pub fn property_registry(&self, scope: &crate::tags::Scope) -> Result<Vec<PropertyCount>> {
        let (scope_sql, args) = self.scope_sql(scope)?;
        let mut stmt = self.conn.prepare(&format!(
            "SELECT p.name, COUNT(DISTINCT p.file_id) AS notes, GROUP_CONCAT(DISTINCT p.kind)
               FROM properties p JOIN files f ON f.id = p.file_id
              WHERE 1=1{scope_sql}
              GROUP BY p.name ORDER BY notes DESC, p.name"
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), |row| {
            let kinds: String = row.get(2)?;
            let mut kinds: Vec<crate::properties::Kind> = kinds
                .split(',')
                .filter_map(crate::properties::Kind::parse)
                .collect();
            kinds.sort_by_key(|k| k.as_str());
            Ok(PropertyCount {
                name: row.get(0)?,
                note_count: row.get::<_, i64>(1)? as usize,
                kinds,
                declared_type: None,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// One property's distinct values, each with its kind and the notes
    /// carrying it, by note count and then value (#66).
    ///
    /// Only the notes `scope` admits count, which by default leaves the archive out (#151).
    pub fn property_values(
        &self,
        name: &str,
        scope: &crate::tags::Scope,
    ) -> Result<Vec<ValueCount>> {
        let (scope_sql, scope_args) = self.scope_sql(scope)?;
        let mut args: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(name.to_string())];
        args.extend(scope_args);
        let mut stmt = self.conn.prepare(&format!(
            "SELECT p.value, p.kind, COUNT(DISTINCT p.file_id) AS notes
               FROM properties p JOIN files f ON f.id = p.file_id
              WHERE p.name = ?{scope_sql}
              GROUP BY p.value, p.kind ORDER BY notes DESC, p.value"
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), |row| {
            let kind: String = row.get(1)?;
            Ok(ValueCount {
                value: row.get(0)?,
                kind: crate::properties::Kind::parse(&kind)
                    .unwrap_or(crate::properties::Kind::Text),
                note_count: row.get::<_, i64>(2)? as usize,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// The property names one note files a link to another under (#66).
    /// Distinct, ordered. Empty when no property carries the link.
    pub fn property_names_for_link(&self, from_file: i64, to_file: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT name FROM properties
              WHERE file_id = ?1 AND target_file = ?2 ORDER BY name",
        )?;
        let rows = stmt.query_map(params![from_file, to_file], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Empty the table. The edge rebuild calls this beside `clear_edges`.
    pub fn clear_properties(&self) -> Result<()> {
        self.conn.execute("DELETE FROM properties", [])?;
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
    fn replacing_a_files_properties_is_idempotent() {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, &generate_docid("a.md"), None, None)
            .unwrap();
        let rows = [
            prop(DOC_LEVEL, "status", "draft", Kind::Text, None),
            prop(DOC_LEVEL, "rating", "5", Kind::Number, None),
        ];
        store.replace_file_properties(a, &rows).unwrap();
        store.replace_file_properties(a, &rows).unwrap();
        let got = store.file_properties(a).unwrap();
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0].name, "rating");
        assert_eq!(got[0].kind, Kind::Number);
        assert_eq!(got[0].chunk_seq, DOC_LEVEL);
        assert_eq!(got[0].heading_path, None);
        assert_eq!(got[1].value, "draft");
    }

    #[test]
    fn a_file_removal_cascades_its_property_rows() {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, &generate_docid("a.md"), None, None)
            .unwrap();
        store
            .replace_file_properties(a, &[prop(DOC_LEVEL, "status", "draft", Kind::Text, None)])
            .unwrap();
        store.delete_file(a).unwrap();
        let n: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM properties", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn a_target_removal_keeps_the_row_and_clears_the_target() {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, &generate_docid("a.md"), None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 0, &generate_docid("b.md"), None, None)
            .unwrap();
        store
            .replace_file_properties(a, &[prop(DOC_LEVEL, "employer", "b", Kind::Link, Some(b))])
            .unwrap();
        assert_eq!(
            store.file_properties(a).unwrap()[0].target_path.as_deref(),
            Some("b.md")
        );
        store.delete_file(b).unwrap();
        let got = store.file_properties(a).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].target_path, None);
        assert_eq!(got[0].value, "b");
    }

    #[test]
    fn the_registry_counts_a_note_once_and_lists_the_kinds_seen() {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, &generate_docid("a.md"), None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 0, &generate_docid("b.md"), None, None)
            .unwrap();
        store
            .replace_file_properties(
                a,
                &[
                    prop(DOC_LEVEL, "status", "draft", Kind::Text, None),
                    prop(DOC_LEVEL, "status", "review", Kind::Text, None),
                    prop(DOC_LEVEL, "rating", "5", Kind::Number, None),
                ],
            )
            .unwrap();
        store
            .replace_file_properties(b, &[prop(DOC_LEVEL, "rating", "high", Kind::Text, None)])
            .unwrap();
        let reg = store
            .property_registry(&crate::tags::Scope::default())
            .unwrap();
        assert_eq!(reg.len(), 2, "{reg:?}");
        // Ordered by note count, then name.
        assert_eq!(reg[0].name, "rating");
        assert_eq!(reg[0].note_count, 2);
        assert_eq!(reg[0].kinds, vec![Kind::Number, Kind::Text]);
        assert_eq!(reg[0].declared_type, None);
        assert_eq!(reg[1].name, "status");
        assert_eq!(reg[1].note_count, 1, "two values in one note count once");

        let vals = store
            .property_values("status", &crate::tags::Scope::default())
            .unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0].value, "draft");
        assert_eq!(vals[0].note_count, 1);
    }

    #[test]
    fn the_names_behind_a_link_are_distinct_and_ordered() {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, &generate_docid("a.md"), None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 0, &generate_docid("b.md"), None, None)
            .unwrap();
        store
            .replace_file_properties(
                a,
                &[
                    prop(DOC_LEVEL, "mentor", "b", Kind::Link, Some(b)),
                    prop(0, "employer", "b", Kind::Link, Some(b)),
                    prop(1, "employer", "b", Kind::Link, Some(b)),
                ],
            )
            .unwrap();
        assert_eq!(
            store.property_names_for_link(a, b).unwrap(),
            vec!["employer".to_string(), "mentor".to_string()]
        );
        assert!(store.property_names_for_link(b, a).unwrap().is_empty());
    }

    #[test]
    fn doc_properties_for_files_answers_frontmatter_rows_only() {
        use crate::properties::Kind;
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 0, &generate_docid("a.md"), None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 0, &generate_docid("b.md"), None, None)
            .unwrap();
        store
            .replace_file_properties(
                a,
                &[
                    prop(DOC_LEVEL, "status", "draft", Kind::Text, None),
                    prop(0, "mentor", "b", Kind::Link, Some(b)),
                ],
            )
            .unwrap();
        let map = store.doc_properties_for_files(&[a, b]).unwrap();
        assert_eq!(map.get(&a).map(Vec::len), Some(1));
        assert_eq!(map[&a][0].name, "status");
        assert!(!map.contains_key(&b));
        store.clear_properties().unwrap();
        assert!(store.doc_properties_for_files(&[a]).unwrap().is_empty());
    }

    /// The listing's fill reads the same predicate the clause selected on,
    /// so the rows it answers are the rows that matched (#66).
    #[test]
    fn matched_properties_narrow_by_name_value_and_target() {
        let (store, ada, acme, bob) = property_vault();
        let ids = [ada, acme, bob];
        let term = |written: &str| {
            crate::tags::Scope::default()
                .with_filters(Some(written), None, None)
                .unwrap()
                .property
                .unwrap()
        };
        let values = |map: &std::collections::HashMap<i64, Vec<PropertyRow>>, id: i64| {
            map.get(&id)
                .map(|rows| rows.iter().map(|r| r.value.clone()).collect::<Vec<_>>())
                .unwrap_or_default()
        };

        // Name alone: every row under that name, and nothing else the note
        // holds.
        let by_name = store
            .matched_properties_for_files(&ids, &term("status"), None)
            .unwrap();
        assert_eq!(values(&by_name, ada), ["draft"]);
        assert_eq!(values(&by_name, acme), ["active"]);
        assert!(!by_name.contains_key(&bob));

        // Name and value.
        let by_value = store
            .matched_properties_for_files(&ids, &term("status=active"), None)
            .unwrap();
        assert!(values(&by_value, ada).is_empty());
        assert_eq!(values(&by_value, acme), ["active"]);

        // Name and target: the rows that name that note, not every row
        // under the name.
        let by_target = store
            .matched_properties_for_files(&ids, &term("employer"), Some(acme))
            .unwrap();
        assert_eq!(values(&by_target, ada), ["acme"]);
        assert!(
            store
                .matched_properties_for_files(&ids, &term("employer"), Some(bob))
                .unwrap()
                .is_empty(),
            "no employer row names bob"
        );
    }
}
