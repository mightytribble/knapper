//! The `tags` and `file_tags` tables: the vocabulary and which note carries what.

use super::Store;
use anyhow::Result;
use rusqlite::params;

/// A row of the vault's tag vocabulary (#60).
///
/// `note_count` is the notes carrying this exact tag. The listing is flat, so a
/// caller wanting a subtree total adds the rows it asked for.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagCount {
    pub path: String,
    /// The vault's own spelling, and absent when that is the path itself
    /// (#120). A vault that never capitalises a tag carries the field on no
    /// row. Read it through [`TagCount::as_written`], which answers either way.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    pub note_count: usize,
}

impl TagCount {
    /// The tag as the vault wrote it: the display form where there is one,
    /// else the path, which is what the vault wrote.
    pub fn as_written(&self) -> &str {
        self.display.as_deref().unwrap_or(&self.path)
    }
}

impl Store {
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

    /// How many notes carry any tag at all — the denominator `top_tags` counts
    /// against (#138).
    ///
    /// Without it a head count reads against `total_files` and overstates the
    /// vocabulary's reach, which is the difference between a tag filter being
    /// worth reaching for and not.
    pub fn tagged_file_count(&self) -> Result<usize> {
        let count: i64 =
            self.conn
                .query_row("SELECT COUNT(DISTINCT file_id) FROM file_tags", [], |row| {
                    row.get(0)
                })?;
        Ok(count as usize)
    }

    /// The vault's vocabulary, whole or under one term (#60).
    ///
    /// A prefix names a subtree whether or not it carries the `/` marker,
    /// because `--under` has only the subtree reading to give.
    ///
    /// A tag with no notes does not exist: `prune_unused_tags` deletes the row
    /// the last note released.
    pub fn tags_under(&self, prefix: Option<&crate::tags::TagTerm>) -> Result<Vec<TagCount>> {
        let (clause, args) = match prefix {
            Some(term) => {
                let subtree = crate::tags::TagTerm::Subtree(term.path().to_string());
                let (pred, args) = crate::tags::predicate(&subtree);
                (format!("WHERE {pred}"), args)
            }
            None => (String::new(), Vec::new()),
        };
        let mut stmt = self.conn.prepare(&format!(
            "SELECT t.path, t.display, COUNT(ft.file_id) AS notes
               FROM tags t JOIN file_tags ft ON ft.tag_id = t.id
               {clause}
              GROUP BY t.id ORDER BY t.path"
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), |row| {
            let path: String = row.get(0)?;
            let display: String = row.get(1)?;
            Ok(TagCount {
                // The display form earns a field only where it differs (#120).
                display: (display != path).then_some(display),
                path,
                note_count: row.get::<_, i64>(2)? as usize,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
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
    /// `type/undead` are one row whichever spelling arrives, and every reader
    /// of a `tags::TagTerm` — the `list_files` tag filter, `tags_under` —
    /// meets a folded column, because `parse_term` folds before the term
    /// exists.
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

    /// The axes this vault holds, and how many notes each covers (#60).
    ///
    /// knapper names no axis. This reports what the vault wrote: the first
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::store::fixtures::*;

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
    fn tagged_file_count_counts_a_note_once_however_many_tags_it_carries() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 1, "d000001", None, None)
            .unwrap();
        let b = store
            .insert_file("b.md", "h", 1, "d000002", None, None)
            .unwrap();
        store
            .insert_file("untagged.md", "h", 1, "d000003", None, None)
            .unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        store
            .reconcile_file_tags(a, &[tag("dimension/sol-prime"), tag("status/active")])
            .unwrap();
        store
            .reconcile_file_tags(b, &[tag("status/active")])
            .unwrap();

        assert_eq!(store.tagged_file_count().unwrap(), 2);
    }

    #[test]
    fn a_tag_carrying_no_slash_is_an_axis_of_its_own() {
        // The axis is the path's first segment (`tags::Tag`), and a tag with
        // no separator is entirely its own first segment. `vault_map` counts
        // these rows, so a flat vocabulary has to count as many axes as it has
        // tags rather than collapse to one (#138).
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 1, "d000001", None, None)
            .unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        store
            .reconcile_file_tags(
                a,
                &[
                    tag("dimension/sol-prime"),
                    tag("dimension/earth"),
                    tag("status/active"),
                    tag("draft"),
                ],
            )
            .unwrap();

        let rows = store.tag_axes().unwrap();
        let axes: Vec<&str> = rows.iter().map(|(a, _)| a.as_str()).collect();
        assert_eq!(axes, vec!["dimension", "draft", "status"]);
    }

    #[test]
    fn a_tag_no_note_carries_is_not_an_axis() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("a.md", "h", 1, "d000001", None, None)
            .unwrap();
        let tag = |p: &str| crate::tags::Tag {
            path: p.into(),
            display: p.into(),
        };
        store
            .reconcile_file_tags(a, &[tag("dimension/sol-prime"), tag("status/active")])
            .unwrap();
        // The note releases one axis outright.
        store
            .reconcile_file_tags(a, &[tag("dimension/sol-prime")])
            .unwrap();

        assert_eq!(
            store.tag_axes().unwrap(),
            vec![("dimension".to_string(), 1)]
        );
    }

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
        store
            .reconcile_file_tags(one, std::slice::from_ref(&swamp))
            .unwrap();
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
        // And the folded query side meets a folded column: the subtree arm,
        assert_eq!(
            store
                .list_files(
                    &crate::tags::Scope::parse(&["type/".to_string()], &[], &[]).unwrap(),
                    None,
                    Some(10),
                )
                .unwrap()
                .len(),
            2
        );
        // and the exact arm.
        assert_eq!(
            store
                .list_files(
                    &crate::tags::Scope::parse(&["TYPE/UNDEAD".to_string()], &[], &[]).unwrap(),
                    None,
                    Some(10),
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
                &crate::tags::Scope::parse(&["alpha".to_string(), "beta".to_string()], &[], &[])
                    .unwrap(),
                None,
                Some(10),
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "both.md");

        // Obsidian matches a tag without regard to case.
        let folded = store
            .list_files(
                &crate::tags::Scope::parse(&["ALPHA".to_string()], &[], &[]).unwrap(),
                None,
                Some(10),
            )
            .unwrap();
        assert_eq!(folded.len(), 2);
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

    #[test]
    fn the_whole_vocabulary_comes_back_in_path_order() {
        let store = operator_fixture();
        let paths: Vec<String> = store
            .tags_under(None)
            .unwrap()
            .into_iter()
            .map(|t| t.path)
            .collect();
        assert_eq!(
            paths,
            vec!["habitat/swamp", "status/draft", "type/beast", "type/undead"]
        );
    }

    #[test]
    fn a_subtree_prefix_returns_the_subtree_and_counts_each_exact_tag() {
        let store = operator_fixture();
        let rows = store
            .tags_under(Some(&crate::tags::parse_term("type/").unwrap()))
            .unwrap();
        let counted: Vec<(String, usize)> =
            rows.into_iter().map(|t| (t.path, t.note_count)).collect();
        // `type/beast` is on two notes; the count is per exact tag, not a
        // subtree total.
        assert_eq!(
            counted,
            vec![
                ("type/beast".to_string(), 2),
                ("type/undead".to_string(), 1)
            ]
        );
    }

    #[test]
    fn a_bare_exact_prefix_answers_the_same_subtree() {
        let store = operator_fixture();
        let bare = store
            .tags_under(Some(&crate::tags::parse_term("type").unwrap()))
            .unwrap();
        let slash = store
            .tags_under(Some(&crate::tags::parse_term("type/").unwrap()))
            .unwrap();
        let paths = |rows: Vec<TagCount>| -> Vec<(String, usize)> {
            rows.into_iter().map(|t| (t.path, t.note_count)).collect()
        };
        assert_eq!(paths(bare), paths(slash));
    }

    #[test]
    fn a_prefix_that_names_a_tag_returns_that_tag_too() {
        let store = Store::open_memory().unwrap();
        let one = store
            .insert_file("one.md", "h", 1, "d000001", None, None)
            .unwrap();
        store
            .reconcile_file_tags(
                one,
                &[
                    crate::tags::Tag {
                        path: "type".into(),
                        display: "Type".into(),
                    },
                    crate::tags::Tag {
                        path: "type/undead".into(),
                        display: "Type/Undead".into(),
                    },
                ],
            )
            .unwrap();
        let rows = store
            .tags_under(Some(&crate::tags::parse_term("type/").unwrap()))
            .unwrap();
        let displays: Vec<String> = rows.iter().map(|t| t.as_written().to_string()).collect();
        // The vault's own spelling comes back, not the folded path.
        assert_eq!(displays, vec!["Type", "Type/Undead"]);
    }

    /// One tag the vault capitalises and one it does not.
    fn mixed_case_fixture() -> Store {
        let store = Store::open_memory().unwrap();
        let one = store
            .insert_file("one.md", "h", 1, "d000001", None, None)
            .unwrap();
        store
            .reconcile_file_tags(
                one,
                &[
                    crate::tags::Tag {
                        path: "type/undead".into(),
                        display: "Type/Undead".into(),
                    },
                    crate::tags::Tag {
                        path: "active-threat".into(),
                        display: "active-threat".into(),
                    },
                ],
            )
            .unwrap();
        store
    }

    #[test]
    fn a_tag_the_vault_spells_as_its_path_holds_no_separate_display_form() {
        let store = mixed_case_fixture();
        let rows = store.tags_under(None).unwrap();
        let forms: Vec<(&str, Option<&str>)> = rows
            .iter()
            .map(|t| (t.path.as_str(), t.display.as_deref()))
            .collect();
        assert_eq!(
            forms,
            vec![
                ("active-threat", None),
                ("type/undead", Some("Type/Undead"))
            ]
        );
        // A caller asking for the spelling gets one either way.
        assert_eq!(rows[0].as_written(), "active-threat");
        assert_eq!(rows[1].as_written(), "Type/Undead");
    }

    #[test]
    fn a_serialised_row_carries_display_only_where_it_differs_from_the_path() {
        let store = mixed_case_fixture();
        let rows = store.tags_under(None).unwrap();
        let json = serde_json::to_value(&rows).unwrap();
        assert!(
            json[0].get("display").is_none(),
            "a tag written as its path pays for no display field: {}",
            json[0]
        );
        assert_eq!(json[1]["display"], "Type/Undead");
    }

    #[test]
    fn a_prefix_matching_nothing_returns_no_rows() {
        let store = operator_fixture();
        let rows = store
            .tags_under(Some(&crate::tags::parse_term("nowhere/").unwrap()))
            .unwrap();
        assert!(rows.is_empty());
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
        let subtree = crate::tags::Scope::parse(&["type/".to_string()], &[], &[]).unwrap();
        assert_eq!(
            listed_paths(&store, &subtree),
            vec!["beast.md", "plain.md", "undead.md"]
        );

        // The exact query is the tag a note carries and no descendant.
        let exact = crate::tags::Scope::parse(&["type".to_string()], &[], &[]).unwrap();
        assert_eq!(listed_paths(&store, &exact), vec!["plain.md"]);
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

        let subtree = crate::tags::Scope::parse(&["type_a/".to_string()], &[], &[]).unwrap();
        assert_eq!(listed_paths(&store, &subtree), vec!["exact.md", "one.md"]);

        // The exact arm still answers for the tag itself.
        let exact = crate::tags::Scope::parse(&["type_a".to_string()], &[], &[]).unwrap();
        assert_eq!(listed_paths(&store, &exact), vec!["exact.md"]);
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
