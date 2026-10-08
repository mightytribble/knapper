//! The `aliases` table: the other names a note's frontmatter records (#142).

use super::Store;
use super::files::{FILE_COLUMNS, file_from_row};
use super::{FileRecord, group_by_file, id_array};
use crate::fault::Fault;
use anyhow::Result;
use rusqlite::params;

impl Store {
    /// Make `aliases` the aliases of this file (#142).
    ///
    /// Owns the file's rows the way `replace_file_properties` owns its
    /// property rows: delete, then insert. The edge pass calls it once per
    /// file. Rows go in the order given, which is the order `aliases_for_files`
    /// answers. A second spelling of an alias the list already holds is
    /// dropped, the rule `aliases::extract` follows.
    pub fn replace_file_aliases(&self, file_id: i64, aliases: &[String]) -> Result<()> {
        self.conn
            .execute("DELETE FROM aliases WHERE file_id = ?1", params![file_id])?;
        let mut stmt = self.conn.prepare(
            "INSERT OR IGNORE INTO aliases (file_id, folded, display) VALUES (?1, ?2, ?3)",
        )?;
        for alias in aliases {
            stmt.execute(params![file_id, crate::aliases::fold(alias), alias])?;
        }
        Ok(())
    }

    /// The note that carries `alias` (#142), compared case-folded on the
    /// whole alias.
    ///
    /// An alias that more than one note carries is an error naming every
    /// candidate, the rule `find_file_by_fuzzy` follows for equidistant
    /// basenames: picking one would answer with a note the caller may not
    /// have meant, and nothing in the reply would say so.
    pub fn find_file_by_alias(&self, alias: &str) -> Result<Option<FileRecord>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {FILE_COLUMNS}
               FROM files f JOIN aliases a ON a.file_id = f.id
              WHERE a.folded = ?1
              ORDER BY f.path"
        ))?;
        let mut found = stmt
            .query_map(params![crate::aliases::fold(alias)], file_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        // A live note's alias beats an archived note's (#151).
        if found.len() > 1 && found.iter().any(|f| !self.is_archived(&f.path)) {
            found.retain(|f| !self.is_archived(&f.path));
        }
        match found.len() {
            0 | 1 => Ok(found.pop()),
            _ => Err(anyhow::anyhow!(Fault::Ambiguous(format!(
                "ambiguous alias '{}': carried by [{}]",
                alias,
                found
                    .iter()
                    .map(|f| f.path.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )))),
        }
    }

    /// The aliases of each listed note, as the note wrote them and in its
    /// order, keyed by file id (#142).
    ///
    /// One query for a whole listing. A note with no alias has no entry.
    pub fn aliases_for_files(
        &self,
        file_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, Vec<String>>> {
        let mut stmt = self.conn.prepare(
            "SELECT file_id, display FROM aliases
              WHERE file_id IN rarray(?1)
              ORDER BY file_id, id",
        )?;
        group_by_file(stmt.query_map(params![id_array(file_ids)], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::store::fixtures::*;

    #[test]
    fn an_alias_finds_its_note_whatever_the_case() {
        let store = Store::open_memory().unwrap();
        let id = store
            .insert_file("npcs/samantha-hoyle.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store
            .replace_file_aliases(id, &aliases(&["Sam", "Élodie"]))
            .unwrap();

        let found = store.find_file_by_alias("SAM").unwrap().unwrap();
        assert_eq!(found.path, "npcs/samantha-hoyle.md");
        let found = store.find_file_by_alias("éLODIE").unwrap().unwrap();
        assert_eq!(found.path, "npcs/samantha-hoyle.md");
    }

    #[test]
    fn an_alias_matches_on_the_whole_alias_only() {
        let store = Store::open_memory().unwrap();
        let id = store
            .insert_file("npcs/samantha-hoyle.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store
            .replace_file_aliases(id, &aliases(&["Drone 7-422"]))
            .unwrap();

        assert!(store.find_file_by_alias("7-422").unwrap().is_none());
        assert!(store.find_file_by_alias("Drone").unwrap().is_none());
        assert!(store.find_file_by_alias("").unwrap().is_none());
    }

    #[test]
    fn an_alias_two_notes_carry_is_refused_and_names_both() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("npcs/samantha-hoyle.md", "h1", 0, "aaa111", None, None)
            .unwrap();
        let b = store
            .insert_file("npcs/sam-okafor.md", "h2", 0, "bbb222", None, None)
            .unwrap();
        store.replace_file_aliases(a, &aliases(&["Sam"])).unwrap();
        store.replace_file_aliases(b, &aliases(&["sam"])).unwrap();

        let err = store.find_file_by_alias("Sam").unwrap_err().to_string();
        assert!(err.contains("npcs/samantha-hoyle.md"), "{err}");
        assert!(err.contains("npcs/sam-okafor.md"), "{err}");
    }

    #[test]
    fn replacing_a_note_s_aliases_releases_the_old_ones() {
        let store = Store::open_memory().unwrap();
        let id = store
            .insert_file("npcs/samantha-hoyle.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store.replace_file_aliases(id, &aliases(&["Sam"])).unwrap();
        store
            .replace_file_aliases(id, &aliases(&["Dragon"]))
            .unwrap();

        assert!(store.find_file_by_alias("Sam").unwrap().is_none());
        assert!(store.find_file_by_alias("Dragon").unwrap().is_some());
    }

    #[test]
    fn a_file_removal_cascades_its_alias_rows() {
        let store = Store::open_memory().unwrap();
        let id = store
            .insert_file("npcs/samantha-hoyle.md", "h", 0, "aaa111", None, None)
            .unwrap();
        store.replace_file_aliases(id, &aliases(&["Sam"])).unwrap();
        store.delete_file(id).unwrap();

        let rows: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM aliases", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }

    #[test]
    fn an_alias_two_notes_carry_is_ambiguous() {
        let store = Store::open_memory().unwrap();
        for (path, docid) in [("a.md", "aaa111"), ("b.md", "bbb222")] {
            let id = store
                .insert_file(path, "h", 100, docid, None, None)
                .unwrap();
            store.replace_file_aliases(id, &aliases(&["Twin"])).unwrap();
        }
        let err = store.find_file_by_alias("Twin").unwrap_err();
        assert_eq!(
            crate::fault::Fault::of(&err).map(|f| f.kind()),
            Some("ambiguous")
        );
        assert!(
            err.to_string().starts_with("ambiguous alias 'Twin'"),
            "{err}"
        );
    }

    #[test]
    fn aliases_for_files_answers_each_note_s_aliases_as_written_and_in_order() {
        let store = Store::open_memory().unwrap();
        let a = store
            .insert_file("npcs/samantha-hoyle.md", "h1", 0, "aaa111", None, None)
            .unwrap();
        let b = store
            .insert_file("npcs/jeanine-wang.md", "h2", 0, "bbb222", None, None)
            .unwrap();
        let c = store
            .insert_file("npcs/no-aliases.md", "h3", 0, "ccc333", None, None)
            .unwrap();
        store
            .replace_file_aliases(a, &aliases(&["Sam", "Dragon"]))
            .unwrap();
        store
            .replace_file_aliases(b, &aliases(&["El Ja'nadine", "Empress"]))
            .unwrap();

        let by_file = store.aliases_for_files(&[a, b, c]).unwrap();
        assert_eq!(by_file[&a], aliases(&["Sam", "Dragon"]));
        assert_eq!(by_file[&b], aliases(&["El Ja'nadine", "Empress"]));
        assert!(!by_file.contains_key(&c));
    }

    /// An alias a live note and an archived note both carry names the live
    /// one, not two notes (#151).
    #[test]
    fn an_alias_a_live_and_an_archived_note_share_resolves_to_the_live_one() {
        let store = Store::open_memory()
            .unwrap()
            .with_archive_folder("04-Archive");
        let live = store
            .insert_file("people/sam.md", "h", 1, "f00001", None, None)
            .unwrap();
        let old = store
            .insert_file("04-Archive/sam-old.md", "h", 1, "f00002", None, None)
            .unwrap();
        store
            .replace_file_aliases(live, &["Sam".to_string()])
            .unwrap();
        store
            .replace_file_aliases(old, &["Sam".to_string()])
            .unwrap();
        assert_eq!(
            store.find_file_by_alias("sam").unwrap().unwrap().path,
            "people/sam.md"
        );
    }
}
