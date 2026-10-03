//! The `migration_log` table: what a PARA migration moved, so `undo` can move it back.

use super::Store;
use anyhow::Result;
use rusqlite::{OptionalExtension, params};

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

impl Store {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
