//! The `identity_facts` table.

use super::Store;
use anyhow::Result;

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

impl Store {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
