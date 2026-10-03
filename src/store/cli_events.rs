//! The `cli_events` table: the audit log of write operations.

use super::Store;
use anyhow::Result;
use rusqlite::params;

/// A record representing a CLI event (for observability/analytics).
#[derive(Debug, Clone)]
pub struct CliEvent {
    pub id: i64,
    pub timestamp: String,
    pub operation: String,
    pub outcome: String,
    pub detail: Option<String>,
}

impl Store {
    /// Log a CLI event for observability/analytics.
    pub fn log_cli_event(
        &self,
        operation: &str,
        outcome: &str,
        detail: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO cli_events (timestamp, operation, outcome, detail)
             VALUES (datetime('now'), ?1, ?2, ?3)",
            params![operation, outcome, detail],
        )?;
        Ok(())
    }

    /// Get CLI events since a given ISO-8601 date string (e.g., "2020-01-01").
    pub fn get_cli_events_since(&self, since: &str) -> Result<Vec<CliEvent>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, operation, outcome, detail
             FROM cli_events WHERE timestamp >= ?1 ORDER BY timestamp DESC",
        )?;
        let rows = stmt.query_map(params![since], |row| {
            Ok(CliEvent {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                operation: row.get(2)?,
                outcome: row.get(3)?,
                detail: row.get(4)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Prune CLI events older than the given number of days.
    pub fn prune_cli_events(&self, days: u32) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM cli_events WHERE julianday('now') - julianday(timestamp) > ?1",
            params![days],
        )?;
        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_events_insert_and_query() {
        let store = Store::open_memory().unwrap();
        store.log_cli_event("edit", "success", None).unwrap();
        store
            .log_cli_event("edit", "fallback", Some("timeout"))
            .unwrap();
        let events = store.get_cli_events_since("2020-01-01").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].operation, "edit");
        assert_eq!(events[1].operation, "edit");
        // Most recent first
        assert_eq!(events[0].outcome, "fallback");
        assert_eq!(events[0].detail.as_deref(), Some("timeout"));
        assert_eq!(events[1].outcome, "success");
        assert!(events[1].detail.is_none());
    }

    #[test]
    fn test_cli_events_prune() {
        let store = Store::open_memory().unwrap();
        store.log_cli_event("search", "success", None).unwrap();
        // Events inserted just now should NOT be pruned with days=0 (julianday diff ~0)
        let pruned = store.prune_cli_events(1).unwrap();
        assert_eq!(pruned, 0);
        let events = store.get_cli_events_since("2020-01-01").unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_cli_events_table_exists() {
        let store = Store::open_memory().unwrap();
        let tables: Vec<String> = {
            let mut stmt = store
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='cli_events'")
                .unwrap();
            let rows = stmt.query_map([], |row| row.get(0)).unwrap();
            rows.filter_map(|r| r.ok()).collect()
        };
        assert!(tables.contains(&"cli_events".to_string()));
    }
}
