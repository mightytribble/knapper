//! The `folder_centroids` and `placement_corrections` tables.

use super::Store;
use anyhow::Result;
use rusqlite::params;

/// A record of a placement correction (user moved a note from suggested folder).
#[derive(Debug, Clone)]
pub struct PlacementCorrection {
    pub id: i64,
    pub file_path: String,
    pub suggested_folder: String,
    pub actual_folder: String,
    pub corrected_at: String,
}

impl Store {
    pub fn upsert_folder_centroid(
        &self,
        folder: &str,
        centroid: &[f32],
        file_count: usize,
    ) -> Result<()> {
        let blob: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT INTO folder_centroids (folder, centroid, file_count, updated_at)
             VALUES (?1, ?2, ?3, datetime('now'))
             ON CONFLICT(folder) DO UPDATE SET centroid = ?2, file_count = ?3, updated_at = datetime('now')",
            params![folder, blob, file_count as i64],
        )?;
        Ok(())
    }

    pub fn get_folder_centroids(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT folder, centroid FROM folder_centroids")?;
        let rows = stmt.query_map([], |row| {
            let folder: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let centroid: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((folder, centroid))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get a single folder's centroid and file count.
    pub fn get_folder_centroid(&self, folder: &str) -> Result<Option<(Vec<f32>, usize)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT centroid, file_count FROM folder_centroids WHERE folder = ?1")?;
        let mut rows = stmt.query_map(params![folder], |row| {
            let blob: Vec<u8> = row.get(0)?;
            let count: i64 = row.get(1)?;
            let centroid: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((centroid, count as usize))
        })?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Incrementally adjust a folder centroid using online mean math.
    /// If `increment` is true, adds a file vector; if false, removes one.
    pub fn adjust_folder_centroid(
        &self,
        folder: &str,
        file_vec: &[f32],
        increment: bool,
    ) -> Result<()> {
        let existing = self.get_folder_centroid(folder)?;
        match (existing, increment) {
            (None, true) => {
                // New folder — centroid is just this vector
                self.upsert_folder_centroid(folder, file_vec, 1)?;
            }
            (None, false) => {
                // Nothing to remove from — no-op
            }
            (Some((old, n)), true) => {
                // online mean addition: new = (old * n + vec) / (n + 1)
                let nf = n as f32;
                let new_n = n + 1;
                let updated: Vec<f32> = old
                    .iter()
                    .zip(file_vec.iter())
                    .map(|(o, v)| (o * nf + v) / new_n as f32)
                    .collect();
                self.upsert_folder_centroid(folder, &updated, new_n)?;
            }
            (Some((_old, n)), false) if n <= 1 => {
                // Last file — delete centroid row
                self.conn.execute(
                    "DELETE FROM folder_centroids WHERE folder = ?1",
                    params![folder],
                )?;
            }
            (Some((old, n)), false) => {
                // online mean subtraction: new = (old * n - vec) / (n - 1)
                let nf = n as f32;
                let new_n = n - 1;
                let updated: Vec<f32> = old
                    .iter()
                    .zip(file_vec.iter())
                    .map(|(o, v)| (o * nf - v) / new_n as f32)
                    .collect();
                self.upsert_folder_centroid(folder, &updated, new_n)?;
            }
        }
        Ok(())
    }

    /// Record a placement correction (user moved a note from suggested folder).
    pub fn insert_placement_correction(
        &self,
        file_path: &str,
        suggested_folder: &str,
        actual_folder: &str,
    ) -> Result<()> {
        let dt = time::OffsetDateTime::now_utc();
        let now = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            dt.year(),
            dt.month() as u8,
            dt.day(),
            dt.hour(),
            dt.minute(),
            dt.second(),
        );
        self.conn.execute(
            "INSERT INTO placement_corrections (file_path, suggested_folder, actual_folder, corrected_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![file_path, suggested_folder, actual_folder, now],
        )?;
        Ok(())
    }

    /// Get recent placement corrections, latest first.
    pub fn get_placement_corrections(&self, limit: usize) -> Result<Vec<PlacementCorrection>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, suggested_folder, actual_folder, corrected_at
             FROM placement_corrections ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(PlacementCorrection {
                id: row.get(0)?,
                file_path: row.get(1)?,
                suggested_folder: row.get(2)?,
                actual_folder: row.get(3)?,
                corrected_at: row.get(4)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_folder_centroid_increment() {
        let store = Store::open_memory().unwrap();
        // Seed centroid [1.0, 0.0, 0.0] with n=2
        store
            .upsert_folder_centroid("01-Projects", &[1.0, 0.0, 0.0], 2)
            .unwrap();
        // Add [0.0, 1.0, 0.0] → new = (old*2 + new) / 3 = [2/3, 1/3, 0]
        store
            .adjust_folder_centroid("01-Projects", &[0.0, 1.0, 0.0], true)
            .unwrap();
        let (centroid, count) = store
            .get_folder_centroid("01-Projects")
            .unwrap()
            .expect("centroid should exist");
        assert_eq!(count, 3);
        assert!((centroid[0] - 0.6667).abs() < 0.01);
        assert!((centroid[1] - 0.3333).abs() < 0.01);
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_folder_centroid_decrement() {
        let store = Store::open_memory().unwrap();
        // Seed centroid [0.667, 0.333, 0.0] with n=3
        store
            .upsert_folder_centroid("01-Projects", &[0.667, 0.333, 0.0], 3)
            .unwrap();
        // Remove [0.0, 1.0, 0.0] → new = (old*3 - vec) / 2 = [1.0005, ~0.0, 0.0]
        store
            .adjust_folder_centroid("01-Projects", &[0.0, 1.0, 0.0], false)
            .unwrap();
        let (centroid, count) = store
            .get_folder_centroid("01-Projects")
            .unwrap()
            .expect("centroid should exist");
        assert_eq!(count, 2);
        assert!((centroid[0] - 1.0).abs() < 0.01);
        assert!((centroid[1] - 0.0).abs() < 0.02); // (0.333*3 - 1.0)/2 = ~0.0
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_folder_centroid_decrement_last_file() {
        let store = Store::open_memory().unwrap();
        // Seed with n=1
        store
            .upsert_folder_centroid("01-Projects", &[1.0, 0.0, 0.0], 1)
            .unwrap();
        // Remove last file → centroid deleted
        store
            .adjust_folder_centroid("01-Projects", &[1.0, 0.0, 0.0], false)
            .unwrap();
        let result = store.get_folder_centroid("01-Projects").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_adjust_folder_centroid_new_folder() {
        let store = Store::open_memory().unwrap();
        // No existing centroid, increment → creates centroid
        store
            .adjust_folder_centroid("02-Areas", &[0.5, 0.5, 0.0], true)
            .unwrap();
        let (centroid, count) = store
            .get_folder_centroid("02-Areas")
            .unwrap()
            .expect("centroid should exist");
        assert_eq!(count, 1);
        assert!((centroid[0] - 0.5).abs() < 0.01);
        assert!((centroid[1] - 0.5).abs() < 0.01);
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_insert_placement_correction() {
        let store = Store::open_memory().unwrap();
        store
            .insert_placement_correction("notes/test.md", "00-Inbox", "01-Projects/Work")
            .unwrap();

        let corrections = store.get_placement_corrections(10).unwrap();
        assert_eq!(corrections.len(), 1);
        assert_eq!(corrections[0].file_path, "notes/test.md");
        assert_eq!(corrections[0].suggested_folder, "00-Inbox");
        assert_eq!(corrections[0].actual_folder, "01-Projects/Work");
        assert!(!corrections[0].corrected_at.is_empty());
    }

    #[test]
    fn test_get_placement_corrections_ordering() {
        let store = Store::open_memory().unwrap();
        store
            .insert_placement_correction("notes/first.md", "00-Inbox", "01-Projects")
            .unwrap();
        store
            .insert_placement_correction("notes/second.md", "02-Areas", "03-Resources")
            .unwrap();

        let corrections = store.get_placement_corrections(10).unwrap();
        assert_eq!(corrections.len(), 2);
        // Latest first (ORDER BY id DESC)
        assert_eq!(corrections[0].file_path, "notes/second.md");
        assert_eq!(corrections[1].file_path, "notes/first.md");
    }
}
