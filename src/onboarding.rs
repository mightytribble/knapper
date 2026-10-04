//! The two `init` flows: `detect`, which reads the vault and writes nothing,
//! and `apply`, which writes `vault.toml` and indexes.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::config::{Config, db_path};
use crate::indexer::{IndexSettings, run_index};
use crate::profile::{
    self, FolderMap, StructureDetection, StructureMethod, VaultProfile, VaultStats, VaultType,
};
use crate::store::Store;

/// Build a VaultProfile from detected components.
fn build_profile(
    vault_path: &Path,
    vault_type: VaultType,
    structure: StructureDetection,
    stats: VaultStats,
) -> VaultProfile {
    VaultProfile {
        vault_path: vault_path.to_path_buf(),
        vault_type,
        structure,
        stats,
    }
}

/// Non-destructive vault inspection returning JSON. Writes nothing.
pub fn run_detect_json(vault_path: &Path) -> Result<serde_json::Value> {
    let vault_path = vault_path
        .canonicalize()
        .unwrap_or_else(|_| vault_path.to_path_buf());

    let vault_type = profile::detect_vault_type(&vault_path);
    let structure = profile::detect_structure(&vault_path)?;
    let stats = profile::scan_vault_stats(&vault_path)?;

    let vault_type_str = match vault_type {
        VaultType::Obsidian => "obsidian",
        VaultType::Logseq => "logseq",
        VaultType::Plain => "plain",
        VaultType::Custom => "custom",
    };

    let structure_str = match structure.method {
        StructureMethod::Para => "para",
        StructureMethod::Folders => "folders",
        StructureMethod::Flat => "flat",
        StructureMethod::Custom => "custom",
    };

    // Build folders object
    let folders = json!({
        "inbox": structure.folders.inbox,
        "projects": structure.folders.projects,
        "areas": structure.folders.areas,
        "resources": structure.folders.resources,
        "archive": structure.folders.archive,
        "templates": structure.folders.templates,
        "daily": structure.folders.daily,
        "people": structure.folders.people,
    });

    // Check for existing index
    let data_dir = Config::data_dir()?;
    let db_path = db_path(&data_dir);

    let existing_index = if db_path.exists() {
        let store = Store::open(&db_path)?;
        let all_files = store.get_all_files()?;
        let last_indexed = store.get_meta("last_indexed_at")?;

        Some(json!({
            "files": all_files.len(),
            "last_indexed": last_indexed,
        }))
    } else {
        None
    };

    // Warnings
    let mut warnings: Vec<String> = Vec::new();
    if stats.total_files == 0 {
        warnings.push("Vault contains no markdown files".into());
    }
    if stats.files_with_frontmatter == 0 && stats.total_files > 0 {
        warnings
            .push("No files have YAML frontmatter — tags and metadata won't be extracted".into());
    }
    if stats.wikilink_count == 0 && stats.total_files > 5 {
        warnings.push("No wikilinks found — graph features will be limited".into());
    }

    let ready = stats.total_files > 0 && warnings.is_empty();

    // Count daily notes (approximate: files in the daily folder)
    let daily_count = count_daily_notes(&vault_path, &structure.folders);
    let people_count = count_people_notes(&vault_path, &structure.folders);

    Ok(json!({
        "vault_path": vault_path.to_string_lossy(),
        "vault_type": vault_type_str,
        "structure": structure_str,
        "files": stats.total_files,
        "folders": folders,
        "stats": {
            "daily_notes": daily_count,
            "people_notes": people_count,
            "unique_tags": stats.unique_tags,
            "wikilinks": stats.wikilink_count,
        },
        "existing_index": existing_index,
        "ready": ready,
        "warnings": warnings,
    }))
}

/// `init --mode apply`: write the vault profile, run the index, and answer
/// what was done. The config is the caller's; nothing here writes it.
pub fn run_apply_json(
    vault_path: &Path,
    config: &Config,
    settings: IndexSettings,
    data_dir: &Path,
) -> Result<serde_json::Value> {
    let vault_path = vault_path
        .canonicalize()
        .unwrap_or_else(|_| vault_path.to_path_buf());

    let mut steps_completed: Vec<String> = Vec::new();

    // ── Vault Profile ──
    let vault_type = profile::detect_vault_type(&vault_path);
    let structure = profile::detect_structure(&vault_path)?;
    let stats = profile::scan_vault_stats(&vault_path)?;

    let vault_profile = build_profile(&vault_path, vault_type, structure, stats);
    profile::write_vault_toml(&vault_profile, data_dir).context("writing vault profile")?;
    steps_completed.push("vault_profile_written".into());

    // ── Indexing ──
    let index_result = run_index(&vault_path, config, settings, false)?;
    steps_completed.push("index_built".into());

    // ── Build response ──
    let index_stats = json!({
        "new_files": index_result.new_files,
        "updated_files": index_result.updated_files,
        "deleted_files": index_result.deleted_files,
        "total_chunks": index_result.total_chunks,
        "duration_secs": index_result.duration.as_secs_f64(),
    });

    let vault_profile_info = json!({
        "vault_type": format!("{:?}", vault_profile.vault_type),
        "structure": format!("{:?}", vault_profile.structure.method),
        "total_files": vault_profile.stats.total_files,
    });

    Ok(json!({
        "status": "ok",
        "vault_profile": vault_profile_info,
        "index": index_stats,
        "steps_completed": steps_completed,
        "next_steps": [
            "knapper search \"...\"",
            "knapper serve",
        ],
    }))
}

// ── Private helpers for detect ────────────────────────────────────

/// Count markdown files in the daily folder (if detected).
fn count_daily_notes(vault_path: &Path, folders: &FolderMap) -> usize {
    let Some(ref daily) = folders.daily else {
        return 0;
    };
    let daily_dir = vault_path.join(daily);
    if !daily_dir.is_dir() {
        return 0;
    }
    count_md_files_in_dir(&daily_dir)
}

/// Count markdown files in the people folder (if detected).
/// Falls back to scanning common nested paths (e.g. `*/People/`) when the
/// profile doesn't report a top-level people folder.
fn count_people_notes(vault_path: &Path, folders: &FolderMap) -> usize {
    // 1. Use profile-detected folder if available.
    if let Some(ref people) = folders.people {
        let people_dir = vault_path.join(people);
        if people_dir.is_dir() {
            return count_md_files_in_dir(&people_dir);
        }
    }

    // 2. Fallback: walk one level of subdirectories looking for a "People" subfolder.
    let Ok(entries) = std::fs::read_dir(vault_path) else {
        return 0;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let Ok(ft) = entry.file_type() else { continue };
        if !ft.is_dir() {
            continue;
        }
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let subdir = entry.path();
        let Ok(inner) = std::fs::read_dir(&subdir) else {
            continue;
        };
        for inner_entry in inner.filter_map(|e| e.ok()) {
            let Ok(ift) = inner_entry.file_type() else {
                continue;
            };
            if !ift.is_dir() {
                continue;
            }
            let name = inner_entry.file_name();
            let name_lower = name.to_string_lossy().to_ascii_lowercase();
            if name_lower == "people" {
                let count = count_md_files_in_dir(&inner_entry.path());
                if count > 0 {
                    return count;
                }
            }
        }
    }

    0
}

/// Count `.md` files directly in a directory (non-recursive).
fn count_md_files_in_dir(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                        && e.path().extension().map(|ext| ext == "md").unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}
