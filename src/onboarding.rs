use std::path::Path;

use anyhow::{Context, Result};
use console::style;
use serde_json::json;

use crate::config::{Config, db_path};
use crate::identity::{L1Summary, extract_l1_facts};
use crate::indexer::{IndexResult, IndexSettings, run_index};
use crate::profile::{
    self, FolderMap, StructureDetection, StructureMethod, VaultProfile, VaultStats, VaultType,
};
use crate::store::Store;

// ── Public types ──────────────────────────────────────────────────

/// Flags for the interactive CLI onboarding flow.
pub struct InteractiveFlags {
    pub name: Option<String>,
    pub role: Option<String>,
    pub purpose: Option<String>,
    pub identity_only: bool,
    pub reindex_only: bool,
    pub quiet: bool,
}

/// Flags for the non-interactive (JSON) apply flow.
pub struct ApplyFlags {
    pub name: Option<String>,
    pub role: Option<String>,
    pub purpose: Option<String>,
    pub identity_only: bool,
    pub reindex_only: bool,
}

// ── Constants ─────────────────────────────────────────────────────

const VERSION: &str = env!("CARGO_PKG_VERSION");

const PURPOSE_OPTIONS: &[&str] = &[
    "Personal knowledge base",
    "Work tracking",
    "Research & learning",
    "Team wiki",
    "Other",
];

// ── Helpers ───────────────────────────────────────────────────────

/// Print a section divider: `── Title ──` padded to terminal width.
fn print_divider(title: &str) {
    let term = console::Term::stdout();
    let width = term.size().1 as usize;
    let prefix = format!("── {} ", title);
    let remaining = width.saturating_sub(prefix.len() + 2);
    let suffix = "─".repeat(remaining);
    println!();
    println!("  {}{}", style(&prefix).bold(), suffix);
    println!();
}

/// Print the engraph banner box.
fn print_banner() {
    let tag = format!("engraph v{}", VERSION);
    let sub = "vault intelligence for AI agents";
    let inner_width = tag.len().max(sub.len()) + 4;

    let top = format!("  {}{}{}", "╭", "─".repeat(inner_width + 2), "╮");
    let bot = format!("  {}{}{}", "╰", "─".repeat(inner_width + 2), "╯");
    let empty_line = format!("  │{}│", " ".repeat(inner_width + 2));
    let tag_line = format!(
        "  │  {:<width$}  │",
        style(&tag).bold(),
        width = inner_width
    );
    let sub_line = format!("  │  {:<width$}  │", style(sub).dim(), width = inner_width);

    println!();
    println!("{}", top);
    println!("{}", empty_line);
    println!("{}", tag_line);
    println!("{}", sub_line);
    println!("{}", empty_line);
    println!("{}", bot);
    println!();
}

/// Try to get the user's name from `git config user.name`.
fn git_user_name() -> Option<String> {
    std::process::Command::new("git")
        .args(["config", "user.name"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if name.is_empty() { None } else { Some(name) }
            } else {
                None
            }
        })
}

/// Print a green checkmark line.
fn check(msg: &str) {
    println!("  {} {}", style("✓").green(), msg);
}

/// Print a red cross line.
fn cross(msg: &str) {
    println!("  {} {}", style("✗").red(), msg);
}

/// Detect vault profile (type, structure, stats) without writing anything.
fn detect_profile(vault_path: &Path) -> Result<(VaultType, StructureDetection, VaultStats)> {
    let vault_type = profile::detect_vault_type(vault_path);
    let structure = profile::detect_structure(vault_path)?;
    let stats = profile::scan_vault_stats(vault_path)?;
    Ok((vault_type, structure, stats))
}

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

/// Print vault scan results.
fn print_scan_results(vault_type: &VaultType, structure: &StructureDetection, stats: &VaultStats) {
    let type_label = match vault_type {
        VaultType::Obsidian => "Obsidian vault detected",
        VaultType::Logseq => "Logseq vault detected",
        VaultType::Plain => "Plain markdown folder detected",
        VaultType::Custom => "Custom vault detected",
    };
    check(type_label);

    check(&format!("{} markdown files", stats.total_files));

    let structure_label = match structure.method {
        StructureMethod::Para => "PARA structure",
        StructureMethod::Folders => "Folder-based structure",
        StructureMethod::Flat => "Flat structure",
        StructureMethod::Custom => "Custom structure",
    };
    check(structure_label);

    // Show detected folder roles
    if let Some(ref daily) = structure.folders.daily {
        check(&format!(
            "{} daily notes in {}/",
            count_files_in_folder_approx(stats, daily),
            daily
        ));
    }

    if structure.folders.templates.is_some() {
        check("Templates folder detected");
    } else {
        cross("No templates folder detected");
    }

    if let Some(ref people) = structure.folders.people {
        check(&format!("People folder: {}/", people));
    }
}

/// Rough count for daily notes — we don't have per-folder counts, so report the folder name.
fn count_files_in_folder_approx(_stats: &VaultStats, _folder: &str) -> String {
    // We don't track per-folder file counts in VaultStats.
    // The total_files stat is the best we have. Return "some" as placeholder.
    // A more accurate count would require walking the folder again.
    String::new()
}

/// Print L1 summary as a compact table.
fn print_l1_summary(summary: &L1Summary) {
    if summary.active_projects > 0 {
        println!(
            "  {} active projects",
            style(summary.active_projects).cyan()
        );
    }
    if summary.key_people > 0 {
        println!("  {} key people", style(summary.key_people).cyan());
    }
    if summary.current_focus > 0 {
        println!(
            "  {} current focus items",
            style(summary.current_focus).cyan()
        );
    }
    if summary.blocking > 0 {
        println!("  {} blocking items", style(summary.blocking).yellow());
    }
    if summary.ooo > 0 {
        println!("  {} people OOO", style(summary.ooo).yellow());
    }
    if summary.active_projects == 0
        && summary.key_people == 0
        && summary.current_focus == 0
        && summary.blocking == 0
        && summary.ooo == 0
    {
        println!(
            "  {}",
            style("No structured facts extracted yet. Add tags and daily notes to enrich.").dim()
        );
    }
}

/// Print index results.
fn print_index_result(result: &IndexResult) {
    check(&format!(
        "Index built ({} files, {} chunks, {:.1}s)",
        result.new_files + result.updated_files,
        result.total_chunks,
        result.duration.as_secs_f64()
    ));
}

/// Print the "What's Next" section.
fn print_next_steps(config_path: &Path) {
    print_divider("What's Next");

    check(&format!("Identity saved to {}", config_path.display()));
    println!();
    println!("  Try these:");
    println!("    {}", style("engraph search \"...\"").cyan());
    println!("    {}", style("engraph identity").cyan());
    println!("    {}", style("engraph serve").cyan());
    println!();
}

// ── Public functions ──────────────────────────────────────────────

/// Full interactive onboarding flow with banner, prompts, and progress.
pub fn run_interactive(
    vault_path: &Path,
    config: &mut Config,
    data_dir: &Path,
    flags: InteractiveFlags,
) -> Result<()> {
    let quiet = flags.quiet;

    // ── Banner ──
    if !quiet {
        print_banner();
    }

    // ── Vault Scan ──
    let (vault_type, structure, stats) = if !flags.identity_only {
        if !quiet {
            println!("  {}", style("Scanning vault...").dim());
            println!();
        }

        let (vt, st, vs) = detect_profile(vault_path)?;

        if !quiet {
            print_scan_results(&vt, &st, &vs);
        }

        (vt, st, vs)
    } else {
        // identity_only: skip vault scan, use minimal defaults
        (
            VaultType::Plain,
            StructureDetection {
                method: StructureMethod::Flat,
                folders: FolderMap::default(),
            },
            VaultStats::default(),
        )
    };

    // ── Identity Setup ──
    if !flags.reindex_only {
        if !quiet {
            print_divider("Identity Setup");
        }

        // Name
        let name = if let Some(ref n) = flags.name {
            n.clone()
        } else {
            let default_name = git_user_name().unwrap_or_default();
            let mut input = dialoguer::Input::<String>::new().with_prompt("  ? What's your name?");
            if !default_name.is_empty() {
                input = input.default(default_name);
            }
            input.interact_text()?
        };

        // Role
        let role = if let Some(ref r) = flags.role {
            r.clone()
        } else {
            dialoguer::Input::<String>::new()
                .with_prompt("  ? What do you do?")
                .interact_text()?
        };

        // Vault purpose
        let purpose = if let Some(ref p) = flags.purpose {
            p.clone()
        } else {
            let selection = dialoguer::Select::new()
                .with_prompt("  ? What's this vault for?")
                .items(PURPOSE_OPTIONS)
                .default(0)
                .interact()?;

            if selection == PURPOSE_OPTIONS.len() - 1 {
                // "Other" selected — ask for freeform input
                dialoguer::Input::<String>::new()
                    .with_prompt("  ? Describe your vault's purpose")
                    .interact_text()?
            } else {
                PURPOSE_OPTIONS[selection].to_string()
            }
        };

        // Save identity to config
        config.identity.name = Some(name);
        config.identity.role = Some(role);
        config.identity.vault_purpose = Some(purpose);
        config.save().context("saving identity to config")?;
    }

    // ── Vault Profile ──
    if !flags.identity_only {
        let vault_profile = build_profile(vault_path, vault_type, structure, stats);
        profile::write_vault_toml(&vault_profile, data_dir).context("writing vault profile")?;
    }

    // ── Indexing ──
    if !flags.identity_only {
        if !quiet {
            print_divider("Indexing");
        }

        // Confirm if vault is large
        if !quiet && !flags.reindex_only {
            let total = profile::scan_vault_stats(vault_path)
                .map(|s| s.total_files)
                .unwrap_or(0);
            if total > 500 {
                let confirm = dialoguer::Confirm::new()
                    .with_prompt(format!("  {} files found. Ready to index?", total))
                    .default(true)
                    .interact()?;
                if !confirm {
                    println!(
                        "\n  {}",
                        style("Skipped indexing. Run `engraph index` when ready.").dim()
                    );
                    let config_path = Config::data_dir()?.join("config.toml");
                    print_next_steps(&config_path);
                    return Ok(());
                }
            }
        }

        let result = run_index(
            vault_path,
            config,
            IndexSettings::from_config(config),
            false,
        )?;

        if !quiet {
            println!();
            print_index_result(&result);
        }

        // ── L1 Extraction ──
        let db_path = db_path(data_dir);
        if db_path.exists() {
            let store = Store::open(&db_path)?;
            if let Ok(Some(vault_profile)) = Config::load_vault_profile() {
                if !quiet {
                    print_divider("Auto-extracted Context");
                }

                match extract_l1_facts(&store, &vault_profile) {
                    Ok(summary) => {
                        if !quiet {
                            print_l1_summary(&summary);
                        }
                    }
                    Err(e) => {
                        if !quiet {
                            println!("  {} L1 extraction: {}", style("!").yellow(), e);
                        }
                    }
                }
            }
        }
    }

    // ── What's Next ──
    if !quiet {
        let config_path = Config::data_dir()?.join("config.toml");
        print_next_steps(&config_path);
    }

    Ok(())
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

    // Suggested identity
    let git_name = git_user_name();
    let name_source = if git_name.is_some() {
        "git_config"
    } else {
        "none"
    };

    // Check for existing index
    let data_dir = Config::data_dir()?;
    let db_path = db_path(&data_dir);

    let (existing_index, active_projects, key_people) = if db_path.exists() {
        let store = Store::open(&db_path)?;
        let all_files = store.get_all_files()?;
        let last_indexed = store.get_meta("last_indexed_at")?;

        let index_info = json!({
            "files": all_files.len(),
            "last_indexed": last_indexed,
        });

        // Try to get projects and people from L1 facts
        let projects: Vec<String> = store
            .get_identity_facts(1)
            .unwrap_or_default()
            .iter()
            .filter(|f| f.key == "active_project")
            .map(|f| f.value.clone())
            .collect();

        let people: Vec<String> = store
            .get_identity_facts(1)
            .unwrap_or_default()
            .iter()
            .filter(|f| f.key == "key_person")
            .map(|f| f.value.clone())
            .collect();

        (Some(index_info), projects, people)
    } else {
        (None, vec![], vec![])
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
        "suggested_identity": {
            "name": git_name,
            "name_source": name_source,
            "active_projects": active_projects,
            "key_people": key_people,
        },
        "existing_index": existing_index,
        "ready": ready,
        "warnings": warnings,
    }))
}

/// Non-interactive setup with JSON result. Sets identity, detects profile,
/// runs index, extracts L1 facts, and returns a JSON summary.
pub fn run_apply_json(
    vault_path: &Path,
    config: &mut Config,
    settings: IndexSettings,
    data_dir: &Path,
    flags: ApplyFlags,
) -> Result<serde_json::Value> {
    let vault_path = vault_path
        .canonicalize()
        .unwrap_or_else(|_| vault_path.to_path_buf());

    let mut steps_completed: Vec<String> = Vec::new();

    // ── Identity ──
    if !flags.reindex_only {
        if let Some(ref name) = flags.name {
            config.identity.name = Some(name.clone());
        }
        if let Some(ref role) = flags.role {
            config.identity.role = Some(role.clone());
        }
        if let Some(ref purpose) = flags.purpose {
            config.identity.vault_purpose = Some(purpose.clone());
        }
        config.save().context("saving identity to config")?;
        steps_completed.push("identity_saved".into());
    }

    // ── Vault Profile ──
    let vault_profile = if !flags.identity_only {
        let vault_type = profile::detect_vault_type(&vault_path);
        let structure = profile::detect_structure(&vault_path)?;
        let stats = profile::scan_vault_stats(&vault_path)?;

        let vp = build_profile(&vault_path, vault_type, structure, stats);
        profile::write_vault_toml(&vp, data_dir).context("writing vault profile")?;
        steps_completed.push("vault_profile_written".into());
        Some(vp)
    } else {
        None
    };

    // ── Indexing ──
    let index_result = if !flags.identity_only {
        let result = run_index(&vault_path, config, settings, false)?;
        steps_completed.push("index_built".into());
        Some(result)
    } else {
        None
    };

    // ── L1 Extraction ──
    let l1_summary = if !flags.identity_only {
        let db_path = db_path(data_dir);
        if db_path.exists() {
            let store = Store::open(&db_path)?;
            if let Some(ref vp) = vault_profile {
                match extract_l1_facts(&store, vp) {
                    Ok(summary) => {
                        steps_completed.push("l1_extracted".into());
                        Some(summary)
                    }
                    Err(_) => None,
                }
            } else if let Ok(Some(loaded_profile)) = Config::load_vault_profile() {
                match extract_l1_facts(&store, &loaded_profile) {
                    Ok(summary) => {
                        steps_completed.push("l1_extracted".into());
                        Some(summary)
                    }
                    Err(_) => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // ── Build response ──
    let config_path = Config::data_dir()?.join("config.toml");

    let index_stats = index_result.as_ref().map(|r| {
        json!({
            "new_files": r.new_files,
            "updated_files": r.updated_files,
            "deleted_files": r.deleted_files,
            "total_chunks": r.total_chunks,
            "duration_secs": r.duration.as_secs_f64(),
        })
    });

    let identity_summary = json!({
        "name": config.identity.name,
        "role": config.identity.role,
        "vault_purpose": config.identity.vault_purpose,
    });

    let l1_info = l1_summary.as_ref().map(|s| {
        json!({
            "active_projects": s.active_projects,
            "key_people": s.key_people,
            "current_focus": s.current_focus,
            "blocking": s.blocking,
            "ooo": s.ooo,
        })
    });

    let vault_profile_info = vault_profile.as_ref().map(|vp| {
        json!({
            "vault_type": format!("{:?}", vp.vault_type),
            "structure": format!("{:?}", vp.structure.method),
            "total_files": vp.stats.total_files,
        })
    });

    Ok(json!({
        "status": "ok",
        "config_path": config_path.to_string_lossy(),
        "vault_profile": vault_profile_info,
        "index": index_stats,
        "identity": identity_summary,
        "l1": l1_info,
        "steps_completed": steps_completed,
        "next_steps": [
            "engraph search \"...\"",
            "engraph identity",
            "engraph serve",
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
