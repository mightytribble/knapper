use anyhow::{Context, Result};
use regex::Regex;

use crate::config::Config;
use crate::profile::VaultProfile;
use crate::store::Store;

/// Summary of what L1 extraction found.
#[derive(Debug, Default)]
pub struct L1Summary {
    pub active_projects: usize,
    pub key_people: usize,
    pub current_focus: usize,
    pub ooo: usize,
    pub blocking: usize,
}

/// Extract L1 identity facts from the indexed vault.
///
/// Clears existing tier-1 facts, then populates five categories:
/// active_projects, key_people, current_focus, ooo, blocking.
pub fn extract_l1_facts(store: &Store, profile: &VaultProfile) -> Result<L1Summary> {
    store.clear_identity_facts(1)?;

    let all_files = store.get_all_files()?;
    let mut summary = L1Summary::default();

    // ── Active projects ─────────────────────────────────────────
    for file in &all_files {
        if path_is_in_excluded_folder(&file.path) {
            continue;
        }
        if file.tags.iter().any(|t| t.eq_ignore_ascii_case("project")) {
            let name = file_stem(&file.path);
            store.upsert_identity_fact(1, "active_project", &name, Some(&file.path))?;
            summary.active_projects += 1;
        }
    }

    // ── Key people ──────────────────────────────────────────────
    let people_folder = profile.structure.folders.people.as_deref();
    if let Some(pf) = people_folder {
        let people_files: Vec<_> = all_files
            .iter()
            .filter(|f| path_is_in_folder(&f.path, pf))
            .collect();

        // Sort by incoming edge count (descending), take top 5.
        let mut scored: Vec<(&crate::store::FileRecord, usize)> = people_files
            .iter()
            .filter_map(|f| {
                let incoming = store.get_incoming(f.id, None).ok()?;
                Some((*f, incoming.len()))
            })
            .collect();
        scored.sort_by_key(|b| std::cmp::Reverse(b.1));

        for (file, _count) in scored.into_iter().take(5) {
            let name = file_stem(&file.path);
            store.upsert_identity_fact(1, "key_person", &name, Some(&file.path))?;
            summary.key_people += 1;
        }
    }

    // ── Daily-note based extractions ────────────────────────────
    let daily_folder = profile.structure.folders.daily.as_deref();
    if let Some(df) = daily_folder {
        let mut daily_files: Vec<_> = all_files
            .iter()
            .filter(|f| path_is_in_folder(&f.path, df) && f.note_date.is_some())
            .collect();

        // Sort by note_date descending (most recent first).
        daily_files.sort_by_key(|b| std::cmp::Reverse(b.note_date));

        // ── Current focus (most recent daily note) ──────────────
        if let Some(latest) = daily_files.first()
            && let Ok(chunks) = store.get_chunks_by_file(latest.id)
        {
            let focus_re = Regex::new(r"(?i)morning\s+focus|top\s+priorit|priorities").unwrap();
            for chunk in &chunks {
                if focus_re.is_match(&chunk.heading) {
                    let items = extract_bullet_items(&chunk.snippet, 3);
                    for item in items {
                        store.upsert_identity_fact(1, "current_focus", &item, None)?;
                        summary.current_focus += 1;
                    }
                    break;
                }
            }
        }

        // ── OOO (last 7 daily notes) ───────────────────────────
        if people_folder.is_some() {
            let people_names: Vec<String> = all_files
                .iter()
                .filter(|f| path_is_in_folder(&f.path, people_folder.unwrap()))
                .map(|f| file_stem(&f.path))
                .collect();

            let ooo_re = Regex::new(r"(?i)\b(ooo|out\s+of\s+office|vacation|leave|pto)\b").unwrap();

            for daily in daily_files.iter().take(7) {
                if let Ok(chunks) = store.get_chunks_by_file(daily.id) {
                    for chunk in &chunks {
                        if ooo_re.is_match(&chunk.snippet) {
                            for person in &people_names {
                                if chunk
                                    .snippet
                                    .to_ascii_lowercase()
                                    .contains(&person.to_ascii_lowercase())
                                {
                                    // Extract context around the match.
                                    let detail = extract_ooo_detail(&chunk.snippet, &ooo_re);
                                    let label = format!("{} ({})", person, detail);
                                    store.upsert_identity_fact(1, "ooo", &label, None)?;
                                    summary.ooo += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Blocking (last 3 daily notes) ───────────────────────
        let blocking_re = Regex::new(r"(?i)\b(P0|blocking|blocked)\b").unwrap();

        for daily in daily_files.iter().take(3) {
            if let Ok(chunks) = store.get_chunks_by_file(daily.id) {
                for chunk in &chunks {
                    let items = extract_matching_bullets(&chunk.snippet, &blocking_re);
                    for item in items {
                        store.upsert_identity_fact(1, "blocking", &item, None)?;
                        summary.blocking += 1;
                    }
                }
            }
        }
    }

    Ok(summary)
}

/// Format the identity block combining L0 (config) and L1 (store) facts.
pub fn format_identity_block(config: &Config, store: &Store) -> Result<String> {
    let id = &config.identity;

    let name = id.name.as_deref().unwrap_or("(not set)");
    let role = id.role.as_deref().unwrap_or("(not set)");
    let vault = id.vault_purpose.as_deref().unwrap_or("(not set)");

    let mut out = String::new();
    out.push_str("## Identity (L0)\n");
    out.push_str(&format!("Name: {}\n", name));
    out.push_str(&format!("Role: {}\n", role));
    out.push_str(&format!("Vault: {}\n", vault));

    let facts = store
        .get_identity_facts(1)
        .context("reading L1 identity facts")?;

    if facts.is_empty() {
        out.push_str("\n## Current State (L1)\n");
        out.push_str("[no data — run knapper index]\n");
        return Ok(out);
    }

    // Determine most recent updated_at across all facts.
    let latest_ts = facts
        .iter()
        .map(|f| f.updated_at.as_str())
        .max()
        .unwrap_or("unknown");

    out.push_str(&format!(
        "\n## Current State (L1) [updated {}]\n",
        latest_ts
    ));

    // Group facts by key.
    let project_vals: Vec<&str> = facts
        .iter()
        .filter(|f| f.key == "active_project")
        .map(|f| f.value.as_str())
        .collect();
    let focus_vals: Vec<&str> = facts
        .iter()
        .filter(|f| f.key == "current_focus")
        .map(|f| f.value.as_str())
        .collect();
    let people_vals: Vec<&str> = facts
        .iter()
        .filter(|f| f.key == "key_person")
        .map(|f| f.value.as_str())
        .collect();
    let blocking_vals: Vec<&str> = facts
        .iter()
        .filter(|f| f.key == "blocking")
        .map(|f| f.value.as_str())
        .collect();
    let ooo_vals: Vec<&str> = facts
        .iter()
        .filter(|f| f.key == "ooo")
        .map(|f| f.value.as_str())
        .collect();

    if !project_vals.is_empty() {
        out.push_str(&format!("Active projects: {}\n", project_vals.join(", ")));
    }
    if !focus_vals.is_empty() {
        out.push_str(&format!("Current focus: {}\n", focus_vals.join(", ")));
    }
    if !people_vals.is_empty() {
        out.push_str(&format!("Key people: {}\n", people_vals.join(", ")));
    }
    if !blocking_vals.is_empty() {
        out.push_str(&format!("Blocking: {}\n", blocking_vals.join(", ")));
    }
    if !ooo_vals.is_empty() {
        out.push_str(&format!("OOO: {}\n", ooo_vals.join(", ")));
    }

    Ok(out)
}

// ── Helpers ─────────────────────────────────────────────────────

/// Extract the file stem (name without extension) from a path string.
fn file_stem(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

/// Return true if the path is inside a templates or archive folder and should be
/// excluded from L1 extraction. Matches any path component named "templates",
/// "template", "archive", or "archives" (case-insensitive), as well as PARA-style
/// numbered variants (e.g. "05-Templates", "04-Archive").
fn path_is_in_excluded_folder(path: &str) -> bool {
    for component in path.split('/') {
        let stripped = component
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches(['-', '_', ' ']);
        let lower = stripped.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "templates" | "template" | "archive" | "archives"
        ) {
            return true;
        }
    }
    false
}

/// Check whether a file path belongs to a given folder (case-insensitive prefix match).
fn path_is_in_folder(path: &str, folder: &str) -> bool {
    let normalized = folder.trim_end_matches('/');
    let lower_path = path.to_ascii_lowercase();
    // Match "folder/" prefix or "/folder/" anywhere in the path.
    lower_path.starts_with(&format!("{}/", normalized.to_ascii_lowercase()))
        || lower_path.contains(&format!("/{}/", normalized.to_ascii_lowercase()))
}

/// Extract up to `max` bullet-point items from a snippet.
fn extract_bullet_items(snippet: &str, max: usize) -> Vec<String> {
    let mut items = Vec::new();
    for line in snippet.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            // Strip checkbox markers like [ ] or [x].
            let rest = rest
                .strip_prefix("[ ] ")
                .or_else(|| rest.strip_prefix("[x] "))
                .or_else(|| rest.strip_prefix("[X] "))
                .unwrap_or(rest);
            let clean = rest.trim().to_string();
            if !clean.is_empty() {
                items.push(clean);
                if items.len() >= max {
                    break;
                }
            }
        }
    }
    items
}

/// Extract bullet items that match a regex pattern.
fn extract_matching_bullets(snippet: &str, pattern: &Regex) -> Vec<String> {
    let mut items = Vec::new();
    for line in snippet.lines() {
        let trimmed = line.trim();
        if (trimmed.starts_with("- ") || trimmed.starts_with("* ")) && pattern.is_match(trimmed) {
            let rest = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .unwrap_or(trimmed);
            let rest = rest
                .strip_prefix("[ ] ")
                .or_else(|| rest.strip_prefix("[x] "))
                .or_else(|| rest.strip_prefix("[X] "))
                .unwrap_or(rest);
            let clean = rest.trim().to_string();
            if !clean.is_empty() {
                items.push(clean);
            }
        }
    }
    items
}

/// Extract a short OOO detail string from around the regex match.
fn extract_ooo_detail(snippet: &str, ooo_re: &Regex) -> String {
    for line in snippet.lines() {
        let trimmed = line.trim();
        if ooo_re.is_match(trimmed) {
            // Return the line content (stripped of bullet prefix) as the detail.
            let rest = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .unwrap_or(trimmed);
            let clean = rest.trim();
            if clean.len() > 80 {
                return format!("{}...", &clean[..77]);
            }
            return clean.to_string();
        }
    }
    "OOO".to_string()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::store::Store;

    #[test]
    fn test_format_identity_block_l0_only() {
        let store = Store::open_memory().unwrap();
        let mut config = Config::default();
        config.identity.name = Some("Oleksandr".into());
        config.identity.role = Some("Engineer".into());
        config.identity.vault_purpose = Some("personal knowledge base".into());

        let block = format_identity_block(&config, &store).unwrap();

        assert!(block.contains("Name: Oleksandr"));
        assert!(block.contains("Role: Engineer"));
        assert!(block.contains("Vault: personal knowledge base"));
        assert!(block.contains("no data"));
    }

    #[test]
    fn test_format_identity_block_with_l1() {
        let store = Store::open_memory().unwrap();
        let mut config = Config::default();
        config.identity.name = Some("Test User".into());
        config.identity.role = Some("Developer".into());
        config.identity.vault_purpose = Some("notes".into());

        // Insert L1 facts manually.
        store
            .upsert_identity_fact(
                1,
                "active_project",
                "ProjectA",
                Some("01-Projects/ProjectA.md"),
            )
            .unwrap();
        store
            .upsert_identity_fact(
                1,
                "active_project",
                "ProjectB",
                Some("01-Projects/ProjectB.md"),
            )
            .unwrap();
        store
            .upsert_identity_fact(
                1,
                "key_person",
                "Alice",
                Some("03-Resources/People/Alice.md"),
            )
            .unwrap();
        store
            .upsert_identity_fact(1, "current_focus", "Ship feature X", None)
            .unwrap();
        store
            .upsert_identity_fact(1, "blocking", "CI pipeline broken", None)
            .unwrap();
        store
            .upsert_identity_fact(1, "ooo", "Bob (vacation until Friday)", None)
            .unwrap();

        let block = format_identity_block(&config, &store).unwrap();

        assert!(block.contains("Name: Test User"));
        assert!(block.contains("Role: Developer"));
        assert!(block.contains("Vault: notes"));
        assert!(block.contains("Active projects: ProjectA, ProjectB"));
        assert!(block.contains("Key people: Alice"));
        assert!(block.contains("Current focus: Ship feature X"));
        assert!(block.contains("Blocking: CI pipeline broken"));
        assert!(block.contains("OOO: Bob (vacation until Friday)"));
        assert!(block.contains("## Current State (L1) [updated"));
        assert!(!block.contains("no data"));
    }
}
