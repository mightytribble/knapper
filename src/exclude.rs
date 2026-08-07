//! Exclude-pattern matching for vault ingestion.
//!
//! `config.exclude` is documented as glob patterns; this module is what makes that
//! true. [`walk_vault`](crate::indexer::walk_vault) and the file watcher both match
//! through [`ExcludeMatcher`], so the indexer and the watcher cannot drift apart.
//!
//! Patterns follow `.gitignore` conventions — a separator decides whether a pattern
//! is anchored to the vault root or matches at any depth:
//!
//! | pattern | matches |
//! |---|---|
//! | `*-index.md` | that basename at any depth: `lore/lore-index.md`, `spell-index.md` |
//! | `templates/` | a directory named `templates` at any depth, and everything under it |
//! | `notes/private/**` | anchored at the vault root — the separator pins it there |
//! | `.obsidian` | a file *or* directory named `.obsidian` at any depth |
//!
//! Paths are matched relative to the vault root.

use std::path::Path;

use anyhow::{Result, anyhow};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};

/// Compiled `exclude` patterns.
#[derive(Debug, Clone)]
pub struct ExcludeMatcher {
    set: GlobSet,
}

impl ExcludeMatcher {
    /// Compile `patterns`.
    ///
    /// Returns an error on an unparseable glob so a typo in `config.toml` surfaces
    /// at load time, rather than as a pattern that silently matches nothing.
    pub fn new(patterns: &[String]) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();

        for pattern in patterns {
            let trimmed = pattern.trim();
            if trimmed.trim_matches('/').is_empty() {
                return Err(anyhow!(
                    "exclude pattern {pattern:?} is empty — it would exclude the entire vault"
                ));
            }
            for expanded in expand(trimmed) {
                // `literal_separator` keeps `*` from crossing directory boundaries,
                // so `notes/*.md` means what it does in `.gitignore`. The `**/`
                // prefixes added by `expand` are what reach across depth.
                let glob = GlobBuilder::new(&expanded)
                    .literal_separator(true)
                    .build()
                    .map_err(|e| anyhow!("invalid exclude pattern {pattern:?}: {e}"))?;
                builder.add(glob);
            }
        }

        let set = builder
            .build()
            .map_err(|e| anyhow!("compiling exclude patterns: {e}"))?;
        Ok(Self { set })
    }

    /// Test a path already relative to the vault root.
    pub fn is_match(&self, rel: impl AsRef<Path>) -> bool {
        self.set.is_match(rel.as_ref())
    }

    /// Test an absolute path by first making it relative to `vault_root`.
    pub fn matches_under(&self, path: &Path, vault_root: &Path) -> bool {
        let rel = path.strip_prefix(vault_root).unwrap_or(path);
        self.is_match(rel)
    }

    /// True when no patterns were supplied.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}

/// Expand one user-facing pattern into the globs that implement it.
fn expand(pattern: &str) -> Vec<String> {
    // Trailing slash: a directory, and everything beneath it. Never a file.
    if let Some(dir) = pattern.strip_suffix('/') {
        return if dir.contains('/') {
            vec![format!("{dir}/**")]
        } else {
            vec![format!("**/{dir}/**")]
        };
    }

    // A separator anchors the pattern to the vault root.
    let anchored = pattern.contains('/');
    let base = if anchored {
        pattern.to_string()
    } else {
        format!("**/{pattern}")
    };

    // `.gitignore` treats a bare name as matching a directory's contents too, so
    // `.obsidian` excludes `.obsidian/workspace.md`. Skip when the pattern already
    // ends in `**`, which covers its own subtree.
    if pattern.ends_with("**") {
        vec![base]
    } else {
        vec![format!("{base}/**"), base]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(patterns: &[&str]) -> ExcludeMatcher {
        let owned: Vec<String> = patterns.iter().map(|p| p.to_string()).collect();
        ExcludeMatcher::new(&owned).unwrap()
    }

    #[test]
    fn basename_glob_matches_at_any_depth() {
        let m = matcher(&["*-index.md"]);
        assert!(m.is_match("lore/lore-index.md"));
        assert!(m.is_match("rules/spell-index.md"));
        assert!(m.is_match("spell-index.md"), "must match at the vault root");
        assert!(!m.is_match("lore/archdragon.md"));
        assert!(!m.is_match("lore/index.md"), "requires the '-index' suffix");
    }

    #[test]
    fn glob_pattern_is_not_substring_matching() {
        // The pre-glob implementation used `rel_str.contains(pattern)`, under which
        // `*.canvas` matched nothing at all and `index.md` matched `reindex.md`.
        let m = matcher(&["*.canvas"]);
        assert!(m.is_match("drawings/map.canvas"));
        assert!(!m.is_match("drawings/canvas-notes.md"));
    }

    #[test]
    fn directory_pattern_matches_contents_at_any_depth() {
        let m = matcher(&[".obsidian/"]);
        assert!(m.is_match(".obsidian/workspace.md"));
        assert!(m.is_match(".obsidian/plugins/plugin.md"));
        assert!(m.is_match("vault/.obsidian/workspace.md"));
        assert!(!m.is_match("note.md"));
        assert!(
            !m.is_match(".obsidian"),
            "a trailing slash means directories only"
        );
    }

    #[test]
    fn bare_name_matches_file_and_directory_contents() {
        // Back-compat: `.obsidian` without the trailing slash used to work by
        // substring accident, and existing configs still carry it.
        let m = matcher(&[".obsidian"]);
        assert!(m.is_match(".obsidian"));
        assert!(m.is_match(".obsidian/workspace.md"));
        assert!(m.is_match("vault/.obsidian/plugins/plugin.md"));
        assert!(!m.is_match("obsidian-notes.md"));
    }

    #[test]
    fn separator_anchors_pattern_to_vault_root() {
        let m = matcher(&["templates/**"]);
        assert!(m.is_match("templates/npc.md"));
        assert!(m.is_match("templates/nested/npc.md"));
        assert!(
            !m.is_match("world/templates/npc.md"),
            "anchored patterns do not match at depth"
        );
    }

    #[test]
    fn anchored_path_without_wildcard_covers_its_subtree() {
        let m = matcher(&["notes/private"]);
        assert!(m.is_match("notes/private"));
        assert!(m.is_match("notes/private/secret.md"));
        assert!(!m.is_match("notes/public/secret.md"));
    }

    #[test]
    fn nested_directory_pattern_is_anchored() {
        let m = matcher(&["notes/archive/"]);
        assert!(m.is_match("notes/archive/old.md"));
        assert!(!m.is_match("archive/old.md"));
    }

    #[test]
    fn multiple_patterns_all_apply() {
        let m = matcher(&["*-index.md", "templates/**", ".obsidian/"]);
        assert!(m.is_match("lore/lore-index.md"));
        assert!(m.is_match("templates/npc.md"));
        assert!(m.is_match(".obsidian/workspace.md"));
        assert!(!m.is_match("npcs/archivist-lenne.md"));
    }

    #[test]
    fn empty_pattern_list_matches_nothing() {
        let m = matcher(&[]);
        assert!(m.is_empty());
        assert!(!m.is_match("note.md"));
    }

    #[test]
    fn star_does_not_cross_directory_boundaries() {
        let m = matcher(&["notes/*.md"]);
        assert!(m.is_match("notes/one.md"));
        assert!(!m.is_match("notes/sub/deep.md"));
    }

    #[test]
    fn invalid_glob_is_an_error_not_a_silent_no_op() {
        let err = ExcludeMatcher::new(&["[unclosed.md".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("invalid exclude pattern"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_pattern_is_rejected() {
        for pattern in ["", "   ", "/"] {
            let err = ExcludeMatcher::new(&[pattern.to_string()]).unwrap_err();
            assert!(
                err.to_string().contains("entire vault"),
                "pattern {pattern:?} gave: {err}"
            );
        }
    }

    #[test]
    fn matches_under_strips_the_vault_root() {
        let m = matcher(&["*-index.md"]);
        let root = Path::new("/vault");
        assert!(m.matches_under(Path::new("/vault/lore/lore-index.md"), root));
        assert!(!m.matches_under(Path::new("/vault/lore/archdragon.md"), root));
        // A path outside the root is matched as-is rather than silently passing.
        assert!(m.matches_under(Path::new("other/lore-index.md"), root));
    }
}
