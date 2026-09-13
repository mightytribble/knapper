//! The alias rule (#142): the other names a note's frontmatter records for it.
//!
//! `aliases` is the key Obsidian reserves for those names, and Obsidian 1.9
//! reads it as a list. `alias`, the singular key, is not read, which is the
//! rule `properties::BUILT_IN` already follows.

use serde_yaml::Value;

/// The aliases a note's YAML block records, in the order it wrote them.
///
/// `frontmatter` is the block with no `---` fences, as `files.frontmatter`
/// holds it. Each list item is one alias. A plain string is also one alias,
/// and a comma inside it does not split it, because a name can hold a comma.
/// A number or a boolean item is read as its text, since YAML gives `1984`
/// a type the note did not mean. A nested item, an empty item and a block
/// that does not parse record nothing.
///
/// Two spellings that fold to one alias are one alias, and the first spelling
/// is kept: the store holds one row per note and folded alias.
pub fn extract(frontmatter: &str) -> Vec<String> {
    let Ok(Value::Mapping(map)) = serde_yaml::from_str::<Value>(frontmatter) else {
        return Vec::new();
    };
    let items = match map.get("aliases") {
        Some(Value::Sequence(items)) => items.iter().collect(),
        Some(scalar) => vec![scalar],
        None => return Vec::new(),
    };
    let mut seen = std::collections::HashSet::new();
    items
        .into_iter()
        .filter_map(scalar_text)
        .filter(|alias| seen.insert(fold(alias)))
        .collect()
}

/// One item's text, or `None` for an item that names nothing.
fn scalar_text(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(s) => s.trim().to_string(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => return None,
    };
    (!text.is_empty()).then_some(text)
}

/// The form two spellings of one alias share: lowercase, as a tag's path is.
///
/// It folds in Rust and not in SQL, because SQLite's `lower()` folds ASCII
/// letters only.
pub fn fold(alias: &str) -> String {
    alias.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_list_is_one_alias_per_item() {
        assert_eq!(
            extract("tags:\n  - person\naliases:\n  - Sam\n  - Dragon\n"),
            vec!["Sam", "Dragon"]
        );
    }

    #[test]
    fn a_flow_list_is_one_alias_per_item() {
        assert_eq!(extract("aliases: [Max, \"M. D.\"]"), vec!["Max", "M. D."]);
    }

    #[test]
    fn a_plain_string_is_one_alias_and_a_comma_does_not_split_it() {
        assert_eq!(extract("aliases: Smith, John"), vec!["Smith, John"]);
    }

    #[test]
    fn a_number_or_a_boolean_item_is_read_as_its_text() {
        assert_eq!(
            extract("aliases:\n  - 1984\n  - true\n"),
            vec!["1984", "true"]
        );
    }

    #[test]
    fn an_empty_or_null_value_records_no_alias() {
        assert!(extract("aliases: []").is_empty());
        assert!(extract("aliases:").is_empty());
        assert!(extract("aliases:\n  - \"\"\n  -   \n").is_empty());
    }

    #[test]
    fn only_the_plural_key_is_read() {
        assert!(extract("alias: Sam").is_empty());
        assert!(extract("Aliases: [Sam]").is_empty());
    }

    #[test]
    fn a_block_with_no_aliases_key_or_no_mapping_records_none() {
        assert!(extract("").is_empty());
        assert!(extract("title: Note").is_empty());
        assert!(extract("- a\n- b").is_empty());
        assert!(extract("aliases: [unclosed").is_empty());
    }

    #[test]
    fn a_nested_item_is_skipped_and_its_siblings_are_kept() {
        assert_eq!(
            extract("aliases:\n  - Sam\n  - {name: Dragon}\n  - [x]\n  - Samantha\n"),
            vec!["Sam", "Samantha"]
        );
    }

    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(extract("aliases:\n  - \"  Sam \"\n"), vec!["Sam"]);
    }

    #[test]
    fn a_second_spelling_of_one_alias_is_dropped_and_the_first_is_kept() {
        assert_eq!(
            extract("aliases: [Sam, SAM, sam, Dragon]"),
            vec!["Sam", "Dragon"]
        );
    }

    #[test]
    fn fold_ignores_case_beyond_ascii() {
        assert_eq!(fold("Élodie"), fold("éLODIE"));
        assert_eq!(fold("Sam"), "sam");
    }
}
