//! What each capability takes, declared once (#62).
//!
//! One struct per capability, read by all three surfaces. `clap::Args` is
//! what the CLI parses, `Deserialize` is what MCP and HTTP read, and
//! `JsonSchema` is what an MCP client is shown. Because the three derive
//! from one declaration, a parameter cannot be named differently on two
//! surfaces or exist on one and not another.
//!
//! Container encoding still differs where a surface forces it: a GET route
//! reads `?all=a,b` as one comma-separated value, because `serde_urlencoded`
//! reads no sequence (#61). Names and meaning do not differ.

use clap::Args;
use rmcp::schemars;
use rmcp::schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Search {
    /// The search query.
    pub query: String,
    /// Number of results to return (default 10).
    #[arg(short = 'n', long)]
    pub top_n: Option<usize>,
    /// Show the per-lane score breakdown for each result.
    #[arg(long)]
    #[serde(default)]
    pub explain: bool,
    /// Return one result per matching section, or one per document.
    #[arg(long, value_enum)]
    pub group_by: Option<crate::config::GroupBy>,
    /// Filter to notes with all listed tags. An alias of `all`.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub tags: Vec<String>,
    /// Filter to notes carrying every term. A term is a tag path; a trailing
    /// `/` or `/*` matches the tag and its descendants. An unknown term is
    /// an error naming the nearest tag the vault holds (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// Filter to notes carrying at least one of these terms. An unknown
    /// term is an error naming the nearest tag the vault holds (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Filter out notes carrying any of these terms. An unknown term here
    /// is ignored (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Read {
    /// File path, basename, or #docid.
    pub file: String,
    /// Read one section by its heading. Omit for the whole note.
    #[arg(long)]
    pub section: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct List {
    /// Filter to a folder path prefix.
    #[arg(long)]
    pub folder: Option<String>,
    /// Filter to notes carrying every term. A term is a tag path; a trailing
    /// `/` or `/*` matches the tag and its descendants. An unknown term is
    /// an error naming the nearest tag the vault holds (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// An alias of `all`.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub tags: Vec<String>,
    /// Filter to notes carrying at least one of these terms. An unknown
    /// term is an error naming the nearest tag the vault holds (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Filter out notes carrying any of these terms. An unknown term here
    /// is ignored (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
    /// Filter to notes created by one agent.
    #[arg(long)]
    pub created_by: Option<String>,
    /// Maximum results (default 20). Raising it adds results below the same
    /// ranking; it does not change what the top of the ranking holds.
    #[arg(long, default_value = "20")]
    #[serde(default = "default_limit", deserialize_with = "deserialize_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    20
}

fn deserialize_limit<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_number_or_default(deserializer, default_limit())
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Tags {
    /// Limit to one tag and its descendants, as `type/` or `type/*`. Omit
    /// for the whole vocabulary.
    #[arg(long)]
    pub under: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct VaultMap {}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Who {
    /// Person name. It matches a filename in the People folder.
    pub name: String,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Project {
    /// Project name. It matches a filename.
    pub name: String,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Topic {
    /// The topic to gather context for.
    pub query: String,
    /// Character budget. 32000 is about 8000 tokens.
    #[arg(long, default_value = "32000")]
    #[serde(default = "default_budget", deserialize_with = "deserialize_budget")]
    pub budget: usize,
}

fn default_budget() -> usize {
    32000
}

fn deserialize_budget<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_number_or_default(deserializer, default_budget())
}

/// A number that falls back to `default` when the field is absent, and also
/// when a caller sends an explicit JSON `null` — `#[serde(default = ...)]`
/// alone only covers the absent case; a present `null` still reaches
/// `usize::deserialize` and fails there (#60, the same lesson as
/// `deserialize_tag_list`).
fn deserialize_number_or_default<'de, D>(deserializer: D, default: usize) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<usize>::deserialize(deserializer)?.unwrap_or(default))
}

/// One field, three shapes: a JSON array of strings, one comma-separated
/// string, or `null`.
///
/// A GET query string reads as the comma-separated shape, because
/// `serde_urlencoded` has no sequence support (#61). A JSON body reads
/// either the array shape or, for a caller that serialises an absent
/// optional as an explicit `null` — routine in JavaScript and Python — the
/// `null` shape, which must not fail deserialization (#60).
fn deserialize_tag_list<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct TagListVisitor;

    impl<'de> serde::de::Visitor<'de> for TagListVisitor {
        type Value = Vec<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON array of strings, one comma-separated string, or null")
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Vec::new())
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect())
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut out = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                out.push(s);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_any(TagListVisitor)
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Create {
    /// Note content. The CLI reads stdin when this is omitted.
    #[arg(long)]
    pub content: Option<String>,
    /// Filename, without `.md`.
    #[arg(long)]
    pub filename: Option<String>,
    /// A hint at the note's kind, used for placement.
    #[arg(long)]
    pub type_hint: Option<String>,
    /// Tags to resolve against the vault's vocabulary.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub tags: Vec<String>,
    /// Folder to place the note in. Placement chooses one when omitted.
    #[arg(long)]
    pub folder: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Delete {
    /// File path, basename, or #docid.
    pub file: String,
    /// `soft` archives the note; `hard` removes it permanently.
    #[arg(long, default_value = "soft")]
    #[serde(
        default = "default_delete_mode",
        deserialize_with = "deserialize_delete_mode"
    )]
    pub mode: String,
}

fn default_delete_mode() -> String {
    "soft".to_string()
}

fn deserialize_delete_mode<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_string_or_default(deserializer, default_delete_mode())
}

/// A string that falls back to `default` when the field is absent, and also
/// when a caller sends an explicit JSON `null` — the same lesson as
/// `deserialize_number_or_default` (#60): `#[serde(default = ...)]` alone
/// only covers the absent case.
fn deserialize_string_or_default<'de, D>(
    deserializer: D,
    default: String,
) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or(default))
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Move {
    /// File path, basename, or #docid.
    pub file: String,
    /// New folder path, relative to the vault root.
    #[arg(long)]
    pub new_folder: String,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Archive {
    /// File path, basename, or #docid.
    pub file: String,
    /// Restore a note the archive holds, instead of archiving one.
    #[arg(long)]
    #[serde(default)]
    pub undo: bool,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Index {
    /// Rebuild the index from scratch.
    #[arg(long)]
    #[serde(default)]
    pub rebuild: bool,
    /// Index files that `.gitignore` or `.ignore` would exclude.
    #[arg(long)]
    #[serde(default)]
    pub no_gitignore: bool,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct ReindexFile {
    /// File path relative to the vault root.
    pub file: String,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Status {}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Health {}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Identity {
    /// Re-extract the L1 facts without a full re-index.
    #[arg(long)]
    #[serde(default)]
    pub refresh: bool,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Init {
    /// `detect` inspects the vault and writes nothing; `apply` configures
    /// identity and indexes. The CLI runs its interactive flow when this is
    /// omitted, which is the one thing the other surfaces cannot do.
    #[arg(long)]
    pub mode: Option<String>,
    /// User name, for `apply`.
    #[arg(long)]
    pub name: Option<String>,
    /// User role, for `apply`.
    #[arg(long)]
    pub role: Option<String>,
    /// Vault purpose, for `apply`.
    #[arg(long)]
    pub purpose: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Migrate {
    /// `preview`, `apply` or `undo`.
    #[arg(long)]
    pub mode: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// One struct, three readers: the names clap parses are the names the
    /// JSON schema publishes.
    #[test]
    fn search_names_the_same_parameters_to_clap_and_to_json_schema() {
        #[derive(Debug, clap::Parser)]
        struct Probe {
            #[command(flatten)]
            inner: Search,
        }
        let cmd = Probe::command();
        let clap_names: std::collections::BTreeSet<String> = cmd
            .get_arguments()
            .map(|a| a.get_id().to_string())
            .filter(|n| n != "help")
            .collect();

        let schema = rmcp::schemars::schema_for!(Search);
        let json = serde_json::to_value(&schema).unwrap();
        let schema_names: std::collections::BTreeSet<String> = json["properties"]
            .as_object()
            .expect("schema has properties")
            .keys()
            .cloned()
            .collect();

        assert_eq!(clap_names, schema_names);
    }

    /// A JSON array of strings is the MCP and HTTP-JSON shape.
    #[test]
    fn tag_list_reads_a_json_array() {
        let list: List = serde_json::from_str(r#"{"all":["a","b"]}"#).unwrap();
        assert_eq!(list.all, vec!["a".to_string(), "b".to_string()]);
    }

    /// One comma-separated string is what a GET query string sends, but a
    /// JSON caller may send it too (#61); the split trims and drops empties.
    #[test]
    fn tag_list_reads_a_comma_separated_string() {
        let list: List = serde_json::from_str(r#"{"all":"a, b"}"#).unwrap();
        assert_eq!(list.all, vec!["a".to_string(), "b".to_string()]);
    }

    /// An explicit JSON `null` — what a JS/Python caller sends for an absent
    /// optional — must not fail deserialization (#60).
    #[test]
    fn tag_list_reads_null_as_empty() {
        let list: List = serde_json::from_str(r#"{"all":null}"#).unwrap();
        assert!(list.all.is_empty());
    }

    /// `serde_urlencoded` has no sequence support, so `GET /api/list` reads
    /// the comma-separated shape through the same helper (#61).
    #[test]
    fn tag_list_reads_from_a_query_string() {
        let list: List = serde_urlencoded::from_str("all=a,b").unwrap();
        assert_eq!(list.all, vec!["a".to_string(), "b".to_string()]);
    }

    /// A number field takes the same null-must-not-fail rule as the tag
    /// lists (#60) — `#[serde(default = ...)]` alone only covers a field
    /// that is missing, not one sent as an explicit `null`.
    #[test]
    fn limit_reads_null_as_the_default() {
        let list: List = serde_json::from_str(r#"{"limit":null}"#).unwrap();
        assert_eq!(list.limit, 20);
    }
}
