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
    /// Number of results to return.
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
    /// Filter to notes carrying every term. A trailing `/` or `/*` matches
    /// the tag and its descendants.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// An alias of `all`.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub tags: Vec<String>,
    /// Filter to notes carrying at least one term.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Filter out notes carrying any of these terms.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
    /// Filter to notes created by one agent.
    #[arg(long)]
    pub created_by: Option<String>,
    /// Maximum results.
    #[arg(long, default_value = "20")]
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    20
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Tags {
    /// Limit to one tag and its descendants, as `type/` or `type/*`.
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
    #[serde(default = "default_budget")]
    pub budget: usize,
}

fn default_budget() -> usize {
    32000
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
}
