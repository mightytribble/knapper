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
    /// Number of results to return. Omit it for the configured default,
    /// which is `top_n` in `config.toml` and is the same number on every
    /// surface (#62).
    #[arg(short = 'n', long)]
    pub top_n: Option<usize>,
    /// Show the per-lane score breakdown for each result.
    ///
    /// The breakdown is text, and `run_search` prints it only when the output
    /// is not JSON. Asking for both is a usage error rather than a flag that
    /// is silently dropped, which is what the CLI answered before the command
    /// took this struct (#62).
    #[arg(long, conflicts_with = "json")]
    #[serde(default)]
    pub explain: bool,
    /// Return one result per matching section, or one per document.
    #[arg(long, value_enum)]
    pub group_by: Option<crate::config::GroupBy>,
    /// An alias of `all`. A term starting with `/` is a directory path from
    /// the vault root instead of a tag, case-sensitive; a trailing `/`
    /// scopes to its subtree.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub scope: Vec<String>,
    /// Filter to notes carrying every term. A term is a tag path; a trailing
    /// `/` or `/*` matches the tag and its descendants. A term starting with
    /// `/` is a directory path from the vault root instead, case-sensitive,
    /// with a trailing `/` scoping to its subtree. An unknown term is an
    /// error naming the nearest tag or folder the vault holds (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// Filter to notes carrying at least one of these terms. A term starting
    /// with `/` is a directory path from the vault root instead of a tag,
    /// case-sensitive; a trailing `/` scopes to its subtree. An unknown
    /// term is an error naming the nearest tag or folder the vault holds
    /// (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Filter out notes carrying any of these terms. A term starting with
    /// `/` is a directory path from the vault root instead of a tag,
    /// case-sensitive; a trailing `/` scopes to its subtree. An unknown
    /// term here is ignored (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Read {
    /// File path, basename, or #docid.
    pub file: String,
    /// Read one section by its heading. Omit for the whole note.
    ///
    /// The heading is an ATX `#` heading and the match folds case, so
    /// `spells` finds `## Spells`. A section read narrows `content` and
    /// `byte_count` to that section; the note's tags and links are reported
    /// either way, because a section's are its file's (#62).
    #[arg(long)]
    pub section: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct List {
    /// Filter to a folder path prefix.
    #[arg(long)]
    pub folder: Option<String>,
    /// Filter to notes carrying every term. A term is a tag path; a trailing
    /// `/` or `/*` matches the tag and its descendants. A term starting with
    /// `/` is a directory path from the vault root instead, case-sensitive,
    /// with a trailing `/` scoping to its subtree. An unknown term is an
    /// error naming the nearest tag or folder the vault holds (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// An alias of `all`. A term starting with `/` is a directory path from
    /// the vault root instead of a tag, case-sensitive; a trailing `/`
    /// scopes to its subtree.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub scope: Vec<String>,
    /// Filter to notes carrying at least one of these terms. A term starting
    /// with `/` is a directory path from the vault root instead of a tag,
    /// case-sensitive; a trailing `/` scopes to its subtree. An unknown
    /// term is an error naming the nearest tag or folder the vault holds
    /// (#60).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Filter out notes carrying any of these terms. A term starting with
    /// `/` is a directory path from the vault root instead of a tag,
    /// case-sensitive; a trailing `/` scopes to its subtree. An unknown
    /// term here is ignored (#60).
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
    /// Person name. It resolves as a `#docid`, a path or a basename first, and
    /// then as a keyword search answered by the first hit that sits under the
    /// profile's People folder, carries a `person` or `people` tag, or has the
    /// name for its filename.
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
    /// An alias of `all`. A term starting with `/` is a directory path from
    /// the vault root instead of a tag, case-sensitive; a trailing `/`
    /// scopes to its subtree.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub scope: Vec<String>,
    /// Gather context from notes carrying every term. A term is a tag path; a
    /// trailing `/` or `/*` matches the tag and its descendants. A term
    /// starting with `/` is a directory path from the vault root instead,
    /// case-sensitive, with a trailing `/` scoping to its subtree. An
    /// unknown term is an error naming the nearest tag or folder the vault
    /// holds (#64).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// Gather context from notes carrying at least one of these terms. A
    /// term starting with `/` is a directory path from the vault root
    /// instead of a tag, case-sensitive; a trailing `/` scopes to its
    /// subtree. An unknown term is an error naming the nearest tag or
    /// folder the vault holds (#64).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Leave out notes carrying any of these terms. A term starting with
    /// `/` is a directory path from the vault root instead of a tag,
    /// case-sensitive; a trailing `/` scopes to its subtree. An unknown
    /// term here is ignored (#64).
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
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
    /// Set to false to skip automatic wikilink resolution. Defaults to true.
    #[arg(long)]
    pub auto_link: Option<bool>,
}

/// What one edit does to what it names. An enum and not a string, because
/// this is what publishes the four legal values to an MCP client and to the
/// OpenAPI spec: a wrong spelling is refused at the boundary (#62).
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum EditMode {
    Replace,
    Prepend,
    Append,
    Remove,
}

impl From<EditMode> for crate::writer::EditMode {
    fn from(mode: EditMode) -> Self {
        match mode {
            EditMode::Replace => crate::writer::EditMode::Replace,
            EditMode::Prepend => crate::writer::EditMode::Prepend,
            EditMode::Append => crate::writer::EditMode::Append,
            EditMode::Remove => crate::writer::EditMode::Remove,
        }
    }
}

/// One edit, as a caller writes it. `section` and `property` are the two
/// ways to name a target and an edit names at most one; naming neither is
/// the note's body (#62).
///
/// The struct is closed: an unknown key is an error and not a key serde
/// drops. `rewrite` took a `preserve_frontmatter` flag and a body edit here
/// always keeps the frontmatter, so a caller that still sends the flag has
/// to read the error rather than get a write it did not ask for.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    /// The heading of the section to edit.
    pub section: Option<String>,
    /// The frontmatter property to edit.
    pub property: Option<String>,
    /// `replace`, `append`, `prepend` or `remove`.
    pub mode: EditMode,
    /// A string, or a list of strings for a list-valued property.
    ///
    /// The field is a `Value` because either shape is legal and which one a
    /// caller sends is what it means. `EditContent` is the schema that says
    /// so, so a client reading the schema is told the two shapes rather
    /// than being told nothing (#62).
    #[schemars(with = "Option<EditContent>")]
    pub content: Option<serde_json::Value>,
}

/// The two shapes `Edit::content` takes. It exists for the schema alone:
/// deserialization reads the raw `Value`, and `content_of` is what refuses a
/// third shape with a message a caller can act on (#62).
#[derive(Debug, JsonSchema)]
#[serde(untagged)]
pub enum EditContent {
    /// One value, for a body, a section or a scalar property.
    Text(String),
    /// A sequence, for a list-valued property such as tags or aliases.
    List(Vec<String>),
}

/// Every edit one note takes in one write (#62).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Update {
    /// File path, basename, or #docid.
    pub file: String,
    /// The edits to apply, in order, in one write.
    pub edits: Vec<Edit>,
}

impl Edit {
    /// The writer's edit this one means.
    fn to_writer_edit(&self) -> anyhow::Result<crate::writer::NoteEdit> {
        let target = match (&self.section, &self.property) {
            (Some(_), Some(_)) => {
                anyhow::bail!("an edit names one of section or property, not both")
            }
            (Some(section), None) => crate::writer::EditTarget::Section(section.clone()),
            (None, Some(property)) => crate::writer::EditTarget::Property(property.clone()),
            (None, None) => crate::writer::EditTarget::Body,
        };
        Ok(crate::writer::NoteEdit {
            target,
            mode: self.mode.into(),
            content: content_of(self.content.as_ref())?,
        })
    }
}

/// A string is one value and a list of strings is a sequence. Which one a
/// caller sends is what separates setting a scalar property from setting a
/// list-valued one, so the two shapes are the whole grammar and any third
/// shape is an error (#62).
fn content_of(
    value: Option<&serde_json::Value>,
) -> anyhow::Result<Option<crate::writer::EditContent>> {
    Ok(match value {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(crate::writer::EditContent::Text(s.clone())),
        Some(serde_json::Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    serde_json::Value::String(s) => out.push(s.clone()),
                    other => anyhow::bail!("content is a string or a list of strings, not {other}"),
                }
            }
            Some(crate::writer::EditContent::List(out))
        }
        Some(other) => anyhow::bail!("content is a string or a list of strings, not {other}"),
    })
}

impl Update {
    /// The writer's edit list this request means. Every edit is read before
    /// any of them is applied, so a request that names an impossible target
    /// writes nothing (#62).
    pub fn to_writer_edits(&self) -> anyhow::Result<Vec<crate::writer::NoteEdit>> {
        self.edits.iter().map(Edit::to_writer_edit).collect()
    }

    /// The request `engraph update`'s one-edit form names.
    ///
    /// MCP and HTTP send this struct as JSON, where a string and an array are
    /// two different things a caller writes. A command line has no such
    /// distinction, so `--content` decides it by how many times it appears:
    /// one occurrence is one string, whatever characters it holds, and
    /// repeating the flag is the list a list-valued property reads. The value
    /// is never split, because prose holds commas and there would be no way
    /// to write one (#62).
    pub fn from_cli_edit(
        file: String,
        section: Option<String>,
        property: Option<String>,
        mode: EditMode,
        content: Vec<String>,
    ) -> Self {
        let content = match content.len() {
            0 => None,
            1 => Some(serde_json::Value::String(
                content.into_iter().next().expect("one element"),
            )),
            _ => Some(serde_json::Value::Array(
                content.into_iter().map(serde_json::Value::String).collect(),
            )),
        };
        Update {
            file,
            edits: vec![Edit {
                section,
                property,
                mode,
                content,
            }],
        }
    }

    /// The request an `engraph update` names, whichever of its two forms the
    /// caller used.
    ///
    /// `--edits` is the whole grammar; the flags beside it are the one-edit
    /// form of the same thing, so both build one `Update` and
    /// `to_writer_edits` stays the one converter (#62).
    ///
    /// An edit that needs content and was given none reads stdin through
    /// `read_stdin`, which is how `write append` took a body before `update`
    /// absorbed it. `--edits` carries its own content and a property `remove`
    /// needs none, so neither of those reads it. The read is a closure, and
    /// not the read itself, so the decision is testable without a process
    /// standard input (#62).
    ///
    /// An empty read is refused for a body or a section `replace`. `--mode`
    /// defaults to `replace`, and the calls this absorbed were stricter — a
    /// required `--content` on `write rewrite` and `write edit`, and an
    /// `append` default on the latter — so a piped command that produces
    /// nothing would otherwise blank the note it names. `--content ""` is the
    /// spelling for a deliberate one.
    pub fn from_cli(
        file: String,
        section: Option<String>,
        property: Option<String>,
        mode: EditMode,
        content: Vec<String>,
        edits: Option<String>,
        read_stdin: impl FnOnce() -> anyhow::Result<String>,
    ) -> anyhow::Result<Self> {
        if let Some(json) = edits {
            let edits = serde_json::from_str::<Vec<Edit>>(&json)
                .map_err(|e| anyhow::anyhow!("--edits is not a JSON array of edits: {e}"))?;
            return Ok(Update { file, edits });
        }
        let content = if content.is_empty() && !matches!(mode, EditMode::Remove) {
            let text = read_stdin()?;
            let targets_text = property.is_none();
            if text.trim().is_empty() && targets_text && matches!(mode, EditMode::Replace) {
                anyhow::bail!(
                    "a replace of {} read no content from stdin. \
                     Pass --content, or --content \"\" to blank it deliberately",
                    match &section {
                        Some(heading) => format!("section '{heading}'"),
                        None => "the note's body".to_string(),
                    }
                );
            }
            vec![text]
        } else {
            content
        };
        Ok(Update::from_cli_edit(
            file, section, property, mode, content,
        ))
    }
}

/// What `delete` does to the note it names. An enum and not a string, for the
/// reason [`EditMode`] is one: a string read as `"hard" => Hard, _ => Soft`
/// archives the note whenever the caller misspells `hard`, and the caller is
/// told nothing. The four legal spellings are published to an MCP client and
/// to the OpenAPI spec, and a fifth is refused at the boundary (#62).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, serde::Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum DeleteMode {
    /// Move the note to the archive folder.
    #[default]
    Soft,
    /// Remove the note from disk and from the index.
    Hard,
}

// `clap::ValueEnum` is derived apart from the rest so that the CLI's two
// spellings are the serde ones, lower-case, and not clap's kebab default.
impl clap::ValueEnum for DeleteMode {
    fn value_variants<'a>() -> &'a [Self] {
        &[DeleteMode::Soft, DeleteMode::Hard]
    }
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(match self {
            DeleteMode::Soft => clap::builder::PossibleValue::new("soft"),
            DeleteMode::Hard => clap::builder::PossibleValue::new("hard"),
        })
    }
}

impl std::fmt::Display for DeleteMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DeleteMode::Soft => "soft",
            DeleteMode::Hard => "hard",
        })
    }
}

impl From<DeleteMode> for crate::writer::DeleteMode {
    fn from(mode: DeleteMode) -> Self {
        match mode {
            DeleteMode::Soft => crate::writer::DeleteMode::Soft,
            DeleteMode::Hard => crate::writer::DeleteMode::Hard,
        }
    }
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Delete {
    /// File path, basename, or #docid.
    pub file: String,
    /// `soft` (default) archives the note; `hard` removes it permanently.
    #[arg(long, value_enum, default_value = "soft")]
    #[serde(default, deserialize_with = "deserialize_delete_mode")]
    pub mode: DeleteMode,
}

/// The same null-must-not-fail rule the other defaulted fields take (#60):
/// `#[serde(default)]` alone covers a missing field and not an explicit
/// `null`.
fn deserialize_delete_mode<'de, D>(deserializer: D) -> Result<DeleteMode, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<DeleteMode>::deserialize(deserializer)?.unwrap_or_default())
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

/// `apply` moves files, and it moves them against the preview named here. A
/// misspelled key would read as no key at all and send `apply` to whatever
/// preview was saved last, so an unknown field is refused (#62).
#[derive(Debug, Args, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Migrate {
    /// `preview`, `apply` or `undo`.
    #[arg(long)]
    pub mode: String,
    /// The preview `apply` acts on. This is the one argument the servers
    /// take and the CLI does not: a server caller holds the JSON that
    /// `preview` returned it and passes it back, while the CLI's `preview`
    /// saves the plan itself and its `apply` reads that copy. `#[arg(skip)]`
    /// is what keeps it off the command line, where JSON has no spelling.
    #[arg(skip)]
    #[serde(default)]
    pub preview: Option<serde_json::Value>,
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

    /// `mode` is an enum, so a spelling outside the four is refused where the
    /// request is read and never reaches the writer. This is the successor to
    /// the unknown-op test that went with `FrontmatterOpInput` (#62).
    #[test]
    fn edit_reads_the_four_modes_and_rejects_a_fifth() {
        for mode in ["replace", "prepend", "append", "remove"] {
            let json = format!(r#"{{"mode":"{mode}"}}"#);
            serde_json::from_str::<Edit>(&json)
                .unwrap_or_else(|e| panic!("mode {mode} must parse: {e}"));
        }
        let err = serde_json::from_str::<Edit>(r#"{"mode":"upsert"}"#).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("unknown variant") && message.contains("replace"),
            "the error must name the legal values, got: {message}"
        );
    }

    /// A misspelled `preview` key would read as no preview at all, and
    /// `apply` would then move files against whichever plan was saved last.
    /// The struct refuses the key instead (#62).
    #[test]
    fn migrate_refuses_a_key_it_does_not_know() {
        let good =
            serde_json::from_str::<Migrate>(r#"{"mode":"apply","preview":{"a":1}}"#).unwrap();
        assert!(good.preview.is_some());

        let err =
            serde_json::from_str::<Migrate>(r#"{"mode":"apply","previews":{"a":1}}"#).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("unknown field") && message.contains("previews"),
            "the error must name the key it refused, got: {message}"
        );
    }

    /// One `--content` is one string, commas and all. Splitting it would make
    /// ordinary prose unwritable through the flag form, and would turn a
    /// scalar property's value into a two-item list with no error (#62).
    #[test]
    fn one_cli_content_reaches_the_writer_as_one_string() {
        let u = Update::from_cli_edit(
            "n.md".into(),
            Some("Notes".into()),
            None,
            EditMode::Replace,
            vec!["Hello, world".into()],
        );
        let edits = u.to_writer_edits().unwrap();
        assert!(
            matches!(
                edits[0].content,
                Some(crate::writer::EditContent::Text(ref t)) if t == "Hello, world"
            ),
            "got {:?}",
            edits[0].content
        );
    }

    /// Repeating the flag is how the command line spells a sequence.
    #[test]
    fn a_repeated_cli_content_reaches_the_writer_as_a_list() {
        let u = Update::from_cli_edit(
            "n.md".into(),
            None,
            Some("tags".into()),
            EditMode::Replace,
            vec!["a".into(), "b".into()],
        );
        let edits = u.to_writer_edits().unwrap();
        assert!(
            matches!(
                edits[0].content,
                Some(crate::writer::EditContent::List(ref v)) if v == &["a".to_string(), "b".to_string()]
            ),
            "got {:?}",
            edits[0].content
        );
    }

    /// No `--content` at all is no content, which is what a property `remove`
    /// takes.
    #[test]
    fn no_cli_content_is_no_content() {
        let u = Update::from_cli_edit(
            "n.md".into(),
            None,
            Some("status".into()),
            EditMode::Remove,
            vec![],
        );
        let edits = u.to_writer_edits().unwrap();
        assert!(edits[0].content.is_none());
    }

    /// The stdin read is a closure, so the decision is testable without a
    /// process standard input (#62).
    fn from_cli(
        section: Option<&str>,
        property: Option<&str>,
        mode: EditMode,
        content: Vec<String>,
        edits: Option<&str>,
        stdin: &str,
    ) -> anyhow::Result<Update> {
        let stdin = stdin.to_string();
        Update::from_cli(
            "n.md".into(),
            section.map(String::from),
            property.map(String::from),
            mode,
            content,
            edits.map(String::from),
            move || Ok(stdin),
        )
    }

    /// An omitted `--content` reads stdin, which is how `write append` took a
    /// body before `update` absorbed it (#62).
    #[test]
    fn an_omitted_content_reads_stdin() {
        let u = from_cli(
            Some("Notes"),
            None,
            EditMode::Append,
            vec![],
            None,
            "from a pipe",
        )
        .unwrap();
        let edits = u.to_writer_edits().unwrap();
        assert!(matches!(
            edits[0].content,
            Some(crate::writer::EditContent::Text(ref t)) if t == "from a pipe"
        ));
    }

    /// `--mode` defaults to `replace`, and the calls this absorbed were
    /// stricter — `write rewrite --content` and `write edit --content` were
    /// required, and `write edit`'s default was `append`. A piped command that
    /// produces nothing must not blank the note (#62).
    #[test]
    fn an_empty_stdin_does_not_blank_a_body_or_a_section() {
        for section in [None, Some("Notes")] {
            let err = from_cli(section, None, EditMode::Replace, vec![], None, "")
                .expect_err("an empty replace must be refused");
            let message = format!("{err}");
            assert!(message.contains("read no content from stdin"), "{message}");
            assert!(message.contains("--content \"\""), "{message}");

            // Whitespace alone is the same accident.
            assert!(from_cli(section, None, EditMode::Replace, vec![], None, "\n\n").is_err());
        }
    }

    /// The refusal is for a `replace` that would blank the text. An empty
    /// append is a no-op and needs no refusal, a property edit is not the
    /// body, and `--content ""` is the deliberate spelling (#62).
    #[test]
    fn the_empty_stdin_refusal_covers_a_replace_and_nothing_else() {
        assert!(from_cli(None, None, EditMode::Append, vec![], None, "").is_ok());
        assert!(from_cli(None, Some("status"), EditMode::Replace, vec![], None, "").is_ok());
        assert!(
            from_cli(
                None,
                None,
                EditMode::Replace,
                vec![String::new()],
                None,
                "unread"
            )
            .is_ok()
        );
    }

    /// A `remove` of a property needs no content, and `--edits` carries its
    /// own, so neither reads stdin (#62).
    #[test]
    fn the_forms_that_carry_their_own_content_do_not_read_stdin() {
        let unreadable = || anyhow::bail!("stdin must not be read");

        let u = Update::from_cli(
            "n.md".into(),
            None,
            Some("status".into()),
            EditMode::Remove,
            vec![],
            None,
            unreadable,
        )
        .unwrap();
        assert!(u.to_writer_edits().unwrap()[0].content.is_none());

        let u = Update::from_cli(
            "n.md".into(),
            None,
            None,
            EditMode::Replace,
            vec![],
            Some(r#"[{"mode":"append","content":"x"}]"#.into()),
            unreadable,
        )
        .unwrap();
        assert_eq!(u.edits.len(), 1);

        let err = Update::from_cli(
            "n.md".into(),
            None,
            None,
            EditMode::Replace,
            vec![],
            Some("not json".into()),
            unreadable,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("--edits is not a JSON array of edits"));
    }

    #[test]
    fn an_edit_naming_a_section_and_a_property_is_an_error() {
        let u = Update {
            file: "n.md".into(),
            edits: vec![Edit {
                section: Some("H".into()),
                property: Some("tags".into()),
                mode: EditMode::Replace,
                content: None,
            }],
        };
        let err = u.to_writer_edits().unwrap_err();
        assert!(format!("{err}").contains("one of"));
    }

    #[test]
    fn an_edit_naming_neither_targets_the_body() {
        let u = Update {
            file: "n.md".into(),
            edits: vec![Edit {
                section: None,
                property: None,
                mode: EditMode::Append,
                content: Some(serde_json::json!("line")),
            }],
        };
        let edits = u.to_writer_edits().unwrap();
        assert!(matches!(edits[0].target, crate::writer::EditTarget::Body));
    }

    #[test]
    fn a_list_valued_content_reaches_the_writer_as_a_list() {
        let u = Update {
            file: "n.md".into(),
            edits: vec![Edit {
                section: None,
                property: Some("tags".into()),
                mode: EditMode::Replace,
                content: Some(serde_json::json!(["a", "b"])),
            }],
        };
        let edits = u.to_writer_edits().unwrap();
        assert!(matches!(
            edits[0].content,
            Some(crate::writer::EditContent::List(ref v)) if v == &["a".to_string(), "b".to_string()]
        ));
    }
}
