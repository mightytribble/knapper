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
    ///
    /// It counts the results a caller is shown, so a merged block is one
    /// (#102). Fewer come back only when the vault holds fewer: the
    /// candidates are capped at `[ranking] candidates` and the answer floor
    /// removes what is not an answer, so a large `top_n` reaches that
    /// ceiling rather than the number asked for.
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
    /// Filter to notes carrying a custom property, as `NAME`, or carrying
    /// it with one value equal to `VALUE`, as `NAME=VALUE`. The split is on
    /// the first `=`, values compare as text, and a name with spaces is
    /// quoted. One property per call; `properties` lists the names and
    /// values a vault holds (#66).
    #[arg(long)]
    #[serde(default)]
    pub property: Option<String>,
    /// Filter to notes that link to this note, named the way a wikilink
    /// names it. With `property`, only links filed under that property
    /// count. An unknown note is an error naming the nearest one (#66).
    #[arg(long)]
    #[serde(default)]
    pub links_to: Option<String>,
    /// Filter to the notes this note links to, named the way a wikilink
    /// names it. With `property`, only links filed under that property
    /// count. An unknown note is an error naming the nearest one (#66).
    #[arg(long)]
    #[serde(default)]
    pub linked_from: Option<String>,
    /// Token budget for the returned text. Fill is greedy in rank order; the
    /// first result is always included. Omit for the configured default (#35).
    #[arg(long = "tokens")]
    pub budget_tokens: Option<u32>,
    /// Return every result's full text, ignoring the token budget (#35).
    #[arg(long, conflicts_with = "summaries")]
    #[serde(default)]
    pub full: bool,
    /// Return breadcrumb and provenance only, no text, for every result (#35).
    #[arg(long)]
    #[serde(default)]
    pub summaries: bool,
    /// Include the cross-encoder's relevance score on each result (#35).
    #[arg(long)]
    #[serde(default)]
    pub scores: bool,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Read {
    /// File path, basename, or #docid.
    pub file: String,
    /// Read one section by its heading. Omit for the whole note.
    ///
    /// The heading is one heading's own text, or its full path from the
    /// note's top heading down, joined with ` > `: `Spells` finds the first
    /// section of that name, and `Stat Block > Spells` finds the one under
    /// `Stat Block`. A partial path finds nothing. The match folds case, and
    /// a bold-only line is a section too, in either spelling — `**Spells**`
    /// and `Spells` name the same one (#69).
    ///
    /// A section read narrows `content` to that section's body. The heading
    /// comes back beside it, in the `section` object with the level and the
    /// span, so what a read returns is what an `update` takes back (#96).
    #[arg(long)]
    pub section: Option<String>,
    /// Return the note's metadata — its frontmatter, its inbound and outbound
    /// links, and its size — instead of its content. It describes the whole
    /// note, so it cannot be combined with `--section` (#80).
    #[arg(long, conflicts_with = "section")]
    #[serde(default)]
    pub metadata: bool,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct List {
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
    /// Filter to notes carrying a custom property, as `NAME`, or carrying
    /// it with one value equal to `VALUE`, as `NAME=VALUE`. The split is on
    /// the first `=`, values compare as text, and a name with spaces is
    /// quoted. One property per call; `properties` lists the names and
    /// values a vault holds (#66).
    #[arg(long)]
    #[serde(default)]
    pub property: Option<String>,
    /// Filter to notes that link to this note, named the way a wikilink
    /// names it. With `property`, only links filed under that property
    /// count. An unknown note is an error naming the nearest one (#66).
    #[arg(long)]
    #[serde(default)]
    pub links_to: Option<String>,
    /// Filter to the notes this note links to, named the way a wikilink
    /// names it. With `property`, only links filed under that property
    /// count. An unknown note is an error naming the nearest one (#66).
    #[arg(long)]
    #[serde(default)]
    pub linked_from: Option<String>,
    /// Filter to notes created by one agent.
    #[arg(long)]
    pub created_by: Option<String>,
    /// Maximum notes to answer. Absent, the listing holds every note the
    /// scope admits — a caller that wants less names a scope or a limit,
    /// and one cap on one surface would make this field mean two things
    /// (#62, #68). `0` answers none, which is what the number says.
    #[arg(long)]
    #[serde(default)]
    pub limit: Option<usize>,
    /// Answer each note's heading outline beneath its path. It reads every
    /// listed note from disk, because the index does not hold the outline;
    /// an undetailed listing touches no file (#68).
    #[arg(long)]
    #[serde(default)]
    pub detailed: bool,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Match {
    /// The literal string to look for. It is text and not a pattern: `.`,
    /// `*` and `[` are themselves.
    pub pattern: String,
    /// Compare the pattern exactly. The default folds case, which is what
    /// the keyword index does.
    #[arg(long)]
    #[serde(default)]
    pub case_sensitive: bool,
    /// An alias of `all`. A term starting with `/` is a directory path from
    /// the vault root instead of a tag, case-sensitive; a trailing `/`
    /// scopes to its subtree.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub scope: Vec<String>,
    /// Look only in notes carrying every term. A term is a tag path; a
    /// trailing `/` or `/*` matches the tag and its descendants. A term
    /// starting with `/` is a directory path from the vault root instead,
    /// case-sensitive, with a trailing `/` scoping to its subtree.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// Look only in notes carrying at least one of these terms.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Skip notes carrying any of these terms.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
    /// Maximum matched lines to report. Absent, every one comes back — a
    /// caller that wants less names a scope or a limit, which is `list`'s
    /// rule for the same field (#68). `0` reports none. The note and line
    /// counts are whole whatever this says, because the count is the answer
    /// to the absence question.
    #[arg(long)]
    #[serde(default)]
    pub limit: Option<usize>,
}

impl Match {
    /// The notes to look in, with `scope` folded into `all`.
    pub fn scope(&self) -> anyhow::Result<crate::tags::Scope> {
        let all = crate::tags::merge_scope_alias(self.scope.clone(), self.all.clone());
        crate::tags::Scope::parse(&all, &self.any, &self.none)
    }
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Tags {
    /// Limit to one tag and its descendants, as `type/` or `type/*`. Omit
    /// for the whole vocabulary.
    #[arg(long)]
    pub under: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Properties {
    /// One property's distinct values, each with its kind and the notes
    /// carrying it, instead of the vocabulary. The call to make before
    /// filtering with `property=NAME=VALUE` (#66).
    #[arg(long)]
    pub name: Option<String>,
}

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct VaultMap {}

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

/// The frontmatter in `content` is written as given. `tags` are added to
/// the note's own `tags` list; no other key is written.
#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Create {
    /// Note content. The CLI reads stdin when this is omitted.
    #[arg(long)]
    pub content: Option<String>,
    /// Filename for the note. A bare name gets `.md` appended; a name that
    /// already ends in `.md` is kept. It becomes the note's breadcrumb root,
    /// so name the file the way it should read as provenance (#47).
    #[arg(long)]
    pub filename: String,
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
    /// The section to edit: a heading's own text, or its full path from the
    /// note's top heading down, joined with ` > ` (#69).
    pub section: Option<String>,
    /// The frontmatter property to edit. A property edit keeps the key
    /// where it is and in the list style the note already uses, and a list
    /// with no items writes an empty list. A key the note does not already
    /// carry is appended at the end of the frontmatter, unless `after` or
    /// `before` names where it goes (#113).
    pub property: Option<String>,
    /// Place a property key the note does not already carry directly below
    /// the key named here. It names one of the note's own frontmatter keys,
    /// and a key the note does not carry is refused. A key the note already
    /// has keeps the place the file gave it, so this is inert there and an
    /// edit can be re-run (#113).
    pub after: Option<String>,
    /// Place a property key the note does not already carry directly above
    /// the key named here, and above the comment line that introduces it.
    /// It is the only way to name the top of the frontmatter. The rules are
    /// `after`'s, and naming both is refused (#113).
    pub before: Option<String>,
    /// The section's new heading text, when the edit renames it. It renames
    /// the section `section` names, so an edit that carries a heading and no
    /// section is refused, and `content` is optional beside it, because a
    /// rename does not restate the body.
    ///
    /// The value is the heading's text and the note keeps its markup: a
    /// `###` stays a `###` and a promoted bold line keeps its markers. A name
    /// another section of the note already holds is refused, since two
    /// sections of one name leave both unaddressable by name (#97).
    pub heading: Option<String>,
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
            heading: self.heading.clone(),
            mode: self.mode.into(),
            content: content_of(self.content.as_ref())?,
            placement: crate::frontmatter::KeyPlacement::new(
                self.after.as_deref(),
                self.before.as_deref(),
            )?,
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

/// The one edit `knapper update`'s flag form names, as the command line
/// spells it: the target — `--section`, `--property`, or neither for the
/// body — the `--heading` that renames a section, and what the edit does to
/// what it names. `--edits` is the whole grammar and this is the one-edit
/// form of the same thing, so both build one [`Update`] (#62, #97).
#[derive(Debug)]
pub struct CliEdit {
    pub section: Option<String>,
    pub property: Option<String>,
    pub heading: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub mode: EditMode,
    pub content: Vec<String>,
}

impl Update {
    /// The writer's edit list this request means. Every edit is read before
    /// any of them is applied, so a request that names an impossible target
    /// writes nothing (#62).
    pub fn to_writer_edits(&self) -> anyhow::Result<Vec<crate::writer::NoteEdit>> {
        self.edits.iter().map(Edit::to_writer_edit).collect()
    }

    /// The request `knapper update`'s one-edit form names.
    ///
    /// MCP and HTTP send this struct as JSON, where a string and an array are
    /// two different things a caller writes. A command line has no such
    /// distinction, so `--content` decides it by how many times it appears:
    /// one occurrence is one string, whatever characters it holds, and
    /// repeating the flag is the list a list-valued property reads. The value
    /// is never split, because prose holds commas and there would be no way
    /// to write one (#62).
    pub fn from_cli_edit(file: String, edit: CliEdit) -> Self {
        let CliEdit {
            section,
            property,
            heading,
            after,
            before,
            mode,
            content,
        } = edit;
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
                heading,
                after,
                before,
                mode,
                content,
            }],
        }
    }

    /// The request a `knapper update` names, whichever of its two forms the
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
        edit: CliEdit,
        edits: Option<String>,
        read_stdin: impl FnOnce() -> anyhow::Result<String>,
    ) -> anyhow::Result<Self> {
        let CliEdit {
            section,
            property,
            heading,
            after,
            before,
            mode,
            content,
        } = edit;
        if let Some(json) = edits {
            let edits = serde_json::from_str::<Vec<Edit>>(&json)
                .map_err(|e| anyhow::anyhow!("--edits is not a JSON array of edits: {e}"))?;
            return Ok(Update { file, edits });
        }
        // A rename does not restate the body, so `--heading` with no
        // `--content` is a complete edit and there is nothing to read (#97).
        let content =
            if content.is_empty() && !matches!(mode, EditMode::Remove) && heading.is_none() {
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
            file,
            CliEdit {
                section,
                property,
                heading,
                after,
                before,
                mode,
                content,
            },
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

#[derive(Debug, Args, Deserialize, JsonSchema)]
pub struct Validate {
    /// A vault-relative note reference (path or basename) to check one file.
    /// Omit it to check a scope or the whole vault. Mutually exclusive with
    /// the scope filters.
    #[arg(conflicts_with_all = ["scope", "all", "any", "none"])]
    pub path: Option<String>,
    /// An alias of `all`. A term starting with `/` is a directory path from
    /// the vault root instead of a tag, case-sensitive; a trailing `/` scopes
    /// to its subtree.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub scope: Vec<String>,
    /// Check only notes carrying every term. A directory term (leading `/`,
    /// case-sensitive, trailing `/` its subtree) scopes by folder instead.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub all: Vec<String>,
    /// Check only notes carrying at least one of these terms.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub any: Vec<String>,
    /// Skip notes carrying any of these terms.
    #[arg(long, value_delimiter = ',')]
    #[serde(default, deserialize_with = "deserialize_tag_list")]
    pub none: Vec<String>,
    /// Treat warnings as gating: exit non-zero when any warning is present.
    #[arg(long)]
    #[serde(default)]
    pub strict: bool,
}

impl Validate {
    /// The addressing target, or an error when a note reference and a scope
    /// are both given (the servers' equivalent of the CLI's `conflicts_with`).
    pub fn target(&self) -> anyhow::Result<crate::validate::Target> {
        let all_terms = crate::tags::merge_scope_alias(self.scope.clone(), self.all.clone());
        let has_scope = !all_terms.is_empty() || !self.any.is_empty() || !self.none.is_empty();
        match (&self.path, has_scope) {
            (Some(_), true) => {
                anyhow::bail!("a note reference and a scope are mutually exclusive")
            }
            (Some(p), false) => Ok(crate::validate::Target::Note(p.clone())),
            (None, true) => Ok(crate::validate::Target::Scope(crate::tags::Scope::parse(
                &all_terms, &self.any, &self.none,
            )?)),
            (None, false) => Ok(crate::validate::Target::Vault),
        }
    }
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

    /// An absent limit is no limit: a bare listing answers every note the
    /// scope admits, so an agent that scoped to a subtree gets that
    /// subtree (#68).
    #[test]
    fn limit_reads_absent_as_no_limit() {
        let list: List = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(list.limit, None);
    }

    /// An explicit JSON `null` — what a JS or Python caller sends for an
    /// absent optional — reads the same as absent and must not fail
    /// deserialization (#60, #68).
    #[test]
    fn limit_reads_null_as_no_limit() {
        let list: List = serde_json::from_str(r#"{"limit":null}"#).unwrap();
        assert_eq!(list.limit, None);
    }

    /// `serde_urlencoded` reads `detailed=true` and not a bare `detailed`,
    /// so the value is required on the query string; absent, the field is
    /// false and the listing touches no file (#68).
    #[test]
    fn detailed_reads_from_a_query_string_and_defaults_to_false() {
        let list: List = serde_urlencoded::from_str("detailed=true").unwrap();
        assert!(list.detailed);
        let bare: List = serde_urlencoded::from_str("all=type/").unwrap();
        assert!(!bare.detailed);
    }

    /// A GET query string carries the number as text; `serde_urlencoded`
    /// reads it into the same field (#68).
    #[test]
    fn limit_reads_a_number_from_a_query_string() {
        let list: List = serde_urlencoded::from_str("limit=5").unwrap();
        assert_eq!(list.limit, Some(5));
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
            CliEdit {
                section: Some("Notes".into()),
                property: None,
                heading: None,
                mode: EditMode::Replace,
                content: vec!["Hello, world".into()],
                after: None,
                before: None,
            },
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
            CliEdit {
                section: None,
                property: Some("tags".into()),
                heading: None,
                mode: EditMode::Replace,
                content: vec!["a".into(), "b".into()],
                after: None,
                before: None,
            },
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
            CliEdit {
                section: None,
                property: Some("status".into()),
                heading: None,
                mode: EditMode::Remove,
                content: vec![],
                after: None,
                before: None,
            },
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
            CliEdit {
                section: section.map(String::from),
                property: property.map(String::from),
                heading: None,
                mode,
                content,
                after: None,
                before: None,
            },
            edits.map(String::from),
            move || Ok(stdin),
        )
    }

    /// The same, for the call that renames the section it names (#97).
    fn from_cli_rename(
        section: &str,
        heading: &str,
        content: Vec<String>,
        stdin: &str,
    ) -> anyhow::Result<Update> {
        let stdin = stdin.to_string();
        Update::from_cli(
            "n.md".into(),
            CliEdit {
                section: Some(section.to_string()),
                property: None,
                heading: Some(heading.to_string()),
                mode: EditMode::Replace,
                content,
                after: None,
                before: None,
            },
            None,
            move || Ok(stdin),
        )
    }

    /// A rename does not restate the body, so `--heading` with no `--content`
    /// is a whole edit: it reads no stdin, and the refusal that guards an
    /// empty piped replace does not fire on it (#97).
    #[test]
    fn a_rename_with_no_content_reads_no_stdin() {
        let u = from_cli_rename("Old name", "New name", vec![], "").unwrap();
        let edits = u.to_writer_edits().unwrap();
        assert!(edits[0].content.is_none());
        assert_eq!(edits[0].heading.as_deref(), Some("New name"));
    }

    /// A rename that also writes a body is one edit, and the content is the
    /// body alone (#97).
    #[test]
    fn a_rename_carries_content_when_it_is_given_some() {
        let u = from_cli_rename("Old name", "New name", vec!["The new body.".into()], "").unwrap();
        let edits = u.to_writer_edits().unwrap();
        assert_eq!(edits[0].heading.as_deref(), Some("New name"));
        assert!(matches!(
            edits[0].content,
            Some(crate::writer::EditContent::Text(ref t)) if t == "The new body."
        ));
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
            CliEdit {
                section: None,
                property: Some("status".into()),
                heading: None,
                mode: EditMode::Remove,
                content: vec![],
                after: None,
                before: None,
            },
            None,
            unreadable,
        )
        .unwrap();
        assert!(u.to_writer_edits().unwrap()[0].content.is_none());

        let u = Update::from_cli(
            "n.md".into(),
            CliEdit {
                section: None,
                property: None,
                heading: None,
                mode: EditMode::Replace,
                content: vec![],
                after: None,
                before: None,
            },
            Some(r#"[{"mode":"append","content":"x"}]"#.into()),
            unreadable,
        )
        .unwrap();
        assert_eq!(u.edits.len(), 1);

        let err = Update::from_cli(
            "n.md".into(),
            CliEdit {
                section: None,
                property: None,
                heading: None,
                mode: EditMode::Replace,
                content: vec![],
                after: None,
                before: None,
            },
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
                heading: None,
                mode: EditMode::Replace,
                content: None,
                after: None,
                before: None,
            }],
        };
        let err = u.to_writer_edits().unwrap_err();
        assert!(format!("{err}").contains("one of"));
    }

    /// Two anchors name two places, and the refusal has to reach the edit
    /// list rather than living in the frontmatter module alone (#113).
    #[test]
    fn an_edit_naming_both_after_and_before_is_an_error() {
        let u = Update {
            file: "n.md".into(),
            edits: vec![Edit {
                section: None,
                property: Some("ties".into()),
                heading: None,
                mode: EditMode::Replace,
                content: None,
                after: Some("name".into()),
                before: Some("tags".into()),
            }],
        };
        let err = u.to_writer_edits().unwrap_err();
        assert!(format!("{err}").contains("not both"), "{err}");
    }

    #[test]
    fn an_edit_naming_neither_targets_the_body() {
        let u = Update {
            file: "n.md".into(),
            edits: vec![Edit {
                section: None,
                property: None,
                heading: None,
                mode: EditMode::Append,
                content: Some(serde_json::json!("line")),
                after: None,
                before: None,
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
                heading: None,
                mode: EditMode::Replace,
                content: Some(serde_json::json!(["a", "b"])),
                after: None,
                before: None,
            }],
        };
        let edits = u.to_writer_edits().unwrap();
        assert!(matches!(
            edits[0].content,
            Some(crate::writer::EditContent::List(ref v)) if v == &["a".to_string(), "b".to_string()]
        ));
    }

    /// The target resolves to whichever of a note, a scope or the whole vault
    /// the request names, and a note and a scope together are refused (#70).
    #[test]
    fn validate_target_is_note_scope_or_vault() {
        // nothing -> whole vault
        let v = Validate {
            path: None,
            scope: vec![],
            all: vec![],
            any: vec![],
            none: vec![],
            strict: false,
        };
        assert!(matches!(
            v.target().unwrap(),
            crate::validate::Target::Vault
        ));
        // a path -> that note
        let v = Validate {
            path: Some("a.md".into()),
            scope: vec![],
            all: vec![],
            any: vec![],
            none: vec![],
            strict: false,
        };
        assert!(matches!(
            v.target().unwrap(),
            crate::validate::Target::Note(_)
        ));
        // a scope -> that scope
        let v = Validate {
            path: None,
            scope: vec!["/Work/".into()],
            all: vec![],
            any: vec![],
            none: vec![],
            strict: false,
        };
        assert!(matches!(
            v.target().unwrap(),
            crate::validate::Target::Scope(_)
        ));
        // path and scope together -> error
        let v = Validate {
            path: Some("a".into()),
            scope: vec!["/Work/".into()],
            all: vec![],
            any: vec![],
            none: vec![],
            strict: false,
        };
        assert!(v.target().is_err());
    }
}
