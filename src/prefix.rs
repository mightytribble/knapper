//! What a chunk is embedded as: the document template's `title:` field
//! (issue #36) and the body, which can carry a contextual prefix (issue #2,
//! **off by default** — see below). [`embed_inputs`] composes both, and it is the
//! one composition the indexer and the write pipeline share.
//!
//! # The contextual prefix
//!
//! Only the raw chunk body reaches the embedder, so a chunk is findable only
//! through words appearing inside its own prose. That misses two things: a
//! document's name, which encyclopedic reference material often states once and
//! then refers to as "it"; and a chunk's ancestor headings, which
//! [`crate::chunker::structure_chunk`] drops when it makes a subsection a
//! sibling of its parent rather than a child.
//!
//! When enabled, each chunk is embedded as a prefix plus its text, the prefix
//! carrying the document's display name, path, aliases and tags along with the
//! chunk's heading path. The prefix reaches the embedder and nothing else:
//! stored `text`, `snippet` and FTS content are untouched, so it cannot leak
//! into a displayed result or be matched as a keyword.
//!
//! # Why the default is off
//!
//! The prefix is a **per-file constant**. Adding the same string to every chunk
//! of a document moves all of its vectors the same way, so what it buys in
//! separation *between* documents it spends on separation *within* one — and
//! since issue #6 the within-document ordering is what decides which section a
//! search returns.
//!
//! Measured on the eval vault (`eval/probes.md`), it moved the conceptual probe
//! from rank 5 to rank 2 and pushed the exact-name probe's answer out of the top
//! 20 entirely. The damage scales with the prefix's share of the embedded text —
//! a 20-token section given a 30-token identity prefix becomes mostly identity —
//! so every component is separately switchable and the trade is left to the
//! caller's own measurements.

use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;
use crate::config::BreadcrumbRoot;
use crate::llm::{DocumentTitle, EmbedDoc};

/// Which components the contextual prefix carries.
///
/// Every component is separately switchable because prefix *length* turned out
/// to be the thing that matters — see the module docs. The display name is the
/// exception: it is the irreducible identity signal, so `enabled` covers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrefixConfig {
    /// Build a contextual prefix at all. Off means chunks are embedded exactly
    /// as they are stored, which is upstream's behaviour and the default.
    pub enabled: bool,
    /// Include the vault-relative path. Largely restates the display name for
    /// vaults whose filenames are their titles.
    pub path: bool,
    /// Include the chunk's ancestor heading path. The only component that
    /// varies between chunks of one document, so the only one that cannot
    /// flatten them together.
    pub heading: bool,
    /// Include the document's frontmatter `tags`.
    pub tags: bool,
    /// Include the document's frontmatter `aliases`. Often near-duplicates of
    /// the name (`Archdragon` / `Archdragons` / `Arch-Dragon`), which multiplies
    /// the name's weight rather than adding a new signal.
    pub aliases: bool,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: true,
            heading: true,
            tags: true,
            aliases: true,
        }
    }
}

impl PrefixConfig {
    /// Every component on. What `enabled = true` in `config.toml` means before
    /// any component is switched off, and what the eval runs were measured with.
    pub fn full() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}

/// Document-level identity, parsed once per file and shared by all its chunks.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DocContext {
    /// Frontmatter `name` if present, else the filename stem.
    ///
    /// Not the breadcrumb root since #46 — `name` is a convention engraph reads
    /// and never writes, and it is not Obsidian's. See [`BreadcrumbRoot`].
    pub name: String,
    /// The filename without its extension or folders. Not identifying: 14 stems
    /// in the calibration vault name more than one file.
    pub stem: String,
    /// Vault-relative path, as stored in `files.path`. The breadcrumb root.
    pub rel_path: String,
    pub aliases: Vec<String>,
    pub tags: Vec<String>,
}

impl DocContext {
    /// Parse a file's identity from its path and raw content (frontmatter included).
    pub fn from_file(rel_path: &str, content: &str) -> Self {
        let (frontmatter, _body) = crate::writer::split_frontmatter(content);
        let (scalars, tags, aliases) = crate::writer::parse_frontmatter_fields(&frontmatter);

        let name = scalars
            .get("name")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| filename_stem(rel_path));

        Self {
            name,
            stem: filename_stem(rel_path),
            rel_path: rel_path.to_string(),
            aliases,
            tags,
        }
    }

    /// Build the prefix for one chunk. Empty when `cfg.enabled` is false, and
    /// never ends in a newline — [`embed_text`] adds the separator.
    fn prefix_for(&self, chunk: &Chunk, cfg: PrefixConfig) -> String {
        if !cfg.enabled {
            return String::new();
        }

        let mut lines: Vec<String> = Vec::with_capacity(3);
        lines.push(match cfg.path && !self.rel_path.is_empty() {
            true => format!("{} — {}", self.name, self.rel_path),
            false => self.name.clone(),
        });

        // Aliases and tags share a line: both are short lists, and a line each
        // would spend prefix budget on labels rather than terms.
        let mut meta: Vec<String> = Vec::with_capacity(2);
        if cfg.aliases && !self.aliases.is_empty() {
            meta.push(format!("aliases: {}", self.aliases.join(", ")));
        }
        if cfg.tags && !self.tags.is_empty() {
            meta.push(format!("tags: {}", self.tags.join(", ")));
        }
        if !meta.is_empty() {
            lines.push(meta.join(" | "));
        }

        // The chunk's own text already opens with its heading line, but only its
        // own: `### Combat` under `## Abilities` loses the parent entirely once
        // structure_chunk makes them sibling chunks. The path restores it, in
        // plain text rather than `#` syntax.
        if cfg.heading && !chunk.heading_path.is_empty() {
            lines.push(chunk.heading_path.join(" > "));
        }

        lines.join("\n")
    }
}

/// The body one chunk is embedded as. Storage is unaffected.
fn embed_text(doc: &DocContext, chunk: &Chunk, cfg: PrefixConfig) -> String {
    let prefix = doc.prefix_for(chunk, cfg);
    if prefix.is_empty() {
        return chunk.text.clone();
    }
    format!("{prefix}\n{}", chunk.text)
}

/// Everything that decides what a chunk is embedded as: the title field
/// (issue #36) and how the body is composed (issue #2).
///
/// The two travel together because both paths into the embedder —
/// [`crate::indexer::index_file`] and the write pipeline — have to use the same
/// pair. A caller that threads one and forgets the other writes vectors into a
/// space it does not share, and nothing downstream can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbedComposition {
    pub prefix: PrefixConfig,
    pub title: DocumentTitle,
    /// What leads the breadcrumb (issue #46). Travels with the other two for
    /// the same reason they travel together: it changes what a stored vector
    /// means, and a caller that threads one and forgets it writes into a space
    /// it does not share.
    pub root: BreadcrumbRoot,
}

impl EmbedComposition {
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            prefix: config.embedding_prefix,
            title: config.embedding_prompt.document_title,
            root: config.breadcrumb_root,
        }
    }
}

/// One chunk exactly as the embedder is shown it: the title field and the body.
///
/// Owned, because callers hold the whole file's worth and borrow
/// [`EmbedDoc`]s from it — [`crate::llm::EmbedModel::embed_batch`] takes
/// borrowed pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedInput {
    /// What fills the document template's `title:` field (issue #36).
    pub title: String,
    /// The body, prefix included when `[embedding_prefix]` is on (issue #2).
    pub text: String,
}

impl EmbedInput {
    pub fn as_doc(&self) -> EmbedDoc<'_> {
        EmbedDoc::new(&self.title, &self.text)
    }
}

/// What a file's chunks are embedded as, both fields, in one place.
///
/// Both callers of the embed path go through this — `indexer::index_file` and
/// `writer::precompute_chunks` — because a note written through the write
/// pipeline lands in the same vector space as an indexed one, and two
/// compositions that can disagree eventually do.
pub fn embed_inputs(doc: &DocContext, chunks: &[Chunk], cfg: EmbedComposition) -> Vec<EmbedInput> {
    chunks
        .iter()
        .map(|chunk| EmbedInput {
            title: title_for(doc, chunk, cfg.title, cfg.root),
            text: embed_text(doc, chunk, cfg.prefix),
        })
        .collect()
}

/// The string that fills the document template's `title:` field (issue #36).
///
/// An empty result is the documented literal `none` — see
/// [`crate::llm::PromptFormat::format_document`].
fn title_for(doc: &DocContext, chunk: &Chunk, cfg: DocumentTitle, root: BreadcrumbRoot) -> String {
    match cfg {
        DocumentTitle::None => String::new(),
        DocumentTitle::Note => doc.name.clone(),
        DocumentTitle::Breadcrumb => breadcrumb(doc, chunk, root),
    }
}

/// `Note Title > H1 > H2 > H3` — design §5.4's rule, and the one string all
/// three limbs of it carry.
///
/// Two limbs read this function. The embedding limb fills the document
/// template's `title:` field with it (issue #36); the lexical limb stores it in
/// `chunks.heading_path`, which is the column the keyword index is declared
/// over (issue #37). One function, because two compositions of the same rule
/// eventually disagree, and the disagreement would be invisible: each limb
/// looks correct on its own.
pub fn breadcrumb(doc: &DocContext, chunk: &Chunk, root: BreadcrumbRoot) -> String {
    let head = match root {
        BreadcrumbRoot::Path => doc.rel_path.as_str(),
        BreadcrumbRoot::Name => doc.name.as_str(),
        BreadcrumbRoot::Stem => doc.stem.as_str(),
    };
    let mut parts: Vec<&str> = Vec::with_capacity(1 + chunk.heading_path.len());
    if !head.is_empty() {
        parts.push(head);
    }
    parts.extend(chunk.heading_path.iter().map(String::as_str));
    parts.join(" > ")
}

/// What the keyword lane indexes beside a chunk's body (issue #37).
///
/// Stored on the `chunks` row rather than written into the FTS index directly:
/// `chunks_fts` is an external-content table over `chunks`, so these columns
/// *are* what it indexes, and `'rebuild'` re-derives the index from them. That
/// is what makes #11's bug class — the keyword index holding a different string
/// from the chunk table — unreachable rather than fixed.
///
/// Both fields are stored whatever `[fts]` says. The config decides which
/// columns the index is *declared* over, so turning one off costs a rebuild of
/// the keyword index and never a re-index of the vault.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LexicalFields {
    /// The breadcrumb. See [`breadcrumb`].
    pub heading_path: String,
    /// The file's frontmatter tags, sorted and space separated. Sorted because
    /// the order frontmatter happens to list them in is not information, and a
    /// stable string keeps two indexes of the same file byte-identical.
    pub tags_text: String,
}

/// The lexical fields for each chunk of a file, in chunk order.
///
/// The counterpart of [`embed_inputs`], and shared by the same two callers for
/// the same reason: `indexer::index_file` and `writer::precompute_chunks` write
/// rows into one table.
pub fn lexical_fields(
    doc: &DocContext,
    chunks: &[Chunk],
    root: BreadcrumbRoot,
) -> Vec<LexicalFields> {
    let mut tags: Vec<&str> = doc.tags.iter().map(String::as_str).collect();
    tags.sort_unstable();
    tags.dedup();
    let tags_text = tags.join(" ");

    chunks
        .iter()
        .map(|chunk| LexicalFields {
            heading_path: breadcrumb(doc, chunk, root),
            tags_text: tags_text.clone(),
        })
        .collect()
}

/// `lore/bestiary/archdragon.md` → `archdragon`.
fn filename_stem(rel_path: &str) -> String {
    std::path::Path::new(rel_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| rel_path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lexical fields of a file: a breadcrumb per chunk, one tag string for
    /// all of them, sorted (issue #37).
    #[test]
    fn the_lexical_fields_are_a_breadcrumb_and_the_sorted_tags() {
        let doc = DocContext {
            name: "Archdragon".into(),
            stem: "archdragon".into(),
            rel_path: "lore/archdragon.md".into(),
            aliases: vec![],
            tags: vec!["zebra".into(), "apex".into(), "apex".into()],
        };
        let chunks = [
            chunk(&["Abilities", "Combat"], "Opens at range."),
            chunk(&[], "No heading here."),
        ];

        // The path leads, extension included: a breadcrumb whose first segment
        // names a file on disk (issue #46).
        let fields = lexical_fields(&doc, &chunks, BreadcrumbRoot::Path);
        assert_eq!(
            fields[0].heading_path,
            "lore/archdragon.md > Abilities > Combat"
        );
        assert_eq!(fields[1].heading_path, "lore/archdragon.md");
        assert!(fields.iter().all(|f| f.tags_text == "apex zebra"));

        // The control #46 was measured against, and the two rejected roots.
        let named = lexical_fields(&doc, &chunks, BreadcrumbRoot::Name);
        assert_eq!(named[0].heading_path, "Archdragon > Abilities > Combat");
        let stem = lexical_fields(&doc, &chunks, BreadcrumbRoot::Stem);
        assert_eq!(stem[0].heading_path, "archdragon > Abilities > Combat");
    }

    /// One rule, one composition: the string the keyword index stores is the
    /// string the embedder is given as a title.
    #[test]
    fn both_limbs_of_the_rule_carry_one_string() {
        let doc = DocContext {
            name: "Archdragon".into(),
            stem: "archdragon".into(),
            rel_path: "lore/archdragon.md".into(),
            aliases: vec![],
            tags: vec![],
        };
        let chunks = [chunk(&["Abilities"], "Flight.")];

        // Every root, because the invariant is that the two limbs agree — not
        // that either of them says any particular thing.
        for root in [
            BreadcrumbRoot::Path,
            BreadcrumbRoot::Name,
            BreadcrumbRoot::Stem,
        ] {
            let composition = EmbedComposition {
                prefix: PrefixConfig::default(),
                title: DocumentTitle::Breadcrumb,
                root,
            };
            assert_eq!(
                embed_inputs(&doc, &chunks, composition)[0].title,
                lexical_fields(&doc, &chunks, root)[0].heading_path,
                "{root:?}"
            );
        }
    }

    fn chunk(heading_path: &[&str], text: &str) -> Chunk {
        Chunk {
            heading: heading_path.last().map(|h| format!("## {h}")),
            heading_path: heading_path.iter().map(|s| s.to_string()).collect(),
            text: text.to_string(),
            snippet: text.chars().take(200).collect(),
        }
    }

    const ARCHDRAGON: &str = "---\nname: Archdragon\naliases:\n  - Elder Wyrm\ntags:\n  - dragon\n  - apex\n---\n\n## Definition\n**Rank**: SS\n";

    #[test]
    fn doc_context_reads_frontmatter_identity() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        assert_eq!(doc.name, "Archdragon");
        assert_eq!(doc.rel_path, "lore/bestiary/archdragon.md");
        assert_eq!(doc.aliases, vec!["Elder Wyrm"]);
        assert_eq!(doc.tags, vec!["dragon", "apex"]);
    }

    #[test]
    fn doc_name_falls_back_to_the_filename_stem() {
        let doc = DocContext::from_file("npcs/archivist-lenne.md", "No frontmatter here.\n");
        assert_eq!(doc.name, "archivist-lenne");
        assert!(doc.tags.is_empty());
    }

    /// The defect that motivated the issue: this chunk's text never says
    /// "Archdragon", so before the prefix nothing in its vector did either.
    #[test]
    fn a_chunk_that_never_names_its_subject_is_embedded_with_the_name() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let c = chunk(
            &["Definition"],
            "## Definition\n**Rank**: SS • **Levels**: 150-511",
        );
        assert!(!c.text.contains("Archdragon"));

        let embedded = embed_text(&doc, &c, PrefixConfig::full());
        assert!(embedded.contains("Archdragon"));
        assert!(embedded.contains("lore/bestiary/archdragon.md"));
        assert!(embedded.ends_with(&c.text));
    }

    #[test]
    fn ancestor_headings_survive_into_the_prefix() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let c = chunk(&["Abilities", "Combat"], "### Combat\nBreath weapon.");

        let embedded = embed_text(&doc, &c, PrefixConfig::full());
        // `Abilities` appears nowhere in the chunk's own text — structure_chunk
        // makes subsections siblings of their parent, not children.
        assert!(!c.text.contains("Abilities"));
        assert!(embedded.contains("Abilities > Combat"));
    }

    #[test]
    fn full_prefix_shape() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let c = chunk(&["Definition"], "## Definition\n**Rank**: SS");

        assert_eq!(
            embed_text(&doc, &c, PrefixConfig::full()),
            "Archdragon — lore/bestiary/archdragon.md\n\
             aliases: Elder Wyrm | tags: dragon, apex\n\
             Definition\n\
             ## Definition\n**Rank**: SS"
        );
    }

    #[test]
    fn tags_and_aliases_are_individually_switchable() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let c = chunk(&["Definition"], "body");

        let no_tags = embed_text(
            &doc,
            &c,
            PrefixConfig {
                tags: false,
                ..PrefixConfig::full()
            },
        );
        assert!(no_tags.contains("aliases: Elder Wyrm"));
        assert!(!no_tags.contains("tags:"));

        let neither = embed_text(
            &doc,
            &c,
            PrefixConfig {
                tags: false,
                aliases: false,
                ..PrefixConfig::full()
            },
        );
        // The whole metadata line goes, not an empty separator.
        assert_eq!(
            neither,
            "Archdragon — lore/bestiary/archdragon.md\nDefinition\nbody"
        );
    }

    #[test]
    fn disabled_embeds_exactly_what_is_stored() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let c = chunk(&["Definition"], "## Definition\n**Rank**: SS");
        let cfg = PrefixConfig::default();
        assert!(
            !cfg.enabled,
            "the prefix is off unless config.toml turns it on"
        );
        assert_eq!(embed_text(&doc, &c, cfg), c.text);
    }

    #[test]
    fn a_chunk_before_the_first_heading_gets_no_heading_line() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let c = chunk(&[], "Opening paragraph.");
        assert_eq!(
            embed_text(&doc, &c, PrefixConfig::full()),
            "Archdragon — lore/bestiary/archdragon.md\n\
             aliases: Elder Wyrm | tags: dragon, apex\n\
             Opening paragraph."
        );
    }

    #[test]
    fn embed_inputs_prefixes_every_chunk_and_preserves_order() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = vec![
            chunk(&["Definition"], "first"),
            chunk(&["Human Forms"], "second"),
        ];

        let inputs = embed_inputs(
            &doc,
            &chunks,
            EmbedComposition {
                prefix: PrefixConfig::full(),
                title: DocumentTitle::None,
                root: BreadcrumbRoot::default(),
            },
        );
        assert_eq!(inputs.len(), 2);
        assert!(inputs[0].text.contains("Definition\nfirst"));
        assert!(inputs[1].text.contains("Human Forms\nsecond"));
        assert!(inputs.iter().all(|i| i.text.starts_with("Archdragon — ")));
    }

    // ── The title field (#36) ────────────────────────────────────────────────

    fn titles(doc: &DocContext, chunks: &[Chunk], cfg: DocumentTitle) -> Vec<String> {
        embed_inputs(
            doc,
            chunks,
            EmbedComposition {
                title: cfg,
                ..Default::default()
            },
        )
        .into_iter()
        .map(|i| i.title)
        .collect()
    }

    #[test]
    fn the_none_title_is_empty_which_the_template_spells_none() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [chunk(
            &["Abilities", "Combat"],
            "### Combat\nBreath weapon.",
        )];
        assert_eq!(
            titles(&doc, &chunks, DocumentTitle::None),
            vec![String::new()]
        );
    }

    /// The shipped default leaves the field empty, so the template writes the
    /// model card's `none`. The breadcrumb rule reaches the index through the
    /// keyword lane instead (#37, #38).
    #[test]
    fn the_default_title_is_empty() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [chunk(
            &["Abilities", "Combat"],
            "### Combat\nBreath weapon.",
        )];
        assert_eq!(
            titles(&doc, &chunks, DocumentTitle::default()),
            vec![String::new()]
        );
    }

    #[test]
    fn the_note_title_is_the_documents_effective_name() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [chunk(
            &["Abilities", "Combat"],
            "### Combat\nBreath weapon.",
        )];
        assert_eq!(
            titles(&doc, &chunks, DocumentTitle::Note),
            vec!["Archdragon".to_string()]
        );
    }

    #[test]
    fn the_breadcrumb_is_the_file_path_then_every_ancestor_heading() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [chunk(
            &["Abilities", "Combat"],
            "### Combat\nBreath weapon.",
        )];
        assert_eq!(
            titles(&doc, &chunks, DocumentTitle::Breadcrumb),
            vec!["lore/bestiary/archdragon.md > Abilities > Combat".to_string()]
        );
    }

    /// The mechanism that separates #36 from #2: the breadcrumb differs between
    /// the sections of one document, so it cannot move all of a file's vectors
    /// the same way. The note title alone can, and does.
    #[test]
    fn the_breadcrumb_varies_between_sections_of_one_document_and_the_file_path_does_not() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [
            chunk(&["Definition"], "first"),
            chunk(&["Human Forms"], "second"),
        ];

        let note = titles(&doc, &chunks, DocumentTitle::Note);
        assert_eq!(note[0], note[1]);

        let breadcrumb = titles(&doc, &chunks, DocumentTitle::Breadcrumb);
        assert_eq!(breadcrumb[0], "lore/bestiary/archdragon.md > Definition");
        assert_eq!(breadcrumb[1], "lore/bestiary/archdragon.md > Human Forms");
    }

    #[test]
    fn a_chunk_before_the_first_heading_has_the_file_path_as_its_whole_breadcrumb() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [chunk(&[], "Opening paragraph.")];
        assert_eq!(
            titles(&doc, &chunks, DocumentTitle::Breadcrumb),
            vec!["lore/bestiary/archdragon.md".to_string()]
        );
    }

    /// The title is a field, not a prefix: `chunks.text` and the FTS index keep
    /// the raw chunk, so it cannot leak into a displayed result or be matched as
    /// a keyword. Same contract as #2's prefix.
    #[test]
    fn the_title_field_does_not_touch_the_body() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let body = "## Definition\n**Rank**: SS";

        for cfg in [
            DocumentTitle::None,
            DocumentTitle::Note,
            DocumentTitle::Breadcrumb,
        ] {
            let chunks = [chunk(&["Definition"], body)];
            let inputs = embed_inputs(
                &doc,
                &chunks,
                EmbedComposition {
                    title: cfg,
                    ..Default::default()
                },
            );
            assert_eq!(inputs[0].text, body, "{cfg:?} changed the body");
        }
    }

    #[test]
    fn as_doc_borrows_both_fields() {
        let doc = DocContext::from_file("lore/bestiary/archdragon.md", ARCHDRAGON);
        let chunks = [chunk(&["Definition"], "body")];
        let inputs = embed_inputs(
            &doc,
            &chunks,
            EmbedComposition {
                title: DocumentTitle::Breadcrumb,
                ..Default::default()
            },
        );

        let as_doc = inputs[0].as_doc();
        assert_eq!(as_doc.title, "lore/bestiary/archdragon.md > Definition");
        assert_eq!(as_doc.text, "body");
    }
}
