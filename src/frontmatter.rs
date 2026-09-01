//! Editing a note's frontmatter without rewriting it.
//!
//! A vault holds its notes to a shape — a key order, a list style, a comment
//! that explains a field. knapper is a guest in that vault, so a write
//! changes the keys it is told to change and leaves every other byte alone.
//!
//! A [`Block`] is the frontmatter as an ordered list of items, each carrying
//! its own original text: one item per top-level key, and one per comment or
//! blank line between them. An operation rewrites one item; [`Block::render`]
//! concatenates them back between the note's own fences and appends the body
//! byte for byte. A block nothing edited therefore renders what it parsed
//! (#92).

use anyhow::{Result, bail};

/// How a list is written in the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListStyle {
    /// `tags: [a, b]`
    Inline,
    /// `tags:` with one `- a` line per item, indented by `indent` spaces.
    Block { indent: usize },
}

/// What an entry's value is, as far as an edit can address it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Scalar,
    List(ListStyle),
    /// A value no line edit can address. The text names what was found, so a
    /// refusal can say it.
    Opaque(&'static str),
}

/// One item of the block. Every byte between the fences belongs to exactly
/// one item, so concatenating them reproduces the block.
#[derive(Debug, Clone)]
enum Item {
    /// A top-level key and every line under it.
    Entry {
        key: String,
        value: Value,
        text: String,
    },
    /// A comment line at column 0, or a blank line between entries. Carried
    /// verbatim and never rewritten.
    Filler(String),
}

impl Item {
    fn text(&self) -> &str {
        match self {
            Item::Entry { text, .. } => text,
            Item::Filler(text) => text,
        }
    }
}

/// A note's frontmatter block, opened for editing.
#[derive(Debug, Clone)]
pub struct Block {
    open_fence: String,
    items: Vec<Item>,
    close_fence: String,
    /// What goes between the closing fence and the body. Empty for a parsed
    /// block, whose body already carries whatever followed the fence.
    separator: String,
    body: String,
    /// The line ending the note uses, for an item an edit constructs fresh.
    /// Unread until an edit operation exists to read it (Task 2).
    #[allow(dead_code)]
    newline: String,
}

impl Block {
    /// Open `text` for editing.
    ///
    /// `Ok(None)` means the note has no block, and a caller that must write a
    /// key opens one with [`Block::open`]. An error means a block is there
    /// that no line edit can address, and the caller must not write.
    pub fn parse(text: &str) -> Result<Option<Block>> {
        let Some((open, inner, close, body)) = split_fences(text)? else {
            return Ok(None);
        };
        let items = parse_items(inner)?;
        let mut seen: Vec<&str> = Vec::new();
        for item in &items {
            if let Item::Entry { key, .. } = item {
                if seen.contains(&key.as_str()) {
                    bail!(
                        "frontmatter holds `{key}` twice, and knapper cannot tell which one the note means"
                    );
                }
                seen.push(key);
            }
        }
        Ok(Some(Block {
            open_fence: open.to_string(),
            items,
            close_fence: close.to_string(),
            separator: String::new(),
            body: body.to_string(),
            newline: newline_of(text).to_string(),
        }))
    }

    /// The block's text, spliced back onto the note's own body.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.body.len() + 128);
        out.push_str(&self.open_fence);
        for item in &self.items {
            out.push_str(item.text());
        }
        out.push_str(&self.close_fence);
        out.push_str(&self.separator);
        out.push_str(&self.body);
        out
    }

    /// What `key`'s value is, or `None` when the block does not hold it.
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.items.iter().find_map(|i| match i {
            Item::Entry { key: k, value, .. } if k == key => Some(value),
            _ => None,
        })
    }

    /// True when the block holds no key. A block of comments alone is empty.
    pub fn is_empty(&self) -> bool {
        !self.items.iter().any(|i| matches!(i, Item::Entry { .. }))
    }

    /// Everything after the block, byte for byte.
    pub fn body(&self) -> &str {
        &self.body
    }
}

// ── Parsing ──────────────────────────────────────────────────────

/// The line ending a text uses: CRLF when its first break is one.
fn newline_of(text: &str) -> &'static str {
    match text.find('\n') {
        Some(i) if i > 0 && text.as_bytes()[i - 1] == b'\r' => "\r\n",
        _ => "\n",
    }
}

/// Split `text` into lines, each keeping its own line ending.
fn lines_with_endings(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            out.push(&text[start..=i]);
            start = i + 1;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn strip_ending(line: &str) -> &str {
    line.trim_end_matches(['\n', '\r'])
}

fn is_blank(line: &str) -> bool {
    strip_ending(line).trim().is_empty()
}

fn is_comment(line: &str) -> bool {
    line.starts_with('#')
}

/// The top-level key a line opens, if it opens one: no leading space, no
/// quote, and a `:` that ends the line or is followed by a space.
fn key_of(line: &str) -> Option<String> {
    let trimmed = strip_ending(line);
    if trimmed.is_empty() || trimmed.starts_with([' ', '\t', '-', '#', '"', '\'']) {
        return None;
    }
    let colon = trimmed.find(':')?;
    let after = &trimmed[colon + 1..];
    if after.is_empty() || after.starts_with(' ') {
        Some(trimmed[..colon].trim_end().to_string())
    } else {
        None
    }
}

/// The block's four parts: the opening fence line, the text between the
/// fences, the closing fence line, and the body. `Ok(None)` when the text
/// does not open with a fence.
#[allow(clippy::type_complexity)]
fn split_fences(text: &str) -> Result<Option<(&str, &str, &str, &str)>> {
    let lines = lines_with_endings(text);
    let Some(first) = lines.first() else {
        return Ok(None);
    };
    if strip_ending(first) != "---" {
        return Ok(None);
    }
    let mut offset = first.len();
    for line in &lines[1..] {
        if strip_ending(line) == "---" {
            return Ok(Some((
                first,
                &text[first.len()..offset],
                &text[offset..offset + line.len()],
                &text[offset + line.len()..],
            )));
        }
        offset += line.len();
    }
    bail!("frontmatter opens with `---` and never closes")
}

/// Cut the text between the fences into items. An entry runs from its key
/// line to the line before the next key line or column-0 comment; a run of
/// blank lines at its tail belongs between the entries instead, so a blank
/// line inside a list stays with the list and one between keys does not.
fn parse_items(inner: &str) -> Result<Vec<Item>> {
    let lines = lines_with_endings(inner);
    let mut items = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if is_blank(lines[i]) || is_comment(lines[i]) {
            items.push(Item::Filler(lines[i].to_string()));
            i += 1;
            continue;
        }
        let Some(key) = key_of(lines[i]) else {
            bail!(
                "frontmatter is not a mapping knapper can edit: `{}`",
                strip_ending(lines[i]).trim()
            );
        };
        let start = i;
        i += 1;
        while i < lines.len() && key_of(lines[i]).is_none() && !is_comment(lines[i]) {
            i += 1;
        }
        let mut end = i;
        while end > start + 1 && is_blank(lines[end - 1]) {
            end -= 1;
        }
        let text: String = lines[start..end].concat();
        let value = classify(&text);
        items.push(Item::Entry { key, value, text });
        for line in &lines[end..i] {
            items.push(Item::Filler(line.to_string()));
        }
    }
    Ok(items)
}

/// The text after the key's colon on the entry's first line.
fn value_text(text: &str) -> &str {
    let first = text.split_inclusive('\n').next().unwrap_or(text);
    match first.find(':') {
        Some(i) => strip_ending(&first[i + 1..]),
        None => "",
    }
}

fn is_scalar(v: &serde_yaml::Value) -> bool {
    matches!(
        v,
        serde_yaml::Value::String(_)
            | serde_yaml::Value::Number(_)
            | serde_yaml::Value::Bool(_)
            | serde_yaml::Value::Null
    )
}

/// The indent of the entry's first `- ` line.
fn first_item_indent(text: &str) -> usize {
    text.lines()
        .find(|l| l.trim_start().starts_with("- "))
        .map(|l| l.len() - l.trim_start().len())
        .unwrap_or(2)
}

/// What an edit could do to this entry. The YAML parse decides scalar from
/// list from anything else; the raw text decides the style the parse throws
/// away.
fn classify(text: &str) -> Value {
    let head = value_text(text).trim_start();
    if head.starts_with('&') || head.starts_with('*') {
        return Value::Opaque("an anchor or alias");
    }
    if head.starts_with('|') || head.starts_with('>') {
        return Value::Opaque("a block scalar");
    }
    let Ok(serde_yaml::Value::Mapping(map)) = serde_yaml::from_str::<serde_yaml::Value>(text)
    else {
        return Value::Opaque("a value that is not one mapping entry");
    };
    let Some(v) = map.values().next() else {
        return Value::Opaque("a value that is not one mapping entry");
    };
    match v {
        serde_yaml::Value::Sequence(items) if items.iter().all(is_scalar) => {
            if head.starts_with('[') {
                Value::List(ListStyle::Inline)
            } else {
                Value::List(ListStyle::Block {
                    indent: first_item_indent(text),
                })
            }
        }
        serde_yaml::Value::Sequence(_) => Value::Opaque("a list of lists or mappings"),
        serde_yaml::Value::Mapping(_) => Value::Opaque("a nested mapping"),
        serde_yaml::Value::Tagged(_) => Value::Opaque("a tagged value"),
        _ => Value::Scalar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole promise, as one assertion: a block that nothing edited
    /// renders back as the bytes it was parsed from.
    fn assert_identity(text: &str) {
        let block = Block::parse(text).unwrap().expect("a block");
        assert_eq!(block.render(), text);
    }

    #[test]
    fn an_unedited_block_renders_the_bytes_it_was_parsed_from() {
        assert_identity("---\nname: Probe\naliases: []\ntags: [type/lore]\n---\n\nBody.\n");
        assert_identity("---\ntags:\n  - a\n  - b\nname: Probe\n---\n\nBody.\n");
        assert_identity("---\n# why this note exists\nname: Probe\n\ntags: [a]\n---\nBody.\n");
        assert_identity("---\r\nname: Probe\r\ntags: [a]\r\n---\r\n\r\nBody.\r\n");
        assert_identity("---\nname: Probe\n---\n\nBody with no final break");
        assert_identity("---\n---\n\nAn empty block.\n");
    }

    #[test]
    fn a_note_with_no_block_parses_to_no_block() {
        assert!(Block::parse("# Title\n\nBody.\n").unwrap().is_none());
        assert!(Block::parse("").unwrap().is_none());
    }

    #[test]
    fn a_block_that_never_closes_is_an_error() {
        let err = Block::parse("---\nname: Probe\n\nBody.\n").unwrap_err();
        assert!(err.to_string().contains("never closes"), "{err}");
    }

    #[test]
    fn a_block_holding_one_key_twice_is_an_error() {
        let err = Block::parse("---\ntags: [a]\nname: X\ntags: [b]\n---\n").unwrap_err();
        assert!(err.to_string().contains("twice"), "{err}");
    }

    #[test]
    fn a_block_that_is_not_a_mapping_is_an_error() {
        let err = Block::parse("---\n- one\n- two\n---\n").unwrap_err();
        assert!(err.to_string().contains("not a mapping"), "{err}");
    }

    #[test]
    fn a_value_is_classified_by_what_an_edit_could_address() {
        let text = concat!(
            "---\n",
            "name: Probe\n",
            "inline: [a, b]\n",
            "blocked:\n    - a\n    - b\n",
            "nested:\n  inner: 1\n",
            "folded: |\n  a line\n",
            "---\n",
        );
        let block = Block::parse(text).unwrap().unwrap();
        assert_eq!(block.value("name"), Some(&Value::Scalar));
        assert_eq!(block.value("inline"), Some(&Value::List(ListStyle::Inline)));
        assert_eq!(
            block.value("blocked"),
            Some(&Value::List(ListStyle::Block { indent: 4 }))
        );
        assert_eq!(
            block.value("nested"),
            Some(&Value::Opaque("a nested mapping"))
        );
        assert_eq!(
            block.value("folded"),
            Some(&Value::Opaque("a block scalar"))
        );
        assert_eq!(block.value("absent"), None);
    }

    #[test]
    fn a_blank_line_inside_a_list_belongs_to_the_list() {
        // The blank line between the items is part of the `tags` entry, and
        // the one before `name` is not: identity proves both, because a
        // mis-split would still round trip, so the edit tests in Task 2 are
        // what pin it. Here we assert only that the parse accepts it.
        let text = "---\ntags:\n  - a\n\n  - b\n\nname: Probe\n---\n";
        assert_identity(text);
        let block = Block::parse(text).unwrap().unwrap();
        assert!(matches!(block.value("tags"), Some(Value::List(_))));
    }

    #[test]
    fn an_empty_block_and_a_block_of_only_comments_are_empty() {
        assert!(Block::parse("---\n---\n").unwrap().unwrap().is_empty());
        assert!(
            Block::parse("---\n# just a note\n---\n")
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert!(
            !Block::parse("---\nname: X\n---\n")
                .unwrap()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn the_body_is_everything_after_the_closing_fence() {
        let block = Block::parse("---\nname: X\n---\n\nBody.\n")
            .unwrap()
            .unwrap();
        assert_eq!(block.body(), "\nBody.\n");
    }
}
