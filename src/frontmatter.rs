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

use crate::markdown::{lines_with_endings, newline_of};
use anyhow::{Result, bail};

/// How a list is written in the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListStyle {
    /// `tags: [a, b]`
    Inline,
    /// `tags:` with one `- a` line per item, indented by `indent` spaces.
    Block { indent: usize },
}

/// Which quote character a fresh value takes when it needs quoting. YAML
/// has two, and they spell the same string: `'single'` escapes only its own
/// quote, by doubling it, and `"double"` escapes with a backslash. A vault
/// writes one of them — Obsidian's own property writer emits `"` — and a
/// value re-serialised in the other one changes a line that holds the same
/// value, which is diff churn in a tracked vault (#112).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuoteStyle {
    Single,
    Double,
}

/// Where a key the block does not already hold is written. A key it does
/// hold has a place already — the file's — so a placement says where to
/// *put* a key and never where to move one, and re-running an edit changes
/// nothing (#113).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyPlacement {
    /// After the last key, which is what a caller naming nothing gets.
    End,
    /// Directly below the key named, and above the comment that introduces
    /// whatever follows it.
    After(String),
    /// Directly above the key named, and above the comment that introduces
    /// *it* — a comment on its own line belongs to the key below it. It is
    /// the only way to name the top of the block.
    Before(String),
}

impl KeyPlacement {
    /// The placement two optional anchors mean. Both name a place, and
    /// picking one of the two would be guessing which the caller meant.
    pub fn new(after: Option<&str>, before: Option<&str>) -> Result<KeyPlacement> {
        match (after, before) {
            (Some(_), Some(_)) => {
                bail!("a placement names one of `after` or `before`, not both")
            }
            (Some(key), None) => Ok(KeyPlacement::After(key.to_string())),
            (None, Some(key)) => Ok(KeyPlacement::Before(key.to_string())),
            (None, None) => Ok(KeyPlacement::End),
        }
    }
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

/// One item of a list, read out of an entry for a list edit. `value` is what
/// an edit's `item` argument is matched against; `source` is the item's own
/// text, quotes and all, for an item the block already held. An item a write
/// supplies fresh has no source of its own and renders through `yaml_scalar`
/// instead, so an edit changes only the items it names.
struct ListItem {
    value: String,
    source: Option<String>,
}

impl ListItem {
    /// An item a write supplies fresh.
    fn fresh(value: String) -> Self {
        ListItem {
            value,
            source: None,
        }
    }

    /// An item read from the block, keeping its own source text.
    fn existing(value: String, source: String) -> Self {
        ListItem {
            value,
            source: Some(source),
        }
    }

    /// The item's rendered text: its own source when it has one, else fresh
    /// through `yaml_scalar` in the block's own quote style.
    fn render(&self, quotes: QuoteStyle) -> String {
        match &self.source {
            Some(source) => source.clone(),
            None => yaml_scalar(&self.value, quotes),
        }
    }

    /// The item's rendered text as it will sit inside `[...]`. An item's own
    /// source can carry a trailing `# comment` that is harmless at the end
    /// of its original scalar line, but the same text placed before a `,`
    /// inside brackets makes the comment swallow the rest of the sequence —
    /// `render_list`'s block-style arm never hits this, because there a
    /// comment sits at the end of its own line either way. A source (or a
    /// fresh value) can also hold an unquoted comma: nothing above a flow
    /// sequence's own commas tells its items apart, so a comma with no
    /// quoting around it does not fail to parse, it silently becomes one
    /// more item. The check below must therefore confirm the text reparses
    /// to *exactly one* item equal to this one's own value, not merely that
    /// it parses at all — and the fallback must itself be safe to sit
    /// inside `[...]`, because a fallback that renders the value the same
    /// way the failed check just read it reproduces the same corruption
    /// (#92, C2; R1 regression).
    fn render_in_flow(&self, quotes: QuoteStyle) -> String {
        let text = self.render(quotes);
        if reparses_to_one_flow_item(&text, &self.value) {
            text
        } else {
            flow_safe_scalar(&self.value, quotes)
        }
    }
}

/// A note's frontmatter block, opened for editing.
#[derive(Debug, Clone)]
pub struct Block {
    open_fence: String,
    items: Vec<Item>,
    close_fence: String,
    /// What goes between the closing fence and the body: the blank line
    /// that separates them, when the text after the fence opens with one,
    /// and empty otherwise. A parsed block and an opened one agree on this,
    /// so `body()` means the same thing for both.
    separator: String,
    body: String,
    /// The line ending the note uses, for an item an edit constructs fresh.
    newline: String,
}

impl Block {
    /// Open `text` for editing.
    ///
    /// `Ok(None)` means the note has no block, and a caller that must write a
    /// key opens one with [`Block::open`]. An error means a block is there
    /// that no line edit can address, and the caller must not write.
    pub fn parse(text: &str) -> Result<Option<Block>> {
        let Some((open, inner, close, after)) = split_fences(text)? else {
            return Ok(None);
        };
        let (separator, body) = split_separator(after);
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
            separator: separator.to_string(),
            body: body.to_string(),
            newline: newline_of(text).to_string(),
        }))
    }

    /// A new, empty block above `text`. The separator is the blank line
    /// knapper puts between a block and a body, and there is none when the
    /// note is empty.
    pub fn open(text: &str) -> Block {
        let newline = newline_of(text);
        Block {
            open_fence: format!("---{newline}"),
            items: Vec::new(),
            close_fence: format!("---{newline}"),
            separator: if text.is_empty() {
                String::new()
            } else {
                newline.to_string()
            },
            body: text.to_string(),
            newline: newline.to_string(),
        }
    }

    /// Open `text` for editing, creating a block when the note has none. The
    /// refusals of [`Block::parse`] still apply: a block that is there and
    /// cannot be edited is an error, not a second block.
    pub fn parse_or_open(text: &str) -> Result<Block> {
        Ok(Block::parse(text)?.unwrap_or_else(|| Block::open(text)))
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

    /// True when the block holds nothing at all — no key, and no comment or
    /// blank line either. A block of comments alone is [`Block::is_empty`]
    /// (it counts keys only), but it is not blank: a caller that drops the
    /// block whenever it is merely empty of keys throws the comments away
    /// with it (#92, I2).
    pub fn is_blank(&self) -> bool {
        self.items.is_empty()
    }

    /// The note's content below the block, past the blank line that
    /// separates them, byte for byte.
    pub fn body(&self) -> &str {
        &self.body
    }

    /// Replace the note's content below the block. The block itself — its
    /// fences, its items, the separator between it and the body — is
    /// untouched, so a body edit routed through `render()` cannot move or
    /// re-render a single frontmatter byte (#92, I5).
    pub fn set_body(&mut self, body: String) {
        self.body = body;
    }

    /// The value `key` holds, when it holds a scalar.
    pub fn scalar(&self, key: &str) -> Option<String> {
        let idx = self.find(key)?;
        match self.items[idx] {
            Item::Entry {
                value: Value::Scalar,
                ..
            } => Some(scalar_of(self.items[idx].text())),
            _ => None,
        }
    }

    /// The items `key`'s list holds, in order. A scalar reads back as one
    /// item, and a key the block does not hold reads back as none.
    pub fn list(&self, key: &str) -> Vec<String> {
        self.items_of(key).into_iter().map(|i| i.value).collect()
    }

    /// Write `key` as a scalar, quoted when YAML needs it, in the quote
    /// character the key already used.
    pub fn set_scalar(&mut self, key: &str, value: &str, at: &KeyPlacement) -> Result<()> {
        self.check_editable(key)?;
        let quotes = self.quote_style_for(key);
        let text = format!("{key}: {}{}", yaml_scalar(value, quotes), self.newline);
        self.put(key, text, Value::Scalar, at)
    }

    /// Write `key` as a bare `true` or `false`, which a quoted scalar would
    /// not be.
    pub fn set_bool(&mut self, key: &str, value: bool) -> Result<()> {
        self.check_editable(key)?;
        let text = format!("{key}: {value}{}", self.newline);
        self.put(key, text, Value::Scalar, &KeyPlacement::End)
    }

    /// Write `key` as a list. Every item comes from the caller, so every item
    /// is serialised fresh through `yaml_scalar`. An empty list is written
    /// `[]` in any style, because block style with no items reads back as
    /// null.
    pub fn set_list(&mut self, key: &str, items: &[String], at: &KeyPlacement) -> Result<()> {
        let style = self.list_style_for(key)?;
        let quotes = self.quote_style_for(key);
        let items: Vec<ListItem> = items.iter().cloned().map(ListItem::fresh).collect();
        let text = self.render_list(key, &items, style, quotes);
        self.put(key, text, Value::List(style), at)
    }

    /// Add `item` to `key`'s list, creating the list when the key is absent
    /// and promoting it when the key holds a scalar. An item the list already
    /// holds changes nothing. Every existing item keeps its own source text;
    /// only the added item is serialised through `yaml_scalar`.
    pub fn add_to_list(&mut self, key: &str, item: &str, at: &KeyPlacement) -> Result<()> {
        let style = self.list_style_for(key)?;
        let quotes = self.quote_style_for(key);
        let mut items = self.items_of(key);
        if items.iter().any(|i| i.value == item) {
            return Ok(());
        }
        items.push(ListItem::fresh(item.to_string()));
        let text = self.render_list(key, &items, style, quotes);
        self.put(key, text, Value::List(style), at)
    }

    /// Remove `item` from `key`'s list. A scalar equal to `item` removes the
    /// key; an absent key changes nothing. Every surviving item keeps its own
    /// source text.
    pub fn remove_from_list(&mut self, key: &str, item: &str) -> Result<()> {
        let Some(idx) = self.find(key) else {
            return Ok(());
        };
        self.check_editable_at(key, idx)?;
        if matches!(
            self.items[idx],
            Item::Entry {
                value: Value::Scalar,
                ..
            }
        ) {
            if scalar_of(self.items[idx].text()) == item {
                self.items.remove(idx);
            }
            return Ok(());
        }
        let style = self.list_style_for(key)?;
        let quotes = self.quote_style_for(key);
        let mut items = self.items_of(key);
        items.retain(|i| i.value != item);
        let text = self.render_list(key, &items, style, quotes);
        self.put(key, text, Value::List(style), &KeyPlacement::End)
    }

    /// Remove `key` and the lines it owns. An absent key changes nothing.
    pub fn remove(&mut self, key: &str) -> Result<()> {
        if let Some(idx) = self.find(key) {
            self.items.remove(idx);
        }
        Ok(())
    }

    // ── internals ────────────────────────────────────────────────

    fn find(&self, key: &str) -> Option<usize> {
        self.items
            .iter()
            .position(|i| matches!(i, Item::Entry { key: k, .. } if k == key))
    }

    /// Fail when `key` holds a value no line edit can address.
    fn check_editable(&self, key: &str) -> Result<()> {
        match self.find(key) {
            Some(idx) => self.check_editable_at(key, idx),
            None => Ok(()),
        }
    }

    /// `check_editable`, given the item's index instead of scanning for it.
    fn check_editable_at(&self, key: &str, idx: usize) -> Result<()> {
        if let Item::Entry {
            value: Value::Opaque(found),
            ..
        } = self.items[idx]
        {
            bail!(
                "cannot edit `{key}`: its value is {found}, and knapper edits a scalar or a flat list"
            );
        }
        Ok(())
    }

    /// The style a list write to `key` takes: the key's own when it is a
    /// list, else the style the block already uses.
    fn list_style_for(&self, key: &str) -> Result<ListStyle> {
        self.check_editable(key)?;
        match self.find(key).map(|idx| &self.items[idx]) {
            Some(Item::Entry {
                value: Value::List(style),
                ..
            }) => Ok(*style),
            _ => Ok(self.new_list_style()),
        }
    }

    /// The style a key knapper adds takes: the first list already in the
    /// block, or block style at two spaces when the block holds none.
    fn new_list_style(&self) -> ListStyle {
        self.items
            .iter()
            .find_map(|i| match i {
                Item::Entry {
                    value: Value::List(style),
                    ..
                } => Some(*style),
                _ => None,
            })
            .unwrap_or(ListStyle::Block { indent: 2 })
    }

    /// The quote character a fresh value written under `key` takes: the one
    /// `key`'s own values already use, else the first one the block uses
    /// anywhere, else `"` — which is what Obsidian's own property writer
    /// emits, and so what a vault this tool is a guest in most often holds.
    /// It is the ladder `list_style_for` walks for inline-versus-block, for
    /// the same reason: a write matches what the file already says (#112).
    fn quote_style_for(&self, key: &str) -> QuoteStyle {
        quote_style_of(&self.items_of(key)).unwrap_or_else(|| self.new_quote_style())
    }

    /// The first quote character the block uses, reading its entries in
    /// order. `None` from every one of them means the block quotes nothing,
    /// and the default answers.
    fn new_quote_style(&self) -> QuoteStyle {
        self.items
            .iter()
            .filter_map(|i| match i {
                Item::Entry { key, .. } => quote_style_of(&self.items_of(key)),
                Item::Filler(_) => None,
            })
            .next()
            .unwrap_or(QuoteStyle::Double)
    }

    /// The items `key` holds: its list's, or its scalar as one item, or none.
    /// Each keeps the source text that reproduces it.
    fn items_of(&self, key: &str) -> Vec<ListItem> {
        let Some(idx) = self.find(key) else {
            return Vec::new();
        };
        match self.items[idx] {
            Item::Entry {
                value: Value::List(style),
                ..
            } => list_items(self.items[idx].text(), style),
            Item::Entry {
                value: Value::Scalar,
                ..
            } => {
                let text = self.items[idx].text();
                let value = scalar_of(text);
                let source = scalar_source(text);
                // A bare `tags:` with nothing after the colon classifies as
                // a null scalar: both its value and its own source text are
                // empty. Reading that back as one list item wrote a phantom
                // `- ` blank entry ahead of whatever a list write added — a
                // bare key with no value is a common Obsidian template
                // shape, so this hit `create --tags` on a templated note
                // (#92, I3).
                if value.is_empty() && source.is_empty() {
                    Vec::new()
                } else if text.lines().count() > 1 {
                    // `scalar_source` reads only the entry's first line, so
                    // a multi-line plain scalar — `name: some\n  continued`,
                    // whose value is `some continued` — would promote to a
                    // list item holding only `some` if it kept that source.
                    // Falling back to a fresh serialisation of the parsed
                    // value loses the folding style but keeps every word
                    // (#92, I4).
                    vec![ListItem::fresh(value)]
                } else {
                    vec![ListItem::existing(value, source)]
                }
            }
            _ => Vec::new(),
        }
    }

    fn render_list(
        &self,
        key: &str,
        items: &[ListItem],
        style: ListStyle,
        quotes: QuoteStyle,
    ) -> String {
        if items.is_empty() {
            return format!("{key}: []{}", self.newline);
        }
        match style {
            ListStyle::Inline => {
                let body: Vec<String> = items.iter().map(|i| i.render_in_flow(quotes)).collect();
                format!("{key}: [{}]{}", body.join(", "), self.newline)
            }
            ListStyle::Block { indent } => {
                let pad = " ".repeat(indent);
                let mut out = format!("{key}:{}", self.newline);
                for item in items {
                    out.push_str(&pad);
                    out.push_str("- ");
                    out.push_str(&item.render(quotes));
                    out.push_str(&self.newline);
                }
                out
            }
        }
    }

    fn put(&mut self, key: &str, text: String, value: Value, at: &KeyPlacement) -> Result<()> {
        let entry = Item::Entry {
            key: key.to_string(),
            value,
            text,
        };
        match self.find(key) {
            Some(idx) => self.items[idx] = entry,
            None => {
                let idx = self.insert_index(key, at)?;
                self.items.insert(idx, entry);
            }
        }
        Ok(())
    }

    /// Where a new key's item goes. An anchor the block does not hold is
    /// refused rather than appended: appending anyway is the silence #113
    /// reports, and the caller named a key it believed was there.
    fn insert_index(&self, key: &str, at: &KeyPlacement) -> Result<usize> {
        let (anchor, before) = match at {
            KeyPlacement::End => return Ok(self.items.len()),
            KeyPlacement::After(anchor) => (anchor, false),
            KeyPlacement::Before(anchor) => (anchor, true),
        };
        let Some(idx) = self.find(anchor) else {
            bail!(
                "no property '{anchor}' in the frontmatter to place '{key}' {}",
                if before { "before" } else { "after" }
            );
        };
        if !before {
            // Immediately below the anchor, which leaves a comment on the
            // next line with the key it introduces.
            return Ok(idx + 1);
        }
        // Above the anchor, stepping back over the comment lines that
        // introduce it. A blank line separates groups rather than
        // introducing a key, so it is where the walk stops.
        let mut at = idx;
        while at > 0 {
            match &self.items[at - 1] {
                Item::Filler(line) if is_comment(line) => at -= 1,
                _ => break,
            }
        }
        Ok(at)
    }
}

// ── Parsing ──────────────────────────────────────────────────────

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
    // `crate::markdown::split_frontmatter` — the reader every other module
    // uses — compares `line.trim() == "---"`, so `--- ` with trailing
    // whitespace already is frontmatter to it. Comparing exactly here made
    // the same line invisible to this module, so `parse_or_open` opened a
    // second block above what every reader already treated as the note's
    // real one (#92, I1).
    if strip_ending(first).trim_end() != "---" {
        return Ok(None);
    }
    let mut offset = first.len();
    for line in &lines[1..] {
        if strip_ending(line).trim_end() == "---" {
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

/// Split the text after a closing fence into the blank line that separates
/// it from the body, when there is one, and the body itself. Claims at most
/// one line ending, so a second blank line stays part of the body — this is
/// [`Block::open`]'s own split, run in reverse over what [`split_fences`]
/// returned, so a parsed block and an opened one agree on what `body()`
/// means (#92).
fn split_separator(after: &str) -> (&str, &str) {
    if let Some(rest) = after.strip_prefix("\r\n") {
        (&after[..2], rest)
    } else if let Some(rest) = after.strip_prefix('\n') {
        (&after[..1], rest)
    } else {
        ("", after)
    }
}

/// Split `text` into its frontmatter block's own bytes — the opening fence
/// through the separator that follows the closing one, unparsed — and the
/// body below it. `Ok(None)` means the note has no block.
///
/// This finds the block's span the way [`Block::parse`] does, but does not
/// call [`parse_items`], so it does not need the block's entries to parse:
/// a non-mapping block or one holding a duplicate key still has a byte span
/// this can find, because both are refusals `parse_items` raises, not ones
/// [`split_fences`] does. A body edit never reads or writes a frontmatter
/// byte, so it does not need to itemize the block to leave it alone — only
/// to know where it ends, which is what [`crate::writer::apply_body_edit`]
/// uses this for (#92, R2).
///
/// An error means even the span is unknowable: an opening `---` with no
/// closing one, [`split_fences`]'s own refusal, which no caller can work
/// around because there is no boundary between block and body to trust.
pub fn split_body(text: &str) -> Result<Option<(String, String)>> {
    let Some((open, inner, close, after)) = split_fences(text)? else {
        return Ok(None);
    };
    let (separator, body) = split_separator(after);
    let mut block = String::with_capacity(open.len() + inner.len() + close.len() + separator.len());
    block.push_str(open);
    block.push_str(inner);
    block.push_str(close);
    block.push_str(separator);
    Ok(Some((block, body.to_string())))
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
    // A line the entry's own span folded in that `key_of` did not recognise
    // as a key of its own — a quoted `"other key": v`, say — still parses as
    // a second entry in this text's mapping. Taking `.next()` without this
    // check would silently classify by the *first* entry's shape and let an
    // edit overwrite the whole span, discarding the second key with no
    // error (#92).
    if map.len() != 1 {
        return Value::Opaque("a value that is not one mapping entry");
    }
    let Some(v) = map.values().next() else {
        return Value::Opaque("a value that is not one mapping entry");
    };
    match v {
        serde_yaml::Value::Sequence(items) if items.iter().all(is_scalar) => {
            if head.starts_with('[') {
                match inline_list_addressable(text) {
                    Ok(()) => Value::List(ListStyle::Inline),
                    Err(found) => Value::Opaque(found),
                }
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

/// Whether a flow list's raw text is one `key: [items]` line with nothing
/// after the `]` that closes it. `render_list` writes exactly that shape and
/// has nowhere to put a second line or trailing text, so a flow list that
/// does not already look like that is refused rather than silently
/// collapsed when it is edited.
fn inline_list_addressable(text: &str) -> Result<(), &'static str> {
    if text.lines().count() > 1 {
        return Err("a flow list that spans more than one line");
    }
    let head = value_text(text).trim_start();
    match closing_bracket(head) {
        Some(end) if head[end + 1..].trim().is_empty() => Ok(()),
        _ => Err("a flow list with text after its closing `]`"),
    }
}

/// The index in `head` of the `]` that closes the `[` at its start, skipping
/// a bracket inside a quoted scalar. `None` when there is none. `head` is
/// assumed to start with `[`.
fn closing_bracket(head: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in head.char_indices().skip(1) {
        if escaped {
            escaped = false;
            continue;
        }
        match quote {
            Some('"') if c == '\\' => escaped = true,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == ']' => return Some(i),
            None => {}
        }
    }
    None
}

/// One scalar as YAML, quoted when it has to be, in `style` when there is a
/// choice. serde_yaml decides *whether* a value needs quoting, which is a
/// correctness question and stays its; it also always reaches for `'` first,
/// which is the part `style` overrides. Two answers are not a choice and are
/// kept whatever `style` says: a value that needs no quotes at all, and a
/// value serde_yaml did not single-quote, which means single quoting cannot
/// hold it — they escape nothing, so a line break or a tab has no
/// single-quoted spelling on one line.
fn yaml_scalar(value: &str, style: QuoteStyle) -> String {
    let yaml = serde_yaml::to_string(&serde_yaml::Value::String(value.to_string()))
        .unwrap_or_else(|_| value.to_string())
        .trim_end()
        .to_string();
    if yaml == value || (style == QuoteStyle::Single && yaml.starts_with('\'')) {
        return yaml;
    }
    double_quoted(value)
}

/// One scalar as a YAML double-quoted scalar. JSON's string syntax is one —
/// YAML 1.2 is a superset of JSON, and every escape `serde_json` emits
/// (`\"`, `\\`, `\n`, `\t`, `\r`, `\b`, `\f`, `\uXXXX`) is one YAML defines —
/// so the escaping is the serializer's job rather than this module's, and
/// there is no character it cannot hold.
fn double_quoted(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("\"{value}\""))
}

/// The quote character `items` are written in: the first one that carries a
/// source of its own opening with a quote. Items a write supplied fresh have
/// no source and say nothing, and neither does an unquoted one.
fn quote_style_of(items: &[ListItem]) -> Option<QuoteStyle> {
    items
        .iter()
        .find_map(|i| match i.source.as_deref()?.chars().next()? {
            '\'' => Some(QuoteStyle::Single),
            '"' => Some(QuoteStyle::Double),
            _ => None,
        })
}

/// Whether `text`, placed inside `[...]`, reparses to exactly one item equal
/// to `value`. Parsing at all is not enough: a comma with no quoting around
/// it is legal wherever it sits, so text holding one can parse cleanly into
/// *two* items instead of failing (#92, R1 regression).
fn reparses_to_one_flow_item(text: &str, value: &str) -> bool {
    let Ok(serde_yaml::Value::Mapping(map)) =
        serde_yaml::from_str::<serde_yaml::Value>(&format!("k: [{text}]"))
    else {
        return false;
    };
    matches!(
        map.values().next(),
        Some(serde_yaml::Value::Sequence(items))
            if items.len() == 1 && items.first().map(scalar_string).as_deref() == Some(value)
    )
}

/// `value` as YAML, safe to place inside `[...]`, in `style` where it can
/// be. `yaml_scalar` decides quoting for a value read alone at the top
/// level, where a comma is nothing special — it is what a flow item's own
/// source already went through, and what an unquoted item freshly added to
/// an existing inline list is rendered through too, so it is exactly the
/// rendering [`reparses_to_one_flow_item`] just found unsafe. Quoting
/// closes a flow item against every character flow syntax gives meaning to
/// — `,`, `#`, `:`, `[`, `]`, `{`, `}` — so the fallback is the block's own
/// quote character where that spells the value, and `"` where it does not:
/// single quoting escapes nothing, so a value carrying a line break has no
/// single-quoted spelling that reads back as itself, and writing one
/// silently folded the break into a space.
fn flow_safe_scalar(value: &str, style: QuoteStyle) -> String {
    let plain = yaml_scalar(value, style);
    if reparses_to_one_flow_item(&plain, value) {
        return plain;
    }
    if style == QuoteStyle::Single {
        let single = format!("'{}'", value.replace('\'', "''"));
        if reparses_to_one_flow_item(&single, value) {
            return single;
        }
    }
    double_quoted(value)
}

fn scalar_string(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

/// The scalar a one-entry mapping's value holds.
fn scalar_of(text: &str) -> String {
    match serde_yaml::from_str::<serde_yaml::Value>(text) {
        Ok(serde_yaml::Value::Mapping(map)) => {
            map.values().next().map(scalar_string).unwrap_or_default()
        }
        _ => String::new(),
    }
}

/// A scalar entry's own source text: what follows `key: ` on its line,
/// quotes and all.
fn scalar_source(text: &str) -> String {
    value_text(text).trim().to_string()
}

/// The items a list entry holds, each keeping the source text that
/// reproduces it.
fn list_items(text: &str, style: ListStyle) -> Vec<ListItem> {
    let values = list_values(text);
    let sources = list_sources(text, style);
    // A shortfall here means the source scan found a different number of
    // items than the YAML parse did — some shape the scan does not handle,
    // slipping past whatever guards classify already applies. Zipping the
    // two short would silently drop whichever items ran out first, which is
    // data loss; falling back to fresh values re-serialises every item
    // through yaml_scalar instead, which only loses formatting.
    if values.len() != sources.len() {
        return values.into_iter().map(ListItem::fresh).collect();
    }
    values
        .into_iter()
        .zip(sources)
        .map(|(value, source)| ListItem::existing(value, source))
        .collect()
}

/// The items' parsed values, in order.
fn list_values(text: &str) -> Vec<String> {
    match serde_yaml::from_str::<serde_yaml::Value>(text) {
        Ok(serde_yaml::Value::Mapping(map)) => match map.values().next() {
            Some(serde_yaml::Value::Sequence(items)) => items.iter().map(scalar_string).collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// The items' own source text, in order: the bracket contents split on
/// commas outside quotes for inline style, or the text after `- ` on each
/// item line, line ending stripped, for block style.
fn list_sources(text: &str, style: ListStyle) -> Vec<String> {
    match style {
        ListStyle::Inline => {
            let head = value_text(text).trim();
            let inner = if head.starts_with('[') {
                match closing_bracket(head) {
                    Some(end) => &head[1..end],
                    None => "",
                }
            } else {
                ""
            };
            split_flow_items(inner)
        }
        ListStyle::Block { .. } => text
            .lines()
            .filter_map(|l| l.trim_start().strip_prefix("- "))
            .map(str::to_string)
            .collect(),
    }
}

/// Split a flow sequence's inner text on commas that are not inside a `'` or
/// `"` quoted scalar, each piece trimmed of surrounding spaces. Inside a `"`
/// scalar a `\` escapes the next character, so an escaped quote does not
/// close it; a `'` scalar has no such escape and only the doubled-quote
/// convention closes and reopens it, which a plain toggle already handles.
/// `classify` already guarantees every item is a scalar, so this never has
/// to handle nesting.
fn split_flow_items(inner: &str) -> Vec<String> {
    if inner.trim().is_empty() {
        return Vec::new();
    }
    let mut items = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in inner.chars() {
        if escaped {
            current.push(c);
            escaped = false;
            continue;
        }
        match quote {
            Some('"') if c == '\\' => {
                current.push(c);
                escaped = true;
            }
            Some(q) if c == q => {
                quote = None;
                current.push(c);
            }
            Some(_) => current.push(c),
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                current.push(c);
            }
            None if c == ',' => {
                items.push(current.trim().to_string());
                current.clear();
            }
            None => current.push(c),
        }
    }
    items.push(current.trim().to_string());
    items
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
    fn the_body_is_what_follows_the_blank_line_after_the_block() {
        let block = Block::parse("---\nname: X\n---\n\nBody.\n")
            .unwrap()
            .unwrap();
        assert_eq!(block.body(), "Body.\n");
    }

    #[test]
    fn a_parsed_blocks_separator_and_body_recombine_to_what_followed_the_fence() {
        let with_blank_line = Block::parse("---\nname: X\n---\n\nBody.\n")
            .unwrap()
            .unwrap();
        assert_eq!(with_blank_line.separator, "\n");
        assert_eq!(with_blank_line.body, "Body.\n");

        let with_no_blank_line = Block::parse("---\nname: X\n---\nBody.\n").unwrap().unwrap();
        assert_eq!(with_no_blank_line.separator, "");
        assert_eq!(with_no_blank_line.body, "Body.\n");

        let crlf = Block::parse("---\r\nname: X\r\n---\r\n\r\nBody.\r\n")
            .unwrap()
            .unwrap();
        assert_eq!(crlf.separator, "\r\n");
        assert_eq!(crlf.body, "Body.\r\n");
    }

    /// Parse, apply `edit`, render.
    fn edited(text: &str, edit: impl FnOnce(&mut Block) -> Result<()>) -> String {
        let mut block = Block::parse(text).unwrap().expect("a block");
        edit(&mut block).unwrap();
        block.render()
    }

    #[test]
    fn an_edit_keeps_the_key_in_its_place_and_its_style() {
        let text = "---\nname: Probe\naliases: []\ntags: [type/lore, realm/rudd]\n---\n\nBody.\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "tags",
                &["type/history".into()],
                &KeyPlacement::End
            )),
            "---\nname: Probe\naliases: []\ntags: [type/history]\n---\n\nBody.\n"
        );
    }

    #[test]
    fn a_block_style_list_stays_block_style_at_its_own_indent() {
        let text = "---\ntags:\n    - a\n    - b\nname: Probe\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "c", &KeyPlacement::End)),
            "---\ntags:\n    - a\n    - b\n    - c\nname: Probe\n---\n"
        );
    }

    #[test]
    fn an_empty_list_writes_an_empty_list_and_keeps_the_key() {
        let text = "---\nname: Probe\ntags:\n  - a\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list("tags", &[], &KeyPlacement::End)),
            "---\nname: Probe\ntags: []\n---\n"
        );
    }

    #[test]
    fn a_comment_and_a_blank_line_survive_an_edit_to_a_neighbour() {
        let text = "---\n# why this note exists\nname: Probe\n\ntags: [a]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_scalar(
                "name",
                "Renamed",
                &KeyPlacement::End
            )),
            "---\n# why this note exists\nname: Renamed\n\ntags: [a]\n---\n"
        );
    }

    #[test]
    fn a_key_that_is_new_is_appended_in_the_style_the_block_already_uses() {
        let inline = "---\nname: Probe\ntags: [a]\n---\n";
        assert_eq!(
            edited(inline, |b| b.add_to_list(
                "aliases",
                "Other",
                &KeyPlacement::End
            )),
            "---\nname: Probe\ntags: [a]\naliases: [Other]\n---\n"
        );
        let blocked = "---\nname: Probe\ntags:\n  - a\n---\n";
        assert_eq!(
            edited(blocked, |b| b.add_to_list(
                "aliases",
                "Other",
                &KeyPlacement::End
            )),
            "---\nname: Probe\ntags:\n  - a\naliases:\n  - Other\n---\n"
        );
        let none = "---\nname: Probe\n---\n";
        assert_eq!(
            edited(none, |b| b.add_to_list(
                "aliases",
                "Other",
                &KeyPlacement::End
            )),
            "---\nname: Probe\naliases:\n  - Other\n---\n"
        );
    }

    /// A bare `tags:` with nothing after the colon is a common Obsidian
    /// template shape. `items_of`'s scalar arm used to read it back as one
    /// list item holding an empty string, so promoting it into a list wrote
    /// a phantom blank entry ahead of whatever the caller added (#92, I3).
    #[test]
    fn a_bare_key_with_no_value_gains_only_the_item_a_write_adds() {
        let text = "---\nname: X\ntags:\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "x", &KeyPlacement::End)),
            "---\nname: X\ntags:\n  - x\n---\n"
        );
    }

    /// `scalar_source` reads only an entry's first line. Promoting a
    /// multi-line plain scalar with that source used to drop its
    /// continuation line — `name: some\n  continued`, whose YAML value is
    /// `some continued`, promoted to a list item holding only `some` (#92,
    /// I4).
    #[test]
    fn promoting_a_multi_line_scalar_keeps_its_continuation() {
        let text = "---\nname: some\n  continued\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("name", "x", &KeyPlacement::End)),
            "---\nname:\n  - some continued\n  - x\n---\n"
        );
    }

    #[test]
    fn a_scalar_is_promoted_when_a_list_operation_names_it() {
        let text = "---\ntags: work\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list(
                "tags",
                "archived",
                &KeyPlacement::End
            )),
            "---\ntags:\n  - work\n  - archived\n---\n"
        );
    }

    /// The block between the fences, parsed as YAML rather than merely
    /// itemized: `parse_items` bails only on a structurally unrecognised
    /// top-level line, so it accepts a value a real YAML parser would split
    /// into more than one item (#92, R1 regression) and would have accepted
    /// even the original C2 corruption's comment-swallowed-bracket text if
    /// the rest of the block still looked like lines. A promise that "the
    /// written frontmatter must parse" is only as strong as what parses it.
    fn assert_block_parses_as_yaml(out: &str) -> serde_yaml::Value {
        let (_open, inner, _close, _after) = split_fences(out).unwrap().unwrap();
        serde_yaml::from_str::<serde_yaml::Value>(inner)
            .unwrap_or_else(|e| panic!("the written frontmatter must parse as YAML: {e}\n{out}"))
    }

    /// A scalar's own source is everything after its colon, comment
    /// included. Promoted straight into `[...]`, the comment used to run to
    /// the line's end and swallow the closing bracket, so the note stopped
    /// parsing for every other reader in the crate. The fix drops the
    /// comment rather than write YAML nothing downstream can read (#92, C2).
    #[test]
    fn promoting_a_commented_scalar_into_an_inline_list_writes_parseable_yaml() {
        let text = "---\nother: [z]\ntags: work # keep\n---\n";
        let out = edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End));
        assert_eq!(out, "---\nother: [z]\ntags: [work, new]\n---\n");
        // The promise this exists to keep: the result must itself parse,
        // and parse to the two items the write actually meant to write.
        let parsed = assert_block_parses_as_yaml(&out);
        assert_eq!(
            parsed.get("tags"),
            Some(&serde_yaml::Value::Sequence(vec![
                serde_yaml::Value::String("work".into()),
                serde_yaml::Value::String("new".into()),
            ])),
            "must read back as exactly the two items it was meant to hold: {out}"
        );
    }

    /// The same shape with a quoted scalar, which the finding calls out as
    /// corrupting the same way.
    #[test]
    fn promoting_a_quoted_commented_scalar_into_an_inline_list_writes_parseable_yaml() {
        let text = "---\ntags: \"work\" # keep\n---\n";
        let out = edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End));
        let parsed = assert_block_parses_as_yaml(&out);
        assert_eq!(
            parsed.get("tags"),
            Some(&serde_yaml::Value::Sequence(vec![
                serde_yaml::Value::String("work".into()),
                serde_yaml::Value::String("new".into()),
            ])),
            "must read back as exactly the two items it was meant to hold: {out}"
        );
    }

    /// R1: the fix wave's own flow-safety check asked only "does this parse
    /// at all", which a value holding a comma passes — into *two* items,
    /// not one, with no error at any point. Reproduced against the exact
    /// input the finding gives: a quoted, commented scalar with a comma,
    /// next to another key already in inline style so the promoted `tags`
    /// list takes that style too. The promoted item keeps `tags`'s own `"`
    /// rather than being requoted with `'` (#112).
    #[test]
    fn promoting_a_commented_scalar_holding_a_comma_keeps_it_one_quoted_item() {
        let text = "---\nother: [z]\ntags: \"a, b\" # keep\n---\n\nBody.\n";
        let out = edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End));
        assert_eq!(
            out, "---\nother: [z]\ntags: [\"a, b\", new]\n---\n\nBody.\n",
            "the comma-bearing value must stay one quoted item, not split in two"
        );
        let parsed = assert_block_parses_as_yaml(&out);
        assert_eq!(
            parsed.get("tags"),
            Some(&serde_yaml::Value::Sequence(vec![
                serde_yaml::Value::String("a, b".into()),
                serde_yaml::Value::String("new".into()),
            ])),
        );
    }

    /// The same comma, with no trailing comment. The source is already a
    /// safely double-quoted scalar, so it needs no requoting and keeps its
    /// own quote character rather than being normalised to single quotes.
    #[test]
    fn promoting_a_scalar_holding_a_comma_with_no_comment_keeps_its_own_quoting() {
        let text = "---\nother: [z]\ntags: \"a, b\"\n---\n\nBody.\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End)),
            "---\nother: [z]\ntags: [\"a, b\", new]\n---\n\nBody.\n"
        );
    }

    /// A value already single-quoted around its own comma. Already flow
    /// safe, so its own source text survives untouched.
    #[test]
    fn a_single_quoted_value_holding_a_comma_keeps_its_own_quoting() {
        let text = "---\nother: [z]\ntags: 'a, b'\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End)),
            "---\nother: [z]\ntags: ['a, b', new]\n---\n"
        );
    }

    /// A `#` with no space before it is not a YAML comment, so the fix must
    /// not treat it as one or requote a value that was already flow safe.
    #[test]
    fn a_hash_that_is_not_a_comment_is_kept_verbatim() {
        let text = "---\nother: [z]\ntags: tag#one\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End)),
            "---\nother: [z]\ntags: [tag#one, new]\n---\n"
        );
    }

    /// The ordinary case the fix must not disturb: a plain scalar with
    /// nothing that needs escaping keeps its own source text, unquoted,
    /// exactly as promoting into an inline list already did.
    #[test]
    fn an_ordinary_uncommented_scalar_keeps_its_own_source_text() {
        let text = "---\nother: [z]\ntags: work\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "new", &KeyPlacement::End)),
            "---\nother: [z]\ntags: [work, new]\n---\n"
        );
    }

    /// Not R1 itself, and not something this fix set out to change: a
    /// *fresh* caller-supplied value holding a comma, added to a list that
    /// is already inline, reaches the same `render_in_flow` fallback as an
    /// existing item's promoted source, so fixing the fallback closes this
    /// shape too. Before the fix, `flow_safe_scalar` did not exist and the
    /// old fallback (`yaml_scalar(&self.value)`) was identical to the text
    /// the check had just rejected, so it changed nothing and the comma
    /// still leaked through unquoted. The quote character is `"` because
    /// this block quotes nothing of its own (#112).
    #[test]
    fn a_fresh_value_holding_a_comma_added_to_an_inline_list_is_quoted() {
        let text = "---\naliases: [x]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list(
                "aliases",
                "Smith, John",
                &KeyPlacement::End
            )),
            "---\naliases: [x, \"Smith, John\"]\n---\n"
        );
    }

    /// #112: a vault writes its property values in one quote character —
    /// Obsidian's own writer emits `"` — and a `replace` re-serialises every
    /// item of the key it names. The re-emitted items must take the quoting
    /// the key already used, or every edit writes a changed line for a value
    /// that did not change.
    #[test]
    fn a_replaced_list_keeps_the_double_quotes_the_key_already_used() {
        let text = "---\nspouse_of: [\"[[Kara]]\"]\nparent_of: [\"[[Isabella]]\"]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "parent_of",
                &["[[Isabella]]".into(), "[[Lucian]]".into()],
                &KeyPlacement::End
            )),
            "---\nspouse_of: [\"[[Kara]]\"]\nparent_of: [\"[[Isabella]]\", \"[[Lucian]]\"]\n---\n"
        );
    }

    /// The same rule in the other direction: a vault written by a tool that
    /// prefers `'` keeps `'`, so neither convention churns.
    #[test]
    fn a_replaced_list_keeps_the_single_quotes_the_key_already_used() {
        let text = "---\nparent_of: ['[[Isabella]]']\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "parent_of",
                &["[[Lucian]]".into()],
                &KeyPlacement::End
            )),
            "---\nparent_of: ['[[Lucian]]']\n---\n"
        );
    }

    /// A key whose own items are all unquoted says nothing about quoting, so
    /// the block answers instead — the first quoted value it holds, which is
    /// the ladder `new_list_style` walks for inline-versus-block.
    #[test]
    fn a_key_that_quotes_nothing_takes_the_first_quoting_the_block_uses() {
        let text = "---\nspouse_of: ['[[Kara]]']\ntags: [a]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list(
                "tags",
                "[[Lucian]]",
                &KeyPlacement::End
            )),
            "---\nspouse_of: ['[[Kara]]']\ntags: [a, '[[Lucian]]']\n---\n"
        );
    }

    /// The key wins over the block: `seat` quotes with `'` and is the key
    /// being written, so `realm`'s `\"` does not decide it.
    #[test]
    fn a_key_that_quotes_its_own_value_beats_the_rest_of_the_block() {
        let text = "---\nrealm: \"New Visland\"\nseat: '[[Kara]]'\n---\n";
        assert_eq!(
            edited(text, |b| b.set_scalar(
                "seat",
                "[[Falconridge]]",
                &KeyPlacement::End
            )),
            "---\nrealm: \"New Visland\"\nseat: '[[Falconridge]]'\n---\n"
        );
    }

    /// A block that quotes nothing at all defaults to `\"`, which is what
    /// Obsidian's own property writer emits.
    #[test]
    fn a_block_that_quotes_nothing_defaults_to_double_quotes() {
        let text = "---\naliases: [x]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list(
                "aliases",
                "Smith, John",
                &KeyPlacement::End
            )),
            "---\naliases: [x, \"Smith, John\"]\n---\n"
        );
    }

    /// An apostrophe is the character an Obsidian note title actually holds,
    /// and single quoting is the form that has to double it. Under a
    /// double-quoted key it is written plainly.
    #[test]
    fn an_apostrophe_is_written_plainly_under_double_quotes() {
        let text = "---\nseat_of: [\"[[Kara]]\"]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "seat_of",
                &["[[Dragon's Rest]]".into()],
                &KeyPlacement::End
            )),
            "---\nseat_of: [\"[[Dragon's Rest]]\"]\n---\n"
        );
    }

    /// Whichever quote character a fresh value takes, YAML must read the
    /// value back unchanged. A double-quoted scalar is written with JSON's
    /// escaping, which YAML 1.2 accepts; a single-quoted one cannot hold a
    /// line break or a tab at all, so a value carrying one takes double
    /// quotes whatever the key prefers.
    #[test]
    fn a_fresh_value_survives_whichever_quoting_it_is_given() {
        for value in [
            "[[Dragon's Rest]]",
            "he said \"hi\"",
            "back\\slash",
            "a: b, c",
            "line1\nline2",
            "tab\there",
            "#hash",
            "trailing ",
        ] {
            for block in [
                "---\nk: \"q\"\nother: [x]\n---\n",
                "---\nk: 'q'\nother: [x]\n---\n",
                "---\nk: \"q\"\nother:\n  - x\n---\n",
            ] {
                let out = edited(block, |b| {
                    b.set_list("other", &[value.to_string()], &KeyPlacement::End)
                });
                let parsed = Block::parse(&out)
                    .unwrap_or_else(|e| panic!("{value:?} wrote unparseable YAML: {out:?}: {e}"))
                    .expect("a block");
                assert_eq!(
                    parsed.list("other"),
                    vec![value.to_string()],
                    "{value:?} did not survive {out:?}"
                );
            }
        }
    }

    /// #113: a key the note does not carry has no place of its own, and the
    /// end of the block is not where a vault with a fixed key order wants
    /// it. `after` names the key it sits below.
    #[test]
    fn a_new_key_lands_after_the_key_after_names() {
        let text = "---\nname: X\nparent_of: [a]\nniece_of: [b]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "sibling_of",
                &["c".into()],
                &KeyPlacement::After("parent_of".into())
            )),
            "---\nname: X\nparent_of: [a]\nsibling_of: [c]\nniece_of: [b]\n---\n"
        );
    }

    /// `before` is the other half, and the only way to name the top of the
    /// block: `after` cannot place a key above the first key there is.
    #[test]
    fn a_new_key_lands_before_the_key_before_names() {
        let text = "---\nname: X\ntags: [t]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_scalar(
                "id",
                "7",
                &KeyPlacement::Before("name".into())
            )),
            "---\nid: \"7\"\nname: X\ntags: [t]\n---\n"
        );
    }

    /// A comment on its own line introduces the key below it, so `after`
    /// puts the new key above that comment and not between it and the key
    /// it belongs to.
    #[test]
    fn placing_after_a_key_leaves_the_comment_with_the_key_it_introduces() {
        let text = "---\nparent_of: [a]\n# the non-family ties\nrules: [b]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "sibling_of",
                &["c".into()],
                &KeyPlacement::After("parent_of".into())
            )),
            "---\nparent_of: [a]\nsibling_of: [c]\n# the non-family ties\nrules: [b]\n---\n"
        );
    }

    /// The mirror: `before` steps back over the comment that introduces the
    /// anchor, so the new key joins the group rather than splitting its
    /// heading off. It stops at a blank line, which separates groups — a key
    /// placed before the first of a group belongs below that separator.
    #[test]
    fn placing_before_a_key_steps_over_its_comment_but_not_a_blank_line() {
        let text = "---\nname: X\n\n# the ties\nparent_of: [a]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "child_of",
                &["c".into()],
                &KeyPlacement::Before("parent_of".into())
            )),
            "---\nname: X\n\nchild_of: [c]\n# the ties\nparent_of: [a]\n---\n"
        );
    }

    /// An anchor the block does not hold is the caller's own mistake, and
    /// appending anyway is the silence #113 is about. The error names it.
    #[test]
    fn an_anchor_the_block_does_not_hold_is_an_error() {
        let mut block = Block::parse("---\nname: X\n---\n")
            .unwrap()
            .expect("a block");
        let err = block
            .set_scalar("ties", "none", &KeyPlacement::After("parent_of".into()))
            .unwrap_err();
        assert!(err.to_string().contains("parent_of"), "{err}");
    }

    /// A key the block already holds has a place, and that place is the
    /// file's. The placement says where to *put* a key, so re-running an
    /// edit moves nothing.
    #[test]
    fn a_key_the_block_already_holds_keeps_its_place() {
        let text = "---\nname: X\nties: [a]\ntags: [t]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list(
                "ties",
                &["b".into()],
                &KeyPlacement::After("tags".into())
            )),
            "---\nname: X\nties: [b]\ntags: [t]\n---\n"
        );
    }

    /// The default is unchanged: a new key with no placement is appended.
    #[test]
    fn a_new_key_with_no_placement_is_appended() {
        let text = "---\nname: X\ntags: [t]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_scalar("ties", "none", &KeyPlacement::End)),
            "---\nname: X\ntags: [t]\nties: none\n---\n"
        );
    }

    /// One placement or none. Two anchors name two places, and picking one
    /// would be guessing which the caller meant.
    #[test]
    fn naming_both_after_and_before_is_refused() {
        let err = KeyPlacement::new(Some("a"), Some("b")).unwrap_err();
        assert!(err.to_string().contains("one of"), "{err}");
        assert!(matches!(
            KeyPlacement::new(None, None),
            Ok(KeyPlacement::End)
        ));
    }

    #[test]
    fn adding_an_item_the_list_already_holds_changes_nothing() {
        let text = "---\ntags: [a, b]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "b", &KeyPlacement::End)),
            text
        );
    }

    #[test]
    fn removing_an_item_leaves_the_others_in_their_style() {
        let text = "---\ntags: [a, b, c]\n---\n";
        assert_eq!(
            edited(text, |b| b.remove_from_list("tags", "b")),
            "---\ntags: [a, c]\n---\n"
        );
    }

    #[test]
    fn removing_a_key_removes_its_lines_and_nothing_else() {
        let text = "---\nname: Probe\ntags:\n  - a\n  - b\naliases: []\n---\n\nBody.\n";
        assert_eq!(
            edited(text, |b| b.remove("tags")),
            "---\nname: Probe\naliases: []\n---\n\nBody.\n"
        );
    }

    #[test]
    fn removing_a_key_the_block_does_not_hold_changes_nothing() {
        let text = "---\nname: Probe\n---\n";
        assert_eq!(edited(text, |b| b.remove("absent")), text);
        assert_eq!(edited(text, |b| b.remove_from_list("absent", "x")), text);
    }

    #[test]
    fn a_value_that_needs_quoting_gets_it() {
        let text = "---\nname: Probe\n---\n";
        assert_eq!(
            edited(text, |b| b.set_scalar(
                "reason",
                "semantic similarity: 0.5",
                &KeyPlacement::End
            )),
            "---\nname: Probe\nreason: \"semantic similarity: 0.5\"\n---\n"
        );
    }

    #[test]
    fn a_bool_is_written_unquoted() {
        assert_eq!(
            edited("---\nname: Probe\n---\n", |b| b.set_bool("archived", true)),
            "---\nname: Probe\narchived: true\n---\n"
        );
    }

    #[test]
    fn a_crlf_block_keeps_crlf_on_the_key_it_gains() {
        assert_eq!(
            edited("---\r\nname: Probe\r\n---\r\n", |b| b.set_scalar(
                "x",
                "1",
                &KeyPlacement::End
            )),
            "---\r\nname: Probe\r\nx: \"1\"\r\n---\r\n"
        );
    }

    /// A quoted key `key_of` does not recognise folds into the entry above
    /// it, so the entry's own span is really two mapping keys. Before this
    /// guard, `classify` took `map.values().next()` and reported the
    /// entry's shape from whichever key parsed first, so an edit to `name`
    /// silently deleted `"other key"` along with it (#92, C1).
    #[test]
    fn a_quoted_key_folded_into_the_previous_entry_refuses_the_edit_instead_of_deleting_it() {
        let text = "---\nname: X\n\"other key\": v\ntags: [a]\n---\n\nBody.\n";
        let mut block = Block::parse(text).unwrap().unwrap();
        assert_eq!(
            block.value("name"),
            Some(&Value::Opaque("a value that is not one mapping entry"))
        );
        let err = block
            .set_scalar("name", "Y", &KeyPlacement::End)
            .unwrap_err();
        assert!(err.to_string().contains("`name`"), "{err}");
        assert_eq!(
            block.render(),
            text,
            "the refused write must not touch the file"
        );
    }

    #[test]
    fn an_edit_to_an_opaque_value_is_refused_and_writes_nothing() {
        let text = "---\nname: Probe\nnested:\n  inner: 1\n---\n";
        let mut block = Block::parse(text).unwrap().unwrap();
        let err = block
            .set_list("nested", &["a".into()], &KeyPlacement::End)
            .unwrap_err();
        assert!(err.to_string().contains("nested mapping"), "{err}");
        assert!(err.to_string().contains("`nested`"), "{err}");
        assert_eq!(block.render(), text);
    }

    #[test]
    fn a_scalar_reads_back_as_the_value_it_holds() {
        let block = Block::parse("---\narchived_from: Areas/n.md\nn: 3\n---\n")
            .unwrap()
            .unwrap();
        assert_eq!(block.scalar("archived_from").as_deref(), Some("Areas/n.md"));
        assert_eq!(block.scalar("n").as_deref(), Some("3"));
        assert_eq!(block.scalar("absent"), None);
    }

    #[test]
    fn a_list_reads_back_its_items_however_it_is_written() {
        let block = Block::parse("---\ninline: [a, b]\nblock:\n  - c\n  - d\nscalar: e\n---\n")
            .unwrap()
            .unwrap();
        assert_eq!(block.list("inline"), vec!["a", "b"]);
        assert_eq!(block.list("block"), vec!["c", "d"]);
        assert_eq!(block.list("scalar"), vec!["e"]);
        assert_eq!(block.list("absent"), Vec::<String>::new());
    }

    #[test]
    fn an_inline_list_of_numbers_gains_an_item_and_the_numbers_stay_numbers() {
        let text = "---\nratings: [1, 2, 3]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("ratings", "4", &KeyPlacement::End)),
            "---\nratings: [1, 2, 3, \"4\"]\n---\n"
        );
    }

    #[test]
    fn a_block_list_of_quoted_items_loses_one_and_the_survivors_keep_their_quotes() {
        let text = "---\ntags:\n  - 'a'\n  - 'b'\n  - 'c'\n---\n";
        assert_eq!(
            edited(text, |b| b.remove_from_list("tags", "b")),
            "---\ntags:\n  - 'a'\n  - 'c'\n---\n"
        );
    }

    #[test]
    fn an_inline_item_that_was_quoted_stays_quoted_when_a_neighbour_is_removed() {
        let text = "---\ntags: [a, 'b c', plain]\n---\n";
        assert_eq!(
            edited(text, |b| b.remove_from_list("tags", "a")),
            "---\ntags: ['b c', plain]\n---\n"
        );
    }

    #[test]
    fn an_item_holding_a_comma_inside_quotes_is_one_item_not_two() {
        let text = "---\ntags: [a, \"b, c\", d]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "e", &KeyPlacement::End)),
            "---\ntags: [a, \"b, c\", d, e]\n---\n"
        );
    }

    #[test]
    fn an_item_holding_an_escaped_quote_survives_an_edit_to_a_neighbour() {
        let text = "---\ntags: [\"a\\\"b\", c]\n---\n";
        assert_eq!(
            edited(text, |b| b.remove_from_list("tags", "c")),
            "---\ntags: [\"a\\\"b\"]\n---\n"
        );
    }

    #[test]
    fn an_inline_list_with_a_trailing_comment_refuses_the_write_and_leaves_the_block_unchanged() {
        let text = "---\ntags: [a, b]  # trailing comment\n---\n";
        let mut block = Block::parse(text).unwrap().unwrap();
        let err = block
            .add_to_list("tags", "c", &KeyPlacement::End)
            .unwrap_err();
        assert!(err.to_string().contains("flow list"), "{err}");
        assert!(err.to_string().contains("`tags`"), "{err}");
        assert_eq!(block.render(), text);
    }

    /// A trailing comma in a flow list — `tags: [a, b,]` — is legal YAML,
    /// classifies as an addressable inline list, and reaches this same
    /// guard: `split_flow_items` reads the trailing comma as one more
    /// split, so its sources come back one longer than the two values
    /// `serde_yaml` parses. The guard's fallback is what makes this read
    /// back as two items and add a third correctly rather than losing one.
    #[test]
    fn a_trailing_comma_in_a_flow_list_reaches_the_guard_through_the_public_api() {
        let text = "---\ntags: [a, b,]\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "c", &KeyPlacement::End)),
            "---\ntags: [a, b, c]\n---\n"
        );
    }

    #[test]
    fn the_count_guard_keeps_every_item_when_sources_and_values_disagree() {
        // The trailing-comma case above reaches this guard through the
        // public API; this test drives the internal `list_items` directly
        // for a second shape that also reaches it: an inline entry's text
        // paired with the block style's source extraction finds no `- `
        // lines, so sources come back empty against three parsed values,
        // and the guard must fall back to re-serialising every item rather
        // than lose the two the zip would otherwise drop.
        let items = list_items("tags: [a, b, c]\n", ListStyle::Block { indent: 2 });
        assert_eq!(items.len(), 3);
        assert_eq!(
            items.iter().map(|i| i.value.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert!(items.iter().all(|i| i.source.is_none()));
    }

    #[test]
    fn an_operation_and_its_inverse_return_the_original_bytes() {
        for text in [
            "---\nname: Probe\ntags: [a, b]\n---\n\nBody.\n",
            "---\ntags:\n  - a\n  - b\nname: Probe\n---\n\nBody.\n",
        ] {
            let mut block = Block::parse(text).unwrap().unwrap();
            block.add_to_list("tags", "zz", &KeyPlacement::End).unwrap();
            block.remove_from_list("tags", "zz").unwrap();
            assert_eq!(block.render(), text);
        }
    }

    #[test]
    fn a_note_with_no_block_gets_one_above_its_body() {
        let mut block = Block::parse_or_open("# Title\n\nBody.\n").unwrap();
        block
            .add_to_list("tags", "type/lore", &KeyPlacement::End)
            .unwrap();
        assert_eq!(
            block.render(),
            "---\ntags:\n  - type/lore\n---\n\n# Title\n\nBody.\n"
        );
    }

    #[test]
    fn an_opened_block_that_gains_nothing_is_empty_and_keeps_the_body() {
        let block = Block::parse_or_open("# Title\n").unwrap();
        assert!(block.is_empty());
        assert_eq!(block.body(), "# Title\n");
    }

    #[test]
    fn a_note_with_a_block_is_parsed_rather_than_opened() {
        let block = Block::parse_or_open("---\nname: X\n---\n\nBody.\n").unwrap();
        assert_eq!(block.render(), "---\nname: X\n---\n\nBody.\n");
    }

    /// `crate::markdown::split_frontmatter` compares `line.trim() == "---"`,
    /// so every other reader in the crate already treats a fence with
    /// trailing whitespace as frontmatter. Before this fix `split_fences`
    /// compared exactly and saw no block, so `parse_or_open` opened a
    /// second one above what the rest of the crate reads as the note's
    /// real properties, demoting them to body text (#92, I1).
    #[test]
    fn a_fence_with_trailing_whitespace_is_parsed_not_reopened() {
        let mut block = Block::parse_or_open("--- \nname: X\n---\n\nBody\n").unwrap();
        block
            .add_to_list("tags", "new", &KeyPlacement::End)
            .unwrap();
        assert_eq!(
            block.render(),
            "--- \nname: X\ntags:\n  - new\n---\n\nBody\n",
            "one block, not two, and the fence's own whitespace kept verbatim"
        );
    }

    #[test]
    fn opening_a_block_on_an_empty_note_writes_no_separator() {
        let mut block = Block::parse_or_open("").unwrap();
        block.set_scalar("name", "X", &KeyPlacement::End).unwrap();
        assert_eq!(block.render(), "---\nname: X\n---\n");
    }

    #[test]
    fn opening_a_block_on_a_crlf_note_writes_crlf() {
        let mut block = Block::parse_or_open("# Title\r\n").unwrap();
        block.set_scalar("name", "X", &KeyPlacement::End).unwrap();
        assert_eq!(block.render(), "---\r\nname: X\r\n---\r\n\r\n# Title\r\n");
    }

    #[test]
    fn parse_or_open_still_refuses_a_block_it_cannot_edit() {
        assert!(Block::parse_or_open("---\nname: X\n\nBody.\n").is_err());
    }
}
