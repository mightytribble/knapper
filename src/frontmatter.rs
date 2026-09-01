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
    /// through `yaml_scalar`.
    fn render(&self) -> String {
        match &self.source {
            Some(source) => source.clone(),
            None => yaml_scalar(&self.value),
        }
    }

    /// The item's rendered text as it will sit inside `[...]`. An item's own
    /// source can carry a trailing `# comment` that is harmless at the end
    /// of its original scalar line, but the same text placed before a `,`
    /// inside brackets makes the comment swallow the rest of the sequence —
    /// `render_list`'s block-style arm never hits this, because there a
    /// comment sits at the end of its own line either way. Falling back to a
    /// fresh `yaml_scalar` loses the comment, but a comment that belonged to
    /// the key the write named is a smaller loss than frontmatter nothing
    /// downstream can parse (#92, C2).
    fn render_in_flow(&self) -> String {
        let text = self.render();
        if serde_yaml::from_str::<serde_yaml::Value>(&format!("k: [{text}]")).is_ok() {
            text
        } else {
            yaml_scalar(&self.value)
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

    /// Write `key` as a scalar, quoted when YAML needs it.
    pub fn set_scalar(&mut self, key: &str, value: &str) -> Result<()> {
        self.check_editable(key)?;
        let text = format!("{key}: {}{}", yaml_scalar(value), self.newline);
        self.put(key, text, Value::Scalar);
        Ok(())
    }

    /// Write `key` as a bare `true` or `false`, which a quoted scalar would
    /// not be.
    pub fn set_bool(&mut self, key: &str, value: bool) -> Result<()> {
        self.check_editable(key)?;
        let text = format!("{key}: {value}{}", self.newline);
        self.put(key, text, Value::Scalar);
        Ok(())
    }

    /// Write `key` as a list. Every item comes from the caller, so every item
    /// is serialised fresh through `yaml_scalar`. An empty list is written
    /// `[]` in any style, because block style with no items reads back as
    /// null.
    pub fn set_list(&mut self, key: &str, items: &[String]) -> Result<()> {
        let style = self.list_style_for(key)?;
        let items: Vec<ListItem> = items.iter().cloned().map(ListItem::fresh).collect();
        let text = self.render_list(key, &items, style);
        self.put(key, text, Value::List(style));
        Ok(())
    }

    /// Add `item` to `key`'s list, creating the list when the key is absent
    /// and promoting it when the key holds a scalar. An item the list already
    /// holds changes nothing. Every existing item keeps its own source text;
    /// only the added item is serialised through `yaml_scalar`.
    pub fn add_to_list(&mut self, key: &str, item: &str) -> Result<()> {
        let style = self.list_style_for(key)?;
        let mut items = self.items_of(key);
        if items.iter().any(|i| i.value == item) {
            return Ok(());
        }
        items.push(ListItem::fresh(item.to_string()));
        let text = self.render_list(key, &items, style);
        self.put(key, text, Value::List(style));
        Ok(())
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
        let mut items = self.items_of(key);
        items.retain(|i| i.value != item);
        let text = self.render_list(key, &items, style);
        self.put(key, text, Value::List(style));
        Ok(())
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

    fn render_list(&self, key: &str, items: &[ListItem], style: ListStyle) -> String {
        if items.is_empty() {
            return format!("{key}: []{}", self.newline);
        }
        match style {
            ListStyle::Inline => {
                let body: Vec<String> = items.iter().map(ListItem::render_in_flow).collect();
                format!("{key}: [{}]{}", body.join(", "), self.newline)
            }
            ListStyle::Block { indent } => {
                let pad = " ".repeat(indent);
                let mut out = format!("{key}:{}", self.newline);
                for item in items {
                    out.push_str(&pad);
                    out.push_str("- ");
                    out.push_str(&item.render());
                    out.push_str(&self.newline);
                }
                out
            }
        }
    }

    fn put(&mut self, key: &str, text: String, value: Value) {
        let entry = Item::Entry {
            key: key.to_string(),
            value,
            text,
        };
        match self.find(key) {
            Some(idx) => self.items[idx] = entry,
            None => self.items.push(entry),
        }
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

/// One scalar as YAML, quoted when it has to be.
fn yaml_scalar(value: &str) -> String {
    serde_yaml::to_string(&serde_yaml::Value::String(value.to_string()))
        .unwrap_or_else(|_| value.to_string())
        .trim_end()
        .to_string()
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
            edited(text, |b| b.set_list("tags", &["type/history".into()])),
            "---\nname: Probe\naliases: []\ntags: [type/history]\n---\n\nBody.\n"
        );
    }

    #[test]
    fn a_block_style_list_stays_block_style_at_its_own_indent() {
        let text = "---\ntags:\n    - a\n    - b\nname: Probe\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "c")),
            "---\ntags:\n    - a\n    - b\n    - c\nname: Probe\n---\n"
        );
    }

    #[test]
    fn an_empty_list_writes_an_empty_list_and_keeps_the_key() {
        let text = "---\nname: Probe\ntags:\n  - a\n---\n";
        assert_eq!(
            edited(text, |b| b.set_list("tags", &[])),
            "---\nname: Probe\ntags: []\n---\n"
        );
    }

    #[test]
    fn a_comment_and_a_blank_line_survive_an_edit_to_a_neighbour() {
        let text = "---\n# why this note exists\nname: Probe\n\ntags: [a]\n---\n";
        assert_eq!(
            edited(text, |b| b.set_scalar("name", "Renamed")),
            "---\n# why this note exists\nname: Renamed\n\ntags: [a]\n---\n"
        );
    }

    #[test]
    fn a_key_that_is_new_is_appended_in_the_style_the_block_already_uses() {
        let inline = "---\nname: Probe\ntags: [a]\n---\n";
        assert_eq!(
            edited(inline, |b| b.add_to_list("aliases", "Other")),
            "---\nname: Probe\ntags: [a]\naliases: [Other]\n---\n"
        );
        let blocked = "---\nname: Probe\ntags:\n  - a\n---\n";
        assert_eq!(
            edited(blocked, |b| b.add_to_list("aliases", "Other")),
            "---\nname: Probe\ntags:\n  - a\naliases:\n  - Other\n---\n"
        );
        let none = "---\nname: Probe\n---\n";
        assert_eq!(
            edited(none, |b| b.add_to_list("aliases", "Other")),
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
            edited(text, |b| b.add_to_list("tags", "x")),
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
            edited(text, |b| b.add_to_list("name", "x")),
            "---\nname:\n  - some continued\n  - x\n---\n"
        );
    }

    #[test]
    fn a_scalar_is_promoted_when_a_list_operation_names_it() {
        let text = "---\ntags: work\n---\n";
        assert_eq!(
            edited(text, |b| b.add_to_list("tags", "archived")),
            "---\ntags:\n  - work\n  - archived\n---\n"
        );
    }

    /// A scalar's own source is everything after its colon, comment
    /// included. Promoted straight into `[...]`, the comment used to run to
    /// the line's end and swallow the closing bracket, so the note stopped
    /// parsing for every other reader in the crate. The fix drops the
    /// comment rather than write YAML nothing downstream can read (#92, C2).
    #[test]
    fn promoting_a_commented_scalar_into_an_inline_list_writes_parseable_yaml() {
        let text = "---\nother: [z]\ntags: work # keep\n---\n";
        let out = edited(text, |b| b.add_to_list("tags", "new"));
        assert_eq!(out, "---\nother: [z]\ntags: [work, new]\n---\n");
        // The promise this exists to keep: the result must itself parse.
        let (_open, inner, _close, _after) = split_fences(&out).unwrap().unwrap();
        assert!(
            parse_items(inner).is_ok(),
            "the written frontmatter must parse: {out}"
        );
    }

    /// The same shape with a quoted scalar, which the finding calls out as
    /// corrupting the same way.
    #[test]
    fn promoting_a_quoted_commented_scalar_into_an_inline_list_writes_parseable_yaml() {
        let text = "---\ntags: \"work\" # keep\n---\n";
        let out = edited(text, |b| b.add_to_list("tags", "new"));
        let (_open, inner, _close, _after) = split_fences(&out).unwrap().unwrap();
        assert!(
            parse_items(inner).is_ok(),
            "the written frontmatter must parse: {out}"
        );
    }

    #[test]
    fn adding_an_item_the_list_already_holds_changes_nothing() {
        let text = "---\ntags: [a, b]\n---\n";
        assert_eq!(edited(text, |b| b.add_to_list("tags", "b")), text);
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
            edited(text, |b| b.set_scalar("reason", "semantic similarity: 0.5")),
            "---\nname: Probe\nreason: 'semantic similarity: 0.5'\n---\n"
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
            edited("---\r\nname: Probe\r\n---\r\n", |b| b.set_scalar("x", "1")),
            "---\r\nname: Probe\r\nx: '1'\r\n---\r\n"
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
        let err = block.set_scalar("name", "Y").unwrap_err();
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
        let err = block.set_list("nested", &["a".into()]).unwrap_err();
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
            edited(text, |b| b.add_to_list("ratings", "4")),
            "---\nratings: [1, 2, 3, '4']\n---\n"
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
            edited(text, |b| b.add_to_list("tags", "e")),
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
        let err = block.add_to_list("tags", "c").unwrap_err();
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
            edited(text, |b| b.add_to_list("tags", "c")),
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
            block.add_to_list("tags", "zz").unwrap();
            block.remove_from_list("tags", "zz").unwrap();
            assert_eq!(block.render(), text);
        }
    }

    #[test]
    fn a_note_with_no_block_gets_one_above_its_body() {
        let mut block = Block::parse_or_open("# Title\n\nBody.\n").unwrap();
        block.add_to_list("tags", "type/lore").unwrap();
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
        block.add_to_list("tags", "new").unwrap();
        assert_eq!(
            block.render(),
            "--- \nname: X\ntags:\n  - new\n---\n\nBody\n",
            "one block, not two, and the fence's own whitespace kept verbatim"
        );
    }

    #[test]
    fn opening_a_block_on_an_empty_note_writes_no_separator() {
        let mut block = Block::parse_or_open("").unwrap();
        block.set_scalar("name", "X").unwrap();
        assert_eq!(block.render(), "---\nname: X\n---\n");
    }

    #[test]
    fn opening_a_block_on_a_crlf_note_writes_crlf() {
        let mut block = Block::parse_or_open("# Title\r\n").unwrap();
        block.set_scalar("name", "X").unwrap();
        assert_eq!(block.render(), "---\r\nname: X\r\n---\r\n\r\n# Title\r\n");
    }

    #[test]
    fn parse_or_open_still_refuses_a_block_it_cannot_edit() {
        assert!(Block::parse_or_open("---\nname: X\n\nBody.\n").is_err());
    }
}
