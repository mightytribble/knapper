use crate::markdown::HeadingInfo;

/// Represents a single semantic chunk extracted from a markdown file.
pub struct Chunk {
    /// The heading line (any `#` level), if any.
    pub heading: Option<String>,
    /// Heading *text* of every ancestor section, outermost first, including this
    /// chunk's own heading as the last element. Empty for pre-heading content.
    ///
    /// `## Abilities` / `### Combat` yields `["Abilities", "Combat"]`. Not
    /// persisted, but [`crate::prefix`] embeds it: a chunk's own text opens
    /// with its own heading and no ancestor's, since `structure_chunk` makes
    /// subsections siblings of their parent rather than children.
    pub heading_path: Vec<String>,
    /// Full chunk text (without frontmatter).
    pub text: String,
    /// First 200 chars of `text`, truncated with `"..."` if needed.
    pub snippet: String,
}

impl Chunk {
    /// Build a chunk from a heading line and a body, deriving `text` and `snippet`.
    ///
    /// The heading line is prepended to the body so every chunk *begins* with the
    /// heading it is labelled with. `continuation` appends ` (cont.)` to the label
    /// for the second and later pieces of a split section.
    fn from_section(
        heading_line: Option<&str>,
        heading_path: &[String],
        body: &str,
        continuation: bool,
    ) -> Chunk {
        let heading = heading_line.map(|h| {
            if continuation {
                format!("{h} (cont.)")
            } else {
                h.to_string()
            }
        });
        let text = match &heading {
            Some(h) => format!("{h}\n{}", body.trim()),
            None => body.trim().to_string(),
        };
        let snippet = make_snippet(&text);
        Chunk {
            heading,
            heading_path: heading_path.to_vec(),
            text,
            snippet,
        }
    }

    /// Put a carried heading line at the head of this chunk's text (issue #44).
    ///
    /// The chunk's own `heading` and `heading_path` do not change: the carried
    /// line labels no chunk, it is only kept in the corpus. `text` is what the
    /// keyword index reads, so the line stays searchable there.
    fn prepend_carried(&mut self, carried: Option<&str>) {
        let Some(line) = carried else {
            return;
        };
        self.text = format!("{line}\n{}", self.text);
        self.snippet = make_snippet(&self.text);
    }
}

/// Result of parsing a markdown file.
pub struct ParsedMarkdown {
    /// Tags extracted from YAML frontmatter.
    pub tags: Vec<String>,
    /// Semantic chunks produced by smart break-point scoring.
    pub chunks: Vec<Chunk>,
}

/// A scored candidate position where a chunk boundary could be placed.
pub struct BreakPoint {
    pub byte_offset: usize,
    pub line_number: usize,
    pub score: u32,
    pub inside_code_fence: bool,
}

/// Scan content line by line and assign break-point scores.
///
/// Scoring rules:
/// - `# ` heading: 100
/// - `## ` heading: 90
/// - `### ` heading: 80
/// - `#### ` heading: 70
/// - `##### ` heading: 60
/// - `###### ` heading: 50
/// - `---`/`***`/`___` (thematic breaks): 60
/// - Code fence boundaries (`` ``` ``): 80
/// - Empty lines: 20
/// - List items (`- `, `* `, digit prefix): 5
/// - Other non-empty lines: 1 (excluded from results)
pub fn find_break_points(content: &str) -> Vec<BreakPoint> {
    let mut break_points = Vec::new();
    let mut inside_code_fence = false;
    let mut byte_offset = 0;

    for (line_number, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        let score = if trimmed.starts_with("```") {
            // Toggle fence state; the fence boundary itself is NOT "inside"
            inside_code_fence = !inside_code_fence;
            // Mark as not inside — fence boundaries are valid break points
            let bp_inside = false;
            break_points.push(BreakPoint {
                byte_offset,
                line_number,
                score: 80,
                inside_code_fence: bp_inside,
            });
            byte_offset += line.len()
                + if byte_offset + line.len() < content.len() {
                    1
                } else {
                    0
                };
            continue;
        } else if inside_code_fence {
            // Lines inside code fences: push with inside_code_fence = true
            // so callers can inspect the field; smart_chunk filters them out.
            break_points.push(BreakPoint {
                byte_offset,
                line_number,
                score: 1,
                inside_code_fence: true,
            });
            byte_offset += line.len()
                + if byte_offset + line.len() < content.len() {
                    1
                } else {
                    0
                };
            continue;
        } else if trimmed.starts_with("# ") && !trimmed.starts_with("## ") {
            100
        } else if trimmed.starts_with("## ") && !trimmed.starts_with("### ") {
            90
        } else if trimmed.starts_with("### ") && !trimmed.starts_with("#### ") {
            80
        } else if trimmed.starts_with("#### ") && !trimmed.starts_with("##### ") {
            70
        } else if trimmed.starts_with("##### ") && !trimmed.starts_with("###### ") {
            60
        } else if trimmed.starts_with("###### ") {
            50
        } else if is_thematic_break(trimmed) {
            60
        } else if trimmed.is_empty() {
            20
        } else if is_list_item(trimmed) {
            5
        } else {
            1
        };

        if score > 1 {
            break_points.push(BreakPoint {
                byte_offset,
                line_number,
                score,
                inside_code_fence,
            });
        }

        byte_offset += line.len()
            + if byte_offset + line.len() < content.len() {
                1
            } else {
                0
            };
    }

    break_points
}

/// Check if a line is a thematic break (`---`, `***`, `___` with 3+ chars, optional spaces).
fn is_thematic_break(trimmed: &str) -> bool {
    if trimmed.len() < 3 {
        return false;
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let first = chars[0];
    if first != '-' && first != '*' && first != '_' {
        return false;
    }
    chars.iter().all(|&c| c == first || c == ' ')
        && chars.iter().filter(|&&c| c == first).count() >= 3
}

/// Check if a line starts as a list item.
fn is_list_item(trimmed: &str) -> bool {
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        return true;
    }
    // Check for ordered list: digit(s) followed by `. ` or `) `
    let mut chars = trimmed.chars();
    if let Some(first) = chars.next()
        && first.is_ascii_digit()
    {
        for c in chars {
            if c.is_ascii_digit() {
                continue;
            }
            if c == '.' || c == ')' {
                return true;
            }
            break;
        }
    }
    false
}

/// Approximate token count: ~4 chars per token.
fn approx_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Snap a byte offset to the nearest valid UTF-8 char boundary (forward).
fn snap_to_char_boundary(s: &str, offset: usize) -> usize {
    let offset = offset.min(s.len());
    let mut pos = offset;
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Extract the first heading line from text (any `#` level).
fn extract_heading(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') && trimmed.contains(' ') {
            return Some(line.to_string());
        }
    }
    None
}

/// Smart chunk splitting using scored break points.
///
/// - `target_tokens`: desired chunk size in approximate tokens (~4 chars/token)
/// - `overlap_pct`: percentage of target_tokens to overlap between chunks (e.g. 15 = 15%)
///
/// Never splits inside code fences. Finds the best break point near the token
/// target using a weighted score that considers both inherent score and distance.
pub fn smart_chunk(content: &str, target_tokens: usize, overlap_pct: usize) -> Vec<Chunk> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let break_points = find_break_points(content);
    let target_chars = target_tokens * 4;
    let overlap_chars = (target_chars * overlap_pct) / 100;

    // If the content fits in one chunk, return it as-is
    if approx_tokens(content) <= target_tokens {
        let heading = extract_heading(content);
        let snippet = make_snippet(content.trim());
        return vec![Chunk {
            heading,
            heading_path: Vec::new(),
            text: content.trim().to_string(),
            snippet,
        }];
    }

    let mut chunks = Vec::new();
    let mut start_offset = 0;

    while start_offset < content.len() {
        start_offset = snap_to_char_boundary(content, start_offset);
        if start_offset >= content.len() {
            break;
        }
        let remaining = &content[start_offset..];
        if remaining.trim().is_empty() {
            break;
        }

        // If remaining content fits in one chunk, take it all
        if approx_tokens(remaining) <= target_tokens {
            let text = remaining.trim().to_string();
            if !text.is_empty() {
                let heading = extract_heading(&text);
                let snippet = make_snippet(&text);
                chunks.push(Chunk {
                    heading,
                    heading_path: Vec::new(),
                    text,
                    snippet,
                });
            }
            break;
        }

        // Find the ideal cut point: target_chars from start_offset
        let ideal_end = start_offset + target_chars;

        // Find the best break point near ideal_end
        // Filter to break points that are:
        // 1. After start_offset
        // 2. Not inside code fences
        // 3. Within a reasonable range of ideal_end
        let best_bp = break_points
            .iter()
            .filter(|bp| {
                bp.byte_offset > start_offset
                    && !bp.inside_code_fence
                    && bp.byte_offset <= start_offset + target_chars * 2
            })
            .max_by(|a, b| {
                let score_a = weighted_score(a, ideal_end);
                let score_b = weighted_score(b, ideal_end);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let cut_offset = match best_bp {
            Some(bp) => bp.byte_offset,
            None => {
                // No good break point found; cut at target
                let cut = snap_to_char_boundary(
                    content,
                    (start_offset + target_chars).min(content.len()),
                );
                // Try to find a newline near the cut
                let fallback = if let Some(nl) = content[start_offset..cut.min(content.len())]
                    .rfind('\n')
                    .map(|p| start_offset + p + 1)
                {
                    if nl > start_offset { nl } else { cut }
                } else {
                    cut
                };
                // Guard: always advance by at least one byte to prevent infinite loops
                fallback.max(start_offset + 1).min(content.len())
            }
        };

        let cut_offset = snap_to_char_boundary(content, cut_offset);
        let chunk_text = content[start_offset..cut_offset].trim().to_string();
        if !chunk_text.is_empty() {
            let heading = extract_heading(&chunk_text);
            let snippet = make_snippet(&chunk_text);
            chunks.push(Chunk {
                heading,
                heading_path: Vec::new(),
                text: chunk_text,
                snippet,
            });
        }

        // Move start forward, applying overlap.
        if cut_offset >= content.len() {
            break;
        }
        // Only step back by the overlap window when the chunk we just emitted is
        // LARGER than that window. If the chunk is at or below overlap size,
        // `cut_offset - overlap_chars` lands before this chunk began; the old
        // `.max(start_offset + 1)` guard then advanced the start by a single
        // character, re-selected the same nearby high-score break point, and
        // crawled forward one char at a time — emitting hundreds of near-duplicate
        // sub-chunks per file (observed: a 4.5k-word note shattered into 900+
        // empty-heading chunks). Advancing fully to the cut guarantees real
        // forward progress and eliminates the crawl.
        start_offset = if overlap_chars > 0 && cut_offset > start_offset + overlap_chars {
            cut_offset - overlap_chars
        } else {
            cut_offset
        };
    }

    chunks
}

/// Compute a weighted score that balances break-point quality with proximity to target.
fn weighted_score(bp: &BreakPoint, ideal_offset: usize) -> f64 {
    let distance = (bp.byte_offset as f64 - ideal_offset as f64).abs();
    // Normalize distance: closer to ideal = higher score multiplier
    // At distance 0, multiplier = 1.0; at distance = ideal_offset, multiplier ~= 0
    let distance_factor = 1.0 / (1.0 + distance / 500.0);
    bp.score as f64 * distance_factor
}

/// Byte offset at which each line of `content` begins.
///
/// Uses `split_inclusive` so multi-byte characters and `\r\n` terminators are
/// accounted for exactly. Index positions line up with `str::lines()`, which is
/// what `markdown::parse_headings` reports against.
fn line_offsets(content: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut pos = 0usize;
    for line in content.split_inclusive('\n') {
        offsets.push(pos);
        pos += line.len();
    }
    offsets
}

/// The settings that decide where a chunk boundary falls and are config keys
/// rather than [`limits`] constants.
///
/// They travel together because every path that chunks a file must use the same
/// pair. A note written through the write pipeline at one setting and
/// re-indexed at another is cut into two different sets of rows, and nothing
/// downstream can tell. This is what [`crate::prefix::EmbedComposition`] does
/// for the embedding inputs, applied to the chunker's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkOptions {
    /// The shortest section body that becomes a chunk of its own (issue #43).
    /// A shorter section merges into the preceding chunk of the same file.
    pub min_chars: usize,
    /// Whether a line that is one bold span and nothing else opens a section
    /// (issue #44).
    pub promote_bold: bool,
}

/// Drop a promoted line whose section body is empty.
///
/// `structure_chunk` skips a `#` heading with no body of its own, because its
/// text survives in the `heading_path` of its descendants. A promoted line is
/// flat and has no descendants — the next one pops it off the ancestor stack —
/// so the same skip would delete the line from the corpus. It stays in the
/// enclosing section's body instead (issue #44).
///
/// This is the one place the chunker's set differs from
/// `markdown::headings_with_promotions`, which keeps such a line because a
/// caller addresses an empty section to fill it (issue #69).
fn drop_bodyless_promotions(content: &str, entries: Vec<HeadingInfo>) -> Vec<HeadingInfo> {
    let offsets = line_offsets(content);
    let line_start = |line: usize| offsets.get(line).copied().unwrap_or(content.len());

    let mut out = Vec::with_capacity(entries.len());
    for (i, info) in entries.iter().enumerate() {
        if info.promoted {
            let body_start = line_start(info.line + 1);
            let body_end = entries
                .get(i + 1)
                .map(|next| line_start(next.line))
                .unwrap_or(content.len());
            if content[body_start..body_end.max(body_start)]
                .trim()
                .is_empty()
            {
                continue;
            }
        }
        out.push(info.clone());
    }
    out
}

/// What counts as a heading for chunking (issue #44).
///
/// It is `markdown::headings_with_promotions` with the bodyless promotions
/// dropped: what the parser calls a heading, narrowed to what starts a chunk.
/// With `promote_bold` off, a bold-only line is not a heading at all and the
/// answer is the ATX headings alone, which reproduces the pre-#44 chunking
/// exactly.
fn structure_headings(content: &str, promote_bold: bool) -> Vec<HeadingInfo> {
    if !promote_bold {
        return crate::markdown::parse_headings(content);
    }
    drop_bodyless_promotions(content, crate::markdown::headings_with_promotions(content))
}

/// Structure-first chunking: a chunk boundary is placed at every ATX heading,
/// and size only decides how an *oversized* section is subdivided.
///
/// The hierarchy is walked to build each chunk's ancestor `heading_path`, but
/// chunks are emitted **flat**: a section owns the content between its heading
/// and the next heading of *any* level, so a parent never swallows its
/// subsections. That keeps one topic per vector, which is what makes the
/// heading path worth embedding (issue #2).
///
/// Splitting order, per section:
/// 1. whole section, if it fits in `target_tokens`
/// 2. otherwise pack whole paragraphs (blank-line separated) up to the budget
/// 3. a single paragraph over the budget is emitted whole
///
/// The heading line is re-emitted at the head of every piece, so no chunk is
/// ever labelled with a heading that begins partway through it. Sizes here use
/// the `chars/4` approximation; the real-token wall is enforced by
/// `split_oversized_chunks` downstream.
pub fn structure_chunk(content: &str, target_tokens: usize, opts: ChunkOptions) -> Vec<Chunk> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let offsets = line_offsets(content);
    let headings = structure_headings(content, opts.promote_bold);
    let line_start = |line: usize| offsets.get(line).copied().unwrap_or(content.len());

    let mut chunks = Vec::new();

    // Content before the first heading (frontmatter is already stripped).
    let first_heading = headings
        .first()
        .map(|h| line_start(h.line))
        .unwrap_or(content.len());
    emit_section(
        &content[..first_heading],
        SectionHeading {
            line: None,
            path: &[],
            carried: None,
        },
        target_tokens,
        opts.min_chars,
        &mut chunks,
    );

    // Ancestor stack of (level, heading text) for the heading path.
    let mut ancestors: Vec<(u8, String)> = Vec::new();
    // A heading line skipped for an empty body, waiting for the promoted
    // section that follows it. See the skip below.
    let mut carried: Option<String> = None;

    for (i, heading) in headings.iter().enumerate() {
        while ancestors
            .last()
            .is_some_and(|(level, _)| *level >= heading.level)
        {
            ancestors.pop();
        }
        ancestors.push((heading.level, heading.text.clone()));
        let path: Vec<String> = ancestors.iter().map(|(_, text)| text.clone()).collect();

        let heading_start = line_start(heading.line);
        let body_start = line_start(heading.line + 1);
        // Next heading of ANY level ends this section: subsections are siblings,
        // not children, for the purpose of chunk content.
        let body_end = headings
            .get(i + 1)
            .map(|next| line_start(next.line))
            .unwrap_or(content.len());

        let heading_line = content[heading_start..body_start].trim_end();
        let body = &content[body_start..body_end.max(body_start)];

        // A heading with no body of its own (immediately followed by a
        // subheading) would produce a chunk that is nothing but its own title.
        // Skip it — the text survives in its descendants' heading_path.
        //
        // A promoted line is an ancestor of nothing, so it carries no such
        // path, and a promoted section under `min_chars` merges into a chunk
        // that keeps the host's breadcrumb. The skipped heading would then
        // reach the corpus nowhere. Carry the line into the next section
        // instead, and only when that section is a promoted one: a heading
        // with `#` descendants keeps the behaviour above (issue #44).
        if body.trim().is_empty() {
            if headings.get(i + 1).is_some_and(|next| next.promoted) {
                carried = Some(heading_line.to_string());
            }
            continue;
        }

        emit_section(
            body,
            SectionHeading {
                line: Some(heading_line),
                path: &path,
                carried: carried.as_deref(),
            },
            target_tokens,
            opts.min_chars,
            &mut chunks,
        );
        carried = None;
    }

    chunks
}

/// What labels a section when it becomes a chunk.
struct SectionHeading<'a> {
    /// The section's own heading line, re-emitted at the head of every piece.
    line: Option<&'a str>,
    /// The ancestor breadcrumb, this section's own heading last.
    path: &'a [String],
    /// A heading line skipped for an empty body, carried here so that it stays
    /// in the corpus. It is emitted once, above `line`, at the head of the
    /// first piece only (issue #44).
    carried: Option<&'a str>,
}

/// Emit one or more chunks for a single section body, splitting on paragraph
/// boundaries. A paragraph over the budget, alone or combined with a folded
/// short leader, is emitted whole rather than torn.
///
/// A section whose body is shorter than `min_chars` is not a chunk of its own:
/// it joins the preceding chunk of the same file (issue #43). BM25 normalises
/// by row length, so a section holding one line scores enormously on any query
/// term it happens to carry, and the vault's template scaffolding — `## Rank`,
/// `## Threads`, `## Player Disposition` — is full of them.
///
/// The same minimum applies to a *piece* of a split section, not only to the
/// whole body. A short leading paragraph flushed by an oversized one would
/// otherwise become a stub row of its own. It is folded forward into the
/// following paragraph and the combination is emitted whole, so the heading
/// leads real section content (issue #51). The fold is forward, never
/// backward: a first piece's preceding chunk is a different section, so a
/// backward merge would orphan this section's heading at that section's tail.
fn emit_section(
    body: &str,
    heading: SectionHeading<'_>,
    target_tokens: usize,
    min_chars: usize,
    out: &mut Vec<Chunk>,
) {
    let SectionHeading {
        line: heading_line,
        path: heading_path,
        carried,
    } = heading;
    let body = body.trim();
    if body.is_empty() {
        return;
    }

    // Under the minimum, with somewhere to go. Length is counted the way
    // `approx_tokens` counts it, so the key's unit is the chunker's own.
    if body.len() < min_chars
        && let Some(host) = out.last_mut()
    {
        // The section's heading line travels with its body, so its terms stay
        // in `chunks.text` and therefore in the keyword index, which derives
        // from `chunks` (#37). Dropping them would be #11's bug class. A
        // carried line travels the same way, and for the same reason.
        let addition = match (carried, heading_line) {
            (Some(c), Some(h)) => format!("{c}\n{h}\n{body}"),
            (Some(c), None) => format!("{c}\n{body}"),
            (None, Some(h)) => format!("{h}\n{body}"),
            (None, None) => body.to_string(),
        };
        // These sections run in streaks, so the host is not grown past the
        // target: at the budget the stub becomes a chunk and hosts the rest.
        let merged = format!("{}\n\n{addition}", host.text);
        // Allow the merge to overrun the target by up to one shorty's worth
        // (min_chars), so a trailing short section with nothing after it to
        // host it merges rather than becoming a sub-minimum row (issue #75,
        // the #51 end-shorty residual). The unit is the chunker's own: a body
        // under min_chars chars is under min_chars/4 approx tokens.
        let shorty_slack = min_chars / 4;
        if approx_tokens(&merged) <= target_tokens + shorty_slack {
            // The host keeps its own heading and heading_path: no breadcrumb
            // is invented for the section that merged in.
            host.snippet = make_snippet(&merged);
            host.text = merged;
            return;
        }
    }

    // The heading is re-emitted on every piece, so it spends budget every time.
    // A carried line is emitted once and counted here too, so that no piece can
    // bust the budget.
    let heading_tokens =
        heading_line.map(approx_tokens).unwrap_or(0) + carried.map(approx_tokens).unwrap_or(0);
    let budget = target_tokens.saturating_sub(heading_tokens).max(1);

    if approx_tokens(body) <= budget {
        let mut chunk = Chunk::from_section(heading_line, heading_path, body, false);
        chunk.prepend_carried(carried);
        out.push(chunk);
        return;
    }

    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();

    for paragraph in body.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        // A short leader must not be flushed as a stub row of its own. When it
        // would be, fold it forward into the following paragraph so the size
        // split re-places it (issue #51). "Short" is a body under `min_chars`,
        // the same test the whole-section merge uses (#43); `min_chars == 0`
        // makes it unreachable and reproduces the pre-#51 pieces exactly.
        let short_leader = !current.is_empty() && current.len() < min_chars;

        // A single paragraph or table over the packing budget is emitted whole:
        // it is one coherent unit, so tearing it splits one theme across two
        // vectors. The real-token wall is enforced downstream in
        // `split_oversized_chunks` (issue #75).
        if approx_tokens(paragraph) > budget {
            let unit = if short_leader {
                format!("{}\n\n{paragraph}", std::mem::take(&mut current))
            } else {
                if !current.is_empty() {
                    pieces.push(std::mem::take(&mut current));
                }
                paragraph.to_string()
            };
            pieces.push(unit);
            continue;
        }

        let candidate = if current.is_empty() {
            paragraph.to_string()
        } else {
            format!("{current}\n\n{paragraph}")
        };
        if !current.is_empty() && approx_tokens(&candidate) > budget {
            if short_leader {
                // The short leader rides the front of this paragraph as one
                // whole piece rather than being size-split (issue #51, #75).
                pieces.push(candidate);
                current.clear();
            } else {
                pieces.push(std::mem::replace(&mut current, paragraph.to_string()));
            }
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }

    for (i, piece) in pieces.into_iter().enumerate() {
        let mut chunk = Chunk::from_section(heading_line, heading_path, &piece, i > 0);
        if i == 0 {
            chunk.prepend_carried(carried);
        }
        out.push(chunk);
    }
}

/// Every number that decides where a chunk boundary falls, or how much of the
/// previous chunk a continuation repeats.
///
/// They are named rather than inline because `chunker_fingerprint` hashes them
/// (issue #31): a store built at one set of these values holds different chunks
/// — and therefore different vectors and different FTS rows — than the same
/// vault indexed at another, and nothing else in the database records which set
/// was used. Changing one here is enough to make the next `engraph index`
/// rebuild; there is no second place to remember to edit.
pub mod limits {
    /// Chunk size the break-point search aims for.
    pub const TARGET_TOKENS: usize = 512;
    /// Tokens of the previous sub-chunk repeated at the head of each piece a
    /// [`super::split_oversized_chunks`] split produces.
    pub const OVERLAP_TOKENS: usize = 50;
}

pub use limits::{OVERLAP_TOKENS, TARGET_TOKENS};

/// Parse markdown content into frontmatter tags and structure-first chunks.
///
/// 1. Strip YAML frontmatter (between `---` at start), parse `tags` if present.
/// 2. Run `structure_chunk` on the body at [`TARGET_TOKENS`].
/// 3. Return `ParsedMarkdown { tags, chunks }`.
///
/// `opts` carries `[chunk_min_chars]` and `[promote_bold_headings]`, the config
/// keys that decide where a chunk boundary falls (see [`ChunkOptions`]). It is
/// a parameter rather than constants because they are config keys, and every
/// caller must pass the same pair: two paths chunking a file at different
/// settings write two different sets of rows.
pub fn chunk_markdown(content: &str, opts: ChunkOptions) -> ParsedMarkdown {
    let (tags, body) = parse_frontmatter(content);

    let chunks = structure_chunk(body, TARGET_TOKENS, opts);

    ParsedMarkdown { tags, chunks }
}

/// Split oversized chunks into sub-chunks that fit within `max_tokens`.
///
/// This is the sole place a block is torn. `structure_chunk` packs whole
/// paragraphs to [`TARGET_TOKENS`] and emits a single over-budget block whole;
/// `max_tokens` here is the embed model's real input wall
/// ([`crate::llm::EmbedModel::max_context`]), not that packing target, so a
/// call site passes the model's own ceiling, not a fixed constant.
///
/// - `token_count` counts tokens in a string (closure for testability).
/// - Chunks under `max_tokens` pass through unchanged.
/// - Over-sized chunks are split on sentence boundaries (`. ` or `\n`).
/// - Each sub-chunk after the first includes `overlap_tokens` worth of trailing
///   text from the previous sub-chunk.
/// - Subsequent sub-chunks get ` (cont.)` appended to the parent heading.
pub fn split_oversized_chunks(
    chunks: Vec<Chunk>,
    token_count: &dyn Fn(&str) -> usize,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<Chunk> {
    let mut result = Vec::new();
    for chunk in chunks {
        if token_count(&chunk.text) <= max_tokens {
            result.push(chunk);
            continue;
        }
        // Split text into sentences on `. ` or `\n`
        let sentences = split_sentences(&chunk.text);
        let mut sub_chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for sentence in &sentences {
            let candidate = if current.is_empty() {
                sentence.to_string()
            } else {
                format!("{current}{sentence}")
            };
            if !current.is_empty() && token_count(&candidate) > max_tokens {
                // Flush current sub-chunk
                sub_chunks.push(current.clone());
                // Build overlap prefix from the end of the previous sub-chunk
                let overlap = build_overlap(&current, token_count, overlap_tokens);
                current = format!("{overlap}{sentence}");
            } else {
                current = candidate;
            }
        }
        if !current.trim().is_empty() {
            sub_chunks.push(current);
        }

        // Convert sub-chunks into Chunk structs
        for (i, sub_text) in sub_chunks.into_iter().enumerate() {
            let heading = if i == 0 {
                chunk.heading.clone()
            } else {
                chunk.heading.as_ref().map(|h| format!("{h} (cont.)"))
            };
            let sub_text = sub_text.trim();
            let full_text = match &heading {
                // Structure-first chunks already lead with their heading; only
                // prepend when it isn't there, or the first piece gets it twice.
                Some(h) if !sub_text.starts_with(h.as_str()) => format!("{h}\n{sub_text}"),
                _ => sub_text.to_string(),
            };
            let snippet = make_snippet(&full_text);
            result.push(Chunk {
                heading,
                heading_path: chunk.heading_path.clone(),
                text: full_text,
                snippet,
            });
        }
    }
    result
}

/// Split text into sentence-like segments, preserving delimiters.
/// Splits on `. ` (sentence end) and `\n` (line break).
fn split_sentences(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        current.push(chars[i]);
        if chars[i] == '\n' {
            segments.push(current.clone());
            current.clear();
        } else if chars[i] == '.' && i + 1 < chars.len() && chars[i + 1] == ' ' {
            current.push(' ');
            i += 1; // consume the space
            segments.push(current.clone());
            current.clear();
        }
        i += 1;
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Build an overlap string from the end of `text` that is approximately
/// `overlap_tokens` tokens long, measured by `token_count`.
fn build_overlap(text: &str, token_count: &dyn Fn(&str) -> usize, overlap_tokens: usize) -> String {
    if overlap_tokens == 0 {
        return String::new();
    }
    // Work backwards through words to build overlap
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut overlap = String::new();
    for &word in words.iter().rev() {
        let candidate = if overlap.is_empty() {
            word.to_string()
        } else {
            format!("{word} {overlap}")
        };
        if token_count(&candidate) > overlap_tokens {
            break;
        }
        overlap = candidate;
    }
    if overlap.is_empty() {
        overlap
    } else {
        format!("{overlap} ")
    }
}

/// The leading 200 characters of `text`, with `"..."` if that cut anything.
///
/// [`crate::store::Store::insert_chunk`] calls this too, so that a stored
/// chunk's `snippet` column is the same derivation as a [`Chunk`]'s field
/// rather than a second one that could drift from it.
pub(crate) fn make_snippet(text: &str) -> String {
    if text.len() > 200 {
        let truncated: String = text.chars().take(200).collect();
        format!("{truncated}...")
    } else {
        text.to_string()
    }
}

/// Parse YAML frontmatter and return (tags, body_without_frontmatter).
fn parse_frontmatter(content: &str) -> (Vec<String>, &str) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (Vec::new(), content);
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    let after_first = after_first.trim_start_matches('-'); // handle "----"
    let after_first = after_first.strip_prefix('\n').unwrap_or(after_first);

    if let Some(end_pos) = after_first.find("\n---") {
        let yaml_block = &after_first[..end_pos];
        let body_start = end_pos + 4; // skip "\n---"
        let body = after_first[body_start..]
            .strip_prefix('\n')
            .unwrap_or(&after_first[body_start..]);
        let tags = parse_tags_from_yaml(yaml_block);
        (tags, body)
    } else {
        (Vec::new(), content)
    }
}

/// Parse `tags` field from a YAML block. Supports:
/// - `tags: [a, b, c]`
/// - `tags:\n  - a\n  - b`
fn parse_tags_from_yaml(yaml: &str) -> Vec<String> {
    let lines: Vec<&str> = yaml.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("tags:") {
            let after_colon = trimmed.strip_prefix("tags:").unwrap().trim();
            // Inline list: tags: [a, b]
            if after_colon.starts_with('[') {
                let inner = after_colon.trim_start_matches('[').trim_end_matches(']');
                return inner
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            // If there's content after colon on same line (single tag)
            if !after_colon.is_empty() {
                return vec![after_colon.to_string()];
            }
            // Block list: tags:\n  - a\n  - b
            let mut tags = Vec::new();
            for subsequent in &lines[i + 1..] {
                let st = subsequent.trim();
                if st.starts_with("- ") {
                    tags.push(st.strip_prefix("- ").unwrap().trim().to_string());
                } else if st.is_empty() {
                    continue;
                } else {
                    break;
                }
            }
            return tags;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Break-point detection tests ──────────────────────────────────────

    #[test]
    fn test_find_break_points() {
        let content = "# Title\n\nSome text\n\n## Section\nContent\n### Sub\nMore\n\n---\n";
        let bps = find_break_points(content);

        // Collect (line_number, score) pairs for easy assertion
        let pairs: Vec<(usize, u32)> = bps.iter().map(|bp| (bp.line_number, bp.score)).collect();

        // # Title -> 100
        assert!(
            pairs.contains(&(0, 100)),
            "Expected # heading at line 0 with score 100, got: {:?}",
            pairs
        );
        // empty line -> 20
        assert!(
            pairs.contains(&(1, 20)),
            "Expected empty line at line 1 with score 20"
        );
        // empty line -> 20
        assert!(
            pairs.contains(&(3, 20)),
            "Expected empty line at line 3 with score 20"
        );
        // ## Section -> 90
        assert!(
            pairs.contains(&(4, 90)),
            "Expected ## heading at line 4 with score 90"
        );
        // ### Sub -> 80
        assert!(
            pairs.contains(&(6, 80)),
            "Expected ### heading at line 6 with score 80"
        );
        // empty line -> 20
        assert!(
            pairs.contains(&(8, 20)),
            "Expected empty line at line 8 with score 20"
        );
        // --- -> 60
        assert!(
            pairs.contains(&(9, 60)),
            "Expected thematic break at line 9 with score 60"
        );

        // "Some text", "Content", "More" have score 1 and should NOT appear
        // (only lines inside code fences get score 1 in results)
        for bp in &bps {
            assert!(
                bp.score > 1 || bp.inside_code_fence,
                "Non-fence break points should not include lines with score <= 1"
            );
        }
    }

    #[test]
    fn test_find_break_points_code_fence() {
        let content = "Before\n\n```rust\nlet x = 1;\nlet y = 2;\n```\n\nAfter\n";
        let bps = find_break_points(content);

        // The opening ``` should be a break point with score 80, NOT inside fence
        let opening = bps.iter().find(|bp| bp.line_number == 2).unwrap();
        assert_eq!(opening.score, 80);
        assert!(
            !opening.inside_code_fence,
            "Opening fence should not be marked as inside"
        );

        // The closing ``` should be a break point with score 80, NOT inside fence
        // (it toggles the fence off)
        let closing = bps.iter().find(|bp| bp.line_number == 5).unwrap();
        assert_eq!(closing.score, 80);
        assert!(
            !closing.inside_code_fence,
            "Closing fence should not be marked as inside"
        );

        // Lines inside the fence (let x = 1; let y = 2;) SHOULD appear with inside_code_fence = true
        let inside_bps: Vec<&BreakPoint> = bps
            .iter()
            .filter(|bp| bp.line_number == 3 || bp.line_number == 4)
            .collect();
        assert_eq!(
            inside_bps.len(),
            2,
            "Expected 2 break points inside code fence"
        );
        for bp in &inside_bps {
            assert!(
                bp.inside_code_fence,
                "Line {} inside fence should have inside_code_fence=true",
                bp.line_number
            );
            assert_eq!(
                bp.score, 1,
                "Line {} inside fence should have score 1",
                bp.line_number
            );
        }
    }

    #[test]
    fn test_find_break_points_list_items() {
        let content = "- item one\n* item two\n1. numbered\nplain text\n";
        let bps = find_break_points(content);
        let pairs: Vec<(usize, u32)> = bps.iter().map(|bp| (bp.line_number, bp.score)).collect();
        assert!(
            pairs.contains(&(0, 5)),
            "Expected list item at line 0 with score 5"
        );
        assert!(
            pairs.contains(&(1, 5)),
            "Expected list item at line 1 with score 5"
        );
        assert!(
            pairs.contains(&(2, 5)),
            "Expected numbered list item at line 2 with score 5"
        );
        // "plain text" has score 1, should NOT appear
        assert!(
            !bps.iter().any(|bp| bp.line_number == 3),
            "Plain text should not be a break point"
        );
    }

    // ── Smart chunk tests ────────────────────────────────────────────────

    #[test]
    fn test_smart_chunk_single() {
        // Short content should produce a single chunk
        let content = "# Hello\nSome short content here.";
        let chunks = smart_chunk(content, 512, 15);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Hello"));
        assert!(chunks[0].text.contains("short content"));
    }

    #[test]
    fn test_smart_chunk_splits_large_content() {
        // Build content larger than 512 tokens (~2048 chars)
        let mut content = String::new();
        content.push_str("# Introduction\n\n");
        for i in 0..30 {
            content.push_str(&format!(
                "## Section {}\nThis is paragraph {} with enough text to take up space. \
                 We need each section to have meaningful content so the chunker has \
                 good break points to choose from.\n\n",
                i, i
            ));
        }

        let chunks = smart_chunk(&content, 512, 15);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks for large content, got {}",
            chunks.len()
        );

        // Each chunk should have a snippet
        for c in &chunks {
            assert!(!c.snippet.is_empty());
        }
    }

    #[test]
    fn test_smart_chunk_no_overlap_crawl() {
        // Regression for the overlap-vs-stride crawl: when a high-score break
        // (heading) lands within the overlap window of a chunk start, the old
        // advance logic stepped the start forward one character at a time,
        // re-selecting the same break and emitting hundreds of near-duplicate
        // sub-chunks. A doc of many short headed sections must still produce a
        // bounded, sane chunk count with no degenerate micro-chunks.
        let mut content = String::new();
        content.push_str("# Title\n\n");
        for i in 0..40 {
            content.push_str(&format!(
                "## Section {i}\nA moderate paragraph of prose for section {i} that carries \
                 real content but is shorter than the target chunk size, so the following \
                 heading falls inside the overlap window.\n\n"
            ));
        }
        let chunks = smart_chunk(&content, 512, 15);
        // ~9k chars at a 2048-char target → a healthy handful of chunks, never hundreds.
        assert!(
            chunks.len() < 40,
            "overlap-crawl regression: expected a bounded chunk count, got {}",
            chunks.len()
        );
        // The crawl produced ~1-token fragments; real chunks are substantial.
        for c in &chunks {
            assert!(
                c.text.len() > 20,
                "degenerate micro-chunk produced ({} chars): {:?}",
                c.text.len(),
                c.text
            );
        }
    }

    #[test]
    fn test_smart_chunk_empty() {
        let chunks = smart_chunk("", 512, 15);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_smart_chunk_whitespace_only() {
        let chunks = smart_chunk("   \n\n  \n", 512, 15);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_code_fence_protection() {
        // Content with a code block that should NOT be split
        let mut content = String::new();
        content.push_str("# Before Code\nSome intro text.\n\n");
        content.push_str("```python\n");
        for i in 0..50 {
            content.push_str(&format!("x_{} = compute_value({})\n", i, i));
        }
        content.push_str("```\n\n");
        content.push_str("# After Code\nSome conclusion.\n");

        let bps = find_break_points(&content);
        // Verify no break points inside the code fence are eligible (not inside_code_fence)
        let fence_start_line = 3; // ```python
        let fence_end_line = fence_start_line + 51; // ``` closing

        for bp in &bps {
            if bp.line_number > fence_start_line && bp.line_number < fence_end_line {
                // These should either not exist or be marked inside_code_fence
                assert!(
                    bp.inside_code_fence || bp.score <= 1,
                    "Break point at line {} (score {}) should be inside code fence or excluded",
                    bp.line_number,
                    bp.score
                );
            }
        }
    }

    // ── Structure-first chunking tests ───────────────────────────────────

    /// `ChunkOptions` at a given minimum, with promotion off — the settings
    /// every test written before #44 was written against.
    fn opts(min_chars: usize) -> ChunkOptions {
        ChunkOptions {
            min_chars,
            promote_bold: false,
        }
    }

    #[test]
    fn test_structure_chunk_one_chunk_per_section() {
        let md = "## Alpha\nA body.\n\n## Beta\nB body.\n\n## Gamma\nG body.\n";
        let chunks = structure_chunk(md, 512, opts(0));
        assert_eq!(chunks.len(), 3);
        for (chunk, expected) in chunks.iter().zip(["Alpha", "Beta", "Gamma"]) {
            assert_eq!(chunk.heading.as_deref(), Some(&*format!("## {expected}")));
            assert_eq!(chunk.heading_path, vec![expected.to_string()]);
        }
    }

    #[test]
    fn test_structure_chunk_no_chunk_spans_two_sections() {
        // Six tiny sections: size-driven chunking merged all of these into one.
        let md: String = (0..6)
            .map(|i| format!("## Section {i}\nBody of section {i}.\n\n"))
            .collect();
        let chunks = structure_chunk(&md, 512, opts(0));

        assert_eq!(chunks.len(), 6);
        for chunk in &chunks {
            let headings = chunk
                .text
                .lines()
                .filter(|l| l.trim_start().starts_with("## "))
                .count();
            assert_eq!(headings, 1, "chunk spans sections:\n{}", chunk.text);
        }
    }

    #[test]
    fn test_structure_chunk_heading_always_starts_chunk() {
        // Long enough to force a split inside a section.
        let filler = "Sentence of prose padding this section out. ".repeat(60);
        let md = format!("## First\n{filler}\n\n{filler}\n\n## Second\nShort.\n");
        let chunks = structure_chunk(&md, 128, opts(0));

        assert!(chunks.len() > 2, "expected the first section to split");
        for chunk in &chunks {
            let heading = chunk.heading.as_deref().expect("every chunk is labelled");
            // The label is the chunk's first line, never a heading discovered
            // partway through the text.
            let base = heading.trim_end_matches(" (cont.)");
            assert!(
                chunk.text.starts_with(base),
                "chunk labelled {heading:?} does not begin with it:\n{}",
                chunk.text
            );
        }
        // Continuations are marked, and the section that fits is not.
        assert_eq!(chunks[0].heading.as_deref(), Some("## First"));
        assert_eq!(chunks[1].heading.as_deref(), Some("## First (cont.)"));
        assert_eq!(chunks.last().unwrap().heading.as_deref(), Some("## Second"));
    }

    #[test]
    fn test_structure_chunk_packs_whole_paragraphs() {
        // Three paragraphs, each ~40 approx-tokens; a 60-token budget must pack
        // them one per chunk rather than cutting a paragraph in half.
        let para = |n: usize| format!("Paragraph {n} {}", "word ".repeat(30));
        let md = format!("## Body\n{}\n\n{}\n\n{}\n", para(1), para(2), para(3));
        let chunks = structure_chunk(&md, 64, opts(0));

        assert_eq!(chunks.len(), 3);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.contains(&format!("Paragraph {}", i + 1)),
                "chunk {i} lost its paragraph:\n{}",
                chunk.text
            );
        }
    }

    #[test]
    fn a_single_oversized_paragraph_is_emitted_whole() {
        // One paragraph well over the 512 chars/4 budget, no blank lines inside.
        let para = "word ".repeat(700); // ~3500 chars ~= 875 approx tokens
        let md = format!("## Note\n{para}");
        let chunks = structure_chunk(&md, 512, opts(0));
        assert_eq!(chunks.len(), 1, "an atomic block must not be torn here");
        assert!(chunks[0].text.contains("## Note"));
    }

    #[test]
    fn a_trailing_short_section_merges_into_the_previous_chunk() {
        // A full first section, then a section whose body is under min_chars with
        // nothing after it to host it.
        let big = "sentence. ".repeat(60); // comfortably over min_chars
        let md = format!("## Body\n{big}\n\n## Coda\nshort tail line");
        let chunks = structure_chunk(&md, 512, opts(120));
        assert!(
            chunks
                .iter()
                .all(|c| c.text.len() >= 120 || c.text.contains("## Body")),
            "no non-first chunk may be under min_chars"
        );
        assert!(
            chunks.last().unwrap().text.contains("short tail line"),
            "the trailing shorty is kept, merged into the body chunk"
        );
    }

    #[test]
    fn a_short_leader_and_a_fitting_paragraph_emit_as_one_whole_piece() {
        // A short leader paragraph (under min_chars) followed by a second
        // paragraph that fits the budget alone but busts it combined with the
        // leader. The combination must still land as one whole piece: no
        // content lost, no sub-minimum row produced.
        let leader = "Short lead-in.";
        assert!(leader.len() < 120, "leader must be under min_chars");
        // Sized so leader + blank lines + this paragraph together exceed the
        // 64-token budget (64 approx tokens), but the paragraph alone does not
        // (60 approx tokens, budget 61 after the heading's share).
        let second = "distinctiveword ".repeat(15);
        let md = format!("## Overview\n{leader}\n\n{second}");
        let chunks = structure_chunk(&md, 64, opts(120));

        assert_eq!(
            chunks.len(),
            1,
            "the leader and the paragraph must emit as one whole piece, got {}",
            chunks.len()
        );
        assert!(
            chunks[0].text.contains("Short lead-in."),
            "the leader's words were dropped:\n{}",
            chunks[0].text
        );
        assert!(
            chunks[0].text.contains("distinctiveword"),
            "the paragraph's words were dropped:\n{}",
            chunks[0].text
        );
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                i == 0 || c.text.len() >= 120,
                "chunk {i} is a {}-char stub:\n{}",
                c.text.len(),
                c.text
            );
        }
    }

    #[test]
    fn an_oversized_single_paragraph_is_emitted_whole() {
        // A single paragraph — no blank lines to split on — well over budget.
        // It is emitted as one whole chunk; the real-token wall is enforced
        // downstream by `split_oversized_chunks` (issue #75).
        let giant = "This is one very long unbroken paragraph. ".repeat(80);
        let md = format!("## Wall\n{giant}");
        let chunks = structure_chunk(&md, 64, opts(0));

        assert_eq!(
            chunks.len(),
            1,
            "an atomic block must not be torn here, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].heading.as_deref(), Some("## Wall"));
        assert_eq!(chunks[0].heading_path, vec!["Wall".to_string()]);
        // No content is lost.
        assert!(chunks[0].text.contains("one very long unbroken paragraph"));
        assert!(chunks[0].text.trim_end().ends_with("paragraph."));
    }

    #[test]
    fn test_structure_chunk_subsections_are_siblings_with_ancestor_path() {
        let md = "## Abilities\nOverview text.\n\n### Combat\nSword work.\n\n### Magic\nSpell work.\n\n## Gear\nA sword.\n";
        let chunks = structure_chunk(md, 512, opts(0));

        assert_eq!(chunks.len(), 4);
        // A parent does not swallow its subsections...
        assert_eq!(chunks[0].text, "## Abilities\nOverview text.");
        // ...but each subsection knows its ancestry.
        assert_eq!(
            chunks[1].heading_path,
            vec!["Abilities".to_string(), "Combat".to_string()]
        );
        assert_eq!(
            chunks[2].heading_path,
            vec!["Abilities".to_string(), "Magic".to_string()]
        );
        // Popping back to a shallower level resets the path.
        assert_eq!(chunks[3].heading_path, vec!["Gear".to_string()]);
    }

    #[test]
    fn test_structure_chunk_skips_bodyless_heading() {
        let md = "## Parent\n### Child\nOnly real content.\n";
        let chunks = structure_chunk(md, 512, opts(0));

        // `## Parent` has no body of its own, so it produces no title-only chunk.
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading.as_deref(), Some("### Child"));
        assert_eq!(
            chunks[0].heading_path,
            vec!["Parent".to_string(), "Child".to_string()]
        );
    }

    #[test]
    fn test_structure_chunk_preamble_before_first_heading() {
        let md = "Intro prose with no heading.\n\n## Section\nBody.\n";
        let chunks = structure_chunk(md, 512, opts(0));

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].heading.is_none());
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!(chunks[0].text, "Intro prose with no heading.");
        assert_eq!(chunks[1].heading.as_deref(), Some("## Section"));
    }

    /// A section body comfortably over any minimum these tests set, so it is a
    /// host and never a stub.
    fn host_body() -> String {
        "Real prose in a section that stands on its own. ".repeat(4)
    }

    #[test]
    fn test_structure_chunk_stub_merges_into_the_preceding_chunk() {
        let md = format!(
            "## Stat Block\n{}\n\n## Threads\n_None yet._\n",
            host_body()
        );
        let chunks = structure_chunk(&md, 512, opts(120));

        assert_eq!(chunks.len(), 1);
        // The host's own label and breadcrumb: no breadcrumb is invented.
        assert_eq!(chunks[0].heading.as_deref(), Some("## Stat Block"));
        assert_eq!(chunks[0].heading_path, vec!["Stat Block".to_string()]);
        // The stub's heading line stays in the text, so its terms are still in
        // `chunks.text` and therefore in the keyword index (#37). Dropping them
        // would be #11's bug class.
        assert!(
            chunks[0].text.contains("## Threads"),
            "the stub's heading was dropped:\n{}",
            chunks[0].text
        );
        assert!(chunks[0].text.contains("_None yet._"));
    }

    #[test]
    fn test_structure_chunk_merged_chunk_keeps_its_snippet_derived() {
        let md = format!(
            "## Stat Block\n{}\n\n## Threads\n_None yet._\n",
            host_body()
        );
        let chunks = structure_chunk(&md, 512, opts(120));

        assert_eq!(chunks[0].snippet, make_snippet(&chunks[0].text));
    }

    #[test]
    fn test_structure_chunk_stub_with_no_preceding_chunk_stays() {
        // A file that is one short section is one chunk. Dropping it would make
        // the file unfindable, which is not the case a minimum is for.
        let md = "## Threads\n_None yet._\n";
        let chunks = structure_chunk(md, 512, opts(120));

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "## Threads\n_None yet._");
        assert_eq!(chunks[0].heading.as_deref(), Some("## Threads"));
    }

    #[test]
    fn test_structure_chunk_body_at_the_minimum_is_not_a_stub() {
        let exact = "x".repeat(120);
        let md = format!("## Host\n{}\n\n## Exact\n{exact}\n", host_body());
        let chunks = structure_chunk(&md, 512, opts(120));

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].heading.as_deref(), Some("## Exact"));
    }

    #[test]
    fn test_structure_chunk_stub_streak_starts_a_new_chunk_at_the_budget() {
        // `## Rank`, `## Threads` and `## Player Disposition` run consecutively
        // in the same file, so a streak must not grow one chunk without limit.
        let stubs: String = (0..12)
            .map(|i| format!("\n\n## Stub {i}\nNone yet.\n"))
            .collect();
        let md = format!("## Host\n{}{stubs}", host_body());
        let target_tokens = 64;
        let min_chars = 120;
        let chunks = structure_chunk(&md, target_tokens, opts(min_chars));

        assert!(
            chunks.len() > 1 && chunks.len() < 13,
            "expected the streak to collapse into bounded chunks, got {}",
            chunks.len()
        );
        // The merge may overrun the target by up to one shorty's worth of
        // headroom (issue #75), never past it.
        let bound = target_tokens + min_chars / 4;
        for chunk in &chunks {
            assert!(
                approx_tokens(&chunk.text) <= bound,
                "merged chunk busts the target plus shorty headroom ({bound}):\n{}",
                chunk.text
            );
        }
    }

    #[test]
    fn test_structure_chunk_minimum_of_zero_keeps_every_section() {
        let md = "## Alpha\nA body.\n\n## Threads\n_None yet._\n";
        let chunks = structure_chunk(md, 512, opts(0));

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].text, "## Threads\n_None yet._");
    }

    #[test]
    fn a_short_leader_of_a_split_section_folds_forward_not_into_a_stub() {
        // `## NPC Activity` opens with a one-line lead-in, then a bullet list
        // that alone busts the budget. The split flushed the lead-in as a row
        // of its own — under the minimum, and not the file's first chunk. It
        // must instead ride the front of the section's first real chunk (#51).
        let leader = "**Tandi** — the session's sole active NPC.";
        let bullets: String = (0..40)
            .map(|i| {
                format!("- Bullet {i:02} records an action the NPC took this session, in full.\n")
            })
            .collect();
        let md = format!(
            "## Overview\n{}\n\n## NPC Activity\n{leader}\n\n{bullets}",
            host_body()
        );
        let chunks = structure_chunk(&md, 512, opts(120));

        // The section's first piece keeps the plain heading and carries both the
        // lead-in and real section content: the leader was folded forward, not
        // emitted as a stub.
        let npc = chunks
            .iter()
            .find(|c| c.heading.as_deref() == Some("## NPC Activity"))
            .expect("the section's first piece keeps the plain heading");
        assert!(
            npc.text.contains("sole active NPC"),
            "the lead-in was dropped:\n{}",
            npc.text
        );
        assert!(
            npc.text.contains("Bullet 00"),
            "the lead-in is a stub, not folded into the following content:\n{}",
            npc.text
        );

        // No chunk but the file's first falls under the minimum.
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                i == 0 || c.text.len() >= 120,
                "chunk {i} is a {}-char stub:\n{}",
                c.text.len(),
                c.text
            );
        }
    }

    #[test]
    fn test_chunk_markdown_carries_the_minimum_past_the_frontmatter() {
        let md = format!(
            "---\ntags: [beast]\n---\n\n## Stat Block\n{}\n\n## Threads\n_None yet._\n",
            host_body()
        );
        let parsed = chunk_markdown(&md, opts(120));

        assert_eq!(parsed.tags, vec!["beast".to_string()]);
        assert_eq!(parsed.chunks.len(), 1);
        assert!(parsed.chunks[0].text.contains("## Threads"));
    }

    #[test]
    fn test_structure_chunk_ignores_headings_in_code_fences() {
        let md = "## Real\nSee below:\n\n```md\n## Not A Heading\n```\n\n## Also Real\nBody.\n";
        let chunks = structure_chunk(md, 512, opts(0));

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("## Real"));
        assert!(chunks[0].text.contains("## Not A Heading"));
        assert_eq!(chunks[1].heading.as_deref(), Some("## Also Real"));
    }

    #[test]
    fn test_structure_chunk_handles_multibyte_headings() {
        // Byte offsets, not char counts: a mis-slice here would panic.
        let md = "## Ríoghán's Résumé\nBody — with an em dash.\n\n## 日本語\n本文です。\n";
        let chunks = structure_chunk(md, 512, opts(0));

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("## Ríoghán's Résumé"));
        assert_eq!(chunks[1].heading.as_deref(), Some("## 日本語"));
        assert!(chunks[1].text.contains("本文です。"));
    }

    /// A section body of `n` characters, so a test states its own size against
    /// `min_chars` rather than depending on a fixture's length.
    fn body(n: usize) -> String {
        let mut s = String::new();
        while s.len() < n {
            s.push_str("The dragon hoards gold in the deep places of the world. ");
        }
        s.truncate(n);
        s
    }

    fn promoting(min_chars: usize) -> ChunkOptions {
        ChunkOptions {
            min_chars,
            promote_bold: true,
        }
    }

    #[test]
    fn a_bold_only_line_opens_a_section() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\n{}\n",
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading_path, vec!["Stat Block"]);
        assert_eq!(chunks[1].heading_path, vec!["Stat Block", "Spells"]);
    }

    /// The chunker's set is the parser's with the bodyless promotions
    /// dropped, and that is the whole of the difference between them. A
    /// promoted line with no body starts no chunk, and `find_section` still
    /// addresses it (#69).
    #[test]
    fn the_chunkers_set_drops_a_bodyless_promotion_the_parser_keeps() {
        let md = "## Stat Block\n\n**Spells**\n**Notes**\n\nSee below\n";
        let chunker_headings = structure_headings(md, true);
        let chunker: Vec<&str> = chunker_headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(chunker, vec!["Stat Block", "Notes"]);
        let parser_headings = crate::markdown::headings_with_promotions(md);
        let parser: Vec<&str> = parser_headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(parser, vec!["Stat Block", "Spells", "Notes"]);
    }

    #[test]
    fn a_promoted_line_keeps_its_own_text_in_the_chunk() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\n{}\n",
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert!(chunks[1].text.starts_with("**Spells**"));
        assert_eq!(chunks[1].heading.as_deref(), Some("**Spells**"));
    }

    #[test]
    fn promoted_lines_are_siblings_and_do_not_nest() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\n{}\n\n**Notes**\n{}\n",
            body(200),
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].heading_path, vec!["Stat Block", "Spells"]);
        assert_eq!(chunks[2].heading_path, vec!["Stat Block", "Notes"]);
    }

    #[test]
    fn a_promoted_line_before_any_heading_is_a_top_level_section() {
        let md = format!(
            "{}\n\n**Summary**\n{}\n\n## Stat Block\n{}\n",
            body(200),
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!(chunks[1].heading_path, vec!["Summary"]);
        assert_eq!(chunks[2].heading_path, vec!["Stat Block"]);
    }

    #[test]
    fn a_later_heading_of_any_depth_ends_a_promoted_section() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\n{}\n\n### Notes\n{}\n",
            body(200),
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].heading_path, vec!["Stat Block", "Spells"]);
        // A promoted line is an ancestor of nothing, so the deeper heading
        // hangs off the enclosing `##` and not off the bold line above it.
        assert_eq!(chunks[2].heading_path, vec!["Stat Block", "Notes"]);
    }

    #[test]
    fn a_promoted_line_with_no_body_stays_in_the_enclosing_section() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\n\n**Notes**\n{}\n",
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 2);
        // The line is not a section, and it is not lost either: a flat promoted
        // line has no descendant to carry it in a heading_path.
        assert!(chunks[0].text.contains("**Spells**"));
        assert_eq!(chunks[1].heading_path, vec!["Stat Block", "Notes"]);
    }

    #[test]
    fn a_promoted_section_under_the_minimum_merges_into_the_preceding_chunk() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\nN/A\n\n**Notes**\n{}\n",
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        // The well-bodied `**Notes**` section proves promotion ran at all; if
        // it did not, the whole document would still be one `## Stat Block`
        // chunk and this assertion would fail.
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].heading_path, vec!["Stat Block", "Notes"]);
        // The under-minimum `**Spells**` section merged into the chunk before it.
        assert_eq!(chunks[0].heading_path, vec!["Stat Block"]);
        assert!(chunks[0].text.contains("**Spells**\nN/A"));
    }

    #[test]
    fn a_heading_promotion_empties_survives_in_the_merged_chunk() {
        // `## Spells` has no body of its own once `**Spells**` opens a section,
        // and that section is under the minimum, so it merges into the chunk
        // before it and the host's breadcrumb wins. Without the carry the
        // heading would be in no chunk's text and in no chunk's heading_path.
        let md = format!(
            "## Abilities\n{}\n\n## Spells\n**Spells**\nN/A\n",
            body(350)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, vec!["Abilities"]);
        assert!(chunks[0].text.contains("## Spells\n**Spells**\nN/A"));

        // The control merges the whole `## Spells` section into the chunk
        // before it, heading line included, so the corpus holds the heading at
        // both settings.
        let control = structure_chunk(&md, 512, opts(120));
        assert_eq!(control.len(), 1);
        assert!(control[0].text.contains("## Spells\n**Spells**\nN/A"));
    }

    #[test]
    fn a_heading_promotion_empties_survives_in_a_standalone_chunk() {
        // The same shape, with a promoted body big enough to be its own chunk.
        let md = format!(
            "## Abilities\n{}\n\n## Spells\n**Spells**\n{}\n",
            body(350),
            body(350)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[1].text.starts_with("## Spells\n**Spells**"));
        // The carried line labels nothing: the chunk is still the promoted one.
        assert_eq!(chunks[1].heading.as_deref(), Some("**Spells**"));
        assert_eq!(chunks[1].heading_path, vec!["Spells", "Spells"]);
        assert!(chunks[1].snippet.starts_with("## Spells"));
    }

    #[test]
    fn a_bodyless_heading_with_hash_descendants_is_still_skipped() {
        // Its text survives in the descendants' heading_path, so it must keep
        // behaving as it did before the carry — at both settings.
        let md = format!("## Spells\n### Level 1\n{}\n", body(350));
        for options in [promoting(120), opts(120)] {
            let chunks = structure_chunk(&md, 512, options);
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].heading_path, vec!["Spells", "Level 1"]);
            assert!(chunks[0].text.starts_with("### Level 1"));
            assert!(!chunks[0].text.contains("## Spells"));
        }
    }

    #[test]
    fn a_bold_line_in_a_code_fence_is_not_promoted() {
        let md = format!(
            "## Stat Block\n{}\n\n```\n**Spells**\n```\n{}\n\n**Notes**\n{}\n",
            body(200),
            body(200),
            body(200)
        );
        let chunks = structure_chunk(&md, 512, promoting(120));
        // The trailing `**Notes**` section proves promotion was active; the
        // fenced `**Spells**` line still did not open a section of its own.
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].heading_path, vec!["Stat Block", "Notes"]);
        assert_eq!(chunks[0].heading_path, vec!["Stat Block"]);
    }

    #[test]
    fn promotion_off_reproduces_the_previous_chunking() {
        let md = format!(
            "## Stat Block\n{}\n\n**Spells**\n{}\n\n**Notes**\n{}\n",
            body(200),
            body(200),
            body(200)
        );
        let off = structure_chunk(&md, 512, opts(120));
        assert_eq!(off.len(), 1);
        assert_eq!(off[0].heading_path, vec!["Stat Block"]);
    }

    #[test]
    fn test_split_oversized_chunks_does_not_duplicate_heading() {
        // Structure-first chunks already lead with their heading; the token-aware
        // pass must not prepend it a second time.
        let chunk = Chunk {
            heading: Some("## Section".to_string()),
            heading_path: vec!["Section".to_string()],
            text: "## Section\nShort body.".to_string(),
            snippet: "## Section\nShort body.".to_string(),
        };
        let token_fn = |s: &str| s.split_whitespace().count();
        let result = split_oversized_chunks(vec![chunk], &token_fn, 512, 50);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text.matches("## Section").count(), 1);
        assert_eq!(result[0].heading_path, vec!["Section".to_string()]);
    }

    // ── Existing tests ───────────────────────────────────────────────────

    #[test]
    fn test_chunk_by_headings() {
        let md = "## A\nContent A\n\n## B\nContent B\n";
        let parsed = chunk_markdown(md, opts(0));
        // One chunk per section, regardless of how far under the size target
        // they are. Merging them would put two topics in one vector.
        assert_eq!(parsed.chunks.len(), 2);
        assert_eq!(parsed.chunks[0].heading.as_deref(), Some("## A"));
        assert_eq!(parsed.chunks[0].text, "## A\nContent A");
        assert_eq!(parsed.chunks[1].heading.as_deref(), Some("## B"));
        assert_eq!(parsed.chunks[1].text, "## B\nContent B");
    }

    #[test]
    fn test_no_headings_single_chunk() {
        let md = "Just some plain text\nwith multiple lines.";
        let parsed = chunk_markdown(md, opts(0));
        assert_eq!(parsed.chunks.len(), 1);
        assert!(parsed.chunks[0].heading.is_none());
        assert!(parsed.chunks[0].text.contains("Just some plain text"));
    }

    #[test]
    fn test_frontmatter_excluded() {
        let md = "---\ntags: [a]\n---\n# Title\nBody";
        let parsed = chunk_markdown(md, opts(0));
        assert_eq!(parsed.chunks.len(), 1);
        assert!(!parsed.chunks[0].text.contains("tags"));
        assert!(!parsed.chunks[0].text.contains("---\ntags"));
        assert!(parsed.chunks[0].text.contains("Body"));
    }

    #[test]
    fn test_snippet_truncation() {
        let long_text = "a".repeat(300);
        let md = format!("## Heading\n{long_text}");
        let parsed = chunk_markdown(&md, opts(0));
        assert!(!parsed.chunks.is_empty());
        // At least one chunk should have a truncated snippet
        let has_truncated = parsed.chunks.iter().any(|c| c.snippet.ends_with("..."));
        assert!(
            has_truncated,
            "Expected at least one snippet to be truncated"
        );
        // Verify truncation length
        for c in &parsed.chunks {
            if c.snippet.ends_with("...") {
                assert_eq!(c.snippet.len(), 203);
            }
        }
    }

    #[test]
    fn test_empty_file() {
        let parsed = chunk_markdown("", opts(0));
        assert!(parsed.chunks.is_empty());
    }

    #[test]
    fn test_parse_frontmatter_tags() {
        let md = "---\ntags: [rust, cli, search]\n---\n# Hello\nWorld";
        let parsed = chunk_markdown(md, opts(0));
        assert_eq!(parsed.tags, vec!["rust", "cli", "search"]);
    }

    #[test]
    fn test_long_chunk_split() {
        // Generate ~600 words of text with sentence boundaries
        let sentences: Vec<String> = (0..60)
            .map(|i| {
                format!(
                    "This is sentence number {} with several words to pad it out.",
                    i
                )
            })
            .collect();
        let long_text = sentences.join(" ");
        let word_count = long_text.split_whitespace().count();
        assert!(
            word_count > 512,
            "Test text must exceed 512 tokens (words); got {word_count}"
        );

        let chunk = Chunk {
            heading: Some("## Long Section".to_string()),
            heading_path: vec!["Long Section".to_string()],
            text: format!("## Long Section\n{long_text}"),
            snippet: make_snippet(&format!("## Long Section\n{long_text}")),
        };

        let token_fn = |s: &str| s.split_whitespace().count();
        let result = split_oversized_chunks(vec![chunk], &token_fn, 512, 50);

        assert!(
            result.len() >= 2,
            "Expected at least 2 sub-chunks, got {}",
            result.len()
        );
        // First chunk keeps original heading
        assert_eq!(result[0].heading.as_deref(), Some("## Long Section"));
        // Subsequent chunks get (cont.)
        assert_eq!(
            result[1].heading.as_deref(),
            Some("## Long Section (cont.)")
        );
        // All sub-chunks should be within token limit
        for c in &result {
            let tokens = token_fn(&c.text);
            assert!(tokens <= 512, "Sub-chunk has {tokens} tokens, exceeds 512");
        }
        // Snippets should be regenerated
        for c in &result {
            assert!(!c.snippet.is_empty());
        }
    }

    #[test]
    fn test_short_chunk_no_split() {
        let chunk = Chunk {
            heading: Some("## Short".to_string()),
            heading_path: vec!["Short".to_string()],
            text: "## Short\nJust a few words here.".to_string(),
            snippet: "## Short\nJust a few words here.".to_string(),
        };

        let token_fn = |s: &str| s.split_whitespace().count();
        let result = split_oversized_chunks(vec![chunk], &token_fn, 512, 50);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].heading.as_deref(), Some("## Short"));
        assert_eq!(result[0].text, "## Short\nJust a few words here.");
    }

    #[test]
    fn test_extract_heading() {
        assert_eq!(
            extract_heading("# Title\nBody text"),
            Some("# Title".to_string())
        );
        assert_eq!(extract_heading("## Sub\nBody"), Some("## Sub".to_string()));
        assert_eq!(extract_heading("No heading here"), None);
        assert_eq!(
            extract_heading("Some text\n### Deep heading\nMore"),
            Some("### Deep heading".to_string())
        );
    }

    #[test]
    fn test_thematic_break_detection() {
        assert!(is_thematic_break("---"));
        assert!(is_thematic_break("***"));
        assert!(is_thematic_break("___"));
        assert!(is_thematic_break("- - -"));
        assert!(is_thematic_break("----"));
        assert!(!is_thematic_break("--"));
        assert!(!is_thematic_break("abc"));
    }

    #[test]
    fn test_list_item_detection() {
        assert!(is_list_item("- item"));
        assert!(is_list_item("* item"));
        assert!(is_list_item("1. item"));
        assert!(is_list_item("10. item"));
        assert!(!is_list_item("plain text"));
        assert!(!is_list_item(""));
    }
}
