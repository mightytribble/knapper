# FAQ

Advice, not reference. `knapper <command> --help` has the flags, and
[surfaces.md](surfaces.md) says what each capability is called on the CLI, on
MCP and over HTTP.

## Which build and which models should I use?

Three main configurations:

| | embedder | scorer | build |
|---|---|---|---|
| **Sweet spot** | EmbeddingGemma (default, local) | calibrated fusion, no cross-encoder model | CPU |
| **Best results** | Gemini API | 4B cross-encoder | CUDA |
| **Very long passages** | Gemini API, or Qwen3-Embedding | either | either |

The default is the sweet spot on purpose: no API key, no second download, and
about 22 ms a query. Adjust to taste.

**CUDA** speeds up embedding, but the real reason to want it is that it makes
a cross-encoder affordable. A 4B cross-encoder is a little better than the
default at ordering hard queries, and costs roughly 400 ms a query on a GPU
that can hold it. On Linux and WSL2 the `:cuda` Docker image is the least
painful route — it carries its own toolkit, so the host needs Docker and the
NVIDIA Container Toolkit and never `nvcc`.

**Gemini embeddings** beat EmbeddingGemma, at the cost of money, some
latency, and sending your notes' text to Google. Set
`models.embed = "gemini:<versioned-id>"` and put the key in `GEMINI_API_KEY`.

**Very long passages** are the one case where the default embedder is the
wrong tool. knapper packs a section's paragraphs into chunks of about 500
tokens, so an ordinary long section is simply several chunks and no embedder
sees more than the budget. A single paragraph or table that cannot be split
that way is emitted whole — and if it is over the embedder's input wall, 2048
tokens for EmbeddingGemma, it is torn with overlap before it is embedded.
Qwen3-Embedding (32k) or the Gemini API takes such a block whole, and reads
complex passages better.

Changing the embedder re-indexes the vault, and it invalidates the
calibration — see [how-knapper-searches.md](how-knapper-searches.md#when-you-change-the-embedder).

## Can I run more than one vault?

knapper uses sqlite, which really wants to use one database location for
everything. To get around this restriction, knapper supports one data directory 
per vault by setting the `KNAPPER_HOME` environment variable:

```bash
KNAPPER_HOME=~/.knapper-work    knapper index ~/vaults/work
KNAPPER_HOME=~/.knapper-fiction knapper index ~/vaults/fiction
```

Everything derives from that directory — the database, the models, the vault
profile and the config — so one variable moves the whole install. `--data-dir`
does the same thing and wins over the environment.

To give each project its own vault, set the variable in that project's MCP
registration rather than in your shell. In a project's `.mcp.json`:

```json
{
  "mcpServers": {
    "knapper": {
      "type": "stdio",
      "command": "knapper",
      "args": ["serve"],
      "env": { "KNAPPER_HOME": "/home/you/.knapper-work" }
    }
  }
}
```

The agent in that workspace then sees that vault and no other. Models live
under `<data dir>/models/` and are downloaded per data directory, so symlink
that directory at a shared copy if you do not want a second GGUF on disk.

## How do I write notes that retrieve well?

knapper cuts a note into chunks of about 500 tokens along its headings, and
merges the retrieved chunks of one section back together in the results — so
a tidy section is a unit that retrieves as a unit. None of this is a hard
rule; each one makes the results better.

- **One subject per section.** A section that covers two topics matches both
  weakly and scores well for neither.
- **A section of a few hundred tokens lands in one chunk.** Past about 500 it
  becomes several, and those come back merged only when more than one of them
  ranks — so the tighter the section, the more reliably it retrieves whole.
  This budget is knapper's, not the model's: a bigger embedder does not raise
  it.
- **Use a real heading hierarchy.** Headings are how knapper cuts the note,
  and the breadcrumb — `Note > H1 > H2` — is indexed as searchable text, so a
  descriptive heading is worth more than a decorative one.
- **Group related topics in one note** and separate unrelated ones. Chunks of
  one note merge in the results when they abut and share a section; a sibling
  section stops the merge, and unrelated neighbours still compete.
- **Tag, and use folders.** Both are how an agent narrows a search before it
  runs one — `--scope /projects/`, `--all type/decision` — and a scoped query
  is faster and more accurate than a broad one.
- **Write meaningful properties.** See the next entry.

## How do I use properties as RDF triples?

A property is a named value on a note, and `note → name → value` is a triple.
knapper reads both spellings Obsidian users write: a frontmatter key, and a
Dataview inline field in the body.

```markdown
---
employer: "[[Acme Corp]]"
status: active
---

Mentor:: [[Ada Lovelace]]
```

Both rows are indexed beside the wikilink graph, and both are queryable:

```bash
knapper list --property employer --links-to "Acme Corp"
knapper search "swamp survey" --linked-from "Ada Lovelace"
```

Every search hit carries its note's frontmatter properties, so the agent sees
the relations without a second call — this is the main reason to write them.

Two things to know. Quote a frontmatter link: unquoted `[[X]]` is a nested
sequence in YAML and writes no row, and `knapper validate` warns about it.
And **ranking never reads the property table** — properties filter and inform,
they do not score. If you want a value to be *searchable* as text, put it in
the note body, where the inline `Key:: value` form serves both purposes.

## How do I tell my agent about my vault?

Put a `CLAUDE.md` or `AGENTS.md` at the vault root — or in the project that
works with it — and say three things: what the vault is for, how it is laid
out, and that knapper is how to reach it. Point at the layout rather than
listing it; `knapper vault-map` and `knapper tags` give the agent the current
picture on demand, and a hand-written list goes stale.

A short style guide beside it is worth more than it looks. If you say how you
want notes written — heading depth, which properties matter, how tags are
named — the agent's writes match your vault instead of drifting from it, and
notes written that way retrieve better.

## Can my agent organize the vault for me?

Yes, and it is one of the better uses of the write side. `knapper health`
finds orphan notes, broken wikilinks, links naming a heading that no longer
exists, and stale content; `knapper validate` finds markdown and property
problems; `knapper tags` shows a vocabulary that has drifted. Hand an agent
that output and a rule for what to do about it, and it can work through the
list — reading, editing sections and fixing links one call at a time.

Do it on a vault you can restore. Every write is atomic and audited, but a
few hundred edits at agent speed is still a few hundred edits.

## How do I check an edit landed everywhere?

Use `match`, not `search`. `search` is ranked and cut to `top_n`, so it
answers *something* whatever you ask it — which makes it useless for proving
a string is gone. `match` is the other contract: one literal string, every
note in scope, no ranking.

```bash
knapper match "years old at the start of the story"
```
```
No note holds "years old at the start of the story".
```

The counts come back whole even when `--limit` shortens the listed lines. It
reads the indexed note bodies, so it does not see frontmatter, and it will
not tell you what a note is *about* — that is what `search` is for.

## How do I edit a note without breaking it?

Read the section, change it, write it back. What `read --section` returns is
exactly what `update --section` takes: the body **below** the heading, with
the heading named beside it rather than carried in the text. `knapper list
--detailed` prints each note's heading outline, which is how an agent finds
the section to name before it reads or writes one.

```bash
knapper update "Meeting Notes" --section "Action Items" \
  --mode append --content="- [ ] Follow up with Sarah"
```

Five things the write pipeline will not let you get wrong, and one shell
gotcha:

- **Content that opens with a heading at or above the section's own level is
  refused.** Such a line would end the section rather than fill it.
- **A rename keeps the markup.** `--heading` carries text only, so a `###`
  stays a `###`. A name another section already holds is refused, because two
  sections of one name leave both unaddressable.
- **A write changes only what it names.** A property edit keeps the key's
  place and the note's own list style; the rest of the frontmatter and the
  whole body stay byte for byte.
- **A value knapper cannot edit as a line** — a nested mapping, an anchor, a
  block scalar — refuses the write and says what it found, rather than
  re-styling your frontmatter.
- **Blank lines are handled at the joining edge.** A blank line where the
  content meets the old body asks for a paragraph break; blank lines at the
  other edges are dropped.
- **Write `--content=` with an equals sign** when the value starts with `-`,
  or clap reads it as a flag.

Batch edits are one call: `--edits` takes a JSON list, and the whole list is
one file write, one conflict check and one re-index.

## How do I restructure my vault into PARA?

`knapper migrate` classifies notes into Projects, Areas, Resources and
Archive by heuristic, and the workflow is preview-first:

```bash
knapper migrate --mode preview   # writes a plan to ~/.knapper/
knapper migrate --mode apply     # moves the files
knapper migrate --mode undo      # reverses the last migration
```

Read `~/.knapper/migration_preview.md` before you apply. The signals are open
tasks and active status for **Projects**, recurring topic keywords for
**Areas**, people and reference material for **Resources**, and
done/inactive/unlinked for **Archive**. A note that matches nothing with
enough confidence stays where it is, and daily notes and templates are always
skipped.

The same three modes are available as an MCP tool and an HTTP endpoint, so an
agent can drive it — but on both of those, `apply` takes the plan that
`preview` returned; only the CLI reads the copy on disk.

## Why did my search return nothing?

Because knapper would rather say nothing than hand your agent a plausible
wrong section. Every result is scored as a probability and dropped below a
floor, so an unanswerable query returns
`No relevant content found for this query in the vault.`

If it happens on a query you know the vault answers, the floor is probably
mis-fit for your setup — most often because you changed the embedding model.
[how-knapper-searches.md](how-knapper-searches.md#why-you-got-nothing-back)
has the diagnosis and the three ways out.

## How is this different from plain vector RAG, or Obsidian search?

Obsidian's own search is keyword-only and has no way to hand results to an
agent. A vector-only RAG stack matches meaning but not exact names, ignores
your wikilinks, and — the part that costs you — always returns its top *k*,
so an agent cannot tell a real answer from the nearest thing in the vault.

knapper runs both lanes plus the link graph, scores what comes out as a
calibrated probability, and abstains below the floor. And it writes: an agent
can edit a section by heading, not just read one.
