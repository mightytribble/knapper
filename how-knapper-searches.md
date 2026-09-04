# How knapper searches

Why a query returned what it returned, what the percentage means, and which
knobs are worth turning. For the keys themselves, see
[configuration.md](configuration.md).

## A result is a chunk

knapper does not return files, and it does not quite return sections either.
It cuts each note into **chunks**: it follows the note's headings, and packs
each section's paragraphs up to about 500 tokens. Two things follow from that
budget. A section longer than it becomes several chunks, cut on paragraph
boundaries. A section shorter than `chunk_min_chars` merges into the chunk
before it, heading line included. So a chunk is usually one section, and at
either edge of the budget it is part of one, or a run of small ones.

Each result names the chunk it came from:

```
--- [6e1b70#0] [99%] 02-Areas/Development/Auth-Architecture.md > Auth Architecture (matched: semantic+keyword)
```

`6e1b70` is the note's docid, `#0` the chunk's ordinal within that note, and
`matched:` names the lanes that found it.

**Retrieved chunks then merge.** After ranking, chunks of one note that abut —
consecutive ordinals — and belong to one section, or to the subsections below
it, are presented as a single block: at the strongest member's score, under
the leading member's heading, with the members' text in document order. The
merge stops at a sibling section, because a sibling starts a new topic.

The merge only ever sees chunks the search **retrieved**. A long section comes
back whole when all of its chunks ranked, and comes back in part when only
part of it did. It is also why `top_n` counts blocks and not chunks: the
results are counted after the merge, so ten results are ten blocks.

## The lanes

Up to five lanes run. Each answers a different question:

| lane | finds | reads |
|---|---|---|
| **semantic** | sections that mean the same thing as the query | cosine distance over the embedding of each chunk |
| **keyword** | sections that use the query's words | FTS5 BM25 over the chunk's whole text and its breadcrumb |
| **graph** | sections the answers point at | one personalized-PageRank hop over the wikilink graph |
| **temporal** | notes from the period the query names | a date range parsed out of the query text |
| **cross-encoder** | which candidate actually answers | the query and the candidate read together, if enabled |

Two things are worth knowing about the keyword lane. It indexes each chunk's
**full text**, not the 200-character snippet you see in results. And it
indexes the section's breadcrumb beside the body, so a query that names a
heading can find it even when the body never repeats the words.

The temporal lane turns itself on. "What happened last week", "March 2026
notes" and "recent" all parse to a date range; a query with no date in it
leaves the lane idle.

## How the lanes combine

```
query
  ├── semantic ─┐
  │             ├── RRF ──► shortlist (30) ──► one scorer sorts ──► floor ──► results
  ├── keyword ──┘              ▲
  ├── graph ───────────────────┤ reserved quota (8)
  └── temporal ────────────────┘ reserved quota (4)
```

The two content lanes fuse by **Reciprocal Rank Fusion**: a section that both
lanes rank highly beats one that only a single lane liked. Graph and temporal
candidates do not compete for those places — each has a reserved share of the
shortlist, and an unfilled reserve goes back to the content lanes.

Then **one scorer sorts the whole shortlist**, and everything it puts below
the floor is dropped. Fusion decides what the scorer gets to see; the scorer
decides the order you get.

## What the percentage means

It is the sorting scorer's probability that the section answers your query.
Which scorer produced it depends on what you have configured:

- **No cross-encoder — the default install.** A calibrated logistic scores
  each candidate from two numbers: its cosine against the query, and its BM25
  measured against the highest score that query could possibly reach. The
  second half is what makes the number comparable between queries. It costs
  no model call: about 22 ms a query on a CPU-only machine.
- **Cross-encoder enabled** (`knapper configure --enable-intelligence`). The
  model reads the query and the candidate together and scores the pair. It is
  better at the hardest case — a section that merely mentions your words
  rather than answering them — and it is far slower: about 13.8 seconds a
  query on the same CPU-only machine, and a fraction of that on a GPU build.

On a 240-note test vault the two refuse exactly the same unanswerable
queries, and the calibrated path misses one question the cross-encoder gets.
That is the trade: the cross-encoder buys ordering on hard queries, and it
costs the query time.

`--scores` prints the percentage either way. `--explain` adds each result's
per-lane ranks and raw scores, and, under the calibrated scorer, the query's
BM25 upper bound and per-term IDFs.

## Why you got nothing back

```
No relevant content found for this query in the vault.
```

This is a feature, and it is the reason to prefer knapper over a plain
vector search. Both scorers carry a **floor** — a probability below which a
candidate is not an answer — and a query whose candidates all score below it
returns nothing rather than its nearest miss. An agent that gets nothing back
asks you; an agent that gets a plausible wrong section quotes it.

The floors are per scorer, because each was fit against its own scores:
`[ranking] answer_floor`, 0.30, for the cross-encoder, and `[calibrated]
floor`, 0.75, for the calibrated logistic. Setting either to `0.0` turns
abstention off.

If you get an abstention on a query you know the vault answers, run it with
`--explain`. Lanes hitting and nothing coming back is the floor, not
retrieval — read the next section.

## When you change the embedder

`[calibrated]`'s four numbers — three coefficients and the floor — are
**EmbeddingGemma's fit**, not a global default. They are one fit and they do
not move apart.

The keyword half of the score normalizes itself against each query's own
upper bound, so it travels. The semantic half is a raw cosine, and where a
cosine sits depends on the model that produced it and on how long the notes
being scored are. Point `models.embed` at another model and the floor moves
in an unknown direction: the path then abstains on every query, or on none.

The same is true of a vault whose notes are much shorter than the ones a fit
was built on — the floor can land inside the band that vault's own correct
answers score in.

Three ways out, cheapest first:

1. **Configure a cross-encoder.** It sorts instead, and the whole
   `[calibrated]` section goes inert.
2. **Set `[calibrated] floor = 0.0`.** You keep the ordering and lose the
   abstention.
3. **Refit.** `scripts/calibrated-fusion-eval.py` takes a built store, query
   vectors from `examples/eval_query_embed.rs`, and two files you write
   yourself: your queries, and which chunks of your vault answer each one.
   Nothing ships those and nothing can — they are a judgement about your own
   notes. The shipped fit rests on 33 labeled answers against 1228 labeled
   non-answers, and a fit is worth what its labels are worth.

## What is worth tuning

Most of these are query-time: edit, search again, and compare. The ones that
re-index say so.

| you want | turn | cost |
|---|---|---|
| more or fewer results | `top_n` | none |
| one result per note | `group_by = "file"` | none |
| whole notes back instead of sections | `[ranking] coalesce_adjacent`, `per_note_cap` | none |
| less abstention | the floor for your scorer | none |
| a deeper candidate pool | `[ranking] retrieval_width`, `candidates` | none |
| better ordering on hard queries | `intelligence = true`, or a 4B cross-encoder | query time |
| tags to count as searchable text | `[fts] tags = true` | keyword index rebuild, ~0.1 s |
| a different embedding model | `models.embed` | full re-index, and refit the calibration |

Two knobs are measured losers on the test vault and ship off:
`[embedding_prefix]`, which helps conceptual queries and hurts exact-name
lookup, and `[fts] tags`, which drops a probe's most direct answer because a
tag says what a note *is* rather than what it discusses. Both are switchable
so you can measure them against your own vault rather than take our word.

Scoping is not tuning, and it is usually the better answer. `--scope`,
`--all`, `--any` and `--none` take tag terms and directory terms alike, and
they resolve **before** anything is embedded — a scoped query is faster as
well as narrower. `--explain` prints the scope it resolved and how many notes
it admitted, which is the first thing to check when a scoped search comes
back thin.
