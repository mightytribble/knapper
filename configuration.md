# Configuration

knapper runs with no configuration. Everything below has a default, and the
defaults are what the search behaviour was measured at — change a key when
you have a reason, not to get started.

The file is `~/.knapper/config.toml`. A fresh install writes the whole
catalogue there, commented out under live section headers, so the file
itself is a reference: uncomment a line to set it. `knapper configure`
edits the file in place rather than rewriting it, so your comments, key
order and any key this build does not know survive the write.

Three things decide where a key can be changed cheaply:

| | cost |
|---|---|
| Query-time keys | none — edit and search again |
| `[fts]` flags | the keyword index rebuilds, ~0.1 s, no vault read, no model |
| Chunking, embedding and prompt keys | the vault re-indexes on the next `knapper index` |

The comments below say which is which. knapper enforces this for you: the
keys in the third group are hashed into an index fingerprint, and a
mismatch makes the next index run rebuild what it has to.

## The file

```toml
vault_path = "~/Documents/MyVault"
# How many results a search returns, counting a merged block as one. Fewer
# come back only when the vault holds fewer: `[ranking] candidates` caps the
# pool and the answer floor drops what is not an answer.
top_n = 10
exclude = [".obsidian/", "node_modules/", ".git/", "*-index.md", "templates/"]

# Results address sections, not whole documents. "file" restores one result
# per document; max_chunks_per_file bounds how much of one document can fill
# a page of results (0 = unlimited).
group_by = "chunk"
max_chunks_per_file = 3

# Enable the cross-encoder rerank lane
intelligence = true

# Chunking. All three are hashed into the index fingerprint, so a change to
# any of them re-indexes the vault on the next `knapper index`.
# chunk_min_chars = 120        # shortest section body that becomes its own
#                              # chunk; a shorter one merges into the chunk
#                              # before it. 0 is no minimum.
# promote_bold_headings = true # a bold-only line opens a section
# carry_orphan_headings = true # a bodyless heading is folded into a
#                              # neighbour rather than dropped
# breadcrumb_root = "path"     # what leads a section's breadcrumb:
#                              # "path", "name" or "stem"

# What each lane's rank is worth to RRF fusion. The values below are the
# defaults. They are query-time, so a sweep costs no re-index. Under the
# default ranking stage the cross-encoder sorts the shortlist, so only
# `semantic` and `fts` decide anything; the rest are read by
# `[ranking] mode = "legacy"`, where all five lanes vote.
[lane_weights]
# semantic = 1.0
# fts = 1.0
# graph = 0.8
# rerank = 1.0
# temporal = 0.0

# The ranking stage. Every key here is query-time: a sweep costs no re-index.
[ranking]
# mode = "sorted"              # "legacy" restores five-lane RRF
# retrieval_width = 60         # rows each content lane fetches
# candidates = 30              # size of the shortlist the scorer sorts
# answer_floor = 0.30          # cross-encoder probability below which a
#                              # result is not an answer; 0.0 keeps everything
# per_note_cap = 0             # sections of one note in the results, 0 = no cap
# Abutting chunks of one section — and of the subsections below it — come
# back as one block, stopping at a sibling section. false gives per-chunk
# results.
# coalesce_adjacent = true

# Prepend document identity to each chunk before embedding. Off by default —
# it helped conceptual queries and hurt exact-name lookup on the test vault.
# Needs `knapper index --rebuild` to take effect either way.
[embedding_prefix]
enabled = false
# path = true      # "Archdragon — lore/bestiary/archdragon.md"
# heading = true   # "Abilities > Combat" — ancestor sections
# tags = true
# aliases = true

# Override models for multilingual or custom use
[models]
# embed = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/qwen3-embedding-0.6b-q8_0.gguf"
# rerank = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
# Larger 4B cross-encoder (third-party GGUF): more accurate, slower, larger
# download. It runs at query time, so a switch needs no re-index.
# rerank = "hf:gscoppino/Qwen3-Reranker-4B-GGUF-llama_cpp/Qwen3-Reranker-4B-Q4_K_M.gguf"  # ~2.5 GB
# rerank = "hf:gscoppino/Qwen3-Reranker-4B-GGUF-llama_cpp/Qwen3-Reranker-4B.Q8_0.gguf"    # ~4.3 GB
# Hosted embeddings instead of the local GGUF. The id must be pinned and
# versioned: "gemini:gemini-embedding" and "-latest" are rejected, so a moving
# alias cannot re-point an existing store's vectors at a model that changed
# underneath them. The key is read from GEMINI_API_KEY and never written here.
# embed = "gemini:gemini-embedding-2"

# Non-secret knobs for the hosted embedder above.
# [models.embed_api]
# dim = 1536           # Matryoshka truncation; unset uses the native 3072
# timeout_secs = 30
# max_retries = 4      # on 429, 5xx and transport errors
# endpoint = "..."     # proxy or test-server override

# Registered AI agents
[agents]
# names = ["claude-code", "cursor"]

# What the keyword lane indexes beside the chunk body, and what bm25() pays
# for a hit there. The flags rebuild the keyword index (0.1 s, no vault read,
# no model); the weights are query-time and cost nothing.
[fts]
# heading_path = true          # the breadcrumb — on
# heading_path_weight = 1.0
# tags = false                 # the note's tags — off; a tag says what a note
#                              # is, not what it discusses
# tags_weight = 1.0

# The cross-encoder, when intelligence is on.
# [rerank]
# max_document_chars = 0       # 0 = feed it the whole candidate
# document_title = false       # prepend the note title to each candidate

# Calibrated fusion — what sorts search when no cross-encoder is configured.
# These four numbers are EmbeddingGemma's fit, not a global default. Point
# `models.embed` elsewhere and they no longer describe your scores: refit
# them together, or set `floor = 0.0`. See how-knapper-searches.md.
# [calibrated]
# enabled = true               # false sends a no-model build to legacy RRF
# semantic = 20.777
# keyword = 13.377
# intercept = -8.762
# floor = 0.75                 # probability below which a result is dropped

# The file watcher behind `knapper serve`.
# [watcher]
# backend = "auto"             # "native" (inotify/FSEvents) or "poll".
#                              # auto polls only on a filesystem inotify
#                              # cannot service — a Docker bind mount, NFS
# poll_interval_secs = 10

# The HTTP server, when you start it with `knapper serve --http`.
# [http]
# port = 3000
# host = "127.0.0.1"
# cors_origins = ["http://localhost:3000"]
# rate_limit = 60              # requests per minute per key
# [[http.api_keys]]            # written by `knapper configure --add-api-key`
# key = "kn_..."
# name = "claude-desktop"
# permissions = "write"        # or "read"
```

## Results are sections

Search returns **sections**. A note whose "Counterspell" and "Dispel Magic" sections both answer a query contributes both, up to `max_chunks_per_file`, and each result names the heading it came from. Pass `--group-by file` (or set `group_by = "file"`) for one result per document, representing it by its best-matching section.

## Embedding models

`models.embed` also takes another **local** GGUF. `knapper models list` names the ones that are known to work, and any `hf:<repo>/<file>.gguf` beyond them is accepted — the output width, the input context and the prompt template all come from the model itself, so nothing else has to be told what changed:

| model | dim | context | download | |
|---|---|---|---|---|
| `embeddinggemma-300M-Q8_0` | 768 | 2 048 | 334 MB | default |
| `Qwen3-Embedding-0.6B-Q8_0` | 1 024 | 32 768 | 639 MB | runs on CPU |
| `Qwen3-Embedding-4B-Q8_0` | 2 560 | 40 960 | 4.28 GB | wants a GPU |

Taking one of the Qwen rows costs two things. The store re-indexes at the new width, because the embedder is a fingerprint component and the vector table is declared at the model's dimension. And `[calibrated]`'s four numbers stay EmbeddingGemma's fit — see [how-knapper-searches.md](how-knapper-searches.md#when-you-change-the-embedder) — so refit them with `scripts/calibrated-fusion-eval.py`, set `[calibrated] floor = 0.0`, or configure a cross-encoder, which sorts instead and makes the section inert. `[embedding_prompt]` is EmbeddingGemma's pair of templates and does nothing under another family: Qwen3-Embedding takes its instruct on the query alone and embeds a document as itself.

`models.embed` takes a hosted provider as well as a local GGUF: `gemini:<versioned-id>` routes embedding to the Gemini API, reading the key from `GEMINI_API_KEY` — environment only, never written to config. The id has to end in a version number, so a moving alias cannot silently re-point an existing store's vectors. Documents batch at 100 per request and inputs are capped at 2048 tokens; `[models.embed_api]` bounds the call and can truncate the 3072-wide output. The embedder is a fingerprint component, so switching re-indexes the vault — and the model-free floor is fit against the local embedder, so a hosted one wants `[calibrated] floor = 0.0` or a refit.

## Embedding prefix

`embedding_prefix` puts the document's name, path, tags, aliases and the chunk's ancestor headings into the text that is **embedded**, while what is stored, displayed and keyword-matched stays the raw chunk. It is off by default: because the prefix is the same string for every chunk of a document, it separates documents from each other at the cost of separating a document's own sections, and on the test vault that lost more on exact-name lookup than it gained on conceptual queries (`eval/probes.md`). Each component is switchable so the trade can be measured per vault.

## Excludes

`exclude` takes `.gitignore`-style globs, matched against paths relative to the vault root. A pattern with no `/` matches at any depth (`*-index.md` catches `lore/lore-index.md`); a trailing `/` means a directory and everything under it; an embedded `/` anchors the pattern to the vault root (`drafts/**`). Excluding a path that is already indexed removes it from the store — chunks, vectors, FTS entries and graph edges — on the next index run.

## Data directory

All data stored in `~/.knapper/` — single SQLite database (~10MB typical), GGUF models, and vault profile. Set `KNAPPER_HOME` (used verbatim) or pass `--data-dir` to move that directory; `--data-dir` wins over the environment, which wins over the default.

One data directory holds one vault. To index a second vault, give it a directory of its own — see [faq.md](faq.md#can-i-run-more-than-one-vault).
