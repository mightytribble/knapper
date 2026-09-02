#!/usr/bin/env python3
"""Offline eval: calibrated score fusion against RRF and single lanes.

## The labels are yours to write

This tool fits `[calibrated]`'s four numbers. It does not ship the evidence to
fit them against, and there is no shipped tool that writes it. Two of the four
inputs are labels over your own vault, and nobody else can supply them:

  <arm-dir>            a built knapper store: the directory holding knapper.db
  <query-embeds.json>  query vectors, written by
                       `KNAPPER_HOME=<arm-dir> cargo run --example
                       eval_query_embed <pool-queries.json>`
  <ground-truth.json>  {"queries": {"<id>": {"tier1": [...], "tier2": [...],
                       "noise": [...]}}} — each member `{"file": vault-relative
                       path, "section": chunk seq, "anchor": a substring of the
                       chunk, optional}`. tier1 answers the query, noise does
                       not, tier2 is neutral.
  <pool-queries.json>  [{"id": ..., "query": ..., "cls": ...}] — `cls` is one of
                       positive, false-premise, far, mid, near, adjacent.

So you write two files, the last two. The shipped numbers were fit on 33 tier-1
positives against 1228 labeled negatives. A fit is worth what its labels are
worth.

## Refit when the embedder changes, not when the corpus does

The labels come from your corpus, but the corpus is not what makes the shipped
numbers wrong. Both features are self-normalized per query — BM25 against that
query's own upper bound, cosine against the model's scale — so a new corpus
changes what the features say and not what the coefficients mean. What the
coefficients are tied to is the embedder. Refit when you change `models.embed`.

Tests whether per-query self-calibrated lane scores — anchored cosine and
upper-bound-normalized BM25, fused by a logistic fit on the ground-truth pool —
can replace the cross-encoder's sort on the degraded (no-intelligence) path.

Reads a built arm's store directly (chunks, vectors, chunks_fts) plus query
vectors dumped by `examples/eval_query_embed.rs`. No knapper code runs here;
the FTS MATCH expression, bm25 weights and lane caps replicate what the
binary does at the pin config.

Usage:
  calibrated-fusion-eval.py <arm-dir> <query-embeds.json> <ground-truth.json> <pool-queries.json>

Scoring follows ground-truth.json's own scheme: coverage / rank / inversions
over a top-20 window; tier-2 is neutral; unclassified chunks above the lowest
tier-1 member are reported, not counted as noise. The logistic is evaluated
leave-one-query-out: every reported ranking and abstention number for a query
comes from coefficients fit without that query's labels.
"""

import hashlib
import json
import math
import sqlite3
import sys

import numpy as np

K1, B = 1.2, 0.75          # FTS5 bm25 constants
RRF_K = 60                 # fusion.rs default
WIDTH = 60                 # [ranking] retrieval_width
PER_FILE_CAP = 3           # shortlist_cap, applied per lane
WINDOW = 20                # top_n every ground-truth rank is scored at
BG_SAMPLE = 200            # deterministic background sample size
EPS_CEIL = 0.05            # minimum (ceil - bg) before falling back to raw cos
L2_REG = 1e-4

ARM, EMBEDS_PATH, GT_PATH, POOL_PATH = sys.argv[1:5]


# ---------------------------------------------------------------- store
con = sqlite3.connect(f"file:{ARM}/knapper.db?mode=ro", uri=True)
rows = con.execute(
    "SELECT c.id, c.file_id, f.path, c.seq, c.text, c.vector"
    " FROM chunks c JOIN files f ON f.id = c.file_id ORDER BY c.id"
).fetchall()
N = len(rows)
chunk_id = [r[0] for r in rows]
file_id = np.array([r[1] for r in rows])
path = [r[2] for r in rows]
seq = [r[3] for r in rows]
text = [r[4] for r in rows]
V = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows]).astype(np.float64)
V /= np.linalg.norm(V, axis=1, keepdims=True)
by_path_seq = {(path[i], seq[i]): i for i in range(N)}
by_chunk_id = {chunk_id[i]: i for i in range(N)}
chunks_of_file = {}
for i in range(N):
    chunks_of_file.setdefault(path[i], []).append(i)

# Deterministic background sample: lowest md5(chunk id), so reruns are
# byte-identical and the sample only moves when the store does.
bg_sample_idx = sorted(
    range(N), key=lambda i: hashlib.md5(str(chunk_id[i]).encode()).hexdigest()
)[:BG_SAMPLE]


# ---------------------------------------------------------------- lanes
def tokenize(query):
    seen, out = set(), []
    for t in query.split():
        if not any(ch.isalnum() for ch in t):
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out


def quote(t):
    return '"' + t.replace('"', '""') + '"'


def fts_lane(query):
    """All matching chunks with positive-better bm25, plus the query's upper bound."""
    terms = tokenize(query)
    if not terms:
        return {}, 0.0
    expr = " OR ".join(quote(t) for t in terms)
    hits = con.execute(
        "SELECT chunks_fts.rowid, bm25(chunks_fts, 1.0, 1.0)"
        " FROM chunks_fts WHERE chunks_fts MATCH ?",
        (expr,),
    ).fetchall()
    scores = {by_chunk_id[cid]: -s for cid, s in hits}
    bound = 0.0
    for t in terms:
        df = con.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (quote(t),)
        ).fetchone()[0]
        idf = max(math.log((N - df + 0.5) / (df + 0.5)), 1e-6)
        bound += idf * (K1 + 1)
    return scores, bound


def capped_top(order, k, cap):
    out, per_file = [], {}
    for i in order:
        f = file_id[i]
        if per_file.get(f, 0) >= cap:
            continue
        per_file[f] = per_file.get(f, 0) + 1
        out.append(i)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------- queries
pool = {q["id"]: q for q in json.load(open(POOL_PATH))}
embeds = {e["id"]: e for e in json.load(open(EMBEDS_PATH))}
gt = json.load(open(GT_PATH))["queries"]

Q = {}
anchor_misses = []
for qid, p in pool.items():
    e = embeds[qid]
    qv = np.array(e["q"], dtype=np.float64)
    qv /= np.linalg.norm(qv)
    dv = np.array(e["d"], dtype=np.float64)
    dv /= np.linalg.norm(dv)
    cos = V @ qv
    ceil = float(qv @ dv)
    fts_scores, bound = fts_lane(p["query"])
    bm25n = np.zeros(N)
    for i, s in fts_scores.items():
        bm25n[i] = min(s / bound, 1.0) if bound > 0 else 0.0

    sem_order = list(np.argsort(-cos))
    fts_order = sorted(fts_scores, key=lambda i: (-fts_scores[i], file_id[i], seq[i]))
    sem_lane = capped_top(sem_order, WIDTH, PER_FILE_CAP)
    fts_lane_top = capped_top(fts_order, WIDTH, PER_FILE_CAP)
    candidates = sorted(set(sem_lane) | set(fts_lane_top))

    rrf = {}
    for rank, i in enumerate(sem_lane, 1):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank)
    for rank, i in enumerate(fts_lane_top, 1):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank)

    # ground truth membership, resolved by (path, seq) and verified by anchor
    tier1, tier2, noise = set(), set(), set()
    if qid in gt:
        g = gt[qid]
        for name, target in (("tier1", tier1), ("tier2", tier2), ("noise", noise)):
            for m in g.get(name, []):
                idx = by_path_seq.get((m["file"], m["section"]))
                a = m.get("anchor")
                if a is not None and (idx is None or a not in text[idx]):
                    idx = next(
                        (j for j in chunks_of_file.get(m["file"], []) if a in text[j]),
                        idx,
                    )
                    if idx is None or a not in text[idx]:
                        anchor_misses.append((qid, name, m["file"], m["section"]))
                        continue
                if idx is None:
                    anchor_misses.append((qid, name, m["file"], m["section"]))
                    continue
                target.add(idx)

    Q[qid] = dict(
        cls=p["cls"], query=p["query"], cos=cos, ceil=ceil, bm25n=bm25n,
        bound=bound, candidates=candidates, rrf=rrf,
        sem_lane=sem_lane, fts_lane=fts_lane_top,
        tier1=tier1, tier2=tier2, noise=noise,
        bg_median=float(np.median(cos)), bg_mean=float(np.mean(cos)),
        bg_sampled=float(np.median(cos[bg_sample_idx])),
    )

if anchor_misses:
    print(f"!! {len(anchor_misses)} ground-truth members failed to resolve:")
    for m in anchor_misses:
        print("   ", m)


# ---------------------------------------------------------------- features
def sem_feature(q, variant):
    if variant == "raw":
        return q["cos"]
    bg = {"median": q["bg_median"], "mean": q["bg_mean"], "sampled": q["bg_sampled"]}[
        variant
    ]
    if q["ceil"] - bg < EPS_CEIL:
        return q["cos"]
    return np.clip((q["cos"] - bg) / (q["ceil"] - bg), 0.0, 1.0)


def labeled_points(qids, variant):
    """(X, y) for the fit: tier1 = 1; ruled noise and every candidate of a
    negative query = 0. Tier-2 and unclassified are neutral and excluded."""
    X, y = [], []
    for qid in qids:
        q = Q[qid]
        sem = sem_feature(q, variant)
        if q["cls"] == "positive" or q["cls"] == "false-premise":
            pos, neg = q["tier1"], q["noise"]
        elif q["cls"] in ("far", "mid", "near", "adjacent"):
            pos, neg = set(), set(q["candidates"])
        else:  # control: held out of the fit entirely
            continue
        for i in pos:
            X.append((sem[i], q["bm25n"][i]))
            y.append(1)
        for i in neg:
            X.append((sem[i], q["bm25n"][i]))
            y.append(0)
    return np.array(X), np.array(y)


def fit_logistic(X, y):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    beta = np.zeros(Xb.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-Xb @ beta))
        Wd = w * p * (1 - p) + 1e-9
        H = Xb.T @ (Xb * Wd[:, None]) + L2_REG * np.eye(Xb.shape[1])
        g = Xb.T @ (w * (y - p)) - L2_REG * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def predict(beta, sem, bm25n):
    z = beta[0] * sem + beta[1] * bm25n + beta[2]
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- scoring
def window_metrics(ranking, q):
    win = ranking[:WINDOW]
    t1_ranks = [r for r, i in enumerate(win, 1) if i in q["tier1"]]
    if not t1_ranks:
        return dict(cov=0, first=None, inv=0, uncl=0)
    lowest = max(t1_ranks)
    above = win[: lowest - 1]
    inv = sum(1 for i in above if i in q["noise"])
    uncl = sum(
        1
        for i in above
        if i not in q["tier1"] and i not in q["tier2"] and i not in q["noise"]
    )
    return dict(cov=len(t1_ranks), first=min(t1_ranks), inv=inv, uncl=uncl)


def rank_candidates(q, key):
    return sorted(q["candidates"], key=lambda i: (-key[i], file_id[i], seq[i]))


all_qids = list(Q)
fit_qids = [k for k in all_qids if Q[k]["cls"] != "control"]

arms = {}
for variant in ("median", "raw", "mean", "sampled"):
    per_query, maxp = {}, {}
    betas = []
    for qid in all_qids:
        train = [k for k in fit_qids if k != qid]
        X, y = labeled_points(train, variant)
        beta = fit_logistic(X, y)
        betas.append(beta)
        q = Q[qid]
        sem = sem_feature(q, variant)
        p = {i: predict(beta, sem[i], q["bm25n"][i]) for i in q["candidates"]}
        ranking = rank_candidates(q, p)
        per_query[qid] = window_metrics(ranking, q)
        maxp[qid] = max(p.values()) if p else 0.0
    arms[f"fused-{variant}"] = dict(per_query=per_query, maxp=maxp, betas=betas)

for name, key_of in (
    ("rrf", lambda q: q["rrf"]),
    ("semantic-only", lambda q: {i: q["cos"][i] for i in q["candidates"]}),
    ("fts-only", lambda q: {i: q["bm25n"][i] for i in q["candidates"]}),
):
    per_query = {}
    for qid in all_qids:
        q = Q[qid]
        per_query[qid] = window_metrics(rank_candidates(q, key_of(q)), q)
    arms[name] = dict(per_query=per_query)


# ---------------------------------------------------------------- report
positives = [k for k in all_qids if Q[k]["cls"] in ("positive", "false-premise")]
negatives = [k for k in all_qids if Q[k]["cls"] in ("far", "mid", "near", "adjacent")]

print("\n=== anchors per query (median bg over all chunks; ceiling = self-sim) ===")
for qid in all_qids:
    q = Q[qid]
    print(
        f"  {qid:3} {q['cls']:13} bg={q['bg_median']:.3f} ceil={q['ceil']:.3f}"
        f" bound={q['bound']:6.2f}  top-cos={q['cos'].max():.3f}"
        f"  top-bm25n={q['bm25n'].max():.3f}"
    )

print("\n=== ranking (window 20, leave-one-query-out for fused arms) ===")
hdr = f"{'arm':44}" + "".join(f"{qid:>9}" for qid in positives) + "   cov total   inv"
print(hdr)
for name, arm in arms.items():
    cells, cov_n, cov_d, inv = [], 0, 0, 0
    for qid in positives:
        m = arm["per_query"][qid]
        t1 = len(Q[qid]["tier1"])
        cells.append(f"{m['cov']}/{t1}@{m['first'] or '-'}")
        cov_n += m["cov"]
        cov_d += t1
        inv += m["inv"]
    print(
        f"{name:44}" + "".join(f"{c:>9}" for c in cells) + f"   {cov_n}/{cov_d}     {inv}"
    )
print("  (cells: tier1-in-window / tier1-total @ rank of first tier1;"
      " inv = ruled noise above lowest tier1)")

for name in ("fused-median", "rrf"):
    uncl = sum(arms[name]["per_query"][qid]["uncl"] for qid in positives)
    print(f"  unclassified-above-lowest-tier1 [{name}]: {uncl}")

print("\n=== abstention (max fused p per query, LOO; fused-median arm) ===")
maxp = arms["fused-median"]["maxp"]
for qid in sorted(all_qids, key=lambda k: -maxp[k]):
    q = Q[qid]
    marker = "POS" if qid in positives else ("ctl" if q["cls"] == "control" else "neg")
    print(f"  {maxp[qid]:.3f}  {qid:3} {marker}  {q['query'][:60]}")
scored_pos = [k for k in positives if Q[k]["cls"] == "positive"]
lo = min(maxp[k] for k in scored_pos)
hi = max(maxp[k] for k in negatives + [k for k in all_qids if Q[k]["cls"] == "control"])
print(f"  min over scored positives: {lo:.3f}   max over negatives+control: {hi:.3f}")
print(f"  separation: {'CLEAN — any floor in (' + format(hi, '.3f') + ', ' + format(lo, '.3f') + ')' if lo > hi else 'NOT clean'}")

print("\n=== abstention across variants: negatives killable while keeping all scored positives ===")
for variant in ("median", "raw", "mean", "sampled"):
    mp = arms[f"fused-{variant}"]["maxp"]
    lo = min(mp[k] for k in scored_pos)
    dead = sorted(
        (k for k in negatives + [k for k in all_qids if Q[k]["cls"] == "control"]
         if mp[k] < lo),
        key=lambda k: mp[k],
    )
    print(f"  fused-{variant:8} floor<{lo:.3f} kills {len(dead):2}/12: {' '.join(dead)}")

print("\n=== logistic coefficients (fused-median): [sem, bm25n, intercept] ===")
B_ = np.array(arms["fused-median"]["betas"])
print(f"  LOO mean {B_.mean(axis=0).round(3)}  min {B_.min(axis=0).round(3)}  max {B_.max(axis=0).round(3)}")
X, y = labeled_points(fit_qids, "median")
print(f"  full fit {fit_logistic(X, y).round(3)}   ({int((y==1).sum())} positives, {int((y==0).sum())} negatives)")
