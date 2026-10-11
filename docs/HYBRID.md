# Hybrid retrieval: dense + BM25 (optional)

Dense vector search finds *similar* text; BM25 finds *the exact terms you
asked for*. `hybrid: on` runs both and fuses their rankings, so a memory that
contains your rare keyword is no longer at the mercy of what the embedding
thought it meant.

The brief version lives in the [README](../README.md#optional-hybrid-search-dense--bm25);
this file is the detail layer.

## Config

```json
{ "hybrid": "on" }
```

in `<HERMES_HOME>/qdrant.json`. Default `off`. `on`/`off` are
case-insensitive; any unrecognized value resolves to `off` with one warning —
a typo can never switch a feature on that wasn't asked for.

## The collection rule (read this first)

**Hybrid applies to collections created while it is on — not to existing
ones.** Qdrant 1.19.2 refuses to add a sparse vector to an existing
collection (verified: `update_collection(sparse_vectors_config=...)` →
400 `Not existing vector name`). So:

- **Fresh install + `hybrid: "on"`** → collection is created with a `bm25`
  sparse vector; every memory is embedded both ways; recall is fused.
- **Existing collection + `hybrid: "on"`** → one warning, dense-only for the
  session. No migration, no failed writes, nothing changes. Enabling it on an
  old store means recreating the collection (or pointing at a new name) and
  re-ingesting.
- **`hybrid: "off"` (default)** → byte-identical to the previous behaviour.

`hermes memory status` reports which of the three states you're in
(`on` / `off` / `inactive: collection has no sparse vector`).

## What happens per operation

| operation | hybrid off | hybrid on |
|---|---|---|
| store a memory | dense vector only | dense + `bm25` sparse |
| recall | dense query | two prefetches (dense, bm25) fused with RRF |
| sparse embedder fails | n/a | **memory still stored dense-only**, one warning |
| sparse arm fails at recall | n/a | **falls back to dense recall**, one warning |

The optional arm never takes the required one down: a broken sparse model
costs you the *improvement*, never the memory and never the recall.

Sparse model: `Qdrant/bm25` (10 MB) from fastembed — already a dependency of
this plugin, no new package.

## Measured

**Method:** LongMemEval (`LIXINYI33/longmemeval-s`), one collection with all
three vectors, three query arms on the same questions; answer-hit as a
substring of retrieved text; each example session-filtered. Re-run it
yourself: `scripts/hybrid_eval.py` (scratch collection only, never touches
your live store).

**Preliminary, n=200 (first 200 examples):**

| arm | recall@5 | recall@10 |
|---|---:|---:|
| dense only | 0.455 | 0.500 |
| + bm25 (RRF) | 0.490 | 0.505 |
| + bm42 (RRF) | 0.485 | 0.505 |

**Honest limits of that table:**

- The gain is real but concentrated at **recall@5 (+0.035, ~7 of 200
  examples)**; recall@10 moved +0.005 (1 example) — flat within noise.
- `bm25` and `bm42` are indistinguishable at this size; we ship bm25 because
  it is 10 MB, literal BM25, and a dependency either way.
- The first 200 examples are the *short* half of the dataset (the full
  1,000-example dense baseline is **0.278** recall@10, not 0.500 — the two
  rows are not comparable across sample sizes; only the columns inside one
  table are).
- Substring answer matching is a generous proxy; it flatters all three arms
  equally, so the *deltas* are the signal, not the absolutes.

A full 1,000-example run (both halves) is the confirmation pass; its numbers
replace this table when it lands.

## When *not* to turn it on

- You want zero change on an existing store → leave it `off` (it's the
  default; doing nothing is correct).
- You can't afford a 10 MB model download on first use → leave it `off`.
- You measure everything you ship → run `scripts/hybrid_eval.py` first and
  decide on your own corpus's number.
