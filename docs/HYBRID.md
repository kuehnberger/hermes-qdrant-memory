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

**Full run, n=1,000 (all examples):**

| arm | recall@5 | recall@10 |
|---|---:|---:|
| dense only | 0.398 | 0.434 |
| + bm25 (RRF) | 0.428 | 0.458 |
| + bm42 (RRF) | 0.421 | 0.461 |

Substantive answers only (answer length ≥ 10 chars, n=544) — the harder,
more honest subset:

| arm | recall@5 | recall@10 |
|---|---:|---:|
| dense only | 0.232 | 0.254 |
| + bm25 (RRF) | 0.246 | 0.272 |
| + bm42 (RRF) | 0.244 | 0.279 |

Per-example win/loss vs dense (per 1,000 examples):

| arm | @5 wins / losses | @10 wins / losses |
|---|---:|---:|
| + bm25 | +38 / −8 | **+24 / −0** |
| + bm42 | +30 / −7 | +29 / −2 |

**Honest limits of that table:**

- The gain is modest but consistent: **+0.024–0.027 recall@10** (+5–6%
  relative). bm25 is *strictly additive at @10* — it never lost an example
  dense alone had found.
- `bm42` edges `bm25` at @10 on substantive answers (0.279 vs 0.272) but
  costs 90 MB instead of 10 MB; we ship and recommend **bm25** because the
  trade (−0.007 on the hard subset for −80 MB) favours it for most users.
  (The model is a single constant, `SPARSE_MODEL_NAME` in `__init__.py`;
  swapping it to the bm42 name is a one-line change, not a config knob.)
- Substring answer matching is a generous proxy; it flatters all three arms
  equally, so the *deltas* are the signal, not the absolutes.
- This measures ranking on one benchmark; your corpus may differ. The
  re-runnable script (`scripts/hybrid_eval.py`) is the source of truth.

## When *not* to turn it on

- You want zero change on an existing store → leave it `off` (it's the
  default; doing nothing is correct).
- You can't afford a 10 MB model download on first use → leave it `off`.
- You measure everything you ship → run `scripts/hybrid_eval.py` first and
  decide on your own corpus's number.
