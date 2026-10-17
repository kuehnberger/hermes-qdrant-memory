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

**Hybrid applies from the moment you enable it — a fresh collection declares
the sparse vector at creation, and an existing dense-only collection gets it
added in place.** Qdrant >= 1.18 supports adding a named vector to a live
collection (`create_vector_name`); on an older server the plugin stays
dense-only with one warning. Note that `update_collection` does **not** work
for this — it answers 400 `Not existing vector name` — which is the trap that
made an earlier version of this plugin claim the migration was impossible.

- **Fresh install + `hybrid: "on"`** → collection is created with a `bm25`
  sparse vector (IDF modifier); every memory is embedded both ways; recall is
  fused. No backfill needed.
- **Existing collection + `hybrid: "on"`** → the plugin adds the sparse vector
  in place and activates hybrid. Points written before that moment carry no
  sparse value until they are backfilled — run `scripts/migrate_hybrid.py`
  once (resumable, ~1 min for a 130k store; dense vectors are untouched).
  Until the backfill runs, the sparse arm simply matches the newer points and
  the dense arm covers everything, so recall is never worse than dense-only.
- **`hybrid: "off"` (default)** → byte-identical to the previous behaviour.

`hermes memory status` reports which state you're in
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
substring of retrieved text; each example session-filtered. **Measured against
the shipped schema** — `bm25` and `bm42` declared with `modifier: IDF`, exactly
as the plugin creates them. Re-run it yourself: `scripts/hybrid_eval.py`
(scratch collection only, never touches your live store).

**Full run, n=1,000 (all examples):**

| arm | recall@5 | recall@10 |
|---|---:|---:|
| dense only | 0.398 | 0.434 |
| + bm25 (RRF) | 0.428 | 0.456 |
| + bm42 (RRF) | 0.419 | 0.460 |

Substantive answers only (answer length ≥ 10 chars, n=544) — the harder,
more honest subset:

| arm | recall@5 | recall@10 |
|---|---:|---:|
| dense only | 0.232 | 0.254 |
| + bm25 (RRF) | 0.250 | 0.272 |
| + bm42 (RRF) | 0.246 | 0.279 |

Per-example win/loss vs dense (per 1,000 examples):

| arm | @5 wins / losses | @10 wins / losses |
|---|---:|---:|
| + bm25 | +39 / −9 | **+24 / −2** |
| + bm42 | +31 / −10 | +28 / −2 |

**Honest limits of that table:**

- The gain is modest but consistent: **+0.022–0.026 recall@10** (+5–6%
  relative). bm25 is *near-additive at @10* — it wins 24 examples and loses
  only 2 (the 2 losses are the price of RRF re-ranking, and the same run
  measured them at zero before the IDF modifier was applied; treat the
  strict-additivity claim as withdrawn).
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
