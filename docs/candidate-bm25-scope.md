# Candidate B — BM25 Sparse Hybrid: Scope (2026-10-08)

## What it is
Add a second, **sparse** vector to the qdrant-memory plugin so recall becomes
dense + sparse fused, instead of dense-only. Sparse vectors carry exact-term
signal that dense embeddings wash out — the difference between "the user asked
about X" and "the memory *contains* the word X".

## Why it matters (the number that decides)
LongMemEval baseline, dense-only, 1,000 examples:

| metric | dense-only |
|---|---:|
| recall@5  | 0.228 |
| recall@10 | 0.278 |
| MRR       | 0.219 |
| nDCG@5    | 0.214 |

Candidate B is justified **only if it moves recall@10 materially above 0.278**.
That is the whole decision criterion. If the hybrid lands near it, the
migration risk is not worth it.

## The dependency that removes the cost question
`Qdrant/bm25` ships inside **fastembed 0.8.1 — already a dependency**.
No new model download, no new package, no GPU. The only real cost is the
collection migration.

## Three changes required

### 1. Collection schema — declares the sparse vector
Live collection today (verified 2026-10-08):

```
vectors: {"dense": VectorParams(size=384, distance=COSINE)}
sparse_vectors: None
```

Needs:

```
sparse_vectors: {"bm25": SparseVectorParams()}
```

**Existing collections cannot be altered in place.** Qdrant has no
add-sparse-vector API. The migration is: create new collection → copy points
with sparse vectors recomputed → swap the name → drop the old.

### 2. `add_memory` — upsert the sparse vector
Every new memory must be embedded twice: dense (existing) + sparse (new).
`Embedder.encode` returns a dense `list[float]`; a second call to
`SparseTextEmbedding.encode` returns `(indices, values)` which upserts as
`sparse_vector`.

### 3. `prefetch` — the hybrid query
Today's recall is a single dense query (`query=dense_vec, using="dense"`).
Hybrid needs two prefetches fused:

```
Prefetch(query=dense_vec, using="dense") +
Prefetch(query=sparse_vec, using="bm25") +
query=None  # fusion only, no top-level query
```

Qdrant >=1.19 requires `query=None` when merging prefetches (the plugin already
documents this trap at `__init__.py:753`).

## Risks

| risk | severity | note |
|---|---|---|
| Collection migration loses data on failure | high | mitigated by copy-then-swap, never drop-first |
| Live store is mid-cleanup (LongMemEval contamination) | medium | finish cleanup first; migration on the clean store |
| Sparse model adds ~10 MB and one more encode call per memory | low | acceptable; already a dependency |
| `hybrid: off` default required | — | must not silently change behaviour for existing users |
| RSS disclosure | required | sparse vector is a config knob, must be documented |

## Open questions before starting
1. Which sparse model? `Qdrant/bm25` (10 MB, true BM25) vs `Qdrant/bm42-all-minilm-l6-v2-attentions` (90 MB, learned). bm25 is the honest BM25; bm42 is learned and may score better.
2. Fusion weight? RRF (reciprocal rank fusion) is the default in Qdrant's hybrid examples and needs no tuning. Start there.
3. Migration window: the live store is ~198k points and growing. Copy-then-swap on that is a multi-hour job. Do it during a low-recall window.

## Decisions (GK, 2026-10-08 ~23:30 CEST)
1. **Order: B (full measurement) → A (build).** GK will ping when back at home
   on full power (1-2h). Then: 3-way probe at SAMPLE=1000 via
   `hybrid_probe_3way.py` (already parameterized, caches warm).
2. **A approved with conditions:** opt-in hybrid behind `hybrid: off`, no live
   migration; comprehensive tests (GK loves tests — more is better);
   docs in TWO layers: one brief layer linking to one detail layer, mid-level
   detail upfront (OSS: code is the deep reference) — never overbuild docs.
3. **Probe results (200 examples, 2026-10-08):** dense 0.455/0.500 ·
   +bm25 0.490/0.505 · +bm42 0.485/0.505 (recall@5/@10). Real but modest;
   gain concentrated at @5; bm25 ≈ bm42. The cons are documentable and
   marketable (honest measurement, re-runnable harness).
4. Migration stays a separate, later decision — gated on the full-run number.

## Status
- Scope written: 2026-10-08
- Decisions recorded: 2026-10-08 (above). Awaiting GK's ping for run B.