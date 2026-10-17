#!/usr/bin/env python3
"""3-way probe v2: dense vs +bm25 vs +bm42 on LongMemEval, with batched
embedding, win/loss vs dense, and an answer-length guard.

One scratch collection with three vectors. Never touches hermes_memories.

RAM-lean restructure (2026-10-11, GK directive "smooth operations"):
the original loaded all 3 ONNX models at once (~2-2.5G peak, OOM-killed
twice on the 6.9 GB host). Now models load SEQUENTIALLY per BLOCK of
texts: dense pass -> free -> bm25 pass -> free -> bm42 pass -> free ->
upsert the block. Peak = one model (~0.8-1G) + one block of embeddings.
Resume semantics unchanged: same collection, uuid5 point ids (idempotent
upserts), per-example count-skip, same final marker 'win/loss vs dense'.
"""
import gc
import os
import time
import uuid

from datasets import load_dataset
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

Q = "http://localhost:6333"
PROBE = "hermes_memories_hybrid_probe_v2"
SAMPLE = int(os.environ.get("SAMPLE", "1000"))
MIN_ANS = 10  # substantive-answer threshold for the strict view
# RAM knob (sequential models): texts embedded per model-load cycle.
# 1024 texts -> ~50-100 MB of intermediate vectors, and only 3 model
# loads per block (~3-5 s each from warm cache) instead of per chunk.
BLOCK = 1024
# Per-batch embed size inside a block. 64: 128-sized chunks + ONNX were
# occasionally OOM-killed under concurrent agent load (silent SIGKILL).
CHUNK = 64

DENSE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BM25_NAME = "Qdrant/bm25"
BM42_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"

c = QdrantClient(Q, timeout=300)

existing = {c_.name for c_ in c.get_collections().collections}
if PROBE not in existing:
    c.create_collection(
        PROBE,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
        # IDF modifier, matching the shipped schema (sparse_vector_params()).
        # The published +0.024 was measured WITHOUT it; this run refreshes the
        # number against what actually ships.
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "bm42": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

ds = load_dataset("LIXINYI33/longmemeval-s", split="train")


def turns(ex):
    out = []
    for sess in ex.get("haystack_sessions", []):
        i = 0
        while i < len(sess):
            m = sess[i]
            if (m.get("role") == "user" and i + 1 < len(sess)
                    and sess[i + 1].get("role") == "assistant"):
                u, a = str(m.get("content", "")), str(sess[i + 1].get("content", ""))
                if u.strip() and a.strip():
                    out.append(f"User: {u}\nAssistant: {a}")
                i += 2
                continue
            i += 1
    return out


def to_sv(s):
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def embed_dense_block(texts):
    """Load dense model, embed texts in CHUNK batches, free the model."""
    m = TextEmbedding(DENSE_NAME)
    out = []
    for start in range(0, len(texts), CHUNK):
        out.extend(v.tolist() for v in m.embed(texts[start:start + CHUNK], parallel=1))
    del m
    gc.collect()
    return out


def embed_sparse_block(texts, name):
    """Load one sparse model, embed texts in CHUNK batches, free the model."""
    m = SparseTextEmbedding(name)
    out = []
    for start in range(0, len(texts), CHUNK):
        out.extend(to_sv(s) for s in m.embed(texts[start:start + CHUNK]))
    del m
    gc.collect()
    return out


# ---- ingest: self-resuming, sequential-model block embedding ----
print(f"ingesting up to {SAMPLE} examples into {PROBE} "
      f"(resumable, sequential models, BLOCK={BLOCK})...")
t0 = time.time()
pending = []
for ex in ds.select(range(SAMPLE)):
    sid = str(ex["question_id"])
    ts = turns(ex)
    if not ts:
        continue
    have = c.count(PROBE, count_filter=models.Filter(
        must=[{"key": "session_id", "match": {"value": sid}}])).count
    if have >= len(ts):
        continue  # example fully stored by a previous (interrupted) run
    pending.append((sid, ts))
print(f"  {len(pending)} examples to ingest ({SAMPLE} checked, rest already stored)")

items = [(sid, t) for sid, ts in pending for t in ts]

# Per BLOCK: dense pass -> free -> bm25 pass -> free -> bm42 pass -> free
# -> upsert. Never more than ONE model resident; intermediate vectors are
# bounded by BLOCK (~50-100 MB).
for b0 in range(0, len(items), BLOCK):
    b1 = min(b0 + BLOCK, len(items))
    btexts = [t for _, t in items[b0:b1]]

    dvs = embed_dense_block(btexts)
    s25 = embed_sparse_block(btexts, BM25_NAME)
    s42 = embed_sparse_block(btexts, BM42_NAME)

    per = {}
    for i, (sid, t) in enumerate(items[b0:b1]):
        per.setdefault(sid, []).append(models.PointStruct(
            id=uuid.uuid5(uuid.NAMESPACE_URL, sid + "\n" + t),
            vector={"dense": dvs[i], "bm25": s25[i], "bm42": s42[i]},
            payload={"session_id": sid, "text": t},
        ))
    for pts in per.values():
        c.upsert(PROBE, points=pts)
    print(f"  embedded {b1}/{len(items)} texts ({time.time()-t0:.0f}s)")

# ---- collect valid questions, batch-embed all queries once ----
valid = [ex for ex in ds.select(range(SAMPLE))
         if str(ex.get("question", "")).strip() and str(ex.get("answer", "")).strip()]
queries = [str(ex["question"]).strip() for ex in valid]
answers = [str(ex["answer"]).strip() for ex in valid]
sids = [str(ex["question_id"]) for ex in valid]
del ds  # RAM knob: frees the Arrow table before the query loop
gc.collect()
print(f"querying {len(valid)} examples (batch-embedding queries, sequential models)...")
# Sequential model loads: 1000 short queries are cheap per model; each
# model is freed before the next loads. parallel=1 on dense: ORT default
# (parallel=0 -> cpu_count-1 threads) allocates a per-thread arena; on the
# 6.9 GB host that is the dominant avoidable cost. Slower embedding, but
# the run is self-resuming so wall time is bounded.
q_dv = embed_dense_block(queries)
q_s25 = embed_sparse_block(queries, BM25_NAME)
q_s42 = embed_sparse_block(queries, BM42_NAME)

arms = {"dense": [0, 0], "bm25": [0, 0], "bm42": [0, 0]}       # [r5, r10] all
strict = {"dense": [0, 0], "bm25": [0, 0], "bm42": [0, 0]}     # [r5, r10] len>=MIN_ANS
n_strict = 0
wins = {"bm25": [0, 0], "bm42": [0, 0]}   # [w10, l10] vs dense
wins5 = {"bm25": [0, 0], "bm42": [0, 0]}  # [w5, l5] vs dense

t1 = time.time()
for i in range(len(valid)):
    flt = models.Filter(must=[{"key": "session_id", "match": {"value": sids[i]}}])
    res = {
        "dense": c.query_points(PROBE, query=q_dv[i], using="dense",
                               query_filter=flt, limit=10),
        "bm25": c.query_points(
            PROBE, query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[models.Prefetch(query=q_dv[i], using="dense"),
                      models.Prefetch(query=q_s25[i], using="bm25")],
            query_filter=flt, limit=10),
        "bm42": c.query_points(
            PROBE, query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[models.Prefetch(query=q_dv[i], using="dense"),
                      models.Prefetch(query=q_s42[i], using="bm42")],
            query_filter=flt, limit=10),
    }
    al = answers[i].lower()
    hit = {}
    substantive = len(answers[i]) >= MIN_ANS
    for name, r in res.items():
        hs = [p.payload["text"].lower() for p in r.points]
        h5 = any(al in h for h in hs[:5])
        h10 = any(al in h for h in hs[:10])
        arms[name][0] += h5
        arms[name][1] += h10
        hit[name] = (h5, h10)
        if substantive:
            strict[name][0] += h5
            strict[name][1] += h10
    if substantive:
        n_strict += 1
    for name in ("bm25", "bm42"):
        wins[name][0] += hit[name][1] and not hit["dense"][1]   # win @10
        wins[name][1] += hit["dense"][1] and not hit[name][1]   # loss @10
        wins5[name][0] += hit[name][0] and not hit["dense"][0]
        wins5[name][1] += hit["dense"][0] and not hit[name][0]
    if (i + 1) % 200 == 0:
        print(f"  queried {i + 1}/{len(valid)}  ({time.time()-t1:.0f}s)")

n = len(valid)
print(f"\n=== 3-way hybrid probe: {n} examples, total {time.time()-t0:.0f}s ===")
print(f"{'arm':6s}  recall@5   recall@10   (all answers)")
for name, (r5, r10) in arms.items():
    print(f"{name:6s}  {r5/n:.3f}      {r10/n:.3f}")
print(f"\nsubstantive answers (len>={MIN_ANS}): n={n_strict}")
for name, (r5, r10) in strict.items():
    print(f"{name:6s}  {r5/n_strict:.3f}      {r10/n_strict:.3f}")
print(f"\nwin/loss vs dense (per {n} examples):")
for name in ("bm25", "bm42"):
    print(f"  {name}: @10  +{wins[name][0]} / -{wins[name][1]}"
          f"   @5  +{wins5[name][0]} / -{wins5[name][1]}")
