#!/usr/bin/env python3
"""3-way probe v2: dense vs +bm25 vs +bm42 on LongMemEval, with batched
embedding, win/loss vs dense, and an answer-length guard.

One scratch collection with three vectors. Never touches hermes_memories.
"""
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

c = QdrantClient(Q, timeout=300)
dense = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sp_bm25 = SparseTextEmbedding("Qdrant/bm25")
sp_bm42 = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

existing = {c_.name for c_ in c.get_collections().collections}
if PROBE not in existing:
    c.create_collection(
        PROBE,
        vectors_config={
            "dense": models.VectorParams(
                size=384, distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(),
            "bm42": models.SparseVectorParams(),
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


# ---- ingest: self-resuming, chunk-batched embedding (no multiprocessing) ----
print(f"ingesting up to {SAMPLE} examples into {PROBE} (resumable, chunk-batched)...")
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
CHUNK = 128
for start in range(0, len(items), CHUNK):
    chunk = items[start:start + CHUNK]
    texts = [t for _, t in chunk]
    dvs = [v.tolist() for v in dense.embed(texts, parallel=0)]
    s25 = [to_sv(s) for s in sp_bm25.embed(texts)]
    s42 = [to_sv(s) for s in sp_bm42.embed(texts)]
    per = {}
    for i, (sid, t) in enumerate(chunk):
        per.setdefault(sid, []).append(models.PointStruct(
            id=uuid.uuid5(uuid.NAMESPACE_URL, sid + "\n" + t),
            vector={"dense": dvs[i], "bm25": s25[i], "bm42": s42[i]},
            payload={"session_id": sid, "text": t},
        ))
    for pts in per.values():
        c.upsert(PROBE, points=pts)
    done = start + len(chunk)
    if (start // CHUNK + 1) % 4 == 0 or done == len(items):
        print(f"  embedded {done}/{len(items)} texts ({time.time()-t0:.0f}s)")

# ---- collect valid questions, batch-embed all queries once ----
valid = [ex for ex in ds.select(range(SAMPLE))
         if str(ex.get("question", "")).strip() and str(ex.get("answer", "")).strip()]
queries = [str(ex["question"]).strip() for ex in valid]
answers = [str(ex["answer"]).strip() for ex in valid]
sids = [str(ex["question_id"]) for ex in valid]
print(f"querying {len(valid)} examples (batch-embedding queries)...")
q_dv = [v.tolist() for v in dense.embed(queries, parallel=0)]
q_s25 = [to_sv(s) for s in sp_bm25.embed(queries)]
q_s42 = [to_sv(s) for s in sp_bm42.embed(queries)]

arms = {"dense": [0, 0], "bm25": [0, 0], "bm42": [0, 0]}       # [r5, r10] all
strict = {"dense": [0, 0], "bm25": [0, 0], "bm42": [0, 0]}     # [r5, r10] len>=MIN_ANS
n_strict = 0
wins = {"bm25": [0, 0], "bm42": [0, 0]}   # [w10, l10] vs dense
wins5 = {"bm25": [0, 0], "bm42": [0, 0]}  # [w5, l5] vs dense

t1 = time.time()
for i in range(len(valid)):
    flt = models.Filter(must=[{"key": "session_id", "match": {"value": sids[i]}}])
    res = {
        "dense": c.query_points(
            PROBE, query=q_dv[i], using="dense", query_filter=flt, limit=10,
        ),
        "bm25": c.query_points(
            PROBE, query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[models.Prefetch(query=q_dv[i], using="dense"),
                      models.Prefetch(query=q_s25[i], using="bm25")],
            query_filter=flt, limit=10,
        ),
        "bm42": c.query_points(
            PROBE, query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[models.Prefetch(query=q_dv[i], using="dense"),
                      models.Prefetch(query=q_s42[i], using="bm42")],
            query_filter=flt, limit=10,
        ),
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
