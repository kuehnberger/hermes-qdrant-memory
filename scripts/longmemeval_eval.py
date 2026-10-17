#!/usr/bin/env python3
"""LongMemEval adapter — runs the real tool path against an external corpus.

Loads LongMemEval (cleaned, 500 examples), seeds the eval collection with
haystack sessions via ``handle_tool_call``, then queries with each question
and reports recall@k / MRR / nDCG@5 — the same metric family as the
synthetic harness, but on a citable external dataset.

Safety: writes ONLY to ``hermes_memories_eval`` (same rule as retrieval_eval.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_COLLECTION = "hermes_memories_eval"
SEARCH_LIMIT = 10

LME_PATH = Path(
    "/home/gk/.cache/huggingface/hub/"
    "datasets--xiaowu0162--longmemeval-cleaned/snapshots/"
    "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json"
)


def load_longmemeval(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def session_to_text(session: list[dict]) -> str:
    parts = []
    for msg in session:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    rr = [1.0 / r if r is not None and r > 0 else 0.0 for r in ranks]
    return sum(rr) / len(rr)


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    dcg = [1.0 / (1.0 + (r - 1)) if r is not None and r <= k else 0.0 for r in ranks]
    idcg = [1.0] * len(ranks)
    return sum(dcg) / sum(idcg) if sum(idcg) else 0.0


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = -(-pct * len(ordered) // 100)
    return ordered[max(0, min(len(ordered) - 1, rank - 1))]


def run(keep: bool = False) -> dict:
    # Same import pattern as retrieval_eval.py — tests/conftest.py sets up
    # the path so plugins.memory.qdrant resolves without hermes_yaml.
    tests_dir = REPO_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import conftest  # noqa: F401
    from plugins.memory.qdrant import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    real_collection = provider._collection
    if EVAL_COLLECTION == real_collection:
        raise SystemExit(
            f"refusing to run: eval collection {EVAL_COLLECTION!r} equals the "
            f"configured collection {real_collection!r}"
        )
    provider._collection = EVAL_COLLECTION
    provider.initialize("eval")

    data = load_longmemeval(LME_PATH)
    print(f"LongMemEval: {len(data)} examples")

    # Seed: ingest haystack sessions, track which point_ids are answer sessions
    print("seeding eval collection with haystack sessions...")
    t0 = time.monotonic()
    seeded = 0
    # answer_point_ids[i] = set of point_ids that contain the answer for example i
    answer_point_ids: list[set[str]] = []

    for i, ex in enumerate(data):
        sessions = ex.get("haystack_sessions", [])
        haystack_sids = ex.get("haystack_session_ids", [])
        answer_sids = set(ex.get("answer_session_ids", []))
        answer_pids: set[str] = set()

        for j, session in enumerate(sessions):
            text = session_to_text(session)
            if not text:
                continue
            sid = f"lme_{i}_{j}"
            try:
                result = provider.handle_tool_call("qdrant_upsert", {
                    "text": text[:2000],
                    "session_id": sid,
                })
                m = re.search(r'stored point (\S+)', result)
                if m:
                    pid = m.group(1)
                    seeded += 1
                    # If this session's original ID is an answer session, track it
                    if j < len(haystack_sids) and haystack_sids[j] in answer_sids:
                        answer_pids.add(pid)
            except Exception as e:
                print(f"  seed failed for {i}/{j}: {e}")

        answer_point_ids.append(answer_pids)
        if (i + 1) % 100 == 0:
            print(f"  seeded {i + 1}/{len(data)} examples ({seeded} sessions)")

    seed_ms = int((time.monotonic() - t0) * 1000)
    print(f"seeded {seeded} sessions in {seed_ms}ms")

    # Query
    print("querying...")
    ranks: list[int | None] = []
    latencies: list[float] = []
    misses: list[tuple[str, int | None]] = []

    for i, ex in enumerate(data):
        question = ex.get("question", "")
        if not question:
            continue

        t1 = time.monotonic()
        try:
            result = provider.handle_tool_call("qdrant_search", {
                "query": question,
                "limit": SEARCH_LIMIT,
            })
        except Exception as e:
            print(f"  query failed for {i}: {e}")
            ranks.append(None)
            continue

        latency = (time.monotonic() - t1) * 1000
        latencies.append(latency)

        # Parse result string: "[0.85] (point_id) text"
        found_rank = None
        if isinstance(result, str):
            for rank, line in enumerate(result.splitlines(), start=1):
                m = re.search(r'\(([^)]+)\)', line)
                if m and m.group(1) in answer_point_ids[i]:
                    found_rank = rank
                    break

        ranks.append(found_rank)
        if found_rank is None or found_rank > 5:
            misses.append((question[:80], found_rank))

        if (i + 1) % 100 == 0:
            print(f"  queried {i + 1}/{len(data)}")

    report = {
        "corpus": len(data),
        "seeded": seeded,
        "seed_ms": seed_ms,
        "limit": SEARCH_LIMIT,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mrr": mrr(ranks),
        "ndcg@5": ndcg_at_k(ranks, 5),
        "latency_mean_ms": mean(latencies) if latencies else 0,
        "latency_p95_ms": percentile(latencies, 95) if latencies else 0,
        "misses": misses[:10],
    }

    print()
    print(f"corpus: {report['corpus']} examples, {report['seeded']} sessions seeded")
    print(f"window: top-{report['limit']}  |  collection: {EVAL_COLLECTION}")
    print()
    print(f"  recall@1   {report['recall@1']:.3f}")
    print(f"  recall@5   {report['recall@5']:.3f}")
    print(f"  recall@10  {report['recall@10']:.3f}")
    print(f"  MRR        {report['mrr']:.3f}")
    print(f"  nDCG@5     {report['ndcg@5']:.3f}")
    print(f"  latency    mean {report['latency_mean_ms']:.1f} ms, "
          f"p95 {report['latency_p95_ms']:.1f} ms")
    if misses:
        print(f"\n  misses ({len(misses)} total, showing {len(report['misses'])}):")
        for q, r in report["misses"]:
            print(f"    [{'miss' if r is None else f'rank {r}'}] {q}")

    if not keep:
        try:
            provider._client.delete_collection(EVAL_COLLECTION)
        except Exception:
            pass

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LongMemEval eval")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    run(keep=args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
