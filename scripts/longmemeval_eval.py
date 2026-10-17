#!/usr/bin/env python3
"""LongMemEval adapter — runs the real tool path against an external corpus.

Loads LongMemEval (cleaned, 500 examples), seeds the eval collection with
haystack sessions via ``handle_tool_call``, then queries with each question
and reports recall@k / MRR / nDCG@5 — the same metric family as the
synthetic harness, but on a citable external dataset.

Safety: writes ONLY to the eval collection (same rule as retrieval_eval.py).

v2.0.0 (2026-10-17) — result-boundary parser. v1.x parsed the search output
LINE BY LINE, so a hit whose text spans dozens of lines was scored at the
line number of its last line, not at its result rank; recall@5/@10 and the
miss definition (``found_rank > 5``) were both wrong. v2 splits on result
boundaries (lines matching ``^\\[score\\] \\(point_id\\)``, the exact format
``_tool_search`` emits) and can dump every miss as JSONL with the full
question, gold answer, gold session ids and retrieved point ids/ranks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_COLLECTION = "hermes_memories_eval"
SEARCH_LIMIT = 10
SEED_WINDOW_CHARS = 2000
HARNESS_VERSION = "2.0.0-result-boundary"

LME_PATH = Path(
    "/home/gk/.cache/huggingface/hub/"
    "datasets--xiaowu0162--longmemeval-cleaned/snapshots/"
    "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json"
)
if not LME_PATH.exists():
    # The HF hub cache is pruned on some hosts; fall back to the stable copy
    # shipped with the eval tooling (identical file, sha256
    # 821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c).
    LME_PATH = Path.home() / ".hermes/state/qdrant/eval-data/longmemeval_oracle.json"

# _tool_search emits one header per hit: "[0.85] (point_id) text" where the
# text itself spans many lines. Scores are formatted "%.2f" by the plugin, so
# the two-decimal requirement keeps continuation lines from false-matching.
RESULT_HEAD = re.compile(r"^\[(-?\d+\.\d{2})\] \(([^)]+)\) ")


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


def parse_results(result: str) -> list[dict]:
    """Split qdrant_search output into RESULTS (not lines).

    v1.x enumerated ``result.splitlines()`` and took the line number as the
    rank; a hit at result rank 2 whose text spans 40 lines was scored as
    "rank 41". This splits on the result-boundary header instead and returns
    one record per hit with its true result rank.
    """
    out: list[dict] = []
    for line in result.splitlines():
        m = RESULT_HEAD.match(line)
        if m:
            out.append({
                "rank": len(out) + 1,
                "score": float(m.group(1)),
                "point_id": m.group(2),
                "text": line[m.end():],
            })
    return out


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


def run(
    collection: str = EVAL_COLLECTION,
    hybrid: bool = False,
    misses_out: str | None = None,
    keep: bool = False,
) -> dict:
    # Same import pattern as retrieval_eval.py — tests/conftest.py sets up
    # the path so plugins.memory.qdrant resolves without hermes_yaml.
    tests_dir = REPO_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import conftest  # noqa: F401
    from plugins.memory.qdrant import QdrantMemoryProvider, hybrid_requested

    # Construct from the LIVE plugin config (same as v1.x), guard against
    # the configured collection, then override collection + hybrid on the
    # instance. Passing the collection inside the config dict instead would
    # make the guard compare the argument against itself and always fire.
    provider = QdrantMemoryProvider()
    real_collection = provider._collection
    if collection == real_collection:
        raise SystemExit(
            f"refusing to run: eval collection {collection!r} equals the "
            f"configured collection {real_collection!r}"
        )
    provider._collection = collection
    provider._config["collection"] = collection
    provider._config["progress"] = "off"
    provider._hybrid_requested = hybrid_requested(hybrid)
    provider.initialize("eval")

    data = load_longmemeval(LME_PATH)
    print(f"LongMemEval: {len(data)} examples")
    print(f"collection: {collection}  hybrid: {hybrid}  "
          f"harness: {HARNESS_VERSION}")

    # Seed: ingest haystack sessions, track which point_ids are answer sessions
    print("seeding eval collection with haystack sessions...")
    t0 = time.monotonic()
    seeded = 0
    # answer_point_ids[i] = set of point_ids that contain the answer for example i
    answer_point_ids: list[set[str]] = []
    pid_to_sid: dict[str, str] = {}

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
                    "text": text[:SEED_WINDOW_CHARS],
                    "session_id": sid,
                })
                m = re.search(r'stored point (\S+)', result)
                if m:
                    pid = m.group(1)
                    seeded += 1
                    pid_to_sid[pid] = sid
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
    miss_records: list[dict] = []

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

        # v2: parse by RESULT BOUNDARY, not by line.
        hits = parse_results(result) if isinstance(result, str) else []
        found_rank = None
        for hit in hits:
            if hit["point_id"] in answer_point_ids[i]:
                found_rank = hit["rank"]
                break

        ranks.append(found_rank)
        if found_rank is None or found_rank > 5:
            miss_records.append({
                "harness_version": HARNESS_VERSION,
                "collection": collection,
                "hybrid": hybrid,
                "generated_at": datetime.now(UTC).isoformat(),
                "i": i,
                "question_id": ex.get("question_id", ""),
                "question_type": ex.get("question_type", ""),
                "question_style": ex.get("question_style", ""),
                "question": question,
                "answer": ex.get("answer", ""),
                "question_date": ex.get("question_date", ""),
                "answer_session_ids": ex.get("answer_session_ids", []),
                "haystack_session_ids": ex.get("haystack_session_ids", []),
                "found_rank": found_rank,
                "latency_ms": round(latency, 2),
                "retrieved": [
                    {
                        "rank": h["rank"],
                        "point_id": h["point_id"],
                        "score": h["score"],
                        "session_id": pid_to_sid.get(h["point_id"]),
                        "is_answer": h["point_id"] in answer_point_ids[i],
                    }
                    for h in hits
                ],
            })

        if (i + 1) % 100 == 0:
            print(f"  queried {i + 1}/{len(data)}")

    report = {
        "harness_version": HARNESS_VERSION,
        "corpus": len(data),
        "seeded": seeded,
        "seed_ms": seed_ms,
        "collection": collection,
        "hybrid": hybrid,
        "seed_window_chars": SEED_WINDOW_CHARS,
        "limit": SEARCH_LIMIT,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mrr": mrr(ranks),
        "ndcg@5": ndcg_at_k(ranks, 5),
        "latency_mean_ms": mean(latencies) if latencies else 0,
        "latency_p95_ms": percentile(latencies, 95) if latencies else 0,
        "misses_count": len(miss_records),
        "misses": [(r["question"][:80], r["found_rank"]) for r in miss_records[:10]],
    }

    print()
    print(f"harness:   {HARNESS_VERSION}")
    print(f"corpus:    {report['corpus']} examples, {report['seeded']} sessions seeded")
    print(f"window:    top-{report['limit']}  |  collection: {collection}  |  "
          f"hybrid: {hybrid}")
    print()
    print(f"  recall@1   {report['recall@1']:.3f}")
    print(f"  recall@5   {report['recall@5']:.3f}")
    print(f"  recall@10  {report['recall@10']:.3f}")
    print(f"  MRR        {report['mrr']:.3f}")
    print(f"  nDCG@5     {report['ndcg@5']:.3f}")
    print(f"  latency    mean {report['latency_mean_ms']:.1f} ms, "
          f"p95 {report['latency_p95_ms']:.1f} ms")
    if miss_records:
        shown = len(report["misses"])
        print(f"\n  misses ({len(miss_records)} total, showing {shown}):")
        for q, r in report["misses"]:
            print(f"    [{'miss' if r is None else f'rank {r}'}] {q}")

    if misses_out:
        out = Path(misses_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for rec in miss_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nwrote {out} ({out.stat().st_size} bytes, {len(miss_records)} misses)")

    if not keep:
        try:
            provider._client.delete_collection(collection)
        except Exception:
            pass

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LongMemEval eval")
    parser.add_argument("--collection", default=EVAL_COLLECTION,
                        help="eval collection to seed/query (default: %(default)s)")
    parser.add_argument("--hybrid", action="store_true",
                        help="enable hybrid BM25+dense search (default: dense-only)")
    parser.add_argument("--misses-out", default=None, metavar="PATH",
                        help="dump ALL misses as JSONL to PATH")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    run(collection=args.collection, hybrid=args.hybrid,
        misses_out=args.misses_out, keep=args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
