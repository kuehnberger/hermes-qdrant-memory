#!/usr/bin/env python3
"""Retrieval-quality eval harness for the Qdrant memory provider.

The gap this closes: we CLAIM the provider recalls memories well; this
measures it. Synthetic memory corpus (paraphrased recall queries, one
unambiguous target each), driven through the REAL tool path —
``handle_tool_call`` for both seeding and search, so the numbers cover
embedding, the named-vector query, and the exact formatting the model sees.

Metrics follow the set adopted from the competitor analysis
(``docs/competitor-analysis-entropicmem.md``): recall@k, MRR, nDCG@5 —
the subset that describes a retrieval tool without an injection screen
or prefetch pipeline. Latency is reported because recall that arrives too
slow is not recall.

Safety:
  * writes go ONLY to a dedicated ``hermes_memories_eval`` collection,
    created with the production schema (the provider's own ``initialize()``
    does the creating), and dropped in ``finally`` unless ``--keep``.
  * refuses to run if the eval name ever equals the configured collection —
    the production store must be untouchable from this script.

Usage (from the repo root, any Python that can import the plugin):

    python scripts/retrieval_eval.py                  # run + report
    python scripts/retrieval_eval.py --keep           # leave eval collection up
    python scripts/retrieval_eval.py --min-recall5 0.9  # exit 1 below 0.90

Exit codes: 0 ok, 1 below threshold, 2 setup failure.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_COLLECTION = "hermes_memories_eval"
SEARCH_LIMIT = 10  # metric set tops out at recall@10

# --------------------------------------------------------------------------
# Dataset — (memory, query) pairs. Memory-style facts about one fictional
# user, the actual retrieval shape of a memory provider: the query is how
# the model would ASK, the memory is how it was STORED — same fact,
# different wording. Every target is unique within the corpus.
# --------------------------------------------------------------------------

CASES: list[tuple[str, str]] = [
    ("The user drinks espresso in the morning but switches to herbal tea after 15:00.",
     "What coffee does the user drink, and when do they stop?"),
    ("User's dog is a border collie called Pixel.",
     "What breed is the user's dog and what is his name?"),
    ("Design standup moved from Monday to Tuesday at 10:00.",
     "When does the design team standup happen now?"),
    ("User is allergic to penicillin — never recommend it.",
     "Which medication must be avoided for this user?"),
    ("The user's sister Maren lives in Lisbon and visits every August.",
     "Where does the user's sister live and when does she visit?"),
    ("User always asks for a window seat when booking flights.",
     "Which seat does the user prefer on planes?"),
    ("Project Kestrel's deadline is 12 November; the client is Halden AG.",
     "Who is the client for Project Kestrel and when is it due?"),
    ("The user's anniversary is 3 June — remind them two weeks ahead, not earlier.",
     "When is the user's wedding anniversary?"),
    ("The spare key to the user's flat is with neighbour Tomas, flat 4B.",
     "Where is the spare key to the user's apartment?"),
    ("User broke their left wrist in February 2024 and dislikes "
     "carrying heavy bags.",
     "What old injury affects how the user handles luggage?"),
    ("Weekly budget review with finance happens Thursdays at 14:30.",
     "What time is the budget review with the finance team?"),
    ("The user's favourite hiking trail is the Kaç mountains ridge route.",
     "Which hiking route does the user like best?"),
    ("User's mother's name is Ingrid; send birthday cards to the Zürich address.",
     "What is the user's mother called and where should cards go?"),
    ("The studio printer jams if loaded with anything above 120 gsm paper.",
     "Why does the office printer keep jamming?"),
    ("User gave up alcohol in 2022 — never suggest beer or wine pairings.",
     "Does the user drink alcohol?"),
    ("Project data lives in the eu-central Qdrant cluster; never copy it to us-east.",
     "Where must the project data be stored?"),
    ("The user's car is an electric Volvo, charging capped at 80% "
     "to protect the battery.",
     "What is the charging limit on the user's car?"),
    ("User's favourite encoding model is all-MiniLM-L6-v2 for its 384-dimension speed.",
     "Which embedding model does the user prefer and why?"),
    ("Gym sessions are booked for Wednesdays and Saturdays at 07:00.",
     "Which days does the user work out?"),
    ("The user's passport expires in September 2027 — renew by June.",
     "When does the user's passport expire?"),
    ("User writes tests before implementation; never merge without green CI.",
     "What is the user's workflow preference for writing code?"),
    ("The cafe on Elm Street makes the oat flat white the user likes.",
     "Where does the user get their favourite oat milk drink?"),
    ("User's laptop is a ThinkPad X13; dock firmware must stay on 2.1.",
     "Which laptop does the user use and what dock firmware version is required?"),
    ("Invoice payments to Bergström AS run on the 5th of each month.",
     "When are the Bergström AS invoices paid?"),
    ("The user prefers plain text email; HTML newsletters get flagged as spam.",
     "What email format does the user want?"),
    ("User's daughter Alma is starting school in August 2026.",
     "When does the user's daughter start school?"),
    ("Backup verification runs the first Monday of every month at 09:00.",
     "When are backups verified?"),
    ("The user's favourite pub quiz team name is 'The Mnemonists'.",
     "What does the user call their pub quiz team?"),
    ("User set the office thermostat to 21°C and dislikes temperatures above 23.",
     "What temperature does the user keep the office at?"),
    ("The QGIS mapping workshop is postponed to 28 October, room B12.",
     "When and where is the mapping workshop now?"),
    ("User reviews kernel code in Gerrit, not GitHub PRs.",
     "Which code review system does the user use for kernel work?"),
    ("The user is vegetarian but eats fish on Fridays.",
     "What dietary restrictions does the user have?"),
    ("Night shift handover notes go to the on-call alias, never to the team channel.",
     "Where should night shift handover notes be sent?"),
    ("User's bank requires the TAN list refreshed every 90 days.",
     "How often must the user's banking TAN list be renewed?"),
    ("The aquarium needs salinity checked twice a week — Tuesday and Friday.",
     "When should the aquarium's salt level be tested?"),
    ("User's running goal is a half marathon in Rotterdam, April 2027.",
     "What race is the user training for and when?"),
]

# --------------------------------------------------------------------------
# Metrics — pure functions, unit-tested offline (tests/test_retrieval_eval.py).
# rank is 1-based; None = target not in the returned window.
# --------------------------------------------------------------------------


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of queries whose target ranked within the top-k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank; a miss inside the window scores 0."""
    if not ranks:
        return 0.0
    return mean(1.0 / r if r is not None else 0.0 for r in ranks)


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """nDCG@k with a single relevant document per query: DCG = 1/log2(rank+1),
    IDCG = 1 (the ideal puts the one relevant doc first), so nDCG is the
    discounted gain itself, 0 for a miss beyond k."""
    if not ranks:
        return 0.0
    gains = []
    for r in ranks:
        if r is not None and r <= k:
            gains.append(1.0 / math.log2(r + 1))
        else:
            gains.append(0.0)
    return mean(gains)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation surprises)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


# --------------------------------------------------------------------------
# Runner — heavy imports happen here, never at module import time (keeps the
# fast test suite from pulling the ML stack; see the CI-budget note in the
# competitor analysis).
# --------------------------------------------------------------------------

_LINE = re.compile(r"^\[([\d.]+)\] (.*)$", re.MULTILINE)


def _parse_hits(output: str) -> list[tuple[float, str]]:
    """Format produced by _tool_search: '[0.72] <text>' per line."""
    return [(float(m.group(1)), m.group(2)) for m in _LINE.finditer(output)]


def run(keep: bool = False) -> dict:
    """Seed, query, score, clean up. Returns the metric report dict."""
    tests_dir = REPO_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    # Both imports resolve at runtime through the sys.path entry just
    # installed — the same bootstrap the test suite uses.
    import conftest  # noqa: F401
    from plugins.memory.qdrant import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    real_collection = provider._collection
    if EVAL_COLLECTION == real_collection:
        raise SystemExit(
            f"refusing to run: eval collection {EVAL_COLLECTION!r} equals the "
            f"configured collection {real_collection!r} — this script must "
            f"never write the production store"
        )

    provider._collection = EVAL_COLLECTION
    try:
        provider.initialize("eval")
    except Exception as e:
        raise SystemExit(f"setup failed: {e}") from e

    ranks: list[int | None] = []
    latencies: list[float] = []
    misses: list[tuple[str, int | None]] = []

    try:
        # Seed through the real tool path (embed + named-vector upsert).
        for memory, _query in CASES:
            out = provider.handle_tool_call(
                "qdrant_upsert", {"text": memory, "session_id": "eval"}
            )
            if "stored point" not in out:
                raise SystemExit(f"seeding failed: {out}")

        # Query through the real tool path; score the exact text the model sees.
        for memory, query in CASES:
            t0 = time.perf_counter()
            out = provider.handle_tool_call(
                "qdrant_search", {"query": query, "limit": SEARCH_LIMIT}
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            hits = _parse_hits(out)
            rank = next(
                (i + 1 for i, (_score, text) in enumerate(hits) if text == memory),
                None,
            )
            ranks.append(rank)
            if rank is None or rank > 5:
                misses.append((query, rank))

        report = {
            "corpus": len(CASES),
            "limit": SEARCH_LIMIT,
            "recall@1": recall_at_k(ranks, 1),
            "recall@5": recall_at_k(ranks, 5),
            "recall@10": recall_at_k(ranks, 10),
            "mrr": mrr(ranks),
            "ndcg@5": ndcg_at_k(ranks, 5),
            "latency_mean_ms": mean(latencies),
            "latency_p95_ms": percentile(latencies, 95),
            "misses": misses[:5],
        }
        return report
    finally:
        if not keep:
            try:
                provider._client.delete_collection(EVAL_COLLECTION)
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Retrieval eval").split("\n", 1)[0]
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="leave the eval collection in place for inspection",
    )
    parser.add_argument(
        "--min-recall5", type=float, default=None, metavar="F",
        help="exit 1 if recall@5 falls below F (report-only when omitted)",
    )
    args = parser.parse_args(argv)

    report = run(keep=args.keep)

    print(f"corpus: {report['corpus']} memories, one paraphrased query each")
    print(f"window: top-{report['limit']}  |  collection: {EVAL_COLLECTION}")
    print()
    print(f"  recall@1   {report['recall@1']:.3f}")
    print(f"  recall@5   {report['recall@5']:.3f}")
    print(f"  recall@10  {report['recall@10']:.3f}")
    print(f"  MRR        {report['mrr']:.3f}")
    print(f"  nDCG@5     {report['ndcg@5']:.3f}")
    print(f"  latency    mean {report['latency_mean_ms']:.1f} ms, "
          f"p95 {report['latency_p95_ms']:.1f} ms")
    if report["misses"]:
        print("\n  queries missed or ranked >5:")
        for query, rank in report["misses"]:
            print(f"    [{'miss' if rank is None else f'rank {rank}'}] {query}")

    if args.min_recall5 is not None and report["recall@5"] < args.min_recall5:
        print(f"\nFAIL: recall@5 {report['recall@5']:.3f} < "
              f"threshold {args.min_recall5:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
