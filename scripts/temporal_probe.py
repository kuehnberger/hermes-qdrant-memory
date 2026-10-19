#!/usr/bin/env python3
"""Temporal probe: does the `temporal` knob help when the query names a time?

WHY THIS EXISTS. The LongMemEval ablation cannot answer this question: its
questions rarely carry an explicit time expression, so the temporal window
almost never fires and the arm measures nothing. A feature that cannot be
measured on the available corpus needs a corpus built for it, or it should not
be claimed at all. This is that corpus.

DESIGN. N topics x K months. For each topic, one entry per month states a fact
that is semantically NEAR-IDENTICAL to the same topic's other months — only the
detail changes, so a retrieval that ignores time has to guess among K entries
that all look alike. Each question asks for a topic's detail AT a stated time.
A time-blind retriever lands the right month about 1/K of the time; a retriever
that honours the window has essentially one candidate.

The whole corpus is seeded into ONE session, because that is the only place the
feature can act: `prefetch` scopes recall to the current session, so temporal
filtering operates WITHIN a session. This probe therefore also measures the
feature's real reach, not an imaginary one.

    python3 scripts/temporal_probe.py            # off vs on, then clean up
    python3 scripts/temporal_probe.py --keep     # leave the scratch collection

Run it under the interpreter that has the plugin's deps. It never touches
`hermes_memories`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_COLLECTION = "hermes_temporal_probe"

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Each topic gets one detail per month. The details within a topic are
# deliberately all plausible for that topic, so semantics alone cannot pick the
# right month — only the date can.
TOPICS = {
    "gym session": ["legs", "arms", "back", "chest", "shoulders", "cardio",
                    "core", "glutes"],
    "book club": ["a mystery novel", "a history book", "a poetry collection",
                  "a sci-fi novel", "a biography", "a travel memoir",
                  "a philosophy book", "a short story anthology"],
    "dentist visit": ["a cleaning", "a filling", "a checkup", "a whitening",
                      "a crown", "an x-ray", "a root canal", "a sealant"],
    "sprint review": ["the payments epic", "the search rewrite",
                      "the onboarding flow", "the billing fix",
                      "the mobile release", "the API cleanup",
                      "the caching layer", "the audit log"],
    "hiking trip": ["the ridge trail", "the lake loop", "the summit path",
                    "the forest route", "the coastal walk", "the canyon trail",
                    "the waterfall path", "the meadow loop"],
    "cooking class": ["knife skills", "bread baking", "sushi rolling",
                      "pastry basics", "curry pastes", "pasta making",
                      "fermentation", "plating"],
    "car service": ["an oil change", "new brake pads", "a tyre rotation",
                    "a battery swap", "an alignment", "a coolant flush",
                    "new wipers", "a timing belt"],
    "piano lesson": ["scales", "a Chopin nocturne", "sight reading",
                     "chord inversions", "a Bach invention", "arpeggios",
                     "a jazz standard", "pedalling"],
}


def _load_provider():
    """Import the plugin the way the Hermes loader does (as a package)."""

    tests_dir = REPO_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import conftest  # noqa: F401  (registers plugins.memory.qdrant)
    from plugins.memory.qdrant import QdrantMemoryProvider

    return QdrantMemoryProvider


def _build_corpus(months: int):
    """Return (points, questions). points: [(text, epoch)]; questions: [(q, answer)]."""
    points: list[tuple[str, float]] = []
    questions: list[tuple[str, str]] = []
    for topic, details in TOPICS.items():
        for i in range(months):
            month = MONTHS[i]
            detail = details[i % len(details)]
            day = 10 + (i % 5)
            text = (f"On {day} {month} 2024 I had a {topic} and it covered "
                    f"{detail}.")
            # 2024-<month>-<day> 09:00 UTC
            epoch = time.mktime((2024, i + 1, day, 9, 0, 0, 0, 0, 0))
            points.append((text, epoch))
            questions.append((
                f"What did my {topic} cover in {month} 2024?",
                detail,
            ))
    return points, questions


def _score(hits: list[str], answer: str, k: int) -> int:
    return 1 if any(answer.lower() in h.lower() for h in hits[:k]) else 0


def _rank_of(hits: list[str], answer: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if answer.lower() in h.lower():
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="temporal knob probe")
    parser.add_argument("--months", type=int, default=8,
                        help="months per topic (default: %(default)s)")
    parser.add_argument("--collection", default=PROBE_COLLECTION)
    parser.add_argument("--keep", action="store_true",
                        help="leave the scratch collection in place")
    args = parser.parse_args(argv)

    Provider = _load_provider()
    provider = Provider()
    real = provider._collection
    if args.collection == real:
        raise SystemExit(f"refusing to run against the live collection {real!r}")
    provider._collection = args.collection
    provider._config["collection"] = args.collection
    provider._config["progress"] = "off"
    # Hybrid ON: that is the shipped recommended config, so temporal is measured
    # on top of the retrieval we actually recommend.
    provider._hybrid_requested = True
    provider._knob_temporal = False
    provider.initialize("temporal-probe")
    client = provider._client

    points, questions = _build_corpus(args.months)
    print(f"corpus: {len(TOPICS)} topics x {args.months} months "
          f"= {len(points)} entries")
    print(f"questions: {len(questions)}  |  collection: {args.collection}  "
          f"|  hybrid: {provider._hybrid_active}")

    session = "temporal-probe-session"
    try:
        # Seed: one session, explicit timestamps (the provider's own write path
        # stamps now(), which would erase the very dates under test).
        batch = []
        for i, (text, epoch) in enumerate(points):
            payload = {
                "text": text,
                "session_id": session,
                "timestamp": float(epoch),
                "source": "temporal_probe",
                "kind": "fact",
            }
            vector = {"dense": provider._embed(text)}
            if provider._hybrid_active:
                vector["bm25"] = provider._embed_sparse(text)
            batch.append({"id": i + 1, "vector": vector, "payload": payload})
        client.upsert(collection_name=args.collection, points=batch, wait=True)
        print(f"seeded {len(batch)} entries into one session\n")

        results = {}
        for label, knob in (("temporal OFF", False), ("temporal ON", True)):
            provider._knob_temporal = knob
            hits_at = {1: 0, 5: 0, 10: 0}
            rr = 0.0
            for query, answer in questions:
                out = provider.prefetch(query, session_id=session)
                hits = [ln.split("] ", 1)[1] for ln in out.splitlines()
                        if "] " in ln]
                for k in hits_at:
                    hits_at[k] += _score(hits, answer, k)
                rank = _rank_of(hits, answer)
                rr += (1.0 / rank) if rank else 0.0
            n = len(questions)
            results[label] = {
                "recall@1": hits_at[1] / n,
                "recall@5": hits_at[5] / n,
                "recall@10": hits_at[10] / n,
                "MRR": rr / n,
            }
            print(f"{label}: " + "  ".join(
                f"{k} {v:.3f}" for k, v in results[label].items()))

        off, on = results["temporal OFF"], results["temporal ON"]
        print("\ndelta (ON - OFF):")
        for k in off:
            print(f"  {k:9s} {on[k] - off[k]:+.3f}")

        chance = 1.0 / args.months
        print(f"\nchance level at recall@1 (1/{args.months} months): {chance:.3f}")
        print("A time-blind retriever lands the right month about that often; a")
        print("retriever that honours the window should be far above it.")
    finally:
        if not args.keep:
            try:
                client.delete_collection(args.collection)
                print(f"\ncleaned up {args.collection}")
            except Exception as e:  # pragma: no cover
                print(f"cleanup failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
