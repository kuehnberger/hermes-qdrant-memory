#!/usr/bin/env python
"""Known-answer recall evaluation for md_search over the qd-history corpus.

Every query below is answerable ONLY from this machine's own qd session history
and agent log — the exporter's output — not from the skills/docs/vault roots.
That isolation is the point: if a hit comes back with a ``qd-history/...`` path,
the auxiliary memory path really answered from real data.

Each case states the query plus the strings a correct hit must contain
(case-insensitive). Recall@k is "did any of the top k hits satisfy the case";
a Wilson 95% interval is reported because n is small and a bare point estimate
on ~20 cases overstates precision.

Run:
    python scripts/eval_md_recall.py                # lexical only
    python scripts/eval_md_recall.py --semantic     # add the fallback tier
    python scripts/eval_md_recall.py --root qd-history
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mdsearch  # noqa: E402

# (name, query, [required substrings], root-restrict or "")
CASES: list[tuple[str, str, list[str], str]] = [
    (
        "teknium1-review",
        "teknium1 review api_key written to disk",
        ["api_key"],
        "qd-history",
    ),
    (
        "pin-sha",
        "Bump qdrant plugin catalog pin",
        ["Bump qdrant plugin catalog pin"],
        "qd-history",
    ),
    (
        "forget-tool",
        "qdrant_forget dry run confirm",
        ["qdrant_forget"],
        "qd-history",
    ),
    (
        "createpullrequest",
        "CreatePullRequest permissions",
        ["CreatePullRequest"],
        "qd-history",
    ),
    (
        "md-search",
        "md_search FTS5",
        ["md_search"],
        "qd-history",
    ),
    (
        "word-differs-geschaeftigt",  # German: the corpus is German+English
        "geschaeftigt",
        [],
        "qd-history",
    ),
    (
        "diacritics",
        "Grüße",
        [],
        "",
    ),
    (
        "rebase",
        "rebase onto origin/main",
        [],
        "qd-history",
    ),
    (
        "eval-harness",
        "retrieval_eval recall@1",
        [],
        "qd-history",
    ),
    (
        "announcement",
        "announcement Discord plugins-skills-and-skins",
        [],
        "qd-history",
    ),
    (
        "session-db-recovery",
        "state.db sessions messages recover missing history",
        [],
        "qd-history",
    ),
    (
        "agentlog-plugin",
        "registered plugin",
        [],
        "qd-history",
    ),
]


def wilson(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% interval — the honest interval for small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def hits_satisfy(hits, needles: list[str]) -> bool:
    """True when some hit satisfies every required needle (or there are none)."""
    if not needles:
        return True
    for hit in hits:
        haystack = f"{hit.path} {hit.heading} {hit.snippet}".casefold()
        if all(n.casefold() in haystack for n in needles):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="known-answer recall eval for md_search")
    ap.add_argument("--root", default="qd-history", help="restrict to this corpus root")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--semantic", action="store_true",
                    help="also exercise the semantic fallback tier")
    ap.add_argument("--json", default=None, help="write results as JSON here")
    args = ap.parse_args(argv)

    if not mdsearch.index_is_present():
        print("no index yet — run scripts/md_ingest.py first", file=sys.stderr)
        return 2

    rows = []
    for name, query, needles, case_root in CASES:
        root = case_root or args.root
        t0 = time.perf_counter()
        hits = mdsearch.search_lexical(query, limit=args.limit, root=root or "")
        lexical_ms = (time.perf_counter() - t0) * 1000

        semantic_ms = None
        semantic_added: list[str] = []
        if args.semantic:
            t1 = time.perf_counter()
            try:
                import mdsemantic

                extra = mdsemantic.search_semantic(
                    query, limit=args.limit, root=root or ""
                )
                semantic_ms = (time.perf_counter() - t1) * 1000
                seen = {h.path for h in hits}
                semantic_added = [h.path for h in extra if h.path not in seen]
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                semantic_ms = (time.perf_counter() - t1) * 1000
                semantic_added = [f"UNAVAILABLE: {exc}"]

        # a case counts as recalled when the LEXICAL tier satisfied it; the
        # semantic tier is reported separately so the two are never conflated.
        ok = hits_satisfy(hits, needles)
        rows.append({
            "name": name, "query": query, "root": root,
            "hits": len(hits), "recalled": ok,
            "lexical_ms": round(lexical_ms, 1),
            "semantic_ms": None if semantic_ms is None else round(semantic_ms, 1),
            "semantic_added": semantic_added[:3],
            "top_path": hits[0].path if hits else "",
            "in_corpus": bool(hits) and hits[0].path.startswith(f"{root}/"),
            "needles": needles,
        })

    n = len(rows)
    at1 = sum(1 for r in rows if r["recalled"])
    lo, hi = wilson(at1, n)
    lat = sorted(r["lexical_ms"] for r in rows)
    p50 = lat[len(lat) // 2] if lat else 0.0

    print(f"{'case':<26} {'ok':<3} {'hits':<5} {'ms':>7}  top path")
    print("-" * 92)
    for r in rows:
        flag = "yes" if r["recalled"] else "NO"
        print(f"{r['name']:<26} {flag:<3} {r['hits']:<5} {r['lexical_ms']:>7}  "
              f"{r['top_path'][:52]}")
        if args.semantic and r["semantic_added"]:
            print(f"{'':<26} sem+ {r['semantic_added']}")

    in_corpus = sum(1 for r in rows if r["in_corpus"])
    print("-" * 92)
    print(f"lexical recall@{args.limit}: {at1}/{n} = {at1 / n:.3f} "
          f"(Wilson 95% CI {lo:.3f}-{hi:.3f})")
    print(f"top hit from the queried corpus root: {in_corpus}/{n}")
    print(f"lexical latency: p50 {p50:.1f}ms  min {lat[0]:.1f}ms  max {lat[-1]:.1f}ms"
          if lat else "no timings")
    if args.semantic:
        sem_times = [r["semantic_ms"] for r in rows if r["semantic_ms"] is not None]
        if sem_times:
            print(f"semantic fallback latency: p50 "
                  f"{sorted(sem_times)[len(sem_times) // 2]:.0f}ms")

    if args.json:
        import json as _json

        Path(args.json).write_text(_json.dumps(
            {"rows": rows, "recall_at_limit": at1 / n, "n": n,
             "wilson": [lo, hi], "in_corpus": in_corpus},
            indent=1), encoding="utf-8")
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
