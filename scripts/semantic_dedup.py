#!/usr/bin/env python3
"""Semantic dedup over the live Qdrant collection.

WHY: the same fact arrives from several sessions, and near-identical points
crowd each other out of a fixed recall budget. memory-os merges points whose
cosine similarity exceeds a threshold; this is that pass.

WHAT IT DOES: fetches dense vectors, groups transitively-similar points with
``hygiene.cluster_near_duplicates`` (cosine >= ``--threshold``, default 0.92),
and proposes one merged payload per group with ``hygiene.merge_payloads``.
With ``--apply`` it keeps the strongest member and marks the others
``superseded_by: <kept id>`` + ``archived: true``, which the provider already
excludes from recall.

REVERSIBLE: nothing is deleted and no vector is rewritten; ``--unmerge
--apply`` clears the flags.

COST, STATED PLAINLY: the clustering is greedy first-fit in pure Python, so it
is O(n^2) in the points scanned — fine for thousands, hopeless for the whole
store. That is why --limit is REQUIRED unless --allow-unbounded is passed: a
297k-point scan would run for hours. The scalable path is a Qdrant-side
neighbour search per point (one query_points per point, threshold as a score
floor) instead of an all-pairs comparison; that is a follow-up, not this script.

    python3 scripts/semantic_dedup.py --limit 20000
    python3 scripts/semantic_dedup.py --limit 20000 --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from the repo root without installing: the hygiene primitives live beside
# this script's parent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hygiene import (  # noqa: E402
    cluster_near_duplicates,
    merge_payloads,
)


def _epoch(payload: dict) -> float | None:
    """Numeric epoch from a payload, or None.

    The store carries epoch floats AND (until the sync_turn fix) ISO strings, so
    both are accepted here. A missing or unparseable stamp returns None — never
    0.0, which would read as 1970 and be treated as the oldest point alive.
    """
    raw = (payload or {}).get("timestamp")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        from datetime import datetime

        try:
            return datetime.fromisoformat(
                raw.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _client(args: argparse.Namespace):
    from qdrant_client import QdrantClient

    return QdrantClient(url=args.url, api_key=args.api_key or None, timeout=60)


def _scroll(client, collection: str, batch: int = 512):
    """Yield points from the whole collection, in batches."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        yield from points
        if offset is None:
            return


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--url", default="http://localhost:6333",
                        help="Qdrant REST URL (default: %(default)s)")
    parser.add_argument("--api-key", default="",
                        help="Qdrant API key for a hosted/cloud store")
    parser.add_argument("--collection", default="hermes_memories",
                        help="collection to scan (default: %(default)s)")
    parser.add_argument("--apply", action="store_true",
                        help="WRITE the change; without this the run is a "
                             "dry-run report and nothing is modified")
    parser.add_argument("--batch", type=int, default=256,
                        help="points per write request (default: %(default)s)")
    return parser


def main() -> int:
    parser = _base_parser(__doc__ or "semantic dedup")
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="cosine similarity to treat as a duplicate "
                             "(default: %(default)s)")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan at most this many points (0 = all, which is "
                             "REFUSED unless --allow-unbounded: the clustering "
                             "is O(n^2) in pure Python)")
    parser.add_argument("--allow-unbounded", action="store_true",
                        help="permit an unbounded scan (slow — see the docstring)")
    parser.add_argument("--unmerge", action="store_true",
                        help="clear superseded_by/archived instead of setting them")
    args = parser.parse_args()

    if not args.limit and not args.allow_unbounded:
        parser.error(
            "--limit is required: the clustering is O(n^2) in pure Python, so an "
            "unbounded scan of a large store runs for hours. Pass --limit N to "
            "bound it, or --allow-unbounded if you really mean it."
        )

    client = _client(args)

    # hygiene.cluster_near_duplicates takes (embedding, payload) pairs and
    # returns groups OF PAYLOADS, so each payload carries its own point id.
    items: list[tuple[list[float], dict]] = []
    payloads: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=min(args.batch, 512),
            offset=offset,
            with_payload=True,
            with_vectors=["dense"],
        )
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                vector = vector.get("dense")
            if not vector:
                continue
            pid = str(point.id)
            payload = dict(point.payload or {})
            payload["id"] = pid
            payloads[pid] = payload
            items.append((list(vector), payload))
        if offset is None or (args.limit and len(items) >= args.limit):
            break

    # Newest first: the grouping is greedy first-fit, so the first member
    # defines the cluster centre and becomes the survivor. Newest-first keeps
    # the surviving point the current one.
    items.sort(key=lambda item: _epoch(item[1]) or 0.0, reverse=True)

    print(f"collection      : {args.collection}")
    print(f"points read     : {len(items)}")

    if args.unmerge:
        merged = [pid for pid, payload in payloads.items()
                  if payload.get("superseded_by")]
        for start in range(0, len(merged), args.batch):
            client.set_payload(collection_name=args.collection,
                               payload={"superseded_by": None, "archived": False},
                               points=merged[start:start + args.batch])
        print(f"unmerged        : {len(merged)} point(s)")
        return 0

    print(f"clustering {len(items)} points (O(n^2), bounded by --limit) ...")
    started = time.time()
    groups = cluster_near_duplicates(items, threshold=args.threshold)
    print(f"clustered in {time.time() - started:.1f}s")
    print(f"duplicate groups: {len(groups)}")
    print(f"points involved : {sum(len(g) for g in groups)}")

    if not groups:
        print("\nNothing to merge.")
        return 0

    for group in groups[:5]:
        kept = group[0].get("id")
        merged = merge_payloads(group)
        members = [g.get("id") for g in group]
        print(f"  group of {len(group)}: keep {kept} | "
              f"text {len(str(merged.get('text', '')))} chars | "
              f"members {members[:4]}{'...' if len(members) > 4 else ''}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to mark duplicates.")
        return 0

    marked = 0
    for group in groups:
        kept = str(group[0].get("id") or "")
        losers = [str(g.get("id")) for g in group[1:] if g.get("id")]
        for start in range(0, len(losers), args.batch):
            chunk = losers[start:start + args.batch]
            client.set_payload(collection_name=args.collection,
                               payload={"superseded_by": kept, "archived": True},
                               points=chunk)
            marked += len(chunk)
    print(f"marked superseded: {marked} point(s) across {len(groups)} group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
