#!/usr/bin/env python3
"""Decay scan over the live Qdrant collection.

WHY: the store only grows. memory-os retires stale machine-generated points on
a weekly schedule; without an equivalent, a 300k-point collection keeps every
cron turn forever and recall pays for all of it.

WHAT IT DOES: scrolls the collection, scores each point with
``hygiene.decay_score(age_days)``, exempts whatever ``hygiene.is_decay_exempt``
protects, and reports the distribution. With ``--apply`` it sets
``archived: true`` on points below ``--threshold``.

REVERSIBLE: the only write is a payload flag. No point is deleted and no vector
is rewritten, so ``--unarchive --apply`` restores the previous state exactly.
The provider excludes archived points from recall, which is what makes
archiving mean anything.

DRY RUN BY DEFAULT: without ``--apply`` nothing is written.

    python3 scripts/decay_scan.py
    python3 scripts/decay_scan.py --apply --threshold 0.05
    python3 scripts/decay_scan.py --unarchive --apply
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
    decay_score,
    is_decay_exempt,
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
    parser = _base_parser(__doc__ or "decay scan")
    parser.add_argument("--half-life-days", type=float, default=180.0,
                        help="decay half-life in days (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="archive below this decay score (default: %(default)s)")
    parser.add_argument("--importance", type=float, default=0.7,
                        help="importance at or above this is never archived "
                             "(default: %(default)s)")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan at most this many points (0 = all)")
    parser.add_argument("--unarchive", action="store_true",
                        help="clear the archived flag instead of setting it")
    args = parser.parse_args()

    client = _client(args)
    now = time.time()

    considered = exempt = no_timestamp = archive_ids = 0
    buckets: dict[str, int] = {}
    stale_ids: list[str] = []

    for point in _scroll(client, args.collection):
        if args.limit and considered >= args.limit:
            break
        considered += 1
        payload = point.payload or {}
        if is_decay_exempt(payload, importance_threshold=args.importance):
            exempt += 1
            continue
        stamp = _epoch(payload)
        if stamp is None:
            # No timestamp is not evidence of staleness — leave it alone.
            no_timestamp += 1
            continue
        age_days = max(0.0, (now - stamp) / 86400.0)
        score = decay_score(age_days, args.half_life_days)
        bucket = f"{(int(score * 10) / 10):.1f}"
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if score < args.threshold:
            stale_ids.append(str(point.id))

    print(f"collection      : {args.collection}")
    print(f"points scanned  : {considered}")
    print(f"exempt          : {exempt}  (human-authored / pinned / important)")
    print(f"no timestamp    : {no_timestamp}  (kept — absence is not staleness)")
    print(f"below threshold : {len(stale_ids)}  (score < {args.threshold})")
    print("decay buckets   : " + ", ".join(
        f"{k}:{v}" for k, v in sorted(buckets.items())))

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to mark these.")
        return 0

    if args.unarchive:
        ids, value = ([str(p.id) for p in _scroll(client, args.collection)
                       if (p.payload or {}).get("archived")], False)
        print(f"\nunarchiving {len(ids)} point(s)")
    else:
        ids, value = stale_ids, True

    for start in range(0, len(ids), args.batch):
        chunk = ids[start:start + args.batch]
        client.set_payload(collection_name=args.collection,
                           payload={"archived": value}, points=chunk)
        archive_ids += len(chunk)
    print(f"archived flag set to {value} on {archive_ids} point(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
