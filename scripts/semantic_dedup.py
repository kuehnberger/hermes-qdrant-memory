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

COST, STATED PLAINLY: two paths. The default is greedy first-fit clustering in
pure Python (``hygiene.cluster_near_duplicates``), O(n^2) in the points scanned
— fine for thousands, hopeless for the whole store, which is why --limit is
REQUIRED unless --allow-unbounded is passed. The --scalable path instead asks
Qdrant for each point's near neighbours (one ``query_points`` per point, the
threshold as a score floor) and decides duplicates from those neighbour sets
(``hygiene.plan_neighbour_dedup``) without ever loading the store's dense
vectors in bulk, so a full ~393k-point run is feasible. It is APPROXIMATE
versus exact all-pairs: it keeps the direct-neighbour relation above threshold,
not its transitive closure — acceptable at the high default threshold (0.92)
and fully reversible (--unmerge --apply clears the flags).

    python3 scripts/semantic_dedup.py --limit 20000
    python3 scripts/semantic_dedup.py --limit 20000 --apply
    python3 scripts/semantic_dedup.py --scalable                 # full-store dry run
    python3 scripts/semantic_dedup.py --scalable --apply
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
    plan_neighbour_dedup,
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


def _retrieve_vectors(client, args, ids):
    """Return ``{id_str: dense_vector}`` for ``ids`` in one ``retrieve`` call.

    ``retrieve`` takes a list of ids and returns them with their named ``dense``
    vector; a point whose vector is absent yields nothing. Vectors are held
    only for this chunk, never the whole store.
    """
    vectors: dict[str, list[float]] = {}
    try:
        retrieved = client.retrieve(
            collection_name=args.collection,
            ids=list(ids),
            with_payload=False,
            with_vectors=["dense"],
        )
    except Exception as exc:  # noqa: BLE001 - a failed chunk must surface, not vanish
        # A dedup tool that silently returns "0 duplicates" because a chunk's
        # retrieve failed is a dangerous false negative — someone would trust
        # the 0 and conclude the store is clean. Surface it on stderr and keep
        # scanning; the id-type mismatch that first exposed this (int ids vs
        # the store's UUID strings) 400s here and was previously swallowed whole.
        import sys
        print(
            f"WARNING: retrieve failed for chunk {ids} — {exc!r}; "
            "those points yield no neighbour records",
            file=sys.stderr,
        )
        return vectors
    for point in retrieved:
        vector = point.vector
        if isinstance(vector, dict):
            vector = vector.get("dense")
        if vector:
            vectors[str(point.id)] = list(vector)
    return vectors


def _scalable_records(client, args, ids, epochs):
    """Build ``(id, epoch, neighbour ids)`` records via per-point neighbour search.

    For each id its dense vector is fetched (in chunks) and one
    ``query_points`` at ``--threshold`` returns its near neighbours. Batching
    the queries into a single ``query_points(query=[v1, v2, ...])`` is not an
    option here: qdrant_client 1.19.1 reads a list-of-lists query against a
    single-vector named field as a MULTIVECTOR (verified in-memory —
    "Multivector dense is not found"), so per-point queries are the supported
    path, each capped at ``--neighbours``. No dense vector is held in bulk.
    """
    records: list[tuple[str, float | None, list[str]]] = []
    ids = list(ids)
    for start in range(0, len(ids), args.chunk):
        chunk = ids[start:start + args.chunk]
        vectors = _retrieve_vectors(client, args, chunk)
        for cid in chunk:
            vector = vectors.get(cid)
            if vector is None:
                continue  # no dense vector — a singleton, not a centre
            found = client.query_points(
                collection_name=args.collection,
                query=vector,
                using="dense",
                score_threshold=args.threshold,
                limit=args.neighbours,
                with_payload=False,
                with_vectors=False,
            )
            neighbours = [str(p.id) for p in found.points if str(p.id) != cid]
            records.append((cid, epochs.get(cid), neighbours))
    return records


def _payloads_for(client, args, ids):
    """Fetch payloads (each with ``id`` set) for a merge preview, lazily."""
    if not ids:
        return []
    retrieved = client.retrieve(
        collection_name=args.collection,
        ids=list(ids),
        with_payload=True,
        with_vectors=False,
    )
    by_id: dict[str, dict] = {}
    for point in retrieved:
        payload = dict(point.payload or {})
        payload["id"] = str(point.id)
        by_id[str(point.id)] = payload
    return [by_id[str(i)] for i in ids if str(i) in by_id]


def _run_scalable(client, args):
    """Full-store dedup via per-point neighbour search (``--scalable``).

    Scrolls id+payload only (no vectors), builds a newest-first ordering, asks
    Qdrant for each point's neighbours, and lets ``plan_neighbour_dedup`` decide
    the survivors without loading the store's dense vectors in bulk. --limit
    bounds the scroll; without it the whole store is scanned, which is the
    point of this path.
    """
    epochs: dict[str, float | None] = {}
    order: list[str] = []
    superseded_ids: list[str] = []
    offset = None
    count = 0
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=min(args.batch, 512),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            pid = str(point.id)
            if pid in epochs:
                continue
            payload = dict(point.payload or {})
            epochs[pid] = _epoch(payload)
            order.append(pid)
            if payload.get("superseded_by"):
                superseded_ids.append(pid)
            count += 1
            if args.limit and count >= args.limit:
                offset = None
                break
        if offset is None or (args.limit and count >= args.limit):
            break

    print(f"collection      : {args.collection}")
    print(f"points read     : {len(epochs)}")
    print(f"scanning via    : per-point neighbour search "
          f"(--threshold {args.threshold}, K {args.neighbours})")

    if args.unmerge:
        for start in range(0, len(superseded_ids), args.batch):
            client.set_payload(
                collection_name=args.collection,
                payload={"superseded_by": None, "archived": False},
                points=superseded_ids[start:start + args.batch],
            )
        print(f"unmerged        : {len(superseded_ids)} point(s)")
        return 0

    print(f"neighbour search over {len(order)} points "
          f"(per-point query_points, capped at K={args.neighbours}) ...")
    started = time.time()
    records = _scalable_records(client, args, order, epochs)
    plan = plan_neighbour_dedup(records)
    groups = plan.groups  # keeper id -> older neighbour ids

    print(f"neighbour search  : {time.time() - started:.1f}s")
    print(f"duplicate groups  : {len(groups)}")
    print(f"points to mark    : {sum(len(v) for v in groups.values())}")

    if groups:
        for keeper, losers in list(groups.items())[:5]:
            member_ids = [keeper, *losers]
            payloads = _payloads_for(client, args, member_ids)
            merged = merge_payloads(payloads) if payloads else {}
            print(f"  group of {len(member_ids)}: keep {keeper} | "
                  f"text {len(str(merged.get('text', '')))} chars | "
                  f"members {member_ids[:4]}{'...' if len(member_ids) > 4 else ''}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to mark duplicates.")
        return 0

    marked = 0
    for keeper, losers in groups.items():
        for start in range(0, len(losers), args.batch):
            chunk = losers[start:start + args.batch]
            client.set_payload(
                collection_name=args.collection,
                payload={"superseded_by": keeper, "archived": True},
                points=chunk,
            )
            marked += len(chunk)
    print(f"marked superseded: {marked} point(s) across {len(groups)} group(s)")
    return 0


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
    parser.add_argument("--scalable", action="store_true",
                        help="full-store neighbour-search path (feasible for "
                             "the whole store) instead of the O(n^2) bounded "
                             "clustering; --limit then only bounds the scroll")
    parser.add_argument("--neighbours", type=int, default=64,
                        help="per-point neighbour cap K in scalable mode "
                             "(default: %(default)s)")
    parser.add_argument("--chunk", type=int, default=256,
                        help="points per vector retrieve batch in scalable "
                             "mode (default: %(default)s)")
    args = parser.parse_args()

    if not args.scalable and not args.limit and not args.allow_unbounded:
        parser.error(
            "--limit is required: the clustering is O(n^2) in pure Python, so an "
            "unbounded scan of a large store runs for hours. Pass --limit N to "
            "bound it, --scalable for the feasible neighbour-search path, or "
            "--allow-unbounded if you really mean it."
        )

    client = _client(args)

    if args.scalable:
        return _run_scalable(client, args)

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
