#!/usr/bin/env python
"""Backfill sparse (bm25) vectors onto an existing dense-only collection.

Enabling ``hybrid: on`` on a store that already holds points is two steps:

1. **Schema** — the provider does this itself on the next ``initialize()``
   (Qdrant >= 1.18 ``create_vector_name``), or this script does it if you
   pass ``--add-schema``.
2. **Values** — existing points have no sparse value until they are written
   again. This script is that pass: it reads each point's stored ``text``
   payload, computes the bm25 sparse vector with the same model the live
   write path uses, and attaches it with ``update_vectors`` (dense vector and
   payload untouched).

Why ``update_vectors`` and not a re-upsert: it replaces only the named
vector, so the dense vector never leaves the server and cannot be corrupted
by a re-embed. Verified on Qdrant 1.19.2 — after ``update_vectors`` the point
carries ``{'dense': list, 'bm25': SparseVector}``.

Resumable: a point that already has a bm25 value is skipped, so an
interrupted run is restarted by re-running the same command. Progress is
printed per batch and the run reports how many were backfilled vs skipped.

Cost, measured on this host against a 130k-point store: ~3,400 sparse
embeds/s → about one minute of CPU for the whole collection. The work is
CPU-bound, not memory-bound; it is safe on a small box.

Usage:
    python scripts/migrate_hybrid.py --collection hermes_memories --dry-run
    python scripts/migrate_hybrid.py --collection hermes_memories --yes
    python scripts/migrate_hybrid.py --collection hermes_memories --yes \\
        --url http://localhost:6333 --batch-size 256
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from typing import Any

#: Points per read page and per sparse-embed batch. 256 keeps the embedder
#: fed without holding a large vector list in memory (this box has ~7 GB).
DEFAULT_BATCH_SIZE = 256

#: Payload key holding the text that was embedded. The provider writes
#: ``text`` on every write path (sync_turn, _tool_upsert, and the bulk
#: importers), truncated to 2000 chars — the same truncation is applied here
#: so a backfilled vector matches what a live write would have produced.
TEXT_KEY = "text"
MAX_TEXT_CHARS = 2000


def _sparse_embedder(model_name: str):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name)


def _embed_sparse(embedder, text: str):
    from qdrant_client.models import SparseVector

    sv = next(embedder.embed(text))
    return SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def _iter_points(client, collection: str, batch_size: int,
                 with_sparse: bool) -> Iterator[list[Any]]:
    """Yield pages of points, newest-scroll via ``next_page_offset``.

    ``with_vectors=[SPARSE]`` is what makes resumability cheap: a point that
    already carries the sparse vector comes back with it populated, one that
    does not comes back with ``{}`` — no separate existence query per point.
    """
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=[TEXT_KEY],
            with_vectors=[SPARSE_VECTOR_NAME] if with_sparse else False,
        )
        if points:
            yield points
        if offset is None:
            break


#: Kept in step with the provider constant of the same name. Duplicated
#: rather than imported so the script runs against an installed plugin
#: directory without depending on its import path.
SPARSE_VECTOR_NAME = "bm25"
SPARSE_MODEL_NAME = "Qdrant/bm25"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Backfill bm25 sparse vectors onto an existing collection"
    )
    ap.add_argument("--collection", required=True,
                    help="collection to backfill (e.g. hermes_memories)")
    ap.add_argument("--url", default="http://localhost:6333",
                    help="Qdrant URL (default: http://localhost:6333)")
    ap.add_argument("--api-key", default=None,
                    help="Qdrant API key (Qdrant Cloud only)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"points per page/batch (default: {DEFAULT_BATCH_SIZE})")
    ap.add_argument("--add-schema", action="store_true",
                    help="also add the sparse vector to the schema if missing "
                         "(Qdrant >= 1.18); the provider does this on "
                         "initialize() too")
    ap.add_argument("--model", default=SPARSE_MODEL_NAME,
                    help=f"sparse model (default: {SPARSE_MODEL_NAME})")
    ap.add_argument("--yes", action="store_true",
                    help="actually write; without it the run is a dry-run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing (default)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N points (0 = all); useful for a smoke test")
    args = ap.parse_args(argv)

    write = args.yes and not args.dry_run

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as m
    except ImportError as e:
        print(f"qdrant-client is not installed: {e}", file=sys.stderr)
        return 2

    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=60)

    # -- 1. Refuse to run against a collection that does not exist -----------
    try:
        names = {c.name for c in client.get_collections().collections}
    except Exception as e:
        print(f"cannot reach Qdrant at {args.url}: {e}", file=sys.stderr)
        return 2
    if args.collection not in names:
        print(f"no collection {args.collection!r} at {args.url}", file=sys.stderr)
        return 2

    info = client.get_collection(args.collection)
    total = info.points_count or 0
    declared = info.config.params.sparse_vectors or {}
    print(f"collection {args.collection!r}: {total:,} points, "
          f"sparse vectors declared: {sorted(declared) or 'none'}")

    # -- 2. Schema -----------------------------------------------------------
    if SPARSE_VECTOR_NAME not in declared:
        if not args.add_schema:
            print(
                f"\n{SPARSE_VECTOR_NAME!r} is not in the collection schema.\n"
                f"Run with --add-schema (Qdrant >= 1.18), or set hybrid: on and\n"
                f"let the provider add it on the next initialize().",
                file=sys.stderr,
            )
            return 3
        if not write:
            print(f"[dry-run] would add sparse vector {SPARSE_VECTOR_NAME!r} "
                  f"to the schema")
        else:
            try:
                client.create_vector_name(
                    collection_name=args.collection,
                    vector_name=SPARSE_VECTOR_NAME,
                    vector_name_config=m.SparseVectorNameConfig(
                        sparse=m.SparseVectorConfig(modifier=m.Modifier.IDF)
                    ),
                )
                print(f"added sparse vector {SPARSE_VECTOR_NAME!r} to schema")
            except Exception as e:
                print(f"could not add sparse vector: {e}", file=sys.stderr)
                return 3

    # -- 3. Backfill ---------------------------------------------------------
    embedder = None
    if write:
        try:
            embedder = _sparse_embedder(args.model)
        except Exception as e:
            print(f"sparse model {args.model!r} unavailable: {e}", file=sys.stderr)
            return 2

    scanned = backfilled = skipped = missing_text = failed = 0
    no_sparse_text = 0
    started = time.time()

    for page in _iter_points(client, args.collection, args.batch_size,
                             with_sparse=write):
        updates = []
        for p in page:
            scanned += 1
            if args.limit and scanned > args.limit:
                break
            # Resumable: a point that already carries the sparse vector is
            # done. On a dry-run we cannot read vectors cheaply, so the count
            # is reported as "candidates" instead.
            if write:
                have = getattr(p, "vector", None) or {}
                if SPARSE_VECTOR_NAME in have:
                    skipped += 1
                    continue
            text = (p.payload or {}).get(TEXT_KEY) or ""
            text = text.strip()
            if not text:
                missing_text += 1
                continue
            if not write:
                backfilled += 1  # "would backfill"
                continue
            try:
                sv = _embed_sparse(embedder, text[:MAX_TEXT_CHARS])
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  sparse embed failed for {p.id}: {e}", file=sys.stderr)
                continue
            # A stopword-only text ("how are you doing?") tokenizes to an
            # EMPTY bm25 vector. Qdrant's update_vectors rejects that with
            # 422 "must specify vectors to update for point", so such a point
            # is left dense-only — which is correct, the sparse arm has
            # nothing to match. (A full upsert tolerates the empty vector;
            # update_vectors does not — verified on 1.19.2.)
            if not len(sv.indices):
                no_sparse_text += 1
                continue
            updates.append(m.PointVectors(id=p.id, vector={SPARSE_VECTOR_NAME: sv}))

        if write and updates:
            try:
                client.update_vectors(collection_name=args.collection,
                                      points=updates, wait=True)
                backfilled += len(updates)
            except Exception as e:
                # One rejected point must not cost the other 255 in the
                # batch: retry individually and keep the successes.
                print(f"  batch failed ({str(e)[:120]}); retrying per point",
                      file=sys.stderr)
                for pv in updates:
                    try:
                        client.update_vectors(collection_name=args.collection,
                                              points=[pv], wait=True)
                        backfilled += 1
                    except Exception:
                        failed += 1

        if scanned and scanned % (args.batch_size * 4) == 0:
            rate = scanned / max(time.time() - started, 1e-9)
            print(f"  ... scanned {scanned:,}/{total:,} "
                  f"({rate:,.0f} pts/s, backfilled {backfilled:,})")

        if args.limit and scanned > args.limit:
            break

    elapsed = time.time() - started
    verb = "backfilled" if write else "would backfill"
    print(f"\n{verb}: {backfilled:,} points")
    if write:
        print(f"already had sparse: {skipped:,}")
    print(f"no text payload:    {missing_text:,}")
    if no_sparse_text:
        print(f"stopword-only text: {no_sparse_text:,} (left dense-only)")
    if failed:
        print(f"failed:             {failed:,}")
    print(f"scanned:            {scanned:,} / {total:,}")
    print(f"elapsed:            {elapsed:.1f}s")

    if not write:
        print("\n(dry-run — nothing written; re-run with --yes to apply)")

    # -- 4. Verify -----------------------------------------------------------
    if write and backfilled:
        after = client.get_collection(args.collection)
        print(f"points after: {after.points_count:,} "
              f"(unchanged: {after.points_count == total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
