#!/usr/bin/env python
"""Ingest the markdown corpus into the FTS5 index and (optionally) the docs collection.

This runs as a TRANSIENT process — a one-shot CLI invocation that loads the
embedding model, does its work and exits. It is never on the gateway's hot path,
and nothing in the provider's live path shells out to it: a full corpus ingest
is ~52k chunks (measured 51,670 over 2,678 files on 2026-10-03) ≈ 65 min at the
measured 10-14 chunks/s on 5 cores, which no agent turn may pay for.

Ingest is SHA-incremental: a file whose bytes are unchanged is skipped without
being read into chunks. A file whose bytes CHANGED has both tiers rewritten —
lexically by ``replace_file_chunks`` (drop-and-insert per file) and, with
``--semantic``, by deleting that file's points BEFORE embedding the new ones
(delete-before-write): an upsert only overwrites ids that still exist, so a
section deleted from the file would otherwise keep its old point forever.
Deleted files have their index rows removed by ``--prune``, which with
``--semantic`` also removes their points.

Usage:
    python scripts/md_ingest.py                 # lexical only (fast, no model)
    python scripts/md_ingest.py --semantic      # also embed into hermes_md_docs
    python scripts/md_ingest.py --rebuild      # ignore SHAs, re-chunk everything
    python scripts/md_ingest.py --root skills  # restrict to one corpus root
    python scripts/md_ingest.py --status        # report only, change nothing

``--lexical-only`` is the default on purpose: the FTS5 index is what makes
``md_search`` answer in milliseconds, it needs no model and no server, and it is
the only tier required for the tool to be useful. The semantic tier is an
enhancement for concept queries, and it is the only part that costs RAM.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# A standalone script cannot rely on the caller having the plugin importable,
# and the repo-root import must work from the plugin directory itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mdsearch import (  # noqa: E402
    chunk_text,
    connect,
    db_path,
    drop_file,
    file_row,
    iter_markdown,
    labels_in_index,
    read_meta,
    replace_file_chunks,
    sha_of,
    write_meta,
)


def _load_semantic():
    try:
        import mdsemantic
        return mdsemantic
    except ImportError as exc:
        print(f"semantic ingest unavailable: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest markdown into the md-search index")
    ap.add_argument("--semantic", action="store_true",
                    help="also embed chunks into the hermes_md_docs collection")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-chunk every file, ignoring the SHA short-circuit")
    ap.add_argument("--root", action="append", default=None,
                    help="restrict to one corpus root (repeatable)")
    ap.add_argument("--status", action="store_true",
                    help="print index statistics and exit without changing anything")
    ap.add_argument("--prune", action="store_true",
                    help="drop index rows for files that no longer exist "
                         "(and their points, when --semantic is given)")
    args = ap.parse_args(argv)

    if not db_path().is_file():
        print(f"no index at {db_path()} — nothing to report yet", file=sys.stderr)
        if args.status:
            return 0
    con = connect()
    try:
        if args.status:
            files = con.execute("SELECT count(*), coalesce(sum(chunks), 0) FROM files").fetchone()
            print(f"index:    {db_path()}")
            print(f"files:    {files[0]}")
            print(f"chunks:   {files[1]}")
            print(f"model:    {read_meta(con, 'model') or '(none — lexical only)'}")
            print(f"last run: {read_meta(con, 'last_ingest') or '(never)'}")
            return 0

        roots = None
        if args.root:
            from mdsearch import DEFAULT_ROOTS
            roots = {k: v for k, v in DEFAULT_ROOTS.items() if k in set(args.root)}
            if not roots:
                print(f"no such root(s): {', '.join(args.root)}; known: "
                      f"{', '.join(DEFAULT_ROOTS)}", file=sys.stderr)
                return 2

        semantic = _load_semantic() if args.semantic else None
        if args.semantic and semantic is None:
            return 2
        embedder = None
        client = None
        if semantic is not None:
            embedder = semantic.make_embedder()
            from qdrant_client import QdrantClient

            url = (os.environ.get("QDRANT_URL") or "http://localhost:6333").strip()
            client = QdrantClient(url=url, prefer_grpc=False, timeout=30)

        started = time.time()
        indexed = skipped = removed = 0
        chunks_total = 0
        seen: set[str] = set()

        for cf in iter_markdown(roots):
            seen.add(cf.label)
            existing = file_row(con, cf.label)
            sha = sha_of(cf.path)
            if not sha:
                continue
            if existing and existing[1] == sha and not args.rebuild:
                skipped += 1
                continue
            try:
                text = Path(cf.path).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"skip {cf.label}: {exc}", file=sys.stderr)
                continue
            chunks = chunk_text(text, cf.label)
            if semantic is not None:
                # delete-before-write. `upsert_chunks` only overwrites ids that
                # still exist, so a section this edit REMOVED would keep its
                # point (with the old sha) forever without this call. Doing it
                # unconditionally — not only when `chunks` is non-empty — is
                # what lets a file that became empty shed its points too.
                semantic.delete_file_points(client, cf.label)
                if chunks:
                    rows = [
                        (cf.label, heading, content, sha)
                        for heading, content in chunks
                    ]
                    semantic.upsert_chunks(client, rows, embedder=embedder)
            chunks_total += replace_file_chunks(con, cf.label, cf.label.split("/", 1)[0], sha, chunks)
            indexed += 1
            if indexed % 50 == 0:
                con.commit()
                print(f"  ... {indexed} indexed, {skipped} unchanged "
                      f"({time.time() - started:.0f}s)", flush=True)

        if args.prune:
            for label in sorted(labels_in_index(con) - seen):
                drop_file(con, label)
                removed += 1
                # Lexical-only runs never open a Qdrant client (a dead server
                # must not block a prune), but a run that OWNS the semantic
                # tier has to clear it too — otherwise a deleted file keeps
                # serving vector hits for text that no longer exists.
                if semantic is not None:
                    semantic.delete_file_points(client, label)

        write_meta(con, "last_ingest", time.strftime("%Y-%m-%d %H:%M:%S"))
        if semantic is not None:
            write_meta(con, "model", semantic.DOCS_MODEL)
        con.commit()

        elapsed = time.time() - started
        print(f"indexed {indexed} file(s), {chunks_total} chunk(s); "
              f"{skipped} unchanged, {removed} pruned in {elapsed:.1f}s")
        if semantic is not None:
            try:
                client.close()
            except Exception:
                pass
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
