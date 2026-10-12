#!/usr/bin/env python
"""Prefill the memory store from a profile's existing history.

A memory provider only knows what it saw. Install it today and it starts
empty, no matter how many months of conversations sit in `state.db`. This
script is the one command that closes that gap: it reads a profile's session
database and writes every stored message into the Qdrant collection in the
same shape the live path uses, so recall has something to recall from the
first turn.

It is also the migration tool between providers — anything that writes the
same collection shape can be seeded from the same source.

What it reads
-------------
`<profile>/state.db` — the `sessions` and `messages` tables. That is the
same file Hermes itself writes; it is opened read-only with the WAL included,
so running this against a LIVE profile cannot corrupt or block the gateway.

What it writes
--------------
One point per **2000-character chunk** of each message, which is the shape
already in the store (`kind` = user | assistant | tool, `chunk_index`,
`session_id`, `source`, `timestamp`). Point ids are deterministic
(`uuid5` of session + message + chunk), so re-running the same backfill
overwrites rather than duplicates — an interrupted run is restarted by
re-running the same command.

Scope
-----
Every source in the session DB by default (cli, tui, cron, subagent, email,
…). Narrow it with `--sources`. Sessions already represented in the
collection are skipped unless `--no-skip-existing` is passed, so re-running
against a live store only fills genuine gaps.

Usage
-----
    # what would happen, writing nothing (the default)
    python scripts/backfill_memory.py --profile qd

    # do it
    python scripts/backfill_memory.py --profile qd --yes

    # one source only, and include sessions already present
    python scripts/backfill_memory.py --profile qd --sources cron --yes \\
        --no-skip-existing

Cost: embedding is local. Measured ~2,000 messages/min on this host; a
125-session / 19k-message profile is a few minutes. Safe on a small box —
the work is CPU-bound and streamed in batches.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

#: Characters per stored point. Matches what the store already holds (the
#: live ingest measured max text length exactly 2000).
CHUNK_CHARS = 2000

#: Roles worth storing. `tool` is included because the existing store has
#: them and a recall for "which command produced this" is a real question.
DEFAULT_KINDS = ("user", "assistant", "tool")

#: Payload key holding the text. Same key the provider writes.
TEXT_KEY = "text"

#: Sparse vector name + model, in step with the provider constants. Kept as
#: literals so this script does not import the plugin (which needs the Hermes
#: host on sys.path).
SPARSE_VECTOR_NAME = "bm25"
SPARSE_MODEL_NAME = "Qdrant/bm25"

_REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Host-free imports: embedder.py is stdlib-only, so the dense embedding can
# be shared with the provider without dragging in the Hermes core.
# --------------------------------------------------------------------------
def _load_embedder_module():
    spec = importlib.util.spec_from_file_location(
        "_qdrant_embedder", _REPO / "embedder.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hermes_base() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _profile_dir(profile: str) -> Path:
    base = _hermes_base()
    cand = base / "profiles" / profile
    if cand.is_dir():
        return cand
    if base.name == profile:
        return base
    raise SystemExit(f"no profile directory for {profile!r} (looked at {cand})")


def _plugin_config() -> dict:
    """The provider's own config file, so the backfill targets the same store."""
    path = _hermes_base() / "qdrant.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except ValueError as e:
            raise SystemExit(f"{path} is not valid JSON: {e}") from e
    return {}


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Read-only, WAL included — safe against a live profile."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15.0)


def _chunks(text: str) -> Iterator[str]:
    for i in range(0, len(text), CHUNK_CHARS):
        yield text[i:i + CHUNK_CHARS]


def _existing_sessions(client, collection: str) -> set[str]:
    """Every session_id already in the collection (paginated scroll)."""
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=["session_id"], with_vectors=False,
        )
        for p in points:
            sid = (p.payload or {}).get("session_id")
            if sid:
                seen.add(sid)
        if offset is None:
            break
    return seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prefill the memory store from a profile's session history"
    )
    ap.add_argument("--profile", default=None,
                    help="profile whose state.db to read (default: $HERMES_HOME)")
    ap.add_argument("--profile-dir", default=None,
                    help="explicit profile home, overrides --profile")
    ap.add_argument("--url", default=None,
                    help="Qdrant URL (default: from qdrant.json)")
    ap.add_argument("--collection", default=None,
                    help="collection (default: from qdrant.json)")
    ap.add_argument("--sources", default="",
                    help="comma-separated session sources to include "
                         "(default: all)")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS),
                    help=f"message roles to store (default: {','.join(DEFAULT_KINDS)})")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    default=True, help="skip sessions already in the store (default)")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false",
                    help="include sessions already present (deterministic ids "
                         "make this an overwrite, not a duplicate)")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="stop after N sessions (0 = all); useful for a smoke test")
    ap.add_argument("--yes", action="store_true",
                    help="actually write; without it the run is a dry-run")
    args = ap.parse_args(argv)

    write = args.yes

    # -- resolve inputs ------------------------------------------------------
    if args.profile_dir:
        pdir = Path(args.profile_dir).expanduser()
    elif args.profile:
        pdir = _profile_dir(args.profile)
    else:
        pdir = _hermes_base()
    db_path = pdir / "state.db"
    if not db_path.is_file():
        print(f"no session db at {db_path}", file=sys.stderr)
        return 2

    cfg = _plugin_config()
    url = args.url or cfg.get("url") or "http://localhost:6333"
    collection = args.collection or cfg.get("collection") or "hermes_memories"
    backend = cfg.get("embedder") or "fastembed"
    # The model id comes from embedder.py's own constant, never a literal
    # here: a model string in a call path is an implicit download that
    # bypasses configuration (tests/test_tree_rules.py enforces this).
    em = _load_embedder_module()
    model = cfg.get("model") or em.DEFAULT_MODEL
    hybrid_requested = str(cfg.get("hybrid", "")).strip().lower() in {
        "on", "true", "yes", "1"}

    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        print(f"qdrant-client is not installed: {e}", file=sys.stderr)
        return 2

    client = QdrantClient(url=url, timeout=60)
    try:
        names = {c.name for c in client.get_collections().collections}
    except Exception as e:
        print(f"cannot reach Qdrant at {url}: {e}", file=sys.stderr)
        return 2
    if collection not in names:
        print(f"no collection {collection!r} at {url} — start Hermes once so "
              f"the provider creates it, or pass --collection", file=sys.stderr)
        return 2

    info = client.get_collection(collection)
    sparse_declared = set((info.config.params.sparse_vectors or {}).keys())
    do_sparse = SPARSE_VECTOR_NAME in sparse_declared

    print(f"profile:    {pdir}")
    print(f"store:      {url} / {collection} ({info.points_count:,} points)")
    print(f"embedder:   {backend} / {model}")
    print(f"sparse:     {'yes (hybrid active)' if do_sparse else 'no (dense-only)'}"
          + ("" if do_sparse or not hybrid_requested
             else "  [hybrid: on but the collection has no sparse vector yet]"))

    # -- existing sessions ---------------------------------------------------
    skip: set[str] = set()
    if args.skip_existing:
        t0 = time.time()
        skip = _existing_sessions(client, collection)
        print(f"already in store: {len(skip):,} session_ids "
              f"(scanned in {time.time() - t0:.1f}s)")

    # -- read the history ----------------------------------------------------
    source_filter = {s.strip() for s in args.sources.split(",") if s.strip()}
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}

    con = _open_db(db_path)
    try:
        rows = con.execute(
            "SELECT id, source FROM sessions ORDER BY COALESCE(started_at, 0)"
        ).fetchall()
    except sqlite3.DatabaseError as e:
        con.close()
        print(f"cannot read sessions: {e}", file=sys.stderr)
        return 2

    sessions = [(sid, src) for sid, src in rows
                if not source_filter or (src or "") in source_filter]
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]

    todo = [(sid, src) for sid, src in sessions if sid not in skip]
    print(f"sessions:   {len(todo):,} to ingest "
          f"({len(sessions) - len(todo):,} of {len(sessions):,} already present)")
    if not write:
        print("\n(dry-run — nothing written; re-run with --yes to apply)\n")

    # -- embedders (lazy: only when writing) ---------------------------------
    embedder = None
    sparse_embedder = None
    if write:
        try:
            embedder = em.Embedder(backend=backend, model=model)
        except Exception as e:
            print(f"dense embedder unavailable ({model}): {e}", file=sys.stderr)
            return 2
        if do_sparse:
            try:
                from fastembed import SparseTextEmbedding
                sparse_embedder = SparseTextEmbedding(SPARSE_MODEL_NAME)
            except Exception as e:
                print(f"sparse embedder unavailable: {e} — storing dense-only",
                      file=sys.stderr)

    # -- write ---------------------------------------------------------------
    started = time.time()
    n_sessions = n_points = n_empty = n_failed = 0
    NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

    for sid, src in sessions:
        if sid in skip:
            continue
        try:
            msgs = con.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id", (sid,)
            ).fetchall()
        except sqlite3.DatabaseError as e:
            n_failed += 1
            print(f"  skip session {sid}: {e}", file=sys.stderr)
            continue

        points = []
        for msg_id, role, content, ts in msgs:
            if role not in kinds:
                continue
            text = (content or "").strip()
            if not text:
                n_empty += 1
                continue
            for ci, chunk in enumerate(_chunks(text)):
                payload = {
                    TEXT_KEY: chunk,
                    "session_id": sid,
                    "timestamp": str(ts) if ts is not None else "",
                    "source": src or "backfill",
                    "kind": role,
                    "chunk_index": ci,
                }
                if not write:
                    n_points += 1
                    continue
                try:
                    vec = {"dense": embedder.encode(chunk)}
                    if sparse_embedder is not None:
                        sv = next(sparse_embedder.embed([chunk]))
                        if len(sv.indices):
                            from qdrant_client.models import SparseVector
                            vec[SPARSE_VECTOR_NAME] = SparseVector(
                                indices=sv.indices.tolist(),
                                values=sv.values.tolist())
                    points.append({
                        "id": str(uuid.uuid5(NS, f"{sid}:{msg_id}:{ci}")),
                        "vector": vec,
                        "payload": payload,
                    })
                except Exception as e:
                    n_failed += 1
                    if n_failed <= 5:
                        print(f"  embed failed for {sid}/{msg_id}: {e}",
                              file=sys.stderr)

        if write and points:
            try:
                client.upsert(collection_name=collection, points=points, wait=True)
                n_points += len(points)
            except Exception as e:
                n_failed += len(points)
                print(f"  upsert failed for session {sid}: {str(e)[:150]}",
                      file=sys.stderr)

        n_sessions += 1
        if n_sessions % 25 == 0:
            rate = n_points / max(time.time() - started, 1e-9)
            print(f"  ... {n_sessions:,} sessions, {n_points:,} points "
                  f"({rate:,.0f} pts/s)")

    elapsed = time.time() - started
    con.close()

    verb = "wrote" if write else "would write"
    print(f"\n{verb}: {n_points:,} points from {n_sessions:,} sessions")
    if n_empty:
        print(f"empty messages skipped: {n_empty:,}")
    if n_failed:
        print(f"failed:                 {n_failed:,}")
    print(f"elapsed: {elapsed:.1f}s")
    if write and n_points:
        after = client.get_collection(collection)
        print(f"collection now: {after.points_count:,} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
