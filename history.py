"""Per-point history and usage ledger (rival-hardening: Mem0 + memory-os parity).

A small local SQLite ledger recording what happened to each memory point
(add/update/delete/recall) and the usage counters a trust score is computed
from. Lives OUTSIDE the plugin member dir: the caller passes ``db_path``
explicitly and this module never invents a location. The plugin member dir
must stay free of changing runtime state — Hermes loads it by directory path
as managed install content, and an SQLite database plus its WAL sidecar files
would mutate that directory on every write. The caller owns the data location
(e.g. the profile state dir) and its cleanup.

Design notes (why, not just what):

* Why WAL + busy_timeout: recall bumps and audit writes arrive from
  different threads, and possibly different processes, of the same agent.
  WAL lets a reader proceed while the writer holds the write lock, and the
  busy timeout makes a colliding connection wait up to N ms instead of
  failing instantly with "database is locked" the moment the writer
  commits.
* Why the 50-event prune: the events table is an audit trail for debugging
  and display ("what happened to this point?"), not the source of truth —
  the vector store is. Unbounded, a long-lived point accumulates one row
  per recall forever; the DB grows without limit and every history() read
  scans it. Newest-50-per-point keeps the common questions answerable while
  capping storage and write amplification per point.
* Why trust_score excludes retrieval_count: helpful/unhelpful votes are a
  QUALITY signal; retrieval_count is a POPULARITY signal. Folding
  popularity into trust would let a point that is recalled constantly but
  never voted on (or actively disliked) look trustworthy. The Laplace
  smoothing (helpful + 1) / (helpful + unhelpful + 2) gives a point with no
  feedback a neutral 0.50 — neither trusted nor distrusted — that moves only
  when real feedback arrives.
* Why use-after-close raises MemoryLedgerClosedError instead of the raw
  sqlite3.ProgrammingError: the raw error names an internal sqlite3 module
  detail that means nothing at the plugin boundary; callers catch ours.

Intended interface:

    class MemoryLedger:
        record_event(point_id, event, text=None, ts=None) -> None
        history(point_id) -> list[dict]
        bump_retrieval(point_ids, ts=None) -> None
        bump_helpful(point_ids, delta=1) -> None
        stats(point_id) -> dict
        trust_score(point_id) -> float
        close() -> None
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

#: Event names record_event() accepts. A closed set so an unknown name fails
#: deterministically (ValueError) instead of silently corrupting the log.
KNOWN_EVENTS = frozenset({"add", "update", "delete", "recall", "supersede"})

#: Per-point cap on stored events; see module docstring (bounded growth).
MAX_EVENTS_PER_POINT = 50

#: Milliseconds a colliding connection waits for the lock before raising.
BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    point_id TEXT NOT NULL,
    ts      REAL NOT NULL,
    event   TEXT NOT NULL,
    text    TEXT
);
CREATE INDEX IF NOT EXISTS events_point_ts
    ON events (point_id, ts DESC);

CREATE TABLE IF NOT EXISTS usage (
    point_id        TEXT PRIMARY KEY,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    helpful_count   INTEGER NOT NULL DEFAULT 0,
    unhelpful_count INTEGER NOT NULL DEFAULT 0,
    first_seen      REAL,
    last_accessed   REAL
);
"""


class MemoryLedgerClosedError(RuntimeError):
    """A method was called on a MemoryLedger whose connection is closed."""


class MemoryLedger:
    """SQLite-backed per-point history + usage ledger; see module docstring."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self._db_path, timeout=BUSY_TIMEOUT_MS / 1000
        )
        self._conn.row_factory = sqlite3.Row
        # PRAGMAs run outside any transaction: journal_mode=WAL is a no-op
        # inside one, and busy_timeout is per-connection state, so both are
        # set here rather than inherited from a previous connection.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        # IF NOT EXISTS makes opening an existing database idempotent.
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # audit trail
    # ------------------------------------------------------------------

    def record_event(
        self, point_id: str, event: str, text: str | None = None,
        ts: float | None = None,
    ) -> None:
        """Append one audit event for ``point_id``, newest MAX_EVENTS_PER_POINT kept.

        ``event`` must be one of add/update/delete/recall/supersede; an
        unknown name raises ValueError deterministically rather than
        silently writing a row the rest of the code cannot interpret.
        ``ts`` defaults to time.time(). The insert and the prune commit (or
        roll back) together.
        """
        conn = self._ensure_open()
        if event not in KNOWN_EVENTS:
            raise ValueError(
                f"unknown event {event!r}; expected one of {sorted(KNOWN_EVENTS)}"
            )
        now = time.time() if ts is None else float(ts)
        with conn:  # atomic: insert + prune commit as one unit
            conn.execute(
                "INSERT INTO events (point_id, ts, event, text) VALUES (?, ?, ?, ?)",
                (point_id, now, event, text),
            )
            # rowid breaks ts ties so same-instant inserts keep the newest
            # rows; ordering the kept set by (ts DESC, rowid DESC) matches
            # history()'s read order, so pruning and reading never disagree.
            conn.execute(
                """
                DELETE FROM events
                WHERE point_id = ?
                  AND rowid NOT IN (
                      SELECT rowid FROM events
                      WHERE point_id = ?
                      ORDER BY ts DESC, rowid DESC
                      LIMIT ?
                  )
                """,
                (point_id, point_id, MAX_EVENTS_PER_POINT),
            )

    def history(self, point_id: str) -> list[dict]:
        """Return this point's events newest first.

        Each entry is {"ts": float, "event": str, "text": str | None}.
        A point with no events yields an empty list, never an error.
        """
        conn = self._ensure_open()
        rows = conn.execute(
            "SELECT ts, event, text FROM events "
            "WHERE point_id = ? ORDER BY ts DESC, rowid DESC",
            (point_id,),
        ).fetchall()
        return [{"ts": r["ts"], "event": r["event"], "text": r["text"]} for r in rows]

    # ------------------------------------------------------------------
    # usage counters
    # ------------------------------------------------------------------

    def bump_retrieval(self, point_ids: list[str], ts: float | None = None) -> None:
        """Count one retrieval per id and stamp last_accessed.

        Retrievals feed the popularity signal only — see trust_score's
        docstring for why that must stay out of the quality formula.
        """
        conn = self._ensure_open()
        now = time.time() if ts is None else float(ts)
        with conn:
            for point_id in point_ids:
                conn.execute(
                    """
                    INSERT INTO usage
                        (point_id, retrieval_count, first_seen, last_accessed)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(point_id) DO UPDATE SET
                        retrieval_count = retrieval_count + 1,
                        last_accessed  = excluded.last_accessed
                    """,
                    (point_id, now, now),
                )

    def bump_helpful(self, point_ids: list[str], delta: int = 1) -> None:
        """Apply feedback: positive delta adds helpful votes, negative unhelpful.

        delta=-1 (unhelpful) increments unhelpful_count by 1 — feedback
        never drives a count below zero. A zero delta writes nothing.
        """
        conn = self._ensure_open()
        now = time.time()
        with conn:
            for point_id in point_ids:
                if delta == 0:
                    continue
                if delta > 0:
                    conn.execute(
                        """
                        INSERT INTO usage (point_id, helpful_count, first_seen)
                        VALUES (?, ?, ?)
                        ON CONFLICT(point_id) DO UPDATE SET
                            helpful_count = helpful_count + excluded.helpful_count
                        """,
                        (point_id, delta, now),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO usage (point_id, unhelpful_count, first_seen)
                        VALUES (?, ?, ?)
                        ON CONFLICT(point_id) DO UPDATE SET
                            unhelpful_count = unhelpful_count + excluded.unhelpful_count
                        """,
                        (point_id, -delta, now),
                    )

    # ------------------------------------------------------------------
    # derived views
    # ------------------------------------------------------------------

    def stats(self, point_id: str) -> dict:
        """Counters + trust + event count for one point.

        An unknown point_id returns zeros (and None for the timestamps)
        rather than raising: stats is a read-only report, and "never seen"
        is a valid report. first_seen/last_accessed are the first/last usage
        stamps; the authoritative add timestamp lives in the events table.
        """
        conn = self._ensure_open()
        row = conn.execute(
            """
            SELECT retrieval_count, helpful_count, unhelpful_count,
                   first_seen, last_accessed
            FROM usage
            WHERE point_id = ?
            """,
            (point_id,),
        ).fetchone()
        if row is None:
            retrieval = helpful = unhelpful = 0
            first_seen = last_accessed = None
        else:
            retrieval = row["retrieval_count"]
            helpful = row["helpful_count"]
            unhelpful = row["unhelpful_count"]
            first_seen = row["first_seen"]
            last_accessed = row["last_accessed"]
        event_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE point_id = ?", (point_id,)
        ).fetchone()[0]
        return {
            "point_id": point_id,
            "retrieval_count": retrieval,
            "helpful_count": helpful,
            "unhelpful_count": unhelpful,
            "trust_score": self._trust(helpful, unhelpful),
            "first_seen": first_seen,
            "last_accessed": last_accessed,
            "event_count": event_count,
        }

    def trust_score(self, point_id: str) -> float:
        """Laplace-smoothed helpful ratio in [0, 1], rounded to 4 decimals.

        (helpful + 1) / (helpful + unhelpful + 2): a point with no feedback
        sits at 0.50 — the neutral prior — and moves only with real feedback.
        retrieval_count deliberately does NOT enter this formula: it is a
        popularity signal, and folding popularity into trust would let a point
        that is recalled constantly but never voted on look trustworthy.
        """
        conn = self._ensure_open()
        row = conn.execute(
            "SELECT helpful_count, unhelpful_count FROM usage WHERE point_id = ?",
            (point_id,),
        ).fetchone()
        helpful = row["helpful_count"] if row is not None else 0
        unhelpful = row["unhelpful_count"] if row is not None else 0
        return self._trust(helpful, unhelpful)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Commit anything pending and release the connection. Safe to call twice."""
        if self._conn is None:
            return
        conn, self._conn = self._conn, None
        try:
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _trust(helpful: int, unhelpful: int) -> float:
        """The trust formula, shared by trust_score() and stats()."""
        return round((helpful + 1) / (helpful + unhelpful + 2), 4)

    def _ensure_open(self) -> sqlite3.Connection:
        """Return the live connection or raise a clear closed-ledger error."""
        if self._conn is None:
            raise MemoryLedgerClosedError(
                f"MemoryLedger for {self._db_path!r} is already closed; "
                "construct a new MemoryLedger to keep writing"
            )
        return self._conn
