"""Tests for the per-point history and usage ledger (``history.py``).

Pure SQLite: tmp_path databases, no Qdrant server, no LLM. Each test gets a
fresh ledger via the fixture; the fixture's teardown close() also exercises
the double-close path on every test.
"""
from __future__ import annotations

import sqlite3

import pytest
from plugins.memory.qdrant import history as history_mod
from plugins.memory.qdrant.history import MemoryLedger


@pytest.fixture
def ledger(tmp_path) -> MemoryLedger:
    led = MemoryLedger(tmp_path / "ledger.db")
    yield led
    led.close()  # safe even if the test already closed it


# ----------------------------------------------------------------------
# audit trail
# ----------------------------------------------------------------------

def test_record_then_history_is_newest_first(ledger):
    ledger.record_event("p1", "add", "first version", ts=1.0)
    ledger.record_event("p1", "update", "second version", ts=2.0)
    ledger.record_event("p1", "recall", None, ts=3.0)

    entries = ledger.history("p1")

    assert [e["event"] for e in entries] == ["recall", "update", "add"]
    assert entries[0]["ts"] == 3.0
    assert entries[0]["text"] is None
    assert entries[1]["text"] == "second version"
    assert entries[2]["ts"] == 1.0


def test_history_of_unknown_point_is_empty(ledger):
    assert ledger.history("never-seen") == []


def test_same_timestamp_events_keep_newest_insert_first(ledger):
    # rowid tiebreak: identical ts must not reorder or drop either row.
    ledger.record_event("p1", "recall", "a", ts=1.0)
    ledger.record_event("p1", "recall", "b", ts=1.0)

    assert [e["text"] for e in ledger.history("p1")] == ["b", "a"]


def test_events_pruned_to_newest_50_per_point(ledger):
    for i in range(55):
        ledger.record_event("p1", "recall", f"call {i}", ts=float(i))

    entries = ledger.history("p1")

    assert len(entries) == 50
    assert entries[0]["ts"] == 54.0
    assert entries[-1]["ts"] == 5.0
    assert "call 0" not in [e["text"] for e in entries]
    # the bound is per point: a second point keeps its own full history
    ledger.record_event("p2", "add", "only one", ts=1.0)
    assert len(ledger.history("p2")) == 1


def test_unknown_event_name_raises_value_error(ledger):
    with pytest.raises(ValueError, match="unknown event"):
        ledger.record_event("p1", "explode")

    # the failed write left no trace
    assert ledger.history("p1") == []
    assert ledger.stats("p1")["event_count"] == 0


# ----------------------------------------------------------------------
# usage counters
# ----------------------------------------------------------------------

def test_bump_retrieval_increments_and_sets_last_accessed(ledger):
    ledger.bump_retrieval(["p1"], ts=10.0)
    ledger.bump_retrieval(["p1", "p2"], ts=20.0)

    s1 = ledger.stats("p1")
    assert s1["retrieval_count"] == 2
    assert s1["first_seen"] == 10.0
    assert s1["last_accessed"] == 20.0

    s2 = ledger.stats("p2")
    assert s2["retrieval_count"] == 1
    assert s2["first_seen"] == 20.0
    assert s2["last_accessed"] == 20.0


def test_retrieval_popularity_does_not_move_trust(ledger):
    # trust is feedback-only: 100 retrievals, no votes -> still the prior.
    for _ in range(100):
        ledger.bump_retrieval(["p1"])
    assert ledger.trust_score("p1") == 0.5


def test_helpful_and_unhelpful_move_trust_from_neutral_prior(ledger):
    assert ledger.trust_score("p1") == 0.5  # Laplace prior, no feedback

    ledger.bump_helpful(["p1"], 1)
    assert ledger.trust_score("p1") == pytest.approx(0.6667)  # 2/3

    ledger.bump_helpful(["p1"], 1)
    assert ledger.trust_score("p1") == pytest.approx(0.75)  # 3/4

    ledger.bump_helpful(["p1"], -1)  # unhelpful
    assert ledger.trust_score("p1") == pytest.approx(0.6)  # 3/5

    # unhelpful alone sinks the score below the prior
    ledger.bump_helpful(["p2"], -1)
    assert ledger.trust_score("p2") == pytest.approx(0.3333)  # 1/3


def test_trust_score_is_rounded_to_4_decimals(ledger):
    # 1 helpful / 1 unhelpful -> 2/4 = 0.5 exactly; use 1/2 -> 2/5 = 0.4.
    # 3 helpful / 1 unhelpful -> 4/6 = 0.66666... -> 0.6667.
    ledger.bump_helpful(["p1"], 3)
    ledger.bump_helpful(["p1"], -1)
    assert ledger.trust_score("p1") == 0.6667


def test_stats_reports_event_count(ledger):
    ledger.record_event("p1", "add", "v1", ts=1.0)
    ledger.record_event("p1", "recall", None, ts=2.0)
    ledger.record_event("p1", "recall", None, ts=3.0)

    assert ledger.stats("p1")["event_count"] == 3


# ----------------------------------------------------------------------
# unknown points
# ----------------------------------------------------------------------

def test_unknown_point_stats_are_zeros(ledger):
    s = ledger.stats("nope")

    assert s["point_id"] == "nope"
    assert s["retrieval_count"] == 0
    assert s["helpful_count"] == 0
    assert s["unhelpful_count"] == 0
    assert s["event_count"] == 0
    assert s["first_seen"] is None
    assert s["last_accessed"] is None
    # no feedback -> the neutral prior, not zero
    assert s["trust_score"] == 0.5


# ----------------------------------------------------------------------
# durability / lifecycle
# ----------------------------------------------------------------------

def test_reopening_same_path_preserves_data(tmp_path):
    db = tmp_path / "ledger.db"
    led = MemoryLedger(db)
    led.record_event("p1", "add", "hello", ts=1.0)
    led.bump_retrieval(["p1"], ts=2.0)
    led.bump_helpful(["p1"], 1)
    led.close()

    re = MemoryLedger(db)  # idempotent schema creation on an existing db
    assert re.history("p1") == [{"ts": 1.0, "event": "add", "text": "hello"}]
    s = re.stats("p1")
    assert s["retrieval_count"] == 1
    assert s["helpful_count"] == 1
    assert s["trust_score"] == pytest.approx(0.6667)
    re.close()


def test_double_close_is_safe(tmp_path):
    led = MemoryLedger(tmp_path / "ledger.db")
    led.close()
    led.close()  # must not raise


def test_use_after_close_raises_clear_error(tmp_path):
    led = MemoryLedger(tmp_path / "ledger.db")
    led.close()

    with pytest.raises(history_mod.MemoryLedgerClosedError):
        led.record_event("p1", "add")
    with pytest.raises(history_mod.MemoryLedgerClosedError):
        led.history("p1")
    with pytest.raises(history_mod.MemoryLedgerClosedError):
        led.stats("p1")
    with pytest.raises(history_mod.MemoryLedgerClosedError):
        led.trust_score("p1")
    with pytest.raises(history_mod.MemoryLedgerClosedError):
        led.bump_retrieval(["p1"])
    with pytest.raises(history_mod.MemoryLedgerClosedError):
        led.bump_helpful(["p1"])


def test_db_uses_wal_and_tolerates_a_concurrent_reader(tmp_path):
    db = tmp_path / "ledger.db"
    led = MemoryLedger(db)

    mode = led._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"

    # a separate connection reads while the ledger is open (WAL: no block)
    other = sqlite3.connect(db)
    assert other.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    other.close()

    led.close()
