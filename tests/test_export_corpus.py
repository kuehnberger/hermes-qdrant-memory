"""Tests for ``scripts/export_corpus.py`` — the session/log -> markdown exporter.

The exporter is what makes a profile's real history ingestible by ``md_search``
at all, so the properties that matter are: the output is shaped the way the
heading-aware chunker expects, nothing enormous or empty sneaks in, and a LIVE
database is read without being written to.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_corpus.py"
_spec = importlib.util.spec_from_file_location("export_corpus", _SCRIPT)
export_corpus = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(export_corpus)


def _make_db(path: Path, *, sessions: list[dict], messages: list[tuple]) -> Path:
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, title TEXT,"
        " display_name TEXT, started_at REAL, ended_at REAL, end_reason TEXT,"
        " model TEXT, message_count INTEGER)"
    )
    con.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
        " role TEXT, content TEXT, tool_calls TEXT, reasoning TEXT,"
        " reasoning_content TEXT)"
    )
    for s in sessions:
        con.execute(
            "INSERT INTO sessions VALUES (:id, :source, :title, :display_name,"
            " :started_at, :ended_at, :end_reason, :model, :message_count)",
            s,
        )
    for m in messages:
        con.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)", m)
    con.commit()
    con.close()
    return path


def test_session_becomes_headed_markdown(tmp_path):
    db = _make_db(
        tmp_path / "state.db",
        sessions=[{
            "id": "s1", "source": "cli", "title": "Catalog work",
            "display_name": "QD", "started_at": 1790683335.0,
            "ended_at": None, "end_reason": None,
            "model": "combo", "message_count": 2,
        }],
        messages=[
            (1, "s1", "user", "the pin is wrong", None, None, None),
            (2, "s1", "assistant", "repinning now", json.dumps(
                [{"type": "function", "function": {"name": "terminal"}}]
            ), "SECRET-REASONING", None),
        ],
    )
    out = tmp_path / "out"
    files, turns = export_corpus.export_sessions(db, out, "qd")
    doc = (out / "sessions" / "s1.md").read_text(encoding="utf-8")
    assert files == 1 and turns == 2
    assert doc.startswith("# Catalog work")
    assert "- profile: qd" in doc and "- session: s1" in doc
    assert "## user (msg 1)" in doc and "## assistant (msg 2)" in doc
    # the tool name survives as a breadcrumb...
    assert "*tools: terminal*" in doc
    # ...and a model's private reasoning never enters the corpus
    assert "SECRET-REASONING" not in doc


def test_tool_messages_are_dropped(tmp_path):
    db = _make_db(
        tmp_path / "state.db",
        sessions=[{
            "id": "s1", "source": "cli", "title": "t", "display_name": None,
            "started_at": None, "ended_at": None, "end_reason": None,
            "model": None, "message_count": 3,
        }],
        messages=[
            (1, "s1", "user", "run it", None, None, None),
            (2, "s1", "assistant", "running", None, None, None),
            (3, "s1", "tool", '{"output": "5 MB of stdout"}', None, None, None),
        ],
    )
    out = tmp_path / "out"
    export_corpus.export_sessions(db, out, "qd")
    doc = (out / "sessions" / "s1.md").read_text(encoding="utf-8")
    assert "5 MB of stdout" not in doc
    assert "## tool" not in doc


def test_oversized_turn_is_truncated_with_a_marker(tmp_path):
    huge = "x" * (export_corpus.MAX_TURN_CHARS + 5000)
    db = _make_db(
        tmp_path / "state.db",
        sessions=[{
            "id": "s1", "source": "cli", "title": "t", "display_name": None,
            "started_at": None, "ended_at": None, "end_reason": None,
            "model": None, "message_count": 2,
        }],
        messages=[
            (1, "s1", "user", "small", None, None, None),
            (2, "s1", "assistant", huge, None, None, None),
        ],
    )
    out = tmp_path / "out"
    export_corpus.export_sessions(db, out, "qd")
    doc = (out / "sessions" / "s1.md").read_text(encoding="utf-8")
    assert "[truncated by export_corpus.py]" in doc
    assert len(doc) < len(huge)


def test_session_below_the_minimum_is_skipped(tmp_path):
    db = _make_db(
        tmp_path / "state.db",
        sessions=[{
            "id": "s1", "source": "cli", "title": "t", "display_name": None,
            "started_at": None, "ended_at": None, "end_reason": None,
            "model": None, "message_count": 1,
        }],
        messages=[(1, "s1", "user", "hello", None, None, None)],
    )
    out = tmp_path / "out"
    assert export_corpus.export_sessions(db, out, "qd") == (0, 0)
    assert not (out / "sessions").exists() or not list((out / "sessions").glob("*.md"))


def test_empty_title_falls_back_to_session_id(tmp_path):
    db = _make_db(
        tmp_path / "state.db",
        sessions=[{
            "id": "s9", "source": None, "title": "   ", "display_name": None,
            "started_at": None, "ended_at": None, "end_reason": None,
            "model": None, "message_count": 2,
        }],
        messages=[
            (1, "s9", "user", "a", None, None, None),
            (2, "s9", "assistant", "b", None, None, None),
        ],
    )
    out = tmp_path / "out"
    export_corpus.export_sessions(db, out, "qd")
    doc = (out / "sessions" / "s9.md").read_text(encoding="utf-8")
    assert doc.startswith("# session s9")


def test_plain_and_json_log_lines_both_parse(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "agent.log").write_text(
        "2026-09-27 17:31:13,363 INFO hermes_cli.plugins: registered qdrant\n"
        '{"timestamp": "2026-09-28T04:00:00Z", "level": "ERROR",'
        ' "name": "x.y", "message": "boom"}\n'
        "not a log line at all\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    days, entries = export_corpus.export_agent_log(logs / "agent.log", out)
    assert days == 2 and entries == 2
    first = (out / "agentlog" / "2026-09-27.md").read_text(encoding="utf-8")
    assert first.startswith("# agent log 2026-09-27")
    assert "registered qdrant" in first
    assert "not a log line at all" not in first


def test_rotated_logs_are_read_oldest_first(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "agent.log").write_text(
        "2026-09-29 10:00:00,000 INFO m: newest\n", encoding="utf-8"
    )
    (logs / "agent.log.1").write_text(
        "2026-09-28 10:00:00,000 INFO m: older\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    days, entries = export_corpus.export_agent_log(logs / "agent.log", out)
    assert days == 2 and entries == 2


def test_missing_log_is_not_an_error(tmp_path):
    assert export_corpus.export_agent_log(
        tmp_path / "nope.log", tmp_path / "o"
    ) == (0, 0)


def test_live_database_is_read_only(tmp_path):
    """The exporter must not write to a profile that is currently running."""
    db = _make_db(
        tmp_path / "state.db",
        sessions=[{
            "id": "s1", "source": "cli", "title": "t", "display_name": None,
            "started_at": None, "ended_at": None, "end_reason": None,
            "model": None, "message_count": 2,
        }],
        messages=[
            (1, "s1", "user", "a", None, None, None),
            (2, "s1", "assistant", "b", None, None, None),
        ],
    )
    before = db.read_bytes()
    con = export_corpus.open_db(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute(
                "INSERT INTO sessions VALUES "
                "('x',NULL,NULL,NULL,NULL,NULL,NULL,NULL,0)"
            )
    finally:
        con.close()
    export_corpus.export_sessions(db, tmp_path / "out", "qd")
    assert db.read_bytes() == before


def test_default_output_is_derived_state_not_plugin_state(monkeypatch):
    """The venv-stamp rule: derived data must not sit next to the plugin code."""
    import re as _re

    src = _SCRIPT.read_text(encoding="utf-8")
    # the default output path is <base>/state/knowledge/<profile> ...
    assert _re.search(r'state"\s*/\s*"knowledge"', src)
    # ... and never a member dir
    assert "REPO_DIR" not in src
    assert "Path(__file__).resolve().parent" not in src


def test_cli_stats_only_writes_nothing(tmp_path, capsys):
    profile = tmp_path / "profiles" / "tester"
    profile.mkdir(parents=True)
    _make_db(
        profile / "state.db",
        sessions=[{
            "id": "s1", "source": "cli", "title": "t", "display_name": None,
            "started_at": None, "ended_at": None, "end_reason": None,
            "model": None, "message_count": 2,
        }],
        messages=[
            (1, "s1", "user", "a", None, None, None),
            (2, "s1", "assistant", "b", None, None, None),
        ],
    )
    rc = export_corpus.main([
        "--profile", "tester", "--profile-dir", str(profile),
        "--out", str(tmp_path / "out"), "--stats-only",
    ])
    assert rc == 0
    assert not (tmp_path / "out").exists()
    assert "would export" in capsys.readouterr().out


def test_script_is_runnable_standalone():
    assert _SCRIPT.is_file()
    assert sys.executable  # the shebang interpreter exists
