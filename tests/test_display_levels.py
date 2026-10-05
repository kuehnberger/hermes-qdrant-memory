"""Tests for the 0.1.8 display levels (SPEC-qdrant-display).

Four claims, one per gate the spec set:

* the DEFAULT (``display.level`` absent) renders the chat indicator byte for
  byte as 0.1.7 did — no metrics, no collection, no latency, whatever the
  timing state happens to be;
* ``summary`` puts exactly two new facts in ``provider_label`` — ms and the
  collection — and drops the ``· …ms`` segment when no timing exists instead
  of printing a zero;
* a broken config degrades to ``off`` with ONE warning line, never an
  exception and never a warning per recall;
* ``verbose`` writes numeric operation records (recall / store / md_search)
  to the log and nothing else — no message text, no snippets (spec rule c2).

Core's rendering is reproduced below rather than imported: the indicator is
built by ``MemoryManager.describe_recall`` in hermes-agent, which is not on
this repo's import path in every environment. The format is quoted verbatim
from that function, so a byte-identical claim here is a claim about the
string the user actually sees.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

LOGGER = "hermes.plugins.memory.qdrant"


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Give the test its own ``<HERMES_HOME>/qdrant.json``.

    Also empties the process-wide "already warned" registry: one warning per
    defect per process is the production rule, but each test asserts its own
    defect, so the registry must start clean.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import plugins.memory.qdrant as mod
    monkeypatch.setattr(mod, "_DISPLAY_WARNINGS", set())
    yield home


@pytest.fixture
def provider(monkeypatch):
    """A fresh provider built from defaults, never from operator state."""
    import plugins.memory.qdrant as mod
    monkeypatch.setattr(mod, "_load_plugin_config", lambda: {})
    return mod.QdrantMemoryProvider()


def set_level(home, level) -> None:
    (home / "qdrant.json").write_text(
        json.dumps({"display": {"level": level}}), encoding="utf-8"
    )


def indicator(status) -> str:
    """``MemoryManager.describe_recall``'s exact rendering of one status."""
    count = status.count
    detail = (
        "recalled 1 memory" if count == 1
        else f"recalled {count} memories" if count > 1
        else "recalled relevant memory"
    )
    return f"{status.glyph} {status.provider_label} — {detail}"


def info_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


def hit(text: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(payload={"text": text}, score=score)


def wire_prefetch(provider, points) -> None:
    """An initialized provider whose client and embedder are stubs."""
    provider._initialized = True
    provider._client = MagicMock()
    provider._client.query_points.return_value = SimpleNamespace(points=points)
    provider._embed = lambda text: [0.1] * 4


def wire_store(provider, points_count: int | None = 128456) -> None:
    """An initialized provider that stores into a stub client."""
    provider._initialized = True
    provider._client = MagicMock()
    if points_count is None:
        provider._client.get_collection.side_effect = RuntimeError("down")
    else:
        provider._client.get_collection.return_value = SimpleNamespace(
            points_count=points_count
        )
    provider._status_callback = MagicMock()
    provider._embed = lambda text: [0.1] * 4


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestDisplayLevelConfig:

    def test_absent_file_is_off(self, provider):
        import plugins.memory.qdrant as mod
        assert mod.display_level() == "off"

    def test_absent_key_is_off(self, provider, _isolate_hermes_home):
        (_isolate_hermes_home / "qdrant.json").write_text(
            json.dumps({"progress": "minimal"}), encoding="utf-8"
        )
        import plugins.memory.qdrant as mod
        assert mod.display_level() == "off"

    @pytest.mark.parametrize("level", ["off", "summary", "verbose"])
    def test_each_level_parses(self, provider, _isolate_hermes_home, level):
        set_level(_isolate_hermes_home, level)
        import plugins.memory.qdrant as mod
        assert mod.display_level() == level

    def test_value_is_case_and_whitespace_insensitive(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, " Summary ")
        import plugins.memory.qdrant as mod
        assert mod.display_level() == "summary"

    def test_malformed_json_is_off_and_warns_once(
        self, provider, _isolate_hermes_home, caplog
    ):
        (_isolate_hermes_home / "qdrant.json").write_text(
            "{not json", encoding="utf-8"
        )
        import plugins.memory.qdrant as mod
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert mod.display_level() == "off"
            assert mod.display_level() == "off"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "parse failure must warn once, not per call"
        assert "display.level" in warnings[0].getMessage()

    def test_non_object_document_is_off_and_warns_once(
        self, provider, _isolate_hermes_home, caplog
    ):
        (_isolate_hermes_home / "qdrant.json").write_text(
            '["summary"]', encoding="utf-8"
        )
        import plugins.memory.qdrant as mod
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert mod.display_level() == "off"
            assert mod.display_level() == "off"
        assert len(caplog.records) == 1
        assert "display.level" in caplog.records[0].getMessage()

    def test_display_block_of_wrong_type_is_off_and_warns_once(
        self, provider, _isolate_hermes_home, caplog
    ):
        (_isolate_hermes_home / "qdrant.json").write_text(
            json.dumps({"display": "verbose"}), encoding="utf-8"
        )
        import plugins.memory.qdrant as mod
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert mod.display_level() == "off"
            assert mod.display_level() == "off"
        assert len(caplog.records) == 1

    @pytest.mark.parametrize("level", ["loud", 3, True])
    def test_value_outside_the_vocabulary_is_off_and_warns_once(
        self, provider, _isolate_hermes_home, caplog, level
    ):
        # A typo and a non-string both fail the same membership test, and
        # both are parse failures: off plus one warning, never an exception.
        (_isolate_hermes_home / "qdrant.json").write_text(
            json.dumps({"display": {"level": level}}), encoding="utf-8"
        )
        import plugins.memory.qdrant as mod
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert mod.display_level() == "off"
            assert mod.display_level() == "off"
        assert len(caplog.records) == 1
        assert "display.level" in caplog.records[0].getMessage()

    def test_display_is_not_reported_as_an_unknown_config_key(
        self, provider, _isolate_hermes_home, caplog, monkeypatch
    ):
        """"display" is a documented key of qdrant.json, not a typo.

        The flat-config sweep must not warn about the very key 0.1.8 tells
        users to write (the mnemosyne #482 warning exists to catch mistakes).
        """
        set_level(_isolate_hermes_home, "summary")
        import hermes_cli.config as _cfg
        monkeypatch.setattr(_cfg, "load_config_readonly", lambda: {})
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        import plugins.memory.qdrant as mod
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            merged = mod._load_plugin_config()
        assert "display" not in merged, "the flat config carries flat knobs"
        assert not any(
            "unknown config key" in r.getMessage() for r in caplog.records
        )

    def test_status_config_reports_the_level(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "verbose")
        assert provider.get_status_config({})["display"] == "verbose"

    def test_status_config_defaults_to_off(self, provider):
        assert provider.get_status_config({})["display"] == "off"


# ---------------------------------------------------------------------------
# The indicator
# ---------------------------------------------------------------------------


class TestIndicator:

    def test_default_level_indicator_is_byte_identical(self, provider):
        """The regression the spec names first: off == 0.1.7, byte for byte."""
        provider._last_recall_count = 10
        provider._last_recall_ms = 41  # timing exists but must not surface
        status = provider.recall_status()
        assert status.provider_label == "qdrant"
        assert status.glyph == "📖"
        assert indicator(status) == "📖 qdrant — recalled 10 memories"

    def test_explicit_off_level_is_byte_identical(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "off")
        provider._last_recall_count = 10
        provider._last_recall_ms = 41
        status = provider.recall_status()
        assert indicator(status) == "📖 qdrant — recalled 10 memories"

    @pytest.mark.parametrize(
        "count,expected",
        [
            (1, "📖 qdrant — recalled 1 memory"),
            (0, "📖 qdrant — recalled relevant memory"),
            (3, "📖 qdrant — recalled 3 memories"),
        ],
    )
    def test_off_level_holds_for_every_count(
        self, provider, _isolate_hermes_home, count, expected
    ):
        set_level(_isolate_hermes_home, "off")
        provider._last_recall_count = count
        provider._last_recall_ms = 41
        assert indicator(provider.recall_status()) == expected

    def test_summary_label_is_exactly_the_spec_format(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "summary")
        provider._last_recall_count = 10
        provider._last_recall_ms = 41
        status = provider.recall_status()
        assert status.provider_label == "qdrant · 41ms · hermes_memories"
        assert indicator(status) == (
            "📖 qdrant · 41ms · hermes_memories — recalled 10 memories"
        )

    def test_summary_without_timing_omits_the_ms_segment(
        self, provider, _isolate_hermes_home
    ):
        """First turn: no timing yet → collection only, never a fabricated 0."""
        set_level(_isolate_hermes_home, "summary")
        provider._last_recall_count = 3
        provider._last_recall_ms = None
        assert provider.recall_status().provider_label == (
            "qdrant · hermes_memories"
        )

    def test_summary_never_prints_zero_milliseconds(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "summary")
        provider._last_recall_count = 3
        provider._last_recall_ms = 0
        status = provider.recall_status()
        assert status.provider_label == "qdrant · hermes_memories"
        assert "0ms" not in status.provider_label

    def test_verbose_carries_the_same_label_as_summary(
        self, provider, _isolate_hermes_home
    ):
        """Verbose adds log lines; the chat indicator does not grow again."""
        set_level(_isolate_hermes_home, "verbose")
        provider._last_recall_count = 10
        provider._last_recall_ms = 41
        assert provider.recall_status().provider_label == (
            "qdrant · 41ms · hermes_memories"
        )

    def test_label_follows_the_configured_collection(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "summary")
        provider._collection = "other_collection"
        provider._last_recall_ms = 41
        assert provider.recall_status().provider_label == (
            "qdrant · 41ms · other_collection"
        )


# ---------------------------------------------------------------------------
# Verbose: recall
# ---------------------------------------------------------------------------


class TestVerboseRecall:

    def test_recall_line_is_numbers_and_ids_only(
        self, provider, _isolate_hermes_home, caplog
    ):
        set_level(_isolate_hermes_home, "verbose")
        wire_prefetch(provider, [
            hit("the deployment target is a debian vps in frankfurt", 0.94),
            hit("meeting with the design team on fridays", 0.70),
        ])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.prefetch("where do we deploy", session_id="sess-1")
        messages = info_messages(caplog)
        recall = [m for m in messages if m.startswith("recall ")]
        assert len(recall) == 1, recall
        assert re.fullmatch(
            r"recall 2 pts in \d+ ms \(scores 0\.70–0\.94\) "
            r"session=sess-1 collection=hermes_memories",
            recall[0],
        ), recall[0]
        # spec rule c2: contents never reach the log
        assert "debian" not in caplog.text
        assert "fridays" not in caplog.text
        assert provider._last_recall_ms is not None

    def test_no_recall_line_at_off(self, provider, caplog):
        wire_prefetch(provider, [hit("secret memory text", 0.9)])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.prefetch("anything", session_id="s")
        assert not any(m.startswith("recall ") for m in info_messages(caplog))

    def test_no_recall_line_at_summary(
        self, provider, _isolate_hermes_home, caplog
    ):
        """Summary is a chat-indicator level; the log stays as it was."""
        set_level(_isolate_hermes_home, "summary")
        wire_prefetch(provider, [hit("secret memory text", 0.9)])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.prefetch("anything", session_id="s")
        assert not any(m.startswith("recall ") for m in info_messages(caplog))

    def test_failed_recall_clears_the_latency(
        self, provider, _isolate_hermes_home, caplog
    ):
        """A failed recall must not leave a stale ms on the indicator."""
        set_level(_isolate_hermes_home, "summary")
        wire_prefetch(provider, [])
        provider._client.query_points.side_effect = RuntimeError("boom")
        provider._last_recall_ms = 41
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert provider.prefetch("anything", session_id="s") == ""
        assert provider._last_recall_ms is None
        assert provider.recall_status().provider_label == (
            "qdrant · hermes_memories"
        )


# ---------------------------------------------------------------------------
# Verbose: store
# ---------------------------------------------------------------------------


class TestVerboseStore:

    def test_store_line_is_numbers_only(
        self, provider, _isolate_hermes_home, caplog
    ):
        set_level(_isolate_hermes_home, "verbose")
        wire_store(provider)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.sync_turn("user said what", "assistant replied",
                               session_id="sess-1")
        store = [m for m in info_messages(caplog) if m.startswith("stored ")]
        assert len(store) == 1, store
        assert re.fullmatch(
            r"stored 128,456 pts in \d+ ms \(\+[1-9][0-9]* B payload\) "
            r"collection=hermes_memories",
            store[0],
        ), store[0]
        assert "user said what" not in caplog.text
        assert "assistant replied" not in caplog.text

    def test_chat_store_line_is_unchanged(
        self, provider, _isolate_hermes_home, caplog
    ):
        """The progress line the user already sees must not move a byte."""
        set_level(_isolate_hermes_home, "verbose")
        wire_store(provider)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.sync_turn("hi", "hello", session_id="s")
        emitted = [c.args for c in provider._status_callback.call_args_list]
        assert ("memory_sync", "💾 qdrant — stored (128,456 points)") in emitted

    @pytest.mark.parametrize("level", ["off", "summary"])
    def test_no_store_line_below_verbose(
        self, provider, _isolate_hermes_home, caplog, level
    ):
        set_level(_isolate_hermes_home, level)
        wire_store(provider, points_count=7)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.sync_turn("user text", "assistant text", session_id="s")
        assert not any(m.startswith("stored ") for m in info_messages(caplog))

    def test_store_line_omits_the_count_the_server_did_not_give(
        self, provider, _isolate_hermes_home, caplog
    ):
        """No collection count → say nothing, rather than print a zero."""
        set_level(_isolate_hermes_home, "verbose")
        wire_store(provider, points_count=None)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.sync_turn("user text", "assistant text", session_id="s")
        store = [m for m in info_messages(caplog) if m.startswith("stored ")]
        assert len(store) == 1, store
        assert "pts" not in store[0]
        assert re.fullmatch(
            r"stored in \d+ ms \(\+[1-9][0-9]* B payload\) "
            r"collection=hermes_memories",
            store[0],
        ), store[0]


# ---------------------------------------------------------------------------
# Verbose: md_search
# ---------------------------------------------------------------------------


class TestVerboseMdSearch:

    @pytest.fixture
    def corpus(self, provider, monkeypatch):
        """A present index with two lexical hits; no model, no server."""
        from plugins.memory.qdrant import mdsearch, mdsemantic

        monkeypatch.setattr(mdsearch, "index_is_present", lambda: True)
        monkeypatch.setattr(
            mdsearch, "search_lexical",
            lambda query, limit=5, root="": [
                SimpleNamespace(path="skills/deploy.md", heading="VPS",
                                snippet="the quick brown fox", score=-1.2),
                SimpleNamespace(path="skills/ci.md", heading="Pipelines",
                                snippet="another chunk", score=-0.8),
            ],
        )
        monkeypatch.setattr(
            mdsemantic, "search_semantic",
            lambda query, limit=5, root="": [],
        )
        return provider

    def test_lexical_route_is_logged_with_latency_and_top(
        self, corpus, _isolate_hermes_home, caplog
    ):
        set_level(_isolate_hermes_home, "verbose")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            out = corpus.handle_tool_call("md_search", {"query": "deploy"})
        assert out.startswith("md_search (lexical, 2 result(s)):")
        lines = [m for m in info_messages(caplog) if m.startswith("md_search")]
        assert len(lines) == 1, lines
        assert re.fullmatch(
            r'md_search "deploy" → lexical \d+ ms, 2 hits '
            r"\(top=skills/deploy\.md\)",
            lines[0],
        ), lines[0]
        # c2: the answer's text never reaches the log — only its file label
        assert "quick brown fox" not in caplog.text
        assert "Pipelines" not in caplog.text

    def test_semantic_route_is_recorded_at_the_decision(
        self, corpus, _isolate_hermes_home, caplog
    ):
        set_level(_isolate_hermes_home, "verbose")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            corpus.handle_tool_call(
                "md_search", {"query": "deploy", "semantic": "always"}
            )
        assert corpus._last_md_search["tier"] == "semantic"
        lines = [m for m in info_messages(caplog) if m.startswith("md_search")]
        assert len(lines) == 1, lines
        assert re.fullmatch(
            r'md_search "deploy" → semantic \d+ ms, 2 hits '
            r"\(top=skills/deploy\.md\)",
            lines[0],
        ), lines[0]

    def test_record_is_written_even_when_nothing_is_logged(self, corpus):
        """Tier and latency are state, not a display choice."""
        corpus.handle_tool_call("md_search", {"query": "deploy"})
        record = corpus._last_md_search
        assert record["tier"] == "lexical"
        assert record["hits"] == 2
        assert record["top"] == "skills/deploy.md"
        assert isinstance(record["ms"], int) and record["ms"] >= 0

    @pytest.mark.parametrize("level", ["off", "summary"])
    def test_no_md_search_line_below_verbose(
        self, corpus, _isolate_hermes_home, caplog, level
    ):
        set_level(_isolate_hermes_home, level)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            corpus.handle_tool_call("md_search", {"query": "deploy"})
        assert not any(
            m.startswith("md_search ") for m in info_messages(caplog)
        )
        assert corpus._last_md_search["tier"] == "lexical"

    def test_nothing_is_recorded_when_the_tier_decision_never_happens(
        self, provider, monkeypatch, caplog
    ):
        """No index → no routing decision → no fabricated record."""
        from plugins.memory.qdrant import mdsearch
        monkeypatch.setattr(mdsearch, "index_is_present", lambda: False)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            out = provider.handle_tool_call("md_search", {"query": "deploy"})
        assert out.startswith("md_search: no markdown index yet")
        assert provider._last_md_search is None
        assert not any(
            m.startswith("md_search ") for m in info_messages(caplog)
        )
