"""Tests for the session metrics (0.1.9): hit-rate, status counters, p50/p95.

Three surfaces, one rule each:

* **Summary label** gains a ``hits N/M`` segment — but only after the first
  recall attempt (0/0 is noise) and never at ``off`` (byte-identical default).
  A FAILED attempt counts (that turn got no usable recall); a breaker-skipped
  turn does not (no recall was attempted).
* **``hermes memory status``** (``get_status_config``) reports the session
  counters and the circuit breaker unconditionally (honest zeros in a fresh
  process), and the latency percentiles only once a recall completed — an
  empty window omits the keys rather than faking a 0.
* **The verbose recall line** carries p50/p95 over the rolling window, numbers
  only, never the chat line.

The indicator rendering is reproduced the same way ``test_display_levels``
does it: core's ``MemoryManager.describe_recall`` is not on this repo's import
path everywhere, and the format is quoted verbatim from core.
"""

from __future__ import annotations

import json
import logging
import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

LOGGER = "hermes.plugins.memory.qdrant"


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Give the test its own ``<HERMES_HOME>/qdrant.json``."""
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


def hit(text: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(payload={"text": text}, score=score)


def wire_prefetch(provider, points) -> None:
    """An initialized provider whose client and embedder are stubs."""
    provider._initialized = True
    provider._client = MagicMock()
    provider._client.query_points.return_value = SimpleNamespace(points=points)
    provider._embed = lambda text: [0.1] * 4


def info_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


# ---------------------------------------------------------------------------
# The hit-rate segment of the summary label
# ---------------------------------------------------------------------------


class TestSessionHitRate:
    def test_no_segment_before_first_attempt(
        self, provider, _isolate_hermes_home
    ):
        """No attempts yet → no hits segment (0/0 is noise), ms/collection stay."""
        set_level(_isolate_hermes_home, "summary")
        provider._last_recall_count = 5
        provider._last_recall_ms = 41
        assert provider.recall_status().provider_label == (
            "qdrant · 41ms · hermes_memories"
        )

    def test_segment_appears_once_attempted(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "summary")
        provider._recall_attempts = 5
        provider._recall_hits = 3
        provider._last_recall_count = 3
        provider._last_recall_ms = 41
        assert provider.recall_status().provider_label == (
            "qdrant · 41ms · hermes_memories · hits 3/5"
        )

    def test_off_never_shows_the_segment(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "off")
        provider._recall_attempts = 5
        provider._recall_hits = 3
        assert provider.recall_status().provider_label == "qdrant"

    def test_successful_prefetch_counts_one_hit(
        self, provider, _isolate_hermes_home
    ):
        set_level(_isolate_hermes_home, "summary")
        wire_prefetch(provider, [hit("alpha", 0.9), hit("beta", 0.8)])
        assert provider.prefetch("q", session_id="s1")
        assert provider._recall_attempts == 1
        assert provider._recall_hits == 1
        label = provider.recall_status().provider_label
        assert re.search(r"· hits 1/1$", label), label

    def test_failed_prefetch_counts_attempt_without_hit(
        self, provider, _isolate_hermes_home, caplog
    ):
        """A recall that blows up still happened — the user got nothing."""
        set_level(_isolate_hermes_home, "summary")
        wire_prefetch(provider, [])
        provider._client.query_points.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert provider.prefetch("q", session_id="s1") == ""
        assert provider._recall_attempts == 1
        assert provider._recall_hits == 0
        label = provider.recall_status().provider_label
        assert re.search(r"· hits 0/1$", label), label

    def test_breaker_skip_is_not_an_attempt(
        self, provider, _isolate_hermes_home
    ):
        """Past-the-breaker is the counting line: a skipped turn never ran."""
        set_level(_isolate_hermes_home, "summary")
        wire_prefetch(provider, [hit("alpha", 0.9)])
        # Open = threshold failures reached AND cooldown not yet expired.
        provider._breaker_failures = provider._BREAKER_THRESHOLD
        provider._breaker_open_until = time.time() + 60
        assert provider.prefetch("q", session_id="s1") == ""
        assert provider._recall_attempts == 0
        assert provider._recall_hits == 0
        # summary still shows the collection segment; no ms, no hits.
        assert provider.recall_status().provider_label == (
            "qdrant · hermes_memories"
        )


# ---------------------------------------------------------------------------
# hermes memory status (get_status_config)
# ---------------------------------------------------------------------------


class TestStatusSessionMetrics:
    def test_fresh_process_reports_honest_zeros(self, provider):
        cfg = provider.get_status_config({})
        assert cfg["recall_attempts"] == 0
        assert cfg["recall_hits"] == 0
        assert cfg["breaker_failures"] == 0
        assert cfg["breaker_open"] is False

    def test_breaker_state_is_surfaced(self, provider):
        provider._breaker_failures = provider._BREAKER_THRESHOLD
        provider._breaker_open_until = time.time() + 60
        cfg = provider.get_status_config({})
        assert cfg["breaker_failures"] == provider._BREAKER_THRESHOLD
        assert cfg["breaker_open"] is True

    def test_failures_below_threshold_are_not_open(self, provider):
        provider._breaker_failures = provider._BREAKER_THRESHOLD - 1
        provider._breaker_open_until = time.time() + 60
        assert provider.get_status_config({})["breaker_open"] is False

    def test_percentiles_omitted_while_window_empty(self, provider):
        cfg = provider.get_status_config({})
        assert "recall_ms_p50" not in cfg
        assert "recall_ms_p95" not in cfg

    def test_percentiles_appear_once_a_recall_completed(self, provider):
        provider._recall_ms_window.extend([10, 20, 30, 40])
        cfg = provider.get_status_config({})
        assert cfg["recall_ms_p50"] == 20   # nearest-rank
        assert cfg["recall_ms_p95"] == 40

    def test_completed_recall_fills_the_window(self, provider, monkeypatch):
        wire_prefetch(provider, [hit("alpha", 0.9)])
        provider.prefetch("q", session_id="s1")
        assert len(provider._recall_ms_window) == 1
        cfg = provider.get_status_config({})
        assert "recall_ms_p50" in cfg

    def test_percentile_helper_nearest_rank(self):
        import plugins.memory.qdrant as mod
        assert mod._percentile([], 50) == 0
        assert mod._percentile([7], 50) == 7
        assert mod._percentile([10, 20, 30, 40], 50) == 20
        assert mod._percentile([10, 20, 30, 40], 95) == 40
        assert mod._percentile([5, 1, 3], 100) == 5


# ---------------------------------------------------------------------------
# The verbose recall line
# ---------------------------------------------------------------------------


class TestVerbosePercentiles:
    def test_recall_line_carries_p50_p95_after_two_recalls(
        self, provider, _isolate_hermes_home, caplog
    ):
        set_level(_isolate_hermes_home, "verbose")
        wire_prefetch(provider, [hit("alpha", 0.9)])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.prefetch("q", session_id="s1")
            provider.prefetch("q", session_id="s1")
        recall_lines = [
            m for m in info_messages(caplog) if m.startswith("recall ")
        ]
        assert recall_lines, "no recall line at verbose"
        assert re.search(r"p50=\d+ms p95=\d+ms", recall_lines[-1]), (
            recall_lines[-1]
        )

    def test_first_recall_has_no_percentile_segment(
        self, provider, _isolate_hermes_home, caplog
    ):
        """One sample is not a distribution — the segment waits for the 2nd."""
        set_level(_isolate_hermes_home, "verbose")
        wire_prefetch(provider, [hit("alpha", 0.9)])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            provider.prefetch("q", session_id="s1")
        recall_lines = [
            m for m in info_messages(caplog) if m.startswith("recall ")
        ]
        assert recall_lines
        assert "p50=" not in recall_lines[-1]
