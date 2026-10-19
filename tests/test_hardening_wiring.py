"""Provider-wiring tests for the opt-in rival-hardening knobs (2026-10-19).

The feature modules (``gate``, ``temporal``, ``entities``, ``rerank``,
``cascade``, ``filters``, ``history``) are tested in isolation; nothing
until now checked that the provider actually CALLS them. This file asserts
that wiring:

  * ``__init__`` parses every knob (``knob_enabled``) and reflects it on the
    ``_knob_*`` attributes — truthy spellings on, falsy spellings off,
    unrecognized values off, a default install all-off;
  * ``prefetch`` skips trivial queries only when ``gate`` is on, applies the
    temporal/entity/rerank stages only when their knob is on (in that
    order), and is byte-identical to the plain fused format when every knob
    is off;
  * ``sync_turn`` writes a NUMERIC epoch under ``timestamp`` (plus the ISO
    string and ``kind: "turn"``) — the fix for the two-timestamp-format bug
    that Qdrant range filters cannot read — and records an ``add`` event in
    the ledger when ``ledger`` is on;
  * ``_tool_search`` ANDs a validated ``filters`` spec into the session scope
    before it reaches ``_query_points``, and answers an invalid spec with an
    error STRING (the dispatcher's contract) without querying;
  * the config schema and ``get_status_config`` surface expose every knob.

Pure unit tests: no server, no model, no LLM. The provider's own
``_query_points`` / ``_embed`` / stage methods and the module-level feature
entry points are monkeypatched, so every assertion is on a VALUE or on
whether a collaborator was called — never on log text or source.
"""

from __future__ import annotations

import time
import types
from datetime import datetime
from typing import Any

import plugins.memory.qdrant as mod
import pytest
from plugins.memory.qdrant import QdrantMemoryProvider

# ---------------------------------------------------------------------------
# Shared doubles
# ---------------------------------------------------------------------------

#: (config key, provider attribute) for the six boolean hardening knobs.
KNOBS = (
    ("gate", "_knob_gate"),
    ("temporal", "_knob_temporal"),
    ("rerank", "_knob_rerank"),
    ("entities", "_knob_entities"),
    ("cascade", "_knob_cascade"),
    ("ledger", "_knob_ledger"),
)


class _Point:
    """Minimal stand-in for a Qdrant hit."""

    def __init__(self, pid: str, text: str, score: float,
                 payload: dict | None = None) -> None:
        self.id = pid
        self.score = score
        self.payload = payload if payload is not None else {"text": text}


class _Response:
    def __init__(self, points: list) -> None:
        self.points = points


def _stub_query_points(monkeypatch, provider, points):
    """Patch ``_query_points``; return the list of calls it received."""
    calls: list[dict[str, Any]] = []

    def fake(dense_vec, query, query_filter, limit):
        calls.append({"query": query, "filter": query_filter, "limit": limit})
        return _Response(points)

    monkeypatch.setattr(provider, "_query_points", fake)
    return calls


def _prefetchable(monkeypatch, config, tmp_path, points, session_id=""):
    """A provider with a stubbed embedder and query path, ready for prefetch."""
    provider = QdrantMemoryProvider(config=config)
    provider._initialized = True
    provider._client = object()
    provider._hermes_home = str(tmp_path)
    monkeypatch.setattr(provider, "_embed",
                        lambda text: [0.0] * provider._vector_size)
    calls = _stub_query_points(monkeypatch, provider, points)
    return provider, calls


class _FakeLedger:
    """Records record_event / bump_retrieval calls; no sqlite."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.bumps: list[list[str]] = []

    def record_event(self, point_id, event, text=None, ts=None):
        self.events.append((point_id, event, text))

    def bump_retrieval(self, point_ids, ts=None):
        self.bumps.append([str(i) for i in point_ids])

    def close(self):
        pass


class _FakeClient:
    """Records upsert calls; answers get_collection for the progress count."""

    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert(self, collection_name, points, wait):
        self.upserts.append({"collection": collection_name,
                             "points": list(points), "wait": wait})

    def get_collection(self, collection_name):
        return types.SimpleNamespace(points_count=42)


# ---------------------------------------------------------------------------
# 1. Knob parsing in __init__
# ---------------------------------------------------------------------------

class TestKnobParsing:
    """Every opt-in knob is parsed by knob_enabled and stored on the provider."""

    @pytest.mark.parametrize("key,attr", KNOBS)
    @pytest.mark.parametrize("raw", [True, "on", "ON", "True", "YES", "1"])
    def test_truthy_values_enable_the_knob(self, key, attr, raw):
        provider = QdrantMemoryProvider(config={key: raw})
        assert getattr(provider, attr) is True

    @pytest.mark.parametrize("key,attr", KNOBS)
    @pytest.mark.parametrize("raw", [False, "off", "OFF", "false", "no", "0", "", None])
    def test_falsy_values_disable_the_knob(self, key, attr, raw):
        provider = QdrantMemoryProvider(config={key: raw})
        assert getattr(provider, attr) is False

    @pytest.mark.parametrize("key,attr", KNOBS)
    @pytest.mark.parametrize("raw", ["maybe", "enabled", 2, "2023"])
    def test_unrecognized_value_resolves_off(self, key, attr, raw):
        # A typo must never switch a feature ON — and never raise either.
        provider = QdrantMemoryProvider(config={key: raw})
        assert getattr(provider, attr) is False

    def test_default_config_leaves_all_knobs_off(self, monkeypatch):
        monkeypatch.setattr(mod, "_load_plugin_config", lambda: {})
        provider = QdrantMemoryProvider()
        for _key, attr in KNOBS:
            assert getattr(provider, attr) is False

    def test_mixed_config_reflects_each_knob(self):
        provider = QdrantMemoryProvider(config={"gate": "on", "ledger": "yes"})
        assert provider._knob_gate is True
        assert provider._knob_ledger is True
        assert provider._knob_temporal is False
        assert provider._knob_rerank is False
        assert provider._knob_entities is False
        assert provider._knob_cascade is False

    @pytest.mark.parametrize("raw,expected", [
        (True, True), (False, False), ("on", True), ("off", False),
        (None, False), ("", False), ("yes", True), ("0", False),
    ])
    def test_knob_enabled_matches_hybrid_truth_table(self, raw, expected):
        assert mod.knob_enabled(raw, name="probe") is expected


# ---------------------------------------------------------------------------
# 2. The trivial-message gate
# ---------------------------------------------------------------------------

class TestGateWiring:
    """gate: on skips trivial recall queries; off never skips anything."""

    TRIVIAL = "ok thanks"
    SUBSTANTIVE = "what is the deployment target for the backend?"

    def test_gate_on_skips_trivial_query_without_embedding_or_querying(
        self, monkeypatch, tmp_path
    ):
        provider, calls = _prefetchable(
            monkeypatch, {"gate": "on"}, tmp_path,
            [_Point("id-a", "espresso in the morning", 0.9)],
        )
        out = provider.prefetch(self.TRIVIAL, session_id="s1")
        assert out == ""
        assert calls == [], "a gated trivial query must never reach _query_points"

    def test_gate_on_does_not_skip_a_substantive_query(
        self, monkeypatch, tmp_path
    ):
        provider, calls = _prefetchable(
            monkeypatch, {"gate": "on"}, tmp_path,
            [_Point("id-a", "espresso in the morning", 0.9)],
        )
        provider.prefetch(self.SUBSTANTIVE, session_id="s1")
        assert len(calls) == 1

    def test_gate_off_never_skips(self, monkeypatch, tmp_path):
        provider, calls = _prefetchable(
            monkeypatch, {}, tmp_path,
            [_Point("id-a", "espresso in the morning", 0.9)],
        )
        provider.prefetch(self.TRIVIAL, session_id="s1")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3. The prefetch hardening stages
# ---------------------------------------------------------------------------

class TestPrefetchPlainFormat:
    """With every knob off the output is the plain fused format, byte for byte."""

    POINTS = [
        _Point("id-a", "espresso in the morning", 0.91,
               {"text": "espresso in the morning", "timestamp": 1700000000.0}),
        _Point("id-b", "deploy target is a debian vps", 0.82,
               {"text": "deploy target is a debian vps", "timestamp": 1700000100.0}),
    ]
    EXPECTED = (
        "- [0.91] espresso in the morning\n- [0.82] deploy target is a debian vps"
    )

    def test_all_knobs_off_output_is_byte_identical(self, monkeypatch, tmp_path):
        provider, calls = _prefetchable(monkeypatch, {}, tmp_path, self.POINTS)
        out = provider.prefetch("morning coffee setup", session_id="sess-1")
        assert out == self.EXPECTED

    def test_all_knobs_off_still_passes_the_session_scope(
        self, monkeypatch, tmp_path
    ):
        provider, calls = _prefetchable(monkeypatch, {}, tmp_path, self.POINTS)
        provider.prefetch("morning coffee setup", session_id="sess-1")
        # The archived exclusion rides on every recall filter: the decay scan
        # and the dedup merge mark a point rather than deleting it, so the mark
        # has to be honoured here or archiving would mean nothing. It is a no-op
        # on a store that has never been scanned (a point without the field is
        # not a match for archived: true).
        assert calls[0]["filter"] == {
            "must": [{"key": "session_id", "match": {"value": "sess-1"}}],
            "must_not": [{"key": "archived", "match": {"value": True}}],
        }
        assert calls[0]["limit"] == 10


class TestTemporalStage:
    """temporal: on restricts to the parsed window; off does not parse at all."""

    WINDOW = (1000.0, 2000.0)

    def _points(self):
        return [
            _Point("id-old", "ancient memory about espresso", 0.9,
                   {"text": "ancient memory about espresso", "timestamp": 500.0}),
            _Point("id-new", "recent memory about espresso", 0.8,
                   {"text": "recent memory about espresso", "timestamp": 1500.0}),
            _Point("id-none", "memory without a timestamp", 0.7,
                   {"text": "memory without a timestamp"}),
        ]

    def _window_recorder(self, monkeypatch, window):
        seen: list[str] = []
        monkeypatch.setattr(
            mod, "parse_time_window",
            lambda query: seen.append(query) or window,
        )
        return seen

    def test_window_excludes_old_candidate_and_keeps_timeless_one(
        self, monkeypatch, tmp_path
    ):
        seen = self._window_recorder(monkeypatch, self.WINDOW)
        provider, _calls = _prefetchable(monkeypatch, {"temporal": "on"},
                                        tmp_path, self._points())
        out = provider.prefetch("memories about espresso", session_id="s1")

        assert seen == ["memories about espresso"]
        assert out == ("- [0.80] recent memory about espresso\n"
                       "- [0.70] memory without a timestamp")
        assert "ancient" not in out

    def test_temporal_off_never_calls_parse_time_window(
        self, monkeypatch, tmp_path
    ):
        seen = self._window_recorder(monkeypatch, self.WINDOW)
        provider, _calls = _prefetchable(monkeypatch, {}, tmp_path, self._points())
        out = provider.prefetch("memories about espresso", session_id="s1")

        assert seen == [], "the knob gates the parse, not the query"
        assert out == ("- [0.90] ancient memory about espresso\n"
                       "- [0.80] recent memory about espresso\n"
                       "- [0.70] memory without a timestamp")

    def test_temporal_on_without_a_parsed_window_keeps_everything(
        self, monkeypatch, tmp_path
    ):
        self._window_recorder(monkeypatch, None)
        provider, _calls = _prefetchable(monkeypatch, {"temporal": "on"},
                                        tmp_path, self._points())
        out = provider.prefetch("no time expressed here", session_id="s1")
        assert out.count("- [") == 3


class TestEntityStage:
    """entities: on re-ranks by shared entities; the module is called either way."""

    def _entity_module(self, monkeypatch):
        calls: dict[str, list] = {"extract": [], "boost": []}
        fake = types.SimpleNamespace(
            extract_entities=lambda query: calls["extract"].append(query) or ["Qdrant"],
            apply_entity_boost=lambda pairs, query_entities: (
                calls["boost"].append((list(pairs), list(query_entities)))
                or [(0.5, "the qdrant cluster runs on port 6333"),
                    (0.95, "deploy target is a debian vps")]
            ),
        )
        monkeypatch.setattr(mod, "_entities", fake)
        return calls

    def test_entity_sharing_candidate_outranks_a_higher_scored_one(
        self, monkeypatch, tmp_path
    ):
        calls = self._entity_module(monkeypatch)
        points = [
            _Point("id-plain", "deploy target is a debian vps", 0.95),
            _Point("id-qdrant", "the qdrant cluster runs on port 6333", 0.5),
        ]
        provider, _q = _prefetchable(monkeypatch, {"entities": "on"}, tmp_path, points)
        out = provider.prefetch("what about the Qdrant cluster?", session_id="s1")

        assert out == ("- [0.50] the qdrant cluster runs on port 6333\n"
                       "- [0.95] deploy target is a debian vps")
        assert calls["extract"] == ["what about the Qdrant cluster?"]
        boosted_texts = [text for _score, text in calls["boost"][0][0]]
        assert boosted_texts == ["deploy target is a debian vps",
                                 "the qdrant cluster runs on port 6333"]
        assert calls["boost"][0][1] == ["Qdrant"]

    def test_entities_off_never_calls_the_module(self, monkeypatch, tmp_path):
        calls = self._entity_module(monkeypatch)
        points = [_Point("id-plain", "deploy target is a debian vps", 0.95)]
        provider, _q = _prefetchable(monkeypatch, {}, tmp_path, points)
        provider.prefetch("what about the Qdrant cluster?", session_id="s1")
        assert calls["extract"] == [] and calls["boost"] == []


class TestRerankStage:
    """rerank: on follows the reranker's order; off never consults it."""

    def test_rerank_on_follows_the_reranker_order(self, monkeypatch, tmp_path):
        points = [
            _Point("id-a", "fused first but less relevant", 0.95),
            _Point("id-b", "fused second but more relevant", 0.5),
        ]
        reranked: list[tuple[float, str]] = []

        class _FakeReranker:
            def rerank(self, query, pairs, *, top_k=None):
                reranked.append((query, list(pairs)))
                return [(0.3, "fused second but more relevant"),
                        (0.95, "fused first but less relevant")]

        provider, _q = _prefetchable(monkeypatch, {"rerank": "on"}, tmp_path, points)
        provider._reranker = _FakeReranker()
        out = provider.prefetch("which deployment target", session_id="s1")

        assert out == ("- [0.50] fused second but more relevant\n"
                       "- [0.95] fused first but less relevant")
        assert reranked[0][0] == "which deployment target"
        assert [text for _s, text in reranked[0][1]] == [
            "fused first but less relevant", "fused second but more relevant"]

    def test_rerank_off_never_consults_the_reranker(self, monkeypatch, tmp_path):
        asked: list[str] = []

        class _FakeReranker:
            def rerank(self, query, pairs, *, top_k=None):
                asked.append(query)
                return list(pairs)

        points = [_Point("id-a", "fused order is kept", 0.95)]
        provider, _q = _prefetchable(monkeypatch, {}, tmp_path, points)
        provider._reranker = _FakeReranker()
        out = provider.prefetch("which deployment target", session_id="s1")
        assert asked == []
        assert out == "- [0.95] fused order is kept"

    def test_rerank_model_config_reaches_the_reranker_constructor(
        self, monkeypatch, tmp_path
    ):
        built: list[str] = []

        class _RecordingReranker:
            def __init__(self, model_name, **kwargs):
                built.append(model_name)

            def rerank(self, query, pairs, *, top_k=None):
                return list(pairs)

        monkeypatch.setattr(mod._rerank, "Reranker", _RecordingReranker)
        points = [_Point("id-a", "keeps its fused order", 0.9)]
        provider, _q = _prefetchable(
            monkeypatch, {"rerank": "on", "rerank_model": "org/custom-model"},
            tmp_path, points)
        provider.prefetch("any substantive query", session_id="s1")
        assert built == ["org/custom-model"]

    def test_rerank_without_a_model_config_uses_the_default(
        self, monkeypatch, tmp_path
    ):
        built: list[str] = []

        class _RecordingReranker:
            def __init__(self, model_name, **kwargs):
                built.append(model_name)

            def rerank(self, query, pairs, *, top_k=None):
                return list(pairs)

        monkeypatch.setattr(mod._rerank, "Reranker", _RecordingReranker)
        points = [_Point("id-a", "keeps its fused order", 0.9)]
        provider, _q = _prefetchable(monkeypatch, {"rerank": "on"}, tmp_path, points)
        provider.prefetch("any substantive query", session_id="s1")
        assert built == [mod._rerank.DEFAULT_MODEL]


class TestStageOrder:
    """The stages run temporal -> entities -> rerank, each gated by its knob."""

    def test_all_three_stages_run_in_order(self, monkeypatch, tmp_path):
        order: list[str] = []

        monkeypatch.setattr(mod, "parse_time_window",
                            lambda query: (0.0, 2.0 ** 40))
        monkeypatch.setattr(mod, "_entities", types.SimpleNamespace(
            extract_entities=lambda query: [],
            apply_entity_boost=lambda pairs, query_entities: list(pairs),
        ))
        points = [
            _Point("id-a", "first candidate text", 0.9,
                   {"text": "first candidate text", "timestamp": 100.0}),
            _Point("id-b", "second candidate text", 0.8,
                   {"text": "second candidate text", "timestamp": 200.0}),
        ]
        provider, _q = _prefetchable(
            monkeypatch,
            {"temporal": "on", "entities": "on", "rerank": "on"},
            tmp_path, points)

        originals = (provider._apply_temporal_window,
                     provider._apply_entity_stage,
                     provider._apply_rerank_stage)

        def temporal_spy(cands, window):
            order.append("temporal")
            return originals[0](cands, window)

        def entities_spy(cands, query):
            order.append("entities")
            return originals[1](cands, query)

        def rerank_spy(cands, query):
            order.append("rerank")
            return originals[2](cands, query)

        monkeypatch.setattr(provider, "_apply_temporal_window", temporal_spy)
        monkeypatch.setattr(provider, "_apply_entity_stage", entities_spy)
        monkeypatch.setattr(provider, "_apply_rerank_stage", rerank_spy)

        out = provider.prefetch("a substantive query about candidates", session_id="s1")

        assert order == ["temporal", "entities", "rerank"]
        # Every stage kept its input here, so the fused format is unchanged.
        assert out == ("- [0.90] first candidate text\n"
                       "- [0.80] second candidate text")

    def test_only_the_enabled_stages_run(self, monkeypatch, tmp_path):
        order: list[str] = []
        points = [_Point("id-a", "only candidate text", 0.9)]
        provider, _q = _prefetchable(monkeypatch, {"temporal": "on"},
                                    tmp_path, points)
        originals = provider._apply_temporal_window

        def temporal_spy(cands, window):
            order.append("temporal")
            return originals(cands, window)

        monkeypatch.setattr(provider, "_apply_temporal_window", temporal_spy)
        monkeypatch.setattr(mod, "parse_time_window",
                            lambda query: (0.0, 2.0 ** 40))
        provider.prefetch("a substantive query about candidates", session_id="s1")
        assert order == ["temporal"]


# ---------------------------------------------------------------------------
# The degradation cascade
# ---------------------------------------------------------------------------

class TestCascadeWiring:
    """cascade: off calls _query_points directly; on runs the cascade."""

    def test_cascade_off_calls_query_points_directly(self, monkeypatch, tmp_path):
        cascades: list[list] = []
        monkeypatch.setattr(
            mod, "run_cascade",
            lambda levels, on_error=None: cascades.append(list(levels))
            or mod._cascade.CascadeResult(points=[], level="none",
                                          reason="", attempted=[]),
        )
        points = [_Point("id-a", "espresso in the morning", 0.9)]
        provider, calls = _prefetchable(monkeypatch, {}, tmp_path, points)
        provider.prefetch("morning coffee setup", session_id="s1")

        assert cascades == []
        assert len(calls) == 1

    def test_cascade_on_runs_the_cascade_and_returns_its_points(
        self, monkeypatch, tmp_path
    ):
        levels_seen: list[list[str]] = []

        def fake_run_cascade(levels, *, on_error=None):
            levels_seen.append([name for name, _fn in levels])
            return mod._cascade.CascadeResult(
                points=[_Point("id-c", "cascade answered", 0.7)],
                level="dense", reason="", attempted=["dense", "lexical"],
            )

        monkeypatch.setattr(mod, "run_cascade", fake_run_cascade)
        provider, calls = _prefetchable(
            monkeypatch, {"cascade": "on"}, tmp_path,
            [_Point("id-a", "should not appear", 0.9)],
        )
        out = provider.prefetch("morning coffee setup", session_id="s1")

        assert levels_seen == [["dense", "lexical"]]
        assert calls == [], (
            "with the cascade on, recall no longer calls _query_points directly"
        )
        assert out == "- [0.70] cascade answered"


# ---------------------------------------------------------------------------
# 4. sync_turn's payload and the usage ledger
# ---------------------------------------------------------------------------

class TestSyncTurnPayload:
    """sync_turn writes a NUMERIC epoch under `timestamp`, plus ISO and kind."""

    def _provider(self, monkeypatch, tmp_path, config=None):
        provider = QdrantMemoryProvider(config=config or {})
        provider._initialized = True
        provider._client = _FakeClient()
        provider._hermes_home = str(tmp_path)
        monkeypatch.setattr(provider, "_embed",
                            lambda text: [0.0] * provider._vector_size)
        return provider

    def test_payload_timestamp_is_numeric_epoch_and_iso_parses(
        self, monkeypatch, tmp_path
    ):
        provider = self._provider(monkeypatch, tmp_path)
        provider.sync_turn("we use qdrant for memory", "qdrant is a vector db",
                           session_id="s1")

        assert len(provider._client.upserts) == 1
        upsert = provider._client.upserts[0]
        assert upsert["collection"] == provider._collection
        assert upsert["wait"] is True
        assert len(upsert["points"]) == 1

        point = upsert["points"][0]
        payload = point["payload"]
        assert isinstance(payload["timestamp"], float), (
            "Qdrant range filters accept numbers only — the ISO form is unreadable"
        )
        assert abs(time.time() - payload["timestamp"]) < 60
        stamp = datetime.fromisoformat(payload["timestamp_iso"])
        assert stamp.tzinfo is not None
        assert stamp.timestamp() == pytest.approx(payload["timestamp"], abs=1.0)
        assert payload["kind"] == "turn"
        assert payload["text"].startswith("user: we use qdrant for memory")
        assert payload["session_id"] == "s1"
        assert str(point["id"])

    def test_gate_on_skips_a_trivial_turn(self, monkeypatch, tmp_path):
        provider = self._provider(monkeypatch, tmp_path, config={"gate": "on"})
        provider.sync_turn("ok", "thanks", session_id="s1")
        assert provider._client.upserts == []


class TestLedgerWiring:
    """ledger: on records an add per store and a bump per recall; off, nothing."""

    def _fake_ledger(self, monkeypatch, provider):
        ledger = _FakeLedger()
        getter_calls: list[bool] = []
        monkeypatch.setattr(
            provider, "_get_ledger",
            lambda: getter_calls.append(True) or ledger,
        )
        return ledger, getter_calls

    def test_ledger_off_means_no_ledger_and_no_file(
        self, monkeypatch, tmp_path
    ):
        points = [_Point("id-a", "espresso in the morning", 0.9)]
        provider, _q = _prefetchable(monkeypatch, {}, tmp_path, points)
        getter_calls: list[bool] = []
        monkeypatch.setattr(provider, "_get_ledger",
                            lambda: getter_calls.append(True))
        monkeypatch.setattr(provider, "_client", _FakeClient())

        provider.sync_turn("we use qdrant for memory", "qdrant is a vector db",
                           session_id="s1")
        provider.prefetch("morning coffee setup", session_id="s1")

        assert getter_calls == []
        assert provider._ledger is None
        assert not (tmp_path / "qdrant-memory-ledger.db").exists()

    def test_sync_turn_records_an_add_event_when_ledger_on(
        self, monkeypatch, tmp_path
    ):
        provider = QdrantMemoryProvider(config={"ledger": "on"})
        provider._initialized = True
        client = _FakeClient()
        provider._client = client
        provider._hermes_home = str(tmp_path)
        monkeypatch.setattr(provider, "_embed",
                            lambda text: [0.0] * provider._vector_size)
        ledger, getter_calls = self._fake_ledger(monkeypatch, provider)

        provider.sync_turn("we use qdrant for memory", "qdrant is a vector db",
                           session_id="s1")

        assert getter_calls == [True]
        assert len(ledger.events) == 1
        point_id, event, text = ledger.events[0]
        assert event == "add"
        upserted_id = str(client.upserts[0]["points"][0]["id"])
        assert point_id == upserted_id, "the ledger must name the stored point"
        assert text == (
            "user: we use qdrant for memory\nassistant: qdrant is a vector db"
        )

    def test_prefetch_bumps_retrieval_counts_when_ledger_on(
        self, monkeypatch, tmp_path
    ):
        points = [
            _Point("id-a", "espresso in the morning", 0.91),
            _Point("id-b", "deploy target is a debian vps", 0.82),
        ]
        provider, _q = _prefetchable(monkeypatch, {"ledger": "on"}, tmp_path, points)
        ledger, getter_calls = self._fake_ledger(monkeypatch, provider)

        provider.prefetch("morning coffee setup", session_id="s1")

        assert getter_calls == [True]
        assert ledger.bumps == [["id-a", "id-b"]]


# ---------------------------------------------------------------------------
# 5. _tool_search filter wiring
# ---------------------------------------------------------------------------

class TestToolSearchFilters:
    """A valid spec ANDs the session scope; an invalid one is an error string."""

    def _search(self, monkeypatch, tmp_path, args):
        provider = QdrantMemoryProvider(config={})
        provider._initialized = True
        provider._hermes_home = str(tmp_path)
        monkeypatch.setattr(provider, "_embed",
                            lambda text: [0.0] * provider._vector_size)
        calls = _stub_query_points(
            monkeypatch, provider,
            [_Point("id-a", "espresso in the morning", 0.9,
                    {"text": "espresso in the morning"})],
        )
        out = provider._tool_search(args)
        return out, calls

    def test_valid_spec_reaches_query_points_with_scope_anded_in(
        self, monkeypatch, tmp_path
    ):
        out, calls = self._search(monkeypatch, tmp_path, {
            "query": "coffee", "session_id": "s1", "filters": {"kind": "turn"},
        })
        assert "espresso in the morning" in out
        assert calls[0]["filter"] == {
            "must": [
                {"key": "session_id", "match": {"value": "s1"}},
                {"key": "kind", "match": {"value": "turn"}},
            ],
            "must_not": [{"key": "archived", "match": {"value": True}}],
        }

    def test_range_spec_reaches_query_points_as_a_range_clause(
        self, monkeypatch, tmp_path
    ):
        out, calls = self._search(monkeypatch, tmp_path, {
            "query": "coffee", "session_id": "s1", "filters": {"ts": {"gt": 100}},
        })
        assert "espresso in the morning" in out
        assert calls[0]["filter"]["must"][0] == {
            "key": "session_id", "match": {"value": "s1"},
        }
        assert {"key": "ts", "range": {"gt": 100}} in calls[0]["filter"]["must"]

    def test_absent_filters_leaves_the_scope_only(self, monkeypatch, tmp_path):
        _out, calls = self._search(monkeypatch, tmp_path,
                                   {"query": "coffee", "session_id": "s1"})
        assert calls[0]["filter"] == {
            "must": [{"key": "session_id", "match": {"value": "s1"}}],
            "must_not": [{"key": "archived", "match": {"value": True}}],
        }

    @pytest.mark.parametrize("spec", [
        {"kind": {"bogus_op": "x"}},   # unknown operator
        {"kind": {"gt": "soon"}},      # range operator on a non-number
        {"kind": None},                # null value
        {"kind": {"eq": "turn", "nin": []}},  # degenerate membership
        "not-a-mapping",               # not a mapping at all
    ])
    def test_invalid_spec_is_an_error_string_and_never_queries(
        self, monkeypatch, tmp_path, spec
    ):
        out, calls = self._search(monkeypatch, tmp_path, {
            "query": "coffee", "session_id": "s1", "filters": spec,
        })
        assert isinstance(out, str)
        assert out.startswith("qdrant_search: invalid filters")
        assert calls == [], "an invalid spec must never reach _query_points"

    def test_scope_conflicting_spec_is_rejected(self, monkeypatch, tmp_path):
        out, calls = self._search(monkeypatch, tmp_path, {
            "query": "coffee", "session_id": "s1",
            "filters": {"session_id": "other"},
        })
        assert out.startswith("qdrant_search: invalid filters")
        assert calls == []

    def test_missing_query_is_an_error_and_never_queries(self, monkeypatch, tmp_path):
        out, calls = self._search(monkeypatch, tmp_path,
                                  {"filters": {"kind": "turn"}})
        assert isinstance(out, str)
        assert out.startswith("qdrant_search: missing 'query'")
        assert calls == []


# ---------------------------------------------------------------------------
# 6. Config schema + status surface
# ---------------------------------------------------------------------------

class TestConfigSurface:
    """get_config_schema and get_status_config expose every hardening knob."""

    HARDENING_KEYS = ("gate", "temporal", "rerank", "rerank_model",
                      "entities", "cascade", "ledger")

    def test_schema_declares_every_hardening_field(self, qdrant_provider):
        fields = {f["key"]: f for f in qdrant_provider.get_config_schema()}
        for key in self.HARDENING_KEYS:
            assert key in fields, f"schema is missing the {key} field"
            assert isinstance(fields[key]["type"], str) and fields[key]["type"]
            assert "default" in fields[key]

    def test_schema_defaults_are_off_for_the_six_switches(self, qdrant_provider):
        fields = {f["key"]: f for f in qdrant_provider.get_config_schema()}
        for key in self.HARDENING_KEYS:
            if key == "rerank_model":
                assert fields[key]["default"]
            else:
                assert fields[key]["default"] == "off"

    def test_status_config_reports_each_knob_on(self, monkeypatch, tmp_path):
        provider = QdrantMemoryProvider(
            config={key: "on" for key, _attr in KNOBS})
        provider._hermes_home = str(tmp_path)
        hardening = provider.get_status_config({})["hardening"]
        for key, _attr in KNOBS:
            assert hardening[key] == "on"

    def test_status_config_reports_each_knob_off_by_default(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(mod, "_load_plugin_config", lambda: {})
        provider = QdrantMemoryProvider()
        provider._hermes_home = str(tmp_path)
        hardening = provider.get_status_config({})["hardening"]
        assert hardening == {key: "off" for key, _attr in KNOBS}
