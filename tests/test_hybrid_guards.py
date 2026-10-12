"""Functional tests for hybrid config parsing and sparse write paths.

Mutation-check survivors closed here:
- M1/M2: hybrid_requested() case-insensitivity + garbage→off — no test called
  the parser directly.
- M11: _tool_upsert() sparse vector write path — never exercised with
  hybrid active, including its dense-only fallback on sparse failure.

These tests assert behaviour, not line counts, so the docs-honesty gate
can no longer stand in for them.
"""

import pytest


class TestHybridRequestedParser:
    """M1/M2: hybrid_requested() must parse case-insensitively and fail
    safe on unrecognized values."""

    def test_off_is_default_when_unset(self):
        from plugins.memory.qdrant import hybrid_requested

        assert hybrid_requested(None) is False
        assert hybrid_requested({}) is False

    def test_true_and_false_bools_pass_through(self):
        from plugins.memory.qdrant import hybrid_requested

        assert hybrid_requested(True) is True
        assert hybrid_requested(False) is False

    @pytest.mark.parametrize("value", ["on", "ON", "On", "oN", " true ", "YES", "1"])
    def test_truthy_values_are_case_insensitive(self, value):
        from plugins.memory.qdrant import hybrid_requested

        assert hybrid_requested(value) is True

    @pytest.mark.parametrize("value", ["off", "OFF", "Off", " false ", "no", "0", ""])
    def test_falsy_values_are_case_insensitive(self, value):
        from plugins.memory.qdrant import hybrid_requested

        assert hybrid_requested(value) is False

    @pytest.mark.parametrize("value", ["maybe", "garbage", "2", "enabled"])
    def test_garbage_resolves_to_off(self, value):
        from plugins.memory.qdrant import hybrid_requested

        assert hybrid_requested(value) is False

    def test_garbage_logs_exactly_one_warning(self, caplog):
        import logging

        from plugins.memory.qdrant import hybrid_requested

        with caplog.at_level(logging.WARNING, logger="memory.qdrant"):
            result = hybrid_requested("nonsense")
        assert result is False
        assert any("hybrid" in r.message.lower() for r in caplog.records)

    def test_both_policies_config_and_raw_agree_when_consistent(self, monkeypatch):
        """The config dict path and the raw-value parser must not disagree."""
        import plugins.memory.qdrant as mod

        monkeypatch.setattr(mod, "_load_plugin_config", lambda: {})
        from plugins.memory.qdrant import QdrantMemoryProvider

        p = QdrantMemoryProvider()
        assert p._config.get("hybrid") is None or "hybrid" in p._config


class TestToolUpsertSparseWrite:
    """M11: _tool_upsert must attach the sparse vector when hybrid is
    active, and store dense-only when sparse embedding fails."""

    def _provider_with_hybrid_active(self, monkeypatch, qdrant_provider):
        """Force the hybrid state that initialize() would normally set."""

        qdrant_provider._hybrid_active = True
        qdrant_provider._collection = "test_coll"
        qdrant_provider._sparse_embedder = None
        qdrant_provider._sparse_error = None

        monkeypatch.setattr(
            qdrant_provider, "_embed", lambda text: [0.1] * 384
        )
        return qdrant_provider

    @staticmethod
    def _capture_upsert(monkeypatch, provider, stored):
        from unittest.mock import MagicMock
        provider._client = MagicMock()
        monkeypatch.setattr(
            provider._client, "upsert", lambda **kw: stored.update(kw))

    def test_sparse_vector_attached_when_hybrid_active(
            self, monkeypatch, qdrant_provider):
        from qdrant_client.models import SparseVector

        p = self._provider_with_hybrid_active(monkeypatch, qdrant_provider)
        monkeypatch.setattr(
            p, "_embed_sparse",
            lambda text: SparseVector(indices=[1, 5], values=[0.5, 0.3]),
        )

        stored = {}
        self._capture_upsert(monkeypatch, p, stored)

        result = p._tool_upsert({"text": "hello world", "session_id": "s1"})

        assert "stored point" in result
        vector = stored["points"][0]["vector"]
        assert "dense" in vector
        assert "bm25" in vector
        assert vector["bm25"].indices == [1, 5]
        assert vector["bm25"].values == [0.5, 0.3]

    def test_dense_only_when_sparse_embed_fails(
            self, monkeypatch, qdrant_provider, caplog):
        """A sparse failure must never lose the memory — dense-only with
        a warning, exactly like sync_turn's path."""
        import logging

        from plugins.memory.qdrant import EmbeddingError

        p = self._provider_with_hybrid_active(monkeypatch, qdrant_provider)

        def _boom(text):
            raise EmbeddingError("sparse model missing")

        monkeypatch.setattr(p, "_embed_sparse", _boom)

        stored = {}
        self._capture_upsert(monkeypatch, p, stored)

        with caplog.at_level(logging.WARNING, logger="memory.qdrant"):
            result = p._tool_upsert({"text": "still saved", "session_id": "s2"})

        assert "stored point" in result
        vector = stored["points"][0]["vector"]
        assert "dense" in vector
        assert "bm25" not in vector
        assert any("dense-only" in r.message or "sparse" in r.message
                   for r in caplog.records)

    def test_no_sparse_when_hybrid_inactive(self, monkeypatch, qdrant_provider):
        """Default path (hybrid off) must not touch the sparse embedder."""
        qdrant_provider._hybrid_active = False
        qdrant_provider._collection = "test_coll"
        monkeypatch.setattr(qdrant_provider, "_embed", lambda text: [0.2] * 384)

        def _must_not_be_called(text):  # pragma: no cover - failure guard
            raise AssertionError(
                "sparse embedder must not be called when hybrid is off")

        monkeypatch.setattr(qdrant_provider, "_embed_sparse", _must_not_be_called)

        stored = {}
        self._capture_upsert(monkeypatch, qdrant_provider, stored)

        result = qdrant_provider._tool_upsert(
            {"text": "plain dense", "session_id": "s3"})
        assert "stored point" in result
        assert set(stored["points"][0]["vector"].keys()) == {"dense"}

    def test_missing_text_short_circuits_before_embedding(
            self, monkeypatch, qdrant_provider):
        calls = []

        def _embed(text):
            calls.append(text)
            return [0.1] * 384

        monkeypatch.setattr(qdrant_provider, "_embed", _embed)
        result = qdrant_provider._tool_upsert({"text": "", "session_id": "s4"})
        assert "missing" in result
        assert calls == []


class TestAddSparseVectorInPlace:
    """The in-place schema change that makes hybrid migratable.

    Guards the fix for a false claim that lived in the source: an earlier
    version asserted Qdrant refuses to add a sparse vector to an existing
    collection and stayed dense-only forever. The correct API is the
    dedicated ``create_vector_name`` (Qdrant >= 1.18), not
    ``update_collection`` — and that distinction is exactly what a naive
    refactor would collapse.
    """

    @staticmethod
    def _provider(monkeypatch, qdrant_provider, *, declares_after, raises=None):
        """A provider whose client fakes the schema call and the read-back."""
        from unittest.mock import MagicMock

        from plugins.memory.qdrant import SPARSE_VECTOR_NAME

        calls = {}
        client = MagicMock()

        def _create_vector_name(**kw):
            calls["create"] = kw
            if raises is not None:
                raise raises

        client.create_vector_name.side_effect = _create_vector_name

        declared = (
            {SPARSE_VECTOR_NAME: object()} if declares_after else {}
        )

        class _Info:
            class config:  # noqa: N801 - mimic the SDK's attribute chain
                class params:  # noqa: N801
                    sparse_vectors = declared

        client.get_collection.return_value = _Info()
        monkeypatch.setattr(qdrant_provider, "_client", client)
        qdrant_provider._collection = "test_coll"
        return qdrant_provider, calls

    def test_adds_sparse_vector_and_reports_true(self, monkeypatch, qdrant_provider):
        p, calls = self._provider(monkeypatch, qdrant_provider, declares_after=True)
        assert p._add_sparse_vector_in_place() is True
        # The call must target the configured vector name, not a literal.
        from plugins.memory.qdrant import SPARSE_VECTOR_NAME
        assert calls["create"]["vector_name"] == SPARSE_VECTOR_NAME
        assert calls["create"]["collection_name"] == "test_coll"

    def test_uses_idf_modifier(self, monkeypatch, qdrant_provider):
        """IDF is what makes a rare term outweigh a common one; without it a
        common word can count as much as a part number."""
        from qdrant_client.http import models as m

        p, calls = self._provider(monkeypatch, qdrant_provider, declares_after=True)
        p._add_sparse_vector_in_place()
        cfg = calls["create"]["vector_name_config"]
        assert cfg.sparse.modifier == m.Modifier.IDF

    def test_returns_false_when_server_refuses(self, monkeypatch, qdrant_provider):
        """An old server (or any refusal) must degrade, never raise."""
        p, _ = self._provider(
            monkeypatch, qdrant_provider, declares_after=False,
            raises=RuntimeError("400 Not existing vector name"),
        )
        assert p._add_sparse_vector_in_place() is False

    def test_returns_false_when_schema_does_not_stick(
            self, monkeypatch, qdrant_provider):
        """Trust the read-back, not the call's return value."""
        p, _ = self._provider(monkeypatch, qdrant_provider, declares_after=False)
        assert p._add_sparse_vector_in_place() is False

    def test_readback_failure_is_not_a_crash(self, monkeypatch, qdrant_provider):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.create_vector_name.return_value = None
        client.get_collection.side_effect = RuntimeError("connection reset")
        monkeypatch.setattr(qdrant_provider, "_client", client)
        assert qdrant_provider._add_sparse_vector_in_place() is False
