"""Tests for the qdrant memory provider's compression-checkpoint hook.

``on_pre_compress`` is the lifecycle hook Hermes fires as the bridge to the
next session, right before it compresses the current transcript. These tests
are fully in-process: a fake backend client, no Qdrant, no embedder, no model
download. They assert the checkpoint is written with an idempotent,
session-derived id, fails closed on backend errors, never raises, and — most
importantly — that checkpoints are excluded from recall by construction.
"""

from plugins.memory.qdrant import (
    QdrantMemoryProvider,
    SPARSE_VECTOR_NAME,
    _checkpoint_clause,
    _recall_filter,
)


class _FakeClient:
    """Records upsert calls; optionally fails to prove fail-closed behaviour."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def upsert(self, collection_name=None, points=None, wait=True, **kw):
        self.calls.append(
            {"collection_name": collection_name, "points": points, "wait": wait}
        )
        if self.fail:
            raise RuntimeError("simulated upsert failure")


def _armed(provider) -> "QdrantMemoryProvider":
    provider._client = _FakeClient()
    provider._initialized = True
    provider._hybrid_active = False
    provider._last_session_id = "sess-xyz"
    return provider


def test_on_pre_compress_writes_idempotent_checkpoint(qdrant_provider):
    p = _armed(qdrant_provider)
    msgs = [{"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}]

    out = p.on_pre_compress(msgs)
    assert out == ""  # contributes no text to the compressor

    assert len(p._client.calls) == 1
    call = p._client.calls[0]
    assert call["collection_name"] == p._collection
    (pt,) = call["points"]
    assert pt["id"] == "ckpt::sess-xyz"            # session-derived -> idempotent
    assert set(pt["vector"]) == {"dense"}          # dense only, zero vector
    assert pt["vector"]["dense"] == [0.0] * p._vector_size
    pl = pt["payload"]
    assert pl["kind"] == "checkpoint"
    assert pl["checkpoint"] is True
    assert pl["session_id"] == "sess-xyz"
    assert pl["turn_count"] == 2
    assert pl["source"] == "on_pre_compress"
    assert "timestamp" in pl and "timestamp_iso" in pl


def test_checkpoint_overwrites_same_session(qdrant_provider):
    p = _armed(qdrant_provider)
    p.on_pre_compress([{"role": "user", "content": "a"}])
    p.on_pre_compress([{"role": "user", "content": "b"}])
    ids = [c["points"][0]["id"] for c in p._client.calls]
    assert ids == ["ckpt::sess-xyz", "ckpt::sess-xyz"]   # same id -> Qdrant overwrite


def test_no_session_skips_write(qdrant_provider):
    p = qdrant_provider
    p._client = _FakeClient()
    p._initialized = True
    p._hybrid_active = False
    p._last_session_id = ""

    out = p.on_pre_compress([{"role": "user", "content": "x"}])
    assert out == ""
    assert p._client.calls == []


def test_on_pre_compress_never_raises(qdrant_provider):
    p = _armed(qdrant_provider)
    p._client = _FakeClient(fail=True)

    # A failed checkpoint must not surface to the caller or break compression.
    out = p.on_pre_compress([{"role": "user", "content": "x"}])
    assert out == ""


def test_hybrid_checkpoint_includes_zero_sparse(qdrant_provider):
    p = _armed(qdrant_provider)
    p._hybrid_active = True
    p.on_pre_compress([{"role": "user", "content": "x"}])
    pt = p._client.calls[0]["points"][0]
    assert SPARSE_VECTOR_NAME in pt["vector"]
    assert pt["vector"][SPARSE_VECTOR_NAME] == {"indices": [], "values": []}


def test_recall_filter_excludes_checkpoint_kind():
    f = _recall_filter({"session_id": "s1"})
    assert _checkpoint_clause() in f["must_not"]


def test_recall_filter_excludes_checkpoint_without_scope():
    f = _recall_filter()
    assert _checkpoint_clause() in f["must_not"]
    assert {"key": "archived", "match": {"value": True}} in f["must_not"]


def test_checkpoint_clause_shape():
    assert _checkpoint_clause() == {"key": "kind", "match": {"value": "checkpoint"}}
