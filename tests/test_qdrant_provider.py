"""Tests for the QdrantMemoryProvider plugin (standalone, kind: exclusive).

Covers provider instantiation, config schema, tool schemas, ABC conformance,
and backend connectivity. The connectivity tests need a real Qdrant on
``http://localhost:6333`` and skip themselves when nothing is listening — see
the ``live_qdrant`` fixture in ``conftest.py``.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.qdrant import EmbeddingError, model_is_present

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_qdrant_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so plugin discovery uses tmp_path."""
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


@pytest.fixture
def qdrant_provider(monkeypatch):
    """A fresh provider built from DEFAULTS — never from operator state.

    The provider's config is ``config.json`` sitting next to the module plus
    the ``memory.qdrant:`` overlay in config.yaml, both read by
    ``_load_plugin_config()``. On a configured install that is somebody else's
    state, not test input: live config.json read ``{"progress": "verbose"}``
    (written by ``hermes memory setup`` on 2026-09-29), which made
    ``test_progress_mode_defaults_to_minimal`` fail here while passing in the
    repo — a red suite nobody could attribute to the code under test.

    ``config={}`` is NOT a way to opt out. ``__init__`` reads
    ``config or _load_plugin_config()``, and the empty dict is falsy *on
    purpose* — that fall-through is what lets a config.yaml-less install pick
    up its own ``config.json`` — so ``QdrantMemoryProvider(config={})`` still
    loads the live file (verified: it returned ``verbose`` on the live install
    with the config.json shown above). The loader is
    stubbed instead. Config-loading behaviour itself is covered by
    ``TestQdrantConfigLoading``, which points the loader at a tmp dir.
    """
    import plugins.memory.qdrant as mod
    monkeypatch.setattr(mod, "_load_plugin_config", lambda: {})
    from plugins.memory.qdrant import QdrantMemoryProvider
    return QdrantMemoryProvider()


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------

class TestQdrantProviderBasics:

    def test_name_is_qdrant(self, qdrant_provider):
        assert qdrant_provider.name == "qdrant"

    def test_is_available_without_qdrant_client(self, qdrant_provider, monkeypatch):
        """is_available returns False when qdrant-client is missing."""
        import plugins.memory.qdrant as _qdrant_mod
        monkeypatch.setattr(_qdrant_mod, "_have_qdrant", lambda: False)
        assert qdrant_provider.is_available() is False

    def test_is_available_without_url(self, qdrant_provider, monkeypatch):
        """is_available returns False when no URL configured."""
        import plugins.memory.qdrant as _qdrant_mod
        monkeypatch.setattr(_qdrant_mod, "_have_qdrant", lambda: True)
        monkeypatch.setattr(qdrant_provider, "_url", None)
        assert qdrant_provider.is_available() is False

    def test_get_config_schema_returns_list(self, qdrant_provider):
        schema = qdrant_provider.get_config_schema()
        assert isinstance(schema, list)
        assert len(schema) > 0
        # The contract's field identifier is "key" (see the contract class below);
        # "name" here would be dropped by both real consumers.
        keys = [f["key"] for f in schema]
        assert "url" in keys
        assert "collection" in keys
        assert "vector_size" in keys
        assert "distance" in keys

    def test_get_tool_schemas_returns_list(self, qdrant_provider):
        schemas = qdrant_provider.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        names = [s["name"] for s in schemas]
        assert "qdrant_search" in names
        assert "qdrant_upsert" in names
        assert "qdrant_recall" in names
        assert "qdrant_collect" in names

    def test_backup_paths_returns_empty_list(self, qdrant_provider):
        assert qdrant_provider.backup_paths() == []

    def test_save_config_writes_json(self, qdrant_provider, tmp_path, monkeypatch):
        """save_config writes values to config.json NEXT TO THE MODULE.

        Not under ``$HERMES_HOME/plugins/memory/qdrant/`` — that path does not
        exist for a user-dir install (``~/.hermes/plugins/<name>/``), so the
        write landed somewhere __init__ never read and the value was lost.
        """
        import plugins.memory.qdrant as mod
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(mod, "_config_json_path", lambda: cfg_path)

        qdrant_provider.save_config({"url": "http://localhost:6333"}, str(tmp_path))
        assert cfg_path.exists()
        assert json.loads(cfg_path.read_text())["url"] == "http://localhost:6333"

    def test_recall_status_returns_recall_status_object(self, qdrant_provider):
        from agent.memory_provider import RecallStatus
        rs = qdrant_provider.recall_status()
        assert isinstance(rs, RecallStatus)
        assert rs.provider_label == "qdrant"
        assert rs.count == 0


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------

class TestQdrantABCConformance:

    REQUIRED_METHODS = [
        "name", "is_available", "initialize", "system_prompt_block",
        "prefetch", "queue_prefetch", "recall_status", "sync_turn",
        "handle_tool_call", "on_session_switch", "on_session_end",
        "on_pre_compress", "shutdown", "get_tool_schemas",
        "get_config_schema", "save_config", "backup_paths",
    ]

    def test_all_abstract_methods_implemented(self, qdrant_provider):
        from agent.memory_provider import MemoryProvider
        for method in self.REQUIRED_METHODS:
            assert hasattr(qdrant_provider, method), f"Missing ABC method: {method}"
            attr = getattr(qdrant_provider, method)
            # name is a property; others are methods
            if method == "name":
                continue
            assert callable(attr), f"Method {method} not callable"

    def test_subclass_of_memory_provider(self):
        from plugins.memory.qdrant import QdrantMemoryProvider
        from agent.memory_provider import MemoryProvider
        assert issubclass(QdrantMemoryProvider, MemoryProvider)


# ---------------------------------------------------------------------------
# initialize() kwarg containment
# ---------------------------------------------------------------------------

@pytest.mark.allow_real_home_io
class TestQdrantInitializeKwargContainment:
    """initialize() must tolerate the scoping kwargs MemoryManager injects.

    Regression: the orchestrator's ``initialize_all`` forwards scoping kwargs
    (platform, hermes_home, agent_context, status_callback, warning_callback,
    session_title, gateway identity) alongside the connection intent. Splatting
    them into ``QdrantClient`` aborted every turn with
    ``Client.__init__() got an unexpected keyword argument 'platform'``, so the
    provider never connected.
    """

    URL = "http://localhost:6333"

    SCOPING_KWARGS = dict(
        platform="cli",
        hermes_home="/nonexistent/scoping-kwarg-fixture",
        agent_context="primary",
        status_callback=object(),
        warning_callback=object(),
        session_title="a title",
        session_title_source="user",
        gateway_session_key="chat-1",
        gateway_user_id="gk",
    )

    def _initialize_capturing_client_kwargs(self, provider, **kwargs):
        """Run initialize() against a stub client; return what reached its ctor."""
        captured = {}

        class _FakeClient:
            def __init__(self, **ctor_kwargs):
                captured.update(ctor_kwargs)

            def get_collections(self):
                return types.SimpleNamespace(collections=[types.SimpleNamespace(name="hermes_memories")])

        provider._url = self.URL
        with patch("qdrant_client.QdrantClient", _FakeClient):
            provider.initialize(session_id="s1", **kwargs)
        return captured

    def test_scoping_kwargs_never_reach_the_client(self, qdrant_provider):
        captured = self._initialize_capturing_client_kwargs(
            qdrant_provider, **self.SCOPING_KWARGS)

        assert not set(captured) & set(self.SCOPING_KWARGS), "scoping kwargs leaked into QdrantClient"
        # The connection intent did survive.
        assert captured == {"url": self.URL, "api_key": None,
                            "prefer_grpc": False, "timeout": 30}

    @pytest.mark.parametrize("extra", [{}, {"prefer_grpc": True}, {"timeout": 5}])
    def test_connection_kwargs_pass_through(self, qdrant_provider, extra):
        """Connection params still reach the client, with and without extras."""
        captured = self._initialize_capturing_client_kwargs(qdrant_provider, **extra)

        assert captured["url"] == self.URL
        assert captured["prefer_grpc"] is extra.get("prefer_grpc", False)
        assert captured["timeout"] == extra.get("timeout", 30)


# ---------------------------------------------------------------------------
# Backend liveness reporting (check_backend / unavailable_reason)
# ---------------------------------------------------------------------------

@pytest.mark.allow_real_home_io
class TestQdrantBackendLiveness:

    """A dead server must be distinguishable from a healthy one.

    ``is_available()`` is contractually config-only (no network call), so
    ``check_backend()`` carries the real probe and ``unavailable_reason()``
    carries the user-facing diagnosis. Regression: an unreachable Qdrant used
    to surface as a raw ``qdrant_client`` traceback out of ``initialize()``.
    """

    URL = "http://localhost:6333"

    def test_check_backend_ok_against_live_server(self, qdrant_provider, live_qdrant):
        qdrant_provider._url = live_qdrant
        ok, message = qdrant_provider.check_backend()
        assert ok is True
        assert live_qdrant in message

    def test_check_backend_reports_unreachable(self, qdrant_provider):
        qdrant_provider._url = "http://127.0.0.1:6399"  # nothing listens here
        ok, message = qdrant_provider.check_backend(timeout=1.0)
        assert ok is False
        assert "cannot reach Qdrant" in message

    def test_unavailable_reason_names_missing_dependency(self, qdrant_provider, monkeypatch):
        import plugins.memory.qdrant as _qdrant_mod
        monkeypatch.setattr(_qdrant_mod, "_have_qdrant", lambda: False)
        assert "not installed" in qdrant_provider.unavailable_reason()

    def test_unavailable_reason_names_missing_url(self, qdrant_provider, monkeypatch):
        import plugins.memory.qdrant as _qdrant_mod
        monkeypatch.setattr(_qdrant_mod, "_have_qdrant", lambda: True)
        qdrant_provider._url = None
        assert "url" in qdrant_provider.unavailable_reason().lower()

    def test_unavailable_reason_is_empty_when_healthy(self, qdrant_provider):
        qdrant_provider._url = self.URL
        assert qdrant_provider.unavailable_reason() == ""

    def test_failed_probe_gates_is_available(self, qdrant_provider):
        """A proven-dead backend must not report a false green."""
        qdrant_provider._url = self.URL
        qdrant_provider._backend_error = "cannot reach Qdrant at ...: refused"
        assert qdrant_provider.is_available() is False
        assert "cannot reach" in qdrant_provider.unavailable_reason()

    def test_successful_initialize_clears_stale_error(self, qdrant_provider):
        """A prior failure must not block recovery on the next real probe."""
        qdrant_provider._url = self.URL
        qdrant_provider._backend_error = "cannot reach Qdrant at ...: refused"

        class _FakeClient:
            def __init__(self, **ctor_kwargs):
                pass

            def get_collections(self):
                return types.SimpleNamespace(
                    collections=[types.SimpleNamespace(name="hermes_memories")])

        with patch("qdrant_client.QdrantClient", _FakeClient):
            qdrant_provider.initialize(session_id="s1")

        assert qdrant_provider._backend_error == ""
        assert qdrant_provider.is_available() is True


# ---------------------------------------------------------------------------
# Config schema contract
# ---------------------------------------------------------------------------

class TestQdrantConfigSchemaContract:

    """Every declared knob must be reachable by the surfaces that drive it.

    Regression: fields were declared with ``name`` instead of the contract's
    ``key``. Both consumers index on ``field["key"]``, so the dashboard
    dropped all six fields silently and ``hermes memory setup`` hard-crashed
    with ``KeyError: 'key'`` on the very first field.
    """

    def test_every_field_declares_a_key(self, qdrant_provider):
        for field in qdrant_provider.get_config_schema():
            assert field.get("key"), f"config field missing 'key': {field}"

    def test_setup_wizard_can_walk_the_schema(self, qdrant_provider):
        """The real wizard helper must not raise on our schema shape."""
        from hermes_cli.memory_setup import _prompt_schema_fields
        # No field is secret/choice-bearing in a way that needs a prompt here;
        # reaching the first interactive read is success, KeyError is failure.
        try:
            _prompt_schema_fields("qdrant", qdrant_provider.get_config_schema(), {}, {})
        except KeyError as e:
            pytest.fail(f"setup wizard raised KeyError {e} on our config schema")
        except (EOFError, OSError):
            pass  # non-interactive stdin — schema was walked fine

    def test_dashboard_normalizer_keeps_every_field(self, qdrant_provider):
        """_normalize_memory_provider_schema silently drops key-less fields."""
        from hermes_cli.web_server_memory import _normalize_memory_provider_schema
        fields = _normalize_memory_provider_schema("qdrant", qdrant_provider)
        assert {f["key"] for f in fields} == {
            "url", "api_key", "collection", "vector_size", "distance",
            "embedder", "model", "device", "progress"}

    def test_secret_field_declares_env_var(self, qdrant_provider):
        """Secrets route to .env through the field's env_var, not a config key."""
        secrets = [f for f in qdrant_provider.get_config_schema()
                   if f.get("type") == "secret" or f.get("secret")]
        assert secrets, "api_key must be declared as a secret"
        for field in secrets:
            assert field.get("env_var"), f"secret field lacks env_var: {field}"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestQdrantConfigLoading:

    """A knob the user (or the setup wizard) changes must actually take effect.

    Regression: ``__init__`` read ``self._config = config or {}`` and nothing
    ever loaded the file that ``save_config()`` writes, so every configured
    value was silently ignored and the provider always used its defaults.
    """

    def _write_config(self, tmp_path, values):
        (tmp_path / "config.json").write_text(json.dumps(values))
        return tmp_path

    def test_config_json_values_reach_the_provider(self, tmp_path, monkeypatch):
        import plugins.memory.qdrant as mod
        cfg = self._write_config(tmp_path, {
            "url": "http://qdrant.example.invalid:6333",
            "collection": "custom_memories",
            "vector_size": 768,
            "distance": "Dot",
        })
        monkeypatch.setattr(mod, "_config_json_path", lambda: cfg / "config.json")
        monkeypatch.setattr(mod, "_load_plugin_config", mod._load_plugin_config)

        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider()
        assert p._url == "http://qdrant.example.invalid:6333"
        assert p._collection == "custom_memories"
        assert p._vector_size == 768
        assert p._distance == "Dot"

    def test_explicit_config_arg_still_wins(self, tmp_path, monkeypatch):
        import plugins.memory.qdrant as mod
        cfg = self._write_config(tmp_path, {"collection": "from_disk"})
        monkeypatch.setattr(mod, "_config_json_path", lambda: cfg / "config.json")

        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider(config={"collection": "explicit"})
        assert p._collection == "explicit"

    def test_malformed_config_falls_back_to_defaults(self, tmp_path, monkeypatch):
        """A corrupt file must degrade to defaults, never raise at construction."""
        import plugins.memory.qdrant as mod
        cfg = tmp_path / "config.json"
        cfg.write_text("{ this is not json ")
        monkeypatch.setattr(mod, "_config_json_path", lambda: cfg)

        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider()
        assert p._collection == "hermes_memories"
        assert p._vector_size == 384

    def test_save_config_writes_next_to_the_module(self, tmp_path, monkeypatch):
        """save_config must round-trip through the path __init__ reads."""
        import plugins.memory.qdrant as mod
        cfg = tmp_path / "config.json"
        monkeypatch.setattr(mod, "_config_json_path", lambda: cfg)

        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider()
        p.save_config({"collection": "roundtrip"}, str(tmp_path))
        assert json.loads(cfg.read_text())["collection"] == "roundtrip"

        p2 = QdrantMemoryProvider()
        assert p2._collection == "roundtrip"

    def test_progress_from_config_json_reaches_the_provider(self, tmp_path,
                                                             monkeypatch):
        """A ``progress`` mode written by the wizard must override the default.

        The inverse of ``test_progress_mode_defaults_to_minimal``: on a live
        install config.json says ``{"progress": "verbose"}`` and the provider
        must honour it. That is correct production behaviour — the bug was the
        default test reading the file, not the file being read.
        """
        import plugins.memory.qdrant as mod
        cfg = self._write_config(tmp_path, {"progress": "verbose"})
        monkeypatch.setattr(mod, "_config_json_path", lambda: cfg / "config.json")

        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider()
        assert p._progress_mode == "verbose"


# ---------------------------------------------------------------------------
# Backend connectivity (live Qdrant on :6333)
# ---------------------------------------------------------------------------

@pytest.mark.allow_real_home_io
class TestQdrantBackendConnectivity:

    def test_backend_connect_and_ensure_collection(self, live_qdrant):
        """Backend can connect to Qdrant and create a collection."""
        from plugins.memory.qdrant._backend import QdrantBackend
        b = QdrantBackend(url=live_qdrant)
        b.connect()
        assert b._client is not None
        b.ensure_collection("test_qdrant_abc", vector_size=384, distance="Cosine")
        cols = b._client.get_collections()
        names = [c.name for c in cols.collections]
        assert "test_qdrant_abc" in names

    def test_backend_upsert_and_search(self, live_qdrant):
        """Backend can upsert points and search them back."""
        import time
        from plugins.memory.qdrant._backend import QdrantBackend
        b = QdrantBackend(url=live_qdrant)
        b.connect()
        b.ensure_collection("test_qdrant_search", vector_size=384, distance="Cosine")
        ts = time.time()
        b.upsert(
            points=[
                {"id": 1, "vector": [0.1] * 384, "payload": {"text": "hello", "session_id": "s1", "ts": ts}},
                {"id": 2, "vector": [0.2] * 384, "payload": {"text": "world", "session_id": "s1", "ts": ts}},
            ],
            collection="test_qdrant_search",
        )
        results = b.search(
            query_vector=[0.1] * 384,
            collection="test_qdrant_search",
            session_id="s1",
            limit=5,
        )
        assert len(results) >= 1
        assert results[0]["score"] == pytest.approx(1.0, abs=0.01)
        payloads = [r["payload"] for r in results]
        texts = [p.get("text", "") for p in payloads]
        assert "hello" in texts or "world" in texts

    def test_backend_scroll(self, live_qdrant):
        """Backend scroll returns all points in a collection."""
        import time
        from plugins.memory.qdrant._backend import QdrantBackend
        b = QdrantBackend(url=live_qdrant)
        b.connect()
        b.ensure_collection("test_qdrant_scroll", vector_size=384, distance="Cosine")
        ts = time.time()
        b.upsert(
            points=[
                {"id": 10, "vector": [0.3] * 384, "payload": {"text": "scroll_test", "session_id": "s2", "ts": ts}},
            ],
            collection="test_qdrant_scroll",
        )
        results = b.scroll(collection="test_qdrant_scroll", session_id="s2", limit=10)
        assert len(results) >= 1
        payloads = [r["payload"] for r in results]
        texts = [p.get("text", "") for p in payloads]
        assert "scroll_test" in texts

    def test_backend_list_collections(self, live_qdrant):
        """Backend list_collections returns collection names."""
        from plugins.memory.qdrant._backend import QdrantBackend
        b = QdrantBackend(url=live_qdrant)
        b.connect()
        cols = b.list_collections()
        assert isinstance(cols, list)

# ---------------------------------------------------------------------------
# Embedder contract
# ---------------------------------------------------------------------------

class TestEmbedderContract:
    """The embedder must never fail silently.

    Regression this class exists to prevent: the old ``_embed()`` caught every
    exception, logged at DEBUG, and returned ``None``. At default log level the
    user saw nothing, the write was skipped, and the session reported memory
    saved when nothing had been encoded. A wrong-but-valid default model is the
    same failure in disguise — it loads fine and quietly writes the wrong space.
    """

    def test_embed_returns_a_vector_not_none(self, qdrant_provider):
        vec = qdrant_provider._embed("hello world")
        assert isinstance(vec, list) and vec
        assert all(isinstance(x, float) for x in vec)

    def test_embed_never_returns_none(self, qdrant_provider):
        """Any failure must raise, so a falsy result can never mean 'failed'."""
        try:
            result = qdrant_provider._embed("probe string for a real vector")
        except EmbeddingError:
            return  # raising is the correct outcome
        assert result is not None
        assert len(result) == qdrant_provider._vector_size

    def test_unknown_backend_raises_config_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from plugins.memory.qdrant import (
            Embedder, EmbeddingConfigError, EmbeddingError,
        )
        with pytest.raises(EmbeddingConfigError):
            Embedder(backend="not-a-real-backend", model="whatever")
        assert issubclass(EmbeddingConfigError, EmbeddingError)

    def test_empty_model_name_raises_rather_than_defaulting(self):
        """A blank model must NOT fall back to a library default.

        fastembed's own default is BAAI/bge-small-en-v1.5, not MiniLM, so a
        blank name here would silently produce a different vector space.
        """
        from plugins.memory.qdrant import Embedder, EmbeddingConfigError
        for blank in ("", "   ", None):
            with pytest.raises(EmbeddingConfigError):
                Embedder(backend="fastembed", model=blank or "")

    def test_model_default_is_pinned_to_minilm(self, qdrant_provider):
        from plugins.memory.qdrant import DEFAULT_MODEL
        assert DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
        assert qdrant_provider._model == DEFAULT_MODEL
        assert qdrant_provider._embedder == "fastembed"

    def test_bad_model_raises_on_encode(self, qdrant_provider):
        """A nonexistent model is deterministic — it must raise, not return None."""
        qdrant_provider._model = "definitely/not-a-real-model-xyz"
        qdrant_provider._embedder_impl = None
        with pytest.raises(EmbeddingError):
            qdrant_provider._embed("some text")

    def test_configured_knobs_reach_the_embedder(self, tmp_path, monkeypatch):
        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider({
            "embedder": "sentence-transformers",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        })
        e = p.get_embedder()
        assert e.backend == "sentence-transformers"
        assert e.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert e.device == "cpu"

    def test_embed_error_surfaces_in_unavailable_reason(self, qdrant_provider):
        """A broken embedder must make /status red even with a healthy server."""
        qdrant_provider._embed_error = "embedder misconfigured: model=X"
        assert "model=X" in qdrant_provider.unavailable_reason()

    def test_schema_exposes_both_backends(self, qdrant_provider):
        fields = {f["key"]: f for f in qdrant_provider.get_config_schema()}
        assert "model" in fields and "device" in fields
        assert fields["embedder"]["choices"] == ["fastembed", "sentence-transformers"]
        assert fields["embedder"]["default"] == "fastembed"

    def test_prepare_tool_is_declared_and_dispatched(self, qdrant_provider):
        names = {t["name"] for t in qdrant_provider.get_tool_schemas()}
        assert "qdrant_prepare" in names
        out = qdrant_provider.handle_tool_call("qdrant_prepare", {})
        assert isinstance(out, str) and out.strip()

    def test_validate_vector_spec_names_both_values_on_mismatch(self, qdrant_provider):
        qdrant_provider._vector_size = 768  # model produces 384
        issues = qdrant_provider.validate_vector_spec()
        joined = " ".join(issues)
        assert issues
        assert "768" in joined and "384" in joined

    def test_validate_vector_spec_flags_non_cosine(self, qdrant_provider):
        qdrant_provider._distance = "Dot"
        joined = " ".join(qdrant_provider.validate_vector_spec())
        assert "Dot" in joined and "Cosine" in joined


# ---------------------------------------------------------------------------
# Embedder parity — the guard against a silent model swap
# ---------------------------------------------------------------------------

@pytest.mark.allow_real_home_io
class TestEmbedderParity:
    """A runtime swap must not change the vector space.

    The two backends are numerically interchangeable for the same checkpoint
    (measured cosine 1.0000000162, max component diff 9.4e-08 on
    all-MiniLM-L6-v2). That is the whole reason no re-ingest is required, so it
    is asserted rather than assumed: a future model-name typo fails here loudly
    instead of degrading recall with no signal anywhere.

    Marked ``allow_real_home_io`` because it reads the model cache the live
    install uses — the session-wide pin in ``conftest.py`` — rather than a
    per-test copy.
    """

    PROBES = [
        "The capital of Austria is Vienna.",
        "Vienna is the capital city of Austria.",
        "Machine learning models learn patterns from data.",
    ]

    def _cos(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    @pytest.fixture(autouse=True)
    def _require_both_backends(self):
        fe = pytest.importorskip("fastembed", reason="fastembed not installed")
        st = pytest.importorskip(
            "sentence_transformers", reason="sentence-transformers not installed"
        )
        self._fe, self._st = fe, st

    def test_backends_agree_on_the_same_model(self):
        from plugins.memory.qdrant import DEFAULT_MODEL, pinned_cache_dir
        # cache_dir is the same pin Embedder._build() uses: without it fastembed
        # falls back to $TMPDIR/fastembed_cache and this test silently pulls a
        # second copy of the weights into prunable scratch on every run.
        fe = self._fe.TextEmbedding(model_name=DEFAULT_MODEL,
                                    cache_dir=str(pinned_cache_dir()))
        st = self._st.SentenceTransformer("all-MiniLM-L6-v2")
        for text in self.PROBES:
            a = next(iter(fe.passage_embed([text]))).tolist()
            b = st.encode(text, normalize_embeddings=True).tolist()
            assert self._cos(a, b) > 0.999, (
                f"backend divergence on {text!r}: cosine={self._cos(a, b):.6f}. "
                "The ONNX and torch paths no longer agree — stored vectors would "
                "be wrong."
            )

    def test_dimensions_match(self):
        from plugins.memory.qdrant import DEFAULT_MODEL, pinned_cache_dir
        fe = self._fe.TextEmbedding(model_name=DEFAULT_MODEL,
                                    cache_dir=str(pinned_cache_dir()))
        st = self._st.SentenceTransformer("all-MiniLM-L6-v2")
        a = next(iter(fe.passage_embed(["dimension probe"]))).tolist()
        b = st.encode("dimension probe", normalize_embeddings=True).tolist()
        assert len(a) == len(b) == 384


@pytest.mark.allow_real_home_io
class TestEmbedderParityLive:
    """The decisive check: a fresh embed must match a REAL stored vector.

    Comparing the two backends to each other only proves they agree; comparing
    a fresh embed to what is already in the collection proves the live data is
    still readable. Skips when no Qdrant is listening.
    """

    PROBE_TEXTS = ["Hey! How can I help you today?"]

    def _cos(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def test_fresh_embed_matches_stored_vectors(self, live_qdrant):
        import requests
        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider({"url": live_qdrant})
        # Find a real point whose text we can recompute.
        resp = requests.post(
            f"{live_qdrant}/collections/hermes_memories/points/scroll",
            json={"limit": 25, "with_payload": True, "with_vector": True},
            timeout=60,
        )
        if resp.status_code != 200:
            pytest.skip("could not scroll the live collection")
        points = resp.json().get("result", {}).get("points", [])
        checked = 0
        for pt in points:
            text = (pt.get("payload") or {}).get("text")
            vec = pt.get("vector")
            if isinstance(vec, dict):
                vec = vec.get("dense")
            if not isinstance(text, str) or not text.strip() or not vec:
                continue
            mine = p._embed(text)
            cos = self._cos(mine, vec)
            assert cos > 0.999, (
                f"fresh embed disagrees with stored vector (cosine={cos:.6f}) "
                f"for id={pt.get('id')} — the collection was written by a "
                "different embedding space."
            )
            checked += 1
            if checked >= 3:
                break
        if not checked:
            pytest.skip("no suitable stored points found to compare")


class TestModelPresence:
    """model_is_present() must answer the question the caller is really asking:
    will this embed WITHOUT a download?

    Regression history, in order. First, qdrant_prepare reported a working
    model as missing while embedding succeeded in 0.04s — fastembed's default
    cache is ``tempfile.gettempdir()/fastembed_cache``, not the XDG
    ``~/.cache/fastembed`` a naive check looked at, and Hermes points TMPDIR at
    its scratch dir. Fixing that by searching EVERY cache root then created the
    opposite lie: a copy stranded in scratch (which is reaped after 24 h idle)
    counted as "present" while the pinned load path was empty, so a
    download-free embed still downloaded 90 MB mid-session.

    So presence now tracks the load path exactly — ``pinned_cache_dir()`` for
    fastembed, the huggingface_hub roots for sentence-transformers — and
    ``model_cache_locations()`` keeps the stale copies visible for reporting.
    """

    def _write_model(self, root, dirname="models--qdrant--all-MiniLM-L6-v2-onnx"):
        d = root / dirname / "snapshots" / "abc123"
        d.mkdir(parents=True)
        (d / "model.onnx").write_bytes(b"stub")
        return root

    def test_pinned_cache_is_the_presence_root(self, tmp_path, monkeypatch):
        """The directory reported as the cache IS the one checked for presence."""
        from plugins.memory.qdrant import embedder as emb

        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
        self._write_model(tmp_path)
        assert model_is_present("sentence-transformers/all-MiniLM-L6-v2")
        assert emb.model_cache_dir() == str(tmp_path)
        assert [str(p) for p in emb._roots_for(emb.BACKEND_FASTEMBED)] == [str(tmp_path)]

    def test_stale_tmpdir_copy_is_reported_but_not_present(self, tmp_path, monkeypatch):
        """A copy in a prunable TMPDIR is NOT "present" — but must still be named.

        tempfile.gettempdir() memoises its result on first call, so setting
        TMPDIR after import has no effect — patch the function itself.
        """
        from plugins.memory.qdrant import embedder as emb

        pinned = tmp_path / "pinned"          # the load path, deliberately empty
        pinned.mkdir()
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(pinned))
        stale = tmp_path / "fastembed_cache"  # what fastembed would have used
        self._write_model(stale)
        monkeypatch.setattr(emb.tempfile, "gettempdir", lambda: str(tmp_path))

        assert str(stale) in [str(p) for p in emb._cache_roots()]
        # Lying the other way is what made prepare promise a download-free
        # embed that then downloaded: the load path (pinned) has no model.
        assert not model_is_present("sentence-transformers/all-MiniLM-L6-v2")
        locs = [str(p) for p in emb.model_cache_locations(
            "sentence-transformers/all-MiniLM-L6-v2")]
        assert str(stale) in locs, locs

    def test_pinned_cache_is_one_path_for_every_home(self, tmp_path, monkeypatch):
        """Profile home and base home must pin the SAME directory, outside scratch.

        Regression: fastembed defaulted to $TMPDIR/fastembed_cache, each home has
        its own TMPDIR, and scratch is reaped after SCRATCH_MAX_IDLE_HOURS —
        three simultaneous copies of the 90 MB weights existed on 2026-09-29.
        """
        from plugins.memory.qdrant import embedder as emb

        monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "alice"))
        profile_pin = emb.pinned_cache_dir()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        base_pin = emb.pinned_cache_dir()

        expected = tmp_path / "state" / "qdrant" / "model_cache"
        assert profile_pin == base_pin == expected
        # Within the home itself the pin lives in state/, never in the cache
        # tree that prune_scratch_dir() reaps.
        assert not str(profile_pin).startswith(str(tmp_path / "cache/"))

    def test_fastembed_is_constructed_with_the_pinned_cache_dir(
            self, tmp_path, monkeypatch):
        """The fix itself: fastembed must be TOLD where to load from.

        Without ``cache_dir`` fastembed falls back to $TMPDIR/fastembed_cache
        (prunable) while model_is_present() consults the pinned dir — the two
        disagree, which is the "present but not loadable" mismatch.
        """
        from plugins.memory.qdrant import embedder as emb

        pin = tmp_path / "pin"
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(pin))
        calls: list[dict] = []

        class _RecordingTextEmbedding:
            dim = 384

            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake = types.ModuleType("fastembed")
        fake.TextEmbedding = _RecordingTextEmbedding
        monkeypatch.setitem(sys.modules, "fastembed", fake)

        emb.Embedder(backend=emb.BACKEND_FASTEMBED, model=emb.DEFAULT_MODEL)._build()

        assert calls, "TextEmbedding was never constructed"
        assert calls[-1]["cache_dir"] == str(pin)
        assert calls[-1]["model_name"] == emb.DEFAULT_MODEL
        assert emb.model_cache_dir(emb.BACKEND_FASTEMBED) == str(pin)

    def test_finds_model_under_explicit_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
        self._write_model(tmp_path)
        assert model_is_present("sentence-transformers/all-MiniLM-L6-v2")

    def test_matches_repo_style_dir_name(self, tmp_path, monkeypatch):
        """fastembed stores qdrant/all-MiniLM-L6-v2-onnx, not the HF name.

        The directory does not literally contain the configured model name, so
        a naive full-string comparison would miss it.
        """
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
        self._write_model(tmp_path)
        assert model_is_present("sentence-transformers/all-MiniLM-L6-v2")

    def test_absent_model_is_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
        tmp_path.mkdir(parents=True, exist_ok=True)
        assert not model_is_present("not/a-real-model-999")

    def test_empty_model_is_false(self):
        assert not model_is_present("")

    def test_missing_dir_does_not_raise(self, tmp_path, monkeypatch):
        from plugins.memory.qdrant import embedder as emb

        # _roots_for() is what model_is_present() walks for the fastembed
        # backend — patch that, or the test would pass without exercising
        # a missing root at all.
        monkeypatch.setattr(emb, "_roots_for", lambda backend: [
            tmp_path / "nope", tmp_path / "also-nope" / "fastembed_cache"])
        assert not model_is_present("sentence-transformers/all-MiniLM-L6-v2")

    def test_unreadable_root_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        from plugins.memory.qdrant import embedder as emb

        # A root that exists but is a FILE, not a directory: iterdir() raises
        # NotADirectoryError. It must be swallowed, not propagate.
        blocker = tmp_path / "blocked"
        blocker.write_text("not a dir")
        monkeypatch.setattr(emb, "_roots_for", lambda backend: [blocker])
        assert not model_is_present("sentence-transformers/all-MiniLM-L6-v2")

    def test_live_model_is_reported_present(self, qdrant_provider, monkeypatch):
        """The model the live stack actually has must not be reported missing.

        The autouse fixture points HERMES_HOME at a tmp dir so plugin discovery
        is isolated; the real weights live under the real home's pinned cache,
        so this one live check drops the override.
        """
        monkeypatch.setenv("HERMES_HOME", str(Path.home() / ".hermes"))
        out = qdrant_provider.handle_tool_call("qdrant_prepare", {})
        assert "not found" not in out.lower(), (
            f"qdrant_prepare claims the model is absent but it is cached: {out}"
        )
        assert "state/qdrant/model_cache" in out, (
            f"prepare did not report the pinned cache path: {out}"
        )


class TestStoredVectorParity:
    """A fresh embedding must match vectors ALREADY IN the collection.

    TestEmbedderParity proves the two backends agree with each other. That is
    necessary but not sufficient: what actually matters on this host is whether
    the 127k vectors written by the old torch path are still *correct* under the
    new fastembed backend. Only a comparison against a stored vector can show
    that, and it is the claim the README makes when it says no re-embedding is
    needed.

    The evidence so far lived in a one-off script outside the repo, which means
    CI could not protect it. This is that check, in the suite.

    Uses a dedicated throwaway collection so a run never touches
    hermes_memories, and writes vectors through the SENTENCE-TRANSFORMERS path
    (torch) so the comparison is genuinely old-path vs new-path rather than
    fastembed compared with itself.
    """

    COLLECTION = "hermes_parity_probe"
    PROBES = [
        "The capital of Austria is Vienna.",
        "Retrieval augmented generation grounds answers in a vector store.",
    ]

    def _cos(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def _client(self, url):
        from qdrant_client import QdrantClient
        return QdrantClient(url=url, prefer_grpc=False, timeout=10)

    def test_new_backend_reproduces_stored_vectors(self, live_qdrant):
        """fastembed(text) must match the stored torch-era vector for that text."""
        import uuid

        import pytest as _pytest
        st = _pytest.importorskip(
            "sentence_transformers", reason="needs torch to write the old-path vector")
        from qdrant_client import models

        from plugins.memory.qdrant import DEFAULT_MODEL

        client = self._client(live_qdrant)
        try:
            if client.collection_exists(self.COLLECTION):
                client.delete_collection(self.COLLECTION)
            client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=models.VectorParams(
                    size=384, distance=models.Distance.COSINE),
            )

            # 1. Write with the OLD path (torch / sentence-transformers).
            old = st.SentenceTransformer("all-MiniLM-L6-v2")
            stored = {}
            for i, text in enumerate(self.PROBES):
                vec = old.encode(text, normalize_embeddings=True).tolist()
                stored[text] = vec
                client.upsert(
                    collection_name=self.COLLECTION,
                    points=[models.PointStruct(
                        id=str(uuid.uuid4()), vector=vec,
                        payload={"text": text, "idx": i})],
                )

            # 2. Re-embed each probe with the NEW default backend and compare.
            from plugins.memory.qdrant import QdrantMemoryProvider
            p = QdrantMemoryProvider.__new__(QdrantMemoryProvider)
            p._embedder = "fastembed"
            p._model = DEFAULT_MODEL
            p._device = "auto"
            p._embedder_impl = None

            for text in self.PROBES:
                fresh = p._embed(text)
                cos = self._cos(fresh, stored[text])
                assert cos > 0.999, (
                    f"new backend diverges from a STORED vector for {text!r}: "
                    f"cosine={cos:.6f}. Existing points are in a different vector "
                    "space — the collection must be re-embedded."
                )

            # 3. The reverse direction matters too: can the new backend RETRIEVE
            # the old vector by text? Cosine alone proves the maths; a real
            # query proves the runtime path the provider uses.
            hits = client.query_points(
                collection_name=self.COLLECTION,
                query=p._embed(self.PROBES[0]),
                using="",
                limit=len(self.PROBES),
            ).points
            texts = [(h.payload or {}).get("text") for h in hits]
            assert self.PROBES[0] == texts[0], (
                f"top hit for its own text was {texts[0]!r}, not {self.PROBES[0]!r} "
                f"— the stored vectors are not retrievable under this backend"
            )
        finally:
            try:
                if client.collection_exists(self.COLLECTION):
                    client.delete_collection(self.COLLECTION)
            except Exception:
                pass
            client.close()

    def test_probe_collection_is_cleaned_up(self, live_qdrant):
        """The parity probe must not leave a collection behind."""
        client = self._client(live_qdrant)
        try:
            assert not client.collection_exists(self.COLLECTION), (
                f"{self.COLLECTION} survived a test run — the test is leaking a "
                f"collection into the live server"
            )
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

class TestProgressDisplay:
    """Tests for the progress display feature (status_callback emission)."""

    def test_progress_mode_defaults_to_minimal(self, qdrant_provider):
        """Contract: no ``progress`` key configured => mode ``"minimal"``.

        The fixture supplies no config at all, so this really is the default
        branch — it must not depend on whether the machine running the suite
        has a config.json (see the fixture's docstring).
        """
        assert qdrant_provider._progress_mode == "minimal"

    def test_progress_mode_can_be_set_to_off(self, tmp_path, monkeypatch):
        """Progress mode 'off' disables all progress events."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider(config={"progress": "off"})
        assert p._progress_mode == "off"

    def test_progress_mode_can_be_set_to_verbose(self, tmp_path, monkeypatch):
        """Progress mode 'verbose' enables all progress events."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        from plugins.memory.qdrant import QdrantMemoryProvider
        p = QdrantMemoryProvider(config={"progress": "verbose"})
        assert p._progress_mode == "verbose"

    def test_emit_progress_respects_off_mode(self, qdrant_provider):
        """No events emitted when mode is 'off'."""
        qdrant_provider._progress_mode = "off"
        qdrant_provider._status_callback = MagicMock()
        qdrant_provider._emit_progress("memory_sync", "test")
        qdrant_provider._status_callback.assert_not_called()

    def test_emit_progress_respects_minimal_mode(self, qdrant_provider):
        """Only completion events emitted when mode is 'minimal'."""
        qdrant_provider._progress_mode = "minimal"
        qdrant_provider._status_callback = MagicMock()
        # Verbose events are suppressed in minimal mode
        qdrant_provider._emit_progress("memory_sync", "test", verbose=True)
        qdrant_provider._status_callback.assert_not_called()
        # Completion events are emitted
        qdrant_provider._emit_progress("memory_sync", "test")
        qdrant_provider._status_callback.assert_called_once()

    def test_emit_progress_respects_verbose_mode(self, qdrant_provider):
        """All events emitted when mode is 'verbose'."""
        qdrant_provider._progress_mode = "verbose"
        qdrant_provider._status_callback = MagicMock()
        qdrant_provider._emit_progress("memory_sync", "test", verbose=True)
        qdrant_provider._status_callback.assert_called_once()

    def test_emit_progress_no_callback_is_safe(self, qdrant_provider):
        """No error when status_callback is None."""
        qdrant_provider._status_callback = None
        qdrant_provider._emit_progress("memory_sync", "test")  # must not raise

    def test_emit_progress_callback_exception_is_swallowed(self, qdrant_provider):
        """A failing status_callback must not break memory operations."""
        qdrant_provider._status_callback = MagicMock(side_effect=RuntimeError("boom"))
        qdrant_provider._emit_progress("memory_sync", "test")  # must not raise

    def test_initialize_stores_status_callback(self, qdrant_provider, monkeypatch):
        """initialize() stores status_callback from kwargs."""
        callback = MagicMock()
        qdrant_provider._status_callback = None
        # Mock the QdrantClient to avoid needing a real server
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: mock_client)
        qdrant_provider.initialize("test-session", status_callback=callback)
        assert qdrant_provider._status_callback is callback

    def test_recall_status_returns_last_count(self, qdrant_provider):
        """recall_status returns the actual recall count."""
        qdrant_provider._last_recall_count = 5
        status = qdrant_provider.recall_status()
        assert status.count == 5

    def test_recall_status_default_count_is_zero(self, qdrant_provider):
        """recall_status returns 0 when no prefetch has run."""
        qdrant_provider._last_recall_count = 0
        status = qdrant_provider.recall_status()
        assert status.count == 0

    def test_progress_config_field_in_schema(self, qdrant_provider):
        """The progress config field is in the schema."""
        schema = qdrant_provider.get_config_schema()
        keys = [f["key"] for f in schema]
        assert "progress" in keys

    def test_progress_config_field_has_correct_choices(self, qdrant_provider):
        """The progress config field has off/minimal/verbose choices."""
        schema = qdrant_provider.get_config_schema()
        progress_field = next(f for f in schema if f["key"] == "progress")
        assert progress_field["choices"] == ["off", "minimal", "verbose"]
        assert progress_field["default"] == "minimal"
