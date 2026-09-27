"""Tests for the QdrantMemoryProvider plugin (standalone, kind: exclusive).

Covers provider instantiation, config schema, tool schemas, ABC conformance,
and backend connectivity. The connectivity tests need a real Qdrant on
``http://localhost:6333`` and skip themselves when nothing is listening — see
the ``live_qdrant`` fixture in ``conftest.py``.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

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
def qdrant_provider():
    """Return a fresh QdrantMemoryProvider instance."""
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
            "url", "api_key", "collection", "vector_size", "distance", "embedder"}

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