"""Qdrant memory provider plugin for Hermes.

Self-hostable vector-memory backend using Qdrant (REST). No cloud dependency
required. Registered by the Hermes memory-provider loader via
``ctx.register_memory_provider(QdrantMemoryProvider())``.

What actually works today (see _backend.py for the unwired extras):

  - Dense semantic search over a single named "dense" vector
  - Session-scoped recall via a ``session_id`` payload filter
  - Verbatim turn storage (sync_turn) with payload metadata
  - A circuit breaker (5 failures -> 120s cooldown) around the I/O paths
  - Honest availability reporting: is_available() / check_backend() /
    unavailable_reason() distinguish config errors from a dead server

What exists in ``_backend.py`` but is NOT on the live path — do not advertise
it until it is wired into the provider: hybrid dense+sparse RRF search and
INT8 scalar quantization.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus

# ``embedder`` is a sibling module, but this package is loaded two ways: as
# ``plugins.memory.qdrant`` (normal, relative import works) and as a bare
# top-level ``__init__`` (how the Hermes plugin loader and these tests import
# it), where a relative import has no parent package and raises ImportError.
# Try the relative form first, then fall back to loading the sibling by path.
try:  # pragma: no cover - branch depends on how the package was imported
    from .embedder import (
        BACKEND_FASTEMBED,
        BACKEND_ST,
        BACKENDS,
        DEFAULT_MODEL,
        Embedder,
        EmbeddingConfigError,
        EmbeddingError,
        EmbeddingRuntimeError,
        model_cache_dir,
        model_is_present,
    )
except ImportError:  # pragma: no cover
    import importlib.util as _ilu
    import os as _os

    _spec = _ilu.spec_from_file_location(
        "hermes_qdrant_embedder",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "embedder.py"),
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover
        raise ImportError("cannot locate sibling embedder.py")
    _embedder_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_embedder_mod)
    BACKEND_FASTEMBED = _embedder_mod.BACKEND_FASTEMBED
    BACKEND_ST = _embedder_mod.BACKEND_ST
    BACKENDS = _embedder_mod.BACKENDS
    DEFAULT_MODEL = _embedder_mod.DEFAULT_MODEL
    Embedder = _embedder_mod.Embedder
    EmbeddingConfigError = _embedder_mod.EmbeddingConfigError
    EmbeddingError = _embedder_mod.EmbeddingError
    EmbeddingRuntimeError = _embedder_mod.EmbeddingRuntimeError
    model_cache_dir = _embedder_mod.model_cache_dir
    model_is_present = _embedder_mod.model_is_present

# Re-exported for callers and tests.
__all_embedder_api__ = (
    "BACKEND_FASTEMBED", "BACKEND_ST", "BACKENDS", "DEFAULT_MODEL",
    "Embedder", "EmbeddingConfigError", "EmbeddingError",
    "EmbeddingRuntimeError", "model_cache_dir", "model_is_present",
)

logger = logging.getLogger("hermes.plugins.memory.qdrant")

# ---------------------------------------------------------------------------
# QdrantClient lazy import — graceful degradation if qdrant-client missing
# ---------------------------------------------------------------------------

def _have_qdrant() -> bool:
    try:
        import qdrant_client  # noqa: F401
        return True
    except ImportError:
        return False


def _models():
    """Return qdrant_client.models, or None if qdrant-client is missing."""
    try:
        from qdrant_client import models
        return models
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

PLUGIN_NAME = "qdrant"
PLUGIN_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _config_json_path() -> Any:
    """``config.json`` sitting next to THIS file — wherever the plugin is installed.

    The install location varies (bundled tree, ``~/.hermes/plugins/<name>/``, a
    pip entry point), so the path is derived from ``__file__`` rather than
    assumed. It stays a dict on disk rather than a config.yaml read because
    ``hermes memory setup`` and the dashboard both route through
    ``MemoryProvider.save_config()``, which is contractually file-based.
    """
    from pathlib import Path as _Path
    return _Path(__file__).resolve().parent / "config.json"


def _load_plugin_config() -> dict:
    """Read saved provider config: env < config.yaml's ``memory.qdrant`` < config.json.

    Precedence, lowest to highest:

    1. the built-in defaults in ``__init__``
    2. ``QDRANT_URL`` / ``QDRANT_API_KEY`` from the environment (cloud deploys
       and the dashboard's secret writer, which routes secrets to ``.env``)
    3. ``memory.qdrant:`` in config.yaml, for homes that set knobs directly
    4. ``config.json`` next to this module, which is what ``save_config()``
       writes via ``hermes memory setup`` and the dashboard

    A malformed or missing file yields no override rather than an error, so a
    bad hand-edit can never take memory offline.
    """
    import json as _json

    merged: dict = {}

    try:
        from hermes_cli.config import load_config_readonly  # managed overlay + ${VAR} expansion
        raw = (load_config_readonly().get("memory") or {}).get(PLUGIN_NAME)
        if isinstance(raw, dict):
            merged.update(raw)
    except Exception:
        pass

    try:
        cfg_path = _config_json_path()
        if cfg_path.exists():
            disk = _json.loads(cfg_path.read_text())
            if isinstance(disk, dict):
                merged.update(disk)
    except Exception as e:
        logger.warning("Qdrant config.json unreadable, using config.yaml/env only: %s", e)

    # Secrets live in the env, not the config file — read them last so a
    # rotated key takes effect without touching config.json. ``get_secret``
    # raises UnscopedSecretError under a multiplex gateway with no bound
    # scope, and a config read must never take memory down: skip the env
    # override rather than propagate.
    try:
        from agent.secret_scope import get_secret
        for env_key, cfg_key in (("QDRANT_URL", "url"), ("QDRANT_API_KEY", "api_key")):
            value = get_secret(env_key, default="")
            if value:
                merged[cfg_key] = value
    except Exception as e:
        logger.debug("Qdrant env override unavailable (no secret scope): %s", e)

    return merged


# ---------------------------------------------------------------------------
# QdrantMemoryProvider
# ---------------------------------------------------------------------------

class QdrantMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by a Qdrant vector database.

    Dense semantic search over a named "dense" vector, session-scoped filtering
    through a ``session_id`` payload index, and verbatim turn storage.
    """

    name = PLUGIN_NAME

    # -- circuit breaker -----------------------------------------------------
    _BREAKER_THRESHOLD = 5
    _BREAKER_COOLDOWN_SECS = 120
    _PREFETCH_WAIT_SECS = 3

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or _load_plugin_config()
        self._client: Any = None
        self._collection = self._config.get("collection", "hermes_memories")
        self._vector_size = self._config.get("vector_size", 384)
        self._distance = self._config.get("distance", "Cosine")
        self._url = self._config.get("url", "http://localhost:6333")
        self._api_key = self._config.get("api_key", "")
        self._embedder = self._config.get("embedder") or BACKEND_FASTEMBED
        self._model = self._config.get("model") or DEFAULT_MODEL
        self._device = self._config.get("device") or "auto"
        self._embedder_impl: Embedder | None = None
        # Last embedder failure, surfaced through unavailable_reason() so a
        # dead embedder shows red in /status even when no tool has been called
        # since it broke.
        self._embed_error: str = ""
        self._breaker_open_until: float = 0.0
        self._breaker_failures = 0
        self._initialized = False
        self._prefetch_thread: Any = None
        # Last real backend probe result ("" = unknown/ok, else a message).
        # Only set by check_backend()/initialize(); never guessed.
        self._backend_error: str = ""

    # -- Lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        """Config/deps only, per the MemoryProvider contract (no network call).

        A previously completed backend probe is honoured if it FAILED, so a
        provider that was proven unreachable does not report a false green.
        A successful probe is deliberately not cached as "available": this
        method must not depend on a transient network state, and initialize()
        re-probes every start.
        """
        if not _have_qdrant():
            logger.debug("qdrant-client not installed")
            return False
        if not self._url:
            return False
        # Only a *negative* cached result gates availability.
        if self._backend_error:
            return False
        return True

    def check_backend(self, timeout: float = 3.0) -> tuple[bool, str]:
        """Actually probe the Qdrant server. Returns (ok, message).

        This is the real liveness check: is_available() is contractually
        config-only, so without this a dead Qdrant is indistinguishable from a
        healthy one until the first failed tool call.
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            return False, f"qdrant-client not importable: {e}"
        if not self._url:
            return False, "no url configured"
        client = None
        try:
            client = QdrantClient(url=self._url, api_key=self._api_key or None,
                                  prefer_grpc=False, timeout=timeout)
            client.get_collections()
            return True, f"connected to {self._url}"
        except Exception as e:
            return False, f"cannot reach Qdrant at {self._url}: {type(e).__name__}: {e}"
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def unavailable_reason(self) -> str:
        """User-facing hint shown when the provider is gated off."""
        if not _have_qdrant():
            return "qdrant-client is not installed in this environment"
        if not self._url:
            return "no Qdrant url configured (set memory.qdrant.url)"
        if self._embed_error:
            # An embedder failure must be visible even when the server is fine:
            # a broken embedder leaves is_available() green while every write
            # silently no-ops.
            return self._embed_error
        if self._backend_error:
            return self._backend_error
        return ""

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Connect to Qdrant and ensure the collection exists."""
        # A previous failed probe must not block recovery: re-probe for real
        # rather than trusting the cached negative from an earlier run.
        self._backend_error = ""
        if not _have_qdrant():
            self._backend_error = "qdrant-client is not installed in this environment"
            raise RuntimeError("Qdrant provider not available — qdrant-client missing")
        if not self._url:
            self._backend_error = "no Qdrant url configured (set memory.qdrant.url)"
            raise RuntimeError("Qdrant provider not available — no URL configured")

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # Only connection params reach QdrantClient. MemoryManager.initialize_all passes
        # scoping kwargs too (platform, hermes_home, agent_context, status_callback,
        # warning_callback, session_title, ...); splatting those in makes the client
        # constructor raise "unexpected keyword argument 'platform'" and the provider
        # never initializes.
        self._client = QdrantClient(
            url=self._url,
            api_key=self._api_key or None,
            prefer_grpc=kwargs.get("prefer_grpc", False),
            timeout=kwargs.get("timeout", 30),
        )

        # Create collection if it doesn't exist.
        # The collection MUST declare a NAMED vector "dense" because sync_turn /
        # add_memory / search all upsert and query with vector={"dense": [...]} and
        # using="dense". An unnamed default vector (plain VectorParams) makes Qdrant
        # reject those with 400 "Not existing vector name error: dense".
        try:
            collections = {c.name for c in self._client.get_collections().collections}
        except Exception as e:
            # A dead/unreachable server must produce a diagnosable error, not a
            # raw qdrant_client traceback (which is what this used to do).
            self._backend_error = f"cannot reach Qdrant at {self._url}: {type(e).__name__}: {e}"
            self._initialized = False
            self._client = None
            logger.error("QdrantMemoryProvider initialize failed: %s", self._backend_error)
            raise RuntimeError(self._backend_error) from e
        # Verified reachable: clear any stale negative so a restart can recover.
        self._backend_error = ""
        if self._collection not in collections:
            logger.info("Creating collection %s (dims=%d, distance=%s, named vector 'dense')",
                        self._collection, self._vector_size, self._distance)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self._vector_size,
                        distance=Distance[self._distance.upper()],
                    )
                },
            )
            # Payload indexes for session scoping
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name="session_id",
                    field_schema="keyword",
                )
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name="source",
                    field_schema="keyword",
                )
            except Exception as e:
                logger.debug("Payload index creation: %s", e)

        self._initialized = True
        logger.info("QdrantMemoryProvider initialized (url=%s, collection=%s)",
                    self._url, self._collection)

    def system_prompt_block(self) -> str:
        """Static info about Qdrant status for the system prompt."""
        status = "ready" if self._initialized else "not initialized"
        return (
            f"[Qdrant memory: {status}] "
            f"Collection: {self._collection} | "
            f"URL: {self._url} | "
            f"Vector size: {self._vector_size}"
        )

    # -- Prefetch / Recall ---------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "", **kwargs: Any) -> str:
        """Semantic/hybrid search for memories relevant to *query*."""
        if not self._initialized or not self._client:
            return ""

        if self._is_breaker_open():
            logger.debug("Circuit breaker open — skipping prefetch")
            return ""

        try:
            # Encode query via the configured embedder. A failure raises, and
            # is logged below — it is never swallowed into a silent "".
            dense_vec = self._embed(query)

            # Dense semantic search via named "dense" vector.
            # NOTE 1: Prefetch must be the real qdrant_client.models.Prefetch —
            # building it with type(...) produces a plain object that pydantic
            # rejects ("Input should be a valid dictionary or instance of Prefetch").
            # NOTE 2: Qdrant >=1.19 rejects prefetch=... with query=None
            # ("A query is needed to merge the prefetches"). A single dense
            # query is expressed by passing query= directly, not via prefetch.
            results = self._client.query_points(
                collection_name=self._collection,
                query=dense_vec,
                using="dense",
                query_filter={
                    "must": [{"key": "session_id", "match": {"value": session_id}}]
                } if session_id else None,
                limit=10,
            )

            hits = results.points if hasattr(results, "points") else []
            lines = []
            for pt in hits:
                payload = pt.payload if hasattr(pt, "payload") else {}
                score = pt.score if hasattr(pt, "score") else 0.0
                text = payload.get("text", "")
                if text:
                    lines.append(f"- [{score:.2f}] {text}")

            return "\n".join(lines) if lines else ""

        except Exception as e:
            self._record_failure()
            logger.warning("Qdrant prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "", **kwargs: Any) -> None:
        """Background prefetch — spawn a thread (simplified; real impl uses spawn_context_thread)."""
        import threading

        def _run():
            try:
                self.prefetch(query, session_id=session_id, **kwargs)
            except Exception as e:
                logger.debug("Background prefetch error: %s", e)

        t = threading.Thread(target=_run, daemon=True, name="qdrant-prefetch")
        t.start()
        self._prefetch_thread = t

    def recall_status(self) -> RecallStatus:
        """Return current recall state."""
        return RecallStatus(
            provider_label=self.name,
            count=0,
            glyph="📖",
        )

    # -- Sync / Store --------------------------------------------------------

    def sync_turn(
        self,
        user: str,
        assistant: str,
        *,
        session_id: str = "",
        messages: list | None = None,
        turn_author: str = "",
        **kwargs: Any,
    ) -> None:
        """Store the turn's content as a memory point in Qdrant."""
        if not self._initialized or not self._client:
            return

        if self._is_breaker_open():
            logger.debug("Circuit breaker open — skipping sync_turn")
            return

        try:
            # Combine user + assistant for embedding
            combined = f"user: {user}\nassistant: {assistant}"
            dense_vec = self._embed(combined)

            point_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            self._client.upsert(
                collection_name=self._collection,
                points=[
                    {
                        "id": point_id,
                        "vector": {"dense": dense_vec},
                        "payload": {
                            "text": combined[:2000],  # Truncate for storage
                            "session_id": session_id,
                            "timestamp": timestamp,
                            "source": "sync_turn",
                            "turn_author": turn_author,
                        },
                    }
                ],
                wait=True,
            )
            self._record_success()

        except Exception as e:
            self._record_failure()
            logger.warning("Qdrant sync_turn failed: %s", e)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs: Any) -> str:
        """Dispatch a Qdrant tool call."""
        if tool_name == "qdrant_search":
            return self._tool_search(args)
        if tool_name == "qdrant_upsert":
            return self._tool_upsert(args)
        if tool_name == "qdrant_recall":
            return self._tool_recall(args)
        if tool_name == "qdrant_collect":
            return self._tool_collect(args)
        if tool_name == "qdrant_prepare":
            return self._tool_prepare(args)
        return f"Unknown qdrant tool: {tool_name}"

    # -- Tool implementations ------------------------------------------------

    def _tool_search(self, args: dict) -> str:
        """Semantic/hybrid search with optional filters."""
        query = args.get("query", "")
        session_id = args.get("session_id", "")
        limit = int(args.get("limit", 10))

        if not query:
            return "qdrant_search: missing 'query' parameter"

        try:
            dense_vec = self._embed(query)
        except EmbeddingError as e:
            # Explicit, actionable failure. Never "embedding failed" with no
            # detail — that is the silent-degradation trap this replaced.
            return f"qdrant_search: {e}"

        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=dense_vec,
                using="dense",
                query_filter={
                    "must": [{"key": "session_id", "match": {"value": session_id}}]
                } if session_id else None,
                limit=limit,
            )
            hits = results.points if hasattr(results, "points") else []
            lines = [f"[{pt.score:.2f}] {pt.payload.get('text', '')}"
                     for pt in hits if hasattr(pt, "payload")]
            return "\n".join(lines) if lines else "No results"
        except Exception as e:
            return f"qdrant_search error: {e}"

    def _tool_upsert(self, args: dict) -> str:
        """Store a memory point (text + metadata)."""
        text = args.get("text", "")
        session_id = args.get("session_id", "")
        if not text:
            return "qdrant_upsert: missing 'text' parameter"

        try:
            dense_vec = self._embed(text)
        except EmbeddingError as e:
            # A failed write must never look like a successful save.
            return f"qdrant_upsert: {e}"

        try:
            point_id = str(uuid.uuid4())
            self._client.upsert(
                collection_name=self._collection,
                points=[{
                    "id": point_id,
                    "vector": {"dense": dense_vec},
                    "payload": {
                        "text": text[:2000],
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "tool_upsert",
                    },
                }],
                wait=True,
            )
            return f"qdrant_upsert: stored point {point_id}"
        except Exception as e:
            return f"qdrant_upsert error: {e}"

    def _tool_recall(self, args: dict) -> str:
        """Scroll/filtered bulk recall for a session."""
        session_id = args.get("session_id", "")
        limit = int(args.get("limit", 100))
        if not session_id:
            return "qdrant_recall: missing 'session_id' parameter"

        try:
            results, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter={
                    "must": [{"key": "session_id", "match": {"value": session_id}}]
                },
                limit=limit,
                with_payload=True,
            )
            lines = [f"[{pt.payload.get('score', 0):.2f}] {pt.payload.get('text', '')}"
                     for pt in results if hasattr(pt, "payload")]
            return "\n".join(lines) if lines else "No recalled memories"
        except Exception as e:
            return f"qdrant_recall error: {e}"

    def _tool_collect(self, args: dict) -> str:
        """Collection inspection: list / info. Read-only by design.

        The schema used to advertise a 'delete' action that no branch
        implemented, so the model could call it and get "Unknown action".
        Deliberately no destructive action: this is a tool the agent can call
        unprompted, and dropping the user's only memory corpus should be a
        human decision.
        """
        action = args.get("action", "list")
        try:
            if action == "list":
                cols = [c.name for c in self._client.get_collections().collections]
                return f"Collections: {', '.join(cols)}"
            if action == "info":
                name = args.get("collection") or self._collection
                col = self._client.get_collection(name)
                return f"Collection {name}: {col.points_count} points, {col.config}"
            return f"Unknown action: {action} (supported: list, info)"
        except Exception as e:
            return f"qdrant_collect error: {e}"

    def _tool_prepare(self, args: dict) -> str:
        """Report embedder readiness, optionally downloading the model.

        Returns a human-readable report. Failures are stated plainly rather
        than folded into a generic error, because the whole point of this tool
        is to tell the user what is actually wrong.
        """
        download = bool(args.get("download", False))
        try:
            return self.prepare_report(download=download)
        except EmbeddingError as e:
            return f"qdrant_prepare: {e}"
        except Exception as e:
            return f"qdrant_prepare error: {type(e).__name__}: {e}"

    # -- Session hooks -------------------------------------------------------

    def on_session_switch(self, new_session_id: str, **kwargs: Any) -> None:
        """Rebind session — no-op for payload-level isolation."""
        logger.debug("Session switch to %s", new_session_id)

    def on_session_end(self, messages: list, **kwargs: Any) -> None:
        """Optional: flush/compact on session end."""
        pass

    def on_pre_compress(self, messages: list, *, session_id: str = "", **kwargs: Any) -> None:
        """Called before context compression — optional."""
        pass

    # -- Config + Tool Schemas (ABC methods) --------------------------------

    def get_tool_schemas(self) -> list[dict]:
        """Return the tool schemas for this provider."""
        from .tool_schemas import ALL_TOOL_SCHEMAS
        return list(ALL_TOOL_SCHEMAS)

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write provider config values to disk (config.json next to this module).

        Called by ``hermes memory setup`` (hermes_cli/memory_setup.py) and by the
        dashboard's memory-provider route. Writes next to the module rather than
        into a hardcoded ``plugins/memory/<name>/`` path so it lands beside this
        copy no matter which install location discovery picked.
        """
        import json as _json

        cfg_path = _config_json_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if cfg_path.exists():
            try:
                existing = _json.loads(cfg_path.read_text())
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(values or {})
        cfg_path.write_text(_json.dumps(existing, indent=2))

    def backup_paths(self) -> list[str]:
        """Return paths outside HERMES_HOME for hermes backup/import (none for Qdrant)."""
        return []

    # -- Shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        """Close Qdrant client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._initialized = False
        logger.info("QdrantMemoryProvider shut down")

    # -- Config --------------------------------------------------------------

    @staticmethod
    def get_config_schema() -> list[dict]:
        """Return config field definitions for the dashboard/CLI.

        Fields use the ``MemoryProvider`` contract's vocabulary: a ``key``
        identifies the setting, ``type`` is the HTTP/dashboard kind
        (``string``/``integer``/``select``/``secret``), and ``secret: True``
        routes the value to ``.env``. Consumers are
        ``hermes_cli/memory_setup.py::_prompt_schema_fields`` and
        ``hermes_cli/web_server_memory.py::_normalize_memory_provider_schema``
        — both index on ``field["key"]``, so a field without one silently
        vanishes from the dashboard and hard-crashes the setup wizard.
        """
        return [
            {
                "key": "url",
                "label": "Qdrant URL",
                "type": "string",
                "default": "http://localhost:6333",
                "description": "Qdrant server URL (REST mode)",
            },
            {
                "key": "api_key",
                "label": "API Key",
                "type": "secret",
                "default": "",
                "env_var": "QDRANT_API_KEY",
                "description": "Qdrant API key (cloud mode only)",
            },
            {
                "key": "collection",
                "label": "Collection Name",
                "type": "string",
                "default": "hermes_memories",
                "description": "Qdrant collection to use",
            },
            {
                "key": "vector_size",
                "label": "Vector Size",
                "type": "integer",
                "default": 384,
                "minimum": 1,
                "description": "Embedding dimension (match your embedder model)",
            },
            {
                "key": "distance",
                "label": "Distance Metric",
                "type": "select",
                "default": "Cosine",
                "choices": ["Cosine", "Dot", "Euclid"],
                "description": "Vector distance metric",
            },
            {
                "key": "embedder",
                "label": "Embedder",
                "type": "select",
                "default": "fastembed",
                "choices": ["fastembed", "sentence-transformers"],
                "description": (
                    "Embedding backend. 'fastembed' (default) runs the model on "
                    "ONNX Runtime — CPU-only, ~290 MB peak RSS, no CUDA build "
                    "needed. 'sentence-transformers' uses PyTorch: ~2.2 GB peak "
                    "RSS, but faster on a machine with a GPU. Both produce "
                    "identical vectors for the same model."
                ),
            },
            {
                "key": "model",
                "label": "Embedding Model",
                "type": "string",
                "default": DEFAULT_MODEL,
                "description": (
                    "Model checkpoint, passed explicitly to the backend. Never "
                    "left blank: a library's own default model would silently "
                    "embed in a different vector space and break recall."
                ),
            },
            {
                "key": "device",
                "label": "Device",
                "type": "select",
                "default": "auto",
                "choices": ["auto", "cpu", "cuda"],
                "description": (
                    "Compute device. 'auto' lets the backend choose (CPU for "
                    "ONNX). 'cuda' requires a working CUDA provider and fails "
                    "loudly if none is present."
                ),
            },
        ]

    # -- Internal helpers ----------------------------------------------------

    # -- Embedder ------------------------------------------------------------

    def get_embedder(self) -> Embedder:
        """Return the configured embedder, validating it on first use.

        A bad backend or an empty model name is a deterministic configuration
        error, so it raises here rather than being quietly replaced by some
        other model.
        """
        if self._embedder_impl is None:
            self._embedder_impl = Embedder(
                backend=self._embedder,
                model=self._model,
                device=self._device,
            )
        return self._embedder_impl

    def _embed(self, text: str) -> list[float]:
        """Encode `text` to a dense vector.

        Raises EmbeddingError (a config or runtime subclass) on every failure
        path. It never returns None: a caller that received a vector can trust
        that the text really was encoded, which the previous
        ``try/except: return None`` could not guarantee.
        """
        embedder = self.get_embedder()
        try:
            vec = embedder.encode(text)
        except EmbeddingConfigError:
            # Tier 1: deterministic. Re-raise so the tool boundary can report
            # it verbatim, and record it so /status goes red.
            self._embed_error = (
                f"embedder misconfigured: backend={self._embedder!r} "
                f"model={self._model!r}"
            )
            logger.error("Qdrant embedder configuration error: %s",
                         embedder, exc_info=True)
            raise
        except EmbeddingRuntimeError as e:
            # Tier 2: transient. Do not raise-and-hide, do not return None —
            # report loudly and let the caller decide whether to retry.
            self._embed_error = str(e)
            logger.error("Qdrant embedding failed (transient): %s", e)
            raise
        self._embed_error = ""
        return vec

    # -- Model preparation and validation -------------------------------------

    def ensure_model(self, download: bool = False) -> tuple[bool, str]:
        """Check the configured model is available locally, optionally fetching.

        Returns ``(ok, message)`` where the message is written for a human, not
        for a log file. Called from check_backend() so /status and the
        dashboard report a missing model instead of showing green.
        """
        try:
            embedder = self.get_embedder()
        except EmbeddingConfigError as e:
            self._embed_error = str(e)
            return False, str(e)

        present = model_is_present(self._model)
        if present and not download:
            return True, f"model {self._model!r} is present (backend {self._embedder})"

        if not download:
            msg = (
                f"model {self._model!r} not found in {model_cache_dir()!r}. "
                f"Run qdrant_prepare to download it, or set memory.qdrant.model."
            )
            self._embed_error = msg
            return False, msg

        # download=True: actually construct it, which fetches the weights.
        try:
            dim = embedder.dimension()
        except EmbeddingError as e:
            self._embed_error = str(e)
            return False, f"download failed: {e}"
        msg = f"model {self._model!r} ready (backend {self._embedder}, {dim} dims)"
        self._embed_error = ""
        return True, msg

    def validate_vector_spec(self) -> list[str]:
        """Return user-facing problems/recommendations about the vector setup.

        Deliberately returns prose lines, not booleans: the value is telling the
        user what is wrong, which numbers disagree, and what to do about it.
        Empty list means everything checks out.
        """
        issues: list[str] = []

        # 1. model dimensionality vs configured vector_size
        try:
            model_dim = self.get_embedder().dimension()
        except EmbeddingError as e:
            issues.append(f"embedder unavailable: {e}")
            model_dim = 0

        if model_dim and self._vector_size != model_dim:
            issues.append(
                f"vector_size is {self._vector_size} but model "
                f"{self._model!r} produces {model_dim} dimensions. "
                f"Set memory.qdrant.vector_size to {model_dim} (and recreate the "
                f"collection) or choose a {self._vector_size}-dim model."
            )

        # 2. distance metric
        if self._distance != "Cosine":
            issues.append(
                f"distance is {self._distance!r}; sentence-embedding models "
                "like all-MiniLM-L6-v2 expect 'Cosine' on L2-normalised "
                "vectors. Other metrics change result ranking."
            )

        # 3. config vs the live collection — the silently-wrong-recall state
        live = self._live_vector_spec()
        if live:
            live_size, live_dist = live
            if model_dim and live_size and live_size != model_dim:
                issues.append(
                    f"collection {self._collection!r} stores {live_size}-dim "
                    f"vectors but the configured model produces {model_dim}. "
                    "Recall will be wrong until one of them changes."
                )
            if live_dist and live_dist.lower() != str(self._distance).lower():
                issues.append(
                    f"collection uses distance {live_dist!r} but config says "
                    f"{self._distance!r}."
                )
        return issues

    def _live_vector_spec(self) -> tuple[int, str] | None:
        """Read the collection's actual vector size/distance, or None if unknown.

        Opens its own short-lived client (same pattern as check_backend) rather
        than reusing the provider's, so a validation run never disturbs a live
        connection and never trips the circuit breaker.
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return None
        if not self._url:
            return None

        client = None
        try:
            client = QdrantClient(url=self._url, api_key=self._api_key or None,
                                  prefer_grpc=False, timeout=5)
            info = client.get_collection(self._collection)
            raw = info.config.params.vectors
            spec = None
            if isinstance(raw, dict):
                spec = raw.get("dense")
            elif raw is not None:
                # Unnamed/single vector config — surface its size anyway.
                spec = {"size": getattr(raw, "size", None),
                        "distance": getattr(raw, "distance", None)}
            if spec is None:
                return None
            size = int(getattr(spec, "size", 0) or 0)
            dist = str(getattr(spec, "distance", "") or "")
            return (size, dist) if size else None
        except Exception as e:
            logger.debug("live vector spec unavailable: %s", e)
            return None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def prepare_report(self, download: bool = False) -> str:
        """Human-readable readiness report combining presence and spec checks."""
        lines: list[str] = []
        ok, msg = self.ensure_model(download=download)
        lines.append(("OK  " if ok else "FAIL") + f"  model: {msg}")
        lines.append(f"      backend={self._embedder!r} model={self._model!r} "
                     f"device={self._device!r} cache={model_cache_dir()!r}")

        live = self._live_vector_spec()
        if live:
            lines.append(f"      collection {self._collection!r}: "
                         f"{live[0]} dims, {live[1]}")
        else:
            lines.append(f"      collection {self._collection!r}: not reachable")

        for issue in self.validate_vector_spec():
            lines.append(f"WARN  {issue}")
        return "\n".join(lines)

    def _is_breaker_open(self) -> bool:
        """Check if the circuit breaker is open."""
        if self._breaker_failures < self._BREAKER_THRESHOLD:
            return False
        import time
        if time.time() < self._breaker_open_until:
            return True
        # Cooldown expired — half-open
        self._breaker_failures = 0
        self._breaker_open_until = 0.0
        return False

    def _record_success(self) -> None:
        self._breaker_failures = 0
        self._breaker_open_until = 0.0

    def _record_failure(self) -> None:
        self._breaker_failures += 1
        if self._breaker_failures >= self._BREAKER_THRESHOLD:
            import time
            self._breaker_open_until = time.time() + self._BREAKER_COOLDOWN_SECS
            logger.warning("Circuit breaker OPEN for %ds", self._BREAKER_COOLDOWN_SECS)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Register the QdrantMemoryProvider with Hermes."""
    from agent.memory_provider import MemoryProvider
    provider = QdrantMemoryProvider()
    ctx.register_memory_provider(provider)
    logger.info("QdrantMemoryProvider registered via ctx.register_memory_provider")
