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
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus

# ``embedder`` is a sibling module, but this package loads two ways: as
# ``plugins.memory.qdrant`` (relative import works) and as a bare top-level
# ``__init__`` (how the Hermes plugin loader and these tests import it), where a
# relative import has no parent package and raises ImportError. Load it by path
# in that case. Module-object binding rather than a name list, so the re-export
# block below stays a one-liner per name instead of duplicating the list twice.
try:  # pragma: no cover - branch depends on how the package was imported
    from . import embedder as _embedder
except ImportError:  # pragma: no cover
    import importlib.util
    import os

    _spec = importlib.util.spec_from_file_location(
        "hermes_qdrant_embedder",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedder.py"),
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("cannot locate sibling embedder.py") from None
    _embedder = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_embedder)

Embedder = _embedder.Embedder
EmbeddingError = _embedder.EmbeddingError
EmbeddingConfigError = _embedder.EmbeddingConfigError
EmbeddingRuntimeError = _embedder.EmbeddingRuntimeError
BACKEND_FASTEMBED = _embedder.BACKEND_FASTEMBED
BACKEND_ST = _embedder.BACKEND_ST
BACKENDS = _embedder.BACKENDS
DEFAULT_MODEL = _embedder.DEFAULT_MODEL
model_cache_dir = _embedder.model_cache_dir
model_cache_locations = _embedder.model_cache_locations
model_is_present = _embedder.model_is_present
pinned_cache_dir = _embedder.pinned_cache_dir

logger = logging.getLogger("hermes.plugins.memory.qdrant")

# Ceiling on a model-supplied `limit`. A tool argument is not a human budget:
# an accidental `limit=100000` on qdrant_recall scrolls the whole collection
# into memory and bills the caller one enormous response, and on qdrant_search
# it is a payload-heavy fetch. Capped per tool — recall legitimately wants
# more than search, but neither wants "everything".
MAX_LIMIT_SEARCH = 100
MAX_LIMIT_RECALL = 1000


def _clamp_limit(raw: object, default: int, ceiling: int) -> int:
    """Coerce a model-supplied ``limit`` into ``1..ceiling``.

    A tool argument is not a human budget: an accidental ``limit=100000`` on
    qdrant_recall scrolls the whole collection into memory and returns one
    enormous payload, and on qdrant_search it is a payload-heavy fetch. The
    floor matters too — ``limit=0`` or a negative value is a silent
    empty-result, which reads as "nothing was remembered" rather than as a
    bad argument. A non-integer falls back to the default rather than raising
    an unhandled traceback at the model.
    """
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return min(max(value, 1), ceiling)


def _url_without_userinfo(url: str) -> str:
    """The URL with any ``user:pass@`` credential removed, for display only.

    The system prompt is model-visible text and is routinely echoed back in
    transcripts, logs and bug reports, so a URL carrying basic-auth
    credentials would leak them to the provider and anywhere the prompt is
    quoted. The connection itself is unaffected — this only shapes the string
    we print.
    """
    if not url or "@" not in url:
        return url
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        if not parts.netloc or "@" not in parts.netloc:
            return url
        host = parts.netloc.rsplit("@", 1)[1]
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except Exception:
        # Never let a display helper break the prompt; over-redact instead.
        return url.split("@", 1)[1] if "@" in url else url

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
PLUGIN_VERSION = "0.1.5"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _state_home(hermes_home: str | None = None) -> Any:
    """The profile home whose ROOT holds this provider's two state files.

    Deliberately NOT the plugin directory. The plugin dir is a *build input*:
    ``pm.workspace.members_stamp()`` hashes every file inside it — its
    ``_MEMBER_EXCLUDE`` names only ``.git``/``.venv``/``venv``/``node_modules``/
    ``__pycache__``, so ``status.json`` and ``config.json`` are hashed too — and
    folds that hash into the venv dependency stamp. One byte written next to the
    module therefore invalidated the environment and re-synced dependencies on
    every launch. Writing beside the module also fails outright on a read-only
    (pip/system) install.

    This is the convention, not a workaround unique to us: sibling providers
    keep their state in the home root (``mem0.json``, ``honcho.json``,
    ``supermemory.json``).

    Precedence: the ``hermes_home`` ``save_config()`` was handed → the process's
    context-local override / ``HERMES_HOME`` → ``~/.hermes``.
    """
    from pathlib import Path as _Path
    if hermes_home is not None and str(hermes_home).strip():
        return _Path(str(hermes_home).strip())
    try:
        from hermes_constants import get_hermes_home
        return _Path(get_hermes_home())
    except Exception:
        import os as _os
        env = _os.environ.get("HERMES_HOME", "").strip()
        return _Path(env) if env else _Path.home() / ".hermes"


def _config_json_path(hermes_home: str | None = None) -> Any:
    """``<HERMES_HOME>/qdrant.json`` — outside the plugin member dir.

    Was ``<plugin dir>/config.json``. It stays a dict on disk rather than a
    config.yaml read because ``hermes memory setup`` and the dashboard both
    route through ``MemoryProvider.save_config()``, which is contractually
    file-based; only the directory moved (see :func:`_state_home`).
    """
    return _state_home(hermes_home) / "qdrant.json"


#: State files 0.1.4 and earlier wrote BESIDE the module, i.e. into a directory
#: that is a hermes build input. Left behind by an upgrade, their mere presence
#: keeps ``pm.workspace.members_stamp()`` — and therefore the venv dependency
#: stamp — different from the one recorded at install time, so every launch takes
#: the sync branch. Nothing reads them any more; the live writer and reader both
#: resolve through :func:`_state_home`. Names are literal, not globs: a sweep that
#: matched patterns would risk deleting a real plugin input.
_LEGACY_IN_DIR_STATE = ("status.json", "config.json")


def sweep_legacy_in_dir_state() -> list:
    """Remove pre-0.1.5 state files left in the plugin member dir. Best effort.

    Called once per :meth:`QdrantMemoryProvider.initialize`, which every store
    and recall passes through — that is what makes the cleanup self-healing: a
    long-lived install keeps picking up a residue written by a process that was
    already running the older code, without the user running anything.

    Returns the paths it removed (empty when there was nothing to do or the
    directory is not writable, e.g. a read-only pip install). Never raises:
    failing to tidy residue must not take memory down.
    """
    from pathlib import Path as _Path

    try:
        member_dir = _Path(__file__).resolve().parent
        removed = []
        for name in _LEGACY_IN_DIR_STATE:
            stale = member_dir / name
            try:
                stale.unlink()
            except FileNotFoundError:
                continue
            except OSError as e:
                logger.debug("could not remove legacy %s: %s", stale, e)
                continue
            removed.append(str(stale))
        if removed:
            logger.info(
                "Removed legacy in-dir state that predates 0.1.5: %s. Their "
                "contents now live in %s; while they sat in the member dir they "
                "changed the venv dependency stamp and re-synced dependencies "
                "on every launch.",
                ", ".join(removed), _state_home(),
            )
        return removed
    except Exception as e:  # pragma: no cover - defensive by contract
        logger.debug("legacy in-dir state sweep skipped: %s", e)
        return []


def _load_plugin_config() -> dict:
    """Read saved provider config: env < config.yaml's ``memory.qdrant`` < state file.

    Precedence, lowest to highest:

    1. the built-in defaults in ``__init__``
    2. ``QDRANT_URL`` / ``QDRANT_API_KEY`` from the environment (cloud deploys
       and the dashboard's secret writer, which routes secrets to ``.env``)
    3. ``memory.qdrant:`` in config.yaml, for homes that set knobs directly
    4. ``<HERMES_HOME>/qdrant.json``, which is what ``save_config()`` writes
       via ``hermes memory setup`` and the dashboard

    A malformed or missing file yields no override rather than an error, so a
    bad hand-edit can never take memory offline.
    """
    import json as _json

    merged: dict = {}

    try:
        from hermes_cli.config import (
            load_config_readonly,  # managed overlay + ${VAR} expansion
        )
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
        logger.warning(
            "Qdrant state file (<HERMES_HOME>/qdrant.json) unreadable, "
            "using config.yaml/env only: %s", e
        )

    # Secrets live in the env, not the config file — read them last so a
    # rotated key takes effect without rewriting ``qdrant.json``. ``get_secret``
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

    # Unknown keys are dropped WITH a warning, not silently: a typo'd key that
    # reads as configured-but-inert is the failure class that took a peer
    # project 50 silently-ignored config keys (mnemosyne issue #482). The
    # warning makes the mistake visible; the drop keeps behaviour honest to it.
    known = {"collection", "vector_size", "distance", "url", "api_key",
             "embedder", "model", "device", "progress"}
    unknown = sorted(set(merged) - known)
    if unknown:
        logger.warning(
            "memory.qdrant: ignoring unknown config key(s): %s (known keys: %s)",
            ", ".join(unknown), ", ".join(sorted(known)),
        )
        merged = {k: v for k, v in merged.items() if k in known}
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
        # Progress display: status_callback is passed to initialize() by the
        # orchestrator (agent_init.py:1284). We store it and emit progress
        # events during sync_turn / prefetch so the user sees the plugin
        # working. Mode is controlled by config: "off" | "minimal" | "verbose".
        self._status_callback: Any = None
        self._progress_mode: str = self._config.get("progress", "minimal")
        self._last_recall_count: int = 0
        # Profile home, captured from initialize()'s scoping kwargs (or left
        # empty so _state_home() resolves it from the environment). It decides
        # where qdrant.json / qdrant-status.json live — never the plugin dir.
        self._hermes_home: str = ""

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
        # Before anything else: drop state a pre-0.1.5 process wrote beside the
        # module. It is inert, but pm.workspace.members_stamp() hashes it and the
        # residue therefore re-syncs the venv on every launch until it is gone.
        sweep_legacy_in_dir_state()
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

        # Store status_callback for progress display. The orchestrator passes it
        # (agent_init.py:1284) so the plugin can emit progress events that the
        # CLI/TUI/gateway renders. We never call it directly — only through
        # _emit_progress() which respects the progress mode config.
        self._status_callback = kwargs.get("status_callback")
        # Which profile's home holds our state files. The orchestrator passes
        # it as a scoping kwarg; an empty value lets _state_home() fall back to
        # the process's own HERMES_HOME / context override.
        self._hermes_home = str(kwargs.get("hermes_home") or "").strip()

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
            self._backend_error = (
                f"cannot reach Qdrant at {self._url}: {type(e).__name__}: {e}"
            )
            self._initialized = False
            self._client = None
            logger.error(
                "QdrantMemoryProvider initialize failed: %s", self._backend_error
            )
            raise RuntimeError(self._backend_error) from e
        # Verified reachable: clear any stale negative so a restart can recover.
        self._backend_error = ""
        if self._collection not in collections:
            logger.info(
                "Creating collection %s (dims=%d, distance=%s, named vector 'dense')",
                self._collection, self._vector_size, self._distance,
            )
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
            f"URL: {_url_without_userinfo(self._url)} | "
            f"Vector size: {self._vector_size}"
        )

    # -- Prefetch / Recall ---------------------------------------------------

    @staticmethod
    def _dedup_hits(pairs: list[tuple[float, str]]) -> list[tuple[float, str]]:
        """Drop later hits that substantially repeat an earlier one.

        Word-level Jaccard (>=0.72) and containment (>=0.86) thresholds —
        the two numbers Mnemosyne's `_semantic_dedup_prefetch` uses, adopted
        after comparing implementations. A prefetch that injects three
        near-copies of one turn burns context tokens and makes recall look
        noisier than it is. n <= 10, so the O(n^2) compare is free.
        """
        kept: list[tuple[float, str]] = []
        kept_words: list[set[str]] = []
        for score, text in pairs:
            words = set(text.lower().split())
            if not words:
                continue
            duplicate = False
            for prev in kept_words:
                inter = len(words & prev)
                union = len(words | prev)
                if union and inter / union >= 0.72:
                    duplicate = True
                    break
                if inter / min(len(words), len(prev)) >= 0.86:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((score, text))
                kept_words.append(words)
        return kept

    def prefetch(self, query: str, *, session_id: str = "", **kwargs: Any) -> str:
        """Semantic/hybrid search for memories relevant to *query*."""
        if not self._initialized or not self._client:
            return ""

        if self._is_breaker_open():
            logger.debug("Circuit breaker open — skipping prefetch")
            return ""

        self._emit_progress("memory_sync", "💾 qdrant — retrieving...", verbose=True)

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
            pairs: list[tuple[float, str]] = []
            for pt in hits:
                payload = pt.payload if hasattr(pt, "payload") else {}
                score = pt.score if hasattr(pt, "score") else 0.0
                text = payload.get("text", "")
                if text:
                    pairs.append((score, text))
            pairs = self._dedup_hits(pairs)
            lines = [f"- [{score:.2f}] {text}" for score, text in pairs]

            count = len(lines)
            self._last_recall_count = count
            self._note_status(last_recall=datetime.now(UTC).isoformat(),
                              last_recall_count=count)
            if count > 0:
                self._emit_progress(
                    "memory_sync",
                    f"💾 qdrant — recalled {count} memor{'y' if count == 1 else 'ies'}",
                )
            else:
                self._emit_progress("memory_sync", "💾 qdrant — no relevant memories")

            return "\n".join(lines) if lines else ""

        except Exception as e:
            self._record_failure()
            logger.warning("Qdrant prefetch failed: %s", e)
            return ""

    def queue_prefetch(
        self, query: str, *, session_id: str = "", **kwargs: Any
    ) -> None:
        """Background prefetch on a scope-bound thread.

        ``plugins/AGENTS.md`` requires every memory-provider background job to go
        through ``spawn_context_thread`` so the worker inherits the spawning
        profile's contextvars. A bare ``threading.Thread`` runs with no scope
        and fails closed, or worse, writes into the launch profile's tenant.
        """
        try:
            from agent.memory_provider import spawn_context_thread
        except Exception:
            # Core symbol unavailable (tests stub it, or an old checkout): fall
            # back rather than lose the prefetch entirely, and say so once.
            import threading

            logger.debug(
                "spawn_context_thread unavailable; prefetch thread will run unscoped"
            )
            spawn = threading.Thread
        else:
            spawn = spawn_context_thread

        def _run():
            try:
                self.prefetch(query, session_id=session_id, **kwargs)
            except Exception as e:
                # A background prefetch failure must be visible without being
                # fatal; the turn does not depend on it.
                logger.warning("Qdrant background prefetch failed: %s", e)

        t = spawn(target=_run, name="qdrant-prefetch", daemon=True)
        t.start()
        self._prefetch_thread = t

    def recall_status(self) -> RecallStatus:
        """Return current recall state."""
        return RecallStatus(
            provider_label=self.name,
            count=self._last_recall_count,
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

        self._emit_progress("memory_sync", "💾 qdrant — storing turn...", verbose=True)

        t0 = time.monotonic()
        try:
            # Combine user + assistant for embedding
            combined = f"user: {user}\nassistant: {assistant}"
            dense_vec = self._embed(combined)

            point_id = str(uuid.uuid4())
            timestamp = datetime.now(UTC).isoformat()

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
            self._note_status(
                last_store=datetime.now(UTC).isoformat(),
                last_store_ms=int((time.monotonic() - t0) * 1000),
            )

            # Get the total point count for the progress message
            try:
                info = self._client.get_collection(self._collection)
                count = info.points_count
                self._emit_progress(
                    "memory_sync", f"💾 qdrant — stored ({count:,} points)"
                )
            except Exception:
                self._emit_progress("memory_sync", "💾 qdrant — stored")

        except Exception as e:
            self._record_failure()
            self._note_status(
                last_error=f"sync_turn: {e}"[:300],
                last_error_at=datetime.now(UTC).isoformat(),
            )
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
        limit = _clamp_limit(args.get("limit"), 10, MAX_LIMIT_SEARCH)

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
                        "timestamp": datetime.now(UTC).isoformat(),
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
        limit = _clamp_limit(args.get("limit"), 100, MAX_LIMIT_RECALL)
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
            # A scroll has no query, so there is no similarity score — printing
            # a fabricated [0.00] for every row would be a fake number. Show a
            # score bracket only when the payload actually carries one.
            lines = []
            for pt in results:
                if not hasattr(pt, "payload"):
                    continue
                payload = pt.payload or {}
                text = payload.get("text", "")
                if not text:
                    continue
                score = payload.get("score")
                lines.append(f"[{score:.2f}] {text}"
                             if isinstance(score, (int, float)) else text)
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

    # -- Config + Tool Schemas (ABC methods) --------------------------------

    def get_tool_schemas(self) -> list[dict]:
        """Return the tool schemas for this provider."""
        from .tool_schemas import ALL_TOOL_SCHEMAS
        return list(ALL_TOOL_SCHEMAS)

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write provider config values to disk (``<HERMES_HOME>/qdrant.json``).

        Called by ``hermes memory setup`` (hermes_cli/memory_setup.py) and by the
        dashboard's memory-provider route. The file lives in the profile home
        rather than beside this module so the member directory stays byte-stable
        — see :func:`_state_home`. ``hermes_home`` is honoured when the wizard
        passes one; an empty string (the standalone ``_setup.py``) falls back to
        the process home.
        """
        import json as _json
        import os as _os

        cfg_path = _config_json_path(hermes_home)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if cfg_path.exists():
            try:
                existing = _json.loads(cfg_path.read_text())
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}

        # Never persist a credential here. `api_key` arrives in `values` from
        # a caller that does not know Hermes reads it from the secret scope,
        # and writing it here would put a plaintext key in a JSON file that
        # the dashboard reads and `hermes backup` copies. It reaches the
        # provider through `QDRANT_API_KEY` via `get_secret()` instead, which
        # is also where a rotated key lands without a re-save.
        values = dict(values or {})
        values.pop("api_key", None)
        existing.update(values)

        # If a previous version (or a hand edit) already wrote a key here,
        # scrub it on the next save rather than leaving it in place.
        if existing.pop("api_key", None) is not None:
            logger.info(
                "Removed a legacy api_key from %s; the key now comes from "
                "QDRANT_API_KEY via the secret scope",
                cfg_path,
            )

        # Atomic (tmp + replace) so a reader never sees a truncated file, and
        # 0600 from creation rather than a chmod afterwards: a bare write_text
        # at the default umask briefly exposes the contents to any local
        # reader, and on a platform without POSIX modes the chmod was silently
        # skipped, leaving the README's "0600" claim untrue.
        tmp = cfg_path.with_name(cfg_path.name + ".tmp")
        fd = _os.open(
            tmp,
            _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC,
            0o600,
        )
        with _os.fdopen(fd, "w") as handle:
            handle.write(_json.dumps(existing, indent=2))
        _os.replace(tmp, cfg_path)

    def backup_paths(self) -> list[str]:
        """Paths outside HERMES_HOME for hermes backup/import (none for Qdrant)."""
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
                "secret": True,
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
            {
                "key": "progress",
                "label": "Progress Display",
                "type": "select",
                "default": "minimal",
                "choices": ["off", "minimal", "verbose"],
                "description": (
                    "Progress display for memory operations. 'off' shows nothing. "
                    "'minimal' shows completion events (stored, recalled N). "
                    "'verbose' also shows start events (storing..., retrieving...)."
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

        Presence is judged against the directory the backend LOADS from, and a
        copy found anywhere else is named in the message — a stale copy in a
        prunable ``$TMPDIR`` is precisely the case where "present" used to be a
        lie that then downloaded 90 MB mid-session.
        """
        try:
            embedder = self.get_embedder()
        except EmbeddingConfigError as e:
            self._embed_error = str(e)
            return False, str(e)

        cache = model_cache_dir(self._embedder)
        present = model_is_present(self._model, backend=self._embedder)
        stale: list[str] = []
        if self._embedder == BACKEND_FASTEMBED:
            stale = [str(p) for p in model_cache_locations(self._model)
                     if str(p) != cache]

        if present and not download:
            return True, (f"model {self._model!r} is present "
                          f"(backend {self._embedder}) in {cache}")

        if not download:
            stale_note = ""
            if stale:
                stale_note = (f" A copy sits in {', '.join(stale)}, but "
                              f"{self._embedder} loads from {cache}, so it "
                              f"will be downloaded there.")
            msg = (f"model {self._model!r} not found in {cache!r}."
                   f"{stale_note} "
                   f"Run qdrant_prepare to download it, or set "
                   f"memory.qdrant.model.")
            self._embed_error = msg
            return False, msg

        # download=True: actually construct it, which fetches the weights.
        if not present:
            # Announce the fetch before it starts: the Hugging Face download
            # bars otherwise appear out of nowhere in the middle of a turn.
            self._emit_progress(
                "model_sync",
                f"⬇ qdrant — downloading embedding model to {cache}...",
                verbose=True,
            )
        try:
            dim = embedder.dimension()
        except EmbeddingError as e:
            self._embed_error = str(e)
            return False, f"download failed: {e}"
        self._emit_progress("model_sync", f"⬇ qdrant — model ready in {cache}")
        msg = (f"model {self._model!r} ready (backend {self._embedder}, "
               f"{dim} dims) in {cache}")
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
                     f"device={self._device!r} "
                     f"cache={model_cache_dir(self._embedder)!r}")

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

    # -- Progress display ----------------------------------------------------

    def _emit_progress(
        self, event_type: str, message: str, *, verbose: bool = False
    ) -> None:
        """Emit a progress event via status_callback if enabled.

        The progress mode controls what is shown:
        - "off": no progress events
        - "minimal": only completion events (default)
        - "verbose": both start and completion events

        This is called from sync_turn and prefetch to give the user visible
        feedback that the memory plugin is working. The status_callback is
        passed by the orchestrator (agent_init.py:1284) and renders in the
        CLI/TUI/gateway interface.
        """
        if self._progress_mode == "off":
            return
        if verbose and self._progress_mode != "verbose":
            return
        if self._status_callback is None:
            return
        try:
            self._status_callback(event_type, message)
        except Exception:
            # A progress display failure must never break memory operations
            pass

    def _status_json_path(self) -> Any:
        """``<HERMES_HOME>/qdrant-status.json`` — outside the plugin member dir.

        Was ``<plugin dir>/status.json``, which is rewritten on every
        store/recall and therefore re-synced the venv on every launch (see
        :func:`_state_home`). Not a staticmethod any more: the profile home
        comes from ``initialize()``'s scoping kwargs.
        """
        home = _state_home(getattr(self, "_hermes_home", "") or None)
        return home / "qdrant-status.json"

    def _note_status(self, **fields: Any) -> None:
        """Merge last-operation bookkeeping into ``qdrant-status.json`` — best effort.

        Small atomic write (tmp + os.replace) so a reader never sees a
        truncated file, and a failure here is swallowed: observability must
        never break memory operations. This is the answer to a peer project's
        silent-failure class (mnemosyne #1033: weeks of no writes while every
        health surface said healthy) — 'when did this last store/recall?' has
        to be answerable after the fact, from a fresh process.
        """
        import json as _json
        import os as _os
        path = self._status_json_path()
        state: dict = {}
        try:
            if path.exists():
                state = _json.loads(path.read_text())
                if not isinstance(state, dict):
                    state = {}
        except Exception:
            state = {}
        state.update(fields)
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(state, indent=1))
            _os.replace(tmp, path)
        except Exception as e:
            logger.debug("qdrant-status.json write failed (non-fatal): %s", e)

    def get_status_config(self, provider_config: dict) -> dict:
        """Config block for `hermes memory status` (core calls this hook).

        Config keys the CLI cannot know by itself, plus the last store/recall
        from ``<HERMES_HOME>/qdrant-status.json`` so a fresh process can answer
        "is this thing alive?" without a live connection. api_key is never
        displayed.
        """
        import json as _json
        cfg = dict(provider_config or {})
        display = {
            "url": cfg.get("url", self._url),
            "collection": cfg.get("collection", self._collection),
            "embedder": cfg.get("embedder", self._embedder),
            "model": cfg.get("model", self._model),
            "vector_size": cfg.get("vector_size", self._vector_size),
            "distance": cfg.get("distance", self._distance),
            "progress": cfg.get("progress", self._progress_mode),
            "api_key": "(set)" if cfg.get("api_key") or self._api_key else "(unset)",
        }
        try:
            if self._status_json_path().exists():
                state = _json.loads(self._status_json_path().read_text())
                if isinstance(state, dict):
                    display.update({
                        key: state[key]
                        for key in ("last_store", "last_store_ms",
                                    "last_recall", "last_recall_count",
                                    "last_error")
                        if key in state
                    })
        except Exception:
            pass
        return display


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Register the QdrantMemoryProvider with Hermes."""
    provider = QdrantMemoryProvider()
    ctx.register_memory_provider(provider)
    logger.info("QdrantMemoryProvider registered via ctx.register_memory_provider")
