"""Test bootstrap for the standalone ``hermes-qdrant-memory`` repo.

Hermes Agent is **not on PyPI** — this plugin is loaded by directory path from
``$HERMES_HOME/plugins/qdrant/`` and imports three things from the host install:
the ``MemoryProvider`` ABC (``agent.memory_provider``), ``get_secret``
(``agent.secret_scope``) and ``load_config_readonly`` (``hermes_cli.config``).
So the tests here cannot simply ``import`` them.

Two jobs, and only these two:

1. **Expose the repo root as ``plugins.memory.qdrant``** — the import path the
   provider uses when Hermes loads it, and the one the tests import. Registering
   the real submodules in ``sys.modules`` (rather than copying the source) means
   the tests exercise the shipped files, not a vendored duplicate.

2. **Stub the host.** The stubs below are deliberately *contract-shaped*, not
   core copies: the ABC carries the same abstract methods and the same default
   no-op bodies the real one has, and the two consumers are the actual
   behaviour the plugin's config schema has to survive — index on ``key``,
   drop key-less fields, and never raise on a field the wizard does not
   recognise. When the host install *is* importable (``HERMES_SOURCE`` on the
   path), nothing is stubbed and the real core is used instead.

The tests that require a live Qdrant server on ``http://localhost:6333`` skip
themselves when nothing is listening; see ``_qdrant_available`` below.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. The repo root IS the plugin directory. Make it importable as
#    ``plugins.memory.qdrant`` without vendoring anything.
# ---------------------------------------------------------------------------

_PACKAGE = "plugins.memory.qdrant"
_SUBMODULES = ("_backend", "_setup", "tool_schemas")


def _register_plugin_package() -> None:
    if _PACKAGE in sys.modules:
        return

    # ``plugins.memory`` is a traversal-only parent whose __path__ IS the repo
    # root, so ``from .tool_schemas import ...`` inside the provider resolves
    # through the ordinary import machinery — no per-submodule plumbing.
    plugins = sys.modules.setdefault("plugins", types.ModuleType("plugins"))
    plugins.__path__ = []
    parent = types.ModuleType("plugins.memory")
    parent.__path__ = [str(REPO_ROOT)]
    sys.modules["plugins.memory"] = parent
    plugins.memory = parent

    spec = importlib.util.spec_from_file_location(
        _PACKAGE, REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = module
    spec.loader.exec_module(module)
    parent.qdrant = module

    for sub in _SUBMODULES:  # fail fast if a shipped file does not import
        importlib.import_module(f"{_PACKAGE}.{sub}")


# ---------------------------------------------------------------------------
# 2. Host stubs. Contract-shaped; skipped entirely when real core is importable.
# ---------------------------------------------------------------------------


def _real_core_available() -> bool:
    if os.environ.get("HERMES_SOURCE"):
        src = str(Path(os.environ["HERMES_SOURCE"]).resolve())
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        import agent.memory_provider  # noqa: F401
        import hermes_cli.config  # noqa: F401
    except Exception:
        return False
    return True


def _stub_memory_provider_abc() -> None:
    """``agent.memory_provider`` — the ABC plus ``RecallStatus``.

    Mirrors the host's surface: three abstract methods, every other hook a
    documented no-op. A provider that fails to override an abstract method must
    still fail to instantiate here, or the conformance test proves nothing.
    """
    from dataclasses import dataclass, field

    agent = sys.modules.setdefault("agent", types.ModuleType("agent"))
    agent.__path__ = []
    mod = types.ModuleType("agent.memory_provider")

    @dataclass(frozen=True)
    class RecallStatus:
        provider_label: str
        count: int
        glyph: str = "\U0001f9e0"

    class MemoryProvider(ABC):
        pre_compress_checkpoint_api_version = 1

        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...

        def unavailable_reason(self) -> str:
            return ""

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            return None

        def recall_status(self) -> Optional[RecallStatus]:
            return None

        def sync_turn(self, user_content: str, assistant_content: str, *,
                      session_id: str = "", messages=None, turn_author=None) -> None:
            return None

        @abstractmethod
        def get_tool_schemas(self) -> List[Dict[str, Any]]: ...

        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
            raise NotImplementedError(
                f"Provider {self.name} does not handle tool {tool_name}")

        def shutdown(self) -> None:
            return None

        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
            return None

        def identity_signature(self) -> Dict[str, Any]:
            return {}

        def on_session_end(self, messages) -> None:
            return None

        def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                              reset: bool = False, rewound: bool = False, **kwargs) -> None:
            return None

        def on_pre_compress(self, messages) -> str:
            return ""

        def on_delegation(self, task: str, result: str, *, child_session_id: str = "",
                          **kwargs) -> None:
            return None

        def get_config_schema(self) -> List[Dict[str, Any]]:
            return []

        def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
            return None

        def on_memory_write(self, action: str, target: str, content: str,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
            return None

        def backup_paths(self) -> List[str]:
            return []

    def spawn_context_thread(target, *, name: str, daemon: bool = True,
                             args: tuple = (), kwargs=None):
        import threading
        return threading.Thread(target=target, args=args, kwargs=kwargs or {},
                                name=name, daemon=daemon)

    mod.MemoryProvider = MemoryProvider
    mod.RecallStatus = RecallStatus
    mod.spawn_context_thread = spawn_context_thread
    mod.INDICATOR_GLYPH = "\U0001f9e0"
    agent.memory_provider = mod
    sys.modules["agent.memory_provider"] = mod


def _stub_secret_scope() -> None:
    """``agent.secret_scope.get_secret`` — env lookup, no fail-closed mode.

    The host raises ``UnscopedSecretError`` under a multiplex gateway with no
    bound scope. The provider already treats that as "no override" (it catches
    the exception), so the standalone tests need only the happy path plus the
    same failure surface.
    """
    agent = sys.modules.setdefault("agent", types.ModuleType("agent"))
    agent.__path__ = []
    mod = types.ModuleType("agent.secret_scope")

    class UnscopedSecretError(RuntimeError):
        pass

    def get_secret(name: str, default: str = "") -> str:
        return os.environ.get(name, default)

    mod.get_secret = get_secret
    mod.UnscopedSecretError = UnscopedSecretError
    agent.secret_scope = mod
    sys.modules["agent.secret_scope"] = mod


def _stub_hermes_cli() -> None:
    """``hermes_cli.config`` + the two real config-schema consumers.

    ``load_config_readonly`` is the config.yaml overlay; standalone it is empty
    (there is no host config.yaml), which is also the "no override" case.

    The two consumers are reproduced as their actual contract, because that
    contract is the thing under test: both index on ``field["key"]``, and the
    dashboard normalizer silently drops key-less fields.
    """
    cli = sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
    cli.__path__ = []

    config_mod = types.ModuleType("hermes_cli.config")

    def load_config_readonly() -> Dict[str, Any]:
        return {}

    config_mod.load_config_readonly = load_config_readonly
    cli.config = config_mod
    sys.modules["hermes_cli.config"] = config_mod

    setup_mod = types.ModuleType("hermes_cli.memory_setup")

    def _prompt_schema_fields(name, schema, provider_config, env_writes) -> bool:
        """Walk a schema the way the wizard does: index on ``key``."""
        for field in schema:
            key = field["key"]  # KeyError here is the regression under test
            desc = field.get("description", key)
            if field.get("secret", False):
                env_var = field.get("env_var")
                if not env_var:
                    raise AssertionError(f"secret field lacks env_var: {desc}")
                continue
            provider_config.setdefault(key, field.get("default"))
        return True

    setup_mod._prompt_schema_fields = _prompt_schema_fields
    sys.modules["hermes_cli.memory_setup"] = setup_mod

    web_mod = types.ModuleType("hermes_cli.web_server_memory")

    def _schema_field_kind(raw: Dict[str, Any], choices: list) -> str:
        explicit = str(raw.get("kind") or raw.get("type") or "").strip().lower()
        default = raw.get("default")
        if raw.get("secret"):
            return "secret"
        if choices:
            return "select"
        if explicit in {"bool", "boolean"} or isinstance(default, bool):
            return "boolean"
        if explicit in {"int", "integer"} or (isinstance(default, int)
                                              and not isinstance(default, bool)):
            return "integer"
        if explicit in {"float", "number"} or isinstance(default, float):
            return "number"
        return "text"

    def _normalize_memory_provider_schema(name: str, provider: Any) -> List[Dict[str, Any]]:
        raw_schema = []
        if provider is not None and hasattr(provider, "get_config_schema"):
            try:
                raw = provider.get_config_schema()
                if isinstance(raw, list):
                    raw_schema = [f for f in raw if isinstance(f, dict)]
            except Exception:
                return []
        fields = []
        for raw in raw_schema:
            key = str(raw.get("key") or "").strip()
            if not key:
                continue  # silently dropped — the dashboard's real behaviour
            choices = raw.get("choices") or raw.get("options") or []
            if not isinstance(choices, list):
                choices = []
            fields.append({
                "key": key,
                "label": str(raw.get("label") or key.replace("_", " ").title()),
                "kind": _schema_field_kind(raw, choices),
                "description": str(raw.get("description") or ""),
                "placeholder": str(raw.get("placeholder") or ""),
                "required": bool(raw.get("required", False)),
                "default": raw.get("default", ""),
                "options": [{"value": str(c), "label": str(c), "description": ""}
                            for c in choices],
                "url": str(raw.get("url") or ""),
                "when": raw.get("when") if isinstance(raw.get("when"), dict) else None,
                "minimum": raw.get("minimum"),
                "maximum": raw.get("maximum"),
                "step": raw.get("step"),
                "_env_key": str(raw.get("env_var") or "") or None,
            })
        return fields

    web_mod._normalize_memory_provider_schema = _normalize_memory_provider_schema
    sys.modules["hermes_cli.web_server_memory"] = web_mod


# Stubs FIRST: the provider's module level does ``from agent.memory_provider
# import MemoryProvider, RecallStatus``, so the ABC must already be importable
# when the package is exec'd. Then the plugin package, which needs the stubs.
if not _real_core_available():
    _stub_memory_provider_abc()
    _stub_secret_scope()
    _stub_hermes_cli()
_register_plugin_package()


# ---------------------------------------------------------------------------
# 3. One model cache for the whole session
# ---------------------------------------------------------------------------


def _pin_session_model_cache() -> None:
    """Point every test at ONE model cache instead of one per test.

    The provider pins its weights to ``<home>/state/qdrant/model_cache``, and
    the ``_isolate_qdrant_home`` fixture hands each test its own
    ``HERMES_HOME`` — hence its own empty pin. Any test that really loads the
    model would then download its own 87 MB copy: measured on 2026-09-29, four
    suite runs left 54 copies (1.8 GB) under ``$TMPDIR/pytest-of-gk``.

    ``FASTEMBED_CACHE_PATH`` is the override ``pinned_cache_dir()`` honours
    first, so setting it here — once, at collection time, while ``HERMES_HOME``
    is still the real session value — makes every test share the cache the live
    install already has. An operator-set value is left alone. A machine with an
    empty cache downloads once, exactly like production's first embed.
    """
    if os.environ.get("FASTEMBED_CACHE_PATH", "").strip():
        return
    from plugins.memory.qdrant import pinned_cache_dir

    os.environ["FASTEMBED_CACHE_PATH"] = str(pinned_cache_dir())


_pin_session_model_cache()


# ---------------------------------------------------------------------------
# 4. Live-server gate
# ---------------------------------------------------------------------------


def _qdrant_reachable(url: str = "http://localhost:6333", timeout: float = 1.5) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/collections", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def live_qdrant() -> str:
    """Skip the calling test unless a Qdrant server is listening on :6333.

    ``docker run -p 6333:6333 qdrant/qdrant`` is all it takes to un-skip them.
    """
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if not _qdrant_reachable(url):
        pytest.skip(f"no Qdrant server at {url} — start one to run the live tests")
    return url
