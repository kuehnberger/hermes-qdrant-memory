"""The other half of the boundary: the qd-plugin makes no LLM calls.

The plugin's USP is "no LLM calls in the Hermes process". The reasoning
provider — the retrieval stage's consumer — is store-free (its own
``tests/test_boundary.py``); this plugin must be LLM-free. These tests are
mechanical: a future edit that reaches for a model fails here, not in
production.

Scope: the plugin RUNTIME modules — the top-level ``.py`` files Hermes
loads (``__init__``, ``_backend``, ``_setup``, ``embedder``, ``mdsearch``,
``mdsemantic``, ``tool_schemas``). ``scripts/`` and ``tests/`` are not
loaded by Hermes and are out of scope by design; an eval script that wants
a judge belongs in an eval workspace, not in the plugin.

Three things would each be a boundary violation, and each is checked:

1. an LLM client import (``openai``, ``anthropic``, ``ollama``, …);
2. a ``.llm`` attribute access (``ctx.llm`` — the host's model context);
3. an LLM endpoint string (``chat/completions`` — a raw call in disguise).

Network libraries are banned too, with one carve-out: ``urllib.parse`` is
URL *string* handling (the plugin splits its Qdrant URL), not a network
call. Every other network root (``urllib.request``, ``urllib.error``,
``httpx``, ``requests``, ``aiohttp``, ``socket``, ``http``, ``websockets``)
would be an outbound call the USP does not allow — the plugin's only
network path is Qdrant REST via ``qdrant_client``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The plugin runtime — the modules Hermes loads by directory path.
RUNTIME_MODULES = (
    "__init__.py",
    "_backend.py",
    "_setup.py",
    "embedder.py",
    "mdsearch.py",
    "mdsemantic.py",
    "tool_schemas.py",
)

#: LLM client libraries. None of these may appear in the plugin runtime.
FORBIDDEN_LLM_ROOTS = (
    "openai",
    "anthropic",
    "ollama",
    "llama_cpp",
    "llamasuite",
    "transformers",
    "mistralai",
    "groq",
    "together",
    "cohere",
    "google",
)

#: Network-capable roots. The plugin's only network path is Qdrant REST
#: (via qdrant_client, which is allowed). urllib.parse is carved out below
#: because it is string handling, not I/O.
FORBIDDEN_NETWORK_ROOTS = (
    "httpx",
    "requests",
    "aiohttp",
    "socket",
    "http",
    "websockets",
    "urllib",
)

#: Submodules of banned roots that are pure string/data handling.
ALLOWED_NETWORK_SUBMODULES = {"urllib.parse"}


def _runtime_sources() -> list[Path]:
    return [REPO_ROOT / name for name in RUNTIME_MODULES]


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _imported_full_modules(path: Path) -> set[str]:
    """Every imported module path, e.g. ``urllib.parse``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
    return modules


def _llm_attribute_accesses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "llm":
            found.append(
                f"{node.lineno}: {ast.unparse(node)}"
                if hasattr(ast, "unparse")
                else f"{node.lineno}: .llm"
            )
    return found


def _llm_endpoint_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "chat/completions" in node.value
        ):
            found.append(f"{node.lineno}: {node.value[:60]!r}")
    return found


def test_the_plugin_runtime_exists() -> None:
    """Guard: a renamed module must not silently drop out of the scan."""
    missing = [str(p) for p in _runtime_sources() if not p.is_file()]
    assert not missing, f"plugin runtime modules missing: {missing}"


def test_the_plugin_runtime_imports_no_llm_client() -> None:
    offenders: dict[str, list[str]] = {}
    for source in _runtime_sources():
        bad = sorted(_imported_roots(source) & set(FORBIDDEN_LLM_ROOTS))
        if bad:
            offenders[source.name] = bad
    assert not offenders, f"plugin runtime must not import LLM clients: {offenders}"


def test_the_plugin_runtime_makes_no_outbound_calls() -> None:
    """No network library except Qdrant's own client (and urllib.parse)."""
    offenders: dict[str, list[str]] = {}
    for source in _runtime_sources():
        bad = sorted(_imported_roots(source) & set(FORBIDDEN_NETWORK_ROOTS))
        # urllib.parse is string handling, not a network call.
        if "urllib" in bad:
            full = _imported_full_modules(source)
            non_parse = {
                mod
                for mod in full
                if mod == "urllib" or mod.startswith("urllib.")
            } - ALLOWED_NETWORK_SUBMODULES
            if non_parse:
                offenders[source.name] = sorted(non_parse)
            bad = []
        if bad:
            offenders[source.name] = bad
    assert not offenders, (
        f"plugin runtime must make no outbound calls except Qdrant REST: {offenders}"
    )


def test_the_plugin_runtime_never_touches_the_llm_context() -> None:
    offenders: dict[str, list[str]] = {}
    for source in _runtime_sources():
        found = _llm_attribute_accesses(source)
        if found:
            offenders[source.name] = found
    assert not offenders, (
        f"plugin runtime must never touch .llm (ctx.llm is the host's model "
        f"context): {offenders}"
    )


def test_the_plugin_runtime_contains_no_llm_endpoint_strings() -> None:
    offenders: dict[str, list[str]] = {}
    for source in _runtime_sources():
        found = _llm_endpoint_strings(source)
        if found:
            offenders[source.name] = found
    assert not offenders, (
        f"plugin runtime must not call LLM endpoints: {offenders}"
    )


def test_the_plugin_tool_surface_has_no_llm_tool() -> None:
    """The registered tools are storage tools; none may be model-facing."""
    import plugins.memory.qdrant as plugin_mod

    llm_tools = [
        name
        for name in dir(plugin_mod)
        if not name.startswith("_")
        and any(
            verb in name.lower()
            for verb in ("complete", "generate", "chat", "reason", "answer")
        )
    ]
    assert not llm_tools, f"plugin exposes model-facing tools: {llm_tools}"
