"""Tree-rule and contract gates that the peer analysis surfaced.

Each test here corresponds to a rule that was being violated or a regression
that had no guard. They are cheap, they need no Qdrant server, and they are the
kind of check that fails loudly rather than degrading quietly.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / "__init__.py"


def _plugin_source() -> str:
    return PLUGIN.read_text(encoding="utf-8")


class TestNoBareThread:
    """plugins/AGENTS.md: background work starts via spawn_context_thread.

    A bare ``threading.Thread`` runs with no profile scope, so the worker fails
    closed or writes into the launch profile's tenant. The rule is only
    enforced by review, so pin it mechanically.
    """

    def test_no_bare_threading_thread_in_provider(self):
        src = _plugin_source()
        tree = ast.parse(src)
        offenders = []
        for node in ast.walk(tree):
            # threading.Thread(...) called directly
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Thread"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "threading"
            ):
                offenders.append(node.lineno)
        assert not offenders, (
            f"bare threading.Thread at lines {offenders} — plugins/AGENTS.md "
            f"requires agent.memory_provider.spawn_context_thread so the worker "
            f"inherits the spawning profile's scope"
        )

    def test_thread_imports_are_confined_to_the_fallback(self):
        """A threading import is allowed only inside the documented fallback."""
        src = _plugin_source()
        tree = ast.parse(src)
        imports = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(a.name == "threading" for a in node.names)
        ]
        assert len(imports) <= 1, (
            f"threading imported {len(imports)}x (lines {imports}); it should "
            f"appear once, inside the spawn_context_thread ImportError fallback"
        )

    def test_queue_prefetch_prefers_spawn_context_thread(self):
        """The correct symbol is tried first, and failure degrades loudly-ish."""
        from plugins.memory.qdrant import QdrantMemoryProvider  # type: ignore

        src = inspect.getsource(QdrantMemoryProvider.queue_prefetch)
        assert "spawn_context_thread" in src, (
            "queue_prefetch must attempt agent.memory_provider.spawn_context_thread"
        )
        # The fallback must exist, but must log rather than fail silently.
        assert "logger.debug" in src or "logger.warning" in src, (
            "the fallback path must say something when it runs unscoped"
        )


class TestHookDeclaration:
    """Catalog rule 6: declared capabilities must match what is registered.

    The peer plugin declares five hooks and implements them. We declare none,
    so we must not carry no-op stub bodies: an implemented-but-undeclared hook
    is the same class of mismatch as a declared-but-absent one, and it is
    invisible to the loader.
    """

    NO_HOOK_STUBS = ("on_session_switch", "on_session_end", "on_pre_compress")

    @pytest.mark.parametrize("name", NO_HOOK_STUBS)
    def test_hook_is_not_a_no_op_stub(self, name):
        tree = ast.parse(_plugin_source())
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                fn = node
                break
        if fn is None:
            return  # not implemented at all — correct, and declared absent
        # A no-op is a docstring plus `pass`. Strip both, and anything that is
        # only a debug log, then require real work to remain.
        meaningful = []
        for n in fn.body:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                continue  # docstring
            if isinstance(n, ast.Pass):
                continue  # the no-op marker
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                # logger.debug(...) alone is a log, not behaviour
                f = n.value.func
                if isinstance(f, ast.Attribute) and f.attr.startswith("debug"):
                    continue
            meaningful.append(n)
        assert meaningful, (
            f"{name} is a no-op (docstring + pass, or a debug log only) — either "
            f"delete the stub, because provides_hooks is [] and an "
            f"implemented-but-undeclared hook is the same rule-6 mismatch as a "
            f"declared-but-absent one, or implement it and declare it"
        )

    def test_manifest_declares_no_hooks(self):
        manifest = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        assert "provides_hooks: []" in manifest, (
            "this test assumes we declare no hooks; if hooks are implemented, "
            "declare them here and relax the checks above"
        )


class TestNoHardcodedModelInCallPath:
    """The peer builds a SentenceTransformer at a call site, bypassing its own
    opt-in gate — which is an undeclared network download. Module-level
    constants are the correct home for a model id; a literal inside a function
    body is a silent download that bypasses configuration.

    ``DEFAULT_MODEL`` and ``KNOWN_MODEL_DIMS`` in embedder.py are exactly the
    declarations the rule wants, so they are allowed; a literal passed to a
    backend call is not.
    """

    def test_no_model_literal_inside_a_function_body(self):
        offenders = []
        for py in list(REPO.glob("*.py")) + list((REPO / "scripts").glob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and (
                            node.value.startswith("sentence-transformers/")
                            or node.value.startswith("BAAI/")
                        )
                    ):
                        offenders.append(f"{py.name}:{node.lineno} in {fn.name}()")
        assert not offenders, (
            f"model id hardcoded inside a function at {offenders} — a literal in "
            f"a call path is an implicit download that bypasses the configured "
            f"embedder; declare it at module level"
        )

    def test_model_defaults_are_declared_at_module_level(self):
        """The declarations the rule points at must actually exist."""
        src = (REPO / "embedder.py").read_text(encoding="utf-8")
        assert "DEFAULT_MODEL" in src, "embedder must pin an explicit default model"
        assert "KNOWN_MODEL_DIMS" in src, (
            "embedder must declare the models it knows dimensions for, so a "
            "mismatched vector_size fails before a write"
        )


class TestDocsHonestyGateRuns:
    """The gate script must be runnable and must pass on the committed tree."""

    def test_gate_exits_zero(self):
        import subprocess

        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "check_docs_honesty.py")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, (
            f"docs honesty gate failed (exit {r.returncode}):\n{r.stdout}\n{r.stderr}"
        )


class TestDeclarationParity:
    """manifest <-> tool schemas <-> dispatch must agree, mechanically.

    The peer project's own parity-test docstring is the lesson: "counts
    published across three repositories had drifted to six different numbers
    because nothing checked them". Catalog rule 6 makes a mismatch a security
    issue, so pin it here rather than trusting prose.
    """

    @staticmethod
    def _manifest_tools() -> list[str]:
        import re
        text = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        m = re.search(r"^provides_tools:\n((?:[ \t]+-[ \t]+\S+\n)+)", text, re.M)
        assert m, "provides_tools block not found in plugin.yaml"
        return [ln.strip()[1:].strip() for ln in m.group(1).strip().splitlines()]

    def test_manifest_matches_tool_schemas(self):
        from plugins.memory.qdrant.tool_schemas import ALL_TOOL_SCHEMAS
        declared = set(self._manifest_tools())
        schema = {s["name"] for s in ALL_TOOL_SCHEMAS}
        assert declared == schema, (
            f"plugin.yaml declares {sorted(declared)} but tool_schemas ships "
            f"{sorted(schema)} — catalog rule 6 treats any mismatch as a "
            f"security issue"
        )

    def test_every_schema_name_is_dispatched(self):
        src = _plugin_source()
        from plugins.memory.qdrant.tool_schemas import ALL_TOOL_SCHEMAS
        missing = [s["name"] for s in ALL_TOOL_SCHEMAS
                   if f'== "{s["name"]}"' not in src]
        assert not missing, (
            f"tool schemas without a handle_tool_call branch: {missing} — "
            f"a declared tool that no branch answers would fail at call time"
        )

    def test_no_dispatch_branch_without_a_schema(self):
        import re
        src = _plugin_source()
        branch_names = set(re.findall(r'tool_name == "([a-z_0-9]+)"', src))
        from plugins.memory.qdrant.tool_schemas import ALL_TOOL_SCHEMAS
        declared = {s["name"] for s in ALL_TOOL_SCHEMAS}
        extra = branch_names - declared
        assert not extra, (
            f"handle_tool_call dispatches {sorted(extra)} but no schema "
            f"declares them — the tool would be invisible to the agent"
        )
