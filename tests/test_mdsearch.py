"""Gates for the KNOWLEDGE INDEX feature (``md_search``).

Each test here corresponds to a claim the card makes or a failure mode this
feature could ship with:

* the lexical fast path must not load an embedding model (the whole reason for
  FTS5-first; a regression here is invisible until someone's RSS jumps 240 MB);
* the docs collection must stay SEPARATE from ``hermes_memories`` and carry no
  session scope;
* index state must never be written into the plugin member dir (the venv-stamp
  rule);
* chunking must be heading-aware and the FTS query must be injection-safe.

Nothing here needs a Qdrant server or a model: the semantic tier is exercised
through stubs, so the suite stays fast and hermetic.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def mdsearch(tmp_path, monkeypatch):
    """``mdsearch`` with its index redirected into a tmp state dir."""
    from plugins.memory.qdrant import mdsearch as module

    monkeypatch.setattr(module, "state_dir", lambda: tmp_path / "md-search")
    monkeypatch.setattr(module, "db_path", lambda: tmp_path / "md-search" / "index.sqlite")
    yield module
    try:
        module.connect().close()
    except Exception:
        pass


def _seed(module, label, heading, content, sha="deadbeef", root="skills"):
    con = module.connect()
    try:
        module.replace_file_chunks(con, label, root, sha, [(heading, content)])
        con.commit()
    finally:
        con.close()


class TestFtsPathLoadsNoModel:
    """The FTS5 fast path must never pull in an embedding backend.

    Asserted two ways, because either alone is weak: an import-graph check
    catches a future ``import mdsemantic`` at the top of ``mdsearch``, and a
    live check catches an import hidden inside a function that only runs on some
    branch.
    """

    def test_mdsearch_module_does_not_import_the_semantic_tier(self):
        src = (REPO / "mdsearch.py").read_text(encoding="utf-8")
        assert "mdsemantic" not in src.replace(
            "``mdsemantic`` is imported", ""  # the docstring names it; code must not
        ) or "import mdsemantic" not in src and "from . import mdsemantic" not in src, (
            "mdsearch.py must not import the semantic tier — the lexical path has "
            "to stay model-free"
        )

    def test_lexical_search_in_a_clean_interpreter_loads_no_model(self, tmp_path):
        """A real subprocess: import mdsearch, run a lexical query, assert clean."""
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "import mdsearch\n"
            "con = mdsearch.connect()\n"
            "mdsearch.replace_file_chunks(con, 'skills/a.md', 'skills', 'sha', "
            "    [('Head', 'the quick brown fox jumps')])\n"
            "con.commit()\n"
            "hits = mdsearch.search_lexical('fox')\n"
            "assert hits, 'lexical search returned nothing for its own seeded chunk'\n"
            "leaked = [m for m in sys.modules if m in "
            "('fastembed', 'onnxruntime', 'sentence_transformers', 'mdsemantic')]\n"
            "assert not leaked, f'model modules imported by the lexical path: {leaked}'\n"
            "print('OK')\n",
            encoding="utf-8",
        )
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
               "FASTEMBED_CACHE_PATH": str(tmp_path / "nowhere")}
        result = subprocess.run(
            [sys.executable, str(probe)], capture_output=True, text=True, timeout=180, env=env
        )
        assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
        assert "OK" in result.stdout

    def test_fts_tokenizer_is_diacritic_insensitive(self, mdsearch):
        """An ASCII query must find text carrying real umlauts.

        NOTE what this cannot prove: ``remove_diacritics 1`` is already
        unicode61's DEFAULT, so mutating our DDL to plain ``unicode61`` still
        matches "munchen" -> "München" (measured, sqlite 3.53.1). The observable
        folding behaviour is therefore pinned by the assertions below, and the
        exact tokenizer string by ``test_fts_table_declares_the_expected_tokenizer``,
        which asks sqlite_master rather than reading our own source.
        """
        _seed(mdsearch, "skills/greet.md", "Begrüßung", "Viele Grüße aus München")
        hits = mdsearch.search_lexical("munchen")
        assert hits and hits[0].path == "skills/greet.md", (
            "an ASCII query must find German text with umlauts; got " + repr(hits)
        )
        # And the accented spelling must still work — folding must not break it.
        assert mdsearch.search_lexical("München"), (
            "diacritic folding must not stop the accented spelling from matching"
        )

    def test_fts_table_declares_the_expected_tokenizer(self, mdsearch):
        """Read the tokenizer back out of sqlite_master.

        Asking the DATABASE beats grepping our source: this fails if the DDL
        changes, and it cannot be fooled by the same string appearing in a
        docstring (which is exactly how the first version of this check passed
        against unmutated code).
        """
        con = mdsearch.connect()
        try:
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE name='ftsx'"
            ).fetchone()
            assert row and row[0], "ftsx table not found in sqlite_master"
            assert "unicode61" in row[0], row[0]
            assert "remove_diacritics 2" in row[0], (
                "the FTS5 table must be created with "
                f"'unicode61 remove_diacritics 2'; got: {row[0]}"
            )
        finally:
            con.close()


class TestQuerySafety:
    def test_fts_query_quotes_operators_so_they_are_data(self, mdsearch):
        assert mdsearch.fts_query('foo AND "bar') == '"foo" OR "AND" OR "bar"'

    def test_fts_query_is_none_for_punctuation_only(self, mdsearch):
        assert mdsearch.fts_query("*** ??") is None

    def test_punctuation_only_query_returns_nothing_not_an_error(self, mdsearch):
        _seed(mdsearch, "skills/a.md", "H", "content here")
        assert mdsearch.search_lexical("***") == []


class TestChunking:
    def test_headings_become_a_breadcrumb(self, mdsearch):
        text = "# Top\n\nintro\n\n## Middle\n\nbody\n\n### Deep\n\nmore\n"
        chunks = mdsearch.chunk_text(text, "skills/a.md")
        headings = [h for h, _ in chunks]
        assert "Top > Middle > Deep" in headings, headings

    def test_frontmatter_is_stripped(self, mdsearch):
        text = "---\ntitle: Secret Key Name\ntags: [x]\n---\n\n# H\n\nreal body\n"
        _, content = mdsearch.chunk_text(text, "skills/a.md")[0]
        assert "Secret Key Name" not in content, (
            "frontmatter keys would otherwise outrank real prose for keyword queries"
        )

    def test_oversized_section_splits_into_multiple_chunks(self, mdsearch):
        para = "word " * 200
        text = "# H\n\n" + "\n\n".join([para] * 5)
        chunks = mdsearch.chunk_text(text, "skills/a.md")
        assert len(chunks) > 1
        assert all(len(c) <= mdsearch.MAX_CHUNK_CHARS + 200 for _, c in chunks)

    def test_headings_only_file_still_yields_one_chunk(self, mdsearch):
        """A file of only headings must stay findable, not vanish from the index."""
        chunks = mdsearch.chunk_text("# Only\n\n## Headings\n", "skills/empty.md")
        assert len(chunks) == 1


class TestIndexBookkeeping:
    def test_replace_removes_the_previous_rows_for_that_file(self, mdsearch):
        _seed(mdsearch, "skills/a.md", "H1", "first content")
        _seed(mdsearch, "skills/a.md", "H2", "second content")
        hits = mdsearch.search_lexical("first")
        assert not hits, "stale chunks survived a re-index — the file would return dead text"
        assert mdsearch.search_lexical("second")

    def test_root_filter_narrows_results(self, mdsearch):
        _seed(mdsearch, "skills/a.md", "H", "shared word", root="skills")
        _seed(mdsearch, "vault/b.md", "H", "shared word", root="vault")
        assert len(mdsearch.search_lexical("shared")) == 2
        only_vault = mdsearch.search_lexical("shared", root="vault")
        assert len(only_vault) == 1 and only_vault[0].path == "vault/b.md"

    def test_rebuild_fts_restores_the_searchable_copy(self, mdsearch):
        _seed(mdsearch, "skills/a.md", "H", "recoverable text")
        con = mdsearch.connect()
        try:
            con.execute("DELETE FROM ftsx")
            con.commit()
            assert not mdsearch.search_lexical("recoverable", con=con)
            mdsearch.rebuild_fts_from(con)
            assert mdsearch.search_lexical("recoverable", con=con), (
                "rebuild_fts_from must restore bm25 searchability from fts alone"
            )
        finally:
            con.close()

    def test_index_lives_outside_the_plugin_member_dir(self, tmp_path, monkeypatch):
        """The venv-stamp rule, asserted on the REAL path function.

        The earlier version monkeypatched ``db_path`` to a tmp dir and then
        asserted the result was not inside the repo — a tautology that passed
        even with the production path pointing straight at the member dir. This
        one calls the unpatched ``state_dir()`` and inspects where it lands.
        """
        from plugins.memory.qdrant import mdsearch as real_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        try:
            from hermes_constants import get_hermes_home

            monkeypatch.setattr(
                "hermes_constants.get_hermes_home", lambda: str(tmp_path / "home")
            )
        except ImportError:
            pass

        resolved = real_module.state_dir().resolve()
        assert real_module.REPO_DIR not in resolved.parents and resolved != real_module.REPO_DIR, (
            f"index state would live at {resolved}, inside the plugin member dir — "
            "Hermes hashes every file there into the workspace dependency stamp"
        )
        assert resolved.name == "md-search", resolved

    def test_state_dir_is_under_state_not_the_plugin_dir(self, tmp_path, monkeypatch):
        """The state path, read straight off the unpatched module.

        No reload: ``state_dir()`` reads ``hermes_constants.get_hermes_home()``
        at CALL time, so setting HERMES_HOME is enough and reloading the module
        would only fight the ``mdsearch`` fixture's own monkeypatching.
        """
        from plugins.memory.qdrant import mdsearch as real_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        resolved = real_module.state_dir().resolve()
        assert resolved.parts[-2:] == ("state", "md-search"), resolved
        assert real_module.db_path().name == "index.sqlite"


class TestCollectionSeparation:
    """``hermes_md_docs`` must never become ``hermes_memories``."""

    def test_docs_collection_is_a_distinct_name(self):
        from plugins.memory.qdrant import mdsemantic

        assert mdsemantic.DOCS_COLLECTION == "hermes_md_docs", (
            f"docs collection is {mdsemantic.DOCS_COLLECTION!r}, expected "
            "'hermes_md_docs' — docs and memories are separate corpora"
        )
        assert mdsemantic.DOCS_COLLECTION != "hermes_memories", (
            "docs collection must not be the memory collection: memories are "
            "session-scoped, docs are shared"
        )

    def test_docs_model_differs_from_the_memory_default_model(self):
        from plugins.memory.qdrant import embedder, mdsemantic

        assert mdsemantic.DOCS_MODEL != embedder.DEFAULT_MODEL, (
            "the docs model must differ from the memory provider's default; equal "
            "names would let doc vectors be written into the memory space"
        )

    def test_docs_payload_keys_are_exactly_the_agreed_set(self):
        """Behavioural, not textual: build the payload and inspect its keys.

        A source-substring test on this property passes the moment someone
        reformats the dict literal; the mutation check caught exactly that.
        """
        from plugins.memory.qdrant import mdsemantic

        payload = mdsemantic.payload_for("skills/a.md", "Heading", "sha123")
        assert set(payload) == {"path", "heading", "root", "sha"}, payload
        assert payload["root"] == "skills", (
            "root is the first path segment, so the scope filter can use it"
        )

    def test_docs_payload_has_no_session_scope(self):
        """Docs are shared; a session filter would hide the index from most sessions."""
        from plugins.memory.qdrant import mdsemantic

        payload = mdsemantic.payload_for("skills/a.md", "H", "sha")
        assert "session_id" not in payload, (
            f"the docs payload must carry no session scope, got {sorted(payload)}"
        )
        # Also assert no session ARGUMENT is threaded anywhere in the write/read
        # surface — a caller could filter on it even if the payload lacks it.
        import inspect

        for fn in (mdsemantic.upsert_chunks, mdsemantic.search_semantic,
                   mdsemantic.delete_file_points):
            sig = inspect.signature(fn)
            assert not any("session" in p.lower() for p in sig.parameters), (
                f"{fn.__name__}{sig} takes a session parameter; docs are shared"
            )


class TestIngestCli:
    def test_ingest_script_compiles(self):
        script = REPO / "scripts" / "md_ingest.py"
        assert script.is_file()
        compile(script.read_text(encoding="utf-8"), str(script), "exec")

    def test_ingest_status_exits_zero_on_an_empty_index(self, tmp_path):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
               "HERMES_HOME": str(tmp_path / "home")}
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "md_ingest.py"), "--status"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        assert result.returncode == 0, result.stderr

    def test_unknown_root_is_rejected_with_a_known_roots_list(self, tmp_path):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
               "HERMES_HOME": str(tmp_path / "home")}
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "md_ingest.py"), "--root", "nope"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        assert result.returncode == 2
        assert "known:" in result.stderr


class TestToolWiring:
    def test_md_search_is_dispatched_by_handle_tool_call(self):
        src = (REPO / "__init__.py").read_text(encoding="utf-8")
        assert 'tool_name == "md_search"' in src, (
            "a declared tool with no dispatch branch fails only at call time"
        )

    def test_schema_declares_the_parameters_the_handler_reads(self):
        from plugins.memory.qdrant.tool_schemas import MD_SEARCH_SCHEMA

        props = set(MD_SEARCH_SCHEMA["parameters"]["properties"])
        assert props == {"query", "limit", "root", "semantic"}
        assert MD_SEARCH_SCHEMA["parameters"]["required"] == ["query"]

    def test_missing_query_is_an_error_string_not_an_exception(self):
        from plugins.memory.qdrant import QdrantMemoryProvider

        out = QdrantMemoryProvider().handle_tool_call("md_search", {})
        assert "missing 'query'" in out

    def test_register_also_registers_the_tool_as_a_plugin_tool(self):
        """The provider-independence path, exercised for real.

        A source-substring test for ``register_tool`` inside ``register()`` passed
        even after the call was renamed away — the mutation check caught it. This
        calls the real ``register()`` against a recording context and asserts on
        what was actually registered.
        """
        import sys

        # ``from plugins.memory.qdrant import __init__`` yields the PACKAGE
        # OBJECT, not the module (dunder names are not submodule attributes), so
        # read it out of sys.modules where the loader actually put it.
        plugin = sys.modules["plugins.memory.qdrant"]

        recorded: list[tuple] = []

        class Ctx:
            plugin_config: dict = {}
            profile_name = "default"
            plugin_id = "qdrant_test_probe"

            def register_memory_provider(self, provider):
                self.provider = provider

            def register_tool(self, name, *args, **kwargs):
                recorded.append((name, kwargs.get("schema") or {}))

        plugin.register(Ctx())
        names = [n for n, _ in recorded]
        assert "md_search" in names, (
            f"register() must call ctx.register_tool('md_search', ...); recorded {names}"
        )
        schema = dict(recorded)[ "md_search" ]
        assert schema.get("name") == "md_search"
        assert "parameters" in schema, "the registered schema must carry parameters"

    def test_manifest_kind_is_standalone_so_the_tool_outlives_the_provider(self):
        """The delivery gate, as a standing rule.

        Under ``kind: exclusive`` the general PluginManager skips this plugin, so
        ``register()`` runs only on the memory-activation path and ``md_search``
        disappears when ``memory.provider`` points elsewhere. Measured across all
        four kind x provider combinations before this rule was written; the row
        that justifies it is (exclusive, other provider) -> not served.
        """
        manifest = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        kind = re.search(r"^kind:\s*(\S+)", manifest, re.M)
        assert kind, "plugin.yaml declares no kind"
        assert kind.group(1) == "standalone", (
            f"plugin.yaml kind is {kind.group(1)!r}; md_search is registered via "
            "ctx.register_tool, which the general PluginManager only reaches for "
            "a non-exclusive plugin — reverting to 'exclusive' silently makes the "
            "tool provider-dependent again"
        )

    def test_memory_activation_is_not_broken_by_the_kind_flip(self):
        """The flip must not cost the provider its activation path.

        ``plugins/memory`` has its own scanner (``_is_memory_provider_dir`` looks
        for the string "MemoryProvider" in __init__.py) and does not consult the
        manifest ``kind`` key, so find_provider_dir still resolves this dir.
        """
        src = (REPO / "__init__.py").read_text(encoding="utf-8")
        assert "class QdrantMemoryProvider" in src, (
            "the provider class must keep its name: plugins/memory's "
            "_is_memory_provider_dir greps __init__.py for 'MemoryProvider', so "
            "renaming it would make the plugin undiscoverable as a provider"
        )
        assert "register_memory_provider" in src, (
            "register() must still call ctx.register_memory_provider — memory "
            "activation runs through this call, not through plugin discovery"
        )

    def test_every_registered_tool_is_declared_in_the_manifest(self):
        """Declaration parity both ways — catalog rule 6."""
        manifest = (REPO / "plugin.yaml").read_text(encoding="utf-8")
        block = re.search(r"^provides_tools:\n((?:[ \t]+-[ \t]+\S+\n)+)", manifest, re.M)
        declared = {ln.strip()[1:].strip() for ln in block.group(1).strip().splitlines()}
        from plugins.memory.qdrant.tool_schemas import ALL_TOOL_SCHEMAS

        assert declared == {s["name"] for s in ALL_TOOL_SCHEMAS}
        assert "md_search" in declared