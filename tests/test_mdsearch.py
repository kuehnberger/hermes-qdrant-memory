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
    monkeypatch.setattr(
        module, "db_path", lambda: tmp_path / "md-search" / "index.sqlite"
    )
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
            "assert not leaked, f'model modules imported by the "
            "lexical path: {leaked}'\n"
            "print('OK')\n",
            encoding="utf-8",
        )
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
               "FASTEMBED_CACHE_PATH": str(tmp_path / "nowhere")}
        result = subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert result.returncode == 0, (
            f"probe failed:\n{result.stdout}\n{result.stderr}"
        )
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

    def test_no_chunk_is_ever_empty_including_frontmatter_only_files(self, mdsearch):
        """An empty chunk is what killed the first semantic ingest run.

        The real corpus has ~20 ``skills/<category>/DESCRIPTION.md`` files that
        are a YAML block and nothing else. Stripping the frontmatter left an
        empty fallback string, which the embedder rejects outright
        (``cannot embed empty or non-string text``) and which FTS5 matches
        nothing against. The invariant is checked on every degenerate shape,
        not on the one file that happened to crash.
        """
        cases = {
            "skills/x/DESCRIPTION.md": "---\ndescription: Only prose lives here.\n---",
            "skills/x/fm_no_trailing_nl.md": "---\ndescription: Second prose.\n---\n",
            "skills/empty.md": "",
            "skills/whitespace.md": "   \n\t\n",
            "skills/headings.md": "# Only\n\n## Headings\n",
        }
        for label, text in cases.items():
            chunks = mdsearch.chunk_text(text, label)
            assert chunks, f"{label}: produced no chunk at all"
            for heading, content in chunks:
                assert isinstance(content, str) and content.strip(), (
                    f"{label}: empty chunk under heading {heading!r} — "
                    "the embedder raises on this and FTS can never match it"
                )
        # frontmatter-only files keep their prose: it is the only text they have
        _, content = mdsearch.chunk_text(
            cases["skills/x/DESCRIPTION.md"], "skills/x/DESCRIPTION.md")[0]
        assert "Only prose lives here." in content, content


class TestIndexBookkeeping:
    def test_replace_removes_the_previous_rows_for_that_file(self, mdsearch):
        _seed(mdsearch, "skills/a.md", "H1", "first content")
        _seed(mdsearch, "skills/a.md", "H2", "second content")
        hits = mdsearch.search_lexical("first")
        assert not hits, (
            "stale chunks survived a re-index — the file would return dead text"
        )
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
            monkeypatch.setattr(
                "hermes_constants.get_hermes_home", lambda: str(tmp_path / "home")
            )
        except ImportError:
            pass

        resolved = real_module.state_dir().resolve()
        assert (
            real_module.REPO_DIR not in resolved.parents
            and resolved != real_module.REPO_DIR
        ), (
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


class _FakeEmbedder:
    """Docs-model stand-in: a deterministic vector, no weights, no network.

    Keeps fastembed's contract that an empty/non-string chunk is an ERROR —
    the failure that killed the first ingest run must not be papered over by a
    fake that happily embeds anything.
    """

    def encode(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"cannot embed empty or non-string text: {text!r}")
        return [float(len(text)), float(sum(map(ord, text)) % 997), 1.0]


class _FakeDocsClient:
    """The slice of ``QdrantClient`` ``mdsemantic`` uses, backed by a dict.

    It EVALUATES the delete filter instead of merely recording that
    ``delete()`` was called: a call-recorder passes the stale-point gate even
    when the filter matches nothing, and "the removed section is still there"
    is exactly the symptom under test.
    """

    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self._collections: set[str] = set()

    def get_collection(self, name: str):
        if name not in self._collections:
            raise RuntimeError(f"collection {name!r} does not exist")
        from types import SimpleNamespace

        return SimpleNamespace(points_count=len(self.points))

    def create_collection(self, collection_name: str, vectors_config=None, **kwargs):
        self._collections.add(collection_name)

    def upsert(self, *, collection_name: str, points, wait: bool = False) -> None:
        for point in points:
            self.points[str(point.id)] = {
                "payload": dict(point.payload or {}),
                "vector": point.vector,
            }

    def delete(
        self, *, collection_name: str, points_selector, wait: bool = False
    ) -> None:
        must = list(getattr(points_selector.filter, "must", None) or [])
        doomed = [
            pid
            for pid, record in self.points.items()
            if all(
                record["payload"].get(cond.key) == getattr(cond.match, "value", None)
                for cond in must
            )
        ]
        for pid in doomed:
            del self.points[pid]

    def close(self) -> None:
        pass

    def payloads(self, path: str) -> list[dict]:
        """Payloads of every point written for one file label."""
        return [
            record["payload"]
            for record in self.points.values()
            if record["payload"].get("path") == path
        ]


@pytest.fixture
def ingest(tmp_path, monkeypatch):
    """``md_ingest.main()`` against a tmp corpus, tmp index and a fake Qdrant.

    The script is EXECUTED rather than imported as a library: its module-level
    ``sys.path.insert`` + ``from mdsearch import ...`` is the load path production
    uses, and it binds the BARE top-level modules — so those bare instances are
    what has to be patched (the dotted ``plugins.memory.qdrant.*`` copies the
    rest of the suite uses are different module objects).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "md_ingest_under_test", REPO / "scripts" / "md_ingest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import mdsearch as script_mdsearch

    monkeypatch.setattr(script_mdsearch, "state_dir", lambda: tmp_path / "md-search")
    monkeypatch.setattr(
        script_mdsearch, "db_path", lambda: tmp_path / "md-search" / "index.sqlite"
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setattr(
        script_mdsearch, "DEFAULT_ROOTS", {"skills": (str(corpus), set())}
    )
    monkeypatch.setattr(module, "db_path", script_mdsearch.db_path)

    store = _FakeDocsClient()

    import mdsemantic as script_mdsemantic

    monkeypatch.setattr(script_mdsemantic, "make_embedder", _FakeEmbedder)

    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: store)

    class IngestEnv:
        def __init__(self) -> None:
            self.store = store
            self.corpus = corpus
            self.index = tmp_path / "md-search" / "index.sqlite"

        def write(self, name: str, text: str) -> Path:
            path = self.corpus / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return path

        def run(self, *args: str) -> None:
            rc = module.main(["--semantic", "--root", "skills", *args])
            assert rc == 0, f"md_ingest.main exited {rc}"

        def _rows(self, sql: str, params: tuple = ()) -> list:
            con = sqlite3.connect(str(self.index))
            try:
                return con.execute(sql, params).fetchall()
            finally:
                con.close()

        def fts_rows(self, label: str) -> list:
            return self._rows(
                "SELECT heading, content FROM fts WHERE path = ?", (label,)
            )

        def all_fts_rows(self) -> list:
            return self._rows("SELECT path, heading, content FROM fts")

    return IngestEnv()


class TestSemanticTierConsistency:
    """The two defects the 2026-10-03 ingest run found, fixed here.

    Both are invisible to any lexical check — they live entirely on the Qdrant
    side, which is why they shipped: ``count(fts)`` was green while 8.1% of
    chunks were missing from the vector tier. So every gate below drives the
    REAL ``md_ingest.main()`` end to end and inspects the resulting store
    against the FTS rows, rather than asserting on either tier alone.
    """

    def test_duplicate_heading_chains_yield_two_points(self, ingest):
        """``(path, heading)`` is not a unique key — 801 files prove it.

        Two sections in one file can carry the same heading path. They shared
        one point id, so the last write won and the other chunk silently
        vanished from the vector tier while staying queryable in FTS5: 4,211
        chunks (8.1%) on the real corpus.
        """
        label = "skills/dup.md"
        ingest.write(
            "dup.md",
            "# Top\n\n## Setup\n\nfirst body\n\n## Other\n\nmiddle\n\n"
            "## Setup\n\nsecond body\n",
        )
        ingest.run()

        rows = ingest.fts_rows(label)
        headings = [h for h, _ in rows]
        assert headings.count("Top > Setup") == 2, (
            f"the fixture must contain a duplicate heading chain or this gate "
            f"proves nothing; got {headings}"
        )
        stored = ingest.store.payloads(label)
        assert len(stored) == len(rows), (
            f"the vector tier holds {len(stored)} point(s) for {len(rows)} "
            "FTS chunks — duplicate heading chains are still sharing one "
            "point id"
        )

    def test_removing_a_section_leaves_no_point_with_the_old_sha(self, ingest):
        """Delete-before-write: an upsert cannot remove what it no longer has.

        ``upsert_chunks`` overwrites ids that still exist and leaves the rest,
        so without a delete of the file's points first, every section removed
        by an edit keeps serving hits (with the old ``sha``) forever.
        """
        label = "skills/edit.md"
        ingest.write(
            "edit.md",
            "# One\n\nalpha\n\n## Two\n\nbeta\n\n## Three\n\ngamma\n",
        )
        ingest.run()
        before = ingest.store.payloads(label)
        assert len(before) == 3, before
        old_shas = {p["sha"] for p in before}
        assert len(old_shas) == 1, old_shas

        # Remove section "Three" — the file's sha changes, so this takes the
        # incremental path (no --rebuild), which is the path that regressed.
        ingest.write("edit.md", "# One\n\nalpha\n\n## Two\n\nbeta\n")
        ingest.run()

        stale = [p for p in ingest.store.payloads(label) if p["sha"] in old_shas]
        assert not stale, (
            f"{len(stale)} point(s) still carry sha {sorted(old_shas)[0][:12]}… "
            "after the section was removed — delete-before-write is missing, "
            "so dead text keeps serving vector hits"
        )
        after = ingest.store.payloads(label)
        assert len(after) == len(ingest.fts_rows(label)) == 2, after

    def test_pruning_a_deleted_file_removes_its_points(self, ingest):
        """``--prune`` used to clear FTS only; the vector tier is unverified."""
        label = "skills/gone.md"
        ingest.write("gone.md", "# Gone\n\nvanishing text\n")
        ingest.run()
        assert ingest.store.payloads(label), "fixture produced no points"

        (ingest.corpus / "gone.md").unlink()
        ingest.run("--prune")

        assert not ingest.fts_rows(label), "prune left the FTS rows behind"
        assert not ingest.store.payloads(label), (
            "--prune dropped the FTS rows but left the vector points: a "
            "deleted file would keep serving semantic hits"
        )

    def test_rebuilding_an_unchanged_file_keeps_the_same_point_ids(self, ingest):
        """Ids must be DERIVED, not regenerated: same bytes -> same ids.

        Asserted on the id set rather than the count, because delete-before-write
        makes the count look right even for random ids — churn shows up as every
        point being rewritten (and re-indexed) on each run.
        """
        ingest.write("stable.md", "# A\n\none body\n\n## B\n\ntwo body\n")
        ingest.run()
        before = set(ingest.store.points)
        assert before, "fixture produced no points"

        ingest.run("--rebuild")

        assert set(ingest.store.points) == before, (
            "point ids changed when the bytes did not — ids must be a pure "
            "function of the chunk's position, not regenerated per run"
        )
        assert len(ingest.store.points) == len(
            ingest.all_fts_rows()
        ), "a rebuild must leave one point per FTS chunk"


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
            f"register() must call ctx.register_tool('md_search', ...); "
            f"recorded {names}"
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
        block = re.search(
            r"^provides_tools:\n((?:[ \t]+-[ \t]+\S+\n)+)", manifest, re.M
        )
        declared = {
            ln.strip()[1:].strip() for ln in block.group(1).strip().splitlines()
        }
        from plugins.memory.qdrant.tool_schemas import ALL_TOOL_SCHEMAS

        assert declared == {s["name"] for s in ALL_TOOL_SCHEMAS}
        assert "md_search" in declared
