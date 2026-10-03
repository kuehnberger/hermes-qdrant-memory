#!/usr/bin/env python
"""Mutation check: revert each fix and confirm its gate FAILS with the expected symptom.

A gate that has never failed is not evidence. This script works on a throwaway
COPY of the repo, applies one mutation, runs the specific gate, and requires a
non-zero exit WITH the expected symptom.

This script has already earned its keep: the first run found five gates that
passed against a deliberately broken source (a tautological member-dir
assertion, and three source-substring tests that only matched their own prose).
Keep every entry here — a mutation that stops being caught is a regression in
the gate, not in the code.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent

# (name, file, old, new, test node, expected substring in the failure)
MUTATIONS = [
    (
        "sql-table-alias-regression",
        "mdsearch.py",
        'SELECT ftsx.path, ftsx.heading, ftsx.content, bm25(ftsx) AS score "\n'
        '            "FROM ftsx JOIN files m ON m.path = ftsx.path "\n'
        '            "WHERE ftsx MATCH ? "',
        'SELECT f.path, f.heading, f.content, bm25(f) AS score "\n'
        '            "FROM ftsx AS f JOIN files m ON m.path = f.path "\n'
        '            "WHERE f MATCH ? "',
        "tests/test_mdsearch.py::TestIndexBookkeeping::test_root_filter_narrows_results",
        "no such column",
    ),
    (
        "collapse-docs-into-memories-collection",
        "mdsemantic.py",
        # Anchor on the ASSIGNMENT, not the module docstring that mentions the
        # same string — the first version of this mutation hit the docstring and
        # changed nothing, so the gate "passed" against unmutated code.
        '\nDOCS_COLLECTION = "hermes_md_docs"\n',
        '\nDOCS_COLLECTION = "hermes_memories"\n',
        "tests/test_mdsearch.py::TestCollectionSeparation::test_docs_collection_is_a_distinct_name",
        "expected 'hermes_md_docs'",
    ),
    (
        "swap-docs-model-to-the-memory-default",
        "mdsemantic.py",
        '\nDOCS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"\n',
        '\nDOCS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"\n',
        "tests/test_mdsearch.py::TestCollectionSeparation::test_docs_model_differs_from_the_memory_default_model",
        "must differ",
    ),
    (
        "add-a-session-key-to-the-docs-payload",
        "mdsemantic.py",
        '    return {\n        "path": label,\n        "heading": heading,',
        '    return {\n        "session_id": "default",\n        "path": label,\n        "heading": heading,',
        "tests/test_mdsearch.py::TestCollectionSeparation::test_docs_payload_has_no_session_scope",
        "no session scope",
    ),
    (
        "index-into-the-plugin-member-dir",
        "mdsearch.py",
        '    return home / "state" / "md-search"',
        '    return home / "plugins" / "qdrant"',
        "tests/test_mdsearch.py::TestIndexBookkeeping::test_index_lives_outside_the_plugin_member_dir",
        "member dir",
    ),
    (
        "state-dir-not-under-state",
        "mdsearch.py",
        '    return home / "state" / "md-search"',
        '    return home / "md-search"',
        "tests/test_mdsearch.py::TestIndexBookkeeping::test_state_dir_is_under_state_not_the_plugin_dir",
        "md-search",
    ),
    (
        "stop-declaring-md-search-in-the-manifest",
        "plugin.yaml",
        "  - qdrant_forget\n  - md_search\n",
        "  - qdrant_forget\n",
        "tests/test_mdsearch.py::TestToolWiring::test_every_registered_tool_is_declared_in_the_manifest",
        "md_search",
    ),
    (
        "stop-registering-md-search-as-a-plugin-tool",
        "__init__.py",
        '            ctx.register_tool(\n                name="md_search",',
        '            _discarded(\n                name="md_search",',
        "tests/test_mdsearch.py::TestToolWiring::test_register_also_registers_the_tool_as_a_plugin_tool",
        "must call ctx.register_tool",
    ),
    (
        "drop-the-md-search-dispatch-branch",
        "__init__.py",
        '        if tool_name == "md_search":\n            return self._tool_md_search(args)\n',
        "",
        "tests/test_mdsearch.py::TestToolWiring::test_md_search_is_dispatched_by_handle_tool_call",
        "md_search",
    ),
    (
        "stop-removing-diacritics-in-the-tokenizer",
        "mdsearch.py",
        "tokenize='unicode61 remove_diacritics 2'",
        "tokenize='unicode61'",
        "tests/test_mdsearch.py::TestFtsPathLoadsNoModel::test_fts_table_declares_the_expected_tokenizer",
        "remove_diacritics 2",
    ),
    (
        "make-fts-query-raw-unquoted-sql",
        "mdsearch.py",
        "    return \" OR \".join('\"' + t.replace('\"', '\"\"') + '\"' for t in tokens)",
        "    return \" OR \".join(t for t in tokens)",
        "tests/test_mdsearch.py::TestQuerySafety::test_fts_query_quotes_operators_so_they_are_data",
        "OR",
    ),
    (
        "stop-stripping-frontmatter",
        "mdsearch.py",
        '    text = _FRONTMATTER.sub("", text, count=1)',
        "    pass",
        "tests/test_mdsearch.py::TestChunking::test_frontmatter_is_stripped",
        "Secret Key Name",
    ),
    (
        "stop-replacing-stale-chunks-on-reindex",
        "mdsearch.py",
        '    con.execute("DELETE FROM ftsx WHERE path = ?", (label,))\n'
        '    con.execute("DELETE FROM fts WHERE path = ?", (label,))',
        "    pass",
        "tests/test_mdsearch.py::TestIndexBookkeeping::test_replace_removes_the_previous_rows_for_that_file",
        "stale chunks survived",
    ),
    (
        "let-the-semantic-tier-import-the-lexical-module",
        "mdsearch.py",
        "import sqlite3",
        "import sqlite3\nimport mdsemantic  # noqa: F401",
        "tests/test_mdsearch.py::TestFtsPathLoadsNoModel::test_mdsearch_module_does_not_import_the_semantic_tier",
        "must not import",
    ),
    # --- the two defects this card exists to fix, plus their two neighbours ---
    # Anchors are the assignments/calls themselves: the id scheme is discussed
    # at length in the docstring directly above it, and a mutation that only
    # edited prose would leave every gate green.
    (
        "collapse-duplicate-heading-ids",
        "mdsemantic.py",
        "    return str(uuid.uuid5(_ID_NAMESPACE, "
        'f"{label}\\x00{heading}\\x00{ordinal}"))',
        '    return str(uuid.uuid5(_ID_NAMESPACE, f"{label}\\x00{heading}"))',
        "tests/test_mdsearch.py::TestSemanticTierConsistency::test_duplicate_heading_chains_yield_two_points",
        "still sharing one point id",
    ),
    (
        "regenerate-point-ids-on-every-run",
        "mdsemantic.py",
        "    return str(uuid.uuid5(_ID_NAMESPACE, "
        'f"{label}\\x00{heading}\\x00{ordinal}"))',
        "    return str(uuid.uuid4())",
        "tests/test_mdsearch.py::TestSemanticTierConsistency::test_rebuilding_an_unchanged_file_keeps_the_same_point_ids",
        "point ids changed when the bytes did not",
    ),
    (
        "skip-delete-before-write-on-a-changed-file",
        "scripts/md_ingest.py",
        "                semantic.delete_file_points(client, cf.label)",
        "                pass  # mutation: no delete-before-write",
        "tests/test_mdsearch.py::TestSemanticTierConsistency::test_removing_a_section_leaves_no_point_with_the_old_sha",
        "delete-before-write is missing",
    ),
    (
        "prune-clears-fts-but-not-the-vector-points",
        "scripts/md_ingest.py",
        "                    semantic.delete_file_points(client, label)",
        "                    pass  # mutation: prune spares the vector tier",
        "tests/test_mdsearch.py::TestSemanticTierConsistency::test_pruning_a_deleted_file_removes_its_points",
        "left the vector points",
    ),
]


def main() -> int:
    problems: list[str] = []
    for name, relfile, old, new, test_node, expect in MUTATIONS:
        work = Path(tempfile.mkdtemp(prefix=f"mut-{name}-"))
        dest = work / "repo"
        shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns(
            "__pycache__", ".git", ".venv", ".pytest_cache", "uv.lock"))
        target = dest / relfile
        text = target.read_text(encoding="utf-8")
        if old not in text:
            problems.append(f"{name}: MUTATION DID NOT APPLY (anchor missing in {relfile})")
            shutil.rmtree(work, ignore_errors=True)
            continue
        target.write_text(text.replace(old, new, 1), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_node, "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=dest, capture_output=True, text=True, timeout=300,
            env={"PATH": "/usr/bin:/bin", "HOME": str(work / "home"),
                 "FASTEMBED_CACHE_PATH": str(work / "modelcache")},
        )
        out = proc.stdout + proc.stderr
        if proc.returncode == 0:
            problems.append(f"{name}: GATE STILL PASSED — it does not guard this")
        elif expect not in out:
            problems.append(
                f"{name}: gate failed but not with the expected symptom (wanted {expect!r})"
            )
        else:
            print(f"OK   {name}: gate failed as expected")
        shutil.rmtree(work, ignore_errors=True)

    print()
    if problems:
        print(f"FAIL — {len(problems)} mutation(s) not caught:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"mutation check: all {len(MUTATIONS)} mutations caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
