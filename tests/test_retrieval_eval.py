"""Offline tests for scripts/retrieval_eval.py.

Imports the harness BY FILE PATH with stdlib only — this proves the module
body stays import-able without the ML stack or the plugin bootstrap (both
are lazy inside ``run()``), so the fast suite never pulls fastembed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "retrieval_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("retrieval_eval", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_is_light_and_metrics_pure():
    """Module import must not drag in qdrant/plugin/ML — CI-budget contract."""
    mod = _load()
    for heavy in ("qdrant_client", "conftest", "plugins", "fastembed"):
        assert not hasattr(mod, heavy), f"module body imported {heavy}"
    for fn in ("recall_at_k", "mrr", "ndcg_at_k", "percentile"):
        assert callable(getattr(mod, fn))


def test_recall_at_k():
    mod = _load()
    ranks = [1, 2, 7, None]
    assert mod.recall_at_k(ranks, 1) == 0.25
    assert mod.recall_at_k(ranks, 5) == 0.50
    assert mod.recall_at_k(ranks, 10) == 0.75
    assert mod.recall_at_k([], 5) == 0.0
    assert mod.recall_at_k([None, None], 5) == 0.0


def test_mrr():
    mod = _load()
    # (1/1 + 1/2 + 1/7 + 0) / 4
    expected = (1.0 + 0.5 + 1 / 7 + 0.0) / 4
    assert abs(mod.mrr([1, 2, 7, None]) - expected) < 1e-9
    assert mod.mrr([None, None]) == 0.0
    assert mod.mrr([]) == 0.0


def test_ndcg_at_k_single_relevant_doc():
    mod = _load()
    # rank 1 -> 1/log2(2) = 1.0 ; rank 2 -> 1/log2(3) ; rank 7 > 5 -> 0
    assert abs(mod.ndcg_at_k([1], 5) - 1.0) < 1e-9
    assert abs(mod.ndcg_at_k([2], 5) - 1 / 1.584962500721156) < 1e-6
    assert mod.ndcg_at_k([7], 5) == 0.0
    assert mod.ndcg_at_k([None], 5) == 0.0


def test_percentile_nearest_rank():
    mod = _load()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert mod.percentile(values, 95) == 50.0
    assert mod.percentile(values, 50) == 30.0
    assert mod.percentile([], 95) == 0.0


def test_parse_hits_matches_tool_output_format():
    mod = _load()
    out = "[0.72] first memory text\n[0.65] second memory text"
    hits = mod._parse_hits(out)
    assert hits == [(0.72, "first memory text"), (0.65, "second memory text")]
    assert mod._parse_hits("No results") == []


def test_dataset_is_sound():
    """Uniqueness and paraphrase shape — a duplicate target would make the
    ranking score meaningless (two identical texts, one 'correct' rank)."""
    mod = _load()
    cases = mod.CASES
    assert len(cases) >= 30, "corpus too small to be a meaningful ranking test"
    memories = [m for m, _ in cases]
    queries = [q for _, q in cases]
    assert len(set(memories)) == len(memories), "duplicate memory text"
    assert len(set(queries)) == len(queries), "duplicate query text"
    for memory, query in cases:
        assert memory != query, "query must rephrase, not repeat, the memory"
        assert query not in memories, "query equals some memory verbatim"
