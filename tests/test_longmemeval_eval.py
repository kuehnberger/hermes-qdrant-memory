"""Offline tests for scripts/longmemeval_eval.py.

Imports the harness BY FILE PATH with stdlib only — the module body must stay
import-able without the ML stack or the plugin bootstrap (both are lazy
inside ``run()``), so the fast suite never pulls fastembed.

The parser tests lock the v2.0.0 result-boundary fix: v1.x parsed the search
output line by line, which mis-scored any hit whose text spans multiple
lines (recall@5/@10 and the miss definition were both wrong).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "longmemeval_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("longmemeval_eval", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_is_light():
    """Module import must not drag in qdrant/plugin/ML — CI-budget contract."""
    mod = _load()
    for heavy in ("qdrant_client", "conftest", "plugins", "fastembed"):
        assert not hasattr(mod, heavy), f"module body imported {heavy}"


def test_parse_results_counts_results_not_lines():
    """A hit whose text spans 40 lines is ONE result at its true rank."""
    mod = _load()
    text_a = "\n".join(f"line {n} of hit one" for n in range(40))
    text_b = "\n".join(f"line {n} of hit two" for n in range(25))
    raw = (
        "[0.85] (pid-a) first hit\n" + text_a + "\n"
        "[0.72] (pid-b) second hit\n" + text_b + "\n"
        "[0.61] (pid-c) third hit\n"
    )
    hits = mod.parse_results(raw)
    assert [h["point_id"] for h in hits] == ["pid-a", "pid-b", "pid-c"]
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert hits[0]["score"] == 0.85
    assert hits[0]["text"] == "first hit"
    # v1.x would have reported the second hit at line 42, not rank 2.


def test_parse_results_ignores_non_header_lines():
    """Continuation lines and prose never become results."""
    mod = _load()
    raw = (
        "some preamble\n"
        "[0.90] (pid-1) hit one\n"
        "continuation with (not-a-pid) in parens\n"
        "[0.50] (pid-2) hit two\n"
        "No results\n"
    )
    hits = mod.parse_results(raw)
    assert [h["point_id"] for h in hits] == ["pid-1", "pid-2"]


def test_parse_results_handles_negative_and_integerish_scores():
    mod = _load()
    raw = "[-0.12] (pid-neg) negative score\n[1.00] (pid-one) perfect\n"
    hits = mod.parse_results(raw)
    assert [h["point_id"] for h in hits] == ["pid-neg", "pid-one"]
    assert hits[0]["score"] == -0.12


def test_parse_results_empty_and_non_string():
    mod = _load()
    assert mod.parse_results("") == []
    assert mod.parse_results("No results") == []


def test_recall_at_k():
    mod = _load()
    ranks = [1, 2, 7, None]
    assert mod.recall_at_k(ranks, 1) == 0.25
    assert mod.recall_at_k(ranks, 5) == 0.50
    assert mod.recall_at_k(ranks, 10) == 0.75


def test_mrr_and_ndcg():
    mod = _load()
    ranks = [1, 3, None]
    assert mod.mrr(ranks) == pytest.approx((1.0 + 1.0 / 3.0) / 3.0)
    # Shipped semantics: IDCG is the query count (every query assumed to have
    # a hit at every rank), so DCG@2 = 1.0 over 3 queries = 1/3.
    assert mod.ndcg_at_k(ranks, 2) == pytest.approx(1.0 / 3.0)


def test_percentile():
    mod = _load()
    assert mod.percentile([1, 2, 3, 4], 50) == 2
    assert mod.percentile([], 95) == 0.0
