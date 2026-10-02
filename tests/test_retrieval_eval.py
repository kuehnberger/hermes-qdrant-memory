"""Offline tests for scripts/retrieval_eval.py.

Imports the harness BY FILE PATH with stdlib only — this proves the module
body stays import-able without the ML stack or the plugin bootstrap (both
are lazy inside ``run()``), so the fast suite never pulls fastembed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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


def test_wilson_interval_brackets_the_point_estimate():
    mod = _load()
    # 31/36 = 0.861 — the interval must contain it and be asymmetric (the
    # normal approximation is badly wrong this close to 1 with n this small).
    lo, hi = mod.wilson_interval(31, 36)
    assert lo < 31 / 36 < hi
    assert hi - lo > 0.10, "a 36-case proportion cannot be this precise"
    # hand-checked: phat=0.86111, denom=1+z^2/n=1.106711, centre=0.826200,
    # half=(z/denom)*sqrt(phat(1-phat)/n + z^2/4n^2)=0.112900
    assert (lo, hi) == pytest.approx((0.7133, 0.9391), abs=5e-4)
    # Perfect and empty are the degenerate ends, still bounded.
    assert mod.wilson_interval(36, 36)[1] == 1.0
    assert mod.wilson_interval(0, 36)[0] == 0.0
    assert mod.wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_narrows_as_n_grows():
    mod = _load()
    narrow = (lambda lo_hi: lo_hi[1] - lo_hi[0])(mod.wilson_interval(310, 360))
    wide = (lambda lo_hi: lo_hi[1] - lo_hi[0])(mod.wilson_interval(31, 36))
    assert narrow < wide, "more cases must give a tighter interval"


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


def test_parse_hits_strips_point_id():
    """Current _tool_search format is '[0.72] (uuid) text'. The ID must be
    removed or every case compares as a miss and recall silently scores 0."""
    mod = _load()
    uid = "3f2b1c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
    out = (f"[0.91] ({uid}) The user drinks espresso in the morning.\n"
           f"[0.40] ({uid}) Another memory.")
    assert mod._parse_hits(out) == [
        (0.91, "The user drinks espresso in the morning."),
        (0.40, "Another memory."),
    ]


def test_parse_hits_accepts_signed_scores():
    """Cosine similarity is negative for distant text; a line the regex
    cannot parse is a silently uncounted miss."""
    mod = _load()
    uid = "3f2b1c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
    assert mod._parse_hits(f"[-0.02] ({uid}) unrelated text") == [
        (-0.02, "unrelated text")
    ]
    assert mod._parse_hits("[-0.02] legacy format without id") == [
        (-0.02, "legacy format without id")
    ]


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


def test_distractors_are_sound():
    """A distractor corpus that leaks (duplicates a target, answers its own
    query's text, or misaligns with CASES) would corrupt the confusion
    metric silently — the pairing IS the metric's axis."""
    mod = _load()
    distractors = mod.DISTRACTORS
    cases = mod.CASES

    assert len(distractors) == len(cases), (
        f"distractors ({len(distractors)}) not index-paired to cases "
        f"({len(cases)}) — the confusion metric depends on the pairing"
    )
    assert len(set(distractors)) == len(distractors), "duplicate distractor"
    assert len(distractors) >= 30, "corpus too small to be meaningful"

    memories = [m for m, _ in cases]
    queries = [q for _, q in cases]
    corpus = set(memories) | set(queries)
    for i, distractor in enumerate(distractors):
        assert distractor not in corpus, (
            f"distractor[{i}] duplicates a memory or query"
        )
        memory, query = cases[i]
        assert distractor != memory, f"distractor[{i}] IS its own target"
        assert distractor not in query and query not in distractor, (
            f"distractor[{i}] shares full text with its paired query"
        )
