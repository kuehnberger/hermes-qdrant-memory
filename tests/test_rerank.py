"""Gates for the opt-in cross-encoder rerank arm (``rerank``).

Each test here pins a claim the arm makes, and the claims are all of one kind:
reranking may improve the order, but it must never be able to break it.

* it reorders by CROSS-ENCODER score and says so — the fused score is gone, so
  nothing downstream may compare the two;
* it is lazy: constructing a ``Reranker`` and asking ``available()`` must not
  import the ML stack, build a model, or call the loader;
* every failure (loader raises, model raises, wrong score count, non-finite or
  non-numeric score) degrades to the ORIGINAL list — same order, same objects —
  because an optional arm that can take recall down is worse than no arm;
* empty and single-element inputs never touch the loader;
* ties keep their incoming order, and scores leave as builtin ``float``.

Nothing here needs a network, a Qdrant server, or an ONNX model: the loader is
the injection seam, so every test drives a scripted fake. The one test that
touches real fastembed (``TestRealLoaderWiring``) only reads its static model
registry and an unsupported-name ``ValueError``, both of which happen before
any download.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
from plugins.memory.qdrant import rerank as rerank_module

REPO = Path(__file__).resolve().parent.parent
LOGGER = "hermes.plugins.memory.qdrant.rerank"

#: The one literal the interface promises. Pinned so a "harmless" rename of the
#: default checkpoint cannot ship silently.
EXPECTED_DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class FakeModel:
    """Stands in for a fastembed ``TextCrossEncoder``.

    Records the ``(query, documents)`` it was handed so a test can prove what
    the arm actually sent, and yields scripted scores — one per document, in
    document order, exactly like the real one.
    """

    def __init__(self, scores: Iterable[float]) -> None:
        self._scores = list(scores)
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: Iterable[str]) -> Iterator[float]:
        docs = list(documents)
        self.calls.append((query, docs))
        yield from self._scores


class ExplodingModel:
    """A model that fails during scoring rather than at load time."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def rerank(self, query: str, documents: Iterable[str]) -> Iterator[float]:
        yield from ()
        raise self._error


class ReturningModel:
    """A model whose ``rerank`` returns a value instead of an iterable."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def rerank(self, query: str, documents: Iterable[str]) -> Any:
        return self._value


class RecordingLoader:
    """The injection seam: ``(model_name, cache_dir) -> model``, with a log."""

    def __init__(self, model: Any = None, error: BaseException | None = None) -> None:
        self.model = model
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, model_name: str, cache_dir: str | None) -> Any:
        self.calls.append((model_name, cache_dir))
        if self.error is not None:
            raise self.error
        return self.model


PAIRS: list[tuple[float, str]] = [(0.9, "alpha"), (0.8, "beta"), (0.7, "gamma")]


def _reranker(model: Any, **kwargs: Any) -> tuple[Any, RecordingLoader]:
    loader = RecordingLoader(model)
    return rerank_module.Reranker(loader=loader, **kwargs), loader


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_reorders_by_descending_cross_encoder_score(self):
        """RRF's order is replaced by the cross-encoder's, best first."""
        reranker, _loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        result = reranker.rerank("q", list(PAIRS))
        assert result == [(0.9, "beta"), (0.5, "gamma"), (0.1, "alpha")]

    def test_the_returned_score_is_the_cross_encoder_score(self):
        """Not the fused score — the incoming scores are gone, deliberately."""
        reranker, _loader = _reranker(FakeModel([0.25, 0.25, 0.25]))
        result = reranker.rerank("q", list(PAIRS))
        assert [score for score, _text in result] == [0.25, 0.25, 0.25]
        assert [0.9, 0.8, 0.7] not in [[s for s, _t in result]]

    def test_returns_a_new_list_and_leaves_the_input_untouched(self):
        reranker, _loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        pairs = list(PAIRS)
        result = reranker.rerank("q", pairs)
        assert result is not pairs
        assert pairs == PAIRS, "rerank must not mutate its input"

    def test_sends_the_query_and_the_pair_texts_in_order(self):
        """Documents are the ``text`` of each pair, and the query goes with them."""
        reranker, _loader = _reranker(FakeModel([0.1, 0.2, 0.3]))
        reranker.rerank("who wrote it", list(PAIRS))
        assert _loader.model.calls == [("who wrote it", ["alpha", "beta", "gamma"])]

    def test_ties_keep_their_incoming_relative_order(self):
        reranker, _loader = _reranker(FakeModel([0.5, 0.9, 0.5]))
        result = reranker.rerank("q", list(PAIRS))
        assert result == [(0.9, "beta"), (0.5, "alpha"), (0.5, "gamma")]

    def test_all_equal_scores_preserve_the_incoming_order_exactly(self):
        reranker, _loader = _reranker(FakeModel([0.4, 0.4, 0.4]))
        result = reranker.rerank("q", list(PAIRS))
        assert [text for _score, text in result] == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_keeps_the_best_k(self):
        reranker, _loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        result = reranker.rerank("q", list(PAIRS), top_k=2)
        assert result == [(0.9, "beta"), (0.5, "gamma")]

    @pytest.mark.parametrize("top_k", [None, 0, -3])
    def test_no_truncation_for_none_or_non_positive_top_k(self, top_k):
        reranker, _loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        result = reranker.rerank("q", list(PAIRS), top_k=top_k)
        assert len(result) == len(PAIRS)

    def test_top_k_larger_than_the_input_is_not_an_error(self):
        reranker, _loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        result = reranker.rerank("q", list(PAIRS), top_k=99)
        assert result == [(0.9, "beta"), (0.5, "gamma"), (0.1, "alpha")]


# ---------------------------------------------------------------------------
# Short-circuit: nothing to reorder, no model load
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_empty_input_comes_back_unchanged_without_loading(self):
        reranker, loader = _reranker(FakeModel([0.1]))
        pairs: list[tuple[float, str]] = []
        assert reranker.rerank("q", pairs) is pairs
        assert loader.calls == []

    def test_single_element_input_comes_back_unchanged_without_loading(self):
        reranker, loader = _reranker(FakeModel([0.1]))
        pairs = [(0.42, "only")]
        result = reranker.rerank("q", pairs)
        assert result is pairs
        assert result == [(0.42, "only")], "the fused score is kept, not rescored"
        assert loader.calls == []

    def test_short_circuit_ignores_top_k(self):
        reranker, loader = _reranker(FakeModel([0.1]))
        pairs = [(0.42, "only")]
        assert reranker.rerank("q", pairs, top_k=1) is pairs
        assert loader.calls == []

    def test_two_elements_do_load(self):
        """The boundary is < 2: the arm works for the smallest real case."""
        reranker, loader = _reranker(FakeModel([0.1, 0.2]))
        result = reranker.rerank("q", [(0.9, "a"), (0.8, "b")])
        assert result == [(0.2, "b"), (0.1, "a")]
        assert len(loader.calls) == 1


# ---------------------------------------------------------------------------
# Degradation: every failure keeps the fused order
# ---------------------------------------------------------------------------


def _assert_unchanged(result: list[tuple[float, str]],
                      pairs: list[tuple[float, str]]) -> None:
    assert result is pairs, "degradation must return the ORIGINAL list object"
    assert result == pairs
    for got, original in zip(result, pairs, strict=True):
        assert got is original, "the original tuples must come back untouched"


class TestDegradation:
    def test_loader_raising_returns_the_original_list_unchanged(self, caplog):
        loader = RecordingLoader(error=RuntimeError("no weights on disk"))
        reranker = rerank_module.Reranker(loader=loader)
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert [r.levelno for r in caplog.records] == [logging.WARNING]
        assert "fused order" in caplog.records[0].getMessage()

    def test_model_raising_during_scoring_degrades(self, caplog):
        reranker, _loader = _reranker(ExplodingModel(ValueError("bad onnx")))
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert caplog.records, "a degraded rerank must say so"

    def test_a_model_that_returns_a_non_iterable_degrades(self, caplog):
        reranker, _loader = _reranker(ReturningModel(None))
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert caplog.records

    @pytest.mark.parametrize("scores", [[0.9, 0.5], [0.9, 0.5, 0.4, 0.3], []])
    def test_score_count_mismatch_degrades(self, scores, caplog):
        """One score per document or nothing — a partial ordering is a lie."""
        reranker, _loader = _reranker(FakeModel(scores))
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert caplog.records

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_score_degrades(self, bad, caplog):
        reranker, _loader = _reranker(FakeModel([0.9, bad, 0.1]))
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert caplog.records

    @pytest.mark.parametrize("bad", ["nope", None, [1.0]])
    def test_non_numeric_score_degrades(self, bad, caplog):
        reranker, _loader = _reranker(FakeModel([0.9, bad, 0.1]))
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert caplog.records

    def test_degradation_preserves_a_permuted_input_exactly(self, caplog):
        """Whatever order recall handed over is what comes back."""
        reranker, _loader = _reranker(ExplodingModel(RuntimeError("boom")))
        pairs = [(0.1, "gamma"), (0.95, "alpha"), (0.5, "beta")]
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        assert result == [(0.1, "gamma"), (0.95, "alpha"), (0.5, "beta")]
        assert [text for _score, text in result] == ["gamma", "alpha", "beta"]

    def test_a_failed_load_does_not_poison_the_next_call(self):
        """A transient failure must not silently disable the arm for the process."""
        loader = RecordingLoader(FakeModel([0.1, 0.9, 0.5]),
                                 error=RuntimeError("cold cache"))
        reranker = rerank_module.Reranker(loader=loader)
        pairs = list(PAIRS)
        assert reranker.rerank("q", pairs) is pairs
        loader.error = None
        assert reranker.rerank("q", pairs) == [
            (0.9, "beta"), (0.5, "gamma"), (0.1, "alpha"),
        ]
        assert len(loader.calls) == 2


# ---------------------------------------------------------------------------
# Score typing
# ---------------------------------------------------------------------------


class TestScoreTyping:
    def test_numpy_scores_are_coerced_to_builtin_floats(self):
        """numpy scalars must not leak into the caller's ranking code."""
        np = pytest.importorskip("numpy")
        reranker, _loader = _reranker(FakeModel([np.float32(0.25), np.float64(0.75)]))
        result = reranker.rerank("q", [(0.9, "a"), (0.8, "b")])
        assert [type(score) for score, _text in result] == [float, float]
        assert result[0][0] == pytest.approx(0.75)
        assert result[1][0] == pytest.approx(0.25)

    def test_integer_scores_come_back_as_floats(self):
        reranker, _loader = _reranker(FakeModel([1, 2]))
        result = reranker.rerank("q", [(0.9, "a"), (0.8, "b")])
        assert [type(score) for score, _text in result] == [float, float]
        assert result[0] == (2.0, "b")


# ---------------------------------------------------------------------------
# Laziness and availability
# ---------------------------------------------------------------------------


class TestLazyLoading:
    def test_constructing_a_reranker_calls_no_loader(self):
        loader = RecordingLoader(FakeModel([0.1]))
        rerank_module.Reranker(loader=loader)
        assert loader.calls == []

    def test_available_calls_no_loader_and_returns_a_bool(self):
        loader = RecordingLoader(FakeModel([0.1]))
        reranker = rerank_module.Reranker(loader=loader)
        assert isinstance(reranker.available(), bool)
        assert loader.calls == []

    def test_the_default_construction_loads_nothing(self):
        """No loader, no cache dir, no model — merely saying so is free."""
        reranker = rerank_module.Reranker()
        assert isinstance(reranker.available(), bool)
        assert reranker._model is None, "available() must not build a model"

    def test_the_loader_is_called_once_and_the_model_is_reused(self):
        reranker, loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        pairs = list(PAIRS)
        reranker.rerank("q", pairs)
        reranker.rerank("q", pairs)
        assert len(loader.calls) == 1

    def test_the_loader_receives_the_model_name_and_the_cache_dir(self):
        reranker, loader = _reranker(FakeModel([0.1, 0.9, 0.5]),
                                     model_name="BAAI/bge-reranker-base",
                                     cache_dir="/tmp/some-cache")
        reranker.rerank("q", list(PAIRS))
        assert loader.calls == [("BAAI/bge-reranker-base", "/tmp/some-cache")]

    def test_the_loader_receives_the_default_model_when_none_is_given(self):
        reranker, loader = _reranker(FakeModel([0.1, 0.9, 0.5]))
        reranker.rerank("q", list(PAIRS))
        assert loader.calls == [(EXPECTED_DEFAULT_MODEL, None)]

    def test_available_reports_an_injected_loader_as_available(self, monkeypatch):
        """With a loader injected, the ML stack is not what availability means."""
        monkeypatch.setattr(rerank_module, "fastembed_available", lambda: False)
        loader = RecordingLoader(FakeModel([0.1]))
        assert rerank_module.Reranker(loader=loader).available() is True

    @pytest.mark.parametrize("probe", [True, False])
    def test_available_follows_the_fastembed_import_probe(self, monkeypatch, probe):
        monkeypatch.setattr(rerank_module, "fastembed_available", lambda: probe)
        assert rerank_module.Reranker().available() is probe

    def test_the_real_availability_probe_is_a_bool_and_downloads_nothing(
        self, tmp_path
    ):
        cache = tmp_path / "never-created"
        reranker = rerank_module.Reranker(cache_dir=str(cache))
        assert isinstance(reranker.available(), bool)
        assert not cache.exists(), "available() must not download weights"


# ---------------------------------------------------------------------------
# The ML stack stays out of module scope
# ---------------------------------------------------------------------------


def _module_level_import_roots(path: Path) -> set[str]:
    """Import roots at MODULE scope only — function-level imports excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestNoModuleScopeMlImport:
    ML_ROOTS = ("fastembed", "onnxruntime", "numpy", "sentence_transformers")

    def test_the_module_imports_no_ml_stack_at_module_scope(self):
        roots = _module_level_import_roots(REPO / "rerank.py")
        offenders = sorted(roots & set(self.ML_ROOTS))
        assert not offenders, (
            f"rerank.py imports {offenders} at module scope — the opt-in arm must "
            f"stay importable without the ML stack"
        )

    def test_the_fastembed_import_lives_inside_a_function(self):
        """The lazy import must actually exist, or the seam is dead code."""
        tree = ast.parse((REPO / "rerank.py").read_text(encoding="utf-8"))
        lazy = [
            node
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(fn)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("fastembed")
        ]
        assert lazy, "no function-level fastembed import: the real loader is a stub"

    def test_a_clean_interpreter_imports_the_module_without_the_ml_stack(
        self, tmp_path
    ):
        """The AST check, verified live in a fresh interpreter."""
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('qd_rerank_probe', "
            f"{str(REPO / 'rerank.py')!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "leaked = [m for m in ('fastembed', 'onnxruntime', 'numpy',\n"
            "                      'sentence_transformers') if m in sys.modules]\n"
            "assert not leaked, f'importing rerank pulled in {leaked}'\n"
            "assert mod.DEFAULT_MODEL == 'Xenova/ms-marco-MiniLM-L-6-v2'\n"
            "assert isinstance(mod.Reranker().available(), bool)\n"
            "print('OK')\n",
            encoding="utf-8",
        )
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
        result = subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert result.returncode == 0, (
            f"probe failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# The real seam, without a download
# ---------------------------------------------------------------------------


class TestRealLoaderWiring:
    """Touches real fastembed, but only paths that precede any download."""

    def test_the_default_model_is_in_fastembeds_registry(self):
        cross_encoder = pytest.importorskip("fastembed.rerank.cross_encoder")
        registry = {
            entry["model"]
            for entry in cross_encoder.TextCrossEncoder.list_supported_models()
        }
        assert EXPECTED_DEFAULT_MODEL in registry, (
            f"{EXPECTED_DEFAULT_MODEL} is not a fastembed cross-encoder; the "
            f"default would fail at load time"
        )

    def test_the_real_loader_rejects_an_unknown_model_before_downloading(self):
        """Proof the seam is wired to TextCrossEncoder, and that it validates first."""
        pytest.importorskip("fastembed.rerank.cross_encoder")
        with pytest.raises(ValueError):
            rerank_module._default_loader("not/a-real-reranker", None)

    def test_an_unbuildable_real_model_degrades_instead_of_raising(self, caplog):
        """The real seam, end to end: an unsupported name fails inside
        TextCrossEncoder BEFORE any download, and the fused order survives."""
        pytest.importorskip("fastembed.rerank.cross_encoder")
        reranker = rerank_module.Reranker(model_name="not/a-real-reranker")
        pairs = list(PAIRS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = reranker.rerank("q", pairs)
        _assert_unchanged(result, pairs)
        assert [r.levelno for r in caplog.records] == [logging.WARNING]
