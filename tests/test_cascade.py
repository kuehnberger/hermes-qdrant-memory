"""Tests for the recall fallback cascade.

The cascade is the policy that turns "hybrid, dense, lexical" into one answer
plus a reason, and its whole job is absorbing failure -- so every test here
drives a failure path with a bare Python callable. No server, no model, no
network and no clock: the module is pure by construction and these tests are
pure for the same reason.

Assertions are on values -- the winning level, the attempted order, the reason
string, the reported exception -- never on line counts or log text, because the
contract is the value, not its formatting.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
from plugins.memory.qdrant import cascade


def failing(exc: BaseException) -> Callable[[], list]:
    """A level callable that raises ``exc`` -- the same instance -- when run."""

    def fn() -> list:
        raise exc

    return fn


def make_spy(calls: list[str], name: str, points: list) -> Callable[[], list]:
    """A level callable that records its name in ``calls`` and returns points."""

    def fn() -> list:
        calls.append(name)
        return points

    return fn


# ---------------------------------------------------------------------------
# The first level wins
# ---------------------------------------------------------------------------


class TestFirstLevelWins:

    def test_returns_the_first_non_empty_level_undegraded(self):
        points = [{"id": 1, "score": 0.9}]
        result = cascade.run_cascade([
            ("hybrid", lambda: points),
            ("dense", lambda: [{"id": 2}]),
        ])
        assert result.points is points
        assert result.level == "hybrid"
        assert result.reason == ""
        assert result.attempted == ["hybrid"]
        assert result.degraded is False

    def test_stops_calling_once_a_level_wins(self):
        calls: list[str] = []
        result = cascade.run_cascade([
            ("hybrid", make_spy(calls, "hybrid", [1])),
            ("dense", make_spy(calls, "dense", [2])),
            ("lexical", make_spy(calls, "lexical", [3])),
        ])
        assert calls == ["hybrid"]
        assert result.level == "hybrid"
        assert result.attempted == ["hybrid"]

    def test_level_names_are_the_callers_choice(self):
        """This module orders callables; it does not know the plugin's names."""
        result = cascade.run_cascade([
            ("rerank-then-dense", lambda: [1]),
            ("dense", lambda: [2]),
        ])
        assert result.level == "rerank-then-dense"
        assert result.degraded is False

    def test_canonical_level_names_are_exposed(self):
        assert cascade.LEVELS == ("hybrid", "dense", "lexical", "none")


# ---------------------------------------------------------------------------
# Demotion
# ---------------------------------------------------------------------------


class TestDemotion:

    def test_raising_first_level_demotes_and_reports_the_cause(self):
        boom = RuntimeError("sparse index missing")
        result = cascade.run_cascade([
            ("hybrid", failing(boom)),
            ("dense", lambda: [9]),
        ])
        assert result.level == "dense"
        assert result.points == [9]
        assert result.reason == "hybrid failed: RuntimeError: sparse index missing"
        assert result.attempted == ["hybrid", "dense"]
        assert result.degraded is True

    def test_empty_first_level_demotes(self):
        result = cascade.run_cascade([
            ("hybrid", lambda: []),
            ("dense", lambda: [7]),
        ])
        assert result.level == "dense"
        assert result.points == [7]
        assert result.reason == "hybrid returned no results"
        assert result.attempted == ["hybrid", "dense"]
        assert result.degraded is True

    def test_reason_names_the_first_demotion_even_after_more_failures(self):
        result = cascade.run_cascade([
            ("hybrid", failing(ValueError("sparse index missing"))),
            ("dense", failing(KeyError("embedder offline"))),
            ("lexical", lambda: [1]),
        ])
        assert result.level == "lexical"
        assert result.points == [1]
        assert result.reason == "hybrid failed: ValueError: sparse index missing"
        assert result.attempted == ["hybrid", "dense", "lexical"]

    def test_an_empty_level_before_a_failure_is_still_the_first_cause(self):
        result = cascade.run_cascade([
            ("hybrid", lambda: []),
            ("dense", failing(OSError("refused"))),
            ("lexical", lambda: [1]),
        ])
        assert result.level == "lexical"
        assert result.reason == "hybrid returned no results"

    def test_a_failure_does_not_abort_the_cascade(self):
        """The contract: a raising level must never abort the run."""
        result = cascade.run_cascade([
            ("hybrid", failing(RuntimeError("boom"))),
            ("dense", failing(RuntimeError("boom again"))),
            ("lexical", lambda: [3]),
        ])
        assert result.level == "lexical"
        assert result.points == [3]

    def test_a_base_exception_subclass_is_absorbed_too(self):
        """pytest's Failed and friends are BaseException, not Exception."""

        class Weird(BaseException):
            pass

        result = cascade.run_cascade([
            ("hybrid", failing(Weird("exotic"))),
            ("dense", lambda: [1]),
        ])
        assert result.level == "dense"
        assert result.reason == "hybrid failed: Weird: exotic"


# ---------------------------------------------------------------------------
# Terminal "none"
# ---------------------------------------------------------------------------


class TestTerminalNone:

    def test_every_level_empty_is_a_clean_empty_store(self):
        result = cascade.run_cascade([
            ("hybrid", lambda: []),
            ("dense", lambda: []),
        ])
        assert result.points == []
        assert result.level == "none"
        assert result.reason == "all levels returned no results"
        assert result.attempted == ["hybrid", "dense"]
        assert result.degraded is True

    def test_every_level_raising_reports_the_first_cause(self):
        result = cascade.run_cascade([
            ("hybrid", failing(ValueError("sparse index missing"))),
            ("dense", failing(OSError("embedder offline"))),
            ("lexical", failing(RuntimeError("bm25 down"))),
        ])
        assert result.points == []
        assert result.level == "none"
        assert result.reason == "hybrid failed: ValueError: sparse index missing"
        assert result.attempted == ["hybrid", "dense", "lexical"]

    def test_the_first_cause_survives_even_when_nothing_recovers(self):
        """Not the clean-empty message: something actually failed here."""
        result = cascade.run_cascade([
            ("hybrid", failing(RuntimeError("boom"))),
            ("dense", lambda: []),
            ("lexical", lambda: []),
        ])
        assert result.level == "none"
        assert result.reason == "hybrid failed: RuntimeError: boom"

    def test_no_levels_configured(self):
        result = cascade.run_cascade([])
        assert result.points == []
        assert result.level == "none"
        assert result.reason == "no levels configured"
        assert result.attempted == []
        assert result.degraded is False

    def test_a_single_empty_level_is_none(self):
        result = cascade.run_cascade([("dense", lambda: [])])
        assert result.level == "none"
        assert result.reason == "all levels returned no results"
        assert result.attempted == ["dense"]


# ---------------------------------------------------------------------------
# on_error
# ---------------------------------------------------------------------------


class TestOnError:

    def test_called_once_per_failure_with_name_and_exc(self):
        boom = RuntimeError("sparse index missing")
        seen: list[tuple[str, BaseException]] = []
        result = cascade.run_cascade(
            [("hybrid", failing(boom)), ("dense", lambda: [1])],
            on_error=lambda name, exc: seen.append((name, exc)),
        )
        assert seen == [("hybrid", boom)]
        assert result.level == "dense"

    def test_called_once_for_each_of_several_failures(self):
        seen: list[tuple[str, BaseException]] = []
        result = cascade.run_cascade(
            [
                ("hybrid", failing(ValueError("a"))),
                ("dense", failing(TypeError("b"))),
                ("lexical", lambda: []),
            ],
            on_error=lambda name, exc: seen.append((name, exc)),
        )
        assert [name for name, _ in seen] == ["hybrid", "dense"]
        assert result.level == "none"

    def test_its_own_exception_does_not_escape(self):
        def broken_hook(name: str, exc: BaseException) -> None:
            raise RuntimeError("the hook itself is broken")

        result = cascade.run_cascade(
            [("hybrid", failing(ValueError("level"))), ("dense", lambda: [1])],
            on_error=broken_hook,
        )
        assert result.level == "dense"
        assert result.points == [1]
        assert result.reason == "hybrid failed: ValueError: level"

    def test_not_called_for_a_level_that_merely_returned_empty(self):
        seen: list[tuple[str, BaseException]] = []
        result = cascade.run_cascade(
            [("hybrid", lambda: []), ("dense", lambda: [1])],
            on_error=lambda name, exc: seen.append((name, exc)),
        )
        assert seen == []
        assert result.level == "dense"

    def test_omitting_on_error_leaves_the_cascade_working(self):
        result = cascade.run_cascade([
            ("hybrid", failing(RuntimeError("boom"))),
            ("dense", lambda: [1]),
        ])
        assert result.level == "dense"
        assert result.reason == "hybrid failed: RuntimeError: boom"


# ---------------------------------------------------------------------------
# attempted ordering
# ---------------------------------------------------------------------------


class TestAttemptedOrder:

    def test_names_are_recorded_in_call_order(self):
        calls: list[str] = []
        result = cascade.run_cascade([
            ("hybrid", make_spy(calls, "hybrid", [])),
            ("dense", make_spy(calls, "dense", [])),
            ("lexical", lambda: [1]),
        ])
        assert calls == ["hybrid", "dense"]
        assert result.attempted == ["hybrid", "dense", "lexical"]
        assert result.attempted[-1] == result.level == "lexical"

    def test_a_raising_level_is_still_recorded_as_attempted(self):
        result = cascade.run_cascade([
            ("hybrid", failing(RuntimeError("boom"))),
            ("dense", lambda: [1]),
        ])
        assert result.attempted == ["hybrid", "dense"]

    def test_attempted_is_independent_per_result(self):
        first = cascade.run_cascade([("hybrid", lambda: [1])])
        second = cascade.run_cascade([("dense", lambda: [2])])
        assert first.attempted == ["hybrid"]
        assert second.attempted == ["dense"]
        assert first.points == [1]


# ---------------------------------------------------------------------------
# The reason bound
# ---------------------------------------------------------------------------


class TestReasonBounds:

    def test_short_exception_message_is_kept_verbatim(self):
        result = cascade.run_cascade([("dense", failing(OSError("refused")))])
        assert result.reason == "dense failed: OSError: refused"

    def test_long_exception_message_is_bounded(self):
        result = cascade.run_cascade(
            [("hybrid", failing(RuntimeError("x" * 5000)))]
        )
        prefix = "hybrid failed: RuntimeError: "
        assert result.reason.startswith(prefix)
        assert len(result.reason) == len(prefix) + cascade._MAX_REASON_MSG
        assert result.reason.endswith("...")
        assert "x" * 201 not in result.reason

    def test_reason_is_never_empty_once_a_demotion_happened(self):
        result = cascade.run_cascade([
            ("hybrid", failing(ValueError("boom"))),
            ("dense", lambda: []),
        ])
        assert result.level == "none"
        assert result.reason != ""


# ---------------------------------------------------------------------------
# The value object
# ---------------------------------------------------------------------------


class TestValueObject:

    def test_is_frozen(self):
        result = cascade.run_cascade([("dense", lambda: [1])])
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.level = "hybrid"

    def test_degraded_is_true_only_when_the_winner_is_not_the_strongest(self):
        assert cascade.run_cascade([("dense", lambda: [1])]).degraded is False
        assert cascade.run_cascade([
            ("hybrid", lambda: []), ("dense", lambda: [1])
        ]).degraded is True
        assert cascade.run_cascade([]).degraded is False

    def test_runs_without_touching_stdout_or_stderr(self, capsys):
        cascade.run_cascade([
            ("hybrid", failing(ValueError("boom"))),
            ("dense", lambda: []),
        ])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
