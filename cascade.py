"""Recall fallback cascade (rival-hardening: memory-os parity).

WHY THIS MODULE EXISTS
----------------------
Recall has three arms, strongest first: ``hybrid`` (dense + BM25 RRF),
``dense`` (dense-only) and ``lexical`` (BM25 only). They fail independently --
hybrid needs a sparse index that a collection created before the sparse
migration does not have, dense needs an embedder that may never have finished
loading, and both need a Qdrant round trip that can be refused. Without a
cascade, any one of those shows up as "no memories found", which is
indistinguishable from a genuinely empty store.

So recall walks the levels in order and *says so*. The result carries the level
that actually answered, every level that was tried, and the reason the stronger
ones were skipped, so a degraded turn is observable -- a status line, a metric,
a debug record -- instead of silently looking like a clean empty recall.

WHY IT IS A PURE ORCHESTRATOR
-----------------------------
This module never imports the Qdrant client, the embedder, or anything from the
plugin package. The caller supplies the callables::

    run_cascade([
        ("hybrid",  lambda: self._recall_hybrid(query)),
        ("dense",   lambda: self._recall_dense(query)),
        ("lexical", lambda: self._recall_lexical(query)),
    ], on_error=self._note_recall_failure)

That split is what keeps the *policy* -- order, degradation, reason recording --
unit-testable with no server, no model and no LLM, and keeps every I/O failure,
the thing the cascade exists to absorb, outside this file. It also means the
module holds no clock, no randomness and no global state: a given
``(levels, on_error)`` always produces the same ``CascadeResult``.

Public surface: :data:`LEVELS`, :class:`CascadeResult`, :func:`run_cascade`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

__all__ = ("LEVELS", "CascadeResult", "run_cascade")

#: The canonical recall levels, strongest first. ``"none"`` is the terminal
#: sentinel reported when every level failed -- it is not itself a callable
#: level, and a cascade configured with it as a level would win with no points.
LEVELS: tuple[str, ...] = ("hybrid", "dense", "lexical", "none")

#: Bound, in characters, on the exception text carried into ``reason``. A
#: server's HTML error page or a traceback-shaped message would otherwise turn
#: a one-line status into a wall of text.
_MAX_REASON_MSG = 200

#: Reason reported when the cascade was handed nothing to run.
REASON_NO_LEVELS = "no levels configured"

#: Reason reported when every level ran cleanly and every level came back
#: empty -- a genuinely empty store, not a failure.
REASON_ALL_EMPTY = "all levels returned no results"


@dataclass(frozen=True)
class CascadeResult:
    """The outcome of one cascade: what came back, and at what cost.

    ``attempted`` holds the level names in the order they were actually run, so
    ``attempted[0]`` is the level the caller *wanted* and ``level`` is the level
    that *answered*. Comparing the two is what makes "recall silently dropped a
    rung" detectable -- see :attr:`degraded`.
    """

    #: Whatever the winning level returned (the caller's own list object).
    points: list
    #: The level that produced ``points``: ``"hybrid" | "dense" | "lexical"``,
    #: or ``"none"`` when nothing did.
    level: str
    #: ``""`` when the first level succeeded; otherwise the FIRST cause that
    #: demoted the cascade, e.g. ``"hybrid failed: OSError: refused"`` or
    #: ``"hybrid returned no results"``. Bounded, single-line, no tracebacks.
    reason: str = ""
    #: Level names tried, in order -- including the winner.
    attempted: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """True when recall did not run at the strongest level.

        ``level != attempted[0]`` is the honest test: an empty ``attempted``
        means nothing was ever configured to run, which is a configuration
        error rather than a demotion, so it is *not* reported as degraded.
        """
        return bool(self.attempted) and self.level != self.attempted[0]


def _bounded_message(exc: BaseException) -> str:
    """``str(exc)`` capped at :data:`_MAX_REASON_MSG` characters.

    Kept separate from the format string so the bound is testable on its own
    and so a pathological message cannot smuggle a newline-block into a status
    line.
    """
    text = str(exc).strip()
    if len(text) > _MAX_REASON_MSG:
        text = text[: _MAX_REASON_MSG - 3] + "..."
    return text


def _failure_reason(name: str, exc: BaseException) -> str:
    """``"<level> failed: <ExcType>: <message>"`` -- the demotion's first cause."""
    return f"{name} failed: {type(exc).__name__}: {_bounded_message(exc)}"


def _notify(
    on_error: Callable[[str, BaseException], None], name: str, exc: BaseException
) -> None:
    """Call ``on_error`` and swallow whatever it raises.

    The callback is instrumentation, never a participant: a broken hook must
    not cost the user the recall that was still available at the next level.
    """
    try:
        on_error(name, exc)
    except BaseException:
        return


def run_cascade(
    levels: Iterable[tuple[str, Callable[[], list]]],
    *,
    on_error: Callable[[str, BaseException], None] | None = None,
) -> CascadeResult:
    """Run ``levels`` strongest-first and return the first non-empty result.

    Each entry is a ``(name, fn)`` pair; ``fn`` takes no arguments and returns
    the points it found. The first ``fn`` to return a non-empty list wins and
    the cascade stops -- later levels are never called. A level that raises is
    absorbed: ``on_error(name, exc)`` is called if given, the failure is
    recorded as the demotion reason, and the next level runs. A raising level
    therefore can never abort the cascade.

    ``reason`` names the FIRST cause that demoted the cascade, so a turn that
    fell from hybrid to dense and then to lexical still reports hybrid. The one
    exception is the terminal no-result case: when every level ran cleanly and
    every level came back empty, the reason is ``"all levels returned no
    results"`` -- that is an empty store, not a failure, and reading it as one
    would send an operator hunting for an outage that never happened.

    Pure: no I/O, no logging, no clock, no global state.
    """
    attempted: list[str] = []
    first_demotion = ""
    saw_exception = False

    for name, fn in levels:
        attempted.append(name)
        try:
            points = fn()
        except (KeyboardInterrupt, SystemExit):
            # Process-control signals are not level failures; let them through
            # so Ctrl-C still interrupts a long recall.
            raise
        except BaseException as exc:
            # The cascade's contract is absolute: a level that raises can never
            # abort the run, so everything else -- including BaseException
            # subclasses such as pytest's ``Failed`` -- is absorbed here.
            saw_exception = True
            if not first_demotion:
                first_demotion = _failure_reason(name, exc)
            if on_error is not None:
                _notify(on_error, name, exc)
            continue
        if points:
            return CascadeResult(
                points=points, level=name, reason=first_demotion, attempted=attempted
            )
        if not first_demotion:
            first_demotion = f"{name} returned no results"

    if not attempted:
        return CascadeResult(
            points=[], level="none", reason=REASON_NO_LEVELS, attempted=attempted
        )
    return CascadeResult(
        points=[],
        level="none",
        reason=first_demotion if saw_exception else REASON_ALL_EMPTY,
        attempted=attempted,
    )
