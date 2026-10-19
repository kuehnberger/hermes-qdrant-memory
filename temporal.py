"""Temporal query understanding for recall (rival-hardening: Hindsight parity).

A recall query is often partly a *time* query — "what did we decide about the
migration in March 2023", "what happened last week" — and the store keeps every
turn's epoch seconds in the payload field ``timestamp``. This module does the
two jobs that need no model and no network:

1. ``parse_time_window`` reads a natural-language time expression out of the
   query text and returns a half-open UTC epoch window ``[start, end)``;
   ``range_filter`` turns that window into the Qdrant payload filter that
   restricts the search to it.
2. ``bucket_spread`` re-orders the window's hits so the answer covers the whole
   window instead of the densest stretch of it.

Why not just rank by score
--------------------------
Dense recall is biased toward whatever stretch of time is densest: a "what
happened in 2023" query over a corpus that is 90% September 2023 returns nine
September memories and one from March. The user asked about the year, so the
year is the unit of answer. ``bucket_spread`` cuts the window into equal time
buckets and promotes the best hit *per bucket* ahead of the rest — Hindsight's
temporal-spread behaviour, and the whole reason this module exists. It never
drops, duplicates or invents a hit: the result is a permutation of the input.

Design rules
------------
* **Deterministic.** Every function is pure and ``now`` is injectable, so a
  test or a replay pins the same windows forever. Nothing here reads the clock
  except the documented default of ``parse_time_window``.
* **Conservative.** A false positive is worse than a miss: a query silently
  narrowed to the wrong month returns nothing, which the user reads as "no
  memories" rather than as a bug. So number-shaped non-times — "the 3 memory
  layers", "Python 3.11", "GPT-4", "section 4.2", "top 10 results", "rank 2" —
  must all parse to ``None``, and the only bare number treated as a time at all
  is a year (19xx/20xx), and only when it is cued ("in 2023", "of 2023") or
  leads the query.
* **Half-open windows.** ``end`` is the *start* of the next unit, so
  "in 2023" is ``[2023-01-01T00:00Z, 2024-01-01T00:00Z)`` and adjacent windows
  tile without overlap or gap. An open-ended expression leaves the missing side
  ``None``: "before June 2024" -> ``(None, <2024-06-01T00:00Z>)``, "since 2023"
  -> ``(<2023-01-01T00:00Z>, None)``.
* **UTC, always.** A naive ``now`` is read as UTC and an aware one is converted;
  epoch seconds carry no zone, so a parse can never depend on the host's local
  time. Windows are floats because ``datetime.timestamp()`` returns floats and
  rounding here would silently move a day boundary.

What is recognised
------------------
    relative   today, yesterday, tomorrow; this/last/next week, month, year
    seasons    this/last/next spring, summer, autumn, fall, winter (northern)
    explicit   "in 2023", "in March", "March 2023", "in March 2023",
               "2023-05", "2023-05-01"
    offsets    "past 7 days", "last 3 weeks", "2 months ago", "5 days ago",
               "next 2 weeks"
    open       "before <expr>", "after <expr>", "since <expr>", "until <expr>",
               "up to <expr>", "prior to <expr>", "from <expr>"

Matching is case-insensitive and first-match-wins in exactly that order, because
the narrower reading is the safer one: "last 3 weeks" is a rolling offset, not
the calendar week that bare "last week" means, so calendar relatives are tried
before numeric offsets and the offsets are the only ones that accept a digit.
An open-ended cue whose remainder does not parse falls through to the rest of
the patterns, so "what did we learn from the 3 memory layers" stays ``None``
while "after the migration, in 2023" still finds the year.

Ambiguity resolves toward the most recent *past* unit: "in March" with ``now``
in October 2026 is March 2026, not March 2027 — a memory query is a question
about the past. "this <season>" is the season starting in ``now``'s year and
"last <season>" the previous year's; seasons are meteorological quarters
(spring = Mar–May) and northern-hemisphere by definition, documented rather
than guessed. A rolling offset ("past 7 days", "2 months ago") is the window
ending at ``now``; month and year arithmetic is calendar-aware, so "2 months
ago" from the 31st lands on the 30th rather than overflowing.

Interface:

    parse_time_window(query: str, now: datetime | None = None)
        -> tuple[float | None, float | None] | None
    range_filter(start: float | None, end: float | None, *,
                 key: str = "timestamp") -> dict[str, Any] | None
    bucket_spread(items: list[tuple[float, float, Any]], start: float | None,
                  end: float | None, *, buckets: int = 6,
                  per_bucket: int = 2) -> list[tuple[float, float, Any]]
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = ["bucket_spread", "parse_time_window", "range_filter"]

#: Windows are UTC epoch seconds; ``datetime.UTC`` is the only zone this module
#: ever produces.
_UTC = UTC

#: The payload field holding a turn's epoch seconds — the key ``range_filter``
#: defaults to and the one the provider writes.
DEFAULT_TS_KEY = "timestamp"

#: Month name -> number. Full names only: "mar" would also match "mark", and a
#: month regex that fires on ordinary words is the false positive this module is
#: most careful about.
_MONTHS: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

#: Season -> the month it starts in. Meteorological quarters (spring = Mar–May,
#: summer = Jun–Aug, autumn = Sep–Nov, winter = Dec–Feb), which is what "last
#: summer" means to a user; astronomical dates would drift by a day or two a
#: year and buy nothing for a recall window.
_SEASONS: dict[str, int] = {
    "spring": 3,
    "summer": 6,
    "autumn": 9,
    "fall": 9,
    "winter": 12,
}

#: Cues that bound a window *below* (everything after the expression); every
#: other cue in ``_OPEN_CUE`` bounds it above.
_LOWER_CUES = frozenset({"after", "later than", "since", "from"})

#: Fixed-length units, as timedeltas. Months and years are absent on purpose:
#: they need calendar arithmetic (``_shift_months``), not 30/365-day guesses.
_UNIT_DELTAS: dict[str, timedelta] = {
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
}

#: Month-length units, as a count of months.
_UNIT_MONTHS: dict[str, int] = {"month": 1, "year": 12}

_MONTH_ALT = "|".join(_MONTHS)

#: "before X" / "since X" / "until X". The remainder is parsed recursively, so a
#: cue with nothing temporal after it is harmless: the inner parse returns None
#: and the caller falls through to the remaining patterns.
_OPEN_CUE = re.compile(
    r"\b(before|prior to|earlier than|until|up to|after|later than|since|from)"
    r"\s+(.+)$",
    re.IGNORECASE,
)

#: ISO shapes. The day pattern is tried first: "2023-05-01" also matches the
#: month pattern's prefix, and the day is the narrower (safer) reading.
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b")

#: "March 2023", "March of 2023", "in March 2023".
_MONTH_YEAR = re.compile(
    rf"\b({_MONTH_ALT})\s+(?:of\s+)?((?:19|20)\d{{2}})\b", re.IGNORECASE
)

#: A bare month name needs a preposition. "March" alone is also a verb — "march
#: through the code" is not a date — so only "in March" / "during March" count.
_BARE_MONTH = re.compile(rf"\b(?:in|during)\s+({_MONTH_ALT})\b", re.IGNORECASE)

#: "last summer", "this autumn", "next spring".
_SEASON = re.compile(
    r"\b(this|last|next)\s+(spring|summer|autumn|fall|winter)\b", re.IGNORECASE
)

#: "5 days ago" — a rolling window ending at ``now``.
_AGO = re.compile(r"\b(\d+)\s+(hour|day|week|month|year)s?\s+ago\b", re.IGNORECASE)

#: "past 7 days", "last 3 weeks", "next 2 months". A digit is required, which is
#: what keeps bare "last week" out of this pattern and in the calendar one.
_ROLLING = re.compile(
    r"\b(?:in\s+the\s+)?(past|last|previous|next|coming)\s+(\d+)\s+"
    r"(hour|day|week|month|year)s?\b",
    re.IGNORECASE,
)

#: Calendar relatives, matched as whole phrases so "last week" cannot be read
#: out of "last weekend".
_RELATIVE = re.compile(
    r"\b(today|yesterday|tomorrow|(?:this|last|next)\s+(?:week|month|year))\b",
    re.IGNORECASE,
)

#: A year is the only bare number this module trusts, and only when cued...
_CUED_YEAR = re.compile(
    r"\b(?:in|during|of|year|since|before|after|from|until|through|by|around|"
    r"early|late|mid)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

#: ...or when it leads the query ("2023 roadmap", and the recursive parse of
#: "since 2023", whose remainder is the bare year).
_LEADING_YEAR = re.compile(r"^\s*((?:19|20)\d{2})\b")


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def _as_utc(value: datetime) -> datetime:
    """Read a naive datetime as UTC and convert an aware one to it."""
    if value.tzinfo is None:
        return value.replace(tzinfo=_UTC)
    return value.astimezone(_UTC)


def _shift_months(start: datetime, months: int) -> datetime:
    """Shift by whole months, clamping the day to the target month's length.

    ``timedelta`` cannot express "one month": 2026-03-31 minus one month must be
    2026-02-28, not 2026-03-03. Clamping is the reading a calendar user expects,
    and it keeps month windows exact.
    """
    index = start.year * 12 + (start.month - 1) + months
    year, month = divmod(index, 12)
    month += 1
    return start.replace(
        year=year, month=month, day=min(start.day, monthrange(year, month)[1])
    )


def _shift(start: datetime, amount: int, unit: str) -> datetime:
    """Shift by ``amount`` of ``unit``; calendar-aware for months and years."""
    if unit in _UNIT_MONTHS:
        return _shift_months(start, amount * _UNIT_MONTHS[unit])
    return start + amount * _UNIT_DELTAS[unit]


def _window(start: datetime, end: datetime) -> tuple[float, float]:
    """Epoch-second half-open window ``[start, end)``."""
    return (start.timestamp(), end.timestamp())


def _day_start(year: int, month: int, day: int) -> datetime | None:
    """Midnight UTC of a calendar date, or None when the date does not exist."""
    try:
        return datetime(year, month, day, tzinfo=_UTC)
    except ValueError:
        return None


def _month_start(year: int, month: int) -> datetime | None:
    """Midnight UTC of the first of a month, or None for an impossible month."""
    if not 1 <= month <= 12:
        return None
    return datetime(year, month, 1, tzinfo=_UTC)


# ---------------------------------------------------------------------------
# Expression -> window
# ---------------------------------------------------------------------------


def _relative_window(phrase: str, ref: datetime) -> tuple[float, float]:
    """Window for a calendar relative: today, this week, last month, …"""
    day = ref.replace(hour=0, minute=0, second=0, microsecond=0)
    if phrase == "today":
        return _window(day, day + timedelta(days=1))
    if phrase == "yesterday":
        return _window(day - timedelta(days=1), day)
    if phrase == "tomorrow":
        return _window(day + timedelta(days=1), day + timedelta(days=2))

    direction, unit = phrase.split()  # "this week" / "last month" / "next year"
    step = {"this": 0, "last": -1, "next": 1}[direction]
    if unit == "week":
        monday = day - timedelta(days=day.weekday())
        return _window(
            monday + timedelta(weeks=step), monday + timedelta(weeks=step + 1)
        )
    if unit == "month":
        start = _shift_months(day.replace(day=1), step)
        return _window(start, _shift_months(start, 1))
    start = _shift_months(day.replace(day=1, month=1), step * 12)
    return _window(start, _shift_months(start, 12))


def _season_window(direction: str, season: str, ref: datetime) -> tuple[float, float]:
    """Window for this/last/next + a season, as a three-month quarter."""
    step = {"this": 0, "last": -1, "next": 1}[direction]
    start = datetime(ref.year + step, _SEASONS[season], 1, tzinfo=_UTC)
    return _window(start, _shift_months(start, 3))


def _bare_month_window(month: int, ref: datetime) -> tuple[float, float]:
    """Window for a month name with no year: the most recent one already past."""
    start = _month_start(ref.year, month)
    if start is None or start > ref:
        start = _month_start(ref.year - 1, month)
    assert start is not None  # a valid month is always constructible
    return _window(start, _shift_months(start, 1))


def _year_window(year: int) -> tuple[float, float]:
    """Window for a bare year: the whole calendar year, half-open."""
    start = datetime(year, 1, 1, tzinfo=_UTC)
    return _window(start, _shift_months(start, 12))


def _parse(text: str, ref: datetime) -> tuple[float | None, float | None] | None:
    """First-match-wins dispatch over the recognised expression shapes."""
    match = _OPEN_CUE.search(text)
    if match:
        inner = _parse(match.group(2), ref)
        if inner is not None:
            start, end = inner
            if match.group(1).lower() in _LOWER_CUES:
                bound = start if start is not None else end
                return (bound, None) if bound is not None else None
            return (None, start) if start is not None else None
        # Nothing temporal after the cue ("from the 3 memory layers"): fall
        # through so a time expression elsewhere in the query still parses.

    match = _ISO_DATE.search(text)
    if match:
        day = _day_start(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if day is not None:
            return _window(day, day + timedelta(days=1))

    match = _ISO_MONTH.search(text)
    if match:
        month = _month_start(int(match.group(1)), int(match.group(2)))
        if month is not None:
            return _window(month, _shift_months(month, 1))

    match = _MONTH_YEAR.search(text)
    if match:
        month = _month_start(int(match.group(2)), _MONTHS[match.group(1).lower()])
        assert month is not None  # the pattern only yields real month names
        return _window(month, _shift_months(month, 1))

    match = _SEASON.search(text)
    if match:
        return _season_window(match.group(1).lower(), match.group(2).lower(), ref)

    match = _AGO.search(text)
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        return (_shift(ref, -amount, unit).timestamp(), ref.timestamp())

    match = _ROLLING.search(text)
    if match:
        direction = match.group(1).lower()
        amount, unit = int(match.group(2)), match.group(3).lower()
        if direction in ("next", "coming"):
            return (ref.timestamp(), _shift(ref, amount, unit).timestamp())
        return (_shift(ref, -amount, unit).timestamp(), ref.timestamp())

    match = _RELATIVE.search(text)
    if match:
        return _relative_window(" ".join(match.group(1).lower().split()), ref)

    match = _BARE_MONTH.search(text)
    if match:
        return _bare_month_window(_MONTHS[match.group(1).lower()], ref)

    match = _CUED_YEAR.search(text)
    if match:
        return _year_window(int(match.group(1)))

    match = _LEADING_YEAR.match(text)
    if match:
        return _year_window(int(match.group(1)))

    return None


def parse_time_window(
    query: str, now: datetime | None = None
) -> tuple[float | None, float | None] | None:
    """Return the UTC epoch window a query names, or None if it names none.

    The window is half-open — ``[start, end)`` — so consecutive units tile
    exactly. An open-ended expression leaves the absent side ``None``:
    "before June 2024" gives ``(None, <2024-06-01T00:00Z>)`` and "since 2023"
    gives ``(<2023-01-01T00:00Z>, None)``. Callers therefore must handle either
    bound being None; ``range_filter`` does.

    ``now`` anchors every relative expression and defaults to
    ``datetime.now(timezone.utc)``. Pass it in tests and replays: it is the only
    clock read in this module. A naive ``now`` is read as UTC.

    Returns None for an empty query, a non-string, and — deliberately — for
    every number-shaped non-time, because a wrong window is worse than no
    window: the wrong month silently returns zero memories.
    """
    if not isinstance(query, str) or not query.strip():
        return None
    reference = _as_utc(now) if now is not None else datetime.now(_UTC)
    return _parse(query, reference)


def range_filter(
    start: float | None, end: float | None, *, key: str = DEFAULT_TS_KEY
) -> dict[str, Any] | None:
    """Express a window as a Qdrant payload range filter.

    Emits ``{"key": key, "range": {...}}`` with ``gte`` for ``start`` and
    ``lte`` for ``end``, omitting whichever bound is None — so a one-sided
    window is a one-sided filter, not a filter with a null bound. Returns None
    when both are None, which lets the caller skip passing a filter at all
    (Qdrant reads an empty range as "match nothing", not as "no bound").

    Both bounds are inclusive. ``parse_time_window``'s ``end`` is exclusive, so
    the only difference is an item sitting exactly on a boundary second, which
    is worth more than a window that drops its own endpoint.
    """
    bounds: dict[str, float] = {}
    if start is not None:
        bounds["gte"] = start
    if end is not None:
        bounds["lte"] = end
    if not bounds:
        return None
    return {"key": key, "range": bounds}


def _bucket_of(ts: float, low: float, span: float, count: int) -> int:
    """Index of the time bucket ``ts`` falls in, clamped to the bucket range.

    Clamping keeps an item whose timestamp sits outside an explicitly narrower
    window in the end buckets instead of dropping or mis-indexing it: the
    filter may have let it through, and silently losing a hit is the one thing
    a re-ranker must never do.
    """
    if span <= 0:
        return 0
    index = int((ts - low) / span * count)
    return max(0, min(count - 1, index))


def bucket_spread(
    items: list[tuple[float, float, Any]],
    start: float | None,
    end: float | None,
    *,
    buckets: int = 6,
    per_bucket: int = 2,
) -> list[tuple[float, float, Any]]:
    """Re-order score-sorted hits so they cover the whole time window.

    ``items`` is ``(score, ts, payload)`` tuples already sorted by descending
    score. The window is cut into ``buckets`` equal spans; walking them in time
    order, up to ``per_bucket`` hits are taken from each populated bucket
    (strongest first), and every remaining hit follows in the original score
    order. That prefix is the answer to "what happened in <window>": one strong
    memory per stretch of it, instead of the many from the densest stretch.

    A None ``start`` or ``end`` means "use the items' own extent on that side",
    so a caller with no parsed window still gets a spread over the timestamps it
    was handed. When both are None and every timestamp is identical the span is
    zero and the input order is returned unchanged — no division, no shuffling.

    The result is a permutation of ``items`` (same objects, same length) and is
    fully deterministic: no clock, no randomness, ties broken by input order.
    """
    if not items:
        return []
    stamps = [item[1] for item in items]
    low = start if start is not None else min(stamps)
    high = end if end is not None else max(stamps)
    if high < low:  # a caller-supplied window the wrong way round
        low, high = high, low
    span = high - low
    count = max(1, int(buckets))
    keep = max(0, int(per_bucket))

    grouped: dict[int, list[int]] = {}
    for index, item in enumerate(items):  # score-descending: champions first
        grouped.setdefault(_bucket_of(item[1], low, span, count), []).append(index)

    chosen: list[int] = []
    for bucket in sorted(grouped):
        chosen.extend(grouped[bucket][:keep])

    picked = set(chosen)
    spread = [items[index] for index in chosen]
    spread.extend(item for index, item in enumerate(items) if index not in picked)
    return spread
