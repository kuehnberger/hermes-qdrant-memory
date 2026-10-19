"""Unit tests for ``temporal``: time-expression parsing and spread selection.

Pure stdlib — no server, no model, no clock. Every parse call injects ``NOW``,
so each expectation is a literal epoch built from ``datetime`` in *this* file
rather than a value re-derived by the module under test.

Two halves:

* **Parsing.** The table of expressions, the open-ended shapes, and the
  false-positive set. The false positives are the load-bearing half: a query
  narrowed to the wrong month returns zero memories, which a user reads as "no
  memories", so "Python 3.11" must stay ``None``.
* **``bucket_spread``.** Determinism, spread across buckets (a sparse-stretch
  hit must not stay buried behind a dense cluster), and the no-window fallback
  where the item timestamps themselves define the span.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from plugins.memory.qdrant.temporal import (
    bucket_spread,
    parse_time_window,
    range_filter,
)

#: The injected clock: a Monday, mid-afternoon UTC, in a month with past and
#: future season boundaries on both sides — so week, month and season
#: expectations are unambiguous.
NOW = datetime(2026, 10, 19, 14, 30, tzinfo=UTC)
NOW_TS = NOW.timestamp()

DAY = 86400.0


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    """Epoch seconds of a UTC instant, computed without the module's help."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


# ---------------------------------------------------------------------------
# Expressions with a bounded window on both sides
# ---------------------------------------------------------------------------

PARSE_CASES = [
    # -- calendar relatives
    pytest.param("today", (at(2026, 10, 19), at(2026, 10, 20)), id="today"),
    pytest.param("yesterday", (at(2026, 10, 18), at(2026, 10, 19)), id="yesterday"),
    pytest.param("tomorrow", (at(2026, 10, 20), at(2026, 10, 21)), id="tomorrow"),
    pytest.param("this week", (at(2026, 10, 19), at(2026, 10, 26)), id="this-week"),
    pytest.param("last week", (at(2026, 10, 12), at(2026, 10, 19)), id="last-week"),
    pytest.param("next week", (at(2026, 10, 26), at(2026, 11, 2)), id="next-week"),
    pytest.param("this month", (at(2026, 10, 1), at(2026, 11, 1)), id="this-month"),
    pytest.param("last month", (at(2026, 9, 1), at(2026, 10, 1)), id="last-month"),
    pytest.param("this year", (at(2026, 1, 1), at(2027, 1, 1)), id="this-year"),
    pytest.param("last year", (at(2025, 1, 1), at(2026, 1, 1)), id="last-year"),
    # -- seasons (northern hemisphere, meteorological quarters)
    pytest.param("this spring", (at(2026, 3, 1), at(2026, 6, 1)), id="this-spring"),
    pytest.param("last spring", (at(2025, 3, 1), at(2025, 6, 1)), id="last-spring"),
    pytest.param("this summer", (at(2026, 6, 1), at(2026, 9, 1)), id="this-summer"),
    pytest.param("last summer", (at(2025, 6, 1), at(2025, 9, 1)), id="last-summer"),
    pytest.param("this autumn", (at(2026, 9, 1), at(2026, 12, 1)), id="this-autumn"),
    pytest.param("last fall", (at(2025, 9, 1), at(2025, 12, 1)), id="last-fall"),
    pytest.param("last winter", (at(2025, 12, 1), at(2026, 3, 1)), id="last-winter"),
    # winter crosses the year boundary: "this winter" starts in December.
    pytest.param("this winter", (at(2026, 12, 1), at(2027, 3, 1)), id="this-winter"),
    # -- explicit dates
    pytest.param("in 2023", (at(2023, 1, 1), at(2024, 1, 1)), id="in-year"),
    pytest.param("2023", (at(2023, 1, 1), at(2024, 1, 1)), id="bare-year"),
    pytest.param("in March", (at(2026, 3, 1), at(2026, 4, 1)), id="in-month"),
    pytest.param("March 2023", (at(2023, 3, 1), at(2023, 4, 1)), id="month-year"),
    pytest.param("in March 2023", (at(2023, 3, 1), at(2023, 4, 1)), id="in-month-year"),
    pytest.param("March of 2023", (at(2023, 3, 1), at(2023, 4, 1)), id="month-of-year"),
    pytest.param("2023-05", (at(2023, 5, 1), at(2023, 6, 1)), id="iso-month"),
    pytest.param("2023-05-01", (at(2023, 5, 1), at(2023, 5, 2)), id="iso-date"),
    # -- rolling offsets, anchored at `now`
    pytest.param("past 7 days", (NOW_TS - 7 * DAY, NOW_TS), id="past-7-days"),
    pytest.param("last 3 weeks", (NOW_TS - 21 * DAY, NOW_TS), id="last-3-weeks"),
    pytest.param("5 days ago", (NOW_TS - 5 * DAY, NOW_TS), id="5-days-ago"),
    # calendar-aware: two months before 2026-10-19 is 2026-08-19, same time.
    pytest.param(
        "2 months ago", (at(2026, 8, 19, 14, 30), NOW_TS), id="2-months-ago"
    ),
    pytest.param(
        "in the past 2 months", (at(2026, 8, 19, 14, 30), NOW_TS), id="in-the-past"
    ),
    pytest.param("next 2 weeks", (NOW_TS, NOW_TS + 14 * DAY), id="next-2-weeks"),
    # -- embedded in a real query, case-insensitively
    pytest.param(
        "what did we decide about the migration in March 2023",
        (at(2023, 3, 1), at(2023, 4, 1)),
        id="sentence",
    ),
    pytest.param("In March 2023", (at(2023, 3, 1), at(2023, 4, 1)), id="mixed-case"),
    pytest.param("PAST 7 DAYS", (NOW_TS - 7 * DAY, NOW_TS), id="upper-case"),
    pytest.param("Last Summer", (at(2025, 6, 1), at(2025, 9, 1)), id="title-case"),
    pytest.param("TODAY", (at(2026, 10, 19), at(2026, 10, 20)), id="today-upper"),
]

# ---------------------------------------------------------------------------
# Open-ended expressions: the missing side is None
# ---------------------------------------------------------------------------

OPEN_CASES = [
    pytest.param("before June 2024", (None, at(2024, 6, 1)), id="before"),
    pytest.param("after June 2024", (at(2024, 6, 1), None), id="after"),
    pytest.param("since 2023", (at(2023, 1, 1), None), id="since"),
    pytest.param("SINCE 2023", (at(2023, 1, 1), None), id="since-upper"),
    pytest.param("until 2023", (None, at(2023, 1, 1)), id="until"),
    pytest.param("prior to 2023-05", (None, at(2023, 5, 1)), id="prior-to"),
    pytest.param("before yesterday", (None, at(2026, 10, 18)), id="before-relative"),
    pytest.param("everything since last month", (at(2026, 9, 1), None), id="since-rel"),
    pytest.param("since last summer", (at(2025, 6, 1), None), id="since-season"),
    # the cue's remainder is not temporal, so the year elsewhere still wins
    pytest.param(
        "after the migration, in 2023", (at(2023, 1, 1), None), id="cue-fallthrough"
    ),
]

# ---------------------------------------------------------------------------
# Numbers that are not times — every one of these must stay None
# ---------------------------------------------------------------------------

FALSE_POSITIVES = [
    "the 3 memory layers",
    "Python 3.11",
    "GPT-4",
    "section 4.2",
    "top 10 results",
    "rank 2",
    "we fixed 2 bugs in 3 files",
    "the 3 memory layers are dense and BM25",
    "what did we learn from the 3 memory layers",
    "score 0.87 for chunk 4",
    "version 1.10.0 of qdrant-client",
    "list the top 5 memories by rank",
    "qwen3:8b answered in 12 tokens",
]


class TestParseTimeWindow:
    @pytest.mark.parametrize("query,expected", PARSE_CASES)
    def test_bounded_window(self, query: str, expected: tuple[float, float]) -> None:
        assert parse_time_window(query, NOW) == expected

    @pytest.mark.parametrize("query,expected", OPEN_CASES)
    def test_open_ended_window(
        self, query: str, expected: tuple[float | None, float | None]
    ) -> None:
        assert parse_time_window(query, NOW) == expected

    @pytest.mark.parametrize("query", FALSE_POSITIVES)
    def test_number_shaped_non_times_stay_none(self, query: str) -> None:
        assert parse_time_window(query, NOW) is None

    @pytest.mark.parametrize("query,expected", PARSE_CASES)
    def test_windows_are_ordered_and_half_open(
        self, query: str, expected: tuple[float, float]
    ) -> None:
        start, end = parse_time_window(query, NOW)
        assert start is not None and end is not None
        assert start < end

    @pytest.mark.parametrize("query,expected", PARSE_CASES)
    def test_parse_is_pure(self, query: str, expected: tuple[float, float]) -> None:
        """Same input, same output — nothing here reads a clock or a cache."""
        assert parse_time_window(query, NOW) == parse_time_window(query, NOW)

    def test_adjacent_calendar_units_tile_exactly(self) -> None:
        """Half-open windows must not overlap or leave a gap."""
        last_year = parse_time_window("last year", NOW)
        this_year = parse_time_window("this year", NOW)
        assert last_year is not None and this_year is not None
        assert last_year[1] == this_year[0]
        assert this_year[1] - this_year[0] == 365 * DAY  # 2026 is not a leap year

    def test_naive_now_is_read_as_utc(self) -> None:
        naive = datetime(2026, 10, 19, 14, 30)
        assert parse_time_window("today", naive) == (at(2026, 10, 19), at(2026, 10, 20))

    def test_aware_now_is_converted_to_utc(self) -> None:
        plus_two = datetime(2026, 10, 19, 16, 30, tzinfo=timezone(timedelta(hours=2)))
        # 16:30+02:00 is 14:30 UTC, still the 19th, so "today" is unchanged.
        assert parse_time_window("today", plus_two) == (
            at(2026, 10, 19),
            at(2026, 10, 20),
        )

    def test_default_now_is_a_utc_day(self) -> None:
        """With no injection the clock is read once and a day is 24h wide."""
        window = parse_time_window("today")
        assert window is not None
        start, end = window
        assert end - start == DAY
        assert start % DAY == 0  # a midnight boundary, whatever today is

    @pytest.mark.parametrize("query", ["", "   ", "\n"])
    def test_empty_query_is_none(self, query: str) -> None:
        assert parse_time_window(query, NOW) is None

    def test_non_string_query_is_none(self) -> None:
        assert parse_time_window(None, NOW) is None  # type: ignore[arg-type]


class TestRangeFilter:
    def test_both_bounds_none_is_no_filter(self) -> None:
        assert range_filter(None, None) is None

    def test_lower_bound_only(self) -> None:
        assert range_filter(100.0, None) == {
            "key": "timestamp",
            "range": {"gte": 100.0},
        }

    def test_upper_bound_only(self) -> None:
        assert range_filter(None, 200.0) == {
            "key": "timestamp",
            "range": {"lte": 200.0},
        }

    def test_both_bounds(self) -> None:
        assert range_filter(100.0, 200.0) == {
            "key": "timestamp",
            "range": {"gte": 100.0, "lte": 200.0},
        }

    def test_key_is_overridable(self) -> None:
        assert range_filter(1.0, 2.0, key="ts") == {
            "key": "ts",
            "range": {"gte": 1.0, "lte": 2.0},
        }

    @pytest.mark.parametrize("query", ["in 2023", "last month", "before June 2024"])
    def test_filter_round_trips_a_parsed_window(self, query: str) -> None:
        start, end = parse_time_window(query, NOW)
        flt = range_filter(start, end)
        assert flt is not None
        assert flt["key"] == "timestamp"
        assert flt["range"].get("gte") == start
        assert flt["range"].get("lte") == end


# ---------------------------------------------------------------------------
# bucket_spread
# ---------------------------------------------------------------------------


class TestBucketSpread:
    def test_promotes_one_hit_per_bucket(self) -> None:
        """A mid-window cluster cannot crowd out the sparse stretches.

        Window [0, 12] in six 2-wide buckets; A/B/C share bucket 2, D is alone
        in bucket 0 and E alone in bucket 4. With one hit per bucket, D leads
        despite the lowest-but-one score, and A's same-bucket rivals B and C
        fall to the tail in score order.
        """
        items = [
            (0.9, 5.0, "A"),
            (0.8, 5.5, "B"),
            (0.7, 5.9, "C"),
            (0.6, 1.0, "D"),
            (0.5, 9.0, "E"),
        ]
        got = bucket_spread(items, 0.0, 12.0, buckets=6, per_bucket=1)
        assert [item[2] for item in got] == ["D", "A", "E", "B", "C"]

    def test_per_bucket_takes_the_strongest_first(self) -> None:
        items = [
            (0.9, 7.0, "X"),
            (0.8, 7.0, "Y"),
            (0.7, 2.0, "Z"),
            (0.6, 9.0, "W"),
        ]
        got = bucket_spread(items, 0.0, 10.0, buckets=2, per_bucket=2)
        assert [item[2] for item in got] == ["Z", "X", "Y", "W"]

    def test_no_window_uses_the_item_span(self) -> None:
        """start/end None: the timestamps themselves define the window."""
        items = [
            (0.9, 100.0, "P"),
            (0.85, 105.0, "Q"),
            (0.8, 110.0, "R"),
            (0.75, 400.0, "S"),
        ]
        got = bucket_spread(items, None, None, buckets=2, per_bucket=1)
        assert [item[2] for item in got] == ["P", "S", "Q", "R"]

    def test_half_open_window_uses_the_item_span_on_the_open_side(self) -> None:
        items = [
            (0.9, 100.0, "P"),
            (0.85, 105.0, "Q"),
            (0.8, 110.0, "R"),
            (0.75, 400.0, "S"),
        ]
        got = bucket_spread(items, 100.0, None, buckets=2, per_bucket=1)
        assert [item[2] for item in got] == ["P", "S", "Q", "R"]

    def test_spread_surfaces_a_sparse_stretch_of_the_year(self) -> None:
        """The Hindsight parity case: ten September hits, one from March.

        A "what happened in 2023" answer must lead with a memory from each
        stretch of the year, so the lone March memory outranks ten
        higher-scoring September ones.
        """
        dense = [
            (0.99 - index * 0.001, at(2023, 9, 10 + index % 5), f"sep{index}")
            for index in range(10)
        ]
        items = dense + [(0.10, at(2023, 3, 1), "march")]
        got = bucket_spread(
            items, at(2023, 1, 1), at(2024, 1, 1), buckets=6, per_bucket=1
        )
        assert got[0][2] == "march"
        assert len(got) == len(items)
        assert sorted(item[2] for item in got) == sorted(item[2] for item in items)

    def test_result_is_a_permutation_of_the_input(self) -> None:
        items = [
            (0.9 - index * 0.01, float(index * 7), f"m{index}") for index in range(12)
        ]
        got = bucket_spread(items, 0.0, 100.0, buckets=4, per_bucket=2)
        assert len(got) == len(items)
        assert sorted(id(item) for item in got) == sorted(id(item) for item in items)

    def test_tail_keeps_the_original_score_order(self) -> None:
        items = [
            (0.9, 5.0, "A"),
            (0.8, 5.5, "B"),
            (0.7, 5.9, "C"),
            (0.6, 1.0, "D"),
            (0.5, 9.0, "E"),
        ]
        got = bucket_spread(items, 0.0, 12.0, buckets=6, per_bucket=1)
        assert got[3:] == [items[1], items[2]]  # B, C in score order

    def test_is_deterministic(self) -> None:
        items = [
            (0.9, 5.0, "A"),
            (0.8, 5.5, "B"),
            (0.7, 1.0, "C"),
            (0.6, 9.0, "D"),
        ]
        first = bucket_spread(items, 0.0, 12.0)
        second = bucket_spread(list(items), 0.0, 12.0)
        assert first == second

    def test_equal_scores_keep_input_order(self) -> None:
        items = [(0.5, float(index), f"m{index}") for index in range(6)]
        got = bucket_spread(items, 0.0, 5.0, buckets=6, per_bucket=1)
        assert got == items

    def test_empty_items(self) -> None:
        assert bucket_spread([], 0.0, 10.0) == []
        assert bucket_spread([], None, None) == []

    def test_zero_span_returns_input_order(self) -> None:
        items = [(0.9, 5.0, "A"), (0.8, 5.0, "B")]
        assert bucket_spread(items, 5.0, 5.0) == items
        assert bucket_spread(items, None, None) == items

    def test_single_item(self) -> None:
        items = [(0.9, 5.0, "only")]
        assert bucket_spread(items, None, None) == items

    def test_items_outside_the_window_are_kept(self) -> None:
        """Clamped into the end buckets, never dropped."""
        items = [(0.9, 0.0, "early"), (0.8, 150.0, "late")]
        got = bucket_spread(items, 100.0, 200.0, buckets=2, per_bucket=1)
        assert sorted(item[2] for item in got) == ["early", "late"]

    def test_window_given_backwards_is_normalised(self) -> None:
        items = [(0.9, 5.0, "A"), (0.8, 1.0, "B")]
        assert bucket_spread(
            items, 12.0, 0.0, buckets=6, per_bucket=1
        ) == bucket_spread(items, 0.0, 12.0, buckets=6, per_bucket=1)

    def test_zero_buckets_and_zero_per_bucket_are_safe(self) -> None:
        """Degenerate knobs must not divide by zero or drop a hit."""
        items = [(0.9, 5.0, "A"), (0.8, 1.0, "B")]
        assert bucket_spread(items, 0.0, 12.0, buckets=0, per_bucket=0) == items
        # One bucket means one champion (the strongest) then the rest.
        assert bucket_spread(items, 0.0, 12.0, buckets=0, per_bucket=1) == items
