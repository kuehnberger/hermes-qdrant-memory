"""Unit tests for ``hygiene``: decay scoring, exemption, dedup and merge.

Pure stdlib — no server, no model, no clock. Every expectation is built
from literals or from ``datetime`` in *this* file rather than re-derived by
the module under test.

Two halves:

* **Decay.** The score itself (half-life shape, clamping, the half-life
  validation) and the exemption rule — the two lanes (importance, human
  source) and the cases that must NOT exempt (bool/NaN importance, unknown
  or absent source), because a scan that exempts wrongly archives human
  notes or spares machine noise.
* **Dedup + merge.** Cosine boundary behaviour (``>=`` threshold, zero
  vectors carry no evidence, dimension mismatch is a caller bug), the
  greedy first-fit clustering with its running-mean centroid — including
  the transitive chain, where the first pair joins on 0.8, the third on
  the drifted centroid, although its similarity to the first is below
  threshold — and the merge field rules (longest text, newest epoch as a
  float, max importance, unanimous chunk indices, recorded sessions).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from plugins.memory.qdrant.hygiene import (
    DEFAULT_DUP_THRESHOLD,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_IMPORTANCE_THRESHOLD,
    cluster_near_duplicates,
    decay_score,
    is_decay_exempt,
    merge_payloads,
)


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    """Epoch seconds of a UTC instant, computed without the module's help."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


def point(**overrides: object) -> dict[str, object]:
    """A payload in the live store's shape: epoch float + kind + source."""
    payload: dict[str, object] = {
        "text": "remember the release date",
        "session_id": "s1",
        "timestamp": 1_700_000_000.0,
        "source": "cli",
        "kind": "fact",
    }
    payload.update(overrides)
    return payload


class TestDefaults:
    """The documented default knobs, pinned: they are the interface."""

    def test_documented_defaults(self) -> None:
        assert DEFAULT_HALF_LIFE_DAYS == 180.0
        assert DEFAULT_IMPORTANCE_THRESHOLD == 0.7
        assert DEFAULT_DUP_THRESHOLD == 0.92


# ---------------------------------------------------------------------------
# decay_score
# ---------------------------------------------------------------------------


class TestDecayScore:
    def test_fresh_point_scores_one(self) -> None:
        assert decay_score(0.0) == 1.0

    def test_one_half_life_scores_half(self) -> None:
        assert decay_score(DEFAULT_HALF_LIFE_DAYS) == 0.5

    def test_two_half_lives_score_quarter(self) -> None:
        assert decay_score(DEFAULT_HALF_LIFE_DAYS * 2) == 0.25

    def test_custom_half_life(self) -> None:
        assert decay_score(90.0, 90.0) == 0.5
        assert decay_score(45.0, 90.0) == pytest.approx(0.5 ** 0.5)

    def test_negative_age_clamped_to_zero(self) -> None:
        # A point cannot be younger than the scan; an unclamped negative
        # age would score above 1.0 and poison downstream thresholds.
        assert decay_score(-5.0) == 1.0

    def test_monotonic_decreasing(self) -> None:
        scores = [decay_score(age) for age in (0.0, 30.0, 180.0, 360.0, 720.0)]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 < score <= 1.0 for score in scores)

    @pytest.mark.parametrize("half_life", [0.0, -1.0, -180.0])
    def test_non_positive_half_life_raises(self, half_life: float) -> None:
        # A zero or negative half-life is division by zero or a score that
        # grows with age — a caller bug, not a value to score.
        with pytest.raises(ValueError, match="half_life_days must be positive"):
            decay_score(10.0, half_life)


# ---------------------------------------------------------------------------
# is_decay_exempt
# ---------------------------------------------------------------------------


class TestIsDecayExempt:
    @pytest.mark.parametrize("source", ["cli", "tui", "CLI", "Tui"])
    def test_human_sources_exempt(self, source: str) -> None:
        assert is_decay_exempt(point(source=source)) is True

    @pytest.mark.parametrize(
        "source",
        ["cron", "subagent", "kanban", "sync_turn", "nous_github", "hermes_docs"],
    )
    def test_machine_sources_not_exempt(self, source: str) -> None:
        assert is_decay_exempt(point(source=source)) is False

    def test_unknown_source_not_exempt(self) -> None:
        # "Unknown" is not evidence a human asked to keep the point; the
        # scan's job is pruning machine noise.
        assert is_decay_exempt(point(source="some_future_tool")) is False

    def test_absent_source_not_exempt(self) -> None:
        payload = point()
        del payload["source"]
        assert is_decay_exempt(payload) is False

    def test_importance_at_threshold_exempt(self) -> None:
        # Machine source: only the importance lane can exempt here.
        assert is_decay_exempt(point(source="cron", importance=0.7)) is True

    def test_importance_above_threshold_exempt(self) -> None:
        assert is_decay_exempt(point(source="cron", importance=0.95)) is True

    def test_importance_below_threshold_not_exempt(self) -> None:
        assert is_decay_exempt(point(source="cron", importance=0.69)) is False

    def test_custom_threshold(self) -> None:
        assert (
            is_decay_exempt(
                point(source="cron", importance=0.4), importance_threshold=0.3
            )
            is True
        )
        assert (
            is_decay_exempt(
                point(source="cron", importance=0.4), importance_threshold=0.5
            )
            is False
        )

    def test_importance_outranks_machine_source(self) -> None:
        assert is_decay_exempt(point(source="cron", importance=0.8)) is True

    @pytest.mark.parametrize(
        "raw", [True, False, "high", float("nan"), float("inf"), None]
    )
    def test_non_numeric_importance_ignored(self, raw: object) -> None:
        # bool is an int in Python; reading True as 1.0 would exempt every
        # stray flag. NaN/inf make every comparison silently True or False.
        # Machine source: the verdict must rest on importance alone.
        assert is_decay_exempt(point(source="cron", importance=raw)) is False

    @pytest.mark.parametrize("payload", [None, [], "text", 42])
    def test_non_mapping_payload_not_exempt(self, payload: object) -> None:
        assert is_decay_exempt(payload) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# cluster_near_duplicates
# ---------------------------------------------------------------------------


class TestClusterNearDuplicates:
    def test_identical_vectors_cluster(self) -> None:
        clusters = cluster_near_duplicates(
            [([1.0, 0.0], point(text="a")), ([1.0, 0.0], point(text="a again"))]
        )
        assert len(clusters) == 1
        assert [str(p.get("text")) for p in clusters[0]] == ["a", "a again"]

    def test_orthogonal_vectors_no_cluster(self) -> None:
        assert cluster_near_duplicates([
            ([1.0, 0.0], point(text="a")),
            ([0.0, 1.0], point(text="b")),
        ]) == []

    def test_exact_threshold_boundary_joins(self) -> None:
        # cos((1,0), (0.6,0.8)) == 0.6 exactly: >= is the contract.
        items = [([1.0, 0.0], point(text="a")), ([0.6, 0.8], point(text="b"))]
        assert len(cluster_near_duplicates(items, threshold=0.6)) == 1
        assert cluster_near_duplicates(items, threshold=0.61) == []

    def test_singletons_omitted(self) -> None:
        # A lone point has nothing to merge; only groups of 2+ are returned.
        clusters = cluster_near_duplicates([
            ([1.0, 0.0], point(text="a")),
            ([0.0, 1.0], point(text="b")),
        ])
        assert clusters == []

    def test_chain_joins_on_drifted_centroid(self) -> None:
        # cos(a,b)=0.8 and cos(b,c)=0.96 clear 0.75, but cos(a,c)=0.6 does
        # not: the running-mean centroid drags c in. Transitive chaining is
        # the property that makes near-dup groups bigger than pairs.
        clusters = cluster_near_duplicates([
            ([1.0, 0.0], point(text="a")),
            ([0.8, 0.6], point(text="b")),
            ([0.6, 0.8], point(text="c")),
        ], threshold=0.75)
        groups = [[str(p.get("text")) for p in group] for group in clusters]
        assert groups == [["a", "b", "c"]]

    def test_first_fit_is_order_dependent(self) -> None:
        # Same vectors, different order, different grouping: the first member
        # defines a cluster's centre, and the running mean drifts it. In
        # forward order all three join (c lands on the drifted centroid at
        # ~0.82, above 0.8, despite cos(a,c)=0.6); with c first, a stays
        # alone and the cluster is {c, b}. Callers sort before clustering.
        vectors = [([1.0, 0.0], "a"), ([0.8, 0.6], "b"), ([0.6, 0.8], "c")]
        forward = cluster_near_duplicates(
            [(v, point(text=t)) for v, t in vectors], threshold=0.8
        )
        reordered = cluster_near_duplicates(
            [(v, point(text=t)) for v, t in (vectors[2], *vectors[:2])],
            threshold=0.8,
        )
        assert [[str(p.get("text")) for p in g] for g in forward] == [["a", "b", "c"]]
        assert [[str(p.get("text")) for p in g] for g in reordered] == [["c", "b"]]

    def test_zero_vectors_never_duplicates(self) -> None:
        # No direction, no evidence: cosine 0.0, below any threshold.
        assert cluster_near_duplicates([
            ([0.0, 0.0, 0.0], point(text="a")),
            ([0.0, 0.0, 0.0], point(text="b")),
        ]) == []

    def test_empty_vectors_never_duplicates(self) -> None:
        assert cluster_near_duplicates([
            ([], point(text="a")),
            ([], point(text="b")),
        ]) == []

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal dimensions"):
            cluster_near_duplicates([
                ([1.0, 0.0], point(text="a")),
                ([1.0, 0.0, 0.0], point(text="b")),
            ])

    @pytest.mark.parametrize("threshold", [1.5, -1.5, float("nan")])
    def test_threshold_outside_cosine_range_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="within"):
            cluster_near_duplicates([], threshold=threshold)

    def test_empty_input_returns_empty(self) -> None:
        assert cluster_near_duplicates([]) == []

    def test_bare_embeddings_accepted(self) -> None:
        # A bare 2-dim embedding is not a pair: its second element is a
        # float, not a mapping. The cluster then carries None payloads.
        clusters = cluster_near_duplicates([[1.0, 0.0], [1.0, 0.0]])
        assert clusters == [[None, None]]

    def test_pairs_as_two_element_lists(self) -> None:
        clusters = cluster_near_duplicates([
            [[1.0, 0.0], point(text="a")],
            [[1.0, 0.0], point(text="b")],
        ])
        assert [[str(p.get("text")) for p in g] for g in clusters] == [["a", "b"]]


# ---------------------------------------------------------------------------
# merge_payloads
# ---------------------------------------------------------------------------


class TestMergePayloads:
    def test_newest_epoch_survives_as_float(self) -> None:
        merged = merge_payloads([
            point(text="older", timestamp=100.0),
            point(text="newer", timestamp=200.0),
        ])
        assert merged["timestamp"] == 200.0
        assert isinstance(merged["timestamp"], float)
        assert merged["timestamp_iso"] == datetime.fromtimestamp(
            200.0, tz=UTC
        ).isoformat()

    def test_iso_timestamp_beats_older_epoch(self) -> None:
        # The live store mixes epoch floats and ISO strings; the merge must
        # compare them as instants, not as raw values. Equal-length texts so
        # the longest-text rule does not mask the timestamp comparison.
        merged = merge_payloads([
            point(
                text="iso-version",
                timestamp=datetime(2026, 10, 19, tzinfo=UTC).isoformat(),
            ),
            point(text="epoch-stamp", timestamp=at(2026, 10, 18)),
        ])
        assert merged["timestamp"] == at(2026, 10, 19)
        assert merged["text"] == "iso-version"

    def test_unparseable_timestamps_sort_last(self) -> None:
        # A member with no timestamp must not count as the newest: the
        # survivor's timestamp comes from the dated member (100.0), not
        # from the timeless one. Text follows the longest-text rule
        # independently, so it is the timeless member's.
        timeless = point(text="no time")
        del timeless["timestamp"]
        merged = merge_payloads([timeless, point(text="old", timestamp=100.0)])
        assert merged["timestamp"] == 100.0
        assert merged["text"] == "no time"

    def test_no_parseable_timestamp_preserves_raw(self) -> None:
        garbage = point(text="garbage", timestamp="not a time")
        absent = point(text="absent")
        del absent["timestamp"]
        merged = merge_payloads([garbage, absent])
        assert merged["timestamp"] == "not a time"
        assert "timestamp_iso" not in merged

    def test_longest_text_wins(self) -> None:
        merged = merge_payloads([
            point(text="short", timestamp=200.0),
            point(text="a much longer surviving text", timestamp=100.0),
        ])
        assert merged["text"] == "a much longer surviving text"
        assert merged["timestamp"] == 200.0

    def test_equal_length_text_newest_wins(self) -> None:
        merged = merge_payloads([
            point(text="aaa", timestamp=100.0),
            point(text="bbb", timestamp=200.0),
        ])
        assert merged["text"] == "bbb"

    def test_absent_text_not_added(self) -> None:
        first = point()
        second = point()
        del first["text"]
        del second["text"]
        merged = merge_payloads([first, second])
        assert "text" not in merged

    def test_importance_max_survives(self) -> None:
        merged = merge_payloads([
            point(importance=0.3),
            point(importance=0.9),
            point(importance=0.5),
        ])
        assert merged["importance"] == 0.9

    def test_single_session_id_not_recorded(self) -> None:
        merged = merge_payloads([point(), point(session_id="s1")])
        assert "merged_session_ids" not in merged
        assert merged["session_id"] == "s1"

    def test_absorbed_session_ids_sorted(self) -> None:
        merged = merge_payloads([
            point(session_id="s1"),
            point(session_id="s3"),
            point(session_id="s2"),
        ])
        assert merged["session_id"] == "s1"
        assert merged["merged_session_ids"] == ["s2", "s3"]

    def test_unanimous_chunk_index_kept(self) -> None:
        merged = merge_payloads([
            point(chunk_index=2, sub_index=0),
            point(chunk_index=2, sub_index=0),
        ])
        assert merged["chunk_index"] == 2
        assert merged["sub_index"] == 0

    def test_disagreeing_chunk_index_dropped(self) -> None:
        merged = merge_payloads([
            point(chunk_index=1),
            point(chunk_index=2),
        ])
        assert "chunk_index" not in merged

    def test_primary_scalars_preserved(self) -> None:
        merged = merge_payloads([
            point(source="cron", kind="turn", turn_author="assistant"),
            point(text="x"),
        ])
        assert merged["source"] == "cron"
        assert merged["kind"] == "turn"
        assert merged["turn_author"] == "assistant"

    def test_singleton_group_normalized(self) -> None:
        merged = merge_payloads([point(timestamp="2026-10-19T00:00:00+00:00")])
        assert merged["timestamp"] == at(2026, 10, 19)
        assert merged["timestamp_iso"] == datetime.fromtimestamp(
            at(2026, 10, 19), tz=UTC
        ).isoformat()

    def test_empty_group_raises(self) -> None:
        with pytest.raises(ValueError, match="empty group"):
            merge_payloads([])

    def test_non_mapping_member_raises(self) -> None:
        with pytest.raises(ValueError, match="payload mappings"):
            merge_payloads([point(), "not a mapping"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# End-to-end: cluster feeds merge
# ---------------------------------------------------------------------------


class TestDedupPipeline:
    def test_near_duplicate_pair_merges_into_one_survivor(self) -> None:
        items = [
            (
                [1.0, 0.0],
                point(text="the release is on friday", timestamp=100.0),
            ),
            (
                [0.99, 0.05],
                point(text="the release is on friday, confirmed", timestamp=200.0),
            ),
        ]
        clusters = cluster_near_duplicates(items, threshold=0.9)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

        merged = merge_payloads(clusters[0])
        assert merged["text"] == "the release is on friday, confirmed"
        assert merged["timestamp"] == 200.0

    def test_distinct_points_never_merge(self) -> None:
        items = [
            ([1.0, 0.0], point(text="a", timestamp=100.0)),
            ([0.0, 1.0], point(text="b", timestamp=200.0)),
        ]
        assert cluster_near_duplicates(items) == []
