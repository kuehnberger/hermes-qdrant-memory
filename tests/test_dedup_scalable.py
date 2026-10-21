"""Unit tests for the scalable dedup neighbour rule: ``plan_neighbour_dedup``.

Pure stdlib — no server, no model. The ``--scalable`` path in
``scripts/semantic_dedup.py`` replaces the O(n^2) all-pairs clustering with a
per-point Qdrant neighbour search; this is the rule that decides, from each
point's returned neighbour set, which points get marked superseded. Kept as a
pure function so the newest-survives invariant is testable without a store.

The rule under test: process centres newest-first; a centre claims a neighbour
as superseded ONLY when the neighbour is strictly older and not already
claimed. A claimed centre is skipped, so a superseded point does not go on to
supersede others — the pass is the direct-neighbour relation above threshold,
NOT its transitive closure.
"""

from __future__ import annotations

from plugins.memory.qdrant.hygiene import DedupPlan, plan_neighbour_dedup


def record(point_id: str, epoch: float | None, neighbours):
    """A ``(id, epoch, neighbours)`` triple, neighbours as any iterable."""
    return (point_id, epoch, list(neighbours))


class TestPlanNeighbourDedup:
    """The mark-older-unclaimed-neighbours rule and its invariants."""

    def test_older_neighbour_is_claimed(self) -> None:
        plan = plan_neighbour_dedup([
            record("new", 200.0, ["old"]),
            record("old", 100.0, []),
        ])
        assert set(plan.claimed) == {"old"}
        assert set(plan.survivors) == {"new"}
        assert plan.superseded_by == {"old": "new"}

    def test_newer_point_never_marked_duplicate_of_older_centre(self) -> None:
        # An older centre whose neighbour list names a NEWER point must not
        # claim it; the newer point claims the older one instead. This is the
        # newest-survives invariant, and it holds even when the relation is
        # symmetric.
        plan = plan_neighbour_dedup([
            record("old", 100.0, ["new"]),
            record("new", 200.0, ["old"]),
        ])
        assert set(plan.claimed) == {"old"}
        assert set(plan.survivors) == {"new"}
        assert plan.superseded_by == {"old": "new"}

    def test_strictly_older_required_equal_epochs_neither_claims(self) -> None:
        # epoch(N) < epoch(C) is strict: two points of equal epoch never claim
        # each other, which is what keeps the survivor choice acyclic.
        plan = plan_neighbour_dedup([
            record("a", 100.0, ["b"]),
            record("b", 100.0, ["a"]),
        ])
        assert plan.claimed == []
        assert set(plan.survivors) == {"a", "b"}

    def test_claimed_centre_does_not_propagate(self) -> None:
        # A -> B -> C transitively, but once B is claimed by A it is skipped as
        # a centre, so C is NOT absorbed. Direct-neighbour, not transitive
        # closure — the documented approximation.
        plan = plan_neighbour_dedup([
            record("A", 300.0, ["B"]),
            record("B", 200.0, ["C"]),
            record("C", 100.0, []),
        ])
        assert set(plan.claimed) == {"B"}
        assert set(plan.survivors) == {"A", "C"}
        assert plan.superseded_by == {"B": "A"}

    def test_one_centre_absorbs_several_older_neighbours(self) -> None:
        plan = plan_neighbour_dedup([
            record("keep", 300.0, ["x", "y", "z"]),
            record("x", 100.0, []),
            record("y", 150.0, []),
            record("z", 200.0, []),
        ])
        assert set(plan.claimed) == {"x", "y", "z"}
        assert set(plan.survivors) == {"keep"}
        assert set(plan.groups["keep"]) == {"x", "y", "z"}

    def test_newest_of_a_symmetric_chain_is_the_survivor(self) -> None:
        # Every point lists every older one; only the newest survives, and each
        # older point is claimed by that newest centre.
        plan = plan_neighbour_dedup([
            record("c3", 400.0, ["c2", "c1", "c0"]),
            record("c2", 300.0, ["c3", "c1", "c0"]),
            record("c1", 200.0, ["c3", "c2", "c0"]),
            record("c0", 100.0, ["c3", "c2", "c1"]),
        ])
        assert set(plan.claimed) == {"c0", "c1", "c2"}
        assert set(plan.survivors) == {"c3"}
        assert plan.superseded_by == {"c0": "c3", "c1": "c3", "c2": "c3"}

    def test_neighbour_absent_from_records_is_not_claimed(self) -> None:
        # A neighbour id not among the scanned points has an unknown epoch, so
        # the strict "epoch(N) < epoch(C)" test cannot pass: it is NOT claimed.
        # Marking it would risk archiving a point that is actually newer — the
        # bug a bounded --scalable --limit run would otherwise hit.
        plan = plan_neighbour_dedup([
            record("keep", 100.0, ["ghost"]),
            record("ghost", 999.0, []),   # would-be, but ghost is newer anyway
        ])
        assert plan.claimed == []
        assert set(plan.survivors) == {"keep", "ghost"}

    def test_known_timestampless_neighbour_is_still_claimed(self) -> None:
        # A neighbour present in records with a None epoch is KNOWN-oldest (not
        # absent), so a dated centre does claim it.
        plan = plan_neighbour_dedup([
            record("keep", 100.0, ["noepoch"]),
            record("noepoch", None, []),
        ])
        assert set(plan.claimed) == {"noepoch"}
        assert set(plan.survivors) == {"keep"}

    def test_timeless_centre_claims_nothing(self) -> None:
        # A centre with no usable epoch (None) is oldest: it can never be
        # strictly newer, so it claims nothing and simply survives.
        plan = plan_neighbour_dedup([
            record("timeless", None, ["other"]),
            record("other", 100.0, []),
        ])
        assert plan.claimed == []
        assert set(plan.survivors) == {"timeless", "other"}

    def test_timeless_neighbour_is_claimable(self) -> None:
        # A dated centre CAN claim a neighbour whose epoch is None (oldest).
        plan = plan_neighbour_dedup([
            record("keep", 100.0, ["noepoch"]),
            record("noepoch", None, []),
        ])
        assert set(plan.claimed) == {"noepoch"}
        assert set(plan.survivors) == {"keep"}

    def test_disjoint_points_make_no_claims(self) -> None:
        plan = plan_neighbour_dedup([
            record("a", 200.0, []),
            record("b", 100.0, []),
        ])
        assert plan.claimed == []
        assert set(plan.survivors) == {"a", "b"}

    def test_input_order_does_not_change_the_plan(self) -> None:
        # The function sorts newest-first internally; shuffled input gives the
        # same survivors and claimed set.
        forward = plan_neighbour_dedup([
            record("new", 200.0, ["old"]),
            record("old", 100.0, []),
        ])
        shuffled = plan_neighbour_dedup([
            record("old", 100.0, []),
            record("new", 200.0, ["old"]),
        ])
        assert set(forward.claimed) == set(shuffled.claimed) == {"old"}
        assert set(forward.survivors) == set(shuffled.survivors) == {"new"}

    def test_returns_a_dedup_plan(self) -> None:
        plan = plan_neighbour_dedup([
            record("n", 2.0, ["o"]),
            record("o", 1.0, []),
        ])
        assert isinstance(plan, DedupPlan)
        assert set(plan._fields) == {
            "survivors", "claimed", "superseded_by", "groups",
        }
