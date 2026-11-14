"""Memory hygiene primitives (rival-hardening: memory-os parity).

Pure, deterministic scoring and clustering for the two maintenance jobs:
a decay scan (archive stale machine-generated points) and a semantic dedup
(merge near-duplicate points). Kept separate from the scripts that touch the
live store so the arithmetic is testable without a server or a model.

Why the rules are what they are
-------------------------------
Importance is optional and absent from the live store, so absence means
"not exempt", never "important": a scan that assumed importance 1.0 for
unlabelled points would archive nothing, and one that assumed 0.0 would
archive human notes. Authorship comes from ``source`` — a human at ``cli``
or ``tui`` versus automation (cron, subagent, kanban, sync_turn,
nous_github, hermes_docs). An unrecognised source counts as
machine-generated, because the scan exists to prune machine noise and
"unknown" is not evidence that a human asked to keep the point.

The merge survivor emits the newest parseable timestamp as an epoch float:
Qdrant range filters are numeric-only and the live store mixes epoch floats
(~91%), absence (~9%) and a few ISO strings, so normalising here keeps one
format law for the whole module.

Interface:

    decay_score(age_days, half_life_days=180.0) -> float
    is_decay_exempt(payload, *, importance_threshold=0.7) -> bool
    cluster_near_duplicates(items, *, threshold=0.92) -> list[list]
    cluster_by_neighbour_edges(edges, *, epochs, threshold=0.92) -> list[list[str]]
    merge_payloads(group) -> dict
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, NamedTuple

__all__ = [
    "decay_score",
    "is_decay_exempt",
    "cluster_near_duplicates",
    "plan_neighbour_dedup",
    "DedupPlan",
    "merge_payloads",
]

#: Default decay half-life in days. Six months: old enough that routine
#: context has served its purpose, young enough that a recurring weekly
#: pattern still outruns it.
DEFAULT_HALF_LIFE_DAYS = 180.0

#: importance at or above which a point survives every decay scan. The
#: field is absent by default in the live store; when it does arrive it is
#: an explicit, modelled keep signal.
DEFAULT_IMPORTANCE_THRESHOLD = 0.7

#: Cosine similarity at or above which two points count as duplicates.
DEFAULT_DUP_THRESHOLD = 0.92

#: Sources a human typed at. Every other observed source is automation.
HUMAN_SOURCES = frozenset({"cli", "tui"})

#: Sentinel distinguishing "key absent" from "key present as None".
_MISSING = object()


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


def decay_score(
    age_days: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS
) -> float:
    """Exponential half-life decay: ``0.5 ** (age_days / half_life_days)``.

    A point one half-life old scores 0.5; two half-lives, 0.25. The score
    is a *retention weight*, not a verdict — callers compare it against
    their own threshold, and a steeper half-life is a one-line change, not
    a re-tuned formula.

    ``age_days`` is clamped at 0: a point cannot be younger than the scan
    measuring it, and an unclamped negative age would score above 1.0,
    reading as "more than fresh" and breaking every downstream threshold.
    A non-positive half-life is a caller bug — division by zero, or a
    score that *grows* with age — so it raises instead of returning a
    number that silently poisons everything it feeds.
    """
    if half_life_days <= 0:
        raise ValueError(
            f"half_life_days must be positive, got {half_life_days!r}"
        )
    age = age_days if age_days > 0 else 0.0
    return 0.5 ** (age / half_life_days)


def _importance(payload: Mapping[str, Any]) -> float | None:
    """The payload's importance as a finite float, or None.

    ``bool`` is excluded even though it is an int in Python: ``True`` is
    not a modelled "maximum importance", and reading it as 1.0 would exempt
    every point carrying a stray flag. NaN/inf are refused because every
    comparison against them is silently True or False — a data-dependent
    verdict no operator can audit.
    """
    raw = payload.get("importance")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def is_decay_exempt(
    payload: Mapping[str, Any] | None,
    *,
    importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD,
) -> bool:
    """True when a payload must survive the decay scan.

    Either lane exempts:

    1. ``importance`` at or above ``importance_threshold`` — an explicit,
       modelled keep signal that outranks authorship;
    2. a human ``source`` (``cli``, ``tui``). Cron, subagent, kanban,
       sync_turn, nous_github, hermes_docs and every unrecognised value are
       machine-generated: the scan exists to prune their stale output, and
       an unknown source is not evidence that a human asked to keep the
       point.

    Absence is not importance: with no importance field — the live default —
    only authorship can exempt, so the scan never archives human notes and
    never spares machine ones.
    """
    if not isinstance(payload, Mapping):
        return False
    importance = _importance(payload)
    if importance is not None and importance >= importance_threshold:
        return True
    source = payload.get("source")
    if isinstance(source, str) and source.strip().lower() in HUMAN_SOURCES:
        return True
    return False


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, or 0.0 when the vectors give no evidence.

    0.0 — not an error — for empty or zero-norm inputs: an all-zero vector
    has no direction, and calling it "orthogonal" would be as arbitrary as
    calling it similar. Mismatched lengths DO raise: a similarity computed
    across different dimensions can manufacture a duplicate that is not one,
    and an operator merging points on that basis needs to know the input is
    corrupt rather than receive a confident wrong answer.

    ``math.fsum`` makes each product-sum exact, so the score does not depend
    on summation order: the same pair of vectors scores identically however
    the caller batches them.
    """
    if len(a) != len(b):
        raise ValueError(
            f"cosine needs equal dimensions, got {len(a)} and {len(b)}"
        )
    if not a:
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.fsum(x * x for x in a)
    norm_b = math.fsum(y * y for y in b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _split_item(
    item: tuple[Sequence[float], Mapping[str, Any]] | Sequence[float],
) -> tuple[Sequence[float], Mapping[str, Any] | None]:
    """Accept ``(embedding, payload)`` pairs or bare embeddings.

    A pair is a 2-tuple/list whose second element is a mapping; a 2-dim
    embedding ``[x, y]`` has a float second element and reads as bare.
    """
    if (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[1], Mapping)
    ):
        embedding, payload = item
        return embedding, payload
    return item, None


def cluster_near_duplicates(
    items: Sequence[tuple[Sequence[float], Mapping[str, Any]]],
    *,
    threshold: float = DEFAULT_DUP_THRESHOLD,
) -> list[list[Mapping[str, Any]]]:
    """Group items whose embeddings are near-duplicates of each other.

    Greedy first-fit in input order: each item joins the first cluster whose
    centroid clears ``threshold``, else founds a new one. Order-dependent by
    construction — the first member of a group defines its centre — so a
    caller that cares which points group together sorts first (newest first
    keeps the eventual survivor current). Deterministic for a fixed input.

    The centroid is the running mean of member vectors, updated exactly
    (``mean + (v - mean) / n``), so a cluster's centre drifts toward the
    bulk of its members instead of staying pinned to whichever point arrived
    first.

    Returns only clusters of two or more payloads — a singleton has nothing
    to merge, and ``merge_payloads`` is the next step of the pipeline. Items
    with empty or zero vectors are never duplicates (cosine 0.0, below any
    sensible threshold): there is no embedding evidence to merge on.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be within [-1.0, 1.0], got {threshold!r}"
        )
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []
    for index, item in enumerate(items):
        embedding, _payload = _split_item(item)
        target = -1
        for position, centroid in enumerate(centroids):
            if _cosine(embedding, centroid) >= threshold:
                target = position
                break
        if target < 0:
            clusters.append([index])
            centroids.append(list(embedding))
            continue
        members = clusters[target]
        members.append(index)
        count = len(members)
        centroids[target] = [
            component + (value - component) / count
            for component, value in zip(centroids[target], embedding, strict=True)
        ]
    return [
        [_split_item(items[index])[1] for index in cluster]
        for cluster in clusters
        if len(cluster) >= 2
    ]


class DedupPlan(NamedTuple):
    """Result of a neighbour-search dedup pass (see ``plan_neighbour_dedup``).

    - ``survivors``:     point ids NOT marked superseded, newest-first.
    - ``claimed``:       point ids marked superseded, in the order claimed.
    - ``superseded_by``: mapping of each superseded id -> the kept neighbour
                         that absorbed it (the newest centre listing it that
                         was itself newer).
    - ``groups``:        mapping of each kept centre -> the older ids it
                         absorbed, in claim order — ready to feed
                         ``merge_payloads``.
    """

    survivors: list
    claimed: list
    superseded_by: dict
    groups: dict


def _norm_epoch(epoch: float | None) -> float:
    """Comparable epoch; None/NaN/inf collapse to -inf (treated as oldest).

    The live store mixes epoch floats, ISO strings and absence, and every
    "newest" rule here is numeric-only. A missing or unparseable stamp is
    therefore the OLDEST possible: a centre with no usable timestamp can never
    be strictly newer than anyone, so it claims nothing, and a neighbour with
    none is claimable by any dated centre.
    """
    if epoch is None:
        return float("-inf")
    value = float(epoch)
    return value if math.isfinite(value) else float("-inf")


def plan_neighbour_dedup(
    records: Iterable[tuple[str, float | None, Iterable[str]]],
) -> DedupPlan:
    """Plan a Qdrant neighbour-search dedup from per-point neighbour sets.

    Pure counterpart to the ``--scalable`` path in
    ``scripts/semantic_dedup.py``: where ``cluster_near_duplicates`` needs
    every dense vector in memory at once (all-pairs, O(n^2)), this needs only
    each point's ``id``, its epoch, and the ids Qdrant returned for a per-point
    neighbour search at the similarity threshold. It never touches a vector,
    so the caller can run a full-store pass without holding the store's
    embeddings in RAM.

    The rule, applied newest-first: a centre ``C`` claims a neighbour ``N`` as
    superseded ONLY when ``N`` is strictly OLDER than ``C`` and not already
    claimed. So the newest point in a near-duplicate run always survives and no
    cycle can form — a newer point is never marked a duplicate of an older
    centre. A centre that is itself already claimed is skipped, so a
    superseded point does not go on to supersede others: the pass is the
    direct-neighbour relation above threshold, NOT its transitive closure.

    That non-transitivity is the one approximation versus exact all-pairs
    clustering. It is safe at the high default threshold (0.92) — two points
    each above threshold of a survivor are near-duplicates of it — and the
    whole pass is reversible (the script's ``--unmerge`` clears the flags).

    ``records`` is an iterable of ``(id, epoch, neighbours)``:

    - ``id``         -> the point id (stringified);
    - ``epoch``      -> its numeric timestamp, or None (counted as oldest; see
                        ``_norm_epoch``);
    - ``neighbours`` -> the ids a per-point ``query_points`` returned at the
                        threshold for this centre. A neighbour newer than the
                        centre is ignored. A neighbour whose epoch is unknown
                        (absent from ``records``) is NOT claimed: the rule
                        needs ``epoch(N) < epoch(C)``, which is unknowable for
                        an unscanned point, and marking one could archive a
                        newer point. In a full store scan every point is
                        scrolled, so this case does not arise there.

    Deterministic for a fixed input: the records are sorted newest-first, with
    the stringified id as a stable tie-break, before processing.
    """
    loaded: list[tuple[str, float, list[str]]] = [
        (str(pid), _norm_epoch(epoch), [str(n) for n in neigh])
        for pid, epoch, neigh in records
    ]
    epoch_by_id = {pid: epoch for pid, epoch, _neigh in loaded}
    # Newest first; a stable id tie-break keeps equal-epoch ties deterministic.
    loaded.sort(key=lambda rec: rec[0])
    loaded.sort(key=lambda rec: rec[1], reverse=True)

    claimed: set[str] = set()
    claimed_order: list[str] = []
    superseded_by: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for pid, epoch, neigh in loaded:
        if pid in claimed:
            continue
        for neighbour in neigh:
            if neighbour == pid or neighbour in claimed:
                continue
            neighbour_epoch = epoch_by_id.get(neighbour)
            if neighbour_epoch is None:
                continue  # unscanned point — epoch unknown, so never supersede
            if neighbour_epoch < epoch:
                claimed.add(neighbour)
                claimed_order.append(neighbour)
                superseded_by[neighbour] = pid
                groups.setdefault(pid, []).append(neighbour)

    survivors = [pid for pid, _epoch, _neigh in loaded if pid not in claimed]
    return DedupPlan(
        survivors=survivors,
        claimed=claimed_order,
        superseded_by=superseded_by,
        groups=groups,
    )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _payload_epoch(payload: Mapping[str, Any]) -> float | None:
    """Best-effort numeric timestamp from a point payload.

    The live store carries three shapes — epoch floats (the backfill's
    format), ISO strings (the sync_turn path) and absence (~9%) — and every
    time comparison downstream is numeric-only, so normalising here keeps
    one format law for the module. Absent or unparseable returns None, never
    0.0: a 0.0 would sort as 1970 and silently win every "oldest" comparison.
    A naive ISO string is read as UTC so the result never depends on the
    host's timezone. Mirrors the provider's ``_payload_epoch``; duplicated
    so this module stays import-light for the maintenance scripts
    (``datetime`` is stdlib, the provider is not).
    """
    raw = payload.get("timestamp")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    if isinstance(raw, str) and raw.strip():
        try:
            stamp = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp.timestamp()
    return None


def _text_of(payload: Mapping[str, Any]) -> str:
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def merge_payloads(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse one duplicate cluster into the single payload that survives.

    Which member wins each field:

    - ``text`` — the longest member text; near-duplicates usually differ by
      truncation and the longest form holds the most content. Ties break
      toward the newest timestamp, then the earlier input position, so the
      result is deterministic.
    - ``timestamp`` — the newest parseable epoch, emitted as a float (the
      store's dominant, filter-compatible form) with ``timestamp_iso``
      regenerated beside it so the pair stays consistent. If no member has a
      parseable timestamp, the primary's raw value survives untouched.
    - ``importance`` — the maximum present: absorbing a group must never
      discard the strongest keep signal among its members.
    - ``session_id`` — the primary's; the absorbed ones are listed in
      ``merged_session_ids`` so the merge stays auditable after the others
      are deleted.
    - ``chunk_index`` / ``sub_index`` — kept only when every member agrees:
      they index a position within a turn, and a merged point spanning
      chunks or turns is no longer exactly that chunk.

    The primary is the newest member by parseable timestamp; members with no
    usable timestamp sort last, because unparseable is not a claim of
    recency. Raises ``ValueError`` on an empty group or a non-mapping
    member.
    """
    members = list(group)
    if not members:
        raise ValueError("merge_payloads: empty group")
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError(
                f"merge_payloads: group members must be payload mappings, "
                f"got {type(member).__name__}"
            )

    epochs = [_payload_epoch(member) for member in members]
    primary = 0
    for index in range(1, len(members)):
        candidate = epochs[index]
        current = epochs[primary]
        if candidate is not None and (current is None or candidate > current):
            primary = index

    merged: dict[str, Any] = dict(members[primary])

    # text: longest wins, then newest, then earliest input position.
    def text_rank(index: int) -> tuple[int, float, int]:
        epoch = epochs[index]
        return (
            len(_text_of(members[index])),
            epoch if epoch is not None else -math.inf,
            -index,
        )

    text_index = max(range(len(members)), key=text_rank)
    winner_text = _text_of(members[text_index])
    if winner_text or "text" in merged:
        merged["text"] = winner_text

    # timestamp: newest parseable epoch as a float, ISO regenerated.
    parseable = [epoch for epoch in epochs if epoch is not None]
    if parseable:
        newest = max(parseable)
        merged["timestamp"] = newest
        merged["timestamp_iso"] = datetime.fromtimestamp(newest, tz=UTC).isoformat()

    # importance: the strongest present value survives.
    importances = [
        value
        for value in (_importance(member) for member in members)
        if value is not None
    ]
    if importances:
        merged["importance"] = max(importances)

    # chunk/sub indices: unanimous or gone.
    for key in ("chunk_index", "sub_index"):
        reference = members[primary].get(key, _MISSING)
        if any(member.get(key, _MISSING) != reference for member in members):
            merged.pop(key, None)

    # session ids: the primary's stays; the absorbed ones are recorded.
    primary_session = members[primary].get("session_id")
    absorbed = sorted({
        str(member.get("session_id"))
        for member in members
        if member.get("session_id") is not None
        and str(member.get("session_id")) != str(primary_session)
    })
    if absorbed:
        merged["merged_session_ids"] = absorbed

    return merged
