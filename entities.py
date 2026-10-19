"""Heuristic entity extraction and entity-boosted recall (rival-hardening: Hindsight/Mem0 parity).

WHY THIS MODULE EXISTS
----------------------
A recall query and a stored memory often name the same *thing* in different
words: the query says "the Qdrant recall tool" and the memory says
``qdrant_recall``. Dense similarity is good at topical overlap and weak at exact
identifier overlap, so the candidate that names the query's subject can sit
below one that merely shares vocabulary. Hindsight and Mem0 both close that gap
with an entity layer; this module is the same idea with no model behind it.

The knob is a RE-RANK, never a filter: an entity-poor candidate keeps its fused
score and simply moves down. Nothing is dropped, so a bad extraction costs
ranking quality, never a whole recall.

WHY NO MODEL
------------
Entity extraction is the one place a memory plugin is tempted to call an LLM,
and this plugin's USP is that it does not. A closed set of surface patterns --
capitalised runs, code identifiers, quoted terms -- is deterministic, free, and
fast enough to run per candidate. It is also auditable: a reviewer can read the
patterns and know exactly what will be extracted, which is not true of a prompt.

WHAT IS RECOGNISED
------------------
    capitalised  "Alice Smith", "New York", "Qdrant" -- a run of capitalised
                 words, or a single one
    identifiers  snake_case (``apply_entity_boost``), dotted paths
                 (``entities.py``, ``plugins.memory.qdrant``), hyphenated names
                 (``hermes-plugin-qdrant``), camelCase/PascalCase
                 (``QdrantMemoryProvider``), all-caps acronyms (``API``, ``BM25``)
    quoted       "exact phrase", 'another one', `` `qdrant_recall` ``

Precision guards, each of which exists because the naive rule produces noise: a
single-character "entity" is not one; a capitalised function word that opens a
sentence ("The cat sat") is not a name, and a leading one is stripped from a
longer run ("The New York Times" -> "New York Times"); and "e.g." is an
abbreviation, not a path.

MATCHING
--------
Entities are compared in normalised form: camelCase is split on its case
boundaries and everything is lowercased, so ``QdrantMemoryProvider``,
``Qdrant Memory Provider`` and ``qdrant memory provider`` are one entity. A
query entity counts as mentioned when it equals a text entity or one contains
the other on a word boundary -- "New York" is mentioned by a memory about
"New York City", and the reverse. Word boundaries are what keep "ai" from
matching "email".

Design rules
------------
* **Deterministic and pure.** No clock, no randomness, no global state: the
  same text always yields the same list, in order of first appearance.
* **Conservative.** A miss only forgoes a boost; a false match can reorder a
  recall. Every guard above is therefore biased toward not extracting.
* **Entity-to-entity matching.** A candidate that mentions a name only in a
  form the extractor does not recognise -- a name written in lowercase prose --
  is not matched. The alternative, substring matching against raw tokens, would
  let a common word ("memory") boost every candidate that contains it; a miss
  only forgoes a boost, and dense similarity already covers the lowercase case.
* **Additive, bounded boost.** ``apply_entity_boost`` adds ``weight * overlap``
  to the fused score and sorts by the result. The weight is clamped to [0, 1]
  and the sort is stable, so a single entity mention cannot outrank a much
  better match and equal boosted scores keep the caller's order.
* **Order-only at the call site.** The provider maps the returned order back
  onto its own tuples, so the number a user sees is still the fused score.

Interface:

    extract_entities(text: str) -> list[str]
    entity_overlap(query_entities, text) -> float
    apply_entity_boost(pairs, query_entities, *, weight=0.15)
        -> list[tuple[float, str]]
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = (
    "DEFAULT_WEIGHT",
    "apply_entity_boost",
    "entity_overlap",
    "extract_entities",
)

#: Default ranking weight: one full entity overlap is worth 0.15 of a fused
#: score -- enough to reorder near-ties, far too little to outrank a clearly
#: better match.
DEFAULT_WEIGHT = 0.15

#: Shortest normalised entity kept. Single characters ("I", "A", "R") are
#: sentence furniture, not names.
_MIN_ENTITY_CHARS = 2

#: Longest quoted term kept. A longer quoted run is a pasted paragraph, not a
#: term, and would swamp the boost.
_MAX_QUOTED_CHARS = 120

# The three families, as patterns. Order inside the alternation is the priority
# order at a given position: an identifier must win over the capitalised run it
# starts ("QdrantMemoryProvider" is one identifier, not three capitalised
# words), and the quoted family comes last because a quote is not a word
# character and so can never win earlier.
_CAP_WORD = r"[A-Z][A-Za-z0-9]*"
_SNAKE_ID = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
_HYPHEN_ID = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+"
_DOTTED_ID = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
#: A letter run containing at least one internal capital: ``entityOverlap``,
#: ``QdrantMemoryProvider``, ``HTTPServer``. The first part may itself contain
#: capitals (``HTTPServer`` -> one identifier, not the acronym ``HTTPS``), and
#: the trailing part must end in lowercase so a plain acronym is left to
#: :data:`_ACRONYM`.
_CAMEL_ID = r"[A-Za-z][A-Za-z0-9]*(?:[A-Z][a-z0-9]+)+"
_ACRONYM = r"[A-Z][A-Z0-9]+"
_QUOTED = rf"[\"'`](?:[^\"'`\n]{{1,{_MAX_QUOTED_CHARS}}})[\"'`]"

_ENTITY_RE = re.compile(
    "|".join(
        (
            rf"(?P<dotted>{_DOTTED_ID})",
            rf"(?P<snake>{_SNAKE_ID})",
            rf"(?P<hyphen>{_HYPHEN_ID})",
            rf"(?P<camel>{_CAMEL_ID})",
            rf"(?P<acronym>{_ACRONYM})",
            rf"(?P<cap>{_CAP_WORD}(?:[ \t]+{_CAP_WORD})*)",
            rf"(?P<quoted>{_QUOTED})",
        )
    )
)

#: Sentence starts: the beginning of the text (leading whitespace included) and
#: any position right after sentence-ending punctuation. Used only to strip a
#: leading function word, so an approximate boundary costs nothing.
_SENTENCE_START_RE = re.compile(r"(?:\A\s*|[.!?]\s+)")

#: Function words that can open a sentence. A capitalised one at a sentence
#: start is prose, not a name -- "The cat sat" -- but the same word mid-run is
#: part of a name, so only a *leading* one is stripped ("The New York Times").
#: Deliberately short: every word here is a word that legitimately starts
#: English sentences, and a false strip only forgoes a boost.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "as", "at", "be", "been", "but", "by", "can", "could",
        "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me",
        "might", "more", "most", "must", "my", "no", "not", "now", "of", "on",
        "one", "only", "or", "our", "out", "over", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "to", "too", "up", "us",
        "very", "was", "we", "were", "what", "when", "where", "which", "who",
        "why", "will", "with", "would", "you", "your",
    }
)

#: A lowercase letter or digit followed by an uppercase one: the camelCase
#: boundary in ``entityOverlap`` and ``QdrantMemory``.
_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: An uppercase run followed by a capitalised word: the boundary in
#: ``HTTPServer`` -> ``HTTP Server``.
_CAMEL_ACRONYM_WORD = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _normalize(entity: str) -> str:
    """Canonical comparison form: camelCase split, whitespace collapsed, lowercased.

    Splitting on case boundaries is what lets a query written as
    ``QdrantMemoryProvider`` match a memory that spells it ``Qdrant Memory
    Provider``; the split is applied to every family, and for the others it is
    a no-op because they contain no case boundary.
    """
    spaced = _CAMEL_LOWER_UPPER.sub(" ", entity)
    spaced = _CAMEL_ACRONYM_WORD.sub(" ", spaced)
    return " ".join(spaced.split()).lower()


def _plausible(surface: str) -> bool:
    """False for anything too short, digit-only, or an abbreviation like ``e.g.``

    ``e.g.`` and ``i.e.`` match the dotted-path pattern and are the commonest
    false positive in prose, so a dotted candidate whose every part is a single
    character is rejected: a real path (``entities.py``) always has a longer
    part.
    """
    key = _normalize(surface)
    if len(key) < _MIN_ENTITY_CHARS:
        return False
    if not any(char.isalpha() for char in key):
        return False
    if "." in surface and all(
        len(part) < _MIN_ENTITY_CHARS for part in surface.split(".")
    ):
        return False
    return True


def _strip_leading_stopwords(surface: str) -> str:
    """Drop function words from the front of a sentence-initial capitalised run."""
    words = surface.split()
    while words and words[0].lower() in _STOPWORDS:
        words.pop(0)
    return " ".join(words)


def _surface(match: re.Match[str]) -> str:
    """The matched text, or the inside of the quotes for the quoted family."""
    if match.lastgroup == "quoted":
        return match.group(0)[1:-1]
    return match.group(0)


def extract_entities(text: str) -> list[str]:
    """Entity mentions in ``text``, in order of first appearance.

    Returns the *surface* forms as written (``"Alice Smith"``,
    ``"QdrantMemoryProvider"``), deduplicated by normalised form so the first
    spelling wins. Callers compare through :func:`entity_overlap`, which
    normalises both sides; returning the surface form keeps the list readable
    and debuggable, which matters because the extraction is a heuristic and the
    first question about it is always "what did it see?".

    Pure: no clock, no randomness, no global state. Empty or whitespace-only
    input yields ``[]``.
    """
    if not text:
        return []
    sentence_starts = {m.end() for m in _SENTENCE_START_RE.finditer(text)}
    found: dict[str, str] = {}
    for match in _ENTITY_RE.finditer(text):
        surface = _surface(match)
        if match.lastgroup == "cap" and match.start() in sentence_starts:
            surface = _strip_leading_stopwords(surface)
            if not surface:
                continue
        if not _plausible(surface):
            continue
        key = _normalize(surface)
        if key not in found:
            found[key] = surface
    return list(found.values())


def _contains(haystack: str, needle: str) -> bool:
    """True when ``needle`` appears in ``haystack`` on whole-word boundaries."""
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def _mentioned(entity: str, present: set[str]) -> bool:
    """True when ``entity`` is named by any of the ``present`` entities.

    Equality, or containment in either direction: a query about "New York" is
    answered by a memory about "New York City", and a query about the longer
    name is answered by the shorter one. Word boundaries keep "ai" from
    matching "email" and "art" from matching "particle".
    """
    for other in present:
        if entity == other or _contains(other, entity) or _contains(entity, other):
            return True
    return False


def entity_overlap(query_entities: Sequence[str], text: str) -> float:
    """Fraction of ``query_entities`` that ``text`` mentions, in ``[0, 1]``.

    A containment measure, not a Jaccard index: the question is "how much of
    what the user asked about does this memory name?", and a long memory
    legitimately names many things the query does not. An empty query is 0.0
    (nothing to match, so no boost), and so is a query whose entities are all
    duplicates of one another after normalisation -- the denominator is the
    number of *distinct* entities.
    """
    if not query_entities:
        return 0.0
    wanted = {_normalize(entity) for entity in query_entities}
    wanted.discard("")
    if not wanted:
        return 0.0
    present = {_normalize(entity) for entity in extract_entities(text)}
    if not present:
        return 0.0
    matched = sum(1 for entity in wanted if _mentioned(entity, present))
    return matched / len(wanted)


def _clamp_weight(weight: float) -> float:
    """Keep the boost a tie-breaker: ``[0, 1]``, so it can never dominate a score."""
    return max(0.0, min(1.0, float(weight)))


def apply_entity_boost(
    pairs: Sequence[tuple[float, str]],
    query_entities: Sequence[str],
    *,
    weight: float = DEFAULT_WEIGHT,
) -> list[tuple[float, str]]:
    """Re-rank ``(score, text)`` pairs by entity overlap with the query.

    Each pair's score becomes ``score + weight * overlap`` and the list is
    sorted by that value, descending. The result is a permutation of the input
    -- one entry per pair, no drops, no duplicates -- because this is a
    re-rank: a candidate that shares no entity with the query keeps its score
    and its relative order, and simply cannot climb.

    The sort is stable, so pairs whose boosted scores tie keep the caller's
    order; that is what makes the knob a no-op on an entity-poor query rather
    than a reshuffle. ``weight`` is clamped to ``[0, 1]`` (a negative value
    would invert the boost, a large one would let a single mention outrank a
    much better match), and ``weight=0`` returns the input order unchanged.

    Callers that display a score must keep their own fused value: the provider
    maps this order back onto its tuples precisely so the number a user sees is
    still the fused score, not the boosted one.
    """
    if not pairs or not query_entities:
        return list(pairs)
    boost = _clamp_weight(weight)
    ranked = [
        (score + boost * entity_overlap(query_entities, text), score, text)
        for score, text in pairs
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(score, text) for _boosted, score, text in ranked]
