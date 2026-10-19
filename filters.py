"""Metadata filter operators for search/recall (rival-hardening: Mem0 parity).

Turns a caller-supplied metadata filter spec into a plain Qdrant filter dict,
ANDed with the scope filter recall already applies. No ``qdrant_client`` import:
the module returns plain dicts so it stays testable without the client, and so
the dict shapes — the actual contract handed to the store — are asserted
directly instead of through a mocked client.

Why the scope is not optional
-----------------------------
Recall here is session-scoped, and the payload filter *is* the boundary that
keeps one session's memories out of another's context. A caller-supplied filter
is therefore only safe if it can narrow that boundary, never widen it. So
``build_qdrant_filter`` ANDs ``scope`` into ``"must"`` unconditionally, and
``validate_filter_spec`` reports a spec that restates a scope key with anything
other than the scope's own value: a caller writing ``{"session_id": "other"}``
gets an error rather than an empty result set that reads as "no memories yet".

Emitted shapes (Qdrant's own vocabulary, one clause per constraint)
------------------------------------------------------------------
    exact    {"kind": "assistant"}      -> {"key": "kind", "match": {"value": ...}}
    eq       {"kind": {"eq": "x"}}      -> same clause as exact
    ne       {"kind": {"ne": "x"}}      -> that clause in "must_not"
    in       {"source": {"in": [..]}}   -> {"key": "source", "match": {"any": [..]}}
    nin      {"source": {"nin": [..]}}  -> that clause in "must_not"
    range    {"ts": {"gt": 100}}        -> {"key": "ts", "range": {"gt": 100}}
    text     {"text": {"contains": "x"}} -> {"key": "text", "match": {"text": "x"}}
    wildcard {"source": "*"}            -> {"key": "source", "is_empty": False}
    logical  {"AND"|"OR"|"NOT": [..]}   -> nested "must"/"should"/"must_not"

Every range operator given for one key collapses into a single ``range`` clause
(``{"gt": 100, "lte": 200}`` is one clause, not two). ``AND`` children are
flattened into the parent's ``"must"``; a child that is itself compound nests as
a filter dict, because Qdrant accepts a Filter wherever it accepts a condition.

``contains`` and ``icontains`` are THE SAME CLAUSE on this backend
-----------------------------------------------------------------
Qdrant's ``match.text`` is a *tokenised* full-text match, not a substring match:
it matches whole tokens produced by Qdrant's own tokenizer, and that tokenizer
is already case-insensitive. So ``{"contains": "qdrant"}`` and
``{"icontains": "Qdrant"}`` both emit ``{"key": ..., "match": {"text": ...}}``,
both are case-insensitive, and neither can find a fragment inside a token.
``contains`` here does NOT mean "substring" and must never be documented as
such. Both spellings are accepted only because rival filter DSLs spell them
differently, and silently rejecting one would look like a bug to a caller.

``NOT`` takes a list of constraints; each is negated and all negations must hold
at once (Qdrant's ``must_not`` semantics), so ``{"NOT": [a, b]}`` means "not a
AND not b". An empty logical list is a caller bug — it would match everything or
nothing depending on the backend's reading — so validation rejects it rather
than guessing an intent.

Interface:

    build_qdrant_filter(spec: dict | None, scope: dict | None = None) -> dict | None
    validate_filter_spec(spec: object, scope: dict | None = None) -> list[str]
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["build_qdrant_filter", "validate_filter_spec"]

#: Logical combinators. Their values are non-empty lists of nested filter specs.
LOGICAL_OPERATORS: tuple[str, ...] = ("AND", "OR", "NOT")

#: Equality / membership operators (``in``/``nin`` carry a list).
MATCH_OPERATORS: tuple[str, ...] = ("eq", "ne", "in", "nin")

#: Numeric comparison operators; several may share one ``range`` clause.
RANGE_OPERATORS: tuple[str, ...] = ("gt", "gte", "lt", "lte")

#: Full-text operators. Same clause on Qdrant — see the module docstring.
TEXT_OPERATORS: tuple[str, ...] = ("contains", "icontains")

#: Every supported field operator, in canonical emission order.
OPERATORS: tuple[str, ...] = MATCH_OPERATORS + RANGE_OPERATORS + TEXT_OPERATORS

#: ``{"key": "*"}`` means "this key exists" -> an ``is_empty: false`` clause.
WILDCARD = "*"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _at(path: str) -> str:
    """Error-message prefix naming where in the spec the problem sits."""
    return f"{path}: " if path else ""


def _join(path: str, part: str) -> str:
    """Extend an error path, e.g. ``"" + "AND[0]"`` or ``"AND[0]" + "OR[1]"``."""
    return f"{path}.{part}" if path else part


def _is_scalar(value: object) -> bool:
    """True for the values an exact ``match.value`` can carry."""
    return isinstance(value, (str, int, float, bool))


def _is_number(value: object) -> bool:
    """True for real numbers. ``bool`` is excluded: ``True > 3`` is never meant."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_name(value: object) -> str:
    return type(value).__name__


def _scope_mapping(scope: object) -> Mapping[str, Any]:
    """Coerce ``scope`` to a mapping. Anything else is treated as "no scope"."""
    return scope if isinstance(scope, Mapping) else {}


def _compact(
    must: list[Any], should: list[Any], must_not: list[Any]
) -> dict[str, list[Any]]:
    """Build the filter dict, omitting empty clause lists.

    Omitted rather than empty on purpose: Qdrant reads ``{"must": []}`` as an
    empty condition set, which some versions treat as "match nothing". The
    absent key is unambiguous.
    """
    out: dict[str, list[Any]] = {}
    for name, clauses in (("must", must), ("should", should), ("must_not", must_not)):
        if clauses:
            out[name] = clauses
    return out


def _match(key: str, kind: str, value: Any) -> dict[str, Any]:
    """One ``match`` clause: ``value`` for equality, ``any`` for membership."""
    return {"key": key, "match": {kind: value}}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_operand(key: str, op: str, operand: object, at: str) -> list[str]:
    """Type-check one operator's operand, e.g. reject ``{"gt": "soon"}``."""
    if op in ("eq", "ne"):
        if not _is_scalar(operand) or operand is None:
            return [
                f"{at}key {key!r}: operator {op!r} expects a string, number or "
                f"bool, got {_type_name(operand)}"
            ]
        return []
    if op in ("in", "nin"):
        if not isinstance(operand, list):
            return [
                f"{at}key {key!r}: operator {op!r} expects a list, got "
                f"{_type_name(operand)}"
            ]
        if not operand:
            return [
                f"{at}key {key!r}: operator {op!r} expects a non-empty list; an "
                f"empty one can never be what the caller meant"
            ]
        bad = [v for v in operand if not _is_scalar(v) or v is None]
        if bad:
            return [
                f"{at}key {key!r}: operator {op!r} expects a list of scalars, "
                f"got {_type_name(bad[0])} at index {operand.index(bad[0])}"
            ]
        return []
    if op in RANGE_OPERATORS:
        if not _is_number(operand):
            return [
                f"{at}key {key!r}: operator {op!r} expects a number, got "
                f"{_type_name(operand)}"
            ]
        return []
    # TEXT_OPERATORS
    if not isinstance(operand, str) or not operand:
        return [
            f"{at}key {key!r}: operator {op!r} expects a non-empty string, got "
            f"{_type_name(operand)}"
        ]
    return []


def _scope_conflict(value: object, scope_value: Any) -> str | None:
    """Return why ``value`` contradicts a scope key's fixed value, else None.

    A constraint is a conflict when the scope's own value cannot satisfy it —
    that is exactly the case where the caller's filter silently returns nothing
    (or, for ``ne``/``nin``, silently returns everything outside the scope).
    Range and text operators on a scope key are rejected outright: the scope
    supplies an exact value, so "narrower than the scope" is never what a caller
    wants from a key it does not own.
    """
    if isinstance(value, Mapping):
        if any(op in value for op in RANGE_OPERATORS + TEXT_OPERATORS):
            return "the spec applies a range/text operator to it"
        if "eq" in value and value["eq"] != scope_value:
            return f"the spec requires {value['eq']!r}"
        if "ne" in value and value["ne"] == scope_value:
            return f"the spec excludes {value['ne']!r}"
        if "in" in value and isinstance(value["in"], list):
            if scope_value not in value["in"]:
                return f"the spec requires one of {value['in']!r}"
        if "nin" in value and isinstance(value["nin"], list):
            if scope_value in value["nin"]:
                return f"the spec excludes {value['nin']!r}"
        return None
    if value == WILDCARD:
        return None  # "key exists" is always true of a scope key
    if _is_scalar(value) and value != scope_value:
        return f"the spec requires {value!r}"
    return None


def _validate_field(
    key: str, value: object, scope: Mapping[str, Any], path: str
) -> list[str]:
    """Validate one ``key: value`` entry, then check it against the scope."""
    at = _at(path)
    errors: list[str] = []
    if isinstance(value, Mapping):
        if not value:
            errors.append(
                f"{at}key {key!r}: operator mapping is empty; use an exact value, "
                f"{WILDCARD!r}, or one of {', '.join(OPERATORS)}"
            )
        else:
            for op, operand in value.items():
                if op not in OPERATORS:
                    errors.append(
                        f"{at}key {key!r}: unknown operator {op!r}; supported "
                        f"operators: {', '.join(OPERATORS)}"
                    )
                    continue
                errors.extend(_validate_operand(key, op, operand, at))
    elif value == WILDCARD:
        pass
    elif value is None:
        errors.append(f"{at}key {key!r}: value must not be None")
    elif _is_scalar(value):
        pass
    else:
        errors.append(
            f"{at}key {key!r}: value must be a scalar, {WILDCARD!r} or an "
            f"operator mapping, got {_type_name(value)}; use "
            f"{{'in': [...]}} for a set of allowed values"
        )

    if key in scope:
        reason = _scope_conflict(value, scope[key])
        if reason is not None:
            errors.append(
                f"{at}key {key!r} is fixed by the scope to {scope[key]!r}, but "
                f"{reason} — a caller spec can never widen or escape the scope"
            )
    return errors


def _validate_logical(
    op: str, value: object, scope: Mapping[str, Any], path: str
) -> list[str]:
    """Validate an ``AND``/``OR``/``NOT`` list and recurse into its children."""
    at = _at(path)
    if not isinstance(value, list):
        return [
            f"{at}{op} expects a list of filter specs, got {_type_name(value)}"
        ]
    if not value:
        return [
            f"{at}{op} expects a non-empty list of filter specs; an empty one "
            f"has no meaning (match all, or match nothing?)"
        ]
    errors: list[str] = []
    for index, child in enumerate(value):
        child_path = _join(path, f"{op}[{index}]")
        if not isinstance(child, Mapping):
            errors.append(
                f"{child_path}: {op} entries must be filter mappings, got "
                f"{_type_name(child)}"
            )
            continue
        if not child:
            errors.append(
                f"{child_path}: {op} entries must be non-empty filter mappings"
            )
            continue
        errors.extend(_validate_mapping(child, scope, child_path))
    return errors


def _validate_mapping(
    spec: Mapping[Any, Any], scope: Mapping[str, Any], path: str
) -> list[str]:
    """Walk a filter mapping, collecting every problem (not just the first)."""
    errors: list[str] = []
    for key, value in spec.items():
        if not isinstance(key, str) or not key:
            errors.append(
                f"{_at(path)}filter keys must be non-empty strings, got {key!r}"
            )
            continue
        if key in LOGICAL_OPERATORS:
            errors.extend(_validate_logical(key, value, scope, path))
        else:
            errors.extend(_validate_field(key, value, scope, path))
    return errors


def validate_filter_spec(spec: object, scope: dict | None = None) -> list[str]:
    """Return human-readable errors for ``spec``; ``[]`` means it is usable.

    ``None`` and ``{}`` are valid — "no filter" is a legitimate request. Every
    other non-mapping is rejected. When ``scope`` is given, a spec that restates
    one of its keys with any value the scope cannot satisfy is an error, because
    the scope is ANDed in regardless and the caller would otherwise get a
    silently empty result set.

    All problems are reported at once, each naming its path (``AND[0].OR[1]``),
    so a caller can fix a spec in one pass instead of one error per attempt.
    """
    if spec is None:
        return []
    if not isinstance(spec, Mapping):
        return [
            f"filter spec must be a mapping, got {_type_name(spec)}"
        ]
    return _validate_mapping(spec, _scope_mapping(scope), "")


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _field_clauses(key: str, value: object) -> tuple[list[Any], list[Any]]:
    """Translate one field entry into ``(must, must_not)`` clauses.

    Each operator becomes its own clause (so ``{"eq": 1, "ne": 2}`` is an AND of
    a match and a negation, which is what it reads as), except the range
    operators, which collapse into a single ``range`` clause because Qdrant
    expects one bound object per key.
    """
    if isinstance(value, Mapping):
        positive: list[Any] = []
        negative: list[Any] = []
        bounds: dict[str, Any] = {}
        for op in OPERATORS:  # canonical order -> deterministic output
            if op not in value:
                continue
            operand = value[op]
            if op in RANGE_OPERATORS:
                bounds[op] = operand
            elif op == "eq":
                positive.append(_match(key, "value", operand))
            elif op == "ne":
                negative.append(_match(key, "value", operand))
            elif op == "in":
                positive.append(_match(key, "any", list(operand)))
            elif op == "nin":
                negative.append(_match(key, "any", list(operand)))
            else:  # contains / icontains -> the same tokenised text match
                positive.append({"key": key, "match": {"text": operand}})
        if bounds:
            positive.append({"key": key, "range": bounds})
        return positive, negative
    if value == WILDCARD:
        return [{"key": key, "is_empty": False}], []
    return [_match(key, "value", value)], []


def _as_condition(child: Mapping[str, Any]) -> Any:
    """A logical child as a Qdrant condition, unwrapping the trivial case.

    A child that translates to exactly one ``must`` clause and nothing else
    becomes that bare clause, so ``{"OR": [{"a": 1}, {"b": 2}]}`` emits the
    readable ``{"should": [<a>, <b>]}`` instead of a list of one-clause filters.
    Anything compound stays a nested filter dict — Qdrant accepts a Filter
    anywhere it accepts a condition, and flattening a negation would be wrong.
    """
    nested = _translate(child)
    must = nested.get("must", [])
    if len(must) == 1 and len(nested) == 1:
        return must[0]
    return nested


def _translate(spec: Mapping[Any, Any]) -> dict[str, list[Any]]:
    """Translate a validated filter spec into a Qdrant filter dict."""
    must: list[Any] = []
    should: list[Any] = []
    must_not: list[Any] = []
    for key, value in spec.items():
        if key in LOGICAL_OPERATORS:
            target = {"AND": must, "OR": should, "NOT": must_not}[key]
            target.extend(_as_condition(child) for child in value)
            continue
        positive, negative = _field_clauses(key, value)
        must.extend(positive)
        must_not.extend(negative)
    return _compact(must, should, must_not)


def build_qdrant_filter(
    spec: dict | None, scope: dict | None = None
) -> dict | None:
    """Build a Qdrant payload filter from ``spec``, ANDed with ``scope``.

    ``scope`` clauses always land in ``"must"`` — a caller-supplied spec can
    neither widen nor escape the session boundary. They are emitted first so the
    boundary is the first thing a reader (or a log) sees.

    Returns ``None`` when there is nothing to filter on (no spec and no scope),
    which lets the caller skip passing a filter to Qdrant entirely.

    Raises ``ValueError`` for an invalid spec rather than emitting a filter that
    quietly means something else — a malformed filter is a bug, and an unfiltered
    search is the one outcome a scope-bearing caller must never get by accident.
    """
    errors = validate_filter_spec(spec, scope)
    if errors:
        raise ValueError("invalid metadata filter spec: " + "; ".join(errors))

    scope_clauses = [
        _match(str(key), "value", value)
        for key, value in _scope_mapping(scope).items()
    ]
    translated = _translate(spec) if isinstance(spec, Mapping) else {}
    merged = _compact(
        scope_clauses + list(translated.get("must", [])),
        list(translated.get("should", [])),
        list(translated.get("must_not", [])),
    )
    return merged or None
