"""Unit tests for ``filters``: metadata spec -> Qdrant filter dict, no server.

Every build test asserts the *exact* emitted dict, because that dict is the
contract handed to Qdrant: a shape that merely "looks about right" is how a
filter silently stops filtering. Pure stdlib — nothing here imports
``qdrant_client`` or touches a network, so the shapes can be pinned without a
server.

The scope tests are the security-relevant half: recall is session-scoped, and
the payload filter *is* the boundary. They assert that the scope clause is
always in ``"must"`` and that a spec contradicting it is refused, never quietly
downgraded to an unscoped search.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from plugins.memory.qdrant import filters

SCOPE = {"session_id": "abc"}


# ---------------------------------------------------------------------------
# One case per operator, asserting the exact emitted clause
# ---------------------------------------------------------------------------

EMITTED_CLAUSES = [
    pytest.param(
        {"kind": "assistant"},
        {"must": [{"key": "kind", "match": {"value": "assistant"}}]},
        id="exact-scalar",
    ),
    pytest.param(
        {"turn": 7},
        {"must": [{"key": "turn", "match": {"value": 7}}]},
        id="exact-int",
    ),
    pytest.param(
        {"archived": False},
        {"must": [{"key": "archived", "match": {"value": False}}]},
        id="exact-bool",
    ),
    pytest.param(
        {"kind": {"eq": "assistant"}},
        {"must": [{"key": "kind", "match": {"value": "assistant"}}]},
        id="op-eq",
    ),
    pytest.param(
        {"kind": {"ne": "assistant"}},
        {"must_not": [{"key": "kind", "match": {"value": "assistant"}}]},
        id="op-ne",
    ),
    pytest.param(
        {"source": {"in": ["cli", "tui"]}},
        {"must": [{"key": "source", "match": {"any": ["cli", "tui"]}}]},
        id="op-in",
    ),
    pytest.param(
        {"source": {"nin": ["cron"]}},
        {"must_not": [{"key": "source", "match": {"any": ["cron"]}}]},
        id="op-nin",
    ),
    pytest.param(
        {"ts": {"gt": 100}},
        {"must": [{"key": "ts", "range": {"gt": 100}}]},
        id="op-gt",
    ),
    pytest.param(
        {"ts": {"gte": 100}},
        {"must": [{"key": "ts", "range": {"gte": 100}}]},
        id="op-gte",
    ),
    pytest.param(
        {"ts": {"lt": 200}},
        {"must": [{"key": "ts", "range": {"lt": 200}}]},
        id="op-lt",
    ),
    pytest.param(
        {"ts": {"lte": 200}},
        {"must": [{"key": "ts", "range": {"lte": 200}}]},
        id="op-lte",
    ),
    pytest.param(
        {"ts": {"gt": 100, "lte": 200}},
        {"must": [{"key": "ts", "range": {"gt": 100, "lte": 200}}]},
        id="op-range-window-one-clause",
    ),
    pytest.param(
        {"ts": {"lte": 200, "gt": 100}},
        {"must": [{"key": "ts", "range": {"gt": 100, "lte": 200}}]},
        id="op-range-window-order-independent",
    ),
    pytest.param(
        {"text": {"contains": "qdrant"}},
        {"must": [{"key": "text", "match": {"text": "qdrant"}}]},
        id="op-contains",
    ),
    pytest.param(
        {"text": {"icontains": "Qdrant"}},
        {"must": [{"key": "text", "match": {"text": "Qdrant"}}]},
        id="op-icontains",
    ),
    pytest.param(
        {"source": "*"},
        {"must": [{"key": "source", "is_empty": False}]},
        id="wildcard-exists",
    ),
    pytest.param(
        {"kind": {"ne": "tool", "in": ["a", "b"]}},
        {
            "must": [{"key": "kind", "match": {"any": ["a", "b"]}}],
            "must_not": [{"key": "kind", "match": {"value": "tool"}}],
        },
        id="two-ops-one-key-one-clause-each",
    ),
]


@pytest.mark.parametrize(("spec", "expected"), EMITTED_CLAUSES)
def test_single_field_spec_emits_the_exact_clause(spec, expected) -> None:
    assert filters.build_qdrant_filter(spec) == expected


def test_text_operators_emit_one_and_the_same_clause() -> None:
    """``contains`` is NOT substring matching — pin the shape, not a promise.

    Qdrant's ``match.text`` is a tokenised full-text match and its tokenizer is
    already case-insensitive, so ``contains`` and ``icontains`` are the same
    clause on this backend. Both must emit ``match.text`` (never ``match.value``
    or ``match.any``, which are the equality/membership shapes).
    """
    contains = filters.build_qdrant_filter({"text": {"contains": "qdrant"}})
    icontains = filters.build_qdrant_filter({"text": {"icontains": "qdrant"}})
    expected = {"must": [{"key": "text", "match": {"text": "qdrant"}}]}
    assert contains == icontains == expected
    clause = contains["must"][0]
    assert set(clause["match"]) == {"text"}, (
        "a text operator must emit a tokenised match.text clause; emitting "
        "match.value/any would silently claim exact or membership semantics"
    )


def test_module_stays_client_free() -> None:
    """The dict shapes are the contract, so the module must not need the client."""
    source = Path(filters.__file__).read_text(encoding="utf-8")
    assert "import qdrant_client" not in source
    assert "from qdrant_client" not in source


# ---------------------------------------------------------------------------
# Nothing to filter on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "scope"),
    [
        pytest.param(None, None, id="none-none"),
        pytest.param({}, None, id="empty-none"),
        pytest.param(None, {}, id="none-empty"),
        pytest.param({}, {}, id="empty-empty"),
    ],
)
def test_nothing_to_filter_on_returns_none(spec, scope) -> None:
    assert filters.build_qdrant_filter(spec, scope) is None


# ---------------------------------------------------------------------------
# Scope is always ANDed in, and can never be escaped
# ---------------------------------------------------------------------------


def test_scope_alone_becomes_a_must_clause() -> None:
    assert filters.build_qdrant_filter(None, SCOPE) == {
        "must": [{"key": "session_id", "match": {"value": "abc"}}]
    }


def test_scope_is_emitted_first_and_anded_with_the_spec() -> None:
    assert filters.build_qdrant_filter({"kind": "assistant"}, SCOPE) == {
        "must": [
            {"key": "session_id", "match": {"value": "abc"}},
            {"key": "kind", "match": {"value": "assistant"}},
        ]
    }


def test_scope_survives_a_spec_that_only_negates() -> None:
    assert filters.build_qdrant_filter({"kind": {"ne": "tool"}}, SCOPE) == {
        "must": [{"key": "session_id", "match": {"value": "abc"}}],
        "must_not": [{"key": "kind", "match": {"value": "tool"}}],
    }


def test_scope_stays_in_must_when_the_spec_is_an_or() -> None:
    assert filters.build_qdrant_filter({"OR": [{"a": 1}, {"b": 2}]}, SCOPE) == {
        "must": [{"key": "session_id", "match": {"value": "abc"}}],
        "should": [
            {"key": "a", "match": {"value": 1}},
            {"key": "b", "match": {"value": 2}},
        ],
    }


def test_multiple_scope_keys_all_land_in_must() -> None:
    scope = {"session_id": "abc", "profile": "qd"}
    out = filters.build_qdrant_filter({"kind": "assistant"}, scope)
    assert out == {
        "must": [
            {"key": "session_id", "match": {"value": "abc"}},
            {"key": "profile", "match": {"value": "qd"}},
            {"key": "kind", "match": {"value": "assistant"}},
        ]
    }


EVERY_SPEC_FORM = [
    pytest.param({"kind": "assistant"}, id="exact"),
    pytest.param({"kind": {"eq": "assistant"}}, id="eq"),
    pytest.param({"kind": {"ne": "tool"}}, id="ne"),
    pytest.param({"source": {"in": ["cli"]}}, id="in"),
    pytest.param({"source": {"nin": ["cron"]}}, id="nin"),
    pytest.param({"ts": {"gt": 100, "lte": 200}}, id="range"),
    pytest.param({"text": {"contains": "qdrant"}}, id="contains"),
    pytest.param({"source": "*"}, id="wildcard"),
    pytest.param({"AND": [{"a": 1}, {"b": 2}]}, id="and"),
    pytest.param({"OR": [{"a": 1}, {"b": 2}]}, id="or"),
    pytest.param({"NOT": [{"a": 1}]}, id="not"),
    pytest.param({"AND": [{"a": 1}, {"OR": [{"b": 2}, {"c": 3}]}]}, id="nested"),
]


@pytest.mark.parametrize("spec", EVERY_SPEC_FORM)
def test_scope_clause_is_present_in_must_for_every_spec_form(spec) -> None:
    """No spec form — logical combinators included — may drop the scope."""
    out = filters.build_qdrant_filter(spec, SCOPE)
    assert out is not None
    assert {"key": "session_id", "match": {"value": "abc"}} in out["must"], (
        f"spec {spec!r} produced a filter without the session scope in must: {out!r}"
    )


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_and_flattens_into_the_parent_must() -> None:
    assert filters.build_qdrant_filter({"AND": [{"a": 1}, {"b": 2}]}) == {
        "must": [
            {"key": "a", "match": {"value": 1}},
            {"key": "b", "match": {"value": 2}},
        ]
    }


def test_and_of_or_of_not_nests_one_filter_per_compound_child() -> None:
    spec = {
        "AND": [
            {"kind": "assistant"},
            {
                "OR": [
                    {"source": {"in": ["cli", "tui"]}},
                    {"NOT": [{"text": {"contains": "qdrant"}}]},
                ]
            },
        ]
    }
    assert filters.build_qdrant_filter(spec) == {
        "must": [
            {"key": "kind", "match": {"value": "assistant"}},
            {
                "should": [
                    {"key": "source", "match": {"any": ["cli", "tui"]}},
                    {"must_not": [{"key": "text", "match": {"text": "qdrant"}}]},
                ]
            },
        ]
    }


def test_or_of_and_nests_the_and_child() -> None:
    spec = {"OR": [{"AND": [{"a": 1}, {"b": 2}]}, {"c": 3}]}
    assert filters.build_qdrant_filter(spec) == {
        "should": [
            {
                "must": [
                    {"key": "a", "match": {"value": 1}},
                    {"key": "b", "match": {"value": 2}},
                ]
            },
            {"key": "c", "match": {"value": 3}},
        ]
    }


def test_not_negates_every_listed_constraint() -> None:
    """``{"NOT": [a, b]}`` is "not a AND not b" — Qdrant's must_not semantics."""
    spec = {"NOT": [{"kind": "tool"}, {"source": "cli"}]}
    assert filters.build_qdrant_filter(spec) == {
        "must_not": [
            {"key": "kind", "match": {"value": "tool"}},
            {"key": "source", "match": {"value": "cli"}},
        ]
    }


def test_logical_and_leaf_clauses_coexist() -> None:
    spec = {"kind": "assistant", "AND": [{"a": 1}], "NOT": [{"b": 2}]}
    assert filters.build_qdrant_filter(spec) == {
        "must": [
            {"key": "kind", "match": {"value": "assistant"}},
            {"key": "a", "match": {"value": 1}},
        ],
        "must_not": [{"key": "b", "match": {"value": 2}}],
    }


# ---------------------------------------------------------------------------
# validate_filter_spec: accepted shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(None, id="none"),
        pytest.param({}, id="empty"),
        pytest.param({"kind": "assistant"}, id="exact"),
        pytest.param({"archived": False}, id="exact-bool"),
        pytest.param({"kind": {"ne": "tool", "in": ["a"]}}, id="two-ops"),
        pytest.param({"ts": {"gt": 100, "gte": 1, "lt": 200, "lte": 300}}, id="range"),
        pytest.param({"source": {"in": ["cli", "tui"]}}, id="in"),
        pytest.param({"source": {"nin": ["cron"]}}, id="nin"),
        pytest.param({"text": {"contains": "qdrant"}}, id="contains"),
        pytest.param({"text": {"icontains": "qdrant"}}, id="icontains"),
        pytest.param({"source": "*"}, id="wildcard"),
        pytest.param({"AND": [{"a": 1}, {"OR": [{"b": 2}]}]}, id="nested-logical"),
        pytest.param({"NOT": [{"a": 1}, {"b": 2}]}, id="not-two-children"),
    ],
)
def test_validate_accepts_every_documented_form(spec) -> None:
    assert filters.validate_filter_spec(spec) == []


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param({"session_id": "abc"}, id="restated-exact"),
        pytest.param({"session_id": {"eq": "abc"}}, id="restated-eq"),
        pytest.param({"session_id": {"in": ["abc", "other"]}}, id="restated-in"),
        pytest.param({"session_id": {"nin": ["other"]}}, id="restated-nin"),
        pytest.param({"session_id": "*"}, id="restated-wildcard"),
        pytest.param({"AND": [{"session_id": "abc"}]}, id="restated-nested"),
    ],
)
def test_validate_accepts_a_scope_key_restated_with_its_own_value(spec) -> None:
    assert filters.validate_filter_spec(spec, SCOPE) == []


def test_restating_the_scope_keeps_the_filter_scoped() -> None:
    out = filters.build_qdrant_filter({"session_id": "abc"}, SCOPE)
    assert out is not None
    assert {"key": "session_id", "match": {"value": "abc"}} in out["must"]


# ---------------------------------------------------------------------------
# validate_filter_spec: rejected shapes
# ---------------------------------------------------------------------------

REJECTED_SPECS = [
    pytest.param("kind=assistant", "must be a mapping", id="spec-is-str"),
    pytest.param(["kind"], "must be a mapping", id="spec-is-list"),
    pytest.param(42, "must be a mapping", id="spec-is-int"),
    pytest.param({"ts": {"foo": 1}}, "unknown operator", id="unknown-operator"),
    pytest.param(
        {"ts": {"between": [1, 2]}}, "unknown operator", id="between-op"
    ),
    pytest.param({"ts": {"gt": "soon"}}, "expects a number", id="gt-with-string"),
    pytest.param({"ts": {"lt": True}}, "expects a number", id="lt-with-bool"),
    pytest.param({"ts": {"gte": None}}, "expects a number", id="gte-with-none"),
    pytest.param({"source": {"in": "cli"}}, "expects a list", id="in-with-string"),
    pytest.param({"source": {"in": []}}, "non-empty list", id="in-empty-list"),
    pytest.param(
        {"source": {"nin": [["cli"]]}}, "list of scalars", id="nin-nested"
    ),
    pytest.param(
        {"kind": {"eq": ["a"]}},
        "expects a string, number or bool",
        id="eq-with-list",
    ),
    pytest.param(
        {"text": {"contains": ""}}, "non-empty string", id="contains-empty"
    ),
    pytest.param(
        {"text": {"contains": 5}}, "non-empty string", id="contains-int"
    ),
    pytest.param(
        {"AND": {"kind": "assistant"}}, "expects a list", id="and-not-a-list"
    ),
    pytest.param({"OR": []}, "non-empty list", id="or-empty-list"),
    pytest.param({"NOT": "kind"}, "expects a list", id="not-not-a-list"),
    pytest.param(
        {"AND": [{"kind": "assistant"}, "nope"]},
        "must be filter mappings",
        id="logical-child-not-a-mapping",
    ),
    pytest.param({"AND": [{}]}, "non-empty filter mappings", id="child-empty"),
    pytest.param(
        {"source": ["cli", "tui"]}, "use {'in': [...]}", id="bare-list-value"
    ),
    pytest.param({"kind": None}, "must not be None", id="none-value"),
    pytest.param(
        {"kind": {}}, "operator mapping is empty", id="empty-op-mapping"
    ),
    pytest.param({1: "x"}, "non-empty strings", id="non-string-key"),
    pytest.param(
        {"AND": [{"ts": {"foo": 1}}]}, "AND[0]", id="nested-error-is-located"
    ),
]


@pytest.mark.parametrize(("spec", "expected"), REJECTED_SPECS)
def test_validate_rejects_bad_shapes(spec, expected) -> None:
    errors = filters.validate_filter_spec(spec)
    assert errors, f"{spec!r} should have been rejected"
    assert any(expected in error for error in errors), (
        f"no error for {spec!r} mentioned {expected!r}; got {errors!r}"
    )


SCOPE_CONFLICTS = [
    pytest.param({"session_id": "other"}, "fixed by the scope", id="exact-other"),
    pytest.param(
        {"session_id": {"eq": "other"}}, "requires 'other'", id="eq-other"
    ),
    pytest.param(
        {"session_id": {"ne": "abc"}}, "excludes 'abc'", id="ne-scope-value"
    ),
    pytest.param(
        {"session_id": {"in": ["other"]}},
        "requires one of",
        id="in-without-scope-value",
    ),
    pytest.param(
        {"session_id": {"nin": ["abc"]}},
        "excludes ['abc']",
        id="nin-scope-value",
    ),
    pytest.param(
        {"session_id": {"gt": 5}}, "range/text operator", id="range-on-scope-key"
    ),
    pytest.param(
        {"session_id": {"contains": "a"}},
        "range/text operator",
        id="text-on-scope-key",
    ),
    pytest.param(
        {"OR": [{"session_id": "other"}]},
        "fixed by the scope",
        id="conflict-inside-or",
    ),
    pytest.param(
        {"AND": [{"kind": "assistant"}, {"session_id": "other"}]},
        "fixed by the scope",
        id="conflict-inside-and",
    ),
]


@pytest.mark.parametrize(("spec", "expected"), SCOPE_CONFLICTS)
def test_validate_reports_a_scope_conflict(spec, expected) -> None:
    errors = filters.validate_filter_spec(spec, SCOPE)
    assert errors, f"{spec!r} contradicts the scope and must be reported"
    assert any(expected in error for error in errors), (
        f"no error for {spec!r} mentioned {expected!r}; got {errors!r}"
    )


@pytest.mark.parametrize(("spec", "expected"), SCOPE_CONFLICTS)
def test_build_refuses_a_scope_conflict(spec, expected) -> None:
    """Fail closed: never return a filter the scope would silently empty out."""
    with pytest.raises(ValueError) as excinfo:
        filters.build_qdrant_filter(spec, SCOPE)
    assert "session_id" in str(excinfo.value)


@pytest.mark.parametrize(("spec", "expected"), REJECTED_SPECS)
def test_build_refuses_an_invalid_spec(spec, expected) -> None:
    with pytest.raises(ValueError):
        filters.build_qdrant_filter(spec)


def test_validate_reports_every_problem_not_just_the_first() -> None:
    errors = filters.validate_filter_spec({"ts": {"foo": 1}, "source": {"in": "cli"}})
    assert len(errors) == 2, errors
    assert any("'ts'" in error for error in errors)
    assert any("'source'" in error for error in errors)
