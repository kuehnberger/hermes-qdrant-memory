"""Tests for the heuristic entity layer.

The module is pure stdlib -- no model, no network, no clock -- so every test
here is pure too. Three halves:

* **Extraction.** Each pattern family, the precision guards, and the
  normalisation that makes matching case- and spacing-insensitive. The guards
  are the load-bearing half: an extractor that fires on ordinary prose boosts
  the wrong candidates, and the failure is invisible in a recall that still
  returns *something*.
* **Overlap.** A containment fraction in [0, 1], case-insensitive, on word
  boundaries -- the boundary is the whole point, because "ok" is a substring of
  "token" and "hi" of "this".
* **Boost.** Additive and bounded, stable, a permutation of the input, and a
  no-op on an empty query or a non-positive weight.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from plugins.memory.qdrant import entities

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Extraction: one family at a time
# ---------------------------------------------------------------------------


class TestExtraction:

    def test_a_single_capitalised_word(self):
        assert entities.extract_entities("Qdrant is a vector database") == ["Qdrant"]

    def test_a_capitalised_run_is_one_entity(self):
        assert entities.extract_entities("Alice Smith joined the team") == [
            "Alice Smith"
        ]

    def test_a_three_word_run(self):
        assert entities.extract_entities("New York City is big") == ["New York City"]

    def test_plain_lowercase_prose_yields_nothing(self):
        """The guard that matters most: ordinary sentences are not entities."""
        assert entities.extract_entities(
            "the quick brown fox jumps over the lazy dog"
        ) == []

    def test_snake_case_identifier(self):
        assert entities.extract_entities(
            "call apply_entity_boost then extract_entities"
        ) == ["apply_entity_boost", "extract_entities"]

    def test_dotted_path(self):
        assert entities.extract_entities(
            "see entities.py and plugins.memory.qdrant"
        ) == ["entities.py", "plugins.memory.qdrant"]

    def test_hyphenated_name(self):
        """The plugin's own package name is a hyphenated identifier."""
        assert entities.extract_entities(
            "install hermes-plugin-qdrant today"
        ) == ["hermes-plugin-qdrant"]

    def test_camel_case_identifier(self):
        assert entities.extract_entities("QdrantMemoryProvider is a class") == [
            "QdrantMemoryProvider"
        ]

    def test_all_caps_acronym(self):
        assert entities.extract_entities("the API and BM25") == ["API", "BM25"]

    def test_double_quoted_term(self):
        assert entities.extract_entities('he said "exact phrase" loudly') == [
            "exact phrase"
        ]

    def test_single_quoted_term(self):
        assert entities.extract_entities("the 'exact phrase' here") == [
            "exact phrase"
        ]

    def test_backticked_code_span(self):
        assert entities.extract_entities("run `qdrant_recall` now") == [
            "qdrant_recall"
        ]

    def test_an_abbreviation_is_not_a_path(self):
        """"e.g." matches the dotted pattern; it is prose, not a file."""
        assert entities.extract_entities("use e.g. this") == []


# ---------------------------------------------------------------------------
# Extraction: the precision guards
# ---------------------------------------------------------------------------


class TestPrecisionGuards:

    def test_sentence_initial_function_word_is_not_a_name(self):
        assert entities.extract_entities("The cat sat on the mat.") == []

    def test_a_leading_function_word_is_stripped_from_a_run(self):
        """"The New York Times" is a name; the leading "The" is prose."""
        assert entities.extract_entities("The New York Times reported it.") == [
            "New York Times"
        ]

    def test_a_mid_sentence_function_word_is_kept(self):
        """"The Rock" is a name, and it is not at a sentence start."""
        assert entities.extract_entities("I saw The Rock at the cinema") == [
            "The Rock"
        ]

    def test_single_characters_are_not_entities(self):
        assert entities.extract_entities("I and A") == []

    def test_a_quoted_term_is_capped(self):
        """A quoted run past the cap is a pasted paragraph, not a term."""
        long_quote = '"' + ("word " * 40).strip() + '"'
        assert len(long_quote) > entities._MAX_QUOTED_CHARS
        assert entities.extract_entities(long_quote) == []

    def test_the_cap_is_a_limit_not_an_off_switch(self):
        """Control for the test above: a short quoted term is still extracted."""
        assert entities.extract_entities('the "ok" path') == ["ok"]

    def test_empty_and_whitespace_input(self):
        assert entities.extract_entities("") == []
        assert entities.extract_entities("   ") == []


# ---------------------------------------------------------------------------
# Extraction: order and dedup
# ---------------------------------------------------------------------------


class TestOrderAndDedup:

    def test_entities_come_back_in_text_order(self):
        assert entities.extract_entities("Zoe met Alice") == ["Zoe", "Alice"]

    def test_repeats_are_deduplicated(self):
        assert entities.extract_entities("Alice Smith saw Alice") == [
            "Alice Smith",
            "Alice",
        ]

    def test_dedup_is_case_insensitive(self):
        """"ALICE" and "Alice" are one entity; the first spelling is kept."""
        assert entities.extract_entities("Alice met ALICE") == ["Alice"]

    def test_dedup_ignores_spacing_and_casing_of_a_identifier(self):
        """camelCase and its spaced form are one entity after normalisation."""
        assert entities.extract_entities(
            "QdrantMemoryProvider and Qdrant Memory Provider"
        ) == ["QdrantMemoryProvider"]

    def test_extraction_is_deterministic(self):
        text = "Alice met Bob in New York"
        assert entities.extract_entities(text) == entities.extract_entities(text)


# ---------------------------------------------------------------------------
# entity_overlap
# ---------------------------------------------------------------------------


class TestEntityOverlap:

    def test_exact_match_is_one(self):
        assert entities.entity_overlap(["Alice"], "Alice went home") == 1.0

    def test_partial_match_is_a_fraction(self):
        assert entities.entity_overlap(["Alice", "Bob"], "Alice went home") == 0.5

    def test_no_match_is_zero(self):
        assert entities.entity_overlap(["Alice"], "nothing relevant here") == 0.0

    def test_empty_query_is_zero(self):
        assert entities.entity_overlap([], "Alice went home") == 0.0

    def test_query_case_does_not_matter(self):
        assert entities.entity_overlap(["alice"], "Alice went home") == 1.0

    def test_text_case_does_not_matter(self):
        """A memory that writes the name in lowercase still earns the boost."""
        assert entities.entity_overlap(["Alice"], "alice went home") == 1.0

    def test_duplicate_query_entities_count_once(self):
        assert entities.entity_overlap(["Alice", "Alice"], "Alice") == 1.0

    def test_a_multi_word_entity_matches_its_longer_form(self):
        assert entities.entity_overlap(["New York"], "New York City") == 1.0

    def test_a_longer_query_entity_matches_its_shorter_form(self):
        assert entities.entity_overlap(["New York City"], "New York") == 1.0

    def test_camel_case_matches_its_spaced_form(self):
        assert entities.entity_overlap(
            ["QdrantMemoryProvider"], "the Qdrant Memory Provider class"
        ) == 1.0

    @pytest.mark.parametrize(
        ("entity", "text"),
        [
            pytest.param("ok", "the token was accepted", id="ok-not-in-token"),
            pytest.param("hi", "this is history", id="hi-not-in-this"),
            pytest.param("hi", "a history of time", id="hi-not-in-history"),
            pytest.param("ai", "email arrived", id="ai-not-in-email"),
        ],
    )
    def test_substrings_are_not_word_matches(self, entity: str, text: str):
        """Word boundaries, not substrings: the trap the naive rule falls into."""
        assert entities.entity_overlap([entity], text) == 0.0

    def test_the_result_is_always_a_fraction(self):
        queries = (["Alice"], ["ok"], ["hi"], ["New York"])
        texts = ("x", "Alice went home", "this is history", "New York City")
        for query in queries:
            for text in texts:
                overlap = entities.entity_overlap(query, text)
                assert 0.0 <= overlap <= 1.0


# ---------------------------------------------------------------------------
# apply_entity_boost
# ---------------------------------------------------------------------------


class TestApplyEntityBoost:

    def test_a_matching_candidate_moves_up(self):
        boosted = entities.apply_entity_boost(
            [(0.5, "Alice went home"), (0.6, "bob")], ["Alice"]
        )
        assert boosted == [(0.65, "Alice went home"), (0.6, "bob")]

    def test_the_boost_is_weight_times_overlap(self):
        """0.1 + 0.15 * 1.0 = 0.25 for the match; the other pair is untouched."""
        boosted = entities.apply_entity_boost(
            [(0.1, "Alice went home"), (0.9, "other")], ["Alice"]
        )
        assert boosted == [(0.9, "other"), (0.25, "Alice went home")]

    def test_ties_keep_the_callers_order(self):
        """Stability is what makes the knob a no-op on an entity-poor query."""
        assert entities.apply_entity_boost(
            [(0.5, "a"), (0.5, "b")], ["zzz"]
        ) == [(0.5, "a"), (0.5, "b")]

    def test_a_non_matching_candidate_keeps_its_score(self):
        boosted = entities.apply_entity_boost([(0.9, "other")], ["Alice"])
        assert boosted == [(0.9, "other")]

    def test_the_result_is_a_permutation_of_the_input(self):
        pairs = [(0.5, "a"), (0.9, "b"), (0.1, "c")]
        boosted = entities.apply_entity_boost(pairs, ["a"])
        assert sorted(text for _score, text in boosted) == ["a", "b", "c"]
        assert len(boosted) == len(pairs)

    def test_the_input_is_not_mutated(self):
        pairs = [(0.5, "Alice"), (0.9, "Bob")]
        entities.apply_entity_boost(pairs, ["Alice"])
        assert pairs == [(0.5, "Alice"), (0.9, "Bob")]

    def test_an_empty_query_short_circuits(self):
        pairs = [(0.5, "a"), (0.9, "b")]
        assert entities.apply_entity_boost(pairs, []) == pairs

    def test_empty_pairs_short_circuits(self):
        assert entities.apply_entity_boost([], ["Alice"]) == []

    @pytest.mark.parametrize("weight", [0.0, -1.0, -0.15])
    def test_a_non_positive_weight_disables_the_boost(self, weight: float):
        """No score is raised: the pairs come back with their original scores."""
        pairs = [(0.5, "Alice went home"), (0.9, "Bob")]
        boosted = entities.apply_entity_boost(pairs, ["Alice"], weight=weight)
        assert sorted(boosted) == sorted(pairs)

    def test_the_weight_is_clamped_to_one(self):
        """A huge weight would let one mention outrank a much better match."""
        boosted = entities.apply_entity_boost(
            [(0.5, "Alice went home")], ["Alice"], weight=9.0
        )
        assert boosted == [(1.5, "Alice went home")]

    def test_boosting_is_deterministic(self):
        pairs = [(0.5, "Alice went home"), (0.6, "bob")]
        assert entities.apply_entity_boost(pairs, ["Alice"]) == (
            entities.apply_entity_boost(pairs, ["Alice"])
        )


# ---------------------------------------------------------------------------
# The interface and the no-LLM boundary
# ---------------------------------------------------------------------------


class TestInterfaceAndBoundary:

    def test_the_documented_interface_exists(self):
        for name in ("extract_entities", "entity_overlap", "apply_entity_boost"):
            assert callable(getattr(entities, name)), name

    def test_the_module_imports_only_the_stdlib(self):
        """The no-LLM rule, pinned for this module too: stdlib only, no model."""
        tree = ast.parse(
            (REPO_ROOT / "entities.py").read_text(encoding="utf-8"),
            filename="entities.py",
        )
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    roots.add(node.module.split(".")[0])
        assert roots <= {"__future__", "re", "collections"}, (
            f"entities.py must stay stdlib-only; it imports {sorted(roots)}"
        )
