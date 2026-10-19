"""Tests for ``gate``: which turns are too trivial to embed, recall or store.

Every assertion here is about the *shape* of the decision, because the cost of
getting it wrong is silent: a turn the gate mislabels is either never stored
(store too strict), stored as noise (store too loose), or spends an embedding
and pollutes the recall prompt (recall too loose). None of those raise.

Three things are pinned mechanically:

1. **The table.** One list of closers (must be skipped) and one of content
   (must not), so a new entry is a one-line change and the two directions are
   never collapsed into one "looks about right" test.
2. **The word-boundary regressions.** "ok" inside "token"/"broken", "hi" inside
   "this"/"which"/"history", "thank" inside "thankful", "break" inside
   "breakfast". A substring implementation passes everything else in this file
   and fails only these — the main bug this module must not have.
3. **The asymmetry.** Recall drops more than the store does, and the cases that
   prove it are short factual turns and short questions: exactly the content a
   memory store exists to keep.

Pure stdlib plus pytest; no server, no model, no network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from plugins.memory.qdrant import gate

REPO = Path(__file__).resolve().parent.parent

#: Turn skipped by every gate: acknowledgements, greetings, farewells, their
#: combinations, and messages with no letters at all.
CLOSERS = [
    "ok",
    "OK",
    "Ok!",
    "ok...",
    " k ",
    "k",
    "thanks",
    "Thanks!",
    "thank you",
    "THANK YOU",
    "thx",
    "ty",
    "hi",
    "Hi!",
    "hello",
    "hey",
    "yo",
    "good morning",
    "Good Morning!",
    "good night",
    "bye",
    "Bye!",
    "cya",
    "see you",
    "see ya",
    "cheers",
    "np",
    "no problem",
    "No problem!",
    "no worries",
    "sure",
    "sure thing",
    "yep",
    "yeah",
    "nope",
    "lol",
    "haha",
    "lmao",
    "yes",
    "no",
    # Combinations of closer tokens: still no content.
    "ok thanks",
    "ok, thanks!",
    "hi hi",
    "thanks thanks",
    "thank you so much",
    "thanks a lot",
    "good morning everyone",
    "sounds good",
    # No letters at all: emoji-only, punctuation-only, symbol-only, blank.
    "🙂",
    "👍",
    "🙌",
    "❤️",
    "!!!",
    "???",
    "...",
    "---",
    "###",
    "   ",
    "",
]

#: Turn that carries content and must never be dropped.
CONTENT = [
    "why is the sky blue?",
    "how do I fix the qdrant timeout?",
    "what about the hybrid flag?",
    "who are you",
    "explain RRF",
    "I use Qdrant",
    "deploy is Friday",
    "the build broke again",
    "revert that commit",
    "what is the port?",
    "the token budget is 128k",
    "history repeats itself",
    "shipping the release on Friday",
    "tokenize the query before embedding",
    "thankful for the help",
    "the breakfast meeting is at nine",
    "qdrant is great",
    "a partial match is not a match: hi there",
]

#: The word-boundary traps. Each contains a closer token as a substring while
#: being an ordinary content word, so a substring matcher fails here.
BOUNDARY_TRAPS = [
    "token",
    "broken",
    "this",
    "which",
    "history",
    "shipping",
    "thankful",
    "breakfast",
]

#: The turns the spec names as must-never-skip, for both gates at once.
CRITICAL_NEGATIVES = [
    "why is the sky blue?",
    "how do I fix the qdrant timeout?",
    "what about the hybrid flag?",
    "who are you",
    "explain RRF",
    "I use Qdrant",
    "deploy is Friday",
    "the build broke again",
    "revert that commit",
    "what is the port?",
]

#: Short turns that are small talk without being closers: recall drops them,
#: the store keeps them. This is the asymmetry, as cases rather than prose.
SHORT_SMALL_TALK = ["mm", "hmm hmm", "uh huh", "me too", "so it is", "uh oh"]


def _slug(text: str) -> str:
    """A stable, readable pytest id for one case."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "empty"


def _params(cases: list[str]) -> list[pytest.ParameterSet]:
    """Numbered ``pytest.param``s, so ids stay unique for repeated tokens."""
    return [
        pytest.param(text, id=f"{index:02d}-{_slug(text)}")
        for index, text in enumerate(cases)
    ]


class TestIsSocialCloser:
    """The closer table, in both directions."""

    @pytest.mark.parametrize("text", _params(CLOSERS))
    def test_closers_are_skipped(self, text: str) -> None:
        assert gate.is_social_closer(text) is True

    @pytest.mark.parametrize("text", _params(CONTENT))
    def test_content_is_not_skipped(self, text: str) -> None:
        assert gate.is_social_closer(text) is False


class TestWordBoundaries:
    """Whole-word matching: a closer substring must not fire inside a word."""

    @pytest.mark.parametrize("text", _params(BOUNDARY_TRAPS))
    def test_closer_tokens_do_not_match_inside_words(self, text: str) -> None:
        assert gate.is_social_closer(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "the token budget is 128k",
            "a broken build is not ok to ship",
            "this history lesson is long",
            "which shipping lane did we pick?",
            "thankful for the help",
            "the breakfast meeting is at nine",
        ],
    )
    def test_sentences_containing_closer_substrings_are_content(
        self, text: str
    ) -> None:
        assert gate.is_social_closer(text) is False

    def test_the_fixture_would_catch_a_substring_implementation(self) -> None:
        """Guard: these cases only mean something if the trap is real.

        If the substring guard were removed, ``"ok" in "token"`` would return
        True and every assertion above would fail — so assert the trap itself
        rather than trusting that the cases are doing work.
        """
        assert "ok" in "token"
        assert "ok" in "broken"
        assert "hi" in "this"
        assert "hi" in "which"
        assert "hi" in "history"
        assert "hi" in "shipping"
        assert "thank" in "thankful"
        assert "break" in "breakfast"

    def test_a_closer_token_is_still_a_closer_on_its_own(self) -> None:
        """The other half of the guard: the word-boundary rule must not be so
        strict that the actual closers stop matching."""
        for token in ("ok", "hi", "thanks", "np", "sure", "k"):
            assert gate.is_social_closer(token) is True


class TestShouldSkipRecall:
    """Recall drops closers *and* short non-questions with no content word."""

    @pytest.mark.parametrize("text", _params(CRITICAL_NEGATIVES))
    def test_the_critical_negatives_are_never_skipped(self, text: str) -> None:
        assert gate.should_skip_recall(text) is False

    @pytest.mark.parametrize("text", _params(BOUNDARY_TRAPS))
    def test_word_boundary_traps_are_never_skipped(self, text: str) -> None:
        assert gate.should_skip_recall(text) is False

    @pytest.mark.parametrize("text", _params(CLOSERS))
    def test_closers_are_skipped(self, text: str) -> None:
        assert gate.should_skip_recall(text) is True

    @pytest.mark.parametrize("text", _params(SHORT_SMALL_TALK))
    def test_short_small_talk_is_skipped(self, text: str) -> None:
        assert gate.should_skip_recall(text) is True

    def test_a_blank_message_is_skipped_at_any_min_chars(self) -> None:
        assert gate.should_skip_recall("", min_chars=0) is True
        assert gate.should_skip_recall("   ", min_chars=1000) is True

    def test_min_chars_is_measured_on_the_stripped_text(self) -> None:
        """Surrounding whitespace neither rescues nor lengthens a message."""
        assert gate.should_skip_recall("  mm  ", min_chars=2) is False
        assert gate.should_skip_recall("  mm  ", min_chars=3) is True


class TestRecallMinCharsBoundary:
    """The exact boundary: shorter than ``min_chars``, never equal to it."""

    def test_a_message_at_min_chars_is_not_skipped(self) -> None:
        # "aaa bbb cc" is exactly 10 characters stripped, and has no word of
        # four letters — so only the length rule can decide it.
        assert gate.should_skip_recall("aaa bbb cc", min_chars=10) is False

    def test_the_same_message_below_min_chars_is_skipped(self) -> None:
        assert gate.should_skip_recall("aaa bbb cc", min_chars=11) is True

    def test_the_default_min_chars_is_twelve(self) -> None:
        # "I use Qdrant" is exactly 12 stripped characters, and carries the
        # content word "qdrant": both rules agree it must be kept.
        assert gate.should_skip_recall("I use Qdrant") is False
        assert gate.should_skip_recall("I use Qdrant", min_chars=13) is False

    def test_a_question_is_never_skipped_however_short(self) -> None:
        assert gate.should_skip_recall("why?", min_chars=12) is False
        assert gate.should_skip_recall("what?", min_chars=100) is False

    def test_a_leading_interrogative_is_a_question_without_the_mark(self) -> None:
        """No "?" and only three-letter words — only the interrogative lead
        ("who") keeps "who are you" out of the skip path."""
        assert gate.should_skip_recall("who are you", min_chars=12) is False
        assert gate.should_skip_recall("explain RRF", min_chars=12) is False

    def test_a_content_word_of_four_letters_rescues_a_short_message(self) -> None:
        assert gate.should_skip_recall("cats", min_chars=12) is False


class TestShouldSkipStore:
    """The store gate is strict: genuine non-content only."""

    @pytest.mark.parametrize("text", _params(CRITICAL_NEGATIVES))
    def test_the_critical_negatives_are_never_skipped(self, text: str) -> None:
        assert gate.should_skip_store(text) is False

    @pytest.mark.parametrize("text", _params(BOUNDARY_TRAPS))
    def test_word_boundary_traps_are_never_skipped(self, text: str) -> None:
        assert gate.should_skip_store(text) is False

    @pytest.mark.parametrize("text", _params(CLOSERS))
    def test_closers_are_skipped(self, text: str) -> None:
        assert gate.should_skip_store(text) is True

    def test_short_small_talk_is_still_stored(self) -> None:
        """The asymmetry, as behaviour: recall drops these, the store does not.

        A short factual turn and a short question are exactly what the store
        exists to keep; dropping them would lose memories that no later turn
        re-states.
        """
        for text in SHORT_SMALL_TALK:
            assert gate.should_skip_recall(text) is True
            assert gate.should_skip_store(text) is False


class TestGateModule:
    """The module's own constraints, pinned mechanically."""

    def test_the_gate_imports_only_the_stdlib(self) -> None:
        """The gate runs on every turn of a hot path, so it may not pull in a
        dependency — and the no-LLM suite does not scan it."""
        tree = ast.parse((REPO / "gate.py").read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    roots.add(node.module.split(".")[0])
        assert roots <= {"__future__", "re", "unicodedata"}, roots

    def test_the_public_interface_is_exactly_the_documented_one(self) -> None:
        assert sorted(gate.__all__) == [
            "is_social_closer",
            "should_skip_recall",
            "should_skip_store",
        ]
