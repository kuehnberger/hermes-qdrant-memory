"""Trivial-message gate (rival-hardening: memory-os parity).

A Hermes turn is usually content; a noticeable minority is not. "ok", "thanks",
"hi", an emoji, "!!!" — none of those belongs in a vector store, and none of
them should spend a query embedding. This module decides, with no model and no
network, which turns are trivial.

Two thresholds, deliberately asymmetric
----------------------------------------
``should_skip_recall`` is the loose one: it also drops *short* turns that are
not questions and carry no content word of four letters or more ("mm", "noted").
Those are small talk too, and recall is where small talk hurts — a "thanks"
query embeds a vector, retrieves unrelated points, and pollutes the prompt with
them.

``should_skip_store`` is the strict one: it drops only *genuine* non-content —
social closers, blanks, emoji/punctuation-only. A short but factual turn ("I use
Qdrant", "deploy is Friday") is exactly what verbatim storage exists for, and a
short question is exactly what the store should be able to answer later. The
store keeps more than recall because a false positive there is a memory that
never existed, while a false positive here merely wastes an embedding.

Why token matching and not substring matching
---------------------------------------------
The naive implementation — ``if "ok" in text`` — fires on "token" and "broken",
and ``"hi" in text`` fires on "this", "which" and "history". Those are ordinary
content words in an ordinary codebase's traffic, so the gate would silently
drop real turns: recall would never see them and the store would never keep
them, with no error anywhere. So this module tokenises into whole words and
matches whole tokens only, and it *segments* the token sequence into known
acknowledgements and greeting phrases rather than testing one substring.

Design rules
------------
* **Case-insensitive, punctuation-tolerant.** Matching happens on the lowered
  text with everything that is not a letter stripped out, so "OK!", " ok " and
  "ok..." are the same turn.
* **No letters at all is trivial.** Blank, whitespace, emoji-only,
  punctuation-only, symbol-only and digits-only messages have no word tokens and
  are closers by definition — none of them carries a retrievable fact.
* **Word length is measured in letters.** "qdrant" is a content word; "ok", "hi"
  and "np" are not, so the four-letter cut keeps acknowledgements out of the
  content side without maintaining a stopword list.
* **Deterministic and dependency-free.** Pure ``re`` over the stdlib — no model,
  no clock, no I/O — because the gate runs on every turn of a hot path.

Interface:

    is_social_closer(text: str) -> bool
    should_skip_recall(text: str, *, min_chars: int = 12) -> bool
    should_skip_store(text: str) -> bool
"""
from __future__ import annotations

import re

__all__ = ["is_social_closer", "should_skip_recall", "should_skip_store"]

#: A whole word: a run of Unicode letters. Digits and the underscore are not
#: word characters here on purpose — "123" and "v1_2" are not words to match
#: against, and a hyphenated or apostrophised token ("l'été", "y'all") is split
#: into its letter runs, which is the reading a token-level gate wants.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Whole-word acknowledgements, greetings and farewells — single tokens only. A
#: multi-word closer ("thank you") belongs in ``_PHRASES`` instead, because its
#: parts ("you") are ordinary content words ("who are you").
_CLOSER_WORDS = frozenset({
    "ok", "okay", "okey", "k", "kk",
    "thanks", "thx", "ty", "thanx", "thnks",
    "hi", "hello", "hey", "yo", "howdy", "sup",
    "bye", "cya", "ciao", "cheers", "later",
    "np", "sure", "yep", "yup", "yeah", "nope",
    "lol", "lmao", "haha", "rofl",
    "yes", "no", "good", "great", "nice", "perfect", "cool",
    "morning", "afternoon", "evening", "night",
})

#: Multi-word closers, as token tuples. Matched exhaustively by
#: :func:`_segmentable`, so "thank you", "good morning" and "no problem" are
#: closers while "who are you" is not.
_PHRASES = frozenset({
    ("thank", "you"),
    ("thank", "you", "very", "much"),
    ("thank", "you", "so", "much"),
    ("thanks", "a", "lot"),
    ("thanks", "a", "bunch"),
    ("thanks", "a", "million"),
    ("good", "morning"),
    ("good", "morning", "everyone"),
    ("good", "afternoon"),
    ("good", "evening"),
    ("good", "evening", "everyone"),
    ("good", "night"),
    ("see", "you"),
    ("see", "ya"),
    ("no", "problem"),
    ("no", "worries"),
    ("sure", "thing"),
    ("sounds", "good"),
})

#: Longest phrase in ``_PHRASES`` — bounds the segmentation window.
_MAX_PHRASE = max(len(phrase) for phrase in _PHRASES)

#: Shortest letter-run that counts as carrying content. Four letters keeps
#: "ok", "hi", "np", "thx" and "sure" on the trivial side while every ordinary
#: noun, verb, name and acronym is longer.
_MIN_CONTENT_WORD = 4

#: A leading interrogative word. "who are you" asks a question without a "?",
#: and it must never be read as small talk.
_QUESTION_LEADS = frozenset({
    "who", "what", "why", "how", "when", "where", "which", "whose", "whom",
    "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
    "should", "would", "will", "shall", "may", "might", "must", "have", "has",
    "had", "explain", "help", "tell", "show", "describe", "clarify", "list",
})


def _tokens(text: str) -> list[str]:
    """Lowercased whole words: letters only, punctuation and digits dropped."""
    return _WORD.findall(text.lower())


def _segmentable(tokens: list[str]) -> bool:
    """True when every token is a closer word or part of a closer phrase.

    Dynamic programming over the token sequence: ``reach[i]`` is True when
    ``tokens[:i]`` is entirely covered by single-word closers and phrase tuples.
    Exhaustive rather than greedy because the coverings overlap — "good" alone
    is a closer *and* the head of the phrase "good morning", so a first match
    must not decide the answer for the whole message.
    """
    reach = [False] * (len(tokens) + 1)
    reach[0] = True
    for index in range(1, len(tokens) + 1):
        first = max(0, index - _MAX_PHRASE)
        for start in range(index - 1, first - 1, -1):
            if not reach[start]:
                continue
            span = tokens[start:index]
            if len(span) == 1:
                if span[0] in _CLOSER_WORDS:
                    reach[index] = True
                    break
            elif tuple(span) in _PHRASES:
                reach[index] = True
                break
    return reach[-1]


def is_social_closer(text: str) -> bool:
    """True when ``text`` carries no memory-worthy content.

    Covers pure acknowledgements, greetings and farewells — case-insensitively,
    with surrounding punctuation and whitespace ignored — plus emoji-only,
    punctuation-only and symbol-only messages, plus any combination of those
    tokens ("ok thanks", "hi hi", "thanks!").

    Matching is on whole words: "ok" must not fire inside "token" or "broken",
    and "hi" must not fire inside "this", "which" or "history".
    """
    tokens = _tokens(text)
    if not tokens:
        # No letters at all: blank, whitespace, emoji-only, punctuation-only,
        # symbol-only, digits-only. Nothing retrievable, so nothing to store.
        return True
    return _segmentable(tokens)


def _is_question(text: str, tokens: list[str]) -> bool:
    """True when ``text`` reads as a question, with or without a ``?``.

    A ``?`` anywhere is the obvious signal. Without one, a leading
    interrogative ("who are you", "how do I fix this", "explain RRF") is still
    a question — the alternative is treating "who are you" as small talk and
    never remembering the answer.
    """
    if "?" in text:
        return True
    return bool(tokens) and tokens[0] in _QUESTION_LEADS


def _has_content_word(tokens: list[str]) -> bool:
    """True when any token is at least ``_MIN_CONTENT_WORD`` letters long."""
    return any(len(token) >= _MIN_CONTENT_WORD for token in tokens)


def should_skip_recall(text: str, *, min_chars: int = 12) -> bool:
    """True when a turn should not be embedded and not reach the recall prompt.

    True for a social closer, a blank/whitespace message, or a short turn that
    is neither a question nor carrying a content word of four letters or more.
    ``min_chars`` is measured on the *stripped* text, so surrounding whitespace
    neither rescues a short message nor lengthens a long one.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if is_social_closer(text):
        return True
    if len(stripped) >= min_chars:
        return False
    tokens = _tokens(stripped)
    if _is_question(stripped, tokens):
        return False
    return not _has_content_word(tokens)


def should_skip_store(text: str) -> bool:
    """True when a turn should not be written to the vector store at all.

    Stricter than :func:`should_skip_recall` on purpose: only genuine
    non-content is dropped — social closers, blanks, emoji/punctuation-only. A
    short factual statement ("I use Qdrant", "deploy is Friday") and a short
    question are both kept. The store is the durable record; a turn it drops is
    a memory that never existed, while a turn recall drops merely wastes an
    embedding.
    """
    if not text.strip():
        return True  # subsumed by is_social_closer, stated for the contract
    return is_social_closer(text)
