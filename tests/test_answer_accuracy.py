#!/usr/bin/env python3
"""Pure, offline tests for the answer-accuracy eval arm (scripts/answer_accuracy.py).

No Qdrant, no network, no LLM — the judge is driven by an injected stub `chat_fn`
and the fetcher by a stub `fetcher`, exactly as the design requires. Runnable
standalone as `python3 tests/test_answer_accuracy.py` (stdlib only) for a quick
green, as well as via pytest.

Counts towards scripts/check_docs_honesty.py §1b (test_funcs across tests/*.py).
"""
import re
import sys
from pathlib import Path

# Make scripts/answer_accuracy.py importable without hermes config or a venv.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import answer_accuracy as aa  # noqa: E402


def _extract(text, label):
    """Pull the rest of the line after 'label:' from a prompt body."""
    m = re.search(rf"{re.escape(label)}:\s*(.+)", text)
    return m.group(1).strip() if m else ""


_JUDGE_TAG = "strict binary judge"


def _is_judge(messages):
    return messages[0]["role"] == "system" and _JUDGE_TAG in messages[0]["content"]


def _stub_fetcher(ids):
    """id -> stored text so the generator sees a deterministic context."""
    texts = {"p_answer": "the answer is Paris", "p_noise": "unrelated noise"}
    return [texts.get(i, "") for i in ids]


def _stub_chat_yes(messages, model, max_tokens):
    """Content-aware: judge agrees iff exact_match holds; generator echoes gold
    only when the context actually contains it."""
    if _is_judge(messages):
        gold = _extract(messages[1]["content"], "Gold answer")
        ans = _extract(messages[1]["content"], "Model answer")
        return "YES" if aa.exact_match_answer(gold, ans) else "NO"
    # generator
    if "Paris" in messages[1]["content"]:
        return "the answer is Paris"
    return "I don't know"


def _stub_chat_no(messages, model, max_tokens):
    """Judge always NO; generator always 'I don't know' -> answer_accuracy 0.0."""
    if _is_judge(messages):
        return "NO"
    return "I don't know"


FIXTURE_EXAMPLES = [
    {"question": "capital of france?", "answer": "Paris"},
    {"question": "capital of germany?", "answer": "Berlin"},
]


# --- normalize_answer / exact_match ----------------------------------------

def test_normalize_answer_collapses_whitespace_and_punctuation():
    assert (
        aa.normalize_answer("The Eiffel Tower, in Paris!")
        == "the eiffel tower in paris"
    )
    assert aa.normalize_answer("  'quoted'  ") == "quoted"


def test_exact_match_is_symmetric_substring():
    assert aa.exact_match_answer("Paris", "It is in Paris.") is True
    assert aa.exact_match_answer("It is in Paris.", "Paris") is True  # gold longer
    assert aa.exact_match_answer("Berlin", "Paris") is False
    assert aa.exact_match_answer("", "anything") is False


# --- parse_judge_response --------------------------------------------------

def test_parse_judge_yes():
    assert aa.parse_judge_response("The answer is correct. YES") is True


def test_parse_judge_no():
    assert aa.parse_judge_response("No, that is wrong. NO") is False


def test_parse_judge_lowercase_matches_case_insensitively():
    assert aa.parse_judge_response("yes") is True
    assert aa.parse_judge_response("no") is False


def test_parse_judge_truly_ambiguous_returns_none():
    # No standalone YES/NO word as a token -> ambiguous / non-compliant.
    assert aa.parse_judge_response("I am not sure about that") is None


def test_parse_judge_empty_returns_none():
    assert aa.parse_judge_response("") is None
    assert aa.parse_judge_response("   ") is None


# --- prompt builders -------------------------------------------------------

def test_build_judge_prompt_contains_all_fields_and_instruction():
    msgs = aa.build_judge_prompt("Q?", "gold", "model said gold")
    assert msgs[0]["role"] == "system"
    assert "YES or NO" in msgs[0]["content"]
    body = msgs[1]["content"]
    assert "Question: Q?" in body
    assert "Gold answer: gold" in body
    assert "Model answer: model said gold" in body


def test_build_generate_prompt_contains_question_and_context():
    msgs = aa.build_generate_prompt("what tower?", "[1] the eiffel tower")
    assert msgs[1]["content"].startswith("Question: what tower?")
    assert "[1] the eiffel tower" in msgs[1]["content"]


# --- answer_row shape ------------------------------------------------------

def test_answer_row_fields():
    r = aa.answer_row("q1", "what?", "gold", "gold", True, True, True, 3)
    assert set(r) == {"qid", "question", "gold", "answer", "judge_yes",
                      "exact_match", "recall_at_1_hit", "n_hits"}
    assert r["judge_yes"] is True and r["n_hits"] == 3


# --- evaluate (stubbed LLM + stubbed fetcher) ------------------------------

def test_evaluate_judge_yes_path():
    ranks = [1, None]  # ex0 recalled@1, ex1 missed
    retrieved = [["p_answer"], ["p_noise"]]
    out = aa.evaluate(FIXTURE_EXAMPLES, retrieved, ranks,
                      fetcher=_stub_fetcher, chat=_stub_chat_yes,
                      answer_model="stub", judge_model="stub", topn=5,
                      logger=lambda *a: None)
    assert out["n"] == 2 and out["judged"] == 2
    assert out["answer_accuracy"] == 0.5  # ex0 yes, ex1 no (wrong answer)
    assert out["exact_match"] == 0.5
    assert out["recall_at_1"] == 0.5
    assert out["rows"][0]["judge_yes"] is True
    assert out["rows"][1]["judge_yes"] is False


def test_evaluate_judge_no_path():
    ranks = [1, 1]
    retrieved = [["p_answer"], ["p_answer"]]
    out = aa.evaluate(FIXTURE_EXAMPLES, retrieved, ranks,
                      fetcher=_stub_fetcher, chat=_stub_chat_no,
                      answer_model="stub", judge_model="stub", topn=5,
                      logger=lambda *a: None)
    assert out["judged"] == 2
    assert out["answer_accuracy"] == 0.0
    assert out["rows"][0]["judge_yes"] is False
    # recall@1 still computed from ids/ranks, independent of the judge
    assert out["recall_at_1"] == 1.0


def test_evaluate_isolates_per_example_error():
    # First example's fetcher raises -> row recorded as error, not fatal.
    def bad_fetcher(ids):
        raise RuntimeError("retrieve exploded")

    def ok_chat(messages, model, max_tokens):
        if _is_judge(messages):
            return "YES"
        return "ok"
    out = aa.evaluate([{"question": "q", "answer": "a"}], [["p"]], [1],
                      fetcher=bad_fetcher, chat=ok_chat,
                      answer_model="stub", judge_model="stub", topn=5,
                      logger=lambda *a: None)
    assert out["n"] == 1
    assert out["judged"] == 0  # the one example errored before judging
    assert out["rows"][0]["answer"] == "<error: retrieve exploded>"
    assert out["rows"][0]["judge_yes"] is None
    # recall@1 + n_hits still computed from ids/ranks (no LLM needed)
    assert out["rows"][0]["recall_at_1_hit"] is True
    assert out["rows"][0]["n_hits"] == 1


# --- pure import sanity ----------------------------------------------------

def test_module_imports_without_llm_or_network_side_effects():
    # answer_accuracy must import cleanly (stdlib only) — no hidden runtime pull.
    import importlib
    mod = importlib.import_module("answer_accuracy")
    assert mod._DEFAULT_BASE_URL  # env-derived default present
    assert callable(mod.evaluate) and callable(mod.parse_judge_response)


if __name__ == "__main__":
    # Standown runner: no pytest needed to verify the deterministic core.
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)
           and not isinstance(v, type)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} tests passed "
          f"(standalone runner)")
    sys.exit(1 if failures else 0)
