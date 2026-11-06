#!/usr/bin/env python3
"""Answer-accuracy eval arm for the LongMemEval harness.

The comparison lesson (see docs/competitor-analysis-entropicmem.md §"Eval harness"
and references/eval-claims-matrix.md): rivals such as EntropicMem publish
end-to-end ANSWER ACCURACY — an LLM judge scores whether an answer derived from
retrieved context matches the gold — whereas we publish recall@k, a
retrieval-only metric. Juxtaposing the two is a category error: recall measures
the index, answer-accuracy measures index+generator+judge.

This module closes the gap by adding a **companion** answer-accuracy arm to
`scripts/longmemeval_eval.py` so that on the SAME LongMemEval 1,000-example set
we can report recall@k (retrieval) AND answer-accuracy (judge) from one run — a
like-for-like basis for any comparison. It is strictly opt-in (`--answer-accuracy`
flag) and OFF by default, so the default recall path and its numbers are
unchanged.

Design rules (from memory-retrieval-eval, "Head-to-head against an external
competitor"):
  * the deterministic core runs OFFLINE against a hand-written fixture corpus;
  * the live call is a single injected `chat_fn`, so the pure logic is unit-tested
    with a stub and never needs a real model to be correct;
  * the judge is a forced YES/NO question, not a json_schema, because most free
    nous/* models 400 on json_schema but answer a binary chat reliably;
  * a single shared generator answer + one YES/NO judge per example;
  * only the standard library is used (urllib.request) — no `openai`/`anthropic`
    pinned, and this lives in `scripts/` (outside the no-LLM boundary that
    test_no_llm_calls.py scans for runtime modules).

Honest scope: this adds the *tool* to measure answer-accuracy. It does NOT add a
quotable number. A full measurement is a heavy, corpus-scale run that belongs on
the VPS (hand off via default + heavy-run.sh; pilot ~40 examples minimum before
any figure is stated). The judge transport targets the local lightproxy
(http://172.17.0.1:8088/v1, no auth, <provider>/<model> form per
hermes-memory-provider-plugin §LLM routing) — that endpoint is not live on this
WSL host (hermes-meter-proxy.service is inactive), so the transport is not
smoke-tested here; only the deterministic core is verified in-process.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable, Iterable, Sequence

# Local lightproxy base URL. Per hermes-memory-provider-plugin §LLM routing:
# http://172.17.0.1:8088/v1, bearer ignored, model sent as <provider>/<model>.
# Override with ANSWER_ACC_BASE_URL for tests / the VPS run. The module never
# hardcodes a real model — operators always name one explicitly.
_DEFAULT_BASE_URL = os.environ.get("ANSWER_ACC_BASE_URL", "http://172.17.0.1:8088/v1")
_DEFAULT_MAX_TOKENS = 256

# A chat function: (messages, model, max_tokens) -> content string.
ChatFn = Callable[[Sequence[dict], str, int], str]

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[.?!,;:'\"()\[\]{}\-]+")
# Forced binary judge: match a whole word YES/NO (case-insensitive).
_YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


# --- deterministic core (no LLM, no network) ---------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding/trailing punctuation.

    Pure function: cheap to hold and to test, and the flattery is applied
    IDENTICALLY to the gold and the model answer, so a delta measured downstream
    is a real delta between the two arms being compared.
    """
    t = _WS_RE.sub(" ", text or "").strip().lower()
    t = t.strip("'\")")
    t = _TRAILING_PUNCT.sub("", t).strip()
    return t


def exact_match_answer(gold: str, answer: str) -> bool:
    """Generous substring proxy (matches HYBRID.md's stated proxy).

    Correct if the normalized gold is a substring of the normalized answer (the
    model quoted/contained the gold), or vice-versa, or they are equal.
    This is a fallback signal only — the judge is the primary axis.
    """
    g = normalize_answer(gold)
    a = normalize_answer(answer)
    if not g or not a:
        return False
    return g == a or g in a or a in g


def build_judge_prompt(question: str, gold: str, answer: str) -> list[dict]:
    """YES/NO judge prompt: answer is the question, gold, and model answer."""
    return [
        {"role": "system",
         "content": ("You are a strict binary judge. Answer exactly YES or NO "
                     "and nothing else.")},
        {"role": "user",
         "content": (
             f"Question: {question}\n\n"
             f"Gold answer: {gold}\n\n"
             f"Model answer: {answer}\n\n"
             "Did the model answer sufficiently address the question compared "
             "to the gold answer? Reply with exactly YES or NO."
         )},
    ]


def build_generate_prompt(question: str, context_text: str) -> list[dict]:
    """Generator prompt: answer from the retrieved context only."""
    return [
        {"role": "system",
         "content": ("You answer concisely from the provided context. If the "
                     "context does not contain the answer, say you don't know.")},
        {"role": "user",
         "content": (
             f"Question: {question}\n\n"
             f"Context:\n{context_text}\n\n"
             "Answer:"
         )},
    ]


def parse_judge_response(text: str) -> bool | None:
    """Return True/False from a forced YES/NO response; None if ambiguous.

    Takes the LAST YES/NO token the model emitted (a forced binary should end
    on its verdict), so trailing re-statement or repetition does not confuse it.
    """
    if not text:
        return None
    matches = _YES_NO_RE.findall(text)
    if not matches:
        return None
    last = matches[-1].upper()
    if last == "YES":
        return True
    if last == "NO":
        return False
    return None


def format_context(retrieved_texts: Iterable[str]) -> str:
    """Join retrieved memory texts with numbered separators the generator cites."""
    parts = []
    for i, t in enumerate(retrieved_texts, 1):
        parts.append(f"[{i}] {t.strip()}")
    return "\n".join(parts)


def answer_row(
    qid, question, gold, answer,
    correct: bool | None, exact: bool, recall_at_1_hit: bool, n_hits: int,
) -> dict:
    return {
        "qid": qid,
        "question": question,
        "gold": gold,
        "answer": answer,
        "judge_yes": correct,
        "exact_match": exact,
        "recall_at_1_hit": recall_at_1_hit,
        "n_hits": n_hits,
    }


# --- LLM transport (isolated behind chat_fn for testability) -----------------

def lightproxy_chat(messages: Sequence[dict], model: str,
                    max_tokens: int = _DEFAULT_MAX_TOKENS) -> str:
    """OpenAI-compatible chat to the local lightproxy (no auth header)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{_DEFAULT_BASE_URL.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310  (local LAN)
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def default_chat(messages, model, max_tokens=_DEFAULT_MAX_TOKENS) -> str:
    """Default chat fn pointed at the local lightproxy. Override in tests."""
    return lightproxy_chat(messages, model, max_tokens)


class AnswerAccuracyJudge:
    """Generate an answer from retrieved context, then judge it against gold."""

    def __init__(self, chat: ChatFn, answer_model: str, judge_model: str):
        self._chat = chat
        self.answer_model = answer_model
        self.judge_model = judge_model

    def generate(self, question: str, context_text: str) -> str:
        msgs = build_generate_prompt(question, context_text)
        return self._chat(msgs, self.answer_model, _DEFAULT_MAX_TOKENS)

    def judge(self, question: str, gold: str, answer: str) -> bool | None:
        msgs = build_judge_prompt(question, gold, answer)
        # Judge answers are one token; cap small to fail fast and cheap.
        return parse_judge_response(self._chat(msgs, self.judge_model, 16))


# --- fetcher (Qdrant retrieve bridge) ----------------------------------------
# Kept as a factory so the pure `evaluate` loop never imports the provider.

def fetcher_from_provider(provider, collection: str):
    """Return a point_id -> text fetcher backed by provider._client.retrieve.

    Uses the SAME collection the eval just seeded/queried. Defensive about id
    type (UUID vs str) and payload shape (PointStruct vs dict).
    """
    client = provider._client

    def _fetch(point_ids: Sequence[str]) -> list[str]:
        ids = [str(p) for p in point_ids if p]
        if not ids:
            return []
        try:
            pts = client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return ["" for _ in ids]
        by_id: dict[str, str] = {}
        for pt in pts:
            pid = str(getattr(pt, "id", pt.get("id") if isinstance(pt, dict) else None))
            payload = getattr(pt, "payload", None) or (pt.get("payload") if isinstance(pt, dict) else None) or {}
            by_id[pid] = (payload.get("text") or "") if isinstance(payload, dict) else ""
        return [by_id.get(i, "") for i in ids]

    return _fetch


# --- orchestrator (pure; depends only on injected fetcher + chat) -------------

def evaluate(
    examples,
    retrieved_ids_per_example: list[list[str]],
    ranks: list[int | None],
    fetcher: Callable[[Sequence[str]], list[str]],
    chat: ChatFn,
    answer_model: str,
    judge_model: str,
    topn: int = 10,
    logger=print,
    log_every: int = 100,
) -> dict:
    """Run the answer-accuracy arm over `examples`.

    For each example: take the top-`topn` retrieved point ids (rank order),
    fetch their text, generate an answer, then judge YES/NO against the gold.

    `fetcher` and `chat` are injected so the entire loop is unit-testable with
    stubs and never needs Quan t or a live model. A per-example error is
    recorded (answer set to "<error>", judge_yes None) and not fatal — one
    bad example must not abort a 1,000-example run.
    """
    judge = AnswerAccuracyJudge(chat=chat, answer_model=answer_model,
                                judge_model=judge_model)
    rows: list[dict] = []
    n = len(examples)
    for i, ex in enumerate(examples):
        qid = ex.get("question_id", "") or str(i)
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        ids = (retrieved_ids_per_example[i]
               if i < len(retrieved_ids_per_example) else [])[:topn]
        n_hits = len(ids)
        recall_at_1_hit = bool(ranks[i]) and ranks[i] is not None and ranks[i] == 1
        answer = ""
        correct: bool | None = None
        try:
            texts = fetcher(ids)
            context_text = format_context(texts)
            answer = judge.generate(question, context_text)
            correct = judge.judge(question, gold, answer)
        except Exception as e:  # one bad example must not abort the whole run
            answer = f"<error: {e}>"
            correct = None
        exact = exact_match_answer(gold, answer)
        rows.append(answer_row(qid, question, gold, answer, correct, exact,
                               recall_at_1_hit, n_hits))
        if log_every and (i + 1) % log_every == 0:
            logger(f"  answer-accuracy: {i + 1}/{n}")

    judged = [r for r in rows if r["judge_yes"] is not None]
    exact_hits = sum(1 for r in rows if r["exact_match"])
    acc = (sum(1 for r in judged if r["judge_yes"]) / len(judged)
           if judged else 0.0)
    r1 = (sum(1 for r in rows if r["recall_at_1_hit"]) / n
          if n else 0.0)
    return {
        "enabled": True,
        "n": n,
        "judged": len(judged),
        "answer_accuracy": round(acc, 4),
        "exact_match": round(exact_hits / n, 4) if n else 0.0,
        "recall_at_1": round(r1, 4),
        "answer_model": answer_model,
        "judge_model": judge_model,
        "note": ("n is the number of judged examples. A corpus-scale pilot "
                 "(>=40 examples recommended) must be run on the VPS before any "
                 "figure is quotable; this arm does not change the recall metrics."),
        "rows": rows,
    }
