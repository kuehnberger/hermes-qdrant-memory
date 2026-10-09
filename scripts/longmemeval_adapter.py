#!/usr/bin/env python3
"""LongMemEval adapter — imports the LIVE plugin from ~/.hermes/plugins/qdrant."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset


def _flatten_sessions(sessions: list[list[dict]]) -> list[tuple[str, str]]:
    """haystack_sessions is a list of sessions; each session is a list of messages.
    Flatten all sessions into (user, assistant) turn pairs."""
    turns = []
    for session in sessions:
        i = 0
        while i < len(session):
            msg = session[i]
            if msg.get("role") == "user" and i + 1 < len(session):
                nxt = session[i + 1]
                if nxt.get("role") == "assistant":
                    u = str(msg.get("content", ""))
                    a = str(nxt.get("content", ""))
                    if u.strip() and a.strip():
                        turns.append((u, a))
                    i += 2
                    continue
            i += 1
    return turns


def _recall_at_k(hits: list[str], answer: str, k: int) -> int:
    ans = answer.lower().strip()
    for h in hits[:k]:
        if ans in h.lower():
            return 1
    return 0


def _mrr(hits: list[str], answer: str) -> float:
    ans = answer.lower().strip()
    for rank, h in enumerate(hits, 1):
        if ans in h.lower():
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(hits: list[str], answer: str, k: int) -> float:
    import math
    ans = answer.lower().strip()
    for rank, h in enumerate(hits[:k], 1):
        if ans in h.lower():
            return 1.0 / math.log2(rank + 1)
    return 0.0


def main():
    # ---- load dataset ----
    print("Loading LongMemEval (LIXINYI33/longmemeval-s)...")
    ds = load_dataset("LIXINYI33/longmemeval-s", split="train")
    print(f"  {len(ds)} examples")

    # ---- import the LIVE plugin from its install dir ----
    PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "qdrant"
    sys.path.insert(0, str(PLUGIN_DIR.parent))  # insert ~/.hermes/plugins
    import qdrant as provider_mod

    provider = provider_mod.QdrantMemoryProvider()
    if provider is None:
        print("FATAL: provider instantiation failed")
        return 2

    # Clear env layer
    for k in ("QDRANT_URL", "QDRANT_API_KEY"):
        os.environ.pop(k, None)

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        os.environ["HERMES_HOME"] = str(home)
        # The provider reads its collection from config (default "hermes_memories").
        # A separate eval collection is what keeps benchmark data out of the
        # live store — the default would silently ingest into production.
        EVAL_COLLECTION = "hermes_memories_eval"
        (home / "qdrant.json").write_text(json.dumps({
            "display": {"level": "off"},
            "collection": EVAL_COLLECTION,
        }))
        provider.initialize(session_id="longmemeval-eval", hermes_home=str(home))
        # Hard guard: refuse to run if the resolved collection is the live one.
        # This check is what failed silently on 2026-10-08, when the default
        # collection was used and 127,850 benchmark points landed in production.
        if provider._collection == "hermes_memories":
            print("FATAL: provider resolved to the production collection "
                  f"'{provider._collection}'. Refusing to ingest benchmark data.")
            return 3
        print(f"Provider initialized against {provider._collection}")

        # ---- ingest each example's haystack ----
        print("Ingesting haystack conversations...")
        for idx, ex in enumerate(ds):
            sid = ex.get("question_id", f"ex{idx}")
            sessions = ex.get("haystack_sessions", [])
            turns = _flatten_sessions(sessions)
            for u, a in turns:
                provider.handle_tool_call("qdrant_upsert", {"text": f"User: {u}\nAssistant: {a}", "session_id": sid})
        print(f"  ingested {idx + 1} examples")

        # ---- evaluate recall ----
        print("Running recall queries...")
        recall5 = recall10 = 0
        mrr_sum = ndcg5_sum = 0.0
        valid = 0

        for idx, ex in enumerate(ds):
            q = str(ex.get("question", "")).strip()
            ans = str(ex.get("answer", "")).strip()
            if not q or not ans:
                continue
            res = provider.handle_tool_call("qdrant_search", {"query": q, "limit": 10, "session_id": ex.get("question_id", f"ex{idx}")})
            hits = [line.split("] ", 1)[1] if "] " in line else line for line in res.splitlines()]
            recall5 += _recall_at_k(hits, ans, 5)
            recall10 += _recall_at_k(hits, ans, 10)
            mrr_sum += _mrr(hits, ans)
            ndcg5_sum += _ndcg_at_k(hits, ans, 5)
            valid += 1

        if valid == 0:
            print("No valid examples to evaluate")
            return 1

        print(f"\n=== LongMemEval results on {valid} examples ===")
        print(f"recall@5 : {recall5 / valid:.3f}")
        print(f"recall@10: {recall10 / valid:.3f}")
        print(f"MRR      : {mrr_sum / valid:.3f}")
        print(f"nDCG@5   : {ndcg5_sum / valid:.3f}")
        return 0


if __name__ == "__main__":
    import os
    sys.exit(main())