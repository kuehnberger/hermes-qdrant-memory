# Recall ablation: what each hardening arm actually buys

Measured 2026-10-19 on LongMemEval-S, 500 questions, 948 sessions seeded, top-10
window, harness `2.0.0-result-boundary`, collection `hermes_memories_eval`. Each
arm runs the live plugin end to end; the arms ran serially.

| arm | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| dense (default) | 0.170 | 0.472 | 0.598 | 0.308 | 0.291 |
| `hybrid` | **0.430** | 0.676 | 0.768 | **0.535** | 0.522 |
| `hybrid` + `temporal` | 0.430 | **0.682** | 0.768 | 0.533 | 0.521 |
| `hybrid` + `temporal` + `rerank` | 0.430 | 0.680 | 0.764 | 0.534 | 0.522 |

## Reading it honestly

**Hybrid is the whole win.** recall@1 +26.0 points, recall@5 +20.4, MRR +0.227.
Every other arm is inside the noise.

**`temporal` is neutral here, and that is not a verdict on the feature.**
LongMemEval questions rarely carry an explicit time expression, so the window
almost never fires. Judging temporal recall needs a corpus built for it; this
run cannot.

**`rerank` is neutral-to-slightly-negative.** This arm is not broken — the
cross-encoder model downloads and demonstrably reorders candidates. The reason
is structural: reranking PERMUTES a candidate set that retrieval already
returned, so it can change rank but cannot add a hit that was never retrieved.
On this data its ordering was marginally worse than RRF's, which is why
`rerank` ships opt-in and **off by default**.

## Known discrepancy — do not publish a dense figure yet

This run measures dense recall@1 at 0.170; an earlier run recorded 0.340.
Hybrid is stable across both (0.430 now vs 0.426 then). The dense gap is
unresolved — most likely a different embedding model or eval-collection state.
Reconcile before citing a dense baseline.

## Reproduce

```bash
bash ~/.hermes/state/qd/run_ablation.sh          # serial, ~5 minutes
python3 scripts/longmemeval_eval.py --hybrid --temporal   # a single arm
```

Run it under the **hermes venv** — that is the interpreter with `datasets`; the
repo venv does not have it. The dataset is staged at
`~/.hermes/state/qdrant/eval-data/`, so no network is needed.
