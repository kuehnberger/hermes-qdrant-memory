# Recall ablation: what each hardening arm actually buys

Measured 2026-10-19 on LongMemEval-S, 500 questions, 948 sessions seeded, top-10
window, harness `2.0.0-result-boundary`, collection `hermes_memories_eval`. Each
arm drives the live plugin end to end; the arms ran serially.

| arm | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| dense (default) | 0.340 | 0.600 | 0.686 | 0.449 | 0.437 |
| `hybrid` | **0.426** | 0.680 | **0.768** | 0.532 | 0.517 |
| `hybrid` + `temporal` | 0.432 | 0.678 | 0.764 | 0.536 | 0.521 |
| `hybrid` + `temporal` + `rerank` | **0.438** | 0.680 | **0.768** | **0.539** | **0.523** |

## Reading it honestly

**Hybrid is the large, clear win:** recall@1 0.340 -> 0.426 (+8.6 points),
recall@10 0.686 -> 0.768 (+8.2), MRR 0.449 -> 0.532. This reproduces the figure
already recorded in the CHANGELOG (0.340 -> 0.426, "+8.6 pts") exactly, on a
freshly re-run harness. It is the number to publish.

**`temporal` adds a small amount** — +0.006 recall@1, +0.004 MRR over hybrid.
LongMemEval questions rarely carry an explicit time expression, so the window
fires on a minority of them. The gain is real but small, and on n=500 a delta of
0.006 is 3 questions. **Not evidence of harm, and not strong evidence of
benefit** — judging temporal recall needs a corpus built for it.

**`rerank` adds a small amount** — +0.012 recall@1 over hybrid (0.426 -> 0.438),
+0.007 MRR. Same caveat: 6 questions on n=500, inside the noise band. It is a
re-rank, so it can only permute candidates retrieval already returned and cannot
add a hit that was never retrieved; that structural ceiling is why it ships
opt-in and **off by default** despite the positive sign.

**Nothing here justifies turning a knob on by default.** Hybrid earns it;
temporal and rerank are defensible only on corpora that exercise them.

## The `temporal` arm, measured on a corpus that CAN judge it

LongMemEval cannot judge `temporal`: its questions rarely carry an explicit time
expression, so the window almost never fires and the arm measures nothing.
`scripts/temporal_probe.py` builds the corpus that can. 8 topics x 8 months = 64
entries, all in ONE session (the only scope where the feature can act), with each
topic's entries semantically near-identical so the DATE is the sole
discriminator. Chance at recall@1 is 1/8 = 0.125.

| arm | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| `temporal` off | 0.562 | 0.969 | 0.969 | 0.755 |
| `temporal` on | **0.969** | 0.969 | 0.969 | **0.969** |
| delta | **+0.406** | +0.000 | +0.000 | **+0.214** |

Reproduced twice (MRR 0.755 / 0.758 — float noise only).

Reading it honestly:

- **When the query names a time, `temporal` is a large RANKING win** — recall@1
  +40.6 points, MRR +0.214.
- recall@5 and @10 were already at ceiling without it, so the gain is entirely
  in ordering: it promotes the right month to rank 1. It does not retrieve
  anything that was missing, and it cannot.
- This corpus is built to make time the discriminator, so it is the **best case**
  for the feature, not a naturalistic one. Real-world value tracks how often your
  queries actually carry a time expression — rare in LongMemEval, plausibly
  common in a personal assistant.
- **Scope limits it.** `prefetch` filters recall to the current session, so
  temporal reasoning operates WITHIN a session. It cannot reach across sessions,
  and no setting changes that.
- It is never harmful: with no time expression the window does not fire and the
  arm is inert (see the LongMemEval table above, where the two arms agree).

## A contaminated run, recorded so it is not repeated

The first attempt at this ablation produced dense recall@1 **0.170** and made
temporal and rerank look useless or harmful. That run was WRONG.

Cause, established with evidence: `hermes_memories_eval` already existed at the
time (left over from earlier work), and `QdrantMemoryProvider.initialize()` only
creates a collection when it is absent — so the FIRST arm inherited a stale
collection, seeded into it, and queried a store polluted by leftover points.
Every later arm created a fresh collection (the harness drops it at the end of
each run), which is why only the first arm was affected. Re-running with the
collection absent reproduced the recorded baseline exactly (0.340/0.600/0.686).

The contaminated logs are kept at `~/.hermes/state/qd/ablation-run1-contaminated/`
as the counter-example. `run_ablation.sh` now drops the eval collection before
the first arm so a stale store cannot contaminate an arm again.

**Lesson worth keeping: an eval arm that inherits pre-existing store state
measures the store, not the arm.**

## Reproduce

```bash
bash ~/.hermes/state/qd/run_ablation.sh          # serial, ~5 minutes
FORCE=1 bash ~/.hermes/state/qd/run_ablation.sh  # re-run even if results exist
python3 scripts/longmemeval_eval.py --hybrid --temporal   # a single arm
```

Run it under the **hermes venv** — that is the interpreter with `datasets`; the
repo venv does not have it. The dataset is staged at
`~/.hermes/state/qdrant/eval-data/`, so no network is needed.

## Answer-accuracy: the metric the comparison exposed

The recall table above is a retrieval-only measurement. EntropicMem and other
peers publish end-to-end **answer-accuracy** — an LLM judge scores whether an
answer derived from retrieved context matches the gold — which is a different
axis (index + generator + judge, not just the index). Printing "recall@10 0.768
vs their 0.X" is a category error unless both axes are on the table.

`scripts/longmemeval_eval.py` now has an opt-in `--answer-accuracy` arm
(`scripts/answer_accuracy.py`) that, after the recall pass, does exactly the
shared-generator + YES/NO-judge shape prescribed by the memory-retrieval-eval
skill's head-to-head recipe:
- take the top-10 retrieved point ids (rank order) from the recall pass;
- fetch the SAME text payload the agent would actually receive via
  `provider._client.retrieve` (not the rank-only line parse);
- generate an answer from that context, then judge YES/NO against the gold;
- report `answer_accuracy` + `exact_match` + `recall_at_1` together, so the
  retrieval ceiling stays visible next to the generator/judge score.

Off by default; the recall metrics and the published numbers above are
unchanged when it is not requested. The judge is forced to a YES/NO reply
(json_schema fails on most free `nous/*` models), and the deterministic core is
unit-tested with stubs in `tests/test_answer_accuracy.py` — 14/14 in-process,
no Qdrant/network.

Honest scope and a hard non-claim:
- No quotable number is added here. The published recall figures stand on their
  own; an answer-accuracy figure would require a full judgment run.
- The judge transport targets the local lightproxy
  (`http://172.17.0.1:8088/v1`, no auth, `<provider>/<model>` form). That
  service is **inactive** on this host (`hermes-meter-proxy.service` is dead), so
  the transport is not smoke-tested here — only the pure logic is.
- A corpus-scale pilot (>=40 examples on the LongMemEval 1,000-set, per the
  memory-retrieval-eval skill) belongs on the VPS via `default` +
  `heavy-run.sh`. Until then the arm is a measurement tool, not a result.
