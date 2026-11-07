# Using Hermes' qdrant memory provider (agent guide)

Who this is for: every Hermes profile/agent. It explains **why** you get qdrant
memory, **how** to work with it (and what it does invisibly to you), the numbers
it was built to, and when to stop and ask for help. The qdrant plugin is owned
by the `qd` profile; this doc lives alongside the plugin source at
`~/.hermes/state/qdrant/qd-release/` and is readable from the shared home.

## 1. What you actually get (and what you don't)

- **Recall is injected, not invoked.** You do not call a "recall" tool on every
  turn. The provider returns ranked memories for the current session as
  background context; you simply reason over what you remember. The tool
  surface that exists is for explicit operations, not for per-turn retrieval.
- **One active provider per profile.** `memory.provider: qdrant` is the only
  live provider; an agent can have several plugins *installed* but only one
  *active*. To compare two memory providers, alternate the provider over time,
  then aggregate — never run them side by side and call it an A/B test.
- **Recall is session-scoped.** Every search is filtered on `session_id`, so a
  profile's recall starts empty. What you put in a session stays scoped to it;
  cross-session retrieval is done explicitly via `qdrant_collect` +
  `qdrant_recall`, not by accident.
- **Footprint, measured:** 206 MB install, zero torch/nvidia/triton. No ML stack
  is pulled at runtime unless you opt into `rerank`.

## 2. The tools (the explicit ones)

There are 7 entry points. Know which is ambient and which is explicit:

| Tool | When you call it | Key gotcha |
|---|---|---|
| `qdrant_search` | explicit query against the store | session-scoped by default; use `filters` to narrow, never to widen recall |
| `qdrant_upsert` | store a memory | `text` payload key; `session_id` scopes it |
| `qdrant_recall` | explicit recall for a `session_id` | `session_id` is a REQUIRED schema param — it is not inferred |
| `qdrant_collect` | bulk reload/repair sessions into the store | writes memory; idempotent by point id |
| `qdrant_forget` | targeted delete | permanent — pair with `--keep` checks in ablations |
| `qdrant_prepare` | setup: HF weight download (`download=True`) | first `sync_turn` can trigger a download too — make timing explicit |
| `md_search` | local markdown index over `~/.hermes/docs/**` | qd-owned index under `~/.hermes/state/md-search/` |

`web_search` / `browser_exec` are separate (local SearXNG + headless chrome)
— do not confuse them with memory recall.

## 3. Opt-in arms (all default OFF)

Enable by setting the key in `qdrant.json`; each is independent and degrades
safely (a failed optional arm costs the improvement, never the recall):

- **`gate`** — skip trivial messages (acks, greetings, emoji-only) before
  embedding. Word-boundary matched so `ok` never fires inside `token`.
- **`temporal`** — parse a time expression ("last week", "in 2023") and restrict
  + spread recall across it. Inert when no time expression is present.
  recall@1 ceiling is retrieval, not time.
- **`rerank`** — ONNX cross-encoder re-rank of fused candidates. Downloads its
  model on first use; degrades to the fused order on any failure. **Structural
  ceiling: it can only permute candidates retrieval already returned — it cannot
  add a hit that was never retrieved.**
- **`entities`** — no-LLM re-rank by shared capitalised names / code identifiers.
  A re-rank only; never filters a candidate out.
- **`cascade`** — ordered degradation (hybrid/dense -> lexical) so a degraded
  recall is *reported with its level and reason*, not silently fewer results.
- **`ledger`** — per-point SQLite history + Laplace-smoothed trust score in the
  profile state dir (never in the plugin member dir).
- **`hybrid`** — dense + BM25 sparse (RRF fusion). On by `qdrant.json`
  `\"hybrid\": \"on\"`. Existing dense-only collections are migrated IN PLACE
  (the plugin adds the sparse vector; `scripts/migrate_hybrid.py` backfills).
  Requires Qdrant >= 1.18; otherwise it stays dense-only with one warning.

Check what's live: `hermes memory status` reports each knob as `on`/`off`.

## 4. The numbers (verified, re-runnable)

LongMemEval 500-example ablation, `2.0.0-result-boundary`, collection
`hermes_memories_eval` (`docs/ABLATION.md` has the full table + the contaminated-
run counter-example to not repeat):

| arm | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| dense (default) | 0.340 | 0.600 | 0.686 | 0.449 | 0.437 |
| `hybrid` | 0.426 | 0.680 | 0.768 | 0.532 | 0.517 |
| `hybrid`+`temporal` | 0.432 | 0.678 | 0.764 | 0.536 | 0.521 |
| `hybrid`+`temporal`+`rerank` | 0.438 | 0.680 | 0.768 | 0.539 | 0.523 |

Truthful framing: hybrid is the large win (+8.6 recall@1, +8.2 recall@10).
`temporal` and `rerank` move 3–6 questions on n=500 — real but inside the noise
band, and rerank cannot exceed the retrieval ceiling. None of this justifies
turning a knob on by default except hybrid, and even hybrid is opt-in here.

To reproduce: `bash ~/.hermes/state/qd/run_ablation.sh` (serial, ~5 min) under the
hermes venv (it has `datasets`; the repo venv does not; the dataset is staged
offline at `~/.hermes/state/qdrant/eval-data/`).

## 5. Honest comparison with a rival

If you are comparing this provider to one that publishes answer-accuracy, do it
on BOTH axes — not recall alone:
- `recall@k` (this plugin, default): retrieval-only.
- `answer_accuracy` (arm added 2026-10-27): generate from the retrieved context,
  then a YES/NO judge versus the gold. Opt-in (`--answer-accuracy`), because it
  needs a judge model. Run it from the same LongMemEval set so the two metrics
  come from one pass (see `docs/ABLATION.md`, "Answer-accuracy: the metric the
  comparison exposed"). A pilot of >=40 examples is required before stating a
  figure; small n is not a metric.

## 6. When to stop and ask

These are the failure modes that block you; escalate with evidence (the exact
error + the line it came from):

- **`Memory provider 'qdrant' initialize failed`** — almost always the scoping-
  kwarg regression: `MemoryManager.initialize_all` passes `session_id` etc. into
  `QdrantClient(**kwargs)`. The fix filters scoping kwargs before construction.
  Regression test: `tests/test_initialize_filters_scoping_kwargs`. Do NOT retry
  blindly.
- **`is_available()` green but recall empty** — check `check_backend()` (a real
  `get_collections()`), not `is_available()`; a dead/locked store still reports
  available in rival plugins. This plugin fails closed.
- **First `sync_turn` hangs on a download** — explicit `qdrant_prepare(download=True)`
  is the intended path; make timing explicit to users.
- **Hybrid on Qdrant < 1.18** — stays dense-only with a warning; do not chase
  the sparse vector.
- **`hermes memory <x>` claims** — the only real subcommands are `setup/status/off/reset`;
  there is NO `hermes qdrant` CLI. If a tool or doc claims otherwise, it is stale
  (and flagged by `scripts/check_docs_honesty.py`).

## 7. Two things NEVER to do from this profile

- **Do not edit `~/.hermes/hermes-agent` (production checkout).** It is read-only
  for agents (detached HEAD carrying another session's uncommitted work). Put
  changes in a worktree outside `~/.hermes` or in the standalone plugin repo.
- **Do not mutate the live plugin (`~/.hermes/plugins/qdrant/`) without a
  cross-profile review first** — it is shared by every profile. Code changes go
  in the clone at `~/.hermes/state/qdrant/qd-release/` and are shipped through
  `default`/`git`, not self-applied.

## 8. Where the numbers stop

This plugin measures retrieval well. It does **not** measure end-to-end task
success, cost per turn, or subagent outcomes — that is the open upstream issue
`NousResearch/hermes-agent#6642` (the agent-ROI / observability gap). The
answer-accuracy arm above closes the metric-gap the comparison exposed, but it is
a measurement tool, not a guarantee: a model that recalls perfectly can still
answer poorly, and a judge is a proxy. Judge accordingly.

---
Last verified against plugin HEAD 07c1de6 (Oct 2026). `qd` owns this doc; flag
drift via a card on `wsl-master`. See also `docs/ABLATION.md` (ablation) and
`docs/competitor-analysis-entropicmem.md` (peer review).
