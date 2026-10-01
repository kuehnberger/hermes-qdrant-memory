# Reference: EntropicMem (competitor analysis)

Analysis of `Ufonik88/EntropicMem` at pinned commit `7e02412`, a community
memory provider in the same catalog category. Three days old at the time of
writing (repo created 2026-09-26), version 2.8.1.

Read this before adopting any idea from a peer plugin. Half of what makes that
plugin look strong is **not** worth copying, and two of its patterns are
actively unsafe. This file records which is which, with code citations, so the
next person does not have to re-derive it.

The headline finding: **we are ahead on the properties that matter most**
(availability honesty, error tiering, dependency weight) and behind on
**measurement and lifecycle**. Copy the second category. Do not touch the first
— we are the good version of it.

## Scale context (not a quality signal)

| | EntropicMem | Ours |
|---|---|---|
| Python lines | 13,604 (plugin) / ~32,500 (repo) | 3,064 (plugin, excl. tests) |
| `__init__.py` | 1,693 | 1,449 |
| Tools | 7 | 5 |
| Hooks | 5 | 0 (three no-op stubs) |
| Tests | 895 | 134 (plus 2,451 lines of test code) |
| CLI commands | 34 | 0 |
| Screenshots | 5 | 0 |
| Eval harness | `evals/` with datasets + baselines | `scripts/retrieval_eval.py` (36-query paraphrase corpus, recall@k/MRR/nDCG@5) — **added 2026-10-02, `fc18af9`**, closing the gap this table recorded |

A 34-command CLI is a **second user interface** layered on top of the tool
surface, and it must be documented, tested, and kept in sync. We have no reason
to take that on. If we ever want operator actions, the right shape is a handful
of `hermes qdrant` subcommands for status/backup/restore — not 34.

Reviewer-facing consequence: every feature added for parity **raises review
cost** at a SHA pin. Our smaller surface is a *better submission shape*, not a
gap. Do not close the feature-count gap; close the evidence gap.

## Where they are RIGHT and we should adopt

### 1. `on_pre_compress` with a fail-closed checkpoint contract

`__init__.py:1270-1319`. Their `on_pre_compress` sets
`pre_compress_checkpoint_api_version = 2`; a non-empty return *guarantees* the
checkpoint landed, else it **raises** rather than claiming success.

Why it matters: compaction is the one irreversible event in a session. Theirs
extracts standing constraints ("never push to main without CI") and durably
checkpoints them as an episode, so they survive compaction. Ours drops the
transcript. For a vector store the fix is a **session-digest point** — one
upsert before the tail is gone, with an id derived from the session id for
idempotence.

Adopt: the v2 fail-closed contract, and the digest point. Effort M.

### 2. A retrieval-quality eval harness

`evals/metrics.py`, `evals/runner.py:23`, `evals/noise.py`. They measure
`recall@5 / MRR / nDCG@5 / abstain_correct / noise_rate / must_not_ok /
prefetch_tokens`, with a **direction-aware regression gate**:

```python
GATED_METRICS = frozenset({"recall@5","ndcg@5","abstain_correct","noise_rate","must_not_ok"})
...
    "regressed": worsened and gated,      # CI fails only on the gated set
```

Their noise generator uses word banks **deliberately disjoint from probe
keywords**, so "a noise hit" is always a ranking failure and never a
shared-token coincidence. That is a genuinely careful detail.

This was our **single biggest evidence gap**: for a vector store, recall quality
*is* the product, and we had no measurement of it at all.

**STATUS: CLOSED 2026-10-02 (`fc18af9`).** `scripts/retrieval_eval.py` ships
and runs: a 36-memory corpus with one paraphrased recall query per memory,
seeded and queried through the real tool path, reporting
recall@1/5/10 + MRR + nDCG@5 + latency. First run: recall@1 0.944,
recall@5 1.000, MRR 0.968, nDCG@5 0.976, mean 22.9 ms — reproduced on a second
run. `--min-recall5` gives the direction-aware floor this section asked for
(exit 1 below threshold).

Two deliberate departures from their harness, recorded so the gap is not
reopened by assumption:
- **No injected-noise experiment.** Their `noise.py` seeds adversarial points
  to measure `noise_rate`/`must_not_ok`; ours has no injection screen, so those
  metrics would measure a feature we do not have. Their word banks being
  disjoint from probe keywords is still the right idea if that is ever built.
- **No distractor seeding in the first version.** Every corpus item is its own
  target, so the harness measures *ranking*, not *discrimination under
  near-miss competition*. Adding same-topic distractors is the obvious next
  step and would make recall@1 far less flattering — worth doing before any
  claim about robustness, not after.

### 3. An AST import-consistency test

`tests/test_plugin_imports.py` exists because a tool was broken for a release by
importing a name a module never exported, caught only at call time. Our
`tool_schemas` / `handle_tool_call` dispatch string-matches names — the same
class of bug is live in our `_tool_*` dispatch. Effort S.

### 4. A CI budget test

`tests/evals/test_ci_budget.py` asserts the fast suite **never imports the ML
stack** and finishes in ≤60s. This directly protects our 287 MB-vs-2.2 GB
fastembed advantage from silently regressing. Effort S.

### 5. Atomic config writes

`__init__.py:658-659` (theirs, `:525-529`). They write `config.json` via
temp-file + `tmp.replace(config_path)`, so a crash mid-write cannot lose every
setting. Ours uses a plain `write_text`, which is non-atomic. Effort S.

### 6. A resolver that verifies a sentinel file, not a directory

```python
def resolve_scripts_dir(hermes_home: Path) -> Optional[Path]:
    candidates = [ ... ]
    for c in candidates:
        if (c / "memory_engine.py").is_file():   # sentinel, not "dir exists"
            return c.resolve()
    return None
```

This is the same discipline that caught our own `~/.cache` vs
`tempfile.gettempdir()` model bug. Apply to any future bundled asset. Effort S.

### 7. A fake-host contract harness

`tests/harness/fake_host.py` (752 lines) replays the real `MemoryManager` call
sequence and asserts hook order, threading, and signature filtering. Per our own
skill: a gate that constructs the object itself does not test the contract. We
have the kwarg-containment half but not the ordering/threading half. Effort S–M.

## Where they are RIGHT about disclosure and docs

- **`Disclosure —` is a real site convention, not an alarm.** `website/src/components/PluginCatalog/catalog.ts:165` defines `splitDisclosure()` ("catalog convention for behaviour a user opts into"); `PluginPage.tsx:158-160` renders it as a 72ch amber-bordered `<aside>`. The site also uses the *prose only* for `metaDescription` (`.slice(0, 160)`), deliberately excluding the disclosure from SEO/OG copy. 128/349 entries use it, including our closest peer `cognee.yaml`. Adopt the shape — we have two genuine opt-ins (the HF weight download; pointing the plugin at an endpoint you choose).
- **A dedicated `## Network Access` table** listing every egress with when/where. Theirs was written *because of* the catalog review (their CHANGELOG: "every network touchpoint is listed in one place … Matches the Hermes catalog disclosure as amended in the 2.8.0 review"). We have exactly one egress and it is currently mentioned in three places and findable in none. This is the change most likely to be *asked for* at review.
- **`## Known Limitations (version)`** with measured numbers, closing with "pinned by strict xfail tests". Our `## Not implemented` reads as a to-do list rather than a disclosure.
- **Atomic `save_config` that merges rather than clobbers** sibling keys.

## Where they are WRONG — do NOT copy

### 1. A false green on availability (their worst defect)

```python
def is_available(self) -> bool:
    hh = self._hermes_home or hermes_home_from_kwargs({})
    scripts = resolve_scripts_dir(hh)
    return scripts is not None          # <- checks FILES ON DISK, not the store
```

There is **no `check_backend()` and no `unavailable_reason()` anywhere in their
tree** (grepped repo-wide: zero hits). A dead or locked SQLite DB therefore
still reports available. And `initialize()` swallows the failure:

```python
    if not self._scripts_dir:
        logger.warning("EntropicMem skill scripts not found — run /learn EntropicMem")
        return          # <- "succeeds" with no backend attached
```

Worse, `prefetch()` collapses to `""` at `logger.debug` (invisible at default
level):

```python
        except Exception as e:
            logger.debug("EntropicMem prefetch failed: %s", e)
            return ""
```

This is precisely the false-green / silent-degradation class our own skill
documents. Their one mitigation: nothing negative is cached, so recovery is
automatic. **Ours is strictly better** — `check_backend()` does a real
`get_collections()`, `initialize()` clears `_backend_error` first so a cached
negative can never block recovery, and `unavailable_reason()` carries embedder
errors that `is_available()` structurally cannot see. Effort 0. Hold the line.

### 2. Fail-open injection screening

`injection_screen.py:525-537`: *"Any internal error is swallowed (fail-open) …
a broken screen can never take retrieval down"* → returns `flagged=False`.
`__init__.py:223-244` states it plainly: *"`screen_text` is fail-open by design —
on any screen failure the text ships unmarked."*

**A security gate that silently disables itself on exception is worse than no
gate**, because the operator believes it is on. If we ever add injection
screening, fail **closed** or fail to a hard "unscreened" marker, never to
"clean". Do not copy the fail-open form.

### 3. Unguarded network egress on a path declared off-by-default

`retrieval.py:158-160` constructs `SentenceTransformer("all-MiniLM-L6-v2")`
**bypassing the plugin's own `embeddings_enabled()` opt-in gate**, plus an
unguarded `sklearn` import we do not have. Constructing a SentenceTransformer
downloads weights from Hugging Face — so a path the plugin advertises as
"no network egress unless enabled" performs network egress anyway. Do not copy.

### 4. Writes failing invisibly

`memory_engine.py` itself is disciplined (0 `logger.debug`, only 3 bare
`except Exception`), but the *plugin* wraps every hook at debug level —
`on_session_end`, `_flush_session_digest`, `on_memory_write`, `prefetch` — so a
failed write produces no visible signal. Our contract is the opposite: a failed
write must never look like a successful save, and tool errors carry the cause.

### 5. Bare-name imports via `sys.path` mutation

`ensure_scripts_on_path` → `from memory_engine import MemoryEngine`. The plugin's
import graph depends on global state, which is what forced their import-AST
test. Our relative-import-plus-importlib-fallback is cleaner. Do not copy.

### 6. Config and secret handling

They read `config.yaml` directly with PyYAML and hand-merge `plugins.*` over
`memory.*`, and take secrets from env vars (`ENTROPICMEM_VAULT_PATH` etc.). The
tree rule says secrets come from `agent.secret_scope`. **We are compliant; do not
regress to env vars.**

### 7. Deliberately not adopting

`on_memory_write` mirroring (we *are* the store — there is no second store to
mirror into), their Fernet-at-rest `security.py` (Qdrant owns encryption;
duplicate key management is a liability), `parity_audit.py` multi-store sync
(we have one store), and the entire vault/graph/CLI surface.

## One finding I checked and REJECTED

A subagent reported that our `on_session_switch` does not rebind `_session_id`,
causing wrong-session recall. **This is not a bug.** There is no `self._session_id`
attribute at all — our provider is stateless per call:

- `sync_turn(..., session_id: str = "")` takes it as a keyword
- the orchestrator passes it on every turn (`agent/memory_manager.py:549`:
  `kwargs = {"session_id": session_id}`)
- `qdrant_recall` carries `session_id` as a **required** schema parameter

There is no state that could go stale. The claim was inferred from a debug log
line rather than from the data flow. Recorded here because the reasoning error
— inferring a bug from a log statement instead of tracing the call path — is
the kind of thing that gets "fixed" by adding state that was never needed.

## Our own open gaps this analysis exposed

Not from their code, but surfaced by comparing against it:

- **Three no-op hook stubs.** We implement `on_session_switch`, `on_session_end`
  and `on_pre_compress` as debug-log/pass bodies while `plugin.yaml` declares
  `provides_hooks: []`. Under catalog rule 6 a declared-vs-actual mismatch is a
  security issue in both directions. Either make them do something and declare
  them, or delete the stubs. **Submission blocker.**
- **Bare `threading.Thread` in `queue_prefetch`.** `plugins/AGENTS.md:79-83`
  mandates `agent.memory_provider.spawn_context_thread` for background work;
  theirs uses the correct wrapper with a fallback. **Tree rule, not a catalog
  rule, but a reviewer reads AGENTS.md.**
- **No `agent_context` write gate.** We write durable memory from subagent and
  cron turns. Theirs gates on `agent_context in (primary|subagent|cron|flush)`.
  Their value is questionable (it *blocks* subagent writes); the real lesson is
  to be deliberate about it. Effort S.
- **The HF model download is an implicit side effect of the first turn.**
  `qdrant_prepare(download=True)` exists, but the first `sync_turn` will trigger
  a download anyway. Our description says weights come from HF "on first use",
  which is honest, but we should make *when* explicit. The description string is
  a review surface; honesty here is cheap. Effort S.
- **No `Network Access` section** and no `Security` section in the README. Our
  SECURITY.md is strong, but the catalog page renders only README.md.

## Verification note

The CLI-related findings (`--provider` vs positional, no `register_cli` in our
tree, no `post_setup`) were confirmed by hand against
`hermes_cli/subcommands/memory.py:20-33` and by grepping our `__init__.py`,
rather than taken on the subagent's word. EntropicMem's cited line numbers were
read directly from a clone at the pinned commit. The session-id claim was
traced through `agent/memory_manager.py` and rejected on the evidence.
