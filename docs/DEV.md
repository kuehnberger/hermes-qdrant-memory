## Back up, restore, move

Memories live in the Qdrant collection — the plugin keeps no local state besides
`<HERMES_HOME>/qdrant.json` (and the `qdrant-status.json` bookkeeping next to it), so moving or
backing up means moving the server's data:

- **Snapshot** (recommended): `POST /collections/<name>/snapshots` (the dashboard UI does the same)
  writes a snapshot file **including vectors** — restoring it needs no re-embedding. Measured:
  ~408 MB for a 127k-point collection.
- **Docker volume:** the default `qdrant/qdrant` image stores everything under `/qdrant/storage` —
  copy the volume, or `docker cp` it out, and the whole memory moves with it.
- **`hermes backup` does not carry your points.** This provider reports no `backup_paths()`, so
  `hermes backup` captures config only; back the server up separately (snapshot or volume).

To verify a restore: `qdrant_collect(action="info")` shows the point count, and a session recall
should return the same lines as before the move.

## Display levels

`display.level` is a **nested** key of the same `<HERMES_HOME>/qdrant.json` that holds
`md_docs_roots`:

```json
{ "display": { "level": "summary" } }
```

| level | what the user sees |
|---|---|
| `off` (default) | nothing new — the indicator is byte-identical to 0.1.7 |
| `summary` | the chat indicator gains latency, collection and the session hit-rate: 📖 qdrant · 41ms · hermes_memories · hits 3/5 — recalled 10 memories |
| `verbose` | the `summary` label **plus** one numeric line per recall, per store and per `md_search` in `agent.log` (the recall line carries `p50`/`p95` over the rolling latency window once two recalls have completed) |

The default is `off`, so an existing install — including a catalog install — behaves exactly as
before until someone writes that key. The verbose lines carry numbers and identifiers only
(latency, counts, score range, session id, collection, file labels), never recalled message text
or search-result snippets. When no timing exists yet — the first turn, or a recall that failed —
the `· …ms` segment is omitted rather than printed as a zero. The `hits N/M`
segment follows the same rule: it appears only once a recall has been attempted
in this process — N results, M attempts, where a *failed* attempt counts (that
turn got no usable recall) and a breaker-skipped turn does not (no recall was
attempted) — and never at `off`.

`hermes memory status` is the numbers surface: next to the effective level it
reports `recall_attempts` and `recall_hits` (process-lifetime counters — a
fresh process shows honest zeros), `breaker_failures` and `breaker_open`
(circuit-breaker state), and `recall_ms_p50` / `recall_ms_p95` over a rolling
window of completed recalls — the percentiles appear only once a recall
completed, omitted rather than faked while the window is empty.

The metrics live in `provider_label`, never in `glyph`: `glyph` is a symbol field no peer
overrides, and `provider_label` is documented free text that interpolates verbatim.

The file is read defensively, like `md_docs_roots`: absent means `off`, and a malformed file, a
wrong-typed `display` block or a value outside `off|summary|verbose` degrades to `off` with a
single warning line instead of an error. `hermes memory status` reports the effective level.

**This is not the `progress` knob.** The older top-level `progress` key (`off | minimal |
verbose`) is a separate, untouched control for status events. The two vocabularies overlap, and
that is exactly how a wrong word ships — name which one a sentence is about.

## Disclosures

Two things a user should know before installing, in the spirit of catalog rule 13:

**The embedding model is loaded in-process, once per process.** The plugin registers native
Hermes tools; it is not a server, so the model weights are `mmap`-ed into whatever process is
doing the embedding — each Hermes gateway, dashboard, CLI session and cron worker pays for its
own copy. Measured on this host with the default `fastembed` backend (MiniLM-L6-v2, 384 dims),
5 concurrent processes each loading the model and embedding 20 texts:

| | 1 process | 5 concurrent processes |
|---|---|---|
| RSS after first embed | 211 MB | 211–216 MB each |
| total resident | 211 MB | **1,071 MB** (≈194–197 MB marginal per extra process) |
| wall time (load + 20 embeds) | 0.9 s | 2.9 s |

This is the same shape as an MCP server configured over **stdio**, where the protocol gives you one
server process per client. It is not fixable by configuration on our side, and we deliberately
do not hide it behind a "lightweight" claim: the honest lever is how many Hermes processes you run
against this provider. Memory *data* is shared (one Qdrant collection); the *model* is not.

**The plugin starts no MCP server, no sidecar, and no background process.** It registers tools
in-process and talks to your Qdrant server over REST. If you build something that exposes this
plugin to other agents as an MCP server, prefer a **single shared HTTP server** over a per-client
stdio entry — a stdio `command:`/`args:` config spawns one process (plus its own ~200 MB model
copy) per client, which is the most common way this cost gets multiplied by accident.

## Evaluation

`scripts/retrieval_eval.py` measures retrieval quality instead of asserting it: 36 target memories
with one paraphrased recall query each, plus 36 same-topic near-miss distractors, all run through
the real tool path (`qdrant_upsert` / `qdrant_search`, same embedding, same formatting the model
sees), reported as recall@1/5/10 with a 95% Wilson interval, MRR, nDCG@5, a pairwise *beats its
own distractor* rate, and latency:

```bash
python scripts/retrieval_eval.py                      # report only
python scripts/retrieval_eval.py --min-recall5 0.9    # exit 1 below 0.90
```

It writes only to a scratch `hermes_memories_eval` collection — created with the production schema,
dropped when the run ends (`--keep` to inspect it) — and refuses to start if that name ever equals
the configured collection, so the production store cannot be written by this script. The metrics
themselves are offline-unit-tested in `tests/test_retrieval_eval.py`.

Current numbers on the shipped harness: recall@1 **0.889** (95% CI 0.75–0.96) against same-topic
distractors, median margin 0.17, 10 of 36 cases decided by less than 0.10. That is the cost of the
206 MB default, reported rather than hidden.

## Not implemented

Documented here so nobody has to read the source to find out:

- **Lifecycle hooks** — none. The provider takes `session_id` on every call
  (`sync_turn(..., session_id=...)`, and `session_id` is a *required* parameter of
  `qdrant_recall`), so there is no session state to rebind and nothing to flush at a session
  boundary. `on_pre_compress` would be a real addition — a digest point written before the
  transcript is discarded — and is not built.
- **Hybrid dense+sparse RRF search** and **INT8 scalar quantization** exist in `_backend.py` but
  are *not wired into the provider* — the provider talks to `QdrantClient` directly. Unreachable
  today.
- **gRPC** transport is not supported; the client is REST.
- **No collection pruning.** No TTL, no dedup, no pruning job — a long-lived collection grows
  unbounded. `qdrant_forget` removes individual memories by point ID but nothing reaps them
  automatically.
- **No bulk deletion.** `qdrant_forget` is point-targeted on purpose: there is no delete-all, no
  delete-by-filter, no delete-by-query. It is a dry run unless called with `confirm: true`, and it
  shows the memory text it is about to remove so the call can be checked before it is irreversible.
  Wiping a memory store stays a human decision, taken outside the agent.
- **No LLM extraction.** Memories are verbatim turns, not summaries.

## Honesty rules

- **Prose stays true.** Every claim in the README, catalog description and tags is only allowed
  while the code makes it true (teknium1's review rule, 2026-10-01). `scripts/check_docs_honesty.py`
  enforces the cheap claims and runs inside pytest (`TestDocsHonestyGateRuns`) and CI.
- **No pinned counts in prose.** A number in an intent file is a claim that drifts: the validator
  check count and the test count both move with the next commit and nothing re-measures the prose.
  Run the gates (`hermes plugins validate .`, `uv run pytest -q`) to see the current numbers
  instead of reading them here.
- **The one validator warning is expected.** `hermes plugins validate .` reports
  `provides_tools declares … but register() did not register them`. That is correct for a memory
  provider: its tools reach the agent through the provider's schema accessor on the
  memory-activation path, not through `register(ctx)`, which the static probe is the only thing that
  can see. Do not "fix" it by deleting the declaration or by re-registering stubs from
  `register()` — the tools would then be declared and absent.
- **The duplicated `md_search` registration is load-bearing.** `register()` registers `md_search`
  twice (provider path + `ctx.register_tool`), and the manifest is `kind: standalone` precisely so
  the tool survives a non-qdrant `memory.provider`. "Deduplicating" it breaks the tool.

## Development

```bash
uv venv && uv sync      # dependencies only — this is a directory plugin
uv run pytest -q        # live-server tests self-skip without a Qdrant on :6333
```

The repo root *is* the plugin directory, so `tests/conftest.py` imports the shipped files as
`plugins.memory.qdrant` and stubs the three Hermes core symbols the provider needs. Point
`HERMES_SOURCE` at a checkout of [hermes-agent](https://github.com/NousResearch/hermes-agent) to
run the same tests against the real core instead of the stubs.

Tests that touch a real Qdrant server need one on a non-default port
(`docker run -p 16333:6333 qdrant/qdrant`, then
`QDRANT_URL=http://localhost:16333 uv run pytest`). `tests/conftest.py` *fails loudly* if
`QDRANT_URL` names Qdrant's default ports (6333 REST / 6334 gRPC — where production lives) and
skips when no scratch server is named.

## Platform support

Linux and macOS on x86_64 and arm64. Windows x86_64 works; **Windows on ARM does not** —
`grpcio` (pulled in by `qdrant-client`) has no `win_arm64` wheel.