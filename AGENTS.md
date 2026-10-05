# AGENTS.md — hermes-qdrant-memory

Intent layer for AI assistants and humans working on this repo. The mechanics live in
[CONTRIBUTING.md](CONTRIBUTING.md) (ground rules, layout, dev commands, PR checklist) — read it
before your first edit. [README.md](README.md) is the user-facing doc,
[CHANGELOG.md](CHANGELOG.md) the history, [SECURITY.md](SECURITY.md) the threat model and reporting
route. This file carries only the *decisions*: what this repo is, what it deliberately is not, and
the rules whose reversal costs a review round. Do not restate the other files here — a duplicated
fact is a fact that rots.

## What this repo is

A **standalone Hermes memory-provider plugin**. The repo root *is* the plugin directory; the Hermes
catalog installs it by pinning a **40-char commit SHA** (never a tag). It stores each Hermes turn
as one Qdrant point (`sync_turn`: `user:` + `assistant:` combined, payload `text` truncated to
2000 chars, plus `session_id`, ISO-8601 `timestamp`, `source`, `turn_author`) and recalls by dense
cosine search under a session-scoped payload filter.

## Invariants — breaking one of these is a review round

- **Secrets stay out of files and prose stays true.** `api_key` declares `secret: True` so the
  wizard masks it and routes it to `.env`; `save_config()` pops `api_key` from what it persists and
  scrubs a legacy one; the state file is created `0600` (at open, not chmod after). The key reaches
  the provider through `QDRANT_API_KEY` via `get_secret()`. Every claim in the README, catalog
  description and tags is only allowed while the code makes it true (teknium1's review rule,
  2026-10-01) — `scripts/check_docs_honesty.py` enforces the cheap claims and runs inside pytest
  (`TestDocsHonestyGateRuns`) and CI.
- **The one validator warning is expected. Do not "fix" it.** `hermes plugins validate .` reports
  `provides_tools declares … but register() did not register them`. That is correct for a memory
  provider: its tools reach the agent through the provider's schema accessor on the
  memory-activation path, not through `register(ctx)`, which the static probe is the only thing
  that can see. Deleting the declaration, or registering stubs from `register()` to silence it,
  makes the tools declared-and-absent. Likewise `register()`'s **duplicated** `md_search`
  registration (provider path + `ctx.register_tool`) is load-bearing: the manifest is
  `kind: standalone` precisely so the tool survives a non-qdrant `memory.provider`. "Deduplicating"
  it breaks the tool.
- **Declaration parity is mechanical, not aspirational.** `plugin.yaml` `provides_tools` ↔
  `tool_schemas.py` `ALL_TOOL_SCHEMAS` ↔ `handle_tool_call` dispatch are asserted in **both**
  directions (`TestDeclarationParity`). A new or renamed tool without all three is a failing test,
  not a TODO.
- **Never splat kwargs into a client constructor.** `MemoryManager.initialize_all` injects
  `session_id` plus scoping kwargs (`platform`, `hermes_home`, `agent_context`, `status_callback`,
  `warning_callback`, `session_title`, …). `QdrantClient(**kwargs)` dies on the first one, and the
  orchestrator *logs and swallows* provider init failures — so the provider silently serves no
  memory while every disk-level check stays green. Name every forwarded argument explicitly; pull
  connection params out with `kwargs.get(...)`. Guarded by
  `test_scoping_kwargs_never_reach_the_client`.
- **No runtime state in the plugin member dir.** `pm.workspace.members_stamp()` hashes every file
  in the plugin dir, so a `status.json` written next to `__file__` re-syncs the venv on *every*
  launch. State belongs in `$HERMES_HOME`-resolved paths (`_setup.py`'s state-home resolver);
  `test_state_paths_live_in_the_home_not_the_member_dir` guards it, and `sweep_legacy_in_dir_state()`
  clears pre-0.1.5 residue.
- **Embedding output is a contract.** `fastembed` (ONNX, CPU) is the default at 384 dims on
  `sentence-transformers/all-MiniLM-L6-v2`; the model name is a module-level constant, never a
  literal inside a function body. `sentence-transformers` is the **`gpu` extra**
  (`>=2.7.0,<7`), not a runtime dependency — `hermes plugins install` has no `--extra`, so the
  install path is "install, then approve the dependency" — and `torch` is never declared directly.
  Two parity checks guard this and both must stay green: `TestEmbedderParity` proves the two
  backends agree with each other, `TestStoredVectorParity` proves the *new* default reproduces a
  vector written by the old sentence-transformers path (cosine > 0.999) **and retrieves it by
  text**. The second is the one that protects real stores — a backend change that alters the vector
  space forces a full re-embed of every existing collection.
- **Recall stays session-scoped, and every stored point carries its scope key.** A new profile
  starts with empty recall; cross-session leakage through a missing or widened payload filter is a
  security bug, not a feature request. Write `session_id` on every store path (a full-collection
  census once found a five-figure share of a production store unreachable for exactly this reason),
  keep one timestamp representation per collection, and never hardcode an origin field the caller
  supplied.
- **Destructive actions stay human.** `qdrant_forget` takes exact point IDs only, batch-capped, dry
  run unless `confirm: true`; there is no delete-all or delete-by-filter, and dropping a user's
  corpus is a human decision. Tool dispatch answers with error **strings**, never exceptions.
- **The display ladder is `off | summary | verbose` — and it is not the `progress` knob.**
  `display.level` (nested key of `<HERMES_HOME>/qdrant.json`, default `off`, validated against
  `DISPLAY_LEVELS`, unreadable-or-unknown falls back to `off` with one warning per defect) decides
  what `recall_status()` puts in `provider_label`: `off` is 0.1.7 byte for byte; `summary` adds
  two facts, latency and collection; `verbose` adds one numeric log line per operation (counts,
  latencies, provenance — never contents). The metrics live in `provider_label`, never in `glyph`
  (a symbol field no peer overrides). The older top-level `progress` key (`off | minimal |
  verbose`) is a separate, untouched knob — name which one a sentence is about, because the two
  vocabularies overlap and that is exactly how a wrong word ships. Guarded by
  `tests/test_display_levels.py`.
- **Three version surfaces, one number.** `plugin.yaml` (quoted — a bare `0.1` is a YAML float),
  `pyproject.toml`, and `PLUGIN_VERSION`. They drift independently; change all three at a release.
- **No capability in core.** `plugins/memory/` is closed to new providers and third-party plugins
  are barred from in-tree absorption — that is why this ships standalone, and why a need for a new
  hook is an issue asking for the *generic* capability, never a Qdrant special case upstream. Never
  vendor Hermes core: `tests/conftest.py` holds minimal, honest contract stubs and that seam stays
  minimal.

## Testing

```bash
uv run pytest                      # unit suite
uv run ruff check .                # whole tree, same pin as CI
hermes plugins validate .          # passes, "security scan: safe" (+ the one expected warning)
python3 scripts/check_docs_honesty.py
```

**Do not point the suite at a live store.** `tests/conftest.py` *fails loudly* if `QDRANT_URL`
names Qdrant's default ports (6333 REST / 6334 gRPC — where production lives) and skips when no
scratch server is named. Run live tests against a non-default port:

```bash
docker run -p 16333:6333 qdrant/qdrant
QDRANT_URL=http://localhost:16333 uv run pytest
```

Judge a change by the suite and the validator, not by "it didn't crash". There are deliberately no
pinned check counts in this file: a count in prose is a claim that drifts, and the one that used to
live here was already wrong.

## Routing

| You are about to… | Read first |
|---|---|
| change deps or layout | `CONTRIBUTING.md` (Ground rules / Layout), `pyproject.toml` comments |
| add, rename or drop a tool | all three parity surfaces + `TestDeclarationParity` |
| change an embedder backend | `TestEmbedderParity` (backends agree) **and** `TestStoredVectorParity` (vs vectors already in a store) |
| write or change docs of record | `README.md`, then a `CHANGELOG.md` entry under `[Unreleased]` |
| report a security issue | `SECURITY.md` — private advisory, never a public issue |
| release / re-pin | `CONTRIBUTING.md`, "a release is a commit SHA" |
| propose a Hermes core change | open an issue for the generic capability — do not patch core |