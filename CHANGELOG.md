# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- **`qdrant_forget` — point-targeted memory deletion.** The lifecycle gap: the
  provider could store and recall but never remove, and `qdrant_collect`
  deliberately has no destructive action. Takes exact point IDs, validates
  them (Qdrant accepts an unsigned integer or a UUID and nothing else),
  retrieves what exists, and reports the memory *text* alongside each ID.
  **Dry run by default** — without `confirm: true` nothing is deleted and the
  response says what would go. Capped at `MAX_FORGET_BATCH` (100) per call so
  it cannot become a bulk delete wearing a point-ID hat. No delete-all, no
  delete-by-filter, no delete-by-query: wiping a memory store stays a human
  decision.
- `qdrant_search` and `qdrant_recall` now print the point ID with each hit.
  Without this the agent has no way to name a memory it wants to forget, and
  "delete the whole collection" is not an acceptable substitute. Output
  format changes from `[0.72] text` to `[0.72] (id) text`.

### Changed

- Tool count 5 → 6, propagated to `plugin.yaml` `provides_tools`, README and
  CONTRIBUTING. The docs-honesty gate caught all four stale claims itself.

### Fixed

- The docs-honesty gate no longer flags a tool-count claim inside a *released*
  CHANGELOG section. Those entries are a historical record and were true when
  written; the check is now scoped to the open section above the first
  released heading, so a future release cannot be pushed into rewriting its own
  history.

## [0.1.5] — 2026-10-01

### Security

- **`api_key` is never written to disk by the plugin.** `save_config()` drops it
  from `values` before writing, and scrubs a key an older version already put
  in the file; the standalone `_setup.py` no longer passes it either, and prints
  the `.env` line to add instead. The key reaches the provider only through
  `QDRANT_API_KEY` via `get_secret()`. Previously it landed in
  `<HERMES_HOME>/qdrant.json` in plaintext — a file the dashboard reads and
  `hermes backup` copies. Found by teknium1 in the catalog review.
- `qdrant.json` is now created with mode `0600` at `os.open` time rather than
  chmod-ed afterwards, so the file is never briefly world-readable between
  creation and chmod (and the mode holds on platforms without POSIX modes,
  where the old `except OSError: pass` left it at the umask default).
- `system_prompt_block()` no longer puts URL credentials in front of the model:
  `https://user:pass@host` is rendered as `https://host`. The system prompt is
  model-visible and gets echoed into transcripts, logs and bug reports. The
  connection itself is unchanged — this only shapes the printed string.

### Fixed

- The model-supplied `limit` on `qdrant_search` and `qdrant_recall` is now
  clamped into `1..100` / `1..1000`. An uncapped `limit=100000` scrolled a whole
  collection into memory and returned one enormous payload. A non-integer
  argument now falls back to the default instead of raising a traceback at the
  model, and `limit=0` returns one result rather than a silent empty set that
  reads as "nothing was remembered".
- `_setup.py`'s install hint quoted `qdrant-client>=1.14.0` and
  `sentence-transformers`, contradicting the actual pyproject floors and the
  fastembed default. It now quotes the real bounds and the real default backend.
- `_setup.py` told the user to run `hermes memory provider qdrant`, which is
  not a subcommand (`memory_sub` has exactly `setup`/`status`/`off`/`reset`).
  Now `hermes memory setup qdrant`.

### Changed

- The docs-honesty gate now also scans **source** for user-facing claims, not
  only prose files: a `hermes memory …` subcommand that does not exist, and a
  dependency floor quoted in code or README that contradicts `pyproject.toml`.
  Both new checks are mutation-verified — planting the exact defects makes the
  gate exit 1. The old gate passed `_setup.py` because it only read `.md` files,
  which is why both of the above survived until review.

## [0.1.4] — 2026-10-01

### Fixed

- **The `api_key` config field is now actually masked.** It declared
  `type: "secret"` but not `secret: True`, and Hermes keys on the `secret`
  flag alone (`hermes_cli/memory_setup.py` `_prompt_schema_fields`,
  `hermes_cli/web_server_memory.py` `_schema_field_kind`). Without it
  `hermes memory setup` prompted in plain text and handed the key to
  `save_config()`, so it landed in `qdrant.json` — or into config.yaml when a
  `memory.qdrant:` block already existed — and the dashboard's config GET
  returned it in plaintext. The wizard now masks the field and the key is
  written only to `QDRANT_API_KEY` in `.env`, which `_load_plugin_config()`
  already read via `get_secret`, so runtime behaviour is unchanged. Found by
  teknium1 in the catalog review of NousResearch/hermes-agent#127847; fixed in
  PR #1 of this repo. `test_api_key_is_flagged_secret` is the regression guard
  (it fails if the flag is dropped or the dashboard stops classifying the field
  as `secret`).
- Not in this release: `_setup.py` (the standalone script) still writes
  `api_key` into `qdrant.json`. That file is now `0600`, but the script should
  tell users to export `QDRANT_API_KEY` instead.

## [0.1.3] — 2026-10-01

### Changed

- **Runtime state moved out of the plugin directory** — `<HERMES_HOME>/qdrant.json`
  (was `config.json`, now written `0600` atomically) and
  `<HERMES_HOME>/qdrant-status.json` (was `status.json`). `pm.workspace.
  members_stamp()` hashes every file in a member dir — `_MEMBER_EXCLUDE` covers
  `.git`/`.venv`/`venv`/`node_modules`/`__pycache__` only — and folds that hash
  into the venv dependency stamp, so a `status.json` write on every
  store/recall changed the stamp continuously and the environment re-synced on
  nearly every launch (48 sync receipts on this box, 3 of them "already in
  sync"). The core-side stamp logic is the root cause: a gitignored file is not
  a build input. The plugin no longer gives it anything to hash, and lands
  where sibling providers keep theirs (`mem0.json`, `honcho.json`,
  `supermemory.json`). Side benefit: a read-only pip/system install no longer
  fails on the first `save_config()`. `save_config()` honours the
  `hermes_home` it is handed; `_state_home()` falls back to the context-local
  override, then `HERMES_HOME`. Existing installs migrate by moving the two
  files once; `TestStateStaysOutOfTheMemberDir` guards it (5 tests, red against
  the pre-move code).
- Residual, upstream: `.pytest_cache/` inside an installed copy is still
  hashed by `members_stamp()` — running the suite in the install dir moves the
  stamp. Only a core fix (exclude gitignored state) closes that; this plugin
  cannot.
- **`fastembed` is now bounded on both ends** (`>=0.4.0,<1`), closing the last
  bare dependency floor. The lower bound is measured, not assumed: the plugin
  touches only `TextEmbedding(model_name=…, cache_dir=…)` and `.embed()`, both
  of which work on 0.4.0 — installed and executed at that exact version, not
  read off a changelog. The upper bound is the next major. Under PM's unified
  resolve, an unbounded floor lets a future release refuse the whole install
  rather than fail one plugin.

### Fixed

- The `api_key` config field now sets `secret: True`, which Hermes checks to
  treat a field as a secret (`type: "secret"` alone is ignored). Without it,
  `hermes memory setup` prompted for the key in plain text and saved it to
  `qdrant.json` (or into config.yaml when a `memory.qdrant:` block existed),
  and the dashboard's config GET returned it. The key now goes only to
  `QDRANT_API_KEY` in `.env`.
- `Embedder.dimension()` measured nothing: it read `fastembed`'s `.dim`, which
  **has never existed** — verified by installing 0.4.0, 0.5.0, 0.6.0, 0.7.0,
  0.8.0 and 0.8.1 and inspecting a live `TextEmbedding`. Every call raised
  `AttributeError` into a bare `except`, so the method returned the static
  table while its docstring claimed a measurement. Consequence: a model
  outside `KNOWN_MODEL_DIMS` reported 0 dims, and `validate_vector_spec()`
  skipped the vector-size cross-check without saying so. It now reads
  `embedding_size` (an int on 0.8.x) and only falls back to the table when the
  attribute is missing or non-positive. `TestMeasuredDimension` (5 tests) pins
  the measured-wins, table-fallback, zero-is-honest and bogus-value paths.
- The test suite can no longer write to the production store: live tests need
  an explicit `QDRANT_URL` on a non-default port (6333 REST **and** 6334 gRPC
  are refused — the production process owns both), and every collection a test
  creates is dropped in teardown. This is the leak that had left 4 stray
  collections next to the 128k-point store.
- Version drift: `pyproject.toml` / `PLUGIN_VERSION` said `0.1.0` while
  `plugin.yaml` said `0.1.2` — aligned to `0.1.2`.
- Two config tests ignored the documented env-beats-file precedence and only
  passed while `QDRANT_URL` happened to be unset.


## [0.1.2] — 2026-09-30

### Fixed

- **Model cache is pinned to one durable directory.** The ~90 MB fastembed
  weights defaulted to `$TMPDIR/fastembed_cache`. Hermes points `TMPDIR` at its
  scratch tree and reaps scratch entries idle for 24 h, so the model was
  re-downloaded on every prune — once per distinct `TMPDIR`, leaving three
  simultaneous copies (base scratch, a profile scratch, `~/.hermes/tmp`) on
  2026-09-29. `Embedder._build()` now passes `cache_dir` explicitly and
  `pinned_cache_dir()` resolves the same path (profile home and base home agree),
  defaulting to `<hermes home>/state/qdrant/model_cache`, which is not pruned.
  `FASTEMBED_CACHE_PATH` still overrides it.
- `model_is_present()` now answers "will this embed without a download" — it
  checks the directory the backend actually loads from instead of any cache on
  disk, so a copy stranded in scratch can no longer be reported as present
  while the load path is empty. Stale copies stay visible through
  `model_cache_locations()`, and `qdrant_prepare` names them instead of
  silently downloading a second copy beside them.
- The test session pins `FASTEMBED_CACHE_PATH` for the whole run. The
  per-test `HERMES_HOME` isolation was giving each embedding test an empty
  pin, so every test downloaded its own copy: four suite runs left 54 copies
  / 1.8 GB under `$TMPDIR/pytest-of-gk`. The suite now runs in 54 s instead
  of 167 s and writes no model weights of its own.
- The default-progress test no longer reads the install's own `config.json`.
  `QdrantMemoryProvider.__init__` falls through `config or _load_plugin_config()`,
  so the previously suggested `config={}` opt-out still loaded the live file: on
  a configured install (`{"progress": "verbose"}`, written by
  `hermes memory setup`) `test_progress_mode_defaults_to_minimal` failed while
  the same suite passed in a fresh checkout — a red suite that had nothing to
  do with the code under test. The shared provider fixture now stubs
  `_load_plugin_config()`, and `TestQdrantConfigLoading` covers the inverse
  contract: a `progress` mode written to `config.json` must reach the provider.
- `qdrant_recall` no longer prints a fabricated `[0.00]` score for every row:
  a scroll has no query, so a score bracket is shown only when the payload
  actually carries one.
- README said `~230 MB RSS` while `plugin.yaml` says `~287 MB peak RSS` for
  the same measurement — unified to `~287 MB peak RSS`.

### Added

- **`get_status_config()`** — `hermes memory status` now prints our config
  block (url, collection, embedder, model, vector_size, distance, progress,
  api_key set/unset) plus **last store/recall timestamps and counts** read
  from a new `status.json`, so a fresh process can answer "is this working,
  and when did it last store/recall?" without a live connection. This is the
  answer to the silent-failure class a peer project's issue history shows
  (weeks of no writes while every health surface said healthy).
- **Prefetch dedup** — near-duplicate hits are dropped before injection using
  word-level Jaccard (>=0.72) / containment (>=0.86) thresholds, adopted from
  Mnemosyne's `_semantic_dedup_prefetch` after comparing implementations.
  A prefetch no longer injects three near-copies of one turn.
- **Unknown config keys warn instead of being silently ignored** (mnemosyne
  issue #482 class): a typo'd `memory.qdrant` key now logs a warning naming
  the unknown key and the known set, then is dropped.
- **Declaration-parity tests** — manifest `provides_tools` <-> `tool_schemas`
  <-> `handle_tool_call` dispatch branches are asserted mechanically in both
  directions (catalog rule 6; the peer project's own parity test documents
  six drifted tool counts as the failure this prevents).
- Tester-report issue template now also captures the Hermes version, a
  recall round-trip (store→recall paste; an empty result is the most
  valuable report), and the redacted `memory.qdrant` config block.

### Documentation

- New "Back up, restore, move" README section: snapshots (vectors included),
  the docker volume, and the honest note that `hermes backup` carries config
  only for this provider.

## [0.1.1] — 2026-09-29

### Added

- **Progress display** — optional status events during memory operations.
  `sync_turn` and `prefetch` now emit `status_callback` events that the
  CLI/TUI renders as `💾 qdrant — stored (127,778 points)` and
  `💾 qdrant — recalled 3 memories`. Controlled by the `progress` config key:
  `off`, `minimal` (default, completion events only), `verbose` (start +
  completion events). The callback is stored from `initialize()` kwargs and
  never blocks memory operations.
- `recall_status()` now returns the actual recall count from the last
  `prefetch` call, so the recall indicator shows a real number.

## [0.1.0] — 2026-09-26

Initial standalone release of the Qdrant memory provider for Hermes Agent.

### Added

- `QdrantMemoryProvider`, a `kind: exclusive` memory provider registered through
  the Hermes memory-provider loader.
- Dense semantic search over a single named `dense` vector, with session-scoped
  recall through a `session_id` payload filter.
- Verbatim turn storage via `sync_turn` with payload metadata.
- A circuit breaker (5 failures → 120 s cooldown) around the I/O paths.
- Honest availability reporting: `is_available()` (config-only, no network),
  `check_backend()` (real probe) and `unavailable_reason()` (user-facing
  diagnosis) distinguish a misconfiguration from a dead server.
- Five agent tools: `qdrant_search`, `qdrant_upsert`, `qdrant_recall`,
  `qdrant_collect`, `qdrant_prepare`.
- `hermes memory setup` schema via `get_config_schema()` / `save_config()`.
  There is no `hermes qdrant` CLI subcommand; the provider registers no CLI.
- Configuration precedence: `QDRANT_URL` / `QDRANT_API_KEY` env (secrets) →
  `memory.qdrant:` in `config.yaml` → `config.json` next to the module.

### Known limitations

- Hybrid dense+sparse RRF search and INT8 scalar quantization exist in
  `_backend.py` but are **not** on the live path; they are not advertised.
- `sentence-transformers` pulls `torch` (~600 MB RSS). Weights download from
  Hugging Face on first use; embedding happens on-device.
- Local embedded Qdrant mode is not supported — point the provider at a running
  Qdrant server (REST).
- `grpcio` has no `win_arm64` wheel, so Windows on ARM is unsupported.
