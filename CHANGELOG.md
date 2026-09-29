# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

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
