# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/).

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
