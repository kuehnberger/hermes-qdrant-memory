# Qdrant memory provider for Hermes

Self-hostable vector memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
backed by [Qdrant](https://qdrant.tech). No cloud dependency, no embedding API
key, no per-query cost.

Replaces Hermes' built-in memory with a Qdrant collection: turns are embedded
locally and stored as points, and recall is a dense vector search filtered by
session.

## What it does

- **Dense semantic search** over a named `dense` vector.
- **Session-scoped recall** through a `session_id` payload index.
- **Verbatim turn storage** — each turn is embedded as one point
  (`user: … assistant: …`). There is no LLM summarization step, so recall is
  cheap and quote-faithful.
- **Circuit breaker** — 5 consecutive I/O failures open a 120s cooldown. A
  Qdrant outage degrades to "no new memories", not a crash.
- **Honest availability** — `is_available()` / `check_backend()` /
  `unavailable_reason()` distinguish a bad config from a dead server, so
  `hermes memory status` can tell you which one you have.
- **4 agent tools**: `qdrant_search`, `qdrant_upsert`, `qdrant_recall`,
  `qdrant_collect` (read-only).

## Requirements

- Hermes `>=0.21.4`
- Python `>=3.11`
- A reachable Qdrant. Quickest local option:

  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

  Or [Qdrant Cloud](https://cloud.qdrant.io) (set `api_key`).

- Dependencies are managed by the repo's `pyproject.toml`:

  | package | constraint | why |
  |---|---|---|
  | `qdrant-client` | `>=1.10.0,<2` | 1.10.0 is the oldest floor with `query_points` / `Prefetch` / `FusionQuery` |
  | `sentence-transformers` | `>=2.7.0,<7` | local embeddings — **heavy**, see below |

`torch` comes in transitively and is intentionally not pinned here.

> **Size warning.** `sentence-transformers` is the heavy dependency: it pulls
> `torch`, which is roughly 600 MB of RSS and several hundred MB on disk. The
> wide `>=2.7.0,<7` span is deliberate — a tight pin would freeze users onto an
> old dependency and miss upstream security fixes. If you only intend to use a
> remote Qdrant endpoint, you can skip the local embedder entirely; see the
> `embedder` config key above.

## Setup

```bash
hermes memory setup --provider qdrant
hermes memory provider qdrant
```

The setup wizard writes the config and probes the server. To configure by hand,
create `config.json` next to the plugin's `__init__.py` (it is `.gitignore`d and
chmod 0600, because it can hold an API key):

```json
{
  "url": "http://localhost:6333",
  "collection": "hermes_memories",
  "vector_size": 384,
  "distance": "Cosine"
}
```

Or set `memory.provider: qdrant` in `config.yaml`. `QDRANT_URL` and
`QDRANT_API_KEY` are optional environment overrides — a local server on the
default URL needs neither.

## Configuration

| key | type | default | meaning |
|---|---|---|---|
| `url` | string | `http://localhost:6333` | Qdrant server (REST) |
| `api_key` | secret | `""` | Qdrant Cloud only |
| `collection` | string | `hermes_memories` | collection name |
| `vector_size` | integer | `384` | must match your embedder |
| `distance` | select | `Cosine` | `Cosine`, `Dot`, `Euclid` |
| `embedder` | select | `""` | `""` or `sentence-transformers` |

Resolution order, lowest to highest: built-in defaults → `config.yaml`'s
`memory.qdrant` → `config.json` → `QDRANT_URL` / `QDRANT_API_KEY` from the
environment. Secrets are read through Hermes' scoped-secret path and are never
written into `config.yaml`.

## Embeddings

Local, via `sentence-transformers` `all-MiniLM-L6-v2` at **384 dimensions**.
The first call downloads the model (~90 MB) — that is the Hugging Face Hub
warning you will see on a cold start. Nothing leaves your machine.

If you change `vector_size` away from 384 you must also supply a matching
embedding model; the mismatch surfaces as a write error, not a config error.

## Not implemented

Documented here so nobody has to read the source to find out:

- **Hybrid dense+sparse RRF search** and **INT8 scalar quantization** exist in
  `_backend.py` but are *not wired into the provider* — the provider talks to
  `QdrantClient` directly. Unreachable today.
- **gRPC** transport is not supported; the client is REST.
- **No collection pruning.** No TTL, no dedup, no pruning job — a long-lived
  collection grows unbounded.
- **No LLM extraction.** Memories are verbatim turns, not summaries.

## Troubleshooting

**`unavailable_reason()` mentions a missing dependency** — install the two
packages above in the same interpreter Hermes runs from.

**It says available but recall returns nothing** — the server is up but the
collection is empty or empty-filtered; check `qdrant_collect(action="info")`.

**Write errors after changing `vector_size`** — embedder dimension mismatch.
Leave it at 384 unless you have changed the embedding model too.

**First recall is slow** — that is the one-time model download plus load, not
the query path.

## Development

```bash
uv venv && uv sync      # dependencies only — this is a directory plugin
uv run pytest -q        # live-server tests self-skip without a Qdrant on :6333
```

The repo root *is* the plugin directory, so `tests/conftest.py` imports the
shipped files as `plugins.memory.qdrant` and stubs the three Hermes core symbols
the provider needs. Point `HERMES_SOURCE` at a checkout of
[hermes-agent](https://github.com/NousResearch/hermes-agent) to run the same
tests against the real core instead of the stubs.

## Platform support

Linux and macOS on x86_64 and arm64. Windows x86_64 works; **Windows on ARM
does not** — `grpcio` (pulled in by `qdrant-client`) has no `win_arm64` wheel.

## License

MIT
