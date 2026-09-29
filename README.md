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
- **5 agent tools**: `qdrant_search`, `qdrant_upsert`, `qdrant_recall`,
  `qdrant_collect` (read-only), `qdrant_prepare` (reports the model's
  dimensions and cache location before you commit to a collection).
- **Progress display** — optional status events during memory operations.
  When enabled, the CLI/TUI shows `💾 qdrant — stored (127,778 points)` after
  each turn and `💾 qdrant — recalled 3 memories` after each retrieval.
  Controlled by the `progress` config key: `off`, `minimal` (default),
  `verbose` (start + completion events).

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
  | `fastembed` | `>=0.4.0` | default embedder — ONNX, no torch, ~230 MB RSS |
  | `sentence-transformers` | `>=2.7.0,<7` | opt-in GPU backend — **heavy**, see below |

`torch` comes in transitively and is intentionally not pinned here.

> **Size warning.** `sentence-transformers` is the heavy dependency: it pulls
> `torch` plus the CUDA wheels — measured at 1.2 GB of `torch`, 3.2 GB of
> `nvidia-*` and 895 MB of `triton` in a CUDA build, ~5.3 GB total. The
> default `fastembed` backend needs none of it. The wide `>=2.7.0,<7` span is
> deliberate — a tight pin would freeze users onto an old dependency and miss
> upstream security fixes. If you only intend to use a remote Qdrant endpoint,
> you can skip the local embedder entirely; see the `embedder` config key above.

> **If you uninstall torch, it can come back silently.** Removing `torch`
> (or the `nvidia-*` / `triton` wheels) is safe while the default `fastembed`
> backend is in use — nothing in the runtime imports torch. But if any other
> package later pulls in a torch dependency, pip/uv will reinstall the full
> CUDA build and give back all ~5.3 GB without saying why. Selecting
> `embedder: sentence-transformers` after such a removal fails loudly with a
> `ModuleNotFoundError` naming torch; the fix is to reinstall it
> (`uv pip install torch`, or `torch --index-url .../cu124` for the CUDA build)
> rather than to debug the plugin.

> **Keep torch in your development/test venv — do not remove it.** A runtime
> install is well served by dropping torch entirely (the default `fastembed`
> backend never imports it, and neither does the plugin at import time). A
> *development* venv is not: the test suite exercises the `sentence-transformers`
> backend on purpose, both to prove vector parity with the stored collection and
> to prove that selecting it fails loudly when torch is absent. Without torch in
> the dev venv those tests either skip or pass vacuously, and the guarantee
> that a backend swap needs no re-embedding loses its only guard.
>
> Practically: keep torch in whichever venv runs `pytest` (here the 3.11 dev
> venv, with a CPU-only build), and omit it from the venv the gateway serves
> from. The plugin's own imports are backend-agnostic — neither `transformers`
> nor `torch` is loaded by `import`ing the plugin or by taking the `fastembed`
> path, so a torch-free runtime produces no warnings on its own.

## Setup

Start a server first — the provider is REST-only and always needs a reachable
Qdrant (there is no embedded mode):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then install and configure in one step:

```bash
hermes plugins install qdrant
hermes memory setup            # pick "qdrant" in the picker
```

The picker walks the config fields, probes the server, and sets
`memory.provider: qdrant`. To skip the picker — the provider name is a
**positional** argument, there is no `--provider` flag:

```bash
hermes memory setup qdrant
```

That path activates the provider and installs its dependencies, but it does
**not** prompt for settings. To configure by hand, create `config.json` next to
the plugin's `__init__.py` (it is `.gitignore`d and chmod 0600, because it can
hold an API key):

```json
{
  "url": "http://localhost:6333",
  "collection": "hermes_memories",
  "vector_size": 384,
  "distance": "Cosine"
}
```

Verify with `hermes memory status`. `QDRANT_URL` and `QDRANT_API_KEY` are
optional environment overrides — a local server on the default URL needs
neither.

> **There is no `hermes qdrant` CLI.** This plugin registers no CLI commands;
> everything is reached through the five agent tools and `hermes memory status`.

## Configuration

| key | type | default | meaning |
|---|---|---|---|
| `url` | string | `http://localhost:6333` | Qdrant server (REST) |
| `api_key` | secret | `""` | Qdrant Cloud only |
| `collection` | string | `hermes_memories` | collection name |
| `vector_size` | integer | `384` | must match your embedder |
| `distance` | select | `Cosine` | `Cosine`, `Dot`, `Euclid` |
| `embedder` | select | `fastembed` | `fastembed` (default, no torch) or `sentence-transformers` (GPU, needs torch) |
| `model` | string | `sentence-transformers/all-MiniLM-L6-v2` | must match the backend's catalog; changing it changes vector space |
| `device` | select | `cpu` | `auto`, `cpu` — `cuda` requires the `sentence-transformers` backend |

Resolution order, lowest to highest: built-in defaults → `config.yaml`'s
`memory.qdrant` → `config.json` → `QDRANT_URL` / `QDRANT_API_KEY` from the
environment. Secrets are read through Hermes' scoped-secret path and are never
written into `config.yaml`.

## Embeddings

Local, at **384 dimensions**, via one of two backends:

| backend | requires | when to pick it |
|---|---|---|
| `fastembed` (default) | `fastembed` only — ONNX, no torch | CPU hosts; ~230 MB RSS, no torch install |
| `sentence-transformers` | `torch` + CUDA wheels (~5.3 GB) | GPU hosts needing CUDA, or models outside fastembed's catalog |

Both are pinned to `sentence-transformers/all-MiniLM-L6-v2` by default, and
they produce **identical vectors** for that model (measured cosine 1.0000),
so switching backends does not require re-embedding an existing collection.

The first call downloads the model (~90 MB via the Hugging Face Hub for the
`sentence-transformers` backend; fastembed caches an ONNX export under
`$TMPDIR/fastembed_cache`) — that cold-start download is the warning you will
see. Nothing leaves your machine.

Choosing a different `model` **changes the vector space** and invalidates
every existing point; re-embed or start a new collection. `qdrant_prepare`
reports the model's dimensions and cache location before you commit to that.

If you change `vector_size` away from 384 you must also supply a matching
embedding model; the mismatch surfaces as a write error, not a config error.

## Not implemented

Documented here so nobody has to read the source to find out:

- **Lifecycle hooks** — none. The provider takes `session_id` on every call
  (`sync_turn(..., session_id=...)`, and `session_id` is a *required* parameter
  of `qdrant_recall`), so there is no session state to rebind and nothing to
  flush at a session boundary. `on_pre_compress` would be a real addition — a
  digest point written before the transcript is discarded — and is not built.
- **Hybrid dense+sparse RRF search** and **INT8 scalar quantization** exist in
  `_backend.py` but are *not wired into the provider* — the provider talks to
  `QdrantClient` directly. Unreachable today.
- **gRPC** transport is not supported; the client is REST.
- **No collection pruning.** No TTL, no dedup, no pruning job — a long-lived
  collection grows unbounded.
- **No LLM extraction.** Memories are verbatim turns, not summaries.

## Troubleshooting

**`unavailable_reason()` mentions a missing dependency** — install the
packages above in the same interpreter Hermes runs from. The default backend
needs only `fastembed`; a `ModuleNotFoundError` for `torch` means you selected
`sentence-transformers` on a host where torch was removed (see the size note).

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
