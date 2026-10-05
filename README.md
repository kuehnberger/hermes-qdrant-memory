# Qdrant memory provider for Hermes

Self-hostable vector memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
backed by [Qdrant](https://qdrant.tech). No cloud dependency, no embedding API key, no
per-query cost. Your data stays on your machine.

## What it does

Your agent's memory is a Qdrant collection you run yourself. Each turn is embedded
locally and stored as one point; recall is a dense vector search filtered by session,
so a result quotes what was actually said — there is no summarizer quietly rewriting it.

- **Dense semantic search** over your conversation history.
- **Session-scoped recall** — a new profile starts with empty memory.
- **Seven tools**: `qdrant_search`, `qdrant_upsert`, `qdrant_recall`, `qdrant_collect`,
  `qdrant_prepare`, `qdrant_forget`, and `md_search` (search your local markdown).
- **Honest failure states** — a bad config and a dead server are told apart, and a
  circuit breaker turns an outage into "no new memories" instead of a crash.

## Install

```bash
docker run -p 6333:6333 qdrant/qdrant
hermes plugins install qdrant
hermes memory setup qdrant
```

Two runtime dependencies, **206 MB**, no torch, no GPU, no CUDA. The default install is
CPU-only ONNX embedding at 384 dimensions.

## What you get

| | |
|---|---|
| **No vendor lock-in** | REST-only — open your collection with `curl` or any HTTP tool. |
| **No monthly cost** | One model download, then everything is local. |
| **No summarization** | Memories are verbatim turns. You own the conversation. |
| **Auditable** | `qdrant_forget` is point-targeted and dry-run by default; there is no delete-all. |

## First memory

```bash
hermes memory setup qdrant
```

Then talk to your agent. Memory writes on its own; `hermes memory status` shows whether
the provider is connected and what it last stored.

## Configuration

Settings live in `<HERMES_HOME>/qdrant.json` (created `0600`, no credentials):

```json
{
  "url": "http://localhost:6333",
  "collection": "hermes_memories",
  "vector_size": 384,
  "distance": "Cosine"
}
```

The API key, when you need one (Qdrant Cloud only), goes in `.env` as `QDRANT_API_KEY` —
never in a config file.

## Optional: see the work happening

`display.level` in `qdrant.json` controls what the chat indicator shows:

| level | what you see |
|---|---|
| `off` (default) | nothing new — byte-identical to before |
| `summary` | `📖 qdrant · 41ms · hermes_memories — recalled 10 memories` |
| `verbose` | the above, plus one numeric line per operation in the log |

Numbers and identifiers only — never message text.

## Where the detail lives

This file is the entry point. Everything else is in:

- **`docs/DEV.md`** — architecture, embedder backends, the knowledge index, backups,
  the eval harness, and what is deliberately not implemented.
- **`CONTRIBUTING.md`** — ground rules, layout, and the PR checklist.
- **`AGENTS.md`** — the decisions whose reversal costs a review round (for contributors).
- **`CHANGELOG.md`** — what changed in each release.
- **`SECURITY.md`** — how to report a security issue.

## License

MIT