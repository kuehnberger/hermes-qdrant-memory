# Qdrant memory provider for Hermes

**Local-first**, self-hostable vector memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
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

## Prefill from history you already have

A memory provider only knows what it saw, so a fresh install starts empty even
if your agent has months of conversations. One command fixes that — it reads
the profile's session database and writes every stored message into the
collection, so recall works from the first turn:

```bash
# see what it would do (writes nothing)
python scripts/backfill_memory.py --profile <name>

# do it
python scripts/backfill_memory.py --profile <name> --yes
```

Sessions already in the store are skipped, point ids are deterministic, and
the run is safe against a live profile (the database is opened read-only), so
re-running only fills genuine gaps. It is also the migration path between
memory providers, and the way to seed a store you are moving to a new
machine. Detail: **`docs/DEV.md`**.

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

## Optional: hybrid search (dense + BM25)

`"hybrid": "on"` in `qdrant.json` adds a 10 MB BM25 sparse vector next to the
semantic one and fuses both rankings (RRF). Off by default. A fresh collection
declares the sparse vector at creation; an existing dense-only store gets it
added in place and its old points backfilled with one command
(`python scripts/migrate_hybrid.py --collection hermes_memories --yes`).
Measured gain, honest limits, and a re-runnable harness: **`docs/HYBRID.md`**.

## Where the detail lives

This file is the entry point. Everything else is in:

- **`docs/DEV.md`** — architecture, embedder backends, the knowledge index, backups,
  the eval harness, and what is deliberately not implemented.
- **`docs/HYBRID.md`** — hybrid search: measured numbers, the collection rule, limits.
- **`CONTRIBUTING.md`** — ground rules, layout, and the PR checklist.
- **`AGENTS.md`** — the decisions whose reversal costs a review round (for contributors).
- **`CHANGELOG.md`** — what changed in each release.
- **`SECURITY.md`** — how to report a security issue.

## License

MIT