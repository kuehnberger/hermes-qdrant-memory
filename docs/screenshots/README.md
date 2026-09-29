# Screenshots

Placeholder. Up to 6 PNGs, 2:1 or wider, GitHub-hosted and pinned to the release
commit (the catalog pins a SHA, so a moving branch URL would not resolve).

Candidate shots:

1. `hermes memory status` — active provider and its availability diagnosis.
   This is the differentiated surface: it distinguishes a misconfigured URL from
   a dead server.
2. `hermes memory setup` — the config wizard with the declared knobs.
3. A `qdrant_search` tool call and its JSON result, with the `session_id`
   filter visible.
4. `qdrant_collect(action="info")` — point count, vector config, live
   collection.
5. **The circuit breaker firing** — stop the Qdrant container and show a write
   degrading to "no new memories" during the 120 s cooldown. No competitor's
   screenshot shows a failure mode, and this is the one that proves the
   degradation is designed rather than accidental.

## Constraint on capture

There is no `hermes qdrant` CLI — the provider registers none, so any earlier
draft naming `hermes qdrant status` was describing a command that does not
exist. Shots 1 and 2 come from `hermes memory ...`; shots 3-5 come from a live
agent session or a direct tool call.

Terminal captures must stay legible at card size: light background,
generous font, no scrollback trim, no personal data in the output.
