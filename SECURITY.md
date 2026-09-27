# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not open a public issue** for a security report. Use GitHub's private
advisory form: *Security → Report a vulnerability* on this repository. Expect an
acknowledgement within a few days; if you have not heard back in a week, nudge.

Please include: the provider version, your Hermes version, the Qdrant server
version, and a minimal reproduction. A `hermes logs` excerpt around the failure
is usually the most useful thing you can attach.

## Threat model

This plugin is a **local memory backend**. It stores whatever Hermes tells it to
store and reads it back.

- **The Qdrant server is trusted.** The provider talks plain HTTP REST to
  whatever `QDRANT_URL` points at. Anyone who can reach that server can read and
  write your memories; treat it as part of your trust boundary and do not expose
  it to an untrusted network.
- **No credentials are required.** Self-hosted Qdrant on `localhost` needs no
  API key. `QDRANT_API_KEY` is only needed for Qdrant Cloud and is read through
  the Hermes secret scope — it is never written to `config.json` and never
  logged.
- **`config.json` is per-user runtime state** and is git-ignored. If a hand-edit
  puts a credential in it, that is the user's file to protect.
- **Data never leaves your machine** except for the model weights: the
  `sentence-transformers` embedder downloads `all-MiniLM-L6-v2` from Hugging
  Face on first use, then embeds on-device. No memory content is sent to a third
  party.
- **Memory content is untrusted input.** Anything Hermes stores can come from
  web pages, files or messages. The provider does not execute stored content; it
  returns it as recall context.
