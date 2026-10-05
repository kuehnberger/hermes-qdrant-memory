# Contributing

Thanks for helping. This is a **standalone Hermes plugin repo** — the repo root
*is* the plugin directory, not a subdirectory of the Hermes Agent tree.

## Ground rules

- **Never edit the Hermes core tree.** The plugin is discovered from
  `$HERMES_HOME/plugins/qdrant/` and implements the `MemoryProvider` ABC
  (`agent/memory_provider.py`). If you believe core needs a new hook, open an
  issue describing the generic capability you need — do not special-case Qdrant
  in core.
- **Never vendor Hermes core.** Hermes is not on PyPI. `tests/conftest.py` stubs
  the handful of core symbols the provider imports; that stub is the seam, and it
  must stay minimal and honest.
- **A release is a commit SHA, not a tag.** The Hermes plugin catalog pins a
  40-character commit SHA; `hermes plugins install --ref` rejects tags.
- **Dependency pins are oldest-compatible floor + upper bound.** Do not raise a
  floor to a newest-release version "to be safe" — see `pyproject.toml`.

## Layout

```
__init__.py      # QdrantMemoryProvider + register(ctx)
_backend.py      # qdrant_client wrapper
_setup.py        # config-schema driven setup helper (not a CLI subcommand)
tool_schemas.py  # the tool schemas (count asserted by TestDeclarationParity)
plugin.yaml      # the manifest (validated by `hermes plugins validate`)
pyproject.toml   # the sole dependency authority
tests/           # pytest; conftest.py stubs Hermes core
docs/            # banner + screenshots
```

## Development

```bash
uv venv && uv sync          # deps only; this is a directory plugin, not a dist
uv run pytest               # unit tests
```

Tests that touch a real Qdrant server need one on a non-default port
(`docker run -p 16333:6333 qdrant/qdrant`, then
`QDRANT_URL=http://localhost:16333 uv run pytest`). `tests/conftest.py`
*fails loudly* if `QDRANT_URL` names Qdrant's default ports (6333 REST / 6334
gRPC — where production lives) and skips when no scratch server is named.

## Before you open a PR

- `hermes plugins validate .` passes (`security scan: safe`, plus the one expected warning — see AGENTS.md).
- `plugin.yaml` `provides_tools` matches what `get_tool_schemas()` returns.
- `CHANGELOG.md` has an entry under the unreleased heading.
