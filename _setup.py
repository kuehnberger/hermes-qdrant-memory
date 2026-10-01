"""Standalone setup helper for the Qdrant memory provider.

This is a convenience script, NOT the setup path. The canonical path is
``hermes memory setup qdrant``, which walks
``QdrantMemoryProvider.get_config_schema()`` and persists through
``save_config()`` — this file must not become a second source of truth for the
same settings.

Run manually only:
    python -m plugins.memory.qdrant._setup      # from the Hermes install
    python ~/.hermes/plugins/qdrant/_setup.py

It writes exactly the keys ``get_config_schema()`` declares, through the same
``save_config()`` the wizard uses, so the two can never disagree on the shape.

What changed and why (all three were real defects):
  - the config path was hardcoded to ``~/.hermes/plugins/memory/qdrant/``,
    which is not where a user-dir or pip install lands, so writes were lost;
  - it offered an ``ollama`` embedder that ``_embed()`` does not implement
    (it logs "not yet implemented" and silently uses the local model anyway);
  - it wrote ``ollama_url``, a key nothing reads.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("hermes.plugins.memory.qdrant.setup")

def _config_file() -> Path:
    """Where ``save_config()`` writes — resolved, never module-relative.

    The config lives in the profile home (``<HERMES_HOME>/qdrant.json``), not
    beside this file: the plugin directory is a build input that
    ``pm.workspace.members_stamp()`` hashes, so writing there re-synced the venv
    on every launch. Resolved through the provider's own helper so the two can
    never disagree about the path.
    """
    try:
        from plugins.memory.qdrant import _config_json_path
        return Path(_config_json_path())
    except Exception:
        pass
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:
        import os
        raw = os.environ.get("HERMES_HOME", "").strip()
        home = Path(raw).expanduser() if raw else Path.home() / ".hermes"
    return home / "qdrant.json"


def _existing() -> dict:
    try:
        data = json.loads(_config_file().read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _probe(url: str, api_key: str) -> str | None:
    """Return an error string if Qdrant is unreachable, else None."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return ("qdrant-client is not installed. Install it with:\n"
                "    pip install 'qdrant-client>=1.10.0,<2' 'fastembed>=0.4.0,<1'")
    client = None
    try:
        client = QdrantClient(url=url, api_key=api_key or None, timeout=5)
        collections = sorted(c.name for c in client.get_collections().collections)
        print(f"  OK — connected to {url}")
        print(f"  Collections: {', '.join(collections) if collections else '(none)'}")
        return None
    except Exception as e:
        return f"cannot connect to {url}: {type(e).__name__}: {e}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def run_setup() -> int:
    """Interactive Qdrant configuration. Returns a process exit code."""
    print("=" * 60)
    print("Qdrant Memory Provider — standalone setup")
    print("=" * 60)
    print("(the supported path is `hermes memory setup qdrant`)\n")

    current = _existing()

    print("\n[1/4] Qdrant connection")
    print("  1) Local / self-hosted (default http://localhost:6333)")
    print("  2) Qdrant Cloud (needs a URL and an API key)")
    mode = input("  Choice [1/2, default=1]: ").strip() or "1"
    if mode == "2":
        url = input("  Qdrant Cloud URL: ").strip()
        api_key = input("  Qdrant API key: ").strip()
    else:
        url = current.get("url") or "http://localhost:6333"
        api_key = current.get("api_key", "")
        print(f"  URL: {url}")

    print("\n[2/4] API key (blank for a local server)")
    if api_key:
        print(f"  Keeping existing key ...{api_key[-4:]}")
        if input("  Replace it? [y/N]: ").strip().lower() != "y":
            api_key = current.get("api_key", "")
    if not api_key:
        api_key = input("  API key (blank to skip): ").strip()

    print("\n[3/4] Collection")
    collection = input(
        f"  Collection name [default=hermes_memories, current="
        f"{current.get('collection', 'hermes_memories')}]: ").strip()
    collection = collection or current.get("collection") or "hermes_memories"

    print("\n[4/4] Testing connectivity...")
    error = _probe(url, api_key)
    if error:
        print(f"  FAILED — {error}")
        print("  Start a local server with:  docker run -p 6333:6333 qdrant/qdrant")
        return 1

    # Same keys, same values, same writer as the canonical wizard. Embedder and
    # vector_size are left at their defaults: _embed() only implements
    # sentence-transformers/all-MiniLM-L6-v2 at 384 dims.
    # The key is NOT written here: it goes to QDRANT_API_KEY in .env, which is
    # where Hermes reads it from (the secret scope). save_config() also drops
    # any api_key it is handed, so even a caller that still passes one cannot
    # land it in this file.
    from plugins.memory.qdrant import QdrantMemoryProvider
    QdrantMemoryProvider().save_config({
        "url": url,
        "collection": collection,
    }, "")

    if api_key:
        home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        env_path = Path(home) / ".env"
        print(f"\nAPI key NOT saved to { _config_file() } by design.")
        print("Add this line to " + str(env_path) + " (or export it) instead:")
        print(f"    QDRANT_API_KEY={api_key}")

    print(f"\nConfig written to {_config_file()}")
    print("\nEnable the provider:")
    print("    hermes memory setup qdrant")
    print("or in config.yaml:    memory:\n      provider: qdrant")
    return 0


def main() -> int:
    """Entry point: cancellation and import errors exit non-zero, not a traceback."""
    try:
        return run_setup()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled; nothing was written.")
        return 130
    except Exception as e:
        print(f"\nSetup failed: {type(e).__name__}: {e}")
        logger.debug("qdrant standalone setup failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
