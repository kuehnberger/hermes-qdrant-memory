"""Tool schemas for the Qdrant memory provider.

Each tool is a dict conforming to the Hermes tool-call schema:
  {"name", "description", "parameters": {"type", "properties", "required"}}

These are returned by QdrantMemoryProvider.get_tool_schemas() and registered
with the Hermes tool dispatcher.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

QDRANT_SEARCH_SCHEMA = {
    "name": "qdrant_search",
    "description": (
        "Search Qdrant for semantically similar memories. "
        "Uses dense vector search with optional session scoping."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text (will be embedded automatically)",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session ID to scope results",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

QDRANT_UPSERT_SCHEMA = {
    "name": "qdrant_upsert",
    "description": (
        "Store a memory point in Qdrant. "
        "Text is embedded and saved with metadata (session_id, timestamp)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Memory text to store",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session ID for scoping",
            },
        },
        "required": ["text"],
    },
}

QDRANT_RECALL_SCHEMA = {
    "name": "qdrant_recall",
    "description": (
        "Bulk recall memories from Qdrant using filtered scroll. "
        "Returns all memories matching the session_id filter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID to recall memories for",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 100)",
                "default": 100,
            },
        },
        "required": ["session_id"],
    },
}

QDRANT_COLLECT_SCHEMA = {
    "name": "qdrant_collect",
    "description": (
        "Collection management: list collections or get info on the active one. "
        "Use action='list' to see all collections, action='info' for details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: 'list' or 'info'",
                "enum": ["list", "info"],
            },
            "collection": {
                "type": "string",
                "description": "Collection name (only 'info' is supported today; "
                               "the active collection is used when omitted)",
            },
        },
        "required": ["action"],
    },
}

QDRANT_PREPARE_SCHEMA = {
    "name": "qdrant_prepare",
    "description": (
        "Check the memory embedder is ready: is the model present on disk, do "
        "the configured vector size and distance match the model and the live "
        "collection? Use download=true to fetch a missing model. Returns a "
        "readable report with any problems and recommended fixes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "download": {
                "type": "boolean",
                "description": "Fetch the model if it is not cached (default false, "
                               "which only reports)",
                "default": False,
            },
        },
        "required": [],
    },
}

# ---------------------------------------------------------------------------
# All schemas (returned by get_tool_schemas)
# ---------------------------------------------------------------------------

QDRANT_FORGET_SCHEMA = {
    "name": "qdrant_forget",
    "description": (
        "Delete specific memories from Qdrant by point ID. "
        "Point-targeted only: use IDs from qdrant_search/qdrant_recall. "
        "There is no bulk or delete-all action — dropping a whole memory "
        "store is a human decision, made outside the agent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "point_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact point IDs to delete, as returned by qdrant_search "
                    "or qdrant_recall"
                ),
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "Must be true. Without it the call only reports which IDs "
                    "exist and what they contain, and deletes nothing."
                ),
                "default": False,
            },
        },
        "required": ["point_ids"],
    },
}

MD_SEARCH_SCHEMA = {
    "name": "md_search",
    "description": (
        "Search the local markdown corpus (skills, vault, docs) for passages that "
        "match a query. Lexical (SQLite FTS5 + bm25) search runs first and answers "
        "in milliseconds without loading any model; when the lexical matches are "
        "thin, a multilingual embedding fallback runs against the hermes_md_docs "
        "collection. Use this to find reference material you have not already been "
        "given — not for memories, which are qdrant_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search the markdown corpus for",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of passages to return (default 5)",
                "default": 5,
            },
            "root": {
                "type": "string",
                "description": (
                    "Restrict to one corpus root: 'skills', 'vault' or 'docs'. "
                    "Omit to search all of them."
                ),
            },
            "semantic": {
                "type": "string",
                "description": (
                    "'auto' (default) falls back to embeddings only when lexical "
                    "matches are thin, 'never' stays lexical-only, 'always' tries the "
                    "semantic tier too"
                ),
                "enum": ["auto", "never", "always"],
                "default": "auto",
            },
        },
        "required": ["query"],
    },
}


ALL_TOOL_SCHEMAS = [
    QDRANT_SEARCH_SCHEMA,
    QDRANT_UPSERT_SCHEMA,
    QDRANT_RECALL_SCHEMA,
    QDRANT_COLLECT_SCHEMA,
    QDRANT_PREPARE_SCHEMA,
    QDRANT_FORGET_SCHEMA,
    MD_SEARCH_SCHEMA,
]
