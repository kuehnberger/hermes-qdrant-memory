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

# ---------------------------------------------------------------------------
# All schemas (returned by get_tool_schemas)
# ---------------------------------------------------------------------------

ALL_TOOL_SCHEMAS = [
    QDRANT_SEARCH_SCHEMA,
    QDRANT_UPSERT_SCHEMA,
    QDRANT_RECALL_SCHEMA,
    QDRANT_COLLECT_SCHEMA,
]
