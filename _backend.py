"""Qdrant backend abstraction for the Hermes Qdrant memory provider.

A standalone helper wrapping qdrant_client over REST:
  - Collection management (create, list, info, delete)
  - Upsert (single + batch)
  - Dense search, plus hybrid dense+sparse RRF fusion
  - Scroll (filtered bulk recall)
  - Payload indexing
  - INT8 scalar quantization setup

STATUS — read before using. This module is a LIBRARY, not the live path:
``QdrantMemoryProvider`` in ``__init__.py`` talks to ``QdrantClient`` directly
and never constructs a ``QdrantBackend``. Only ``TestQdrantBackendConnectivity``
exercises this class today. ``hybrid_search()`` and
``setup_scalar_quantization()`` in particular are implemented but UNWIRED — do
not advertise either as a provider capability until the provider calls them.

The provider imports nothing from here; it imports qdrant_client lazily so a
missing dependency degrades to "unavailable" instead of an import error.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("hermes.plugins.memory.qdrant.backend")


class QdrantBackend:
    """Abstraction over qdrant_client for the Hermes memory provider.

    Not on the provider's live path — see the module docstring.
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: str = "") -> None:
        self._url = url
        self._api_key = api_key
        self._client: Any = None
        self._collection = "hermes_memories"

    # -- Connection -----------------------------------------------------------

    def connect(self) -> None:
        """Initialize the Qdrant client (REST mode)."""
        from qdrant_client import QdrantClient

        self._client = QdrantClient(
            url=self._url,
            api_key=self._api_key or None,
            prefer_grpc=False,
        )
        logger.info("Qdrant connected (url=%s, REST mode)", self._url)

    def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # -- Collection management -----------------------------------------------

    def ensure_collection(
        self,
        name: str,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> None:
        """Create collection if it doesn't exist."""
        if self._client is None:
            raise RuntimeError("Not connected — call connect() first")

        collections = {c.name for c in self._client.get_collections().collections}
        if name not in collections:
            logger.info("Creating collection %s (dims=%d, distance=%s)",
                        name, vector_size, distance)
            from qdrant_client.models import Distance, VectorParams

            self._client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance[distance.upper()],
                ),
            )
            self._ensure_payload_indexes(name)

    def _ensure_payload_indexes(self, name: str) -> None:
        """Create keyword indexes on session_id and source payload fields."""
        try:
            from qdrant_client.models import PayloadSchemaType
            self._client.create_payload_index(
                collection_name=name,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.debug("Payload index creation: %s", e)

    def list_collections(self) -> list[str]:
        """Return names of all collections."""
        if self._client is None:
            return []
        return [c.name for c in self._client.get_collections().collections]

    def get_collection_info(self, name: str) -> dict[str, Any] | None:
        """Return collection info (point count, config)."""
        if self._client is None:
            return None
        try:
            info = self._client.get_collection(name)
            return {
                "name": info.collection_name,
                "points": info.points_count,
                "config": str(info.config),
            }
        except Exception:
            return None

    def delete_collection(self, name: str) -> None:
        """Delete a collection (destructive)."""
        if self._client is not None:
            self._client.delete_collection(name)

    # -- Upsert ---------------------------------------------------------------

    def upsert(
        self,
        points: list[dict],
        collection: str = "hermes_memories",
        wait: bool = True,
    ) -> None:
        """Upsert a batch of points.

        Each point: {"id": str, "vector": list[float], "payload": dict}
        """
        if self._client is None:
            raise RuntimeError("Not connected")
        self._client.upsert(
            collection_name=collection,
            points=points,
            wait=wait,
        )

    # -- Search ---------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        collection: str = "hermes_memories",
        session_id: str = "",
        limit: int = 10,
    ) -> list[dict]:
        """Dense vector search with optional session_id filter."""
        if self._client is None:
            return []

        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        flt = (
            Filter(
                must=[
                    FieldCondition(
                        key="session_id", match=MatchValue(value=session_id)
                    )
                ]
            )
            if session_id
            else None
        )

        results = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=flt,
            limit=limit,
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results.points
        ]

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict | None = None,
        collection: str = "hermes_memories",
        session_id: str = "",
        limit: int = 10,
        fusion: str = "rrf",
    ) -> list[dict]:
        """Hybrid dense+sparse search with RRF fusion.

        sparse_vector: {"indices": [...], "values": [...]} or None for dense-only.
        """
        if self._client is None:
            return []

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FusionQuery,
            MatchValue,
            Prefetch,
        )

        flt = (
            Filter(
                must=[
                    FieldCondition(
                        key="session_id", match=MatchValue(value=session_id)
                    )
                ]
            )
            if session_id
            else None
        )

        prefetch_queries = [
            Prefetch(query=dense_vector, using="dense", limit=limit),
        ]
        if sparse_vector:
            prefetch_queries.append(
                Prefetch(query=sparse_vector, using="sparse", limit=limit)
            )

        results = self._client.query_points(
            collection_name=collection,
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=fusion) if len(prefetch_queries) > 1 else None,
            query_filter=flt,
            limit=limit,
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results.points
        ]

    # -- Scroll (bulk recall) -------------------------------------------------

    def scroll(
        self,
        collection: str = "hermes_memories",
        session_id: str = "",
        limit: int = 1000,
        with_payload: bool = True,
    ) -> list[dict]:
        """Scroll with optional session_id filter for bulk recall."""
        if self._client is None:
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = (
            Filter(
                must=[
                    FieldCondition(
                        key="session_id", match=MatchValue(value=session_id)
                    )
                ]
            )
            if session_id
            else None
        )

        results, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=limit,
            with_payload=with_payload,
        )
        return [
            {"id": r.id, "payload": r.payload}
            for r in results
        ]

    # -- Quantization ---------------------------------------------------------

    def setup_scalar_quantization(
        self,
        collection: str = "hermes_memories",
        bits: int = 8,
    ) -> None:
        """Apply INT8 scalar quantization to a collection.

        Reduces memory by 4x-32x depending on vector size and cardinality.
        """
        if self._client is None:
            raise RuntimeError("Not connected")

        from qdrant_client.models import QuantizationConfig, ScalarQuantization

        self._client.update_collection(
            collection_name=collection,
            quantization_config=QuantizationConfig(
                scalar=ScalarQuantization(
                    always=False,
                    ignore=False,
                    quantize=False,
                )
            ),
        )
        logger.info("Scalar quantization applied to %s (%d-bit)", collection, bits)
