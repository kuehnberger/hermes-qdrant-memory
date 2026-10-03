"""Semantic fallback for ``md_search``: multilingual embeddings over ``hermes_md_docs``.

Kept in its OWN module on purpose. ``mdsearch.py`` is the FTS5 fast path and
must stay importable without ever touching an embedding library; the only way
to make that guarantee mechanical instead of aspirational is to make the model
reachable from exactly one module, and assert in the test suite that
``mdsearch`` does not import this one.

Model choice
------------
``paraphrase-multilingual-MiniLM-L12-v2`` (384-dim), NOT the memory provider's
``DEFAULT_MODEL`` (``all-MiniLM-L6-v2``). Two reasons, and the second is the
important one:

1. The doc corpus is German and English; an English-only model ranks German
   queries badly.
2. Vectors are only comparable WITHIN one model. These docs live in their own
   collection (``hermes_md_docs``) precisely so this model can differ from the
   memory provider's — mixing them would silently corrupt recall. The memory
   provider's ``DEFAULT_MODEL`` is deliberately not touched by this feature
   (re-embedding 128k existing memories would cost ~36.1M tokens).

Collection separation
---------------------
``DOCS_COLLECTION = "hermes_md_docs"`` is a DIFFERENT collection from the
provider's ``collection=hermes_memories``, and it carries NO session filter.
Docs are shared, per-machine knowledge; memories are per-session. Merging them
would make every recall session-scoped or every doc visible in every session's
recall, and both are wrong.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterable
from typing import Any, NamedTuple

logger = logging.getLogger("hermes.plugins.memory.qdrant.mdsearch_semantic")

#: Separate from hermes_memories. Asserted in tests (test_mdsearch.py).
DOCS_COLLECTION = "hermes_md_docs"

#: 384-dim, same width as the memory collection — but NOT the same model, so
#: the two collections are never queried against each other's vectors.
DOCS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCS_DIM = 384

#: Namespace for the point id. ``uuid5`` (SHA-1 based) rather than uuid4 so the
#: id is DERIVED — see :func:`chunk_point_id`.
_ID_NAMESPACE = uuid.NAMESPACE_URL


class SemanticHit(NamedTuple):
    path: str
    heading: str
    score: float


class DocsBackendUnavailable(RuntimeError):
    """Deterministic: the docs model or the Qdrant collection is not usable.

    Raised rather than returned as ``None`` so no caller can read an empty
    result list as "the semantic search found nothing" when in fact it never
    ran. The ingest CLI and the fallback both turn this into an actionable
    message.
    """


def _load_embedder_module():
    """The provider's ``embedder`` module, in either load mode.

    The plugin is loaded as a package by Hermes and as a bare top-level module
    during test collection, so the relative import needs the importlib fallback.
    """
    try:
        from . import embedder as module
        return module
    except ImportError as err:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "hermes_qdrant_embedder",
            os.path.join(os.path.dirname(__file__), "embedder.py"),
        )
        if spec is None or spec.loader is None:
            raise DocsBackendUnavailable(
                f"could not load the plugin's embedder module from {__file__!r}; "
                "the docs collection cannot embed anything without it"
            ) from err
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def make_embedder():
    """An ``Embedder`` bound to the DOCS model.

    Its own instance, not the provider's: a different model in the same object
    would put doc vectors in the memory provider's space.
    """
    try:
        return _load_embedder_module().Embedder(model=DOCS_MODEL)
    except Exception as exc:
        raise DocsBackendUnavailable(
            f"could not construct the docs embedder for {DOCS_MODEL!r}: {exc}"
        ) from exc


def embed_texts(texts: list[str], embedder: Any | None = None) -> list[list[float]]:
    """Embed with the docs model. Raises ``DocsBackendUnavailable`` on failure."""
    embedder = embedder or make_embedder()
    out: list[list[float]] = []
    for text in texts:
        try:
            out.append(embedder.encode(text))
        except Exception as exc:
            raise DocsBackendUnavailable(
                f"embedding failed for the docs model {DOCS_MODEL!r}: {exc}"
            ) from exc
    return out


def _client():
    """A Qdrant client from the same env seam the provider honours.

    Imported lazily: ``qdrant_client`` is a heavy import and the FTS5 path must
    not pay for it.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise DocsBackendUnavailable(
            f"qdrant-client is not installed ({exc}); the semantic fallback is "
            "unavailable — lexical (FTS5) search still works"
        ) from exc
    url = (os.environ.get("QDRANT_URL") or "http://localhost:6333").strip()
    return QdrantClient(url=url, prefer_grpc=False, timeout=15)


def ensure_collection(client: Any) -> None:
    """Create ``hermes_md_docs`` if absent. Idempotent."""
    from qdrant_client.http import models as qmodels

    try:
        client.get_collection(DOCS_COLLECTION)
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=DOCS_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=DOCS_DIM, distance=qmodels.Distance.COSINE
        ),
    )
    logger.info(
        "md-search: created collection %s (%d-dim cosine)",
        DOCS_COLLECTION,
        DOCS_DIM,
    )


def payload_for(label: str, heading: str, sha: str) -> dict:
    """The exact payload every docs point carries.

    Factored out of :func:`upsert_chunks` and asserted directly by the test
    suite, because the payload's key set IS a contract (``{path, heading, root,
    sha}``, and deliberately NO session scope) — inlining it in the write path
    made that contract untestable without a live Qdrant server.
    """
    return {
        "path": label,
        "heading": heading,
        "root": label.split("/", 1)[0],
        "sha": sha,
    }


def chunk_point_id(label: str, heading: str, ordinal: int) -> str:
    """The deterministic id of one chunk point.

    ``(path, heading)`` alone is NOT unique. Real corpus, measured on the
    2026-10-03 index: 51,670 FTS chunks but only 47,459 distinct
    ``(path, heading)`` pairs — 4,211 chunks (8.1%) across 801 files collided on
    one id, last write won, and the losers stayed queryable in FTS5 while being
    invisible to the vector tier. ``references/llms-full.md`` alone collapsed
    423 chunks.

    The ordinal is the chunk's occurrence count *within* that (path, heading)
    pair, so duplicates coexist and the id stays a pure function of position —
    re-ingesting an unchanged file rewrites exactly the ids it already owns
    instead of accumulating copies.

    Deliberately NOT keyed on the file ``sha``: a content-dependent id turns
    every edit into "delete the old ids, write new ones" for a caller that
    upserts without deleting, i.e. a second stale copy of every point. Section
    *removals* are handled where they belong — ``md_ingest`` deletes the file's
    points before writing a changed one (delete-before-write) — so the id only
    has to be collision-free and stable.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, f"{label}\x00{heading}\x00{ordinal}"))


def upsert_chunks(client: Any, rows: Iterable[tuple[str, str, str, str]],
                  embedder: Any | None = None) -> int:
    """Embed and write ``(label, heading, content, sha)`` rows.

    The payload is :func:`payload_for` — ``{path, heading, root, sha}`` and
    deliberately NO ``session_id``: these are shared documents, and a session
    filter here would make the index visible only to the session that wrote it.

    Ids come from :func:`chunk_point_id`, so two chunks sharing a heading path
    get two points. This function does NOT clear points a caller no longer
    wants: an upsert overwrites ids that still exist and leaves the rest, and
    removing deleted sections is the caller's job (``md_ingest`` does it with
    :func:`delete_file_points` before this call).
    """
    from qdrant_client.http import models as qmodels

    ensure_collection(client)
    rows = list(rows)
    if not rows:
        return 0
    vectors = embed_texts([content for _, _, content, _ in rows], embedder=embedder)
    ordinals: dict[tuple[str, str], int] = {}
    points = []
    for (label, heading, _content, sha), vector in zip(rows, vectors, strict=True):
        key = (label, heading)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        points.append(qmodels.PointStruct(
            id=chunk_point_id(label, heading, ordinal),
            vector=vector,
            payload=payload_for(label, heading, sha),
        ))
    client.upsert(collection_name=DOCS_COLLECTION, points=points, wait=True)
    return len(points)


def delete_file_points(client: Any, label: str) -> None:
    """Remove every chunk point belonging to one file label.

    The delete-before-write half of incremental ingest: an upsert only
    overwrites ids that still exist, so a section removed from a file would
    otherwise keep its point (and its old ``sha``) forever. The filter is on the
    payload's ``path``, which :func:`payload_for` writes on every point.
    """
    from qdrant_client.http import models as qmodels

    client.delete(
        collection_name=DOCS_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="path", match=qmodels.MatchValue(value=label)
                )]
            )
        ),
        wait=True,
    )


def search_semantic(query: str, limit: int = 5, *, client: Any | None = None,
                    embedder: Any | None = None, root: str = "") -> list[SemanticHit]:
    """Vector search over ``hermes_md_docs``.

    Raises ``DocsBackendUnavailable`` when the model or server is unusable, and
    returns ``[]`` only for a genuine no-match. The caller decides whether a
    thin lexical result is worth the fallback; it must never be told "no
    results" when the truth is "the semantic tier never ran".
    """
    from qdrant_client.http import models as qmodels

    own_client = client is None
    if own_client:
        client = _client()
    try:
        vector = embed_texts([query], embedder=embedder)[0]
        query_filter = None
        if root:
            query_filter = qmodels.Filter(must=[qmodels.FieldCondition(
                key="root", match=qmodels.MatchValue(value=root)
            )])
        resp = client.query_points(
            collection_name=DOCS_COLLECTION,
            query=vector,
            limit=max(1, int(limit)),
            query_filter=query_filter,
            with_payload=True,
        )
        hits = []
        for point in resp.points:
            payload = point.payload or {}
            hits.append(SemanticHit(
                path=str(payload.get("path", "")),
                heading=str(payload.get("heading", "")),
                score=float(point.score),
            ))
        return hits
    except DocsBackendUnavailable:
        raise
    except Exception as exc:
        raise DocsBackendUnavailable(
            f"semantic search against {DOCS_COLLECTION!r} failed: {exc}"
        ) from exc
    finally:
        if own_client:
            try:
                client.close()
            except Exception:
                pass


def index_available() -> bool:
    """True when the docs collection exists and carries at least one point."""
    try:
        client = _client()
    except DocsBackendUnavailable:
        return False
    try:
        info = client.get_collection(DOCS_COLLECTION)
        return int(getattr(info, "points_count", 0) or 0) > 0
    except Exception:
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass
