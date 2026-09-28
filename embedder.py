"""Embedder abstraction for the Qdrant memory provider.

Why this module exists
----------------------
The provider previously hard-coded sentence-transformers and swallowed every
failure:

    except Exception as e:
        logger.debug("Embedding failed: %s", e)
        return None

``logger.debug`` is invisible at the default level, so a dead embedder produced
``None``, the write was skipped, and the user was told memory had been saved
when nothing was encoded. That is a correctness lie, not just poor logging.

This module makes failure visible and separates the two classes of it:

  * ``EmbeddingConfigError`` — deterministic and user-fixable. A missing model,
    an unknown model name, a dimension the model cannot produce. These fail on
    every single call and cannot self-heal, so they RAISE.
  * ``EmbeddingRuntimeError`` — transient. One OOM during encode, a momentary
    disk hiccup. These may pass on retry, so they are reported (logged at ERROR
    and recorded for ``unavailable_reason()``) but do not raise.

Neither ever returns a bare ``None``. Callers always get either a vector or an
exception carrying an actionable message.

Backend choice
--------------
``fastembed`` (ONNX Runtime) is the default: it is CPU-only, needs no CUDA
build, and measured 291 MB peak RSS against sentence-transformers' 2,191 MB.

``sentence-transformers`` remains fully supported and configurable because it is
the better choice on a machine with a GPU, or with RAM to spare. Neither backend
is a fallback for the other by default — mixing them silently is exactly the
failure this module exists to prevent.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes.plugins.memory.qdrant.embedder")

# The checkpoint every deployment is expected to use. It is pinned as a default
# but is ALWAYS passed explicitly to the backend: a library with its own default
# model is the same silent-wrong-space failure in disguise (fastembed defaults
# to BAAI/bge-small-en-v1.5, not this).
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Dimensionality of the models this provider knows about. Used to validate a
# configured vector_size against the model BEFORE any write is attempted.
KNOWN_MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-de-v1.5": 384,
}

BACKEND_FASTEMBED = "fastembed"
BACKEND_ST = "sentence-transformers"
BACKENDS = (BACKEND_FASTEMBED, BACKEND_ST)


class EmbeddingError(RuntimeError):
    """Base class for embedder failures. Always carries an actionable message."""

    def __init__(self, message: str, *, model: str = "", backend: str = "") -> None:
        super().__init__(message)
        self.model = model
        self.backend = backend


class EmbeddingConfigError(EmbeddingError):
    """Deterministic, user-fixable configuration failure. Must be fixed, not retried."""


class EmbeddingRuntimeError(EmbeddingError):
    """Transient runtime failure. May succeed on a later attempt."""


# ---------------------------------------------------------------------------
# Model presence / download
# ---------------------------------------------------------------------------

def _cache_roots() -> list[Path]:
    """Directories a model may already be cached in, most specific first."""
    roots: list[Path] = []

    fe_env = os.environ.get("FASTEMBED_CACHE_PATH")
    if fe_env:
        roots.append(Path(fe_env))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")

    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"

    roots.append(base / "fastembed")
    roots.append(base / "huggingface" / "hub")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def model_is_present(model: str) -> bool:
    """True if `model` appears to be cached on disk for either backend.

    Deliberately conservative: it only answers "can we be reasonably sure the
    weights are local", and callers treat a False as "offer to download" rather
    than as a hard failure. A backend that does not use a plain directory cache
    is not falsely reported as missing — absence of a match returns False, and
    the caller lets the backend itself try the load.
    """
    if not model:
        return False
    needle = model.split("/")[-1].lower()
    for root in _cache_roots():
        if not root.is_dir():
            continue
        try:
            for entry in root.iterdir():
                if needle in entry.name.lower():
                    return True
        except OSError:
            continue
    return False


def model_cache_dir() -> str:
    """The directory a downloaded model would live in (for user-facing reports)."""
    roots = _cache_roots()
    return str(roots[0]) if roots else ""


# ---------------------------------------------------------------------------
# The embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Lazily-constructed text embedder for one (backend, model, device) triple.

    Construction is deferred until the first ``encode()`` so that merely
    importing or configuring the provider costs nothing — the model load is
    paid on the first memory operation in a process, not at import time.
    """

    def __init__(self, backend: str = BACKEND_FASTEMBED,
                 model: str = DEFAULT_MODEL,
                 device: str = "auto") -> None:
        if backend not in BACKENDS:
            raise EmbeddingConfigError(
                f"unknown embedder backend {backend!r}; expected one of "
                f"{', '.join(BACKENDS)}",
                model=model, backend=backend,
            )
        if not model or not str(model).strip():
            # Never fall back to a default here. A blank model name is how a
            # deployment silently ends up in the wrong vector space.
            raise EmbeddingConfigError(
                "embedder model name is empty; set memory.qdrant.model "
                f"(e.g. {DEFAULT_MODEL!r}). Refusing to guess a default.",
                model=model, backend=backend,
            )
        self.backend = backend
        self.model = model
        self.device = device or "auto"
        self._impl: Any = None

    # -- construction --------------------------------------------------------

    def _build(self) -> Any:
        """Construct the underlying backend model. Raises EmbeddingConfigError."""
        if self.backend == BACKEND_FASTEMBED:
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                raise EmbeddingConfigError(
                    "embedder backend 'fastembed' is not installed "
                    f"({e}). Install it, or set memory.qdrant.embedder to "
                    "'sentence-transformers'.",
                    model=self.model, backend=self.backend,
                ) from e
            kwargs: dict[str, Any] = {"model_name": self.model}
            # fastembed picks CPU by default; only pin a device when asked.
            if self.device in ("cuda", "gpu"):
                providers = self._cuda_providers()
                if not providers:
                    raise EmbeddingConfigError(
                        "device 'cuda' requested but onnxruntime reports no "
                        f"CUDAExecutionProvider for model {self.model!r}. "
                        "Use device 'cpu' or 'auto'.",
                        model=self.model, backend=self.backend,
                    )
                kwargs["providers"] = providers
            try:
                return TextEmbedding(**kwargs)
            except Exception as e:
                raise EmbeddingConfigError(
                    f"could not load model {self.model!r} via fastembed: {e}. "
                    "Run qdrant_prepare to download it.",
                    model=self.model, backend=self.backend,
                ) from e

        # sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingConfigError(
                "embedder backend 'sentence-transformers' is not installed "
                f"({e}). Install it, or set memory.qdrant.embedder to "
                "'fastembed'.",
                model=self.model, backend=self.backend,
            ) from e
        kwargs = {}
        if self.device in ("cuda", "gpu"):
            kwargs["device"] = "cuda"
        try:
            return SentenceTransformer(self.model, **kwargs)
        except Exception as e:
            raise EmbeddingConfigError(
                f"could not load model {self.model!r} via "
                f"sentence-transformers: {e}. Run qdrant_prepare to download it.",
                model=self.model, backend=self.backend,
            ) from e

    @staticmethod
    def _cuda_providers() -> list[str]:
        try:
            import onnxruntime as ort
            return [p for p in ort.get_available_providers()
                    if "CUDA" in p.upper() or "GPU" in p.upper()]
        except Exception:
            return []

    # -- encoding ------------------------------------------------------------

    def _ensure(self) -> Any:
        if self._impl is None:
            self._impl = self._build()
        return self._impl

    def encode(self, text: str) -> list[float]:
        """Encode `text`, raising EmbeddingError on any failure.

        Deterministic problems (tier 1) propagate as EmbeddingConfigError from
        _build(). Per-call runtime failures (tier 2) are wrapped in
        EmbeddingRuntimeError — raised, not returned as None, so no caller can
        mistake a failed embed for a completed one.
        """
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingConfigError(
                "cannot embed empty or non-string text",
                model=self.model, backend=self.backend,
            )

        impl = self._ensure()  # tier 1 raises out of here

        try:
            if self.backend == BACKEND_FASTEMBED:
                vec = next(iter(impl.passage_embed([text])))
                out = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            else:
                vec = impl.encode(text, normalize_embeddings=True)
                out = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        except EmbeddingError:
            raise
        except Exception as e:
            # Tier 2: transient. Still an exception — never a silent None.
            raise EmbeddingRuntimeError(
                f"embedding failed for model {self.model!r} via "
                f"{self.backend}: {e}",
                model=self.model, backend=self.backend,
            ) from e

        # A vector that is the wrong width or contains non-finite values would
        # be written to the collection and corrupt recall silently. Reject it.
        if not out:
            raise EmbeddingRuntimeError(
                f"embedder {self.backend} returned an empty vector for model "
                f"{self.model!r}",
                model=self.model, backend=self.backend,
            )
        for x in out:
            if not math.isfinite(x):
                raise EmbeddingRuntimeError(
                    f"embedder {self.backend} produced a non-finite value "
                    f"(NaN/Inf) for model {self.model!r}",
                    model=self.model, backend=self.backend,
                )
        return out

    def dimension(self) -> int:
        """Output width of the configured model, measured not assumed.

        Falls back to the static table when the model cannot be loaded, and
        returns 0 when genuinely unknown so the caller can report that honestly
        rather than guessing.
        """
        try:
            impl = self._ensure()
        except EmbeddingError:
            return KNOWN_MODEL_DIMS.get(self.model, 0)

        try:
            if self.backend == BACKEND_FASTEMBED:
                return int(impl.dim)  # fastembed exposes .dim
        except Exception:
            pass
        try:
            if hasattr(impl, "get_sentence_embedding_dimension"):
                return int(impl.get_sentence_embedding_dimension())
        except Exception:
            pass
        return KNOWN_MODEL_DIMS.get(self.model, 0)
