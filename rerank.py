"""Cross-encoder rerank over fused recall candidates (Hindsight parity).

Why this module exists
----------------------
Recall today fuses a dense arm and a BM25 arm with Reciprocal Rank Fusion. RRF
is *order-only*: it never looks at the query and the candidate together, so two
documents whose fused ranks differ by a hair are ordered by arithmetic on
ranks, not by relevance. A small cross-encoder reads the (query, document)
pair jointly and scores it, which is the cheap quality win Hindsight gets from
its rerank stage.

It is an OPTIONAL ARM, and the whole design follows from that:

* **Opt-in.** Nothing here runs unless a caller constructs a ``Reranker`` and
  calls ``rerank()``. Importing the module costs nothing and pulls in no ML
  stack (``fastembed`` is imported inside the loader, never at module scope),
  so a deployment that never opts in never pays for the ONNX runtime.
* **Degradation is mandatory.** Every failure mode — loader raises, model
  cannot be built, the score count does not match the document count, a score
  is not finite — logs a warning and returns the ORIGINAL ``pairs`` list
  unchanged (same order, same objects). A rerank arm that can take recall down
  when it misfires is worse than no rerank arm, so the failure path is the
  fused order, exactly as RRF produced it.
* **Short-circuit.** Empty and single-element inputs are returned as-is
  without touching the loader: there is nothing to reorder, and a one-document
  recall is not worth a model load.

Reading the result
------------------
``rerank()`` returns a NEW list of ``(score, text)`` where ``score`` is the
CROSS-ENCODER score — a query-conditioned relevance number, not the RRF/fused
score the candidate arrived with. The original scores are therefore NOT
preserved and must not be compared against a fused score from elsewhere; they
exist only to order this one result set (descending, ties keeping their
incoming relative order).

Intended interface:

    class Reranker:
        __init__(model_name=DEFAULT_MODEL, *, cache_dir=None, loader=None)
        available() -> bool
        rerank(query, pairs, *, top_k=None) -> list[tuple[float, str]]
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("hermes.plugins.memory.qdrant.rerank")

#: The checkpoint the rerank arm uses by default. Declared at module level (not
#: inline at a call site) so the model id is configuration, not a literal that
#: silently downloads: it is the smallest fastembed cross-encoder
#: (~0.08 GB, 6 layers) and scores ms-marco-style passage relevance, which is
#: what a fused memory-recall candidate list is asking about.
DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

#: ``(fused_score, text)`` — the shape recall hands over and rerank returns.
Pair = tuple[float, str]

#: ``(model_name, cache_dir) -> model``, where ``model.rerank(query, documents)``
#: yields one numeric score per document. This is the injection seam: tests
#: pass a fake so the suite never downloads or loads an ONNX model.
Loader = Callable[[str, "str | None"], Any]


def fastembed_available() -> bool:
    """True when the fastembed cross-encoder import path is importable.

    Deliberately an IMPORT probe, not a model check: it must be answerable
    without downloading weights or building an ONNX session (a recall path
    asking "is the optional arm usable?" cannot afford either). A True here
    therefore means "the ML stack is installed", not "the weights are on
    disk" — the load itself still fails soft through :meth:`Reranker.rerank`.
    """
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401
    except Exception:  # ImportError, and anything fastembed raises on import
        return False
    return True


def _default_loader(model_name: str, cache_dir: str | None = None) -> Any:
    """Build the real fastembed cross-encoder.

    The import lives INSIDE the function on purpose: at module scope it would
    make every process that imports this module pull in onnxruntime (~1-2 s,
    tens of MB of RSS) even when the rerank arm is never used — the opposite of
    an opt-in arm. Raising is fine and expected here; ``Reranker.rerank``
    treats any failure as "no rerank" and keeps the fused order.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)


class Reranker:
    """Opt-in cross-encoder reranker over a fused candidate list.

    Construction is free: no import, no model, no network. The model is built
    on the first ``rerank()`` that actually has work to do, and any failure to
    build it degrades to the input order rather than raising.

    ``loader`` is the injection seam — ``(model_name, cache_dir) -> model``
    where ``model.rerank(query, documents)`` yields one score per document.
    ``None`` means the real fastembed one.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        loader: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        # A blank name would otherwise reach the backend and become an
        # implicit "whatever the library defaults to" — the wrong-checkpoint
        # failure the embedder already refuses to make. Fall back to the
        # declared default instead of guessing.
        self.model_name = str(model_name).strip() or DEFAULT_MODEL
        self.cache_dir = cache_dir
        self._loader = loader
        self._model: Any = None

    def available(self) -> bool:
        """Whether this reranker can plausibly run, without loading anything.

        With an injected loader the caller has already supplied the model
        source, so probing fastembed would be answering the wrong question —
        an injected loader is available by construction. Otherwise this is the
        fastembed import probe. Either way it never constructs a model, never
        touches the network, and never calls the loader.
        """
        if self._loader is not None:
            return True
        return fastembed_available()

    def _load(self) -> Any:
        """Build (once) and return the model. May raise; callers degrade.

        A failed load is NOT cached: the common real cause is a transient
        download or a cold cache, and a permanently-poisoned reranker would
        turn one bad minute into a silently disabled arm for the whole
        process. Each failure is already cheap (it logs and falls back).
        """
        if self._model is None:
            loader = self._loader if self._loader is not None else _default_loader
            self._model = loader(self.model_name, self.cache_dir)
        return self._model

    def rerank(
        self,
        query: str,
        pairs: list[Pair],
        *,
        top_k: int | None = None,
    ) -> list[Pair]:
        """Reorder ``pairs`` by cross-encoder score, best first.

        ``pairs`` are ``(score, text)`` as recall produced them (typically RRF
        output); the documents sent to the model are the ``text`` of each pair,
        in order. Returns a NEW list of ``(cross_encoder_score, text)`` sorted
        by descending cross-encoder score, truncated to ``top_k`` when given
        (``None`` or ``<= 0`` means no truncation).

        The returned float is the CROSS-ENCODER score, NOT the original fused
        score: the incoming scores are not preserved, so the result is only
        comparable within itself.

        Any failure — loader raises, model cannot be built, wrong number of
        scores, a non-finite or non-numeric score — logs a warning and returns
        the ORIGINAL ``pairs`` list unchanged (same order, same objects), so
        the optional arm can never take recall down. Empty and single-element
        inputs come straight back without calling the loader. Ties are stable:
        equal cross-encoder scores keep their incoming relative order.
        """
        if len(pairs) < 2:
            # Nothing to reorder, and a model load is not worth one document.
            return pairs

        texts = [text for _score, text in pairs]
        try:
            model = self._load()
            raw_scores = list(model.rerank(query, texts))
        except Exception as e:  # noqa: BLE001 - the arm must never propagate
            logger.warning(
                "rerank: model %r unusable (%s: %s); keeping the fused order",
                self.model_name, type(e).__name__, e,
            )
            return pairs

        if len(raw_scores) != len(texts):
            logger.warning(
                "rerank: model %r scored %d of %d documents; keeping the "
                "fused order",
                self.model_name, len(raw_scores), len(texts),
            )
            return pairs

        scores: list[float] = []
        for raw in raw_scores:
            try:
                # float() rather than a cast: a numpy float32 that leaks out
                # here would spread a numpy type through the caller's ranking
                # and comparison code.
                value = float(raw)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "rerank: model %r returned a non-numeric score (%r: %s); "
                    "keeping the fused order",
                    self.model_name, raw, e,
                )
                return pairs
            if not math.isfinite(value):
                logger.warning(
                    "rerank: model %r returned a non-finite score (%r); "
                    "keeping the fused order",
                    self.model_name, value,
                )
                return pairs
            scores.append(value)

        # Sort by score descending, index ascending as the tie-break, so equal
        # scores keep their incoming relative order by construction rather than
        # by relying on sort stability.
        order = sorted(range(len(texts)), key=lambda i: (-scores[i], i))
        ranked: list[Pair] = [(scores[i], texts[i]) for i in order]
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]
        return ranked
