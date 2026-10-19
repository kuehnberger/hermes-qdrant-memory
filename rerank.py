"""Cross-encoder rerank over fused recall candidates (rival-hardening: Hindsight parity).

Reranks the RRF-fused candidate list with a small ONNX cross-encoder, opt-in.
The model is loaded lazily and every failure degrades to the unreranked order.

Intended interface:

    class Reranker:
        available() -> bool
        rerank(query, pairs, *, top_k=None) -> list[tuple[float, str]]
"""
from __future__ import annotations
