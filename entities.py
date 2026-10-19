"""Heuristic entity extraction and entity-boosted recall (rival-hardening: Hindsight/Mem0 parity).

Extracts entity mentions without an LLM (capitalised phrases, code identifiers,
quoted terms) and boosts candidate points that share entities with the query.
No model, no network, deterministic.

Intended interface:

    extract_entities(text) -> list[str]
    entity_overlap(query_entities, text) -> float
    apply_entity_boost(pairs, query_entities, *, weight=0.15) -> list[tuple[float, str]]
"""
from __future__ import annotations
