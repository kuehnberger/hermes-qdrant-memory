"""Memory hygiene primitives (rival-hardening: memory-os parity).

Pure, deterministic scoring and clustering for the two maintenance jobs:
a decay scan (archive stale machine-generated points) and a semantic dedup
(merge near-duplicate points). Kept separate from the scripts that touch the
live store so the arithmetic is testable without a server or a model.

Intended interface:

    decay_score(age_days, half_life_days=180.0) -> float
    is_decay_exempt(payload, *, importance_threshold=0.7) -> bool
    cluster_near_duplicates(items, *, threshold=0.92) -> list[list]
    merge_payloads(group) -> dict
"""
from __future__ import annotations
