from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_STRATEGY: dict[str, Any] = {
    "schema_version": 1,
    "extraction": {
        "max_facts_per_turn": 8,
        "dedup_threshold": 0.92,
        "prompt_template": "v1: preserve source_ids and return JSON facts",
    },
    "retrieval": {
        "top_k": 12,
        "rrf_k": 60,
        "weights": {"semantic": 1.0, "lexical": 1.0, "subquery": 1.0},
        "subquery_max_n": 5,
        "rerank_top_n": 12,
        "rerank_attempts": 3,
        "intent_override": "auto",
    },
    "assembly": {
        "token_budget": 1200,
        "profile_fraction": 0.15,
        "min_confidence": 0.2,
    },
    "lifecycle": {
        "recency_tau_hours": 720.0,
        "importance_tau_hours": 4320.0,
        "access_boost": 0.02,
        "skip_penalty": 0.01,
        "archive_threshold": 0.05,
        "compression_policy": "summary_only",
        "source_window": 16,
    },
    "presentation": {"output_format": "evidence_first"},
    "features": {"failure_correction": True, "conflict_handling": True, "conditional_memory_triggers": True, "hypothesis_tree": False},
}


def default_strategy() -> dict[str, Any]:
    return deepcopy(DEFAULT_STRATEGY)
