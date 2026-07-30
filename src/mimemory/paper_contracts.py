"""Published Mi-Memory cross-module contracts and prompt templates.

These schemas encode the public interfaces in Appendix B--F. They intentionally
make unknown author-internal implementation details configurable rather than
silently replacing them with heuristic behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


FACT_EXTRACTION_PROMPT = """You are the MemStack fact extractor. Convert the ordered conversation into typed,
atomic, evidence-grounded memory facts. Preserve entities, quantities, dates, negations, and source turn ids.
Return JSON only with this exact shape:
{
  "facts": [{"content": str, "type": "temporal|location|plan|preference|slot|ledger|other", "source_ids": [str], "confidence": number, "importance": number, "entities": [str], "temporal": str|null}],
  "session_summary": str,
  "profile_updates": [{"content": str, "source_ids": [str], "confidence": number}]
}
Do not invent facts. Every fact must name one or more source_ids supplied in the input."""


QUERY_PLAN_PROMPT = """You are the MemStack retrieval router. Preserve explicit entities, dates, quantities, and
negations from the user query. Return JSON only:
{"intent": str, "subqueries": [str], "requires_procedure": bool, "requires_temporal_grounding": bool}.
Generate at most five subqueries and do not alter the original query."""


RERANK_PROMPT = """You are the independent MemStack cross-encoder-style reranker. Rank candidates only by
their evidence support for the query. Prefer exact entity, temporal, and provenance alignment; penalize unrelated
or contradicted facts. Return JSON only: {"ranking": [{"id": str, "score": number, "reason": str}]}.
Include every candidate id exactly once and score higher for stronger support."""


LAYER_A_PROMPT = """You are the D2ACCI Layer A faithfulness classifier. Given a benchmark question, gold answer,
original source window, and retrieved fact, return JSON only: {"label": "Full_Coverage|Source_Only|Uncertain", "reason": str}.
Full_Coverage means the retrieved fact contains enough answer evidence. Source_Only means the source does but the
fact lost or weakened it. Do not use outside knowledge."""


CRITIC_PROMPT = """You are the E2MEND Critic. Review one bounded strategy mutation with its affected schema path,
dimension reputation, probe improvements/regressions, and examples. Return JSON only:
{"decision": "actionable|risky|duplicate|out_of_scope", "reason": str}. You cannot override hard gates."""


@dataclass(frozen=True, slots=True)
class PublishedContract:
    name: str
    required_fields: tuple[str, ...]
    producer: str
    consumer: str


PUBLISHED_CONTRACTS = (
    PublishedContract(
        "FusedEvent",
        ("id", "source_event_ids", "timestamp", "device_ids", "edge_type", "provenance", "confidence"),
        "MemFuse",
        "MemStack",
    ),
    PublishedContract(
        "PerceptionFact",
        ("fact_id", "image_id", "session/date", "category/name", "caption", "confidence", "source_ids"),
        "MemSense",
        "MemStack",
    ),
    PublishedContract(
        "DiagnosticSignal",
        ("question_id", "retrieved_ids", "context_ids", "answer_evidence", "stage_label", "error_label"),
        "MemStack",
        "E2MEND",
    ),
    PublishedContract(
        "StrategyArtifact",
        ("artifact_id", "schema_version", "mutation_paths", "guardrails", "gate_record", "rollback_key"),
        "E2MEND",
        "MemStack",
    ),
)

