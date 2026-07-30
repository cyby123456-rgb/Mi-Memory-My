"""Paper-faithful MemStack runtime with injectable model roles.

The implementation follows the public Appendix B contracts: LLM fact extraction,
LLM query planning, vector/BM25/subquery RRF, independent reranking, and traceful
context assembly. No fallback heuristic is used when a required model role fails.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from .assembly import ContextAssembler
from .models import ContextBundle, DiagnosticTrace, MemoryLayer, MemoryRecord, RetrievalHit, SourceRef, utc_now
from .paper_contracts import FACT_EXTRACTION_PROMPT, QUERY_PLAN_PROMPT, RERANK_PROMPT
from .providers import EmbeddingProvider, ProviderError, json_from_completion
from .retrieval import HybridRetriever
from .storage import MemoryStore


class CompletionProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float = 0.0) -> str: ...


@dataclass(slots=True)
class MemStackModels:
    extractor: CompletionProvider
    planner: CompletionProvider
    reranker: CompletionProvider
    embeddings: EmbeddingProvider


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


class MemStackRuntime:
    def __init__(self, store: MemoryStore, strategy: dict[str, Any], models: MemStackModels) -> None:
        self.store = store
        self.strategy = deepcopy(strategy)
        self.models = models
        self.assembler = ContextAssembler(self.strategy)

    def ingest(self, messages: list[dict[str, Any]], *, user_id: str, session_id: str) -> list[MemoryRecord]:
        if not messages:
            raise ValueError("MemStack ingestion requires at least one ordered message")
        normalized = []
        for index, message in enumerate(messages):
            content = message.get("content")
            role = message.get("role")
            source_id = str(message.get("source_id", f"{session_id}:{index}"))
            if not isinstance(content, str) or not content.strip() or role not in {"user", "assistant"}:
                raise ValueError("each message requires user/assistant role and non-empty content")
            normalized.append({"source_id": source_id, "role": role, "content": content, "timestamp": message.get("timestamp")})
        response = json_from_completion(
            self.models.extractor.complete(
                [{"role": "system", "content": FACT_EXTRACTION_PROMPT}, {"role": "user", "content": str(normalized)}]
            )
        )
        facts = response.get("facts")
        if not isinstance(facts, list):
            raise ProviderError("fact extractor response must include facts array")
        records: list[MemoryRecord] = []
        max_facts = int(self.strategy["extraction"]["max_facts_per_turn"])
        for item in facts[:max_facts]:
            if not isinstance(item, dict):
                raise ProviderError("fact entry must be an object")
            content = item.get("content")
            source_ids = item.get("source_ids")
            if not isinstance(content, str) or not content.strip() or not isinstance(source_ids, list) or not source_ids:
                raise ProviderError("fact must contain content and source_ids")
            allowed = {message["source_id"] for message in normalized}
            if not set(map(str, source_ids)).issubset(allowed):
                raise ProviderError("fact cites a source id outside the ingestion window")
            record = MemoryRecord(
                content=content,
                layer=MemoryLayer.L0,
                importance=float(item.get("importance", 0.5)),
                confidence=float(item.get("confidence", 1.0)),
                sources=[SourceRef(source_id=str(source_id), source_type="dialogue") for source_id in source_ids],
                metadata={"type": item.get("type", "other"), "entities": item.get("entities", []), "temporal": item.get("temporal"), "user_id": user_id, "session_id": session_id},
            )
            records.append(record)
        summary = response.get("session_summary")
        if isinstance(summary, str) and summary.strip():
            records.append(MemoryRecord(content=summary, layer=MemoryLayer.L1, sources=[SourceRef(source_id=item["source_id"]) for item in normalized], metadata={"user_id": user_id, "session_id": session_id}))
        for update in response.get("profile_updates", []):
            if not isinstance(update, dict) or not isinstance(update.get("content"), str):
                continue
            source_ids = [str(item) for item in update.get("source_ids", [])]
            if source_ids and set(source_ids).issubset({item["source_id"] for item in normalized}):
                records.append(MemoryRecord(content=update["content"], layer=MemoryLayer.L2, importance=0.9, confidence=float(update.get("confidence", 1.0)), sources=[SourceRef(source_id=item) for item in source_ids], metadata={"user_id": user_id, "session_id": session_id}))
        vectors = self.models.embeddings.embed([record.content for record in records])
        if len(vectors) != len(records):
            raise ProviderError("embedding provider did not return one vector per MemStack record")
        for record, vector in zip(records, vectors, strict=True):
            record.metadata["embedding"] = vector
            record.metadata["embedding_model"] = "configured"
            self.store.put(record)
        return records

    def retrieve(self, query: str, *, user_id: str) -> ContextBundle:
        plan = json_from_completion(self.models.planner.complete([{"role": "system", "content": QUERY_PLAN_PROMPT}, {"role": "user", "content": query}]))
        subqueries = plan.get("subqueries")
        if not isinstance(subqueries, list) or not all(isinstance(item, str) for item in subqueries) or len(subqueries) > 5:
            raise ProviderError("query planner returned invalid subqueries")
        records = [record for record in self.store.list() if record.metadata.get("user_id") == user_id]
        query_vector = self.models.embeddings.embed([query])[0]
        vector_rank = sorted(((record.id, _cosine(query_vector, record.metadata.get("embedding", []))) for record in records), key=lambda item: (-item[1], item[0]))
        lexical_engine = HybridRetriever(self.strategy)
        lexical_rank = lexical_engine._bm25(query, records)
        expanded_scores: dict[str, float] = {}
        for subquery in subqueries:
            for record_id, score in lexical_engine._bm25(subquery, records):
                expanded_scores[record_id] = max(expanded_scores.get(record_id, 0.0), score)
        subquery_rank = sorted(expanded_scores.items(), key=lambda item: (-item[1], item[0]))
        channels = {"semantic": vector_rank, "lexical": lexical_rank, "subquery": subquery_rank}
        fused = lexical_engine._rrf(channels, self.strategy["retrieval"])
        candidate_limit = int(self.strategy["retrieval"]["rerank_top_n"])
        by_id = {record.id: record for record in records}
        candidates = [item for item in fused[:candidate_limit] if item[0] in by_id]
        rerank_payload = {"query": query, "intent": plan.get("intent", ""), "candidates": [{"id": item[0], "content": by_id[item[0]].content, "provenance": [source.source_id for source in by_id[item[0]].sources]} for item in candidates]}
        ranked = json_from_completion(self.models.reranker.complete([{"role": "system", "content": RERANK_PROMPT}, {"role": "user", "content": str(rerank_payload)}]))
        rows = ranked.get("ranking")
        if not isinstance(rows, list) or {row.get("id") for row in rows if isinstance(row, dict)} != {item[0] for item in candidates}:
            raise ProviderError("reranker must return every candidate id exactly once")
        rerank_scores = {str(row["id"]): float(row["score"]) for row in rows}
        ordered = sorted(candidates, key=lambda item: (-rerank_scores[item[0]], -item[1], item[0]))
        trace = DiagnosticTrace(query=query, strategy_version=str(self.strategy.get("artifact_id", "default")))
        trace.channel_results = {name: [record_id for record_id, _ in ranking] for name, ranking in channels.items()}
        trace.fused_ranking = [{"id": record_id, "score": score, "channel_ranks": ranks, "rerank_score": rerank_scores.get(record_id)} for record_id, score, ranks, _ in ordered]
        hits = [RetrievalHit(by_id[record_id], rerank_scores[record_id], channel_scores, ranks) for record_id, _, ranks, channel_scores in ordered]
        bundle = self.assembler.assemble(query, hits, trace)
        self.store.write_trace(bundle.trace)
        return bundle
