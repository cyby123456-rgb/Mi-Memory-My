from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from .models import DiagnosticTrace, MemoryRecord, RetrievalHit


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _sparse_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    norm_l = math.sqrt(sum(value * value for value in left.values()))
    norm_r = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


def _semantic_features(text: str) -> Counter[str]:
    normalized = " ".join(tokenize(text))
    features: Counter[str] = Counter(tokenize(normalized))
    compact = normalized.replace(" ", "_")
    features.update(f"#tri:{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2)))
    return features


def expand_subqueries(query: str, max_n: int = 5) -> list[str]:
    pieces = [query.strip()]
    pieces.extend(part.strip() for part in re.split(r"[,;，；]|\b(?:and|or|then)\b|以及|然后|和", query, flags=re.I))
    unique: list[str] = []
    for piece in pieces:
        if piece and piece.casefold() not in {item.casefold() for item in unique}:
            unique.append(piece)
    return unique[:max_n]


@dataclass(slots=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    trace: DiagnosticTrace


class HybridRetriever:
    def __init__(self, strategy: dict[str, Any]) -> None:
        self.strategy = strategy

    def retrieve(self, query: str, records: Iterable[MemoryRecord]) -> RetrievalResult:
        docs = list(records)
        config = self.strategy["retrieval"]
        top_k = int(config["top_k"])
        semantic = self._semantic(query, docs)
        lexical = self._bm25(query, docs)
        subquery = self._subquery(query, docs, int(config["subquery_max_n"]))
        channels = {"semantic": semantic, "lexical": lexical, "subquery": subquery}
        fused = self._rrf(channels, config)

        hits: list[RetrievalHit] = []
        by_id = {record.id: record for record in docs}
        for record_id, score, ranks, scores in fused[:top_k]:
            hits.append(RetrievalHit(by_id[record_id], score, scores, ranks))

        trace = DiagnosticTrace(query=query, strategy_version=str(self.strategy.get("artifact_id", "default")))
        trace.channel_results = {
            name: [record_id for record_id, _ in ranking[:top_k]] for name, ranking in channels.items()
        }
        trace.fused_ranking = [
            {"id": record_id, "score": score, "channel_ranks": ranks}
            for record_id, score, ranks, _ in fused[:top_k]
        ]
        return RetrievalResult(hits=hits, trace=trace)

    def _semantic(self, query: str, records: list[MemoryRecord]) -> list[tuple[str, float]]:
        query_features = _semantic_features(query)
        scored = []
        for record in records:
            text = f"{record.title} {record.summary} {record.content} {' '.join(record.keywords)}"
            score = _sparse_cosine(query_features, _semantic_features(text))
            if score > 0:
                scored.append((record.id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def _bm25(self, query: str, records: list[MemoryRecord]) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        tokenized = [tokenize(f"{item.title} {item.summary} {item.content} {' '.join(item.keywords)}") for item in records]
        if not query_terms or not tokenized:
            return []
        avgdl = sum(map(len, tokenized)) / len(tokenized) or 1
        document_frequency = Counter(term for term in set(query_terms) for doc in tokenized if term in doc)
        scores: list[tuple[str, float]] = []
        k1, b = 1.5, 0.75
        for record, terms in zip(records, tokenized, strict=True):
            counts = Counter(terms)
            score = 0.0
            for term in set(query_terms):
                df = document_frequency[term]
                idf = math.log(1 + (len(records) - df + 0.5) / (df + 0.5))
                tf = counts[term]
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(terms) / avgdl)) if tf else 0
            if score > 0:
                scores.append((record.id, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))

    def _subquery(self, query: str, records: list[MemoryRecord], max_n: int) -> list[tuple[str, float]]:
        aggregate: dict[str, float] = defaultdict(float)
        for subquery in expand_subqueries(query, max_n=max_n):
            for record_id, score in self._bm25(subquery, records):
                aggregate[record_id] = max(aggregate[record_id], score)
        return sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))

    def _rrf(
        self,
        channels: dict[str, list[tuple[str, float]]],
        config: dict[str, Any],
    ) -> list[tuple[str, float, dict[str, int], dict[str, float]]]:
        rrf_k = float(config["rrf_k"])
        weights = config["weights"]
        fused: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for channel, ranking in channels.items():
            for rank, (record_id, score) in enumerate(ranking, start=1):
                fused[record_id] += float(weights[channel]) / (rrf_k + rank)
                ranks[record_id][channel] = rank
                raw_scores[record_id][channel] = score
        return sorted(
            ((record_id, score, ranks[record_id], raw_scores[record_id]) for record_id, score in fused.items()),
            key=lambda item: (-item[1], item[0]),
        )


def lexical_overlap(left: str, right: str) -> float:
    left_terms, right_terms = set(tokenize(left)), set(tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def hours_since(timestamp: str) -> float:
    try:
        moment = datetime.fromisoformat(timestamp)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - moment).total_seconds() / 3600)
    except ValueError:
        return 0.0

