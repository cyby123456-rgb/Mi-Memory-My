from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MemoryLayer(StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    SM = "SM"
    PROCEDURE = "PROCEDURE"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


@dataclass(slots=True)
class SourceRef:
    source_id: str
    source_type: str = "dialogue"
    uri: str | None = None
    timestamp: str = field(default_factory=utc_now)
    device_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceRef:
        return cls(**value)


@dataclass(slots=True)
class FusedEvent:
    id: str
    content: str
    source_event_ids: list[str]
    timestamp: str
    device_ids: list[str]
    edge_type: str
    provenance: list[str]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_event_ids:
            raise ValueError("fused event requires atomic source events")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True)
class PerceptionFact:
    fact_id: str
    image_id: str
    content: str
    session: str | None = None
    date: str | None = None
    category: str | None = None
    name: str | None = None
    caption: str | None = None
    confidence: float = 1.0
    source_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("perception fact requires image_id")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True)
class MemoryRecord:
    content: str
    layer: MemoryLayer = MemoryLayer.L0
    id: str = field(default_factory=lambda: uuid4().hex)
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: float = 0.5
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_accessed_at: str = field(default_factory=utc_now)
    access_count: int = 0
    skip_count: int = 0
    sources: list[SourceRef] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("memory content cannot be empty")
        if not 0 <= self.importance <= 1:
            raise ValueError("importance must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.summary:
            self.summary = self.content[:240]
        if not self.title:
            self.title = self.summary[:72]

    @property
    def token_estimate(self) -> int:
        return max(1, (len(self.content) + 3) // 4)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer"] = self.layer.value
        data["status"] = self.status.value
        data["token_estimate"] = self.token_estimate
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MemoryRecord:
        data = dict(value)
        data.pop("token_estimate", None)
        data["layer"] = MemoryLayer(data.get("layer", MemoryLayer.L0))
        data["status"] = MemoryStatus(data.get("status", MemoryStatus.ACTIVE))
        data["sources"] = [SourceRef.from_dict(item) for item in data.get("sources", [])]
        return cls(**data)


@dataclass(slots=True)
class RetrievalHit:
    record: MemoryRecord
    score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    channel_ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": self.score,
            "channel_scores": self.channel_scores,
            "channel_ranks": self.channel_ranks,
        }


@dataclass(slots=True)
class DiagnosticTrace:
    query: str
    query_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    strategy_version: str = "default"
    channel_results: dict[str, list[str]] = field(default_factory=dict)
    fused_ranking: list[dict[str, Any]] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    filters: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextBundle:
    query: str
    evidence: list[RetrievalHit]
    operational_guidance: list[MemoryRecord]
    text: str
    token_usage: int
    trace: DiagnosticTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "evidence": [hit.to_dict() for hit in self.evidence],
            "operational_guidance": [item.to_dict() for item in self.operational_guidance],
            "text": self.text,
            "token_usage": self.token_usage,
            "trace": self.trace.to_dict(),
        }
