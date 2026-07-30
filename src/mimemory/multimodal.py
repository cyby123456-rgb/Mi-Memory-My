"""MemSense and MemFuse implementations aligned with the public paper artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .models import FusedEvent, PerceptionFact, SourceRef
from .providers import ProviderError, json_from_completion


MEMSENSE_PROMPT = """You are MemSense. Build an Image Knowledge Base entry from a visual observation and its
preceding dialogue. Execute the five public passes: conversation-name extraction, category normalization, temporal
indexing, related-fact binding, and uncertainty marking. Return JSON only with facts: [{"content", "category",
"name", "caption", "confidence", "source_ids"}]. Never invent image evidence."""

MEMFUSE_PROMPT = """You are MemFuse. Given time-ordered atomic cross-device events, form one FusionSession and
propose auditable semantic/causal edges. Return JSON only: {"summary": str, "edge_type": str,
"source_event_ids": [str], "confidence": number, "provenance": [str]}. All source ids must originate in input."""


@dataclass(slots=True)
class ImageObservation:
    image_id: str
    image_reference: str
    source_ids: list[str]
    session: str | None = None
    date: str | None = None
    preceding_dialogue: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ImageKnowledgeBase:
    facts: list[PerceptionFact] = field(default_factory=list)
    by_category: dict[str, list[str]] = field(default_factory=dict)
    by_session: dict[str, list[str]] = field(default_factory=dict)
    by_date: dict[str, list[str]] = field(default_factory=dict)

    def add(self, facts: list[PerceptionFact]) -> None:
        ids = {item.fact_id for item in self.facts}
        for item in facts:
            if item.fact_id in ids: continue
            self.facts.append(item); ids.add(item.fact_id)
            if item.category: self.by_category.setdefault(item.category.casefold(), []).append(item.fact_id)
            if item.session: self.by_session.setdefault(item.session, []).append(item.fact_id)
            if item.date: self.by_date.setdefault(item.date, []).append(item.fact_id)

    def select(self, query: str, *, category: str | None = None, session: str | None = None) -> list[PerceptionFact]:
        terms = set(query.casefold().split())
        result = []
        for fact in self.facts:
            haystack = " ".join(filter(None, [fact.content, fact.category, fact.name, fact.caption])).casefold()
            if category and fact.category != category:
                continue
            if session and fact.session != session:
                continue
            if terms & set(haystack.split()):
                result.append(fact)
        return sorted(result, key=lambda item: (-item.confidence, item.fact_id))

    def ikb_first(self, intent: str, *, category: str | None = None, session: str | None = None, date: str | None = None) -> list[PerceptionFact]:
        """Algorithm 1's deterministic IKB constraints before residual VLM retrieval."""
        ids: set[str] | None = None
        for index, key in ((self.by_category, category.casefold() if category else None), (self.by_session, session), (self.by_date, date)):
            if key is not None:
                current = set(index.get(key, [])); ids = current if ids is None else ids & current
        candidates = [item for item in self.facts if ids is None or item.fact_id in ids]
        if intent == "VR": return sorted(candidates, key=lambda item: item.fact_id)
        if intent in {"VS", "TTL"}: return sorted(candidates, key=lambda item: (-item.confidence, item.fact_id))
        raise ValueError("IKB intent must be VR, VS, or TTL")


class MemSense:
    def __init__(self, vision_model: Any) -> None:
        self.vision_model = vision_model

    def build_ikb(self, observation: ImageObservation) -> list[PerceptionFact]:
        payload = {
            "image_id": observation.image_id,
            "image_reference": observation.image_reference,
            "source_ids": observation.source_ids,
            "session": observation.session,
            "date": observation.date,
            "preceding_dialogue": observation.preceding_dialogue,
        }
        result = json_from_completion(self.vision_model.complete([{"role": "system", "content": MEMSENSE_PROMPT}, {"role": "user", "content": str(payload)}]))
        rows = result.get("facts")
        if not isinstance(rows, list):
            raise ProviderError("MemSense must return facts array")
        facts: list[PerceptionFact] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("content"), str):
                raise ProviderError("MemSense fact must contain content")
            source_ids = row.get("source_ids", [])
            if not isinstance(source_ids, list) or not set(source_ids).issubset(set(observation.source_ids)):
                raise ProviderError("MemSense fact cites an unknown source")
            facts.append(PerceptionFact(
                fact_id=f"{observation.image_id}:{index}", image_id=observation.image_id, content=row["content"],
                session=observation.session, date=observation.date, category=row.get("category"), name=row.get("name"),
                caption=row.get("caption"), confidence=float(row.get("confidence", 1.0)), source_ids=source_ids,
            ))
        return facts


@dataclass(slots=True)
class DeviceEvent:
    event_id: str
    content: str
    timestamp: str
    device_id: str
    source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class MemFuse:
    def __init__(self, fusion_model: Any) -> None:
        self.fusion_model = fusion_model

    def fuse(self, events: list[DeviceEvent]) -> FusedEvent:
        if len(events) < 2:
            raise ValueError("FusionSession requires at least two atomic device events")
        ordered = sorted(events, key=lambda item: item.timestamp)
        payload = [{"event_id": item.event_id, "content": item.content, "timestamp": item.timestamp, "device_id": item.device_id, "source_ids": item.source_ids} for item in ordered]
        result = json_from_completion(self.fusion_model.complete([{"role": "system", "content": MEMFUSE_PROMPT}, {"role": "user", "content": str(payload)}]))
        source_event_ids = result.get("source_event_ids")
        valid_ids = {item.event_id for item in ordered}
        if not isinstance(source_event_ids, list) or not source_event_ids or not set(source_event_ids).issubset(valid_ids):
            raise ProviderError("MemFuse output must cite only input event ids")
        selected = [item for item in ordered if item.event_id in set(source_event_ids)]
        provenance = result.get("provenance", [])
        if not isinstance(provenance, list):
            raise ProviderError("MemFuse provenance must be an array")
        return FusedEvent(
            id=uuid4().hex, content=str(result.get("summary", "")).strip(), source_event_ids=source_event_ids,
            timestamp=selected[-1].timestamp, device_ids=sorted({item.device_id for item in selected}),
            edge_type=str(result.get("edge_type", "RELATED")), provenance=[str(item) for item in provenance],
            confidence=float(result.get("confidence", 1.0)), metadata={"session_start": selected[0].timestamp},
        )
