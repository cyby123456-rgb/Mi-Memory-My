"""Published MemSense and MemFuse data paths.

The paper specifies public artifacts and ordering, rather than the original
authors' private prompts.  This module therefore makes each published pass and
graph mutation explicit and retains the source evidence for every output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from .models import FusedEvent, PerceptionFact
from .providers import ProviderError, json_from_completion


MEMSENSE_PASSES = (
    ("conversation_name", "Extract only names and references established by the preceding dialogue."),
    ("category", "Normalize observations into retrieval categories without adding visual claims."),
    ("temporal", "Bind the observation to its supplied session and date; do not infer a time."),
    ("related_fact", "Bind supported visual facts to the supplied source ids and dialogue references."),
    ("uncertainty", "Mark uncertainty and emit only evidence-grounded final IKB facts."),
)
MEMSENSE_PROMPT = """You are MemSense pass {pass_name}. {instruction}
Return JSON only: {{"facts": [{{"content": str, "category": str|null, "name": str|null,
"caption": str|null, "confidence": number, "source_ids": [str]}}]}}. Never invent image evidence."""

MEMFUSE_PROMPT = """You are MemFuse. Given time-ordered atomic cross-device events, form one FusionSession and
propose auditable semantic/causal edges. Return JSON only: {"summary": str, "edge_type": "BELONG|CAUSES|RELATED",
"source_event_ids": [str], "confidence": number, "provenance": [str]}. All source ids must originate in input."""
IKB_ROUTE_PROMPT = """Classify this multimodal memory query. Return JSON only:
{"intent":"VR|VS|TTL", "category":str|null, "session":str|null, "date":str|null,
"image_ids":[str], "residual_query":str}. VR retrieves visual records, VS finds visual semantics,
and TTL resolves text-to-image links. Do not invent identifiers."""


@dataclass(slots=True)
class ImageObservation:
    image_id: str
    image_reference: str
    source_ids: list[str]
    session: str | None = None
    date: str | None = None
    preceding_dialogue: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MemSensePassArtifact:
    pass_name: str
    input_source_ids: tuple[str, ...]
    output_fact_count: int
    created_at: str
    response: dict[str, Any]


@dataclass(slots=True)
class ImageKnowledgeBase:
    facts: list[PerceptionFact] = field(default_factory=list)
    by_category: dict[str, list[str]] = field(default_factory=dict)
    by_session: dict[str, list[str]] = field(default_factory=dict)
    by_date: dict[str, list[str]] = field(default_factory=dict)

    def add(self, facts: list[PerceptionFact]) -> None:
        ids = {item.fact_id for item in self.facts}
        for item in facts:
            if item.fact_id in ids:
                continue
            self.facts.append(item)
            ids.add(item.fact_id)
            if item.category:
                self.by_category.setdefault(item.category.casefold(), []).append(item.fact_id)
            if item.session:
                self.by_session.setdefault(item.session, []).append(item.fact_id)
            if item.date:
                self.by_date.setdefault(item.date, []).append(item.fact_id)

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
        """Apply Algorithm 1's index constraints before any residual VLM call."""
        ids: set[str] | None = None
        for index, key in ((self.by_category, category.casefold() if category else None), (self.by_session, session), (self.by_date, date)):
            if key is not None:
                current = set(index.get(key, []))
                ids = current if ids is None else ids & current
        candidates = [item for item in self.facts if ids is None or item.fact_id in ids]
        if intent == "VR":
            return sorted(candidates, key=lambda item: item.fact_id)
        if intent in {"VS", "TTL"}:
            return sorted(candidates, key=lambda item: (-item.confidence, item.fact_id))
        raise ValueError("IKB intent must be VR, VS, or TTL")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"facts": [asdict(fact) for fact in self.facts], "by_category": self.by_category,
            "by_session": self.by_session, "by_date": self.by_date}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ImageKnowledgeBase":
        target = Path(path)
        if not target.exists():
            return cls()
        value = json.loads(target.read_text(encoding="utf-8"))
        facts = [PerceptionFact(**item) for item in value.get("facts", [])]
        ikb = cls()
        ikb.add(facts)  # Rebuild indexes from facts rather than trusting stale serialized indexes.
        return ikb


class MemSense:
    def __init__(self, vision_model: Any) -> None:
        self.vision_model = vision_model
        self.pass_artifacts: list[MemSensePassArtifact] = []

    @staticmethod
    def _vision_content(observation: ImageObservation, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Use OpenAI-compatible multimodal content rather than serialising an image URL as text."""
        return [
            {"type": "text", "text": str({"observation": {"image_id": observation.image_id, "source_ids": observation.source_ids,
                "session": observation.session, "date": observation.date, "preceding_dialogue": observation.preceding_dialogue}, "prior_passes": state})},
            {"type": "image_url", "image_url": {"url": observation.image_reference}},
        ]

    def build_ikb(self, observation: ImageObservation) -> list[PerceptionFact]:
        if not observation.source_ids:
            raise ValueError("an image observation requires source ids")
        state: dict[str, Any] = {}
        final_rows: list[dict[str, Any]] = []
        self.pass_artifacts = []
        for pass_name, instruction in MEMSENSE_PASSES:
            result = json_from_completion(self.vision_model.complete([
                {"role": "system", "content": MEMSENSE_PROMPT.format(pass_name=pass_name, instruction=instruction)},
                {"role": "user", "content": self._vision_content(observation, state)},
            ]))
            rows = result.get("facts")
            if not isinstance(rows, list):
                raise ProviderError("each MemSense pass must return a facts array")
            self.pass_artifacts.append(MemSensePassArtifact(pass_name, tuple(observation.source_ids), len(rows), datetime.now(UTC).isoformat(), result))
            state[pass_name] = rows
            # The uncertainty pass is authoritative; accepting earlier rows keeps
            # simple compliant providers useful while preserving a full audit trail.
            if pass_name == "uncertainty" or not final_rows:
                final_rows = rows
        return self._facts(observation, final_rows)

    @staticmethod
    def _facts(observation: ImageObservation, rows: list[dict[str, Any]]) -> list[PerceptionFact]:
        facts: list[PerceptionFact] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("content"), str) or not row["content"].strip():
                raise ProviderError("MemSense fact must contain content")
            source_ids = row.get("source_ids", [])
            if not isinstance(source_ids, list) or not source_ids or not set(source_ids).issubset(set(observation.source_ids)):
                raise ProviderError("MemSense fact cites an unknown source")
            key = (row["content"], row.get("category"), row.get("name"))
            if key in seen:
                continue
            seen.add(key)
            facts.append(PerceptionFact(fact_id=f"{observation.image_id}:{index}", image_id=observation.image_id,
                content=row["content"], session=observation.session, date=observation.date, category=row.get("category"),
                name=row.get("name"), caption=row.get("caption"), confidence=float(row.get("confidence", 1.0)), source_ids=source_ids))
        return facts


@dataclass(frozen=True, slots=True)
class IKBRouteResult:
    intent: Literal["VR", "VS", "TTL"]
    facts: tuple[PerceptionFact, ...]
    image_ids: tuple[str, ...]
    used_residual_vlm: bool


class IKBRouter:
    """Algorithm 1: IKB lookup first, then bounded graph/image and VLM residual work."""
    def __init__(self, route_model: Any, *, vlm_fallback: Callable[[str, list[str]], list[str]] | None = None) -> None:
        self.route_model, self.vlm_fallback = route_model, vlm_fallback

    def route(self, query: str, ikb: ImageKnowledgeBase, *, session_facts: list[PerceptionFact] | None = None,
              graph_image_ids: list[str] | None = None) -> IKBRouteResult:
        result = json_from_completion(self.route_model.complete([
            {"role": "system", "content": IKB_ROUTE_PROMPT}, {"role": "user", "content": query},
        ]))
        intent = result.get("intent")
        if intent not in {"VR", "VS", "TTL"}:
            raise ProviderError("IKB router must classify VR, VS, or TTL")
        facts = ikb.ikb_first(intent, category=result.get("category"), session=result.get("session"), date=result.get("date"))
        if session_facts:
            allowed = {fact.fact_id for fact in session_facts}
            facts = [fact for fact in facts if fact.fact_id in allowed]
        image_ids = list(dict.fromkeys([*result.get("image_ids", []), *(graph_image_ids or []), *[fact.image_id for fact in facts]]))[:8]
        used_residual = False
        if not facts and self.vlm_fallback:
            image_ids = list(dict.fromkeys([*image_ids, *self.vlm_fallback(str(result.get("residual_query", query)), image_ids)]))[:8]
            used_residual = True
        return IKBRouteResult(intent, tuple(facts), tuple(image_ids), used_residual)


@dataclass(slots=True)
class DeviceEvent:
    event_id: str
    content: str
    timestamp: str
    device_id: str
    source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusionSession:
    """The three transient zones retain raw observations until a pack is committed."""
    session_id: str = field(default_factory=lambda: uuid4().hex)
    perception_zone: list[DeviceEvent] = field(default_factory=list)
    interaction_zone: list[DeviceEvent] = field(default_factory=list)
    environment_zone: list[DeviceEvent] = field(default_factory=list)
    max_events_per_zone: int = 64

    def add(self, event: DeviceEvent, zone: Literal["perception", "interaction", "environment"]) -> list[DeviceEvent]:
        target = getattr(self, f"{zone}_zone")
        target.append(event)
        target.sort(key=lambda item: item.timestamp)
        evicted = target[:-self.max_events_per_zone]
        del target[:-self.max_events_per_zone]
        return evicted

    def ordered_events(self) -> list[DeviceEvent]:
        return sorted([*self.perception_zone, *self.interaction_zone, *self.environment_zone], key=lambda item: item.timestamp)


@dataclass(frozen=True, slots=True)
class CausalEdge:
    source_id: str
    target_id: str
    relation: Literal["BELONG", "CAUSES", "RELATED"]
    confidence: float
    provenance: tuple[str, ...]


@dataclass(slots=True)
class CausalMemoryGraph:
    atomic_events: dict[str, DeviceEvent] = field(default_factory=dict)
    memory_packs: dict[str, FusedEvent] = field(default_factory=dict)
    edges: list[CausalEdge] = field(default_factory=list)

    def retain_atomic(self, events: list[DeviceEvent]) -> None:
        self.atomic_events.update({event.event_id: event for event in events})

    def add_pack(self, pack: FusedEvent) -> None:
        self.memory_packs[pack.id] = pack
        for event_id in pack.source_event_ids:
            self.edges.append(CausalEdge(event_id, pack.id, "BELONG", pack.confidence, tuple(pack.provenance)))
        if pack.edge_type in {"CAUSES", "RELATED"} and len(pack.source_event_ids) > 1:
            self.edges.append(CausalEdge(pack.source_event_ids[0], pack.source_event_ids[-1], pack.edge_type, pack.confidence, tuple(pack.provenance)))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"atomic_events": [asdict(item) for item in self.atomic_events.values()],
            "memory_packs": [asdict(item) for item in self.memory_packs.values()], "edges": [asdict(item) for item in self.edges]}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CausalMemoryGraph":
        target = Path(path)
        if not target.exists():
            return cls()
        value = json.loads(target.read_text(encoding="utf-8"))
        graph = cls()
        graph.atomic_events = {item["event_id"]: DeviceEvent(**item) for item in value.get("atomic_events", [])}
        graph.memory_packs = {item["id"]: FusedEvent(**item) for item in value.get("memory_packs", [])}
        graph.edges = [CausalEdge(**{**item, "provenance": tuple(item["provenance"])}) for item in value.get("edges", [])]
        return graph


class MemFuse:
    def __init__(self, fusion_model: Any, *, confidence_floor: float = 0.5, graph_path: str | Path | None = None) -> None:
        self.fusion_model, self.confidence_floor = fusion_model, confidence_floor
        self.graph_path = Path(graph_path) if graph_path else None
        self.graph = CausalMemoryGraph.load(self.graph_path) if self.graph_path else CausalMemoryGraph()

    def _persist_graph(self) -> None:
        if self.graph_path:
            self.graph.save(self.graph_path)

    def fuse(self, events: list[DeviceEvent]) -> FusedEvent:
        if len(events) < 2:
            raise ValueError("FusionSession requires at least two atomic device events")
        ordered = sorted(events, key=lambda item: item.timestamp)
        self.graph.retain_atomic(ordered)
        self._persist_graph()
        payload = [{"event_id": item.event_id, "content": item.content, "timestamp": item.timestamp, "device_id": item.device_id, "source_ids": item.source_ids} for item in ordered]
        result = json_from_completion(self.fusion_model.complete([{"role": "system", "content": MEMFUSE_PROMPT}, {"role": "user", "content": str(payload)}]))
        source_event_ids = result.get("source_event_ids")
        valid_ids = {item.event_id for item in ordered}
        if not isinstance(source_event_ids, list) or not source_event_ids or not set(source_event_ids).issubset(valid_ids):
            raise ProviderError("MemFuse output must cite only input event ids")
        edge_type = result.get("edge_type", "RELATED")
        if edge_type not in {"BELONG", "CAUSES", "RELATED"}:
            raise ProviderError("MemFuse edge must be BELONG, CAUSES, or RELATED")
        selected = [item for item in ordered if item.event_id in set(source_event_ids)]
        provenance = result.get("provenance", [])
        confidence = float(result.get("confidence", 1.0))
        if not isinstance(provenance, list) or not str(result.get("summary", "")).strip():
            raise ProviderError("MemFuse requires summary and provenance")
        if confidence < self.confidence_floor:
            raise ProviderError("MemFuse confidence is below the fusion threshold; atomic events were retained")
        fused = FusedEvent(id=uuid4().hex, content=str(result["summary"]).strip(), source_event_ids=source_event_ids,
            timestamp=selected[-1].timestamp, device_ids=sorted({item.device_id for item in selected}), edge_type=edge_type,
            provenance=[str(item) for item in provenance], confidence=confidence, metadata={"session_start": selected[0].timestamp})
        self.graph.add_pack(fused)
        self._persist_graph()
        return fused
