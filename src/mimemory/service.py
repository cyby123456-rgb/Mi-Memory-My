from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .assembly import ContextAssembler
from .models import (
    ContextBundle,
    FusedEvent,
    MemoryLayer,
    MemoryRecord,
    MemoryStatus,
    PerceptionFact,
    SourceRef,
    utc_now,
)
from .retrieval import HybridRetriever, hours_since, lexical_overlap, tokenize
from .storage import LiteMemStore, MemoryStore
from .strategy import StrategyManager


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])(?:\s+|(?=\S))")


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        *,
        strategy_manager: StrategyManager | None = None,
        strategy: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.strategy_manager = strategy_manager
        if strategy is None:
            if strategy_manager is None:
                raise ValueError("strategy or strategy_manager is required")
            strategy = strategy_manager.current()
        self.strategy = strategy
        self._reload_components()

    @classmethod
    def local(cls, root: str | Path) -> MemoryService:
        root_path = Path(root)
        manager = StrategyManager(root_path / "strategies")
        return cls(LiteMemStore(root_path / "litemem"), strategy_manager=manager)

    def _reload_components(self) -> None:
        self.retriever = HybridRetriever(self.strategy)
        self.assembler = ContextAssembler(self.strategy)

    def refresh_strategy(self) -> None:
        if self.strategy_manager:
            self.strategy = self.strategy_manager.current()
            self._reload_components()

    def add_memory(
        self,
        content: str,
        *,
        layer: MemoryLayer = MemoryLayer.L0,
        title: str = "",
        summary: str = "",
        keywords: Iterable[str] = (),
        importance: float = 0.5,
        confidence: float = 1.0,
        sources: Iterable[SourceRef] = (),
        metadata: dict[str, Any] | None = None,
        supersedes: Iterable[str] = (),
    ) -> MemoryRecord:
        record = MemoryRecord(
            content=content.strip(),
            layer=layer,
            title=title,
            summary=summary,
            keywords=list(keywords) or sorted(set(tokenize(content)))[:16],
            importance=importance,
            confidence=confidence,
            sources=list(sources),
            metadata=metadata or {},
            supersedes=list(supersedes),
        )
        return self.store.put(record)

    def ingest_text(
        self,
        text: str,
        *,
        source_id: str | None = None,
        session_id: str | None = None,
        create_summary: bool = True,
    ) -> list[MemoryRecord]:
        if not text.strip():
            raise ValueError("text cannot be empty")
        source = SourceRef(source_id=source_id or uuid4().hex, timestamp=utc_now())
        self.store.append_raw(text, session_id=session_id)
        max_facts = int(self.strategy["extraction"]["max_facts_per_turn"])
        candidates = [part.strip() for part in SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]
        if not candidates:
            candidates = [text.strip()]
        existing = self.store.list()
        threshold = float(self.strategy["extraction"]["dedup_threshold"])
        created: list[MemoryRecord] = []
        for fact in candidates[:max_facts]:
            if any(lexical_overlap(fact, item.content) >= threshold for item in existing + created):
                continue
            created.append(self.add_memory(fact, layer=MemoryLayer.L0, sources=[source]))
        if create_summary and len(candidates) > 1:
            summary = " ".join(candidates[:4])
            created.append(
                self.add_memory(
                    summary,
                    layer=MemoryLayer.L1,
                    title=f"Session {session_id or source.source_id}",
                    sources=[source],
                    metadata={"session_id": session_id},
                )
            )
        return created

    def remember_profile(self, content: str, *, source_id: str | None = None) -> MemoryRecord:
        if isinstance(self.store, LiteMemStore):
            return self.store.write_profile(content)
        source = SourceRef(source_id=source_id or uuid4().hex)
        return self.add_memory(content, layer=MemoryLayer.L2, importance=0.9, sources=[source], metadata={"kind": "profile"})

    def remember_procedure(
        self,
        trigger: str,
        procedure: list[str],
        *,
        constraints: list[str] | None = None,
        validation: list[str] | None = None,
    ) -> MemoryRecord:
        body = f"When {trigger}: " + " -> ".join(procedure)
        return self.add_memory(
            body,
            layer=MemoryLayer.PROCEDURE,
            importance=0.85,
            metadata={
                "trigger": trigger,
                "procedure": procedure,
                "constraints": constraints or [],
                "validation": validation or [],
            },
        )

    def admit_fused_event(self, event: FusedEvent) -> MemoryRecord:
        sources = [
            SourceRef(
                source_id=source_id,
                source_type="device_event",
                timestamp=event.timestamp,
                device_id=event.device_ids[index] if index < len(event.device_ids) else None,
            )
            for index, source_id in enumerate(event.source_event_ids)
        ]
        return self.add_memory(
            event.content,
            layer=MemoryLayer.L0,
            confidence=event.confidence,
            sources=sources,
            metadata={
                "payload_type": "FusedEvent",
                "fused_event_id": event.id,
                "edge_type": event.edge_type,
                "provenance": event.provenance,
                **event.metadata,
            },
        )

    def admit_perception_fact(self, fact: PerceptionFact) -> MemoryRecord:
        source_ids = fact.source_ids or [fact.image_id]
        sources = [SourceRef(source_id=item, source_type="image", timestamp=fact.date or utc_now()) for item in source_ids]
        return self.add_memory(
            fact.content,
            layer=MemoryLayer.L0,
            confidence=fact.confidence,
            sources=sources,
            metadata={
                "payload_type": "PerceptionFact",
                "fact_id": fact.fact_id,
                "image_id": fact.image_id,
                "session": fact.session,
                "date": fact.date,
                "category": fact.category,
                "name": fact.name,
                "caption": fact.caption,
            },
        )

    def recall(self, query: str, *, persist_trace: bool = True, touch: bool = True) -> ContextBundle:
        result = self.retriever.retrieve(query, self.store.list())
        bundle = self.assembler.assemble(query, result.hits, result.trace)
        if touch and hasattr(self.store, "touch_access"):
            self.store.touch_access(hit.record for hit in bundle.evidence)
        if persist_trace:
            self.store.write_trace(bundle.trace)
        return bundle

    def correct(self, record_id: str, replacement: str, *, source_id: str | None = None) -> MemoryRecord:
        old = self.store.get(record_id)
        if old is None:
            raise KeyError(record_id)
        old.status = MemoryStatus.DEPRECATED
        old.updated_at = utc_now()
        self.store.put(old)
        return self.add_memory(
            replacement,
            layer=MemoryLayer.L2,
            importance=max(old.importance, 0.9),
            confidence=1.0,
            sources=[SourceRef(source_id=source_id or uuid4().hex, source_type="correction")],
            metadata={"kind": "correction", "corrects": record_id},
            supersedes=[record_id],
        )

    def forget(self, record_id: str, *, reason: str = "user_request") -> MemoryRecord:
        record = self.store.get(record_id)
        if record is None:
            raise KeyError(record_id)
        record.status = MemoryStatus.FORGOTTEN
        record.updated_at = utc_now()
        record.metadata["forgotten_at"] = utc_now()
        record.metadata["forget_reason"] = reason
        return self.store.put(record)

    def organize(self) -> dict[str, list[str]]:
        config = self.strategy["lifecycle"]
        archive_threshold = float(config["archive_threshold"])
        tau = float(config["importance_tau_hours"])
        access_boost = float(config["access_boost"])
        skip_penalty = float(config["skip_penalty"])
        archived: list[str] = []
        updated: list[str] = []
        for record in self.store.list():
            age = hours_since(record.updated_at)
            decayed = record.importance * math.exp(-age / tau)
            score = decayed + access_boost * record.access_count - skip_penalty * record.skip_count
            record.metadata["decayed_importance"] = score
            if score < archive_threshold and record.layer not in {MemoryLayer.L2, MemoryLayer.PROCEDURE}:
                record.status = MemoryStatus.ARCHIVED
                archived.append(record.id)
            else:
                updated.append(record.id)
            self.store.put(record)
        return {"archived": archived, "updated": updated}
