from __future__ import annotations

from typing import Any

from .models import ContextBundle, MemoryLayer, MemoryRecord, RetrievalHit
from .retrieval import lexical_overlap


class ContextAssembler:
    def __init__(self, strategy: dict[str, Any]) -> None:
        self.strategy = strategy

    def assemble(self, query: str, hits: list[RetrievalHit], trace) -> ContextBundle:
        config = self.strategy["assembly"]
        budget = int(config["token_budget"])
        min_confidence = float(config["min_confidence"])
        profile_budget = int(budget * float(config["profile_fraction"]))
        dedup_threshold = float(self.strategy["extraction"]["dedup_threshold"])

        filtered: list[RetrievalHit] = []
        guidance: list[MemoryRecord] = []
        for hit in hits:
            if hit.record.confidence < min_confidence:
                trace.filters.append({"id": hit.record.id, "reason": "low_confidence"})
                continue
            if hit.record.layer is MemoryLayer.PROCEDURE:
                guidance.append(hit.record)
                continue
            if any(
                lexical_overlap(hit.record.content, existing.record.content) >= dedup_threshold
                for existing in filtered
            ):
                trace.filters.append({"id": hit.record.id, "reason": "deduplicated"})
                continue
            filtered.append(hit)

        mandatory = [hit for hit in filtered if hit.record.metadata.get("kind") in {"correction", "constraint"}]
        profiles = [hit for hit in filtered if hit.record.layer is MemoryLayer.L2 and hit not in mandatory]
        regular = [hit for hit in filtered if hit not in mandatory and hit not in profiles]

        selected: list[RetrievalHit] = []
        used = 0

        def pack(candidates: list[RetrievalHit], limit: int) -> None:
            nonlocal used
            local_used = 0
            for hit in candidates:
                cost = hit.record.token_estimate + 14
                if used + cost <= budget and local_used + cost <= limit:
                    selected.append(hit)
                    used += cost
                    local_used += cost
                else:
                    trace.dropped_ids.append(hit.record.id)

        pack(mandatory, budget)
        pack(profiles, profile_budget)
        pack(regular, budget)

        sections = []
        for hit in selected:
            source_ids = [source.source_id for source in hit.record.sources]
            sections.append(
                f"[{hit.record.layer.value}:{hit.record.id}] {hit.record.content}\n"
                f"provenance={source_ids or ['local']} updated={hit.record.updated_at}"
            )
        if guidance:
            sections.append(
                "[OPERATIONAL GUIDANCE]\n"
                + "\n".join(f"- {record.content}" for record in guidance)
            )
        trace.selected_ids = [hit.record.id for hit in selected]
        trace.token_usage = used
        return ContextBundle(query, selected, guidance, "\n\n".join(sections), used, trace)
