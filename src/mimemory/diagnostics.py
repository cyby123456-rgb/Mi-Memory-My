"""D2ACCI Layer A evidence-complete diagnostic path (Appendix F.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import MemoryRecord
from .paper_contracts import LAYER_A_PROMPT
from .providers import ProviderError, json_from_completion


@dataclass(slots=True)
class DiagnosticSignal:
    question_id: str
    retrieved_ids: list[str]
    context_ids: list[str]
    answer_evidence: list[str]
    stage_label: str
    error_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def layer_a_diagnose(question_id: str, question: str, gold_answer: str, evidence_source_ids: list[str], stored: list[MemoryRecord], retrieved: list[MemoryRecord], classifier: Any) -> DiagnosticSignal:
    evidence_set = set(evidence_source_ids)
    evidence_facts = [item for item in stored if evidence_set & {source.source_id for source in item.sources}]
    retrieved_evidence = [item for item in retrieved if item.id in {fact.id for fact in evidence_facts}]
    base = {"question_id": question_id, "retrieved_ids": [item.id for item in retrieved], "context_ids": [item.id for item in retrieved], "answer_evidence": evidence_source_ids}
    if not evidence_facts:
        return DiagnosticSignal(**base, stage_label="ingestion", error_label="ingestion_gap")
    if not retrieved_evidence:
        return DiagnosticSignal(**base, stage_label="retrieval", error_label="retrieval_gap")
    chosen = retrieved_evidence[0]
    source_window = [source.source_id for source in chosen.sources]
    response = json_from_completion(classifier.complete([{"role": "system", "content": LAYER_A_PROMPT}, {"role": "user", "content": str({"question": question, "gold_answer": gold_answer, "source_window": source_window, "retrieved_fact": chosen.content})}]))
    label = response.get("label")
    if label == "Full_Coverage":
        return DiagnosticSignal(**base, stage_label="generation", error_label="generation_error")
    if label == "Source_Only":
        return DiagnosticSignal(**base, stage_label="ingestion", error_label="ingestion_gap")
    return DiagnosticSignal(**base, stage_label="uncertain", error_label="uncertain")
