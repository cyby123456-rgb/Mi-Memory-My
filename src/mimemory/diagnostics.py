"""D2ACCI Layer A evidence-complete diagnostic path (Appendix F.4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
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


@dataclass(slots=True)
class D2ACCIArtifact:
    """The six aligned review artifacts, persisted without benchmark answers being rewritten."""
    question_id: str
    source_alignment: dict[str, Any]
    storage_review: dict[str, Any]
    retrieval_review: dict[str, Any]
    filtered_context_review: dict[str, Any]
    answer_review: dict[str, Any]
    diagnosis: DiagnosticSignal
    fixes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["diagnosis"] = self.diagnosis.to_dict()
        return value


class D2ACCI:
    """Appendix E's aligned, append-only diagnostic governance ledger."""
    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path else None

    def review(self, question_id: str, question: str, gold_answer: str, evidence_source_ids: list[str],
               stored: list[MemoryRecord], retrieved: list[MemoryRecord], filtered: list[MemoryRecord],
               answer: str, classifier: Any, *, fixes: list[dict[str, Any]] | None = None) -> D2ACCIArtifact:
        signal = layer_a_diagnose(question_id, question, gold_answer, evidence_source_ids, stored, retrieved, classifier, filtered=filtered)
        artifact = D2ACCIArtifact(question_id,
            {"source_ids": list(evidence_source_ids), "gold_answer": gold_answer},
            {"stored_ids": [item.id for item in stored]}, {"retrieved_ids": [item.id for item in retrieved]},
            {"filtered_ids": [item.id for item in filtered]}, {"answer": answer}, signal, fixes or [])
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return artifact

    @staticmethod
    def category_report(artifacts: list[D2ACCIArtifact], categories: dict[str, str]) -> dict[str, dict[str, int]]:
        """Produce the category-level non-regression surface used by paired review."""
        report: dict[str, dict[str, int]] = {}
        for artifact in artifacts:
            category = categories.get(artifact.question_id, "unknown")
            bucket = report.setdefault(category, {})
            label = artifact.diagnosis.error_label
            bucket[label] = bucket.get(label, 0) + 1
        return report


def layer_a_diagnose(question_id: str, question: str, gold_answer: str, evidence_source_ids: list[str], stored: list[MemoryRecord], retrieved: list[MemoryRecord], classifier: Any, *, filtered: list[MemoryRecord] | None = None) -> DiagnosticSignal:
    evidence_set = set(evidence_source_ids)
    evidence_facts = [item for item in stored if evidence_set & {source.source_id for source in item.sources}]
    retrieved_evidence = [item for item in retrieved if item.id in {fact.id for fact in evidence_facts}]
    filtered = retrieved if filtered is None else filtered
    base = {"question_id": question_id, "retrieved_ids": [item.id for item in retrieved], "context_ids": [item.id for item in filtered], "answer_evidence": evidence_source_ids}
    if not evidence_facts:
        return DiagnosticSignal(**base, stage_label="ingestion", error_label="ingestion_gap")
    if not retrieved_evidence:
        return DiagnosticSignal(**base, stage_label="retrieval", error_label="retrieval_gap")
    if not any(item.id in {fact.id for fact in retrieved_evidence} for item in filtered):
        return DiagnosticSignal(**base, stage_label="filtering", error_label="filtering_gap")
    chosen = retrieved_evidence[0]
    source_window = [source.source_id for source in chosen.sources]
    response = json_from_completion(classifier.complete([{"role": "system", "content": LAYER_A_PROMPT}, {"role": "user", "content": str({"question": question, "gold_answer": gold_answer, "source_window": source_window, "retrieved_fact": chosen.content})}]))
    label = response.get("label")
    if label == "Full_Coverage":
        return DiagnosticSignal(**base, stage_label="generation", error_label="generation_error")
    if label == "Source_Only":
        return DiagnosticSignal(**base, stage_label="ingestion", error_label="ingestion_gap")
    return DiagnosticSignal(**base, stage_label="uncertain", error_label="uncertain")
