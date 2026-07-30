"""Reproducible benchmark adapters and the shared Mi-Memory evaluation harness.

Adapters normalize public benchmark exports into one evidence-complete contract.
The harness keeps the memory runtime, answerer, judge, and diagnostics explicit so
the reported configuration can be replayed without test-set-specific code paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .diagnostics import DiagnosticSignal, layer_a_diagnose
from .memstack import MemStackRuntime
from .models import MemoryRecord


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    user_id: str
    session_id: str
    messages: list[dict[str, Any]]
    query: str
    answer: str
    options: list[str] = field(default_factory=list)
    evidence_source_ids: list[str] = field(default_factory=list)
    category: str = "unknown"


@dataclass(slots=True)
class CaseResult:
    case_id: str
    predicted_answer: str
    gold_answer: str
    correct: bool
    retrieved_ids: list[str]
    trace_id: str
    diagnostic: dict[str, Any]


class AnswerProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float = 0.0) -> str: ...


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    messages = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        content = item.get("content", item.get("text", item.get("message", "")))
        role = item.get("role", item.get("speaker", "user"))
        if isinstance(content, str) and content.strip():
            messages.append({"source_id": str(item.get("id", f"turn-{index}")), "role": "assistant" if str(role).lower() in {"assistant", "bot", "gpt"} else "user", "content": content, "timestamp": item.get("timestamp")})
    return messages


class PublicDatasetAdapter:
    name = "generic"

    def load(self, path: str | Path) -> list[BenchmarkCase]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = raw.get("data", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError(f"{self.name} input must contain an array")
        cases: list[BenchmarkCase] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            conversation = row.get("messages", row.get("conversation", row.get("dialogue", [])))
            questions = row.get("questions", row.get("qa", [row]))
            if not isinstance(questions, list):
                questions = [row]
            for question_index, question in enumerate(questions):
                if not isinstance(question, dict):
                    continue
                query = question.get("query", question.get("question", ""))
                answer = question.get("answer", question.get("gold_answer", ""))
                if not isinstance(query, str) or not isinstance(answer, str):
                    continue
                case_id = str(question.get("id", f"{self.name}-{index}-{question_index}"))
                evidence = question.get("evidence_source_ids", question.get("evidence", question.get("supporting_ids", [])))
                if not isinstance(evidence, list):
                    evidence = []
                cases.append(BenchmarkCase(
                    case_id=case_id, user_id=str(row.get("user_id", row.get("conversation_id", f"{self.name}-user-{index}"))),
                    session_id=str(row.get("session_id", row.get("id", f"{self.name}-session-{index}"))),
                    messages=_messages(conversation), query=query, answer=answer,
                    options=[str(item) for item in question.get("options", []) if isinstance(item, str)],
                    evidence_source_ids=[str(item) for item in evidence], category=str(question.get("category", question.get("type", "unknown"))),
                ))
        return cases


class LoCoMoAdapter(PublicDatasetAdapter): name = "locomo"
class PersonaMemV2Adapter(PublicDatasetAdapter): name = "personamem_v2"
class LongMemEvalAdapter(PublicDatasetAdapter): name = "longmemeval"


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


class BenchmarkHarness:
    def __init__(self, runtime: MemStackRuntime, answerer: AnswerProvider, classifier: AnswerProvider) -> None:
        self.runtime = runtime
        self.answerer = answerer
        self.classifier = classifier

    def run(self, cases: Iterable[BenchmarkCase], output_path: str | Path | None = None) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in cases:
            stored = self.runtime.ingest(case.messages, user_id=case.user_id, session_id=case.session_id)
            context = self.runtime.retrieve(case.query, user_id=case.user_id)
            answer_prompt = {"question": case.query, "options": case.options, "evidence": context.text, "instruction": "Answer only from evidence. Return the answer directly."}
            predicted = self.answerer.complete([{"role": "user", "content": str(answer_prompt)}], temperature=0.0).strip()
            diagnostic = layer_a_diagnose(case.case_id, case.query, case.answer, case.evidence_source_ids, stored, [hit.record for hit in context.evidence], self.classifier)
            results.append(CaseResult(case.case_id, predicted, case.answer, normalize_answer(predicted) == normalize_answer(case.answer), [hit.record.id for hit in context.evidence], context.trace.query_id, diagnostic.to_dict()))
        if output_path:
            path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
        return results

    @staticmethod
    def paired_report(baseline: list[CaseResult], candidate: list[CaseResult]) -> dict[str, Any]:
        base = {item.case_id: item for item in baseline}; cand = {item.case_id: item for item in candidate}
        shared = sorted(set(base) & set(cand)); improved = [key for key in shared if not base[key].correct and cand[key].correct]; regressed = [key for key in shared if base[key].correct and not cand[key].correct]
        return {"count": len(shared), "baseline_accuracy": sum(base[key].correct for key in shared) / len(shared) if shared else 0.0, "candidate_accuracy": sum(cand[key].correct for key in shared) / len(shared) if shared else 0.0, "delta": (len(improved) - len(regressed)) / len(shared) if shared else 0.0, "improved": improved, "regressed": regressed}
