"""Reproducible benchmark adapters and the shared Mi-Memory evaluation harness.

Adapters normalize public benchmark exports into one evidence-complete contract.
The harness keeps the memory runtime, answerer, judge, and diagnostics explicit so
the reported configuration can be replayed without test-set-specific code paths.
"""

from __future__ import annotations

import json
import re
import ast
import csv
import time
from datetime import UTC, datetime
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


@dataclass(slots=True)
class ResourceLedger:
    """Non-sensitive, appendable accounting for a benchmark replay."""

    case_id: str
    elapsed_seconds: float
    ingested_records: int
    retrieved_records: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


class LoCoMoAdapter(PublicDatasetAdapter):
    """Adapter for SNAP's released `locomo10.json` conversation/session schema."""
    name = "locomo"

    @staticmethod
    def _timestamp(value: Any) -> int | None:
        if not isinstance(value, str):
            return None
        try:
            return int(datetime.strptime(value, "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            return None

    def load(self, path: str | Path) -> list[BenchmarkCase]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("LoCoMo input must be an array")
        cases: list[BenchmarkCase] = []
        for item_index, item in enumerate(raw):
            if not isinstance(item, dict) or not isinstance(item.get("conversation"), dict) or not isinstance(item.get("qa"), list):
                raise ValueError("LoCoMo record is missing conversation or qa")
            conversation = item["conversation"]
            speaker_a = conversation.get("speaker_a")
            if not isinstance(speaker_a, str):
                raise ValueError("LoCoMo conversation is missing speaker_a")
            messages: list[dict[str, Any]] = []
            for session_number in range(1, 36):
                session = conversation.get(f"session_{session_number}")
                if session is None:
                    continue
                if not isinstance(session, list):
                    raise ValueError("LoCoMo session must be a turn array")
                timestamp = self._timestamp(conversation.get(f"session_{session_number}_date_time"))
                for turn_index, turn in enumerate(session):
                    if not isinstance(turn, dict) or not isinstance(turn.get("text"), str) or not isinstance(turn.get("speaker"), str):
                        raise ValueError("LoCoMo turn must contain speaker and text")
                    messages.append({"source_id": str(turn.get("dia_id", f"s{session_number}:{turn_index}")),
                        "role": "user" if turn["speaker"] == speaker_a else "assistant", "content": turn["text"],
                        "timestamp": timestamp, "session_id": f"session-{session_number}"})
            user_id = str(item.get("sample_id", f"locomo-{item_index}"))
            for question_index, question in enumerate(item["qa"]):
                if not isinstance(question, dict) or not isinstance(question.get("question"), str) or not isinstance(question.get("answer"), str):
                    raise ValueError("LoCoMo question must contain question and answer")
                cases.append(BenchmarkCase(case_id=f"{user_id}:{question_index}", user_id=user_id,
                    session_id=user_id, messages=messages, query=question["question"], answer=question["answer"],
                    evidence_source_ids=[str(value) for value in question.get("evidence", []) if isinstance(value, str)],
                    category=str(question.get("category", "unknown"))))
        return cases


class PersonaMemV2Adapter(PublicDatasetAdapter):
    """Adapter for PersonaMem-v2's released benchmark CSV and local history files."""
    name = "personamem_v2"

    @staticmethod
    def _decode(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)

    @staticmethod
    def _history_path(csv_path: Path, value: str) -> Path:
        """Resolve both dataset-root and CSV-relative official link layouts."""

        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        for base in (csv_path.parent, *csv_path.parents):
            resolved = base / candidate
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(f"PersonaMem-v2 history file does not exist: {value}")

    def load(self, path: str | Path) -> list[BenchmarkCase]:
        csv_path = Path(path)
        cases: list[BenchmarkCase] = []
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                raw_query = row.get("user_query", "")
                try:
                    query_object = self._decode(raw_query)
                except (ValueError, SyntaxError):
                    query_object = {"content": raw_query}
                query = query_object.get("content", "") if isinstance(query_object, dict) else str(query_object)
                if not isinstance(query, str) or not query.strip():
                    continue
                history_value = row.get("chat_history_32k_link") or row.get("chat_history_link") or ""
                try:
                    history = self._decode(history_value)
                except (ValueError, SyntaxError):
                    history_path = self._history_path(csv_path, history_value)
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(history, dict):
                    history = history.get("conversations", [])
                messages = _messages(history)
                correct = row.get("correct_answer", "")
                try:
                    incorrect = self._decode(row.get("incorrect_answers", "[]"))
                except (ValueError, SyntaxError):
                    incorrect = []
                options = [str(correct), *[str(item) for item in incorrect if isinstance(item, str)]]
                persona_id = str(row.get("persona_id", index))
                cases.append(BenchmarkCase(case_id=f"personamem:{persona_id}:{index}", user_id=f"personamem:{persona_id}",
                    session_id=f"personamem:{persona_id}", messages=messages, query=query, answer=str(correct), options=options,
                    category=str(row.get("pref_type", "unknown"))))
        return cases
class LongMemEvalAdapter(PublicDatasetAdapter):
    """Adapter for the official LongMemEval-S/M session schema.

    A history session maps to one synchronous Add request.  A question is
    searched only after its ordered session history has been admitted.
    """
    name = "longmemeval"

    @staticmethod
    def _timestamp_milliseconds(value: Any) -> int | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value if value > 10_000_000_000 else value * 1000)
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            match = re.fullmatch(r"(\d{4}/\d{2}/\d{2})\s+\([A-Za-z]{3}\)\s+(\d{2}:\d{2})", value)
            if not match:
                return None
            parsed = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y/%m/%d %H:%M")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)

    def load(self, path: str | Path) -> list[BenchmarkCase]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("LongMemEval input must be an array")
        cases: list[BenchmarkCase] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            question_id = row.get("question_id")
            sessions, session_ids, dates = row.get("haystack_sessions"), row.get("haystack_session_ids"), row.get("haystack_dates")
            if not isinstance(question_id, str) or not isinstance(sessions, list) or not isinstance(session_ids, list) or not isinstance(dates, list):
                raise ValueError("LongMemEval record is missing question/session fields")
            messages: list[dict[str, Any]] = []
            for session_index, session in enumerate(sessions):
                if not isinstance(session, list):
                    raise ValueError("LongMemEval session must be a turn array")
                session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
                timestamp = self._timestamp_milliseconds(dates[session_index]) if session_index < len(dates) else None
                for turn_index, turn in enumerate(session):
                    if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"} or not isinstance(turn.get("content"), str):
                        raise ValueError("LongMemEval turn must have user/assistant role and content")
                    messages.append({"source_id": f"{session_id}:{turn_index}", "role": turn["role"], "content": turn["content"], "timestamp": timestamp, "session_id": session_id})
            cases.append(BenchmarkCase(case_id=question_id, user_id=question_id, session_id=question_id, messages=messages,
                query=str(row.get("question", "")), answer=str(row.get("answer", "")),
                evidence_source_ids=[str(item) for item in row.get("answer_session_ids", []) if isinstance(item, (str, int))],
                category=str(row.get("question_type", "unknown"))))
        return cases

    def add_requests(self, case: BenchmarkCase) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for message in case.messages:
            grouped.setdefault(str(message["session_id"]), []).append(message)
        requests: list[dict[str, Any]] = []
        for sequence, (session_id, messages) in enumerate(grouped.items()):
            requests.append({"request_id": f"longmemeval:{case.case_id}:{sequence}", "user_id": case.user_id,
                "session_id": session_id, "messages": [{"role": item["role"], "content": item["content"], "timestamp": item["timestamp"]} for item in messages]})
        return requests


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


class BenchmarkHarness:
    def __init__(self, runtime: MemStackRuntime, answerer: AnswerProvider, classifier: AnswerProvider) -> None:
        self.runtime = runtime
        self.answerer = answerer
        self.classifier = classifier

    def run(self, cases: Iterable[BenchmarkCase], output_path: str | Path | None = None,
            resource_path: str | Path | None = None) -> list[CaseResult]:
        results: list[CaseResult] = []
        resources: list[ResourceLedger] = []
        admitted: dict[tuple[str, str], list[MemoryRecord]] = {}
        for case in cases:
            started = time.monotonic()
            admission_key = (case.user_id, case.session_id)
            stored = admitted.get(admission_key)
            if stored is None:
                stored = self.runtime.ingest(case.messages, user_id=case.user_id, session_id=case.session_id)
                admitted[admission_key] = stored
            context = self.runtime.retrieve(case.query, user_id=case.user_id)
            answer_prompt = {"question": case.query, "options": case.options, "evidence": context.text, "instruction": "Answer only from evidence. Return the answer directly."}
            predicted = self.answerer.complete([{"role": "user", "content": str(answer_prompt)}], temperature=0.0).strip()
            diagnostic = layer_a_diagnose(case.case_id, case.query, case.answer, case.evidence_source_ids, stored, [hit.record for hit in context.evidence], self.classifier)
            results.append(CaseResult(case.case_id, predicted, case.answer, normalize_answer(predicted) == normalize_answer(case.answer), [hit.record.id for hit in context.evidence], context.trace.query_id, diagnostic.to_dict()))
            resources.append(ResourceLedger(case.case_id, round(time.monotonic() - started, 6), len(stored), len(context.evidence)))
        if output_path:
            path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
            # Keep answer keys in the released dataset, not in replay artifacts.
            path.write_text("\n".join(json.dumps({
                "case_id": item.case_id,
                "predicted_answer": item.predicted_answer,
                "correct": item.correct,
                "retrieved_ids": item.retrieved_ids,
                "trace_id": item.trace_id,
                "diagnostic": item.diagnostic,
            }, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
        if resource_path:
            path = Path(resource_path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in resources) + "\n", encoding="utf-8")
        return results

    @staticmethod
    def report(results: Iterable[CaseResult], categories: dict[str, str] | None = None) -> dict[str, Any]:
        """Return answer and diagnostic aggregates without publishing source data."""

        rows = list(results)
        category_map = categories or {}
        grouped: dict[str, list[CaseResult]] = {}
        for row in rows:
            grouped.setdefault(category_map.get(row.case_id, "unknown"), []).append(row)
        return {
            "cases": len(rows),
            "answer_exact_match": sum(row.correct for row in rows) / len(rows) if rows else 0.0,
            "categories": {
                name: {"cases": len(items), "answer_exact_match": sum(item.correct for item in items) / len(items) if items else 0.0}
                for name, items in sorted(grouped.items())
            },
            "diagnostics": {
                label: sum(row.diagnostic.get("error_label") == label for row in rows)
                for label in sorted({row.diagnostic.get("error_label", "unknown") for row in rows})
            },
        }

    @staticmethod
    def paired_report(baseline: list[CaseResult], candidate: list[CaseResult]) -> dict[str, Any]:
        base = {item.case_id: item for item in baseline}; cand = {item.case_id: item for item in candidate}
        shared = sorted(set(base) & set(cand)); improved = [key for key in shared if not base[key].correct and cand[key].correct]; regressed = [key for key in shared if base[key].correct and not cand[key].correct]
        return {"count": len(shared), "baseline_accuracy": sum(base[key].correct for key in shared) / len(shared) if shared else 0.0, "candidate_accuracy": sum(cand[key].correct for key in shared) / len(shared) if shared else 0.0, "delta": (len(improved) - len(regressed)) / len(shared) if shared else 0.0, "improved": improved, "regressed": regressed}
