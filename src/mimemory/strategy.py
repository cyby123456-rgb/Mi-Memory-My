from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from .config import default_strategy
from .models import utc_now


ALLOWED_PATHS = {
    "extraction.max_facts_per_turn": (1, 64),
    "extraction.dedup_threshold": (0.5, 1.0),
    "retrieval.top_k": (1, 100),
    "retrieval.rrf_k": (1, 200),
    "retrieval.weights.semantic": (0.0, 5.0),
    "retrieval.weights.lexical": (0.0, 5.0),
    "retrieval.weights.subquery": (0.0, 5.0),
    "retrieval.subquery_max_n": (1, 10),
    "retrieval.rerank_top_n": (1, 100),
    "assembly.token_budget": (128, 32768),
    "assembly.profile_fraction": (0.05, 0.5),
    "assembly.min_confidence": (0.0, 1.0),
    "lifecycle.recency_tau_hours": (1.0, 87600.0),
    "lifecycle.importance_tau_hours": (1.0, 175200.0),
    "lifecycle.access_boost": (0.0, 1.0),
    "lifecycle.skip_penalty": (0.0, 1.0),
    "lifecycle.archive_threshold": (0.0, 1.0),
    "lifecycle.source_window": (1, 256),
}
ALLOWED_ENUMS = {
    "retrieval.intent_override": {"auto", "temporal", "entity", "profile"},
    "lifecycle.compression_policy": {"summary_only", "merge_supported", "lossless"},
    "presentation.output_format": {"evidence_first", "concise", "structured"},
}
ALLOWED_BOOLEANS = {
    "features.failure_correction",
    "features.conflict_handling",
    "features.conditional_memory_triggers",
    "features.hypothesis_tree",
}
PROMPT_MARKERS = {"extraction.prompt_template": ("source_ids", "JSON")}


@dataclass(slots=True)
class EvaluationReport:
    baseline_score: float
    candidate_score: float
    category_deltas: dict[str, float] = field(default_factory=dict)
    stable_correct_regressions: int = 0
    replay_passed: bool = True
    full_evaluation_available: bool = True

    @property
    def delta(self) -> float:
        return self.candidate_score - self.baseline_score


@dataclass(slots=True)
class GateDecision:
    accepted: bool
    candidate_id: str
    reason: str
    delta: float
    decided_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyManager:
    """Schema-bounded strategy mutations with gate records and rollback."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = self.root / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"
        self.current_path = self.root / "current.json"
        self.best_path = self.root / "best.json"
        if not self.current_path.exists():
            initial = default_strategy()
            initial["artifact_id"] = "default"
            initial["created_at"] = utc_now()
            self._write_json(self.current_path, initial)
            self._write_json(self.best_path, initial)
            self._write_json(self.checkpoint_root / "default.json", initial)

    def _write_json(self, path: Path, value: dict[str, Any]) -> None:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            temp = Path(stream.name)
        temp.replace(path)

    def current(self) -> dict[str, Any]:
        return json.loads(self.current_path.read_text(encoding="utf-8"))

    def propose(self, changes: dict[str, Any]) -> dict[str, Any]:
        candidate = deepcopy(self.current())
        mutations: list[str] = []
        for path, value in changes.items():
            self._validate_value(path, value)
            target = candidate
            pieces = path.split(".")
            for piece in pieces[:-1]:
                if not isinstance(target.get(piece), dict):
                    raise ValueError(f"strategy path is not mutable: {path}")
                target = target[piece]
            target[pieces[-1]] = value
            mutations.append(path)
        candidate["artifact_id"] = uuid4().hex
        candidate["parent_id"] = self.current().get("artifact_id")
        candidate["mutation_paths"] = mutations
        candidate["created_at"] = utc_now()
        return candidate

    def _validate_value(self, path: str, value: Any) -> None:
        bounds = ALLOWED_PATHS.get(path)
        if bounds is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"strategy value must be numeric: {path}")
            low, high = bounds
            if not low <= value <= high:
                raise ValueError(f"strategy value for {path} must be in [{low}, {high}]")
            return
        if path in ALLOWED_ENUMS:
            if value not in ALLOWED_ENUMS[path]:
                raise ValueError(f"strategy value is not legal for {path}")
            return
        if path in ALLOWED_BOOLEANS:
            if not isinstance(value, bool):
                raise ValueError(f"strategy value must be boolean: {path}")
            return
        if path in PROMPT_MARKERS:
            if not isinstance(value, str) or any(marker not in value for marker in PROMPT_MARKERS[path]):
                raise ValueError(f"prompt-integrity gate failed for {path}")
            return
        raise ValueError(f"out-of-scope strategy path: {path}")

    def evaluate(
        self,
        candidate: dict[str, Any],
        report: EvaluationReport,
        *,
        min_delta: float = 0.0,
        category_tolerance: float = -0.01,
        max_stable_regressions: int = 0,
    ) -> GateDecision:
        candidate_id = str(candidate.get("artifact_id", "unknown"))
        reasons: list[str] = []
        if report.delta < min_delta:
            reasons.append(f"delta {report.delta:.6f} below {min_delta:.6f}")
        regressed = {name: delta for name, delta in report.category_deltas.items() if delta < category_tolerance}
        if regressed:
            reasons.append(f"category regression: {regressed}")
        if report.stable_correct_regressions > max_stable_regressions:
            reasons.append("stable-correct replay regressed")
        if not report.replay_passed:
            reasons.append("replay gate failed")
        decision = GateDecision(not reasons, candidate_id, "; ".join(reasons) or "all gates passed", report.delta)
        event = {
            "candidate": candidate,
            "report": asdict(report),
            "decision": decision.to_dict(),
        }
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        if decision.accepted:
            candidate["gate_record"] = decision.to_dict()
            candidate["rollback_key"] = candidate.get("parent_id", "default")
            self._write_json(self.current_path, candidate)
            self._write_json(self.best_path, candidate)
            self._write_json(self.checkpoint_root / f"{candidate_id}.json", candidate)
        return decision

    def rollback(self) -> dict[str, Any]:
        current = self.current()
        rollback_key = str(current.get("rollback_key") or current.get("parent_id") or "default")
        checkpoint = self.checkpoint_root / f"{rollback_key}.json"
        if not checkpoint.exists():
            checkpoint = self.checkpoint_root / "default.json"
        restored = json.loads(checkpoint.read_text(encoding="utf-8"))
        self._write_json(self.current_path, restored)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "rollback", "at": utc_now(), "artifact_id": restored["artifact_id"]}) + "\n")
        return restored
