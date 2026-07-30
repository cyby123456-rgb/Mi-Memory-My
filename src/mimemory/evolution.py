"""E2MEND bounded strategy evolution (Appendix F.3--F.6)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .diagnostics import DiagnosticSignal
from .paper_contracts import CRITIC_PROMPT
from .providers import ProviderError, json_from_completion
from .strategy import EvaluationReport, GateDecision, StrategyManager


@dataclass(slots=True)
class CandidateOutcome:
    candidate_id: str
    decision: str
    gate: str
    reason: str
    delta: float = 0.0


class E2MEND:
    """Schema-constrained, evidence-gated strategy search.

    The framework, data schema, and evaluator remain locked. The planner can only
    mutate paths declared in StrategyManager.ALLOWED_PATHS.
    """

    def __init__(self, manager: StrategyManager, planner: Any, critic: Any) -> None:
        self.manager = manager
        self.planner = planner
        self.critic = critic
        self.reverted_directions: set[tuple[str, str]] = set()
        self.reputation: dict[str, float] = {}

    def observe(self, signals: list[DiagnosticSignal]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for signal in signals:
            counts[signal.error_label] = counts.get(signal.error_label, 0) + 1
        return {"root_cause_counts": counts, "signals": [signal.to_dict() for signal in signals]}

    def plan(self, digest: dict[str, Any]) -> dict[str, Any]:
        response = json_from_completion(self.planner.complete([{"role": "system", "content": "You are E2MEND Planner. Select only declared mutable paths and numeric values. Return JSON {\"changes\": {path: number}}."}, {"role": "user", "content": str({"digest": digest, "reputation": self.reputation})}]))
        changes = response.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise ProviderError("E2MEND planner must return a non-empty changes object")
        return changes

    def run_round(
        self,
        signals: list[DiagnosticSignal],
        evaluate: Callable[[dict[str, Any], bool], EvaluationReport],
    ) -> CandidateOutcome:
        digest = self.observe(signals)
        changes = self.plan(digest)
        direction = next(iter(changes.items()))
        if direction in self.reverted_directions:
            return CandidateOutcome("none", "rejected", "novelty", "direction was previously reverted")
        try:
            candidate = self.manager.propose(changes)
        except ValueError as exc:
            return CandidateOutcome("none", "rejected", "structure_or_range", str(exc))
        candidate_id = candidate["artifact_id"]
        # Replay gate: stable-correct samples must not regress before full evaluation.
        probe = evaluate(candidate, True)
        if probe.stable_correct_regressions or not probe.replay_passed or probe.delta <= 0:
            self.reverted_directions.add(direction)
            self.reputation[direction[0]] = self.reputation.get(direction[0], 0.0) - 1.0
            return CandidateOutcome(candidate_id, "rejected", "replay", "targeted screen did not pass", probe.delta)
        review = json_from_completion(self.critic.complete([{"role": "system", "content": CRITIC_PROMPT}, {"role": "user", "content": str({"changes": changes, "reputation": self.reputation, "probe": asdict(probe)})}]))
        if review.get("decision") != "actionable":
            self.reverted_directions.add(direction)
            self.reputation[direction[0]] = self.reputation.get(direction[0], 0.0) - 0.5
            return CandidateOutcome(candidate_id, "rejected", "critic", str(review.get("reason", "critic rejected")), probe.delta)
        full = evaluate(candidate, False)
        decision: GateDecision = self.manager.evaluate(candidate, full)
        if not decision.accepted:
            self.reverted_directions.add(direction)
            self.reputation[direction[0]] = self.reputation.get(direction[0], 0.0) - 1.0
            return CandidateOutcome(candidate_id, "rejected", "paired_acceptance", decision.reason, full.delta)
        self.reputation[direction[0]] = self.reputation.get(direction[0], 0.0) + full.delta
        return CandidateOutcome(candidate_id, "accepted", "five_gate", decision.reason, full.delta)
