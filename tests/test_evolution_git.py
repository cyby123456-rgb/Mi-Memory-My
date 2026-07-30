from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory.diagnostics import DiagnosticSignal
from mimemory.evolution import E2MEND
from mimemory.git_provenance import GitProvenance
from mimemory.strategy import EvaluationReport, StrategyManager


class FakeChat:
    def __init__(self, value): self.value = value
    def complete(self, messages, *, model=None, temperature=0.0): return json.dumps(self.value)


class EvolutionAndGitTests(unittest.TestCase):
    def test_e2mend_accepts_only_after_probe_critic_and_paired_gate(self):
        with TemporaryDirectory() as root:
            engine = E2MEND(StrategyManager(Path(root) / "strategies"), FakeChat({"changes": {"retrieval.top_k": 16}}), FakeChat({"decision": "actionable", "reason": "localized retrieval gap"}))
            signal = DiagnosticSignal("q", [], [], [], "retrieval", "retrieval_gap")
            outcome = engine.run_round([signal], lambda candidate, probe: EvaluationReport(0.8, 0.82 if probe else 0.85, {"temporal": 0.0}))
            self.assertEqual((outcome.decision, outcome.gate), ("accepted", "five_gate"))
            self.assertTrue(outcome.gate_records["acceptance"])
            self.assertTrue((Path(root) / "strategies" / "e2mend-artifacts.jsonl").exists())

    def test_e2mend_rejects_replayed_regression(self):
        with TemporaryDirectory() as root:
            engine = E2MEND(StrategyManager(Path(root) / "strategies"), FakeChat({"changes": {"retrieval.top_k": 16}}), FakeChat({"decision": "actionable", "reason": "ok"}))
            signal = DiagnosticSignal("q", [], [], [], "retrieval", "retrieval_gap")
            outcome = engine.run_round([signal], lambda candidate, probe: EvaluationReport(0.8, 0.81, stable_correct_regressions=1))
            self.assertEqual((outcome.decision, outcome.gate), ("rejected", "replay"))

    def test_e2mend_rolls_back_on_checkpoint_drift(self):
        with TemporaryDirectory() as root:
            manager = StrategyManager(Path(root) / "strategies")
            candidate = manager.propose({"retrieval.top_k": 16})
            manager.evaluate(candidate, EvaluationReport(0.8, 0.9))
            engine = E2MEND(manager, FakeChat({"changes": {"retrieval.top_k": 18}}), FakeChat({"decision": "actionable", "reason": "ok"}), drift_tolerance=0.01)
            self.assertTrue(engine.rollback_on_drift(0.7, 0.9))
            self.assertEqual(manager.current()["artifact_id"], "default")

    def test_litemem_git_provenance_commits_and_diffs(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "memory.md"; path.write_text("one\n", encoding="utf-8")
            git = GitProvenance(root)
            self.assertIsNotNone(git.commit("capture memory"))
            path.write_text("two\n", encoding="utf-8")
            self.assertIsNotNone(git.commit("correct memory"))
            self.assertIn("-one", git.diff())
            self.assertIn("+two", git.diff())


if __name__ == "__main__": unittest.main()
