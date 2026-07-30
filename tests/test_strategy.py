from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory.strategy import EvaluationReport, StrategyManager


class StrategyManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.manager = StrategyManager(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_schema_rejects_framework_changes_and_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-scope"):
            self.manager.propose({"framework.storage": "postgres"})
        with self.assertRaisesRegex(ValueError, "must be in"):
            self.manager.propose({"retrieval.top_k": 1000})

    def test_candidate_requires_positive_non_regressive_evidence(self) -> None:
        candidate = self.manager.propose({"retrieval.top_k": 20})
        rejected = self.manager.evaluate(
            candidate,
            EvaluationReport(0.8, 0.82, {"temporal": -0.03}),
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(self.manager.current()["artifact_id"], "default")

        candidate = self.manager.propose({"retrieval.top_k": 16})
        accepted = self.manager.evaluate(
            candidate,
            EvaluationReport(0.8, 0.83, {"temporal": 0.01, "profile": 0.0}),
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(self.manager.current()["retrieval"]["top_k"], 16)

    def test_rollback_restores_parent_checkpoint(self) -> None:
        candidate = self.manager.propose({"retrieval.top_k": 16})
        self.manager.evaluate(candidate, EvaluationReport(0.8, 0.9))
        restored = self.manager.rollback()
        self.assertEqual(restored["artifact_id"], "default")
        self.assertEqual(restored["retrieval"]["top_k"], 12)


if __name__ == "__main__":
    unittest.main()

