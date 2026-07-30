from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory.benchmarks import BenchmarkCase, BenchmarkHarness, LoCoMoAdapter, LongMemEvalAdapter
from mimemory.config import default_strategy
from mimemory.memstack import MemStackModels, MemStackRuntime
from mimemory.storage import LiteMemStore


class QueueChat:
    def __init__(self, values): self.values = list(values)
    def complete(self, messages, *, model=None, temperature=0.0): return json.dumps(self.values.pop(0))


class Vectors:
    def embed(self, inputs, *, model=None): return [[float(len(item)), 1.0] for item in inputs]


class TextAnswer:
    def complete(self, messages, *, model=None, temperature=0.0): return "car"


class BenchmarkTests(unittest.TestCase):
    def test_adapter_normalizes_conversation_and_questions(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "locomo.json"
            path.write_text(json.dumps([{"conversation_id": "c1", "conversation": [{"id": "t1", "role": "user", "content": "Bag is in car"}], "qa": [{"id": "q1", "question": "where", "answer": "car", "evidence": ["t1"]}]}]), encoding="utf-8")
            cases = LoCoMoAdapter().load(path)
            self.assertEqual((cases[0].case_id, cases[0].messages[0]["source_id"]), ("q1", "t1"))

    def test_harness_writes_replayable_results_and_paired_report(self):
        with TemporaryDirectory() as root:
            extractor = QueueChat([{"facts": [{"content": "The bag is in the car.", "source_ids": ["t1"]}], "session_summary": "Bag location", "profile_updates": []}])
            planner = QueueChat([{"intent": "location", "subqueries": ["bag car"], "requires_procedure": False, "requires_temporal_grounding": False}])
            reranker = QueueChat([{"ranking": []}])
            runtime = MemStackRuntime(LiteMemStore(root), default_strategy(), MemStackModels(extractor, planner, reranker, Vectors()))
            case = BenchmarkCase("q1", "u1", "s1", [{"source_id": "t1", "role": "user", "content": "The bag is in the car."}], "Where is bag?", "car", evidence_source_ids=["t1"])
            # Reranker ids are only known after ingestion; construct a wrapper that fills them lazily.
            original = reranker.complete
            def rerank(messages, **kwargs):
                if reranker.values and reranker.values[0]["ranking"] == []:
                    store = runtime.store.list(); reranker.values[0]["ranking"] = [{"id": x.id, "score": 1.0, "reason": "support"} for x in store]
                return original(messages, **kwargs)
            reranker.complete = rerank
            harness = BenchmarkHarness(runtime, TextAnswer(), QueueChat([{"label": "Full_Coverage"}]))
            output = Path(root) / "results.jsonl"; results = harness.run([case], output)
            self.assertTrue(results[0].correct); self.assertTrue(output.exists())
            report = harness.paired_report(results, results)
            self.assertEqual(report["delta"], 0.0)

    def test_longmemeval_adapter_preserves_session_order_for_add_requests(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "longmemeval_s.json"
            path.write_text(json.dumps([{"question_id": "q1", "question_type": "temporal-reasoning", "question": "Where?", "answer": "car", "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"], "haystack_sessions": [[{"role": "user", "content": "Bag is at home"}], [{"role": "assistant", "content": "Bag is in car"}]], "answer_session_ids": ["s2"]}]), encoding="utf-8")
            adapter = LongMemEvalAdapter()
            requests = adapter.add_requests(adapter.load(path)[0])
            self.assertEqual([request["session_id"] for request in requests], ["s1", "s2"])
            self.assertEqual(requests[1]["messages"][0]["timestamp"], 1704153600000)


if __name__ == "__main__": unittest.main()
