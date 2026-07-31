from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory.benchmarks import BenchmarkCase, BenchmarkHarness, LoCoMoAdapter, LongMemEvalAdapter, PersonaMemV2Adapter
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
            path.write_text(json.dumps([{"sample_id": "c1", "conversation": {"speaker_a": "Alice", "speaker_b": "Bob", "session_1_date_time": "1:56 pm on 08 May, 2023", "session_1": [{"dia_id": "D1:1", "speaker": "Alice", "text": "Bag is in car"}]}, "qa": [{"question": "where", "answer": "car", "evidence": ["D1:1"], "category": 2}]}]), encoding="utf-8")
            cases = LoCoMoAdapter().load(path)
            self.assertEqual((cases[0].case_id, cases[0].messages[0]["source_id"]), ("c1:0", "D1:1"))
            self.assertEqual(cases[0].messages[0]["timestamp"], 1683554160000)

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

    def test_harness_reuses_one_admission_and_writes_resource_ledger(self):
        with TemporaryDirectory() as root:
            extractor = QueueChat([{"facts": [{"content": "The bag is in the car.", "source_ids": ["t1"]}], "session_summary": "Bag location", "profile_updates": []}])
            planner = QueueChat([{"intent": "location", "subqueries": ["bag car"], "requires_procedure": False, "requires_temporal_grounding": False}, {"intent": "location", "subqueries": ["bag"], "requires_procedure": False, "requires_temporal_grounding": False}])
            reranker = QueueChat([{"ranking": []}, {"ranking": []}])
            runtime = MemStackRuntime(LiteMemStore(root), default_strategy(), MemStackModels(extractor, planner, reranker, Vectors()))
            original = reranker.complete
            def rerank(messages, **kwargs):
                if reranker.values and reranker.values[0]["ranking"] == []:
                    reranker.values[0]["ranking"] = [{"id": x.id, "score": 1.0, "reason": "support"} for x in runtime.store.list()]
                return original(messages, **kwargs)
            reranker.complete = rerank
            cases = [BenchmarkCase(f"q{index}", "u1", "s1", [{"source_id": "t1", "role": "user", "content": "The bag is in the car."}], "Where is bag?", "car", evidence_source_ids=["t1"], category="location") for index in range(2)]
            resources = Path(root) / "resources.jsonl"
            results = BenchmarkHarness(runtime, TextAnswer(), QueueChat([{"label": "Full_Coverage"}, {"label": "Full_Coverage"}])).run(cases, resource_path=resources)
            self.assertEqual(len(extractor.values), 0)
            self.assertEqual(len(resources.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(BenchmarkHarness.report(results, {"q0": "location", "q1": "location"})["categories"]["location"]["cases"], 2)

    def test_longmemeval_adapter_preserves_session_order_for_add_requests(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "longmemeval_s.json"
            path.write_text(json.dumps([{"question_id": "q1", "question_type": "temporal-reasoning", "question": "Where?", "answer": "car", "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"], "haystack_sessions": [[{"role": "user", "content": "Bag is at home"}], [{"role": "assistant", "content": "Bag is in car"}]], "answer_session_ids": ["s2"]}]), encoding="utf-8")
            adapter = LongMemEvalAdapter()
            requests = adapter.add_requests(adapter.load(path)[0])
            self.assertEqual([request["session_id"] for request in requests], ["s1", "s2"])
            self.assertEqual(requests[1]["messages"][0]["timestamp"], 1704153600000)

    def test_personamem_adapter_reads_released_csv_contract(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "benchmark.csv"
            path.write_text(
                "persona_id,user_query,correct_answer,incorrect_answers,chat_history_32k_link,pref_type\n"
                '7,"{""role"": ""user"", ""content"": ""What should I cook?""}",pasta,"[""salad""]","[{""role"": ""user"", ""content"": ""I like pasta""}]",food\n',
                encoding="utf-8",
            )
            case = PersonaMemV2Adapter().load(path)[0]
            self.assertEqual((case.user_id, case.query, case.options), ("personamem:7", "What should I cook?", ["pasta", "salad"]))

    def test_personamem_adapter_resolves_dataset_root_history_paths(self):
        with TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "history").mkdir()
            (root_path / "history" / "p1.json").write_text('[{"role": "user", "content": "I like pasta"}]', encoding="utf-8")
            benchmark = root_path / "benchmark" / "multimodal"
            benchmark.mkdir(parents=True)
            path = benchmark / "benchmark.csv"
            path.write_text(
                "persona_id,user_query,correct_answer,incorrect_answers,chat_history_32k_link,pref_type\n"
                '7,"{""role"": ""user"", ""content"": ""What should I cook?""}",pasta,"[""salad""]",history/p1.json,food\n',
                encoding="utf-8",
            )
            case = PersonaMemV2Adapter().load(path)[0]
            self.assertEqual(case.messages[0]["content"], "I like pasta")


if __name__ == "__main__": unittest.main()
