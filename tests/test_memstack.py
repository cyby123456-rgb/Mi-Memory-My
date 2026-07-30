from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory.config import default_strategy
from mimemory.memstack import MemStackModels, MemStackRuntime
from mimemory.storage import LiteMemStore


class FakeChat:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def complete(self, messages, *, model=None, temperature=0.0) -> str:
        if not self.responses:
            raise AssertionError("unexpected model call")
        return json.dumps(self.responses.pop(0))


class FakeEmbeddings:
    def embed(self, inputs, *, model=None):
        return [[float(len(text)), float(sum(ord(char) for char in text) % 97), 1.0] for text in inputs]


class MemStackRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.store = LiteMemStore(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_llm_extraction_embedding_rrf_rerank_and_trace(self) -> None:
        extractor = FakeChat([
            {
                "facts": [{"content": "The blue bag is in the car.", "type": "location", "source_ids": ["turn-1"], "confidence": 0.98, "importance": 0.8, "entities": ["blue bag", "car"], "temporal": None}],
                "session_summary": "The user stored the blue bag in the car.",
                "profile_updates": [],
            }
        ])
        planner = FakeChat([{"intent": "location", "subqueries": ["blue bag car"], "requires_procedure": False, "requires_temporal_grounding": False}])
        reranker = FakeChat([{"ranking": [{"id": "PLACEHOLDER", "score": 0.99, "reason": "exact location"}, {"id": "PLACEHOLDER2", "score": 0.6, "reason": "summary"}]}])
        runtime = MemStackRuntime(self.store, default_strategy(), MemStackModels(extractor, planner, reranker, FakeEmbeddings()))
        records = runtime.ingest([{"source_id": "turn-1", "role": "user", "content": "My blue bag is in the car."}], user_id="u1", session_id="s1")
        # Supply the generated stable ids to the mock reranker after ingestion.
        reranker.responses[0]["ranking"][0]["id"] = records[0].id
        reranker.responses[0]["ranking"][1]["id"] = records[1].id
        bundle = runtime.retrieve("Where is the blue bag?", user_id="u1")
        self.assertEqual(bundle.evidence[0].record.content, "The blue bag is in the car.")
        self.assertIn("semantic", bundle.trace.channel_results)
        self.assertIn("lexical", bundle.trace.channel_results)
        self.assertIn("subquery", bundle.trace.channel_results)
        self.assertTrue((Path(self.temp.name) / "traces" / f"{bundle.trace.query_id}.json").exists())


if __name__ == "__main__":
    unittest.main()

