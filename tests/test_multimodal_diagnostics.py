from __future__ import annotations

import json
import unittest

from mimemory.diagnostics import layer_a_diagnose
from mimemory.models import MemoryRecord, SourceRef
from mimemory.multimodal import DeviceEvent, ImageKnowledgeBase, ImageObservation, MemFuse, MemSense


class FakeChat:
    def __init__(self, response): self.response = response
    def complete(self, messages, *, model=None, temperature=0.0): return json.dumps(self.response)


class MultimodalAndDiagnosticTests(unittest.TestCase):
    def test_memsense_five_pass_contract_and_ikb(self):
        sense = MemSense(FakeChat({"facts": [{"content": "A blue bag contains shoes.", "category": "equipment", "name": "bag", "caption": "blue bag", "confidence": 0.9, "source_ids": ["img-1"]}]}))
        facts = sense.build_ikb(ImageObservation("image-1", "https://example/image", ["img-1"], session="s1"))
        ikb = ImageKnowledgeBase(); ikb.add(facts)
        self.assertEqual(ikb.select("blue bag")[0].image_id, "image-1")

    def test_memfuse_requires_auditable_event_provenance(self):
        fuse = MemFuse(FakeChat({"summary": "The bag moved to the car.", "edge_type": "CAUSES", "source_event_ids": ["camera", "car"], "confidence": 0.8, "provenance": ["camera", "car"]}))
        fused = fuse.fuse([DeviceEvent("camera", "Bag left home", "2026-01-01T10:00:00+00:00", "cam", ["camera"]), DeviceEvent("car", "Bag arrived in car", "2026-01-01T10:05:00+00:00", "car", ["car"])])
        self.assertEqual(fused.edge_type, "CAUSES")
        self.assertEqual(fused.source_event_ids, ["camera", "car"])

    def test_layer_a_covers_deterministic_and_llm_branches(self):
        fact = MemoryRecord(content="Bag is in car", sources=[SourceRef("turn-1")])
        ingestion = layer_a_diagnose("q1", "where", "car", ["turn-1"], [], [], FakeChat({"label": "Uncertain"}))
        retrieval = layer_a_diagnose("q1", "where", "car", ["turn-1"], [fact], [], FakeChat({"label": "Uncertain"}))
        generation = layer_a_diagnose("q1", "where", "car", ["turn-1"], [fact], [fact], FakeChat({"label": "Full_Coverage"}))
        self.assertEqual((ingestion.error_label, retrieval.error_label, generation.error_label), ("ingestion_gap", "retrieval_gap", "generation_error"))


if __name__ == "__main__": unittest.main()
