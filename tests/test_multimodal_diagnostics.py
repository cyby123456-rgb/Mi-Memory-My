from __future__ import annotations

import json
import unittest

from mimemory.diagnostics import D2ACCI, layer_a_diagnose
from mimemory.models import MemoryRecord, SourceRef
from mimemory.multimodal import (
    DeviceEvent, FusionSession, IKBRouter, ImageKnowledgeBase, ImageObservation,
    MemFuse, MemSense,
)


class FakeChat:
    def __init__(self, response): self.response = response
    def complete(self, messages, *, model=None, temperature=0.0): return json.dumps(self.response)


class MultimodalAndDiagnosticTests(unittest.TestCase):
    def test_memsense_five_pass_contract_and_ikb(self):
        sense = MemSense(FakeChat({"facts": [{"content": "A blue bag contains shoes.", "category": "equipment", "name": "bag", "caption": "blue bag", "confidence": 0.9, "source_ids": ["img-1"]}]}))
        facts = sense.build_ikb(ImageObservation("image-1", "https://example/image", ["img-1"], session="s1"))
        ikb = ImageKnowledgeBase(); ikb.add(facts)
        self.assertEqual(ikb.select("blue bag")[0].image_id, "image-1")
        self.assertEqual([artifact.pass_name for artifact in sense.pass_artifacts], ["conversation_name", "category", "temporal", "related_fact", "uncertainty"])

    def test_ikb_router_uses_indexes_before_bounded_residual_vlm(self):
        sense = MemSense(FakeChat({"facts": [{"content": "A blue bag contains shoes.", "category": "equipment", "name": "bag", "caption": "blue bag", "confidence": 0.9, "source_ids": ["img-1"]}]}))
        fact = sense.build_ikb(ImageObservation("image-1", "data:image/png;base64,eA==", ["img-1"], session="s1", date="2026-01-01"))[0]
        ikb = ImageKnowledgeBase(); ikb.add([fact])
        route = IKBRouter(FakeChat({"intent": "VS", "category": "equipment", "session": "s1", "date": "2026-01-01", "image_ids": ["declared"], "residual_query": "bag"}))
        result = route.route("which bag", ikb, graph_image_ids=["g1"])
        self.assertEqual(result.intent, "VS")
        self.assertEqual(result.facts, (fact,))
        self.assertEqual(result.image_ids, ("declared", "g1", "image-1"))
        self.assertFalse(result.used_residual_vlm)

    def test_memfuse_requires_auditable_event_provenance(self):
        fuse = MemFuse(FakeChat({"summary": "The bag moved to the car.", "edge_type": "CAUSES", "source_event_ids": ["camera", "car"], "confidence": 0.8, "provenance": ["camera", "car"]}))
        fused = fuse.fuse([DeviceEvent("camera", "Bag left home", "2026-01-01T10:00:00+00:00", "cam", ["camera"]), DeviceEvent("car", "Bag arrived in car", "2026-01-01T10:05:00+00:00", "car", ["car"])])
        self.assertEqual(fused.edge_type, "CAUSES")
        self.assertEqual(fused.source_event_ids, ["camera", "car"])
        self.assertEqual(len(fuse.graph.atomic_events), 2)
        self.assertEqual([edge.relation for edge in fuse.graph.edges], ["BELONG", "BELONG", "CAUSES"])

    def test_fusion_session_evicts_only_its_transient_zone(self):
        session = FusionSession(max_events_per_zone=1)
        one = DeviceEvent("one", "one", "2026-01-01T10:00:00+00:00", "cam", ["one"])
        two = DeviceEvent("two", "two", "2026-01-01T10:01:00+00:00", "cam", ["two"])
        self.assertEqual(session.add(one, "perception"), [])
        self.assertEqual(session.add(two, "perception"), [one])
        self.assertEqual(session.ordered_events(), [two])

    def test_layer_a_covers_deterministic_and_llm_branches(self):
        fact = MemoryRecord(content="Bag is in car", sources=[SourceRef("turn-1")])
        ingestion = layer_a_diagnose("q1", "where", "car", ["turn-1"], [], [], FakeChat({"label": "Uncertain"}))
        retrieval = layer_a_diagnose("q1", "where", "car", ["turn-1"], [fact], [], FakeChat({"label": "Uncertain"}))
        generation = layer_a_diagnose("q1", "where", "car", ["turn-1"], [fact], [fact], FakeChat({"label": "Full_Coverage"}))
        self.assertEqual((ingestion.error_label, retrieval.error_label, generation.error_label), ("ingestion_gap", "retrieval_gap", "generation_error"))

    def test_d2acci_records_filtered_context_gap(self):
        fact = MemoryRecord(content="Bag is in car", sources=[SourceRef("turn-1")])
        review = D2ACCI().review("q1", "where", "car", ["turn-1"], [fact], [fact], [], "unknown", FakeChat({"label": "Uncertain"}))
        self.assertEqual(review.diagnosis.error_label, "filtering_gap")
        self.assertEqual(review.filtered_context_review["filtered_ids"], [])


if __name__ == "__main__": unittest.main()
