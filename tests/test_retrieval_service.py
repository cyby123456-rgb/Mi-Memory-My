from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory import FusedEvent, MemoryLayer, MemoryService, MemoryStatus, PerceptionFact


class MemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.service = MemoryService.local(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_ingest_recall_and_trace(self) -> None:
        records = self.service.ingest_text(
            "The blue training bag is in the car. It contains spare shoes.",
            source_id="turn-7",
            session_id="training",
        )
        self.assertEqual(len(records), 3)
        bundle = self.service.recall("Where are my training shoes?")
        contents = [hit.record.content for hit in bundle.evidence]
        self.assertTrue(any("car" in content for content in contents))
        self.assertTrue(bundle.trace.channel_results["lexical"])
        self.assertTrue(bundle.trace.selected_ids)
        self.assertIn("provenance", bundle.text)

    def test_chinese_sentences_are_split_without_spaces(self) -> None:
        records = self.service.ingest_text("训练包在车里。备用鞋在包内。", create_summary=False)
        self.assertEqual(len(records), 2)

    def test_correction_supersedes_and_forget_hides(self) -> None:
        old = self.service.add_memory("The bag is at home.")
        correction = self.service.correct(old.id, "The bag is now in the car.")
        self.assertEqual(self.service.store.get(old.id).status, MemoryStatus.DEPRECATED)
        self.assertEqual(correction.layer, MemoryLayer.L2)
        self.assertEqual(correction.supersedes, [old.id])
        bundle = self.service.recall("Where is the bag?")
        self.assertEqual(bundle.evidence[0].record.id, correction.id)
        self.service.forget(correction.id)
        self.assertFalse(self.service.recall("Where is the bag?").evidence)

    def test_procedure_is_separate_from_factual_evidence(self) -> None:
        procedure = self.service.remember_procedure(
            "planning a workout", ["check the bag", "check the calendar"]
        )
        bundle = self.service.recall("Help me plan a workout and check the bag")
        self.assertIn(procedure.id, [item.id for item in bundle.operational_guidance])
        self.assertNotIn(procedure.id, [hit.record.id for hit in bundle.evidence])

    def test_low_confidence_memory_is_filtered(self) -> None:
        record = self.service.add_memory("Possible location is the garage", confidence=0.1)
        bundle = self.service.recall("garage location")
        self.assertNotIn(record.id, bundle.trace.selected_ids)
        self.assertIn({"id": record.id, "reason": "low_confidence"}, bundle.trace.filters)

    def test_organize_archives_low_value_memory(self) -> None:
        record = self.service.add_memory("transient detail", importance=0.0)
        result = self.service.organize()
        self.assertIn(record.id, result["archived"])
        self.assertEqual(self.service.store.get(record.id).status, MemoryStatus.ARCHIVED)

    def test_typed_expansion_payloads_keep_atomic_provenance(self) -> None:
        fused = self.service.admit_fused_event(
            FusedEvent(
                id="fused-1",
                content="The training bag moved from home to the car.",
                source_event_ids=["camera-1", "car-1"],
                timestamp="2026-07-10T08:00:00+00:00",
                device_ids=["home-camera", "vehicle"],
                edge_type="CAUSES",
                provenance=["camera-1", "car-1"],
            )
        )
        visual = self.service.admit_perception_fact(
            PerceptionFact(
                fact_id="fact-1",
                image_id="image-1",
                content="A blue bag contains spare shoes.",
                category="training equipment",
            )
        )
        self.assertEqual([source.source_id for source in fused.sources], ["camera-1", "car-1"])
        self.assertEqual(fused.metadata["payload_type"], "FusedEvent")
        self.assertEqual(visual.metadata["image_id"], "image-1")


if __name__ == "__main__":
    unittest.main()
