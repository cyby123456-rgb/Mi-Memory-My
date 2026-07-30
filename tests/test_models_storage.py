from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mimemory.models import DiagnosticTrace, MemoryLayer, MemoryRecord, MemoryStatus, SourceRef
from mimemory.storage import LiteMemStore


class LiteMemStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = LiteMemStore(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_record_roundtrip_preserves_typed_fields(self) -> None:
        record = MemoryRecord(
            content="The training bag is in the car.",
            layer=MemoryLayer.L2,
            sources=[SourceRef("turn-1", device_id="phone")],
            metadata={"kind": "correction"},
        )
        self.store.put(record)
        loaded = self.store.get(record.id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.layer, MemoryLayer.L2)
        self.assertEqual(loaded.sources[0].device_id, "phone")
        self.assertEqual(loaded.metadata["kind"], "correction")
        path = self.root / json.loads((self.root / "_index.json").read_text())[record.id]
        self.assertTrue(path.exists())

    def test_inactive_records_are_hidden_by_default(self) -> None:
        record = MemoryRecord(content="stale", status=MemoryStatus.FORGOTTEN)
        self.store.put(record)
        self.assertEqual(self.store.list(), [])
        self.assertEqual(len(self.store.list(include_inactive=True)), 1)

    def test_rebuild_index_recovers_missing_index(self) -> None:
        record = self.store.put(MemoryRecord(content="recover me"))
        (self.root / "_index.json").write_text("not json", encoding="utf-8")
        rebuilt = self.store.rebuild_index()
        self.assertIn(record.id, rebuilt)
        self.assertEqual(self.store.get(record.id).content, "recover me")

    def test_raw_log_and_trace_are_persisted(self) -> None:
        raw_path = self.store.append_raw("hello", session_id="s1")
        trace = DiagnosticTrace(query="hello")
        trace_path = self.store.write_trace(trace)
        self.assertIn("session=s1", raw_path.read_text())
        self.assertEqual(json.loads(trace_path.read_text())["query_id"], trace.query_id)

    def test_singleton_context_is_prepended_to_file_native_route(self) -> None:
        style = self.store.write_profile("Be concise.", style=True)
        profile = self.store.write_profile("User likes tea.")
        evidence = self.store.put(MemoryRecord(content="The tea is in the cupboard."))
        routed = self.store.route_file_native("where is tea")
        self.assertEqual([item.id for item in routed[:2]], [style.id, profile.id])
        self.assertIn(evidence.id, [item.id for item in routed])
        self.assertIn("SUMMARY_END", (self.root / "user" / "style.md").read_text(encoding="utf-8"))

    def test_replacing_a_singleton_removes_its_stale_index_owner(self) -> None:
        first = self.store.put(MemoryRecord(content="Old profile", layer=MemoryLayer.L2, metadata={"kind": "profile"}))
        second = self.store.put(MemoryRecord(content="New profile", layer=MemoryLayer.L2, metadata={"kind": "profile"}))
        self.assertIsNone(self.store.get(first.id))
        self.assertEqual(self.store.get(second.id).content, "New profile")


if __name__ == "__main__":
    unittest.main()
