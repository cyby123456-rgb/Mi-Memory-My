from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from mimemory.api import create_server
from mimemory.service import MemoryService


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        service = MemoryService.local(Path(self.temp.name))
        self.server = create_server(service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path: str, body: dict | None = None) -> tuple[int, object]:
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        with urlopen(request, timeout=2) as response:
            return response.status, json.load(response)

    def test_health_ingest_and_recall(self) -> None:
        status, health = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, records = self.request("/ingest", {"text": "My keys are in the kitchen."})
        self.assertEqual(status, 201)
        self.assertEqual(len(records), 1)
        status, bundle = self.request("/recall", {"query": "Where are my keys?"})
        self.assertEqual(status, 200)
        self.assertIn("kitchen", bundle["text"])


if __name__ == "__main__":
    unittest.main()

