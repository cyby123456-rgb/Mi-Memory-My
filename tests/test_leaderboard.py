from __future__ import annotations

import json
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mimemory.leaderboard import ContractError, LeaderboardAdapter, create_leaderboard_server


class LeaderboardAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.adapter = LeaderboardAdapter(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_payload(self, user_id: str = "eval:run:locomo:conv-0") -> dict:
        return {
            "request_id": "eval:run:locomo_refined:conv-0:chunk-0",
            "messages": [
                {
                    "role": "user",
                    "timestamp": 1704067200000,
                    "content": "The blue training bag is in the car.",
                },
                {"role": "assistant", "content": "I will remember that."},
            ],
            "user_id": user_id,
            "session_id": "eval:run:sample:0",
        }

    def test_add_echoes_contract_and_is_synchronously_searchable(self) -> None:
        payload = self.add_payload()
        response = self.adapter.add(payload)
        self.assertEqual(
            response,
            {
                "success": True,
                "request_id": payload["request_id"],
                "user_id": payload["user_id"],
                "session_id": payload["session_id"],
            },
        )
        result = self.adapter.search(
            {"query": "Where is the training bag?", "user_id": payload["user_id"], "top_k": 100}
        )
        self.assertEqual(result["data"][0]["content"], "The blue training bag is in the car.")
        self.assertEqual(result["data"][0]["created_at"], "2024-01-01T00:00:00+00:00")
        self.assertIsInstance(result["data"][0]["score"], float)

    def test_user_id_is_a_hard_retrieval_boundary(self) -> None:
        self.adapter.add(self.add_payload("user-a"))
        empty = self.adapter.search({"query": "training bag", "user_id": "user-b", "top_k": 10})
        self.assertEqual(empty, {"data": []})

    def test_add_is_idempotent_and_rejects_changed_reuse(self) -> None:
        payload = self.add_payload()
        self.adapter.add(payload)
        self.adapter.add(payload)
        service = self.adapter.registry.service_for(payload["user_id"])
        self.assertEqual(len(service.store.list()), 2)
        changed = self.add_payload()
        changed["messages"][0]["content"] = "Changed content"
        with self.assertRaisesRegex(ContractError, "already used"):
            self.adapter.add(changed)

    def test_missing_timestamp_remains_idempotent(self) -> None:
        payload = self.add_payload()
        payload["messages"] = [{"role": "user", "content": "No timestamp."}]
        self.adapter.add(payload)
        self.adapter.add(payload)
        self.assertEqual(len(self.adapter.registry.service_for(payload["user_id"]).store.list()), 1)

    def test_search_honors_top_k_and_choice_options(self) -> None:
        payload = self.add_payload()
        self.adapter.add(payload)
        result = self.adapter.search(
            {
                "query": "Which answer matches?",
                "options": ["A. kitchen", "B. training bag in the car"],
                "user_id": payload["user_id"],
                "top_k": 1,
            }
        )
        self.assertEqual(len(result["data"]), 1)
        self.assertIn("car", result["data"][0]["content"])

    def test_contract_validation(self) -> None:
        with self.assertRaisesRegex(ContractError, "messages"):
            self.adapter.add({"request_id": "r", "messages": [], "user_id": "u", "session_id": "s"})
        with self.assertRaisesRegex(ContractError, "top_k"):
            self.adapter.search({"query": "q", "user_id": "u", "top_k": 0})

    def test_retention_purge_removes_only_expired_scope(self) -> None:
        old_payload = self.add_payload("old-user")
        new_payload = self.add_payload("new-user")
        self.adapter.add(old_payload)
        self.adapter.add(new_payload)
        old_metadata = self.adapter.registry.scope_path("old-user") / ".scope.json"
        old_metadata.write_text(
            json.dumps({"user_id": "old-user", "last_seen_at": (datetime.now(UTC) - timedelta(days=31)).isoformat()}),
            encoding="utf-8",
        )
        removed = self.adapter.registry.purge_expired(30)
        self.assertIn(self.adapter.registry.scope_path("old-user").name, removed)
        self.assertFalse(self.adapter.registry.scope_path("old-user").exists())
        self.assertTrue(self.adapter.registry.scope_path("new-user").exists())


class LeaderboardHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.adapter = LeaderboardAdapter(Path(self.temp.name))
        self.server = create_leaderboard_server(self.adapter, "127.0.0.1", 0, api_token="secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path: str, payload: dict | None = None, *, token: str | None = "secret", auth_scheme: str = "Bearer") -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"{auth_scheme} {token}"
        request = Request(
            self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def test_http_contract_status_paths_and_auth(self) -> None:
        status, health = self.request("/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")

        payload = LeaderboardAdapterTests.add_payload(self)
        status, unauthorized = self.request("/add", payload, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(unauthorized["error"], "unauthorized")

        status, added = self.request("/add", payload)
        self.assertEqual(status, 200)
        self.assertTrue(added["success"])

        token_payload = LeaderboardAdapterTests.add_payload(self, user_id="token-user")
        status, added = self.request("/v1/add", token_payload, auth_scheme="Token")
        self.assertEqual(status, 200)
        self.assertTrue(added["success"])

        status, searched = self.request(
            "/search", {"query": "training bag", "user_id": payload["user_id"], "top_k": 100}
        )
        self.assertEqual(status, 200)
        self.assertTrue(searched["data"])


if __name__ == "__main__":
    unittest.main()
