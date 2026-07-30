from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen
from uuid import uuid4


def post(url: str, payload: dict, token: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=15) as response:
        return response.status, json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the leaderboard Add/Search contract")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--token")
    args = parser.parse_args()
    run_id = uuid4().hex
    request_id = f"smoke:{run_id}:chunk-0"
    user_id = f"smoke:{run_id}:user-0"
    session_id = f"smoke:{run_id}:session-0"
    marker = f"blue-orchid-{run_id}"
    add_payload = {
        "request_id": request_id,
        "messages": [{"role": "user", "content": f"The private marker is {marker}."}],
        "user_id": user_id,
        "session_id": session_id,
    }
    add_status, added = post(args.base_url.rstrip("/") + "/add", add_payload, args.token)
    expected = {
        "success": True,
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
    }
    if add_status != 200 or added != expected:
        raise SystemExit(f"Add contract failed: status={add_status}, response={added}")
    search_status, searched = post(
        args.base_url.rstrip("/") + "/search",
        {"query": "What is the private marker?", "user_id": user_id, "top_k": 10},
        args.token,
    )
    if search_status != 200 or not any(marker in item.get("content", "") for item in searched.get("data", [])):
        raise SystemExit(f"Search contract failed: status={search_status}, response={searched}")
    print("Add/Search smoke test passed")


if __name__ == "__main__":
    main()

