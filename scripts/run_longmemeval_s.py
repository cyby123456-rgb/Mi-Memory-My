"""Run the official LongMemEval-S session-level Add/Search flow with checkpoints.

The caller must explicitly opt in to live providers via
MIMEMORY_LIVE_PROVIDER_APPROVED=1. Results deliberately contain identifiers and
retrieval metadata only, not dataset conversations or environment credentials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep
from typing import Any

from mimemory.benchmarks import LongMemEvalAdapter
from mimemory.config import default_strategy
from mimemory.leaderboard import PaperLeaderboardAdapter
from mimemory.memstack import MemStackModels, MemStackRuntime
from mimemory.providers import OpenAICompatibleClient, PaperModelRoles
from mimemory.storage import LiteMemStore


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every official sample")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--max-case-attempts", type=int, default=0, help="0 retries a failed case until it succeeds")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    results_path, errors_path = root / "results.jsonl", root / "errors.jsonl"
    completed = {
        json.loads(line)["question_id"]
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if results_path.exists() else set()

    roles = PaperModelRoles.from_environment()

    def runtime_factory(user_id: str) -> MemStackRuntime:
        scope = root / "memory" / user_id
        return MemStackRuntime(LiteMemStore(scope), default_strategy(), MemStackModels(
            extractor=OpenAICompatibleClient(roles.extraction),
            planner=OpenAICompatibleClient(roles.extraction),
            reranker=OpenAICompatibleClient(roles.evaluator),
            embeddings=OpenAICompatibleClient(roles.embedding),
        ))

    adapter, dataset = PaperLeaderboardAdapter(runtime_factory), LongMemEvalAdapter()
    cases = dataset.load(args.data)
    if args.limit:
        cases = cases[:args.limit]
    for ordinal, case in enumerate(cases, start=1):
        if case.case_id in completed:
            continue
        requests = dataset.add_requests(case)
        attempt = 0
        while True:
            attempt += 1
            try:
                for request in requests:
                    adapter.add(request)
                response = adapter.search({"query": case.query, "user_id": case.user_id, "top_k": args.top_k, "options": []})
                append_jsonl(results_path, {
                    "ordinal": ordinal, "question_id": case.case_id, "question_type": case.category,
                    "add_requests": len(requests), "messages": sum(len(item["messages"]) for item in requests),
                    "search_hit_ids": [item["id"] for item in response["data"]],
                })
                break
            except Exception as exc:  # Keep a retriable audit record without secrets or source text.
                delay = min(args.retry_delay * (2 ** min(attempt - 1, 6)), 300.0)
                append_jsonl(errors_path, {"ordinal": ordinal, "question_id": case.case_id, "attempt": attempt,
                    "retry_delay_seconds": delay, "error": type(exc).__name__, "message": str(exc)[:500]})
                if args.max_case_attempts and attempt >= args.max_case_attempts:
                    raise
                sleep(delay)


if __name__ == "__main__":
    main()
