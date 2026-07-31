"""Run one released benchmark with a fixed Mi-Memory configuration.

The script deliberately keeps answer keys in input-only data and writes only
case identifiers, aggregate metrics, diagnostics, and resource timings.  It
does not perform strategy search or modify a configuration from benchmark
answers.  Remote calls remain separately gated by the environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from mimemory.benchmarks import BenchmarkHarness, LoCoMoAdapter, LongMemEvalAdapter, PersonaMemV2Adapter
from mimemory.config import default_strategy
from mimemory.memstack import MemStackModels, MemStackRuntime
from mimemory.providers import OpenAICompatibleClient, PaperModelRoles
from mimemory.storage import LiteMemStore


ADAPTERS = {
    "locomo": LoCoMoAdapter,
    "longmemeval": LongMemEvalAdapter,
    "personamem-v2": PersonaMemV2Adapter,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-config public benchmark runner")
    parser.add_argument("--benchmark", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--data", required=True, help="Released dataset file; never committed")
    parser.add_argument("--root", required=True, help="Ignored run directory")
    parser.add_argument("--limit", type=int, default=0, help="0 runs all cases")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cases = ADAPTERS[args.benchmark]().load(args.data)
    if args.limit:
        cases = cases[:args.limit]
    roles = PaperModelRoles.from_environment()
    runtime = MemStackRuntime(
        LiteMemStore(root / "memory"),
        default_strategy(),
        MemStackModels(
            extractor=OpenAICompatibleClient(roles.extraction),
            planner=OpenAICompatibleClient(roles.extraction),
            reranker=OpenAICompatibleClient(roles.evaluator),
            embeddings=OpenAICompatibleClient(roles.embedding),
        ),
    )
    # Answer generation is free-form; JSON mode is reserved for the runtime's
    # structured extraction and diagnostic contracts.
    answerer = OpenAICompatibleClient(replace(roles.extraction, json_mode=False))
    classifier = OpenAICompatibleClient(roles.evaluator)
    harness = BenchmarkHarness(runtime, answerer, classifier)
    results = harness.run(cases, root / "results.jsonl", root / "resources.jsonl")
    report = {
        "benchmark": args.benchmark,
        "configuration": "default_strategy",
        "cases": len(results),
        "metrics": harness.report(results, {case.case_id: case.category for case in cases}),
    }
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
