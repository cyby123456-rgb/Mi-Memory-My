# Mi-Memory Clean-room Runtime

An executable, dependency-light reconstruction of the lifecycle memory architecture described in **Mi-Memory: A Lifecycle Memory Framework for Personal AI** ([arXiv:2607.18975](https://arxiv.org/abs/2607.18975)). Version 0.2 also provides a submission-compatible Add/Search wrapper for the Agent Memory Leaderboard.

This repository is an independent implementation based on the public paper. It is not the authors' original implementation and does not claim to reproduce the paper's reported benchmark scores. The authors' public repository contained the paper and project site, but no source code, when this implementation was created.

## What works

- Typed L0 atomic facts, L1 summaries, L2 profiles/corrections, SM state, and procedural guidance
- Markdown files with JSON/YAML-compatible frontmatter, raw daily logs, indexes, audit records, and diagnostic traces
- Semantic-like sparse matching, BM25 lexical retrieval, bounded subquery expansion, and weighted Reciprocal Rank Fusion
- Confidence filtering, duplicate suppression, profile reservation, correction priority, and bounded context packing
- Source identity, timestamps, device identifiers, supersession, user correction, forgetting, access feedback, decay, and archival
- Typed `FusedEvent` and `PerceptionFact` admission interfaces for later MemFuse/MemSense integrations
- Schema-bounded strategy mutations, evaluation gates, checkpoint records, acceptance/rejection history, and rollback
- Zero-dependency CLI and threaded JSON HTTP API
- Leaderboard-compatible synchronous `POST /add` and `POST /search` endpoints with strict `user_id` isolation
- Docker packaging and an end-to-end example

## Architecture mapping

| Paper role | This implementation | Status |
| --- | --- | --- |
| Structure / MemStack | Typed records, hybrid retrieval, RRF, context assembly, traces | Executable baseline |
| Expansion / MemSense | `PerceptionFact` admission contract | Interface implemented; VLM/IKB builder pending |
| Expansion / MemFuse | `FusedEvent` admission contract with atomic provenance | Interface implemented; causal fusion engine pending |
| Evolution / D2ACCI | Per-query traces and explicit candidate evaluation reports | Executable baseline |
| Evolution / E2MEND | Bounded strategy proposals, deterministic gates, checkpoints, rollback | Executable baseline; autonomous planner pending |
| Deployment / LiteMem | Markdown repository, daily write-ahead log, index, decay, audit log | Executable baseline |

## Quick start

Python 3.11 or newer is sufficient. The runtime itself has no third-party dependencies.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

mimemory --root .mimemory ingest \
  "I put my blue training bag in the car. It contains a jersey and spare shoes." \
  --source-id dialogue-1 --session-id training

mimemory --root .mimemory recall "Where is my training bag and what is in it?"
```

Run the example without installing the package:

```bash
PYTHONPATH=src python examples/demo.py
```

## HTTP API

Start the service:

```bash
mimemory --root .mimemory serve --host 127.0.0.1 --port 8765
```

Write and recall memory:

```bash
curl -X POST http://127.0.0.1:8765/ingest \
  -H 'content-type: application/json' \
  -d '{"text":"My keys are in the kitchen.","source_id":"turn-1"}'

curl -X POST http://127.0.0.1:8765/recall \
  -H 'content-type: application/json' \
  -d '{"query":"Where are my keys?"}'
```

Available endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness check |
| `GET` | `/memories` | List active and inactive records |
| `GET` | `/strategy` | Inspect the current strategy artifact |
| `POST` | `/ingest` | Capture dialogue and deterministic facts |
| `POST` | `/memories` | Add a typed memory directly |
| `POST` | `/recall` | Retrieve, assemble, and trace evidence |
| `POST` | `/correct` | Supersede a memory with an L2 correction |
| `POST` | `/forget` | Exclude a memory from serving retrieval |
| `POST` | `/organize` | Apply importance decay and archival policy |
| `POST` | `/strategy/evaluate` | Propose, gate, and optionally accept a strategy mutation |
| `POST` | `/strategy/rollback` | Restore the accepted strategy's parent checkpoint |

## Leaderboard Add/Search API

The Docker image starts the leaderboard wrapper by default:

```bash
mimemory-leaderboard --root .mimemory-eval --host 0.0.0.0 --port 8765
```

It exposes:

```text
GET  /health
POST /add
POST /search
```

Each `user_id` receives a physically separate memory repository. Add requests are synchronous and idempotent; Search accepts the platform's optional choice list and dynamic `top_k`. See `SUBMISSION.md` for the exact platform schema, Docker command, optional authentication, HTTPS deployment note, and 30-day retention procedure.

Run the contract smoke test with:

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:8765
```

## File-native memory

The default data directory follows the paper's LiteMem shape:

```text
.mimemory/
├── litemem/
│   ├── _index.md
│   ├── audit.jsonl
│   ├── daily/
│   ├── entity/
│   ├── sessions/
│   ├── user/profile/
│   ├── knowledge/skill/
│   └── traces/
└── strategies/
    ├── current.json
    ├── best.json
    ├── history.jsonl
    └── checkpoints/
```

Every memory file contains JSON frontmatter between `---` markers. JSON mappings are valid YAML, so the records can be parsed by either JSON-aware code or ordinary YAML tooling without making YAML a runtime dependency.

## Retrieval and context assembly

The retriever produces three ranked lists:

1. sparse token/character features for a deterministic semantic baseline;
2. BM25 lexical ranking;
3. BM25 over bounded query decompositions.

It combines them with the paper's weighted RRF rule using a default smoothing constant of 60. The assembler then applies confidence gates and duplicate suppression, separates procedures from factual evidence, packs corrections first, reserves capacity for L2 profile records, and fills the remaining budget by fused rank. Each recall persists channel rankings, selected IDs, dropped IDs, filters, token usage, and the strategy artifact ID.

This repository deliberately uses a deterministic local semantic baseline instead of downloading an embedding model. A production provider can replace the semantic channel while preserving the retrieval and trace contracts.

## Strategy governance

Runtime behavior lives in a versioned strategy artifact. `StrategyManager.propose` accepts only declared numeric paths with hard bounds. `StrategyManager.evaluate` rejects a candidate when:

- its aggregate delta is below the configured minimum;
- a category delta crosses the non-regression tolerance;
- stable-correct replay regresses;
- the replay gate fails.

Accepted candidates receive a gate record and rollback key. Framework code, storage schemas, and evaluation protocols are outside the mutable strategy space, matching the paper's separation between D2ACCI governance and E2MEND automation.

## Tests

Run the standard-library suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers persistence, index recovery, active/inactive filtering, raw logs, diagnostic traces, English and Chinese ingestion, hybrid retrieval, correction, forgetting, procedural separation, confidence filtering, lifecycle archival, typed multi-source payloads, strategy gates, rollback, both HTTP APIs, Add idempotency, user isolation, dynamic `top_k`, authentication, and retention cleanup.

## Docker

```bash
docker compose up --build
```

Data is stored in the `mimemory-data` volume and the API is exposed on port 8765.

## Reproduction boundary

The following paper claims cannot currently be reproduced exactly from public artifacts:

- the reported 93.59%, 57.24%, and 87.47% results without the original unified harness, prompts, judge setup, traces, and endpoint behavior;
- MemFuseBench, which the paper identifies as internal;
- exact E2MEND trajectory replay, which depends on undisclosed deployment-specific model endpoints and full strategy history;
- the procedural-memory benchmark, which the paper labels design-only;
- the complete MemSense IKB/VLM pipeline and causal MemFuse engine.

The code therefore provides a runnable systems reconstruction and an auditable base for public benchmark adapters. It does not present clean-room behavior as the paper authors' unpublished implementation.

## License

MIT. The linked paper, benchmark datasets, model outputs, and authors' project assets retain their own licenses.
