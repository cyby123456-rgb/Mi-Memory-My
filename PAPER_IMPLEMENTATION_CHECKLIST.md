# Public-Spec Implementation Checklist

This is a clean-room implementation checklist for arXiv:2607.18975. It tracks
published behavior only; it does not infer unpublished code, prompts, datasets,
or endpoint settings.

| Paper basis | Required public behavior | Implementation artifact | Status |
| --- | --- | --- | --- |
| Sec. 3.1, Fig. 2 | Typed evidence, diagnostic trace, strategy, gate/rollback contracts | `paper_contracts.py`, `models.py` | Implemented |
| Sec. 4.2, Fig. 3; App. B Eq. 10 | L0/L1/L2/SM, LLM extraction, vector/BM25/subquery RRF, protected packing, traces | `memstack.py`, `assembly.py`, `retrieval.py` | Implemented; invalidation/decay/promotion are provided by the shared LiteMem lifecycle path |
| App. B Table 15, Eq. 11 | ProcedureEntry as operational guidance, separate from factual evidence | `models.py`, `service.py` | Implemented |
| Sec. 5.2; App. D Table 22, Alg. 1 | Five-pass IKB, category/session/time indexes, VR/VS/TTL IKB-first routing | `multimodal.py` | Implemented: five audited passes, persistent category/session/time IKB, index-constrained router and eight-image residual bound |
| Sec. 5.2; App. C | Atomic device events, FusionSession, dual-layer causal graph, conflict retention | `multimodal.py` | Implemented: three transient zones, persistent atomic/MemoryPack graph, BELONG/CAUSES edges and low-confidence atomic retention |
| Sec. 6; App. E | D2ACCI hypothesis, aligned paired outputs, six-step review artifacts | `diagnostics.py`, `benchmarks.py` | Implemented: six-stage aligned artifact, append-only ledger and category-level root-cause roll-up |
| App. F Alg. 2--3, Table 28, Eq. 19 | E2MEND Observe/Improve/Verify, five gates, Critic, UCB1, pending champion, drift rollback | `evolution.py`, `strategy.py` | Implemented: typed/versioned prompt mutations, hard integrity gate, candidate and dimension history, UCB1 scoring, pending champions and drift rollback |
| Sec. 7; App. G Eq. 20--21, Alg. 4 | File-native router, lazy daily reads, style/profile priority, decay, idle consolidation, atomic index, Git | `storage.py`, `lifecycle.py`, `git_provenance.py` | Implemented: Eq. 20/21 scoring terms, singleton style/profile, correction file, line-window router, priority injection, markers, atomic index and Git hook |
| Sec. 8; App. H | Unified benchmark adapters, traces, paired non-regression reports, resource accounting | `benchmarks.py`, `scripts/run_public_benchmark.py` | Fixed-config public runner emits primary-correctness diagnostics, resource ledger and category report; upstream official scorers remain required for published scores |
| API Guide | Synchronous Add/Search, user isolation, exact response contracts | `leaderboard.py` | Implemented; paper runtime selectable |

## Explicit Non-Claims

- Exact original source code, internal MemFuseBench, original prompts, and deployment
  endpoints are not public and cannot be reconstructed faithfully.
- No paper score is claimed until a separate approved benchmark run produces a
  configuration, trace set, and paired report.
- The generic runner's `primary_correctness` is an audit metric, not an official
  LoCoMo, PersonaMem-v2, or LongMemEval leaderboard score.
