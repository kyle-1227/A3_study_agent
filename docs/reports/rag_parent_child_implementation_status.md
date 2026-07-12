# Parent-Child Hybrid RAG implementation status

Date: 2026-07-11
Analysis baseline: `f14194ea859bef5783ea6c3b7da51e7a776ea094`
Final worktree audit HEAD: `66d1dd8be4450ee7500b855ce2fbbc1693a23876` (the analysis baseline remains an ancestor; the user advanced the branch during implementation)

## Outcome

The isolated parent-child candidate path is implemented through immutable generation build, strict runtime retrieval, offline evaluation, an explicit generation-pinned Graph factory, Shadow/canary control, and health contracts. The candidate Graph gives the Evidence Judge child previews only, then hydrates authoritative parent context after the Judge decision. It has not been selected by the served application, activated, or written to the existing `chroma_store`.

Production recommendation remains blocked. The current corpus has no human or annotated historical gold queries, every primary subject has fewer than three independent sources, and no valid production `config/rag/index.yaml` has been supplied. The repository therefore contains no invented provider/model/base URL or fake gold data.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 - readiness/baseline | Audit implemented and run | Read-only audit reproduces the corpus statistics. A clean network baseline build is blocked until an explicit valid index config and secret are supplied. Synthetic questions remain rollout-ineligible. |
| 1 - strict config/catalog/contracts | Implemented | Strict/frozen/extra-forbid configs, canonical policy recomputation, explicit secrets, and exact SubjectCatalog policy mapping. |
| 2 - page-aware loader | Implemented | Physical/logical page ordinals, empty-page retention, cross-page top/bottom noise detection, and cleaned offset/page spans. Legacy loader remains available only to the baseline path. |
| 3 - parent/child chunking | Implemented | Deterministic parent and child IDs, exact offsets, major-boundary protection, short-unit merge, atomic-block validation, and independent child splitting. |
| 4 - storage/generation | Implemented | Staging-only child Chroma, authoritative SQLite parents, safe per-subject BM25 JSONL, strict manifests, crash-visible registry lifecycle, and explicit activation/rollback/cleanup. |
| 5 - strict hybrid retrieval | Implemented | Exact-subject Vector/BM25, weighted RRF, strict reranker, parent aggregation/source caps, multi-subject coverage quota, generation-keyed resources, and no request fallback. |
| 6 - evaluation | Implemented | Policy-independent gold spans, layered metrics, subject-stratified paired bootstrap, and data/effect/latency/token/final-answer gates. |
| 7 - Judge/CE handoff | Explicit candidate Graph route implemented | The candidate Graph retrieves capped supporting children, binds request plus the complete handoff fingerprint, preserves each child's branch role/purpose, validates every Judge-kept child ID, and hydrates/merges parents before answer or resource routing. Parent bodies remain outside candidate metadata and the Judge prompt. Parent context carries supporting child page/absolute-span provenance and parent window spans. The served `app.py` path still calls the legacy `get_compiled_graph`; candidate activation remains disabled. |
| 8 - Shadow | Implemented | Baseline is the predetermined served path; candidate failure is recorded as failure and never presented as fallback success. |
| 9 - canary/rollback | Implemented | Stable request hashing, subject eligibility, explicit `ROLLBACK_REQUIRED`, and registry activate/rollback CLI. Initial rollout config remains disabled. |
| 10 - monitoring/drift | Implemented | Content-free metric events and validation invalidation on source/subject/policy/embedding/dataset fingerprint drift. |

## New explicit entry points

- `scripts/audit_rag_readiness.py`
- `scripts/doctor_rag_env.py`
- `scripts/build_index.py --pipeline flat-baseline`
- `scripts/build_parent_child_generation.py`
- `scripts/validate_parent_child_candidate.py`
- `scripts/manage_rag_generation.py`
- `src.rag.parent_child.runtime_loader.load_generation_runtime(...)`
- `src.graph.parent_child_nodes.parent_child_graph_runtime_from_loaded(...)`
- `src.graph.builder.build_parent_child_graph(runtime)`
- `src.graph.builder.get_compiled_parent_child_graph(runtime, *, checkpointer)`
- `ParentChildHybridRetriever.retrieve_children_multi(request)`
- `ParentChildHybridRetriever.hydrate_kept_multi(result, kept_child_ids)`

Every path/config/output argument is explicit. Generation build can only produce `READY`; activation is a separate control-plane command.

## Safety decisions

- No fallback or degraded candidate mode was added.
- Provider errors, invalid scores, missing IDs, storage mismatch, request/fingerprint drift, and hydration errors fail with typed exceptions.
- Retrieval-plan priority must be explicitly supplied for candidate multi-branch routing; legacy defaulted values are rejected at the candidate boundary.
- Provider response bodies, secrets, parent bodies, and source bodies are excluded from registry diagnostics and health/Shadow records.
- Parent bodies enter context only after Evidence Judge decisions; Judge candidates and candidate metadata contain child previews and content-free provenance only.
- BM25 uses canonical JSONL and never pickle.
- Failed or obsolete storage is deleted only after root containment and a strict ownership-marker check.
- Storage artifact resolution rejects symlink components before canonical path resolution; a real Windows symlink test is retained but skipped when the host denies symlink creation.
- Importing the legacy reset helper no longer loads `.env` into the caller; CLI execution still loads it explicitly, and the orphan `parent_chunks.jsonl` target remains report-only rather than deleted.
- Existing user Graph changes were preserved. Phase 7 adds an explicit candidate-only Graph factory, while the served application remains on the legacy graph because rollout activation is disabled.
- No frontend or PDF worktree changes were overwritten, and no active index was mutated.

## Verification

- `python -m compileall -q src scripts tests`: passed.
- Scoped Ruff check and format check across 63 implementation/test files: passed.
- Scoped mypy with unavailable-import isolation across 35 new/protected source files: passed.
- Phase 7 Graph/handoff/retrieval focused suite: `23 passed`.
- Full pytest: `1521 passed, 2 skipped`; the skips are environment-dependent integration cases. Pytest also reported the existing cache-directory permission warning and one asynchronous SQLite event-loop shutdown warning.
- Focused security/config suite: `35 passed`; strict benchmark/rollout configuration keeps activation and Shadow disabled.
- `git diff --check`: passed apart from Git's existing LF-to-CRLF notices.
- Repository-wide Ruff remains blocked by 53 pre-existing findings and 77 pre-existing formatting differences outside this implementation scope; those files were not mass-rewritten.

## Tool availability

Python, pytest, Ruff, and mypy are available. Semgrep, import-linter, Pyright, ty, Gitleaks, Bandit, and Vulture are missing; they must not be interpreted as passing gates.

## 2026-07-12 joint evidence candidate update

The Parent-Child runtime is now also bound into an explicit resource-aware evidence candidate through `build_resource_evidence_parent_child_graph(runtime)` and `get_compiled_resource_evidence_parent_child_graph(runtime, *, checkpointer)`. This new graph plans evidence after resource and subject selection, supports at most two targeted supplement rounds, performs terminal-only one-shot parent hydration, and blocks only resources whose required evidence remains incomplete. The older Parent-Child graph remains available as the P0/ablation route, and `app.py` still serves the legacy graph. Activation remains blocked by the production index/gold prerequisites above and by the new P0/PG/PR/PGR gate documented in `evidence_orchestration_implementation_status.md`.
