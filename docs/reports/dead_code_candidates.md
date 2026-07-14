# Dead Code Candidates

Initial governance report created on 2026-06-20. No dead code was deleted.

## Policy

- Vulture and text scans are report-only.
- Do not delete legacy fallback/helper/dead code without explicit approval.
- Verify dynamic references before cleanup, especially LangGraph nodes, FastAPI routes, config keys, prompts, and tests.

## Current Baseline

Vulture has not been run in this first governance round. No dead-code candidates are approved for deletion.

## 2026-07-10 Supervisor phrase-detector candidates

Candidate: Legacy resource-request phrase detectors
File: `src/graph/supervisor.py`
Symbol: `_READABLE_*_MARKERS`, `_detect_requested_resource_types`, `_detect_requested_resource_type`
Evidence: Repository reference scan finds runtime definitions and test imports only; `supervisor_node` uses strict structured output and does not call these helpers.
Confidence: High
Dynamic reference checks: LangGraph builder and prompt/config scans show no dynamic symbol lookup for these private helpers.
Related tests: `tests/test_supervisor.py`
Recommended action: Keep report-only until explicit deletion approval; do not use these helpers as a runtime routing source.

## 2026-07-10 RAG parent JSONL cleanup candidate

Candidate: `parent_chunks.jsonl` cleanup entry
File: `scripts/reset_index.py`
Symbol: cleanup target only; no corresponding writer or reader was found
Evidence: Repository reference scans find the filename only in the reset
script and a metadata-schema test. No production Parent Store implementation
uses it at the approved baseline.
Confidence: Medium; this may be an unfinished design reservation rather than
dead code.
Dynamic reference checks: No config, prompt, CLI, or runtime import constructs
the filename outside the reset path.
Related tests: `tests/test_metadata_schema.py`
Recommended action: Keep report-only. Do not delete the entry until the
generation-owned cleanup path is implemented and separately approved.

## How to Add Findings

Use this format:

```text
Candidate:
File:
Symbol:
Evidence:
Confidence:
Dynamic reference checks:
Related tests:
Recommended action:
```

## Follow-Up Command

```powershell
vulture src tests --min-confidence 80
```

## 2026-07-13 Streaming and Context Window V3 replacement audit

Scope: only code superseded by the approved streaming/context-window refactor.
The older candidates above remain report-only and are explicitly outside this
cleanup. The fixed implementation baseline is
`9bf9950c002d71b7d184fadcd35993ab35306bd5` on
`codex/streaming-context-v3`.

### Replacement parity map

| Superseded behavior | Authoritative replacement | Executable evidence |
| --- | --- | --- |
| Legacy SSE `token`/`text` browser updates | `agent_stream_v2` content blocks plus `LiveTurnState`; only `qa_final`/`resource_final` commit messages | `tests/test_agent_stream_v2.py`, `tests/test_stream_session.py`, `frontend/lib/agent-stream-client.test.ts`, `frontend/lib/live-turn.test.ts` |
| Hand-written SSE chunk splitting | Shared incremental parser/reader with CRLF, multiline data, UTF-8 chunk and EOF handling | `frontend/lib/sse-parser.test.ts`, `frontend/lib/agent-stream-client.test.ts`, `tests/test_streaming_context_v3_replacement_parity.py` |
| Non-resumable request stream | Run journal, request binding, replay lease, Last-Event-ID gap/expiry failures | `tests/test_stream_session.py`, `tests/test_agent_stream_v2.py` |
| Provisional structured JSON forwarding | Incremental `QAResponse.answer` decoder; final Pydantic and business validation remain authoritative | `tests/test_tool_argument_stream.py`, `tests/test_deepseek_tool_stream.py`, `tests/test_deepseek_structured_output.py` |
| V2 thread baseline and next-call prediction | `ThreadContextWindowV3` retained CE memory plus lifetime injection statistics | `tests/test_thread_context_window_v3.py`, `frontend/lib/thread-context-window-v3.test.ts`, `frontend/components/thread-context-capsule.test.tsx` |
| Request-local context percentage reset | Checkpoint-owned `SessionContextMemoryLedgerV1` and active-run V3 snapshot | `tests/test_session_context_memory.py`, `tests/test_provider_dispatch_memory.py`, `tests/test_streaming_context_v3_replacement_parity.py` |
| Transcript used directly as the only model history | Independent `ModelViewProjectionV1`, strict micro/full compaction, compact boundary recovery | `tests/test_model_view_projection.py`, `tests/test_full_compaction.py`, `tests/test_full_compaction_app.py`, `tests/test_compaction_llm.py` |
| False completion while profile input is pending | Persisted profile-completion interrupt with checkpoint resume | `tests/test_sse_lifecycle.py`, `tests/test_run_control.py` |

The focused replacement regression completed with `375 passed` before this
audit. The frontend V3/streaming regression previously completed with 71
Vitest tests, typecheck, ESLint, and production build. Phase 10 additionally
adds an active-run parity test that proves a V3 ledger with more than 50
request identities is preserved without generic run-control truncation.

### Approved deletion candidates for the verified cleanup commit

Candidate: Context Window V2 backend implementation
Files: `src/context_engineering/thread_window.py`, `app.py`,
`src/schemas.py`, `src/run_control.py`
Symbols: `ThreadContextWindowV2`, `build_thread_context_window_v2`,
`thread_context_window_v2`
Evidence: tracked-file reference scan finds the implementation only in the
module itself, `app.py`, the thread status schema/run-control cache, V2-only
tests, and frontend V2 parsing/state. No LangGraph node, provider registry,
prompt, configuration key, or dynamic import refers to the builder.
Confidence: High.
Replacement tests: `tests/test_thread_context_window_v3.py`,
`tests/test_session_context_memory.py`,
`tests/test_streaming_context_v3_replacement_parity.py`.
Approved action: delete the module, public/status fields, cache field, V2-only
tests, fixtures, parsers, and write-only frontend state in the isolated cleanup
commit.

Candidate: Legacy internal SSE `token` and `text` emissions
Files: `app.py` (`on_chat_model_stream` and non-streaming node-end branches),
`src/streaming/adapter.py`
Symbols: legacy payload types `token` and `text`,
`adapt_legacy_sse_stream`
Evidence: both browser pages import `consumeAgentStreamV2` and contain no
`case "token"`, `case "text"`, or manual `split("\\n\\n")` consumer. The
only conversion is the explicitly temporary server adapter used by
`StreamSession`.
Confidence: High after the native producer replaces the adapter.
Replacement tests: `tests/test_agent_stream_v2.py`,
`tests/test_stream_session.py`, frontend shared parser/client/reducer tests.
Approved action: make `StreamSession` consume native V2 event drafts, emit
content-block events directly from graph execution, then delete the adapter,
its export, and adapter-only tests. Do not delete the adapter before that native
producer is executable.

Candidate: Frontend Context Window V2 contract and write-only state
Files: `frontend/app/page.tsx`,
`frontend/lib/observability-contracts.ts`,
`frontend/lib/thread-context-window.test.ts`,
`frontend/test/observability-fixtures.ts`
Symbols: `ThreadContextWindowV2`, `parseThreadContextWindowV2`,
`threadContextWindowV2Payload`, `threadContextWindowV2` state
Evidence: page references only set/reset/parse this state; no rendered
component reads it. The visible capsule consumes `ThreadContextWindowV3`.
Confidence: High.
Replacement tests: `frontend/lib/thread-context-window-v3.test.ts`,
`frontend/components/thread-context-capsule.test.tsx`.
Approved action: delete the V2 interface/parser/fixture/test and all page state
updates in the isolated cleanup commit.

Candidate: V2-only backend tests and fixtures
Files: `tests/test_thread_context_window_v2.py`, V2 assertion in
`tests/test_observability_sse_v2.py`, legacy-token assertions in
`tests/test_sse_lifecycle.py`, `tests/test_agent_stream_adapter.py`
Evidence: each assertion targets a contract scheduled for deletion; the parity
map above identifies an executable V3 or native-stream replacement first.
Confidence: High, conditional on native producer tests passing.
Approved action: remove V2-only coverage and replace legacy SSE lifecycle
assertions with native event-draft/stream-session assertions. Never delete a
failing test solely to obtain green status.

### Explicitly retained boundaries

- `ContextItem`, Collect/Packing/Apply Policy, Provider Registry, Influence
  Ledger, and LLM Input Manifest.
- Provider input budgets, `ContextUsageReport`, complete checkpoint messages,
  durable `task_workspace`, run control, activity timeline, QA/resource final
  contracts, and all unrelated RAG work.
- Every older candidate already recorded in this report.

### Reference and tool evidence

- Tracked reference counts before cleanup: `app.py` 16 V2/adapter references;
  frontend main page 18; V2 frontend contract 6; V2 frontend test 9; V2 fixture
  2; backend V2 module 7; run control 1; schema 1; streaming adapter/session/API
  exports 6; V2/adapter tests 13.
- FastAPI route, LangGraph node, prompt, configuration, and dynamic import scans
  found no hidden V2 entry point.
- Both streaming pages have no manual `split("\\n\\n")` parser.
- `rg.exe` was unavailable with `Access is denied`; tracked-file PowerShell
  scans were used instead.
- Semgrep, Gitleaks, Bandit, import-linter, Vulture were not installed. They are
  recorded as missing, not passing. Manual import/route/config/prompt/test scans
  provide the deletion evidence for this scoped cleanup.

Review output manually and update this report. Do not automatically apply deletions.

## 2026-07-14 Resource Final V3 legacy cleanup evidence

This cleanup is explicitly authorized by the approved volunteer/Agent-node
zero-legacy plan. It is limited to Resource Final contracts and diagnostic
code already replaced by `resource_final_v3` and `agent_stream_v2`; the current
formal graph and checkpoint migration readers remain outside this deletion.

Candidate: Resource Final V1/V2 compatibility projection
Files: `src/graph/resource_final.py`, `app.py`,
`tests/test_resource_final_contract.py`, `tests/test_app.py`,
`tests/test_sse_lifecycle.py`
Symbols: `normalize_resource_final_payload`,
`completed_without_resource_payload`, `_legacy_resource_final_payload`,
`STRUCTURED_RESOURCE_ARTIFACT_KEYS`, `resource:v1`, `payload:v1`, and
`completed_without_resource`
Evidence: `resource_bundle_output` now constructs a validated
`ResourceFinalV3` directly. The stream reads only `resource_final_v3`, checks
thread/request identity, and emits a typed `resource_final_v3_missing`
`stream_error` when resource execution reaches a terminal state without the
authoritative contract. Non-resource evidence summaries now construct and
business-validate `QAResponse` and terminate through `qa_final`.
Confidence: High.
Dynamic reference checks: tracked import, FastAPI route, LangGraph node,
configuration, prompt, frontend consumer, fixture, and test scans found no
remaining runtime import of `src.graph.resource_final` after this diff.
Replacement tests: `tests/test_resource_final_v3_contract.py`,
`tests/test_resource_final_runtime.py`, `tests/test_sse_lifecycle.py`,
`tests/test_postgres_persistence_integration.py`, and frontend Resource Final
V3 parser/reducer tests.
Approved action: delete the compatibility module and V1/V2-only tests; retain
strict V3 schema, runtime builder, checkpoint payload, identity checks, and
typed failure tests.

Candidate: obsolete SSE bubble comparison utility
File: `scripts/compare_sse_bubble_output.py`
Symbols/events: `_simulate_frontend_bubble`, `_event_summary`, `token`, `text`,
`mindmap_result`, and legacy unversioned `resource_final`
Evidence: the file is not imported, registered as a CLI entry point, referenced
by tests, or referenced by documentation. It implements a second hand-written
SSE parser and simulates the deleted browser event branches, so running it can
only assess a contract the application no longer exposes.
Confidence: High.
Dynamic reference checks: tracked filename, report-path, import, FastAPI,
prompt/config, and test scans found no references outside the file itself.
Replacement tests: `frontend/lib/sse-parser.test.ts`,
`frontend/lib/agent-stream-client.test.ts`, `frontend/lib/live-turn.test.ts`,
`tests/test_agent_stream_v2.py`, and `tests/test_stream_session.py`.
Approved action: delete the unreferenced utility in this isolated legacy
cleanup; do not replace it with another private SSE implementation.

Deletion boundary: `src/graph/resource_final_v3.py`,
`src/graph/resource_final_runtime.py`, Resource Final V3 frontend rendering,
`task_workspace`, complete checkpoints, run control, activity timeline, and
all production-switch migration readers are explicitly retained.

Verification after deletion:

- Active-code scans across `app.py`, `src`, `frontend`, `scripts`, `tests`, and
  `config` have no matches for `_legacy_resource_final_payload`,
  `normalize_resource_final_payload`, `completed_without_resource`,
  `resource_final_diagnostic`, `mindmap_result`, `review_doc_result`,
  `resource:v1`, `payload:v1`, or imports of the deleted module.
- Focused Resource Final/stream/checkpoint regression: `287 passed, 1 skipped`.
- Full backend regression after migrating the final Phase-0 fixture:
  `2280 passed, 5 skipped`.
- Frontend: 23 Vitest files / 69 tests, source ESLint, full `npm run lint`,
  typecheck, and production build passed. The build manifest contains no
  `/volunteer` route.
- `python -m compileall -q src tests app.py`, touched-file Ruff check/format,
  three-file Resource Final V3 scoped mypy, eight security tests, and
  `git diff --check` passed.
- Full-repository Ruff remains outside this cleanup: `ruff check .` reports
  60 pre-existing findings and `ruff format --check .` reports 66 pre-existing
  files. The optional Semgrep, import-linter, Gitleaks, Bandit, and Vulture
  executables are missing and were not reported as passing.

## 2026-07-14 Context Apply inert fallback-field cleanup evidence

Candidate: inert Context Apply fallback policy and telemetry fields
Files: `src/context_engineering/packing/apply.py`,
`src/context_engineering/packing/importance.py`,
`src/context_engineering/packing/apply_trace.py`,
`src/context_engineering/packing/node_policy.py`, `app.py`, and
`config/settings.yaml`
Symbols: `fallback_on_error`, `fallback_if_empty_after_drop`, and
`fallback_to_rule_based`
Evidence: tracked reference inspection found no conditional or alternative
execution for the first two fields. The third field was copied from policy to
failure telemetry but never invoked rule-based selection. Official settings
fixed all three to false. Removing them leaves budget degradation, source/drop
accounting, importance scoring success/failure, and provider retry unchanged.
Confidence: High.
Dynamic reference checks: no Pydantic alias, environment resolver, prompt,
LangGraph node lookup, FastAPI route, or frontend parser accesses these names.
All active source/config/test references are removed in this diff.
Replacement tests: Context Apply message, policy, node-policy, budget,
route-rollout, plain/structured LLM, trace, importance, Phase-3B boundary, and
SSE lifecycle suites (`303 passed`). Full backend regression passed with
`2280 passed, 5 skipped`.
Approved action: delete fields, parsers, defaults, telemetry projection, and
fixture arguments. Retain explicit typed failures, dropped reasons,
observe-only scoring, and same-provider bounded transport retry.

## 2026-07-14 legacy memory prompt replacement evidence

Candidate: legacy memory prompt construction and token budget layer
Files: `src/context/context_builder.py`, `src/context/token_manager.py`,
`src/context/errors.py`, `src/context/__init__.py`, `src/memory/prompts.py`,
`src/memory/schema.py`, `src/memory/__init__.py`, `config/settings.yaml`,
`tests/test_context_builder.py`, and `tests/test_token_budget_strict_config.py`
Symbols/config: `build_memory_context`, `format_memory_influence_explanation`,
`MemoryContextInjection`, `MEMORY_CONTEXT_*`,
`MEMORY_INFLUENCE_EXPLANATION_TEMPLATE`, and `memory.token_budget`
Evidence: `generate_answer` now delegates all memory/profile/rules injection to
the active Context Engineering node policy. The production-path replacement
test proves that already-retrieved conversation, episodic, semantic, and
profile state reaches the provider through one CE block and generates
content-free dispatch descriptors. Explicit ignore, pending confirmation, and
cross-thread memory do not reach the provider. Required rules keep the active
path valid when optional memory/profile are absent. The provider does not
import retrieval or embedding modules.
Confidence: High after focused replacement tests; deletion remains a separate
commit. The replacement snapshot now passes 397 focused tests, the full
backend suite (`2297 passed, 5 skipped`), frontend tests/typecheck/lint/build,
compileall, touched Ruff, CE scoped mypy, security tests, and diff check.
Dynamic reference checks: after the replacement diff, active runtime use of
these symbols is confined to the legacy `src/context` package and its exports;
`MemoryContextInjection` and the old prompt constants have no independent
consumer. The old tests and `memory.token_budget` validate only that legacy
layer. `src.memory` storage, retrieval, consolidation, schemas unrelated to
`MemoryContextInjection`, and top-level memory import smoke coverage must be
retained or moved before deletion.
Replacement tests: `tests/test_generate_answer_context_engineering.py`,
`tests/test_memory_context_provider.py`,
`tests/test_profile_rules_providers.py`,
`tests/test_context_influence_ledger.py`, `tests/test_builder.py`,
`tests/test_model_view_projection.py`, session ledger tests, and stream/app
ledger update tests.
Executed action: after replacement snapshot `ed953ac` passed all gates, the
listed legacy package/schema/constants/config/tests were deleted in an
independent cleanup diff. The retained `src.memory` public API smoke moved to
`tests/test_memory_public_api.py`; a dedicated absence guard prevents the old
package, symbols, and configuration from returning. Retain
`src/context_engineering`, `src/memory` business storage/retrieval, complete
transcript/checkpoints, compaction, Context Window V3, and same-provider retry.

Post-deletion verification: 161 focused tests and the final full backend suite
(`2279 passed, 5 skipped`) passed. Frontend Vitest/typecheck/lint/build,
compileall, touched Ruff, retained-memory scoped mypy, security tests, diff
check, and exact active-code/config/test scans passed. The first full run found
two stale phase guards that still required the deleted 4096 budget; they were
converted to absence regressions rather than deleted. Optional Semgrep,
import-linter, Gitleaks, Bandit, and Vulture remain unavailable and were not
reported as passing.
