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
