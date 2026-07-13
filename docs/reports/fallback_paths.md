# Fallback Paths Report

Initial governance report created on 2026-06-20. This is a report-only baseline. No runtime code was changed or deleted.

## Policy

- Do not add new fallback.
- Do not add silent defaults.
- Existing fallback/helper paths are legacy risk until reviewed.
- Discovery does not imply deletion approval.

## Observed Legacy Risk Areas

### 2026-07-13 Structured-output single-mode cleanup

- Removed the legacy `fallback_modes` configuration/API/result/trace contract
  and `get_fallback_modes` helper from the structured-output runtime.
- Each `invoke_structured_llm` call now supplies one explicit `output_mode`;
  retryable parse, schema, and business-validation failures retry only that
  mode, while provider transport retry remains unchanged.
- Removed the now-constant structured-output result flags `fallback_used` and
  `default_used`. Context Apply telemetry and resource-generator fallback
  fields are separate contracts and were not changed in this cleanup.

### 2026-07-10 Parent-child RAG implementation baseline

- `src/rag/retriever.py` catches BM25 construction failures and continues with
  vector-only retrieval. The parent-child candidate path must not call this
  helper and must surface a typed BM25 failure instead.
- `src/rag/reranker.py` catches provider and protocol failures and returns the
  original document order. The parent-child candidate path must use a strict
  reranker boundary and must not treat the original order as a successful
  rerank result.
- Existing `get_setting(..., default)` and environment fallbacks in the legacy
  RAG index/retrieval path remain baseline-only risk. New generation code must
  receive a fully validated configuration object with no production defaults.
- These findings are report-only. No legacy fallback was removed as part of
  the parent-child RAG foundation work.

### 2026-07-12 Performance observability baseline

- `src/tracing/collector.py` retains the legacy OpenTelemetry SQLite fallback
  exporter and environment defaults. The new request-span performance registry
  is independent of that exporter and does not treat exporter availability as
  timing evidence.
- This remains report-only: Phase 3 adds content-free span collection and
  reports incomplete coverage when dependencies are unavailable; it does not
  remove, extend, or rely on the legacy exporter fallback.
- `src/tools/search_tool.py` retains a legacy configuration default for the
  Tavily API-key environment variable. Phase 3 only wrapped the existing
  request boundary in a content-free timing span; it did not extend the
  default, add a search fallback, or alter retrieval behavior.

### 2026-07-08 LLM Input Manifest enforcement

- Added manifest enforcement for active provider transport paths before any
  generated/chat-completion LLM request is sent.
- Existing `invoke_with_fallback()` and `async_invoke_with_fallback()` remain
  report-only legacy helpers because they are covered by historical tests and
  are not used by production graph call sites.
- New production LLM paths must not call these legacy fallback helpers; direct
  provider invocation should stay behind manifest-guarded transport.

### 2026-06-29 Context Engineering Phase 0

- Phase 0 target fallback cleanup and deferred findings are recorded in `docs/reports/context_engineering_phase0_audit.md`.
- Removed production fake-success paths for assessment classification and memory embeddings/retrieval.
- Deferred non-target legacy fallback areas remain report-only pending separate specs.

### 2026-06-28 Run Control implementation note

- Run Control touched `src/graph/llm.py` and `src/llm/structured_output.py` only to emit context-window telemetry.
- Existing provider/model fallback/default paths in those files were observed as legacy risk and were not removed or extended.
- Context usage telemetry reports `context_usage_error` when budget configuration or model windows are missing, rather than fabricating fallback window data.

### `src/graph/llm.py`

- Module docstring and helpers describe resilient fallback/failover behavior.
- `get_node_llm`, `get_primary_llm`, and `get_fallback_llm` include provider/model/base_url/api_key defaults.
- Fallback model/env behavior appears coupled to DeepSeek defaults.

### `src/llm/structured_output.py`

- Structured output runtime is provider-neutral in intent but imports the graph LLM bridge.
- Existing retry/fallback behavior must not be extended without explicit design review.
- This file is protected and was not modified in this governance round.

### `src/graph/academic.py`

- Multiple evidence and sufficiency paths contain fallback/debug fields and deterministic sufficiency fallback references.
- Existing traces include fallback-chain concepts that should remain visible rather than silent.
- This file is protected and was not modified in this governance round.

### Tests

- `tests/test_llm_fallback.py`, `tests/test_deepseek_structured_output.py`, and `tests/test_structured_retry.py` are relevant when changing fallback or structured-output behavior.

## Follow-Up

1. Run Semgrep guard and classify findings as legacy or new.
2. Decide which fallback paths are intentional, deprecated, or removal candidates.
3. Create separate specs before any cleanup.
4. Do not remove fallback paths in unrelated feature work.
