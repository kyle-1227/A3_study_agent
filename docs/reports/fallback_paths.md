# Fallback Paths Report

Initial governance report created on 2026-06-20. This is a report-only baseline. No runtime code was changed or deleted.

## Policy

- Do not add new fallback.
- Do not add silent defaults.
- Existing fallback/helper paths are legacy risk until reviewed.
- Discovery does not imply deletion approval.

## Observed Legacy Risk Areas

### 2026-07-14 Context Apply inert fallback-field cleanup

- Removed the unused Context Apply policy fields `fallback_on_error` and
  `fallback_if_empty_after_drop` from dataclasses, configuration parsing,
  official settings, and test fixtures. Neither field had a runtime branch;
  storing them only advertised behavior that did not exist.
- Removed `fallback_to_rule_based` from importance-scoring policy, aggregate
  telemetry, trace projection, SSE progress, settings, and tests. Importance
  scoring failure still reports typed reason, error type, sanitized warnings,
  and elapsed time; it does not execute or claim a rule-based substitute.
- Whole-item budget degradation, source filtering, drop reasons, required
  source failures, observe-only importance scoring, and same-provider
  transport retry are unchanged.
- Active-code/config/test scans contain none of the three field names. Related
  Context Apply regression passed with 303 tests; full backend regression
  passed with 2280 tests and 5 skips.

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
- The later single-provider cleanup removed `get_fallback_llm`,
  `invoke_with_fallback`, `async_invoke_with_fallback`, and their `FALLBACK_*`
  environment contract. `tests/test_llm_single_provider_retry.py` now guards
  their absence while preserving bounded same-provider transport retry.
- Production LLM paths must remain behind manifest-guarded transport and must
  not reintroduce cross-provider or cross-model retry.

### 2026-06-29 Context Engineering Phase 0

- Phase 0 target fallback cleanup and deferred findings are recorded in `docs/reports/context_engineering_phase0_audit.md`.
- Removed production fake-success paths for assessment classification and memory embeddings/retrieval.
- Deferred non-target legacy fallback areas remain report-only pending separate specs.

### 2026-06-28 Run Control implementation note

- Run Control touched `src/graph/llm.py` and `src/llm/structured_output.py` only to emit context-window telemetry.
- Existing provider/model fallback/default paths in those files were observed as legacy risk and were not removed or extended.
- Context usage telemetry reports `context_usage_error` when budget configuration or model windows are missing, rather than fabricating fallback window data.

### `src/graph/llm.py`

- Cross-model fallback helpers are removed. `get_node_llm` and
  `get_primary_llm` still contain legacy provider/model/base-url defaults and
  OpenRouter header handling; those configuration paths remain separate
  report-only debt and were not changed by the Context Apply cleanup.

### `src/llm/structured_output.py`

- Structured output runtime is provider-neutral in intent but imports the graph LLM bridge.
- Existing retry/fallback behavior must not be extended without explicit design review.
- This file is protected and was not modified in this governance round.

### `src/graph/academic.py`

- Multiple evidence and sufficiency paths contain fallback/debug fields and deterministic sufficiency fallback references.
- Existing traces include fallback-chain concepts that should remain visible rather than silent.
- This file is protected and was not modified in this governance round.

### Tests

- `tests/test_llm_single_provider_retry.py`,
  `tests/test_deepseek_structured_output.py`, and
  `tests/test_structured_retry.py` are relevant when changing provider retry
  or structured-output behavior.

## Follow-Up

1. Run Semgrep guard and classify findings as legacy or new.
2. Decide which fallback paths are intentional, deprecated, or removal candidates.
3. Create separate specs before any cleanup.
4. Do not remove fallback paths in unrelated feature work.

### 2026-07-14 generate_answer memory injection replacement

- Removed the silent `try/except` around the second, node-local memory
  retrieval/prompt builder. `generate_answer` now has one provider-input path:
  active Context Engineering selection followed by the manifest-guarded
  provider dispatch.
- Context Apply remains fail closed. Missing required rules, invalid policy,
  packing failure, identity mismatch, or provider failure is not converted to
  an original-message call or plausible answer.
- `memory_use_policy=ignore` and pending `ask_user` prevent memory collection;
  cross-thread memory is rejected by source policy. No fallback, silent
  default, provider/model override, or OpenRouter DeepSeek path was added.
- The user-visible legacy memory footer was removed. Safe source counts are
  exposed only as content-free Influence Ledger metadata and provider dispatch
  records; memory bodies are not written to those audit surfaces.
- After the verified replacement snapshot, the now-unreferenced `src/context`
  builder/token-budget/error package and its tests were deleted. No
  compatibility import, original-message retry, deterministic memory text, or
  legacy budget adapter remains; retained memory storage/retrieval is unchanged.

### 2026-07-14 Supervisor phrase-routing cleanup

- Removed the private resource-request phrase tables and their singular/plural
  detectors. They were not a transport fallback, but they represented a second,
  query-text-based interpretation path beside the strict structured contract.
- `supervisor_node` now has one auditable routing source:
  `SupervisorOutput` produced by `invoke_structured_llm`, validated by Pydantic
  and `validate_supervisor_output`, then projected into graph state.
- A strong resource-generation phrase paired with a valid `unknown/general` QA
  structured result remains QA and reaches the `qa` route; query text is not
  reparsed to manufacture a resource request.
- No fallback, silent default, alias-normalization bypass, provider/model
  override, or validation bypass was introduced. `_sanitize_valid_intents`
  remains an independently active import-time configuration concern and was
  intentionally not changed in this cleanup.
