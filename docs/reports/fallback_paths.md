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

### 2026-07-14 Assessment journal foundation

- The assessment idempotency journal stores only a stable request hash and the
  strict public terminal. It does not retain request content or private answer
  keys as a replay fallback.
- Invalid restored state, thread/request/final identity drift, journal
  conflicts, capacity exhaustion, callback failure, or an append that is not
  visible on reread all fail closed. No process-local success object is returned
  as a substitute for missing durable state.
- Operation failures are not cached. Retrying re-executes the same explicit
  operation; it does not return a deterministic assessment, neutral score, or
  fabricated adaptive practice task.
- This foundation does not call the legacy placeholder practice generator and
  does not claim the missing FastAPI/provider/PostgreSQL integration is
  complete.

### 2026-07-14 Strict assessment endpoint and failure replay

- The endpoint uses one configured provider/model/output mode for each of
  `error_classifier` and `practice_generator`. It adds no alternate provider,
  alternate model, deterministic answer, neutral score, fabricated task, alias
  repair, or validation bypass.
- The journal foundation note above is superseded for operation failures:
  before provider dispatch the runtime now persists a content-free
  `in_progress` claim. Ordinary failure becomes a content-free `failed`
  terminal and replays without another dispatch. Cancellation/crash leaves
  `in_progress` and returns `assessment_request_recovery_required`; it does not
  silently clear the claim or retry the provider.
- A user may explicitly start a new request with a new UUID after reviewing a
  failed/recovery-required result. The server never changes the UUID, answer,
  or request hash on the user's behalf.
- Same-provider bounded transport and structured semantic retries remain the
  existing configured mechanisms. `sensitive_trace=True` changes diagnostics
  only: it removes content-bearing trace fields and is not a model, provider,
  parser, or business-result fallback.
- Real PostgreSQL and real Provider E2E remain explicit delivery gates. The
  opt-in PostgreSQL concurrency test is skipped when
  `A3_TEST_POSTGRES_URI` is absent and is not reported as passing.

### 2026-07-15 Learning-guidance strict storage reads

- Legacy `SQLiteProfileStore.load()` initializes missing storage, catches all
  profile JSON/schema failures, logs them, and returns `None`. It remains for
  existing consumers and was not deleted or changed in this feature batch.
- Legacy episodic row deserialization converts malformed metadata JSON to an
  empty object and malformed embedding JSON to `None`. Existing memory
  consumers retain that behavior; the production learning-guidance adapter is
  forbidden from calling it.
- Learning guidance now has separate concrete SQLite read methods that open an
  existing database read-only, validate required tables and columns, reject
  duplicate/non-finite JSON and schema drift, and expose content-safe typed
  failures. These methods do not create storage and do not turn damaged rows
  into unavailable profile/history results.
- The strict API is intentionally not added to the abstract legacy store
  interfaces. Production runtime composition must inject the concrete strict
  adapters explicitly; it must not fall back to the legacy methods.

### 2026-07-15 Candidate resource title provenance

- `src/graph/resource_generation.py::_resource_title` retains a pre-existing
  display fallback that substitutes the canonical `resource_type` when neither
  the primary artifact nor its title state contains a title.
- The candidate automatic-recommendation path now binds recommendations to the
  generated Resource Final V3 title. That path must not treat the substituted
  resource type as verified title provenance.
- Candidate activation remains closed. Before rollout, resource generators must
  emit a contract-validated title or the candidate worker must fail with a typed
  missing-title reason; the legacy substitution must not become recommendation
  evidence. This report does not remove or extend the legacy behavior.
