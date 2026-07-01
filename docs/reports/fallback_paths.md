# Fallback Paths Report

Initial governance report created on 2026-06-20. This is a report-only baseline. No runtime code was changed or deleted.

## Policy

- Do not add new fallback.
- Do not add silent defaults.
- Existing fallback/helper paths are legacy risk until reviewed.
- Discovery does not imply deletion approval.

## Observed Legacy Risk Areas

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
