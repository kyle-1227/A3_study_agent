# Fallback Paths Report

Initial governance report created on 2026-06-20. This is a report-only baseline. No runtime code was changed or deleted.

## Policy

- Do not add new fallback.
- Do not add silent defaults.
- Existing fallback/helper paths are legacy risk until reviewed.
- Discovery does not imply deletion approval.

## Observed Legacy Risk Areas

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
