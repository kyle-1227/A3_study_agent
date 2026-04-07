# MILESTONE 2 — Backend Logic Complete

> **Status**: DONE | **Date**: 2026-04-05 | **Task**: BE-05

---

## What was done

### BE-05: Update `app.py` (SSE events, constants, CORS strip, error handling)

**File modified**: `app.py`

#### 1. Constant sets updated (AC-01 Step 6)

- **`ALLOWED_NODES`** (line 78): Removed `"plan_adversarial"`, added `"drafter"`.
  - Final: `{"generate_answer", "drafter", "emotional_response"}`

- **`TEXT_EMIT_NODES`** (line 81): New constant added.
  - Final: `{"plan_output", "handle_unknown"}`

- **`GRAPH_NODES`** (lines 84-102): Removed `"plan_adversarial"`, added 6 flattened adversarial nodes + `"rewrite_query"`.
  - Final: 17 nodes — `supervisor`, `academic_router`, `rag_retrieve`, `web_search`, `generate_answer`, `evaluate_hallucination`, `rewrite_query`, `search_policy`, `gather_intel`, `drafter`, `reviewer_academic`, `reviewer_emotional`, `consensus_check`, `adv_rewrite`, `plan_output`, `emotional_response`, `handle_unknown`

#### 2. `"text"` SSE event emission (AC-02, BUG-02/BUG-04)

- Added in `_stream_graph_events`, within the `on_chain_end` branch (lines 158-168).
- After yielding the `node_event` end payload, checks if `node_name in TEXT_EMIT_NODES`.
- Extracts `messages` from the node output and emits `{"type": "text", "content": ..., "node": ...}` for each AIMessage with content.

#### 3. `"done"` SSE event (BUG-09)

- Added at line 221, after the interrupt check block.
- Only emitted when the stream completes normally (no interrupt, no error).

#### 4. Error SSE event (SEC-03)

- Wrapped the entire `astream_events` loop in `try/except Exception` (lines 118-206).
- On unhandled exception: logs via `logger.exception()`, emits `{"type": "error", "message": str(e)}`, then returns.

#### 5. CORS whitespace stripping (SEC-02)

- Line 71: Changed from `.split(",")` to list comprehension with `.strip()` and empty-string filter.

---

## Verification

### Constant sets match design doc

| Constant | Design Doc | Implementation | Match |
|----------|-----------|----------------|-------|
| `ALLOWED_NODES` | `{"generate_answer", "drafter", "emotional_response"}` | Same | YES |
| `TEXT_EMIT_NODES` | `{"plan_output", "handle_unknown"}` | Same | YES |
| `GRAPH_NODES` | 17 nodes (Section 4, AC-01 Step 6) | 17 nodes | YES |

### Test results

```
pytest tests/test_app.py -v
======================= 12 passed, 1 warning in 46.01s ========================
```

All 12 tests pass:
- `TestCORSConfiguration` (3 tests) — PASSED
- `TestNoGlobalGraph` (3 tests) — PASSED
- `TestPyprojectToml` (5 tests) — PASSED
- `TestEnvExample` (1 test) — PASSED

---

## SSE Protocol — Current State

```
Existing events (unchanged):
  {"type": "token",      "content": "..."}
  {"type": "node_event", "node": "...", "status": "start"|"end", "duration_ms": N, "error": null}
  {"type": "usage",      "node": "...", "input_tokens": N, "output_tokens": N, "total_tokens": N}
  {"type": "thread_id",  "thread_id": "..."}
  {"type": "interrupt",  "draft": "...", "thread_id": "..."}

New events (v0.3.0):
  {"type": "text",  "content": "...", "node": "..."}    — complete text from non-streaming nodes
  {"type": "done"}                                       — stream completion marker
  {"type": "error", "message": "..."}                    — unhandled error during streaming
```

---

## Next step

**BE-07 (Test Migration)** — awaiting user instruction to proceed.
