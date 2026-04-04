# REQ-08: Human-in-the-loop Interrupt/Resume — Backend Ready

## Summary

Backend HIL implementation complete. The planner branch now pauses after adversarial review for human plan editing, then resumes execution with the edited plan.

## Files Changed

| File | Change |
|------|--------|
| `src/graph/planner.py` | Added `interrupt()` call in `plan_adversarial_node` after SubGraph completes. Imports `interrupt` from `langgraph.types`. |
| `src/schemas.py` | Added `ResumeRequest(thread_id: str, edited_plan: str)` Pydantic model. |
| `app.py` | Refactored SSE streaming into `_stream_graph_events()` shared helper. `generate_sse()` now emits `thread_id` event first and checks for interrupts after streaming. Added `generate_resume_sse()` and `POST /resume` endpoint. Updated `GRAPH_NODES` (added `gather_intel`, `plan_adversarial`, `handle_unknown`) and `ALLOWED_NODES` (replaced `generate_plan` with `plan_adversarial`). |
| `tests/test_hil_interrupt.py` | **NEW** — 14 tests covering interrupt call, resume value usage, fallback, LangGraph integration, SSE thread_id/interrupt events, resume streaming, Command usage, endpoint wiring, schema validation. |
| `tests/test_sse_lifecycle.py` | Updated all tests for: (1) `aget_state` mock requirement, (2) `thread_id` first-event offset, (3) new `ALL_NODES` list. |
| `tests/test_checkpointer.py` | Updated `TestSSEWithConfig` to mock `aget_state`. |

## How to Run Tests

```bash
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short
```

Result: **384 passed, 1 skipped, 0 failed** (18 new tests: 14 HIL + 4 new node parameterizations)

## Git Add + Commit Suggestion

```bash
git add src/graph/planner.py src/schemas.py app.py \
        tests/test_hil_interrupt.py tests/test_sse_lifecycle.py tests/test_checkpointer.py \
        context_prompts/backend/reports/REQ-08_hil_backend_ready.md

git commit -m "feat(hil): Human-in-the-loop interrupt/resume (REQ-08)

- planner.py: interrupt() pauses after adversarial plan review
- app.py: POST /resume endpoint, SSE interrupt+thread_id events
- schemas.py: ResumeRequest model
- 384 tests pass (18 new)"
```

---

## Frontend Coordination: SSE Event Schema + /resume API Spec

### New SSE Event Types

#### 1. `thread_id` (emitted first on every `/stream` call)

```json
{"type": "thread_id", "thread_id": "uuid-string"}
```

Frontend MUST store this `thread_id` — it is required for the `/resume` call.

#### 2. `interrupt` (emitted when graph pauses for human review)

```json
{
  "type": "interrupt",
  "draft": "## 学习计划\n- 周一：数学 ...",
  "thread_id": "uuid-string"
}
```

When frontend receives this event:
1. Display `draft` in an editable textarea
2. Show a "Confirm Plan" / "Edit & Submit" button
3. On user confirmation, POST to `/resume`

### POST /resume Endpoint

**URL:** `POST /resume`
**Content-Type:** `application/json`
**Request Body:**

```json
{
  "thread_id": "the-thread-id-from-interrupt-event",
  "edited_plan": "## User's edited plan text..."
}
```

**Response:** `text/event-stream` (SSE) — streams remaining graph events after the interrupt point (typically just the `plan_adversarial` node end event).

### Flow Diagram

```
Frontend                    Backend
   |                           |
   |-- POST /stream ---------->|
   |<-- SSE: thread_id --------|  (store thread_id)
   |<-- SSE: node_events ------|  (supervisor, search_policy, gather_intel, ...)
   |<-- SSE: interrupt --------|  (show draft in textarea)
   |                           |
   | [user edits plan]         |
   |                           |
   |-- POST /resume ---------->|  (thread_id + edited_plan)
   |<-- SSE: node_events ------|  (plan_adversarial end, etc.)
   |<-- SSE: [stream ends] ----|
```

### Notes for Frontend Agent

- If user does NOT edit the plan and clicks "Confirm", send the original `draft` as `edited_plan`
- If user sends empty `edited_plan`, backend uses the original draft (fallback)
- The `/resume` response is also SSE — use the same event handler as `/stream`
- `thread_id` is required for `/resume`; without it the backend cannot locate the interrupted state
