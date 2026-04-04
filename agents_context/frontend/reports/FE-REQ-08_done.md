# FE-REQ-08: Human-in-the-loop Interrupt UI

**Status:** DONE
**Date:** 2026-03-30

## Files Changed

| File | Change |
|------|--------|
| `frontend/app/page.tsx` | Major refactor: extracted SSE processing into `processSSEEvent()` + `consumeSSEStream()` helpers shared by `/stream` and `/resume`; extracted `fetchWithErrorHandling()` for shared HTTP error logic; added HIL state (`threadId`, `interruptDraft`, `isInterrupted`, `isResuming`); handles `thread_id` and `interrupt` SSE events; implemented `handleResume` callback that POSTs to `/resume` and consumes the resume SSE stream; renders `<PlanReview>` component when interrupted |
| `frontend/components/plan-review.tsx` | **NEW** — editable textarea for draft plan review; shows character count; "确认计划" / "修改后确认" button with loading state; "重置" button when modified |
| `frontend/components/right-panel.tsx` | Added `isInterrupted` prop; renders "等待用户审批" status banner with pulsing indicator when graph is paused; added `gather_intel`, `plan_adversarial`, `handle_unknown` to `NODE_LABELS` map |

## Architecture

```
User sends message
    ↓
POST /stream (with X-Access-Token)
    ↓
SSE: thread_id → stored in ref
SSE: node_event, token, usage → normal rendering
SSE: interrupt → { draft, thread_id }
    ↓
isInterrupted=true → PlanReview renders below ChatArea
Right panel shows "等待用户审批"
    ↓
User edits plan → clicks "确认计划"
    ↓
POST /resume { thread_id, edited_plan }
    ↓
Resume SSE stream → same processSSEEvent handler
    ↓
Stream completes → isLoading=false
```

## Key Design Decisions

1. **Shared SSE processor** — `processSSEEvent()` is a single function that handles all event types for both `/stream` and `/resume`, avoiding duplication.
2. **Ref for threadId** — used `useRef` instead of state to avoid stale closures inside the async stream loop.
3. **PlanReview placement** — rendered between ChatArea and bottom edge, inside a flex-col wrapper, so it overlays the input area when the graph is paused.
4. **Auth header** — extracted `getAuthHeaders()` utility used by both `/stream` and `/resume` requests.

## How to Verify

```bash
# Start backend (must have REQ-07 + REQ-08 merged)
cd gaokao_tutor && python -m uvicorn app:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
# Open http://localhost:3000

# E2E test flow:
# 1. Type a planner-intent query, e.g. "帮我制定一个数学复习计划"
# 2. Observe node trail in right panel: supervisor → search_policy → gather_intel → plan_adversarial
# 3. When interrupt event arrives: PlanReview textarea appears with draft plan
# 4. Right panel shows "等待用户审批" with pulsing indicator
# 5. Edit the plan text (or not) → click "确认计划"
# 6. Resume SSE streams remaining events → "Stream complete"
```

## Suggested Commit

```bash
git add frontend/app/page.tsx frontend/components/plan-review.tsx frontend/components/right-panel.tsx
git commit -m "feat(frontend): FE-REQ-08 Human-in-the-loop plan review UI

- page.tsx: SSE interrupt/resume flow with shared event processor
- plan-review.tsx: editable plan textarea + confirm/reset
- right-panel.tsx: interrupt status banner + new node labels"
```
