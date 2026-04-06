# MILESTONE BE-ENH: Backend Enhancement Complete

**Date**: 2026-04-06
**Branch**: `feat/HIL`
**Status**: COMPLETE

---

## Summary

All 6 backend enhancement tasks (BE-ENH-01 through BE-ENH-06) have been implemented and verified. The HIL Feedback Loop (Feature A) is now fully functional on the backend.

---

## Task Completion

### BE-ENH-01: State Extension
**File**: `src/graph/state.py`
- Added 4 new fields to `TutorState`: `hil_action`, `hil_feedback`, `hil_summary`, `feedback_route`
- Total TutorState fields: **26** (was 22)

### BE-ENH-02: Modify `plan_output_node`
**File**: `src/graph/plan_adversarial.py`
- `plan_output_node` now handles both string (confirm) and dict (feedback) responses from `interrupt()`
- Returns `hil_action="feedback"` + `hil_feedback` for dict input
- Returns `hil_action="confirm"` + `plan` + `messages` for string input
- Backward compatible with existing frontend

### BE-ENH-03: Add `feedback_router`, `plan_tweak_node`, and routing functions
**File**: `src/graph/plan_adversarial.py`
- `FeedbackClassification` Pydantic model: validates `route` as `"tweak"` or `"rewrite"`
- `feedback_router`: classifies feedback via supervisor's LLM with structured output, updates `hil_summary`, clears adversarial state on rewrite
- `plan_tweak_node`: single LLM call for surgical draft edits, uses planner model
- `route_after_hil`: routes `plan_output` to END (confirm) or `feedback_router` (feedback)
- `route_feedback`: routes `feedback_router` to `plan_tweak` (tweak) or `drafter` (rewrite)

### BE-ENH-04: Rewire builder.py
**File**: `src/graph/builder.py`
- Added `feedback_router` and `plan_tweak` nodes
- Replaced `plan_output → END` with conditional edges:
  - `plan_output → END` (confirm) or `plan_output → feedback_router` (feedback)
  - `feedback_router → plan_tweak` (tweak) or `feedback_router → drafter` (rewrite)
  - `plan_tweak → plan_output` (loop back for review)
- Graph compiles with **19 nodes** (was 17)

### BE-ENH-05: Update app.py + schemas
**Files**: `app.py`, `src/schemas.py`
- `ResumeRequest.edited_plan` changed from required to optional (default `""`)
- Added `ResumeRequest.feedback: str | None` with `max_length=4096`
- `generate_resume_sse` now accepts `feedback` parameter
- When `feedback` is provided: `Command(resume={"action":"feedback","text":"..."})` 
- When `feedback` is None: `Command(resume=edited_plan)` (backward compatible)
- `ALLOWED_NODES` updated: added `plan_tweak`
- `GRAPH_NODES` updated: added `feedback_router`, `plan_tweak` (19 entries)

### BE-ENH-06: Tests
**Files**: `tests/test_plan_adversarial.py`, `tests/test_builder.py`, `tests/test_sse_lifecycle.py`, `tests/test_hil_interrupt.py`, `tests/test_hil_summary.py`

New test classes added:
- `TestFeedbackClassification` — Pydantic model validation (tweak/rewrite/invalid)
- `TestFeedbackRouter` — tweak preserves draft, rewrite clears state, fallback on error
- `TestPlanTweakNode` — returns modified draft
- `TestPlanOutputNodeFeedback` — dict input sets hil_action/hil_feedback
- `TestPlanOutputNodeConfirm` — string input sets confirm/plan
- `TestRouteAfterHil` — feedback/confirm/missing routing
- `TestRouteFeedback` — tweak/rewrite/missing routing
- `TestHilSummaryEmpty` — summary creation from empty state
- `TestHilSummaryExisting` — summary compression with history
- `TestHilSummaryTruncation` — bounded growth verification
- Resume SSE feedback tests — dict vs string Command construction
- ResumeRequest schema tests — defaults, feedback field, max_length

Updated existing tests:
- `_base_state()` in `test_plan_adversarial.py` — includes 4 new fields
- `test_builder.py` — expected nodes set updated to 19
- `test_sse_lifecycle.py` — `ALL_NODES` list updated to 19
- `test_hil_interrupt.py` — `generate_resume_sse` calls updated for new signature

---

## Verification

### Test Results
```
pytest tests/ -v --tb=short
440 passed, 1 skipped, 9 errors (pre-existing fixture issues in test_integration.py)
```

The 9 errors are all pre-existing `fixture 'graph' not found` errors in `test_integration.py` — they are NOT caused by this enhancement work.

### Targeted Test Results (all modified files)
```
pytest tests/test_plan_adversarial.py tests/test_builder.py tests/test_sse_lifecycle.py tests/test_hil_interrupt.py tests/test_hil_summary.py tests/test_app.py -v
141 passed
```

### Graph Node Count
```
python -c "from src.graph.builder import build_graph; g = build_graph(); print(len(g.nodes))"
19
```

---

## Files Modified
| File | Change |
|------|--------|
| `src/graph/state.py` | +4 fields |
| `src/graph/plan_adversarial.py` | +FeedbackClassification, +feedback_router, +plan_tweak_node, +route_after_hil, +route_feedback, modified plan_output_node |
| `src/graph/builder.py` | +2 nodes, replaced plan_output→END with conditional edges |
| `app.py` | Updated ALLOWED_NODES, GRAPH_NODES, generate_resume_sse, resume_endpoint |
| `src/schemas.py` | ResumeRequest: edited_plan optional, +feedback field |
| `tests/test_plan_adversarial.py` | +7 test classes, updated _base_state |
| `tests/test_builder.py` | Updated expected node set |
| `tests/test_sse_lifecycle.py` | Updated ALL_NODES |
| `tests/test_hil_interrupt.py` | +4 tests, updated existing resume tests |
| `tests/test_hil_summary.py` | NEW file, +4 tests |

## Graph Topology (Final)
```
... → consensus_check → plan_output ──→ END                    (user confirms)
                 ↑              │
           adv_rewrite ←──┐    └──→ feedback_router            (user gives feedback)
                          │           ├─ "tweak" → plan_tweak → plan_output  (loop)
                          │           └─ "rewrite" → drafter → reviewers → ...
                          └── (consensus=False, from reviewers)
```
