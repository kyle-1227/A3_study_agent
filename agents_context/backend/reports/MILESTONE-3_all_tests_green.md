# MILESTONE 3 — All Tests Green

> **Status**: DONE | **Date**: 2026-04-05 | **Task**: BE-07

---

## Test Results

```
pytest tests/ -v (excluding test_integration.py)
================= 404 passed, 1 skipped, 2 warnings in 40.99s =================

pytest tests/ -v (including test_integration.py)
============ 408 passed, 1 skipped, 6 warnings, 9 errors in 39.94s ============
```

The 9 errors in `test_integration.py` are **pre-existing** — that file defines standalone integration tests with a `graph` parameter that is not a pytest fixture. It is designed to run via `python -m tests.test_integration`, not via pytest. These errors existed before the v0.3.0 refactor (commit `375d0e7`, untouched since).

---

## Changes Made

### 1. `tests/test_plan_adversarial.py` — REWRITTEN

| Class | Action |
|-------|--------|
| `TestReviewVerdict` (3 tests) | **Kept unchanged** |
| `TestDrafterNode` (2 tests) | **Updated**: `PlanAdversarialState` → `TutorState`, `"round"` → `"adv_round"` |
| `TestReviewerAcademicNode` (3 tests) | **Updated**: `TutorState` dicts, added `academic_reason` assertions |
| `TestReviewerEmotionalNode` (1 test) | **Updated**: `TutorState` dict, added `emotional_reason` assertion |
| `TestConsensusCheckNode` (5→6 tests) | **Updated**: `TutorState`, `"adv_round"`, removed `"max_rounds"`. **Added** `test_revision_notes_contains_reason_text` |
| `TestRewriteNode` → `TestAdvRewriteNode` (1 test) | **Renamed**, added `academic_reason`/`emotional_reason` clear assertions |
| `TestOutputNode` → `TestPlanOutputNode` (1→2 tests) | **Renamed**, tests `plan_output_node` with `interrupt()` mocked. **Added** `test_interrupt_guard_skips_hil` |
| `TestShouldOutputOrRevise` (2 tests) | **New**: tests routing function |
| `TestBuildSubGraph` (2 tests) | **DELETED** |
| `TestSubGraphIntegration` (3 tests) | **DELETED** |
| **Imports** | Removed `PlanAdversarialState`, `build_adversarial_subgraph`, `output_node`, `rewrite_node`. Added `TutorState`, `adv_rewrite_node`, `plan_output_node`, `should_output_or_revise`. |

### 2. `tests/test_hil_interrupt.py` — PARTIAL REWRITE

| Class | Action |
|-------|--------|
| `TestPlanAdversarialNodeInterrupt` → `TestPlanOutputNodeInterrupt` (3 tests) | **Rewritten**: imports `plan_output_node` from `src.graph.plan_adversarial`, mocks `src.graph.plan_adversarial.interrupt`, uses full `TutorState` dicts |
| `TestResumeSSE` | **Updated**: `"plan_adversarial"` → `"plan_output"` in mock event |
| All other classes | **Kept unchanged** |

### 3. `tests/test_planner.py` — PARTIAL UPDATE

| Class | Action |
|-------|--------|
| `TestGeneratePlan` (2 tests) | **DELETED** |
| Import line | Removed `generate_plan` |

### 4. `tests/test_llm_fallback.py` — PARTIAL UPDATE

| Class | Action |
|-------|--------|
| `TestGeneratePlanFallback` (2 tests) | **DELETED** |

### 5. `tests/test_builder.py` — UPDATED

| Test | Action |
|------|--------|
| `test_graph_has_all_nodes` | **Updated** expected set: removed `"plan_adversarial"`, added 6 flattened nodes |

### 6. `tests/test_sse_lifecycle.py` — UPDATED + NEW TESTS

| Location | Action |
|----------|--------|
| `ALL_NODES` list | **Updated**: 9 → 17 nodes (matches `GRAPH_NODES` from `app.py`) |
| `_parse_payloads` helper | **Updated**: also filters trailing `"done"` events |
| `TestSSETextEvent` (3 tests) | **NEW**: tests `"text"` SSE event for `plan_output`, `handle_unknown`, and non-text-emit nodes |
| `TestSSEDoneEvent` (2 tests) | **NEW**: tests `"done"` event on normal completion and absence on interrupt |

### 7. `tests/test_gather_intel.py` — UPDATED

| Tests | Action |
|-------|--------|
| All 5 tests | **Added assertions** for adversarial init fields (`adv_round`, `draft`, `consensus`, etc.) |

### 8. `tests/test_app.py` — NEW TESTS

| Class | Tests |
|-------|-------|
| `TestInputValidation` (4 tests) | `ChatRequest` rejects query > 4096 chars, accepts normal. `ResumeRequest` rejects plan > 16384 chars, accepts normal. |

---

## Test Count Summary

| Metric | Count |
|--------|-------|
| Tests deleted | 9 (2 SubGraph build + 3 SubGraph integration + 2 GeneratePlan + 2 GeneratePlanFallback) |
| Tests added | 13 (2 PlanOutputNode + 2 ShouldOutputOrRevise + 1 ConsensusCheck reason + 3 TextEvent + 2 DoneEvent + 4 InputValidation - 1 already counted interrupt guard) |
| Net change | +4 |
| Total passing | 404 (excluding `test_integration.py`) / 408 (including) |

---

## New SSE Event Coverage

| Event | Test Location |
|-------|--------------|
| `"text"` | `test_sse_lifecycle.py::TestSSETextEvent` (3 tests) |
| `"done"` | `test_sse_lifecycle.py::TestSSEDoneEvent` (2 tests) |
| `"error"` | Not tested separately (SEC-03 error path — would require raising inside `astream_events`, can be added in a follow-up) |
| Input validation | `test_app.py::TestInputValidation` (4 tests) |
| `interrupt()` guard | `test_plan_adversarial.py::TestPlanOutputNode::test_interrupt_guard_skips_hil` |

---

## Next step

**Frontend tasks (FE-01, FE-02, FE-03)** — awaiting user instruction to proceed.
