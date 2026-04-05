# MILESTONE 1 — Graph Compiles

**Date**: 2026-04-05  
**Branch**: `refactor/architecture_design`  
**Status**: PASS

---

## Tasks Completed

### BE-01: State + Intent Fix
- Added `"unknown"` to `intent` Literal in `TutorState` (BUG-07)
- Added 8 adversarial planning fields to `TutorState`: `draft`, `academic_verdict`, `academic_reason`, `emotional_verdict`, `emotional_reason`, `adv_round`, `consensus`, `revision_notes`
- TutorState now has **22 fields**
- Added `max_length=4096` to `ChatRequest.query` and `max_length=16384` to `ResumeRequest.edited_plan` (SEC-01)

### BE-02: Refactor plan_adversarial.py
- All 6 node functions now accept `TutorState` instead of `PlanAdversarialState`
- `drafter_node`: uses `_last_human_query()` helper, returns `adv_round` key
- `reviewer_academic_node` / `reviewer_emotional_node`: now return `verdict.reason` in `academic_reason`/`emotional_reason`
- `consensus_check_node`: reads `max_rounds` from config via `get_setting()`, uses reason fields in `revision_notes` (AC-03)
- Renamed `rewrite_node` → `adv_rewrite_node`, clears all 4 verdict/reason fields
- Renamed `output_node` → `plan_output_node`, adds `interrupt()` with `try/except ValueError` guard
- Renamed `_should_output_or_revise` → `should_output_or_revise` (public)
- Deleted `PlanAdversarialState` class and `build_adversarial_subgraph()` function

### BE-03: Clean up planner.py + async retrieval
- Updated `gather_intel` return dict with 8 adversarial init fields
- Deleted `plan_adversarial_node` function
- Deleted `generate_plan` function (BUG-08)
- Removed stale imports (`interrupt`, `AIMessage`)
- Wrapped `retrieve()` in `asyncio.to_thread()` in `_gather_resource_intel._rag` (BUG-06)
- Wrapped `retrieve()` in `asyncio.to_thread()` in `academic.py:rag_retrieve` (BUG-06)

### BE-04: Rewire builder.py
- Replaced `plan_adversarial_node` import with 7 flattened node imports from `plan_adversarial`
- Replaced single `plan_adversarial` node with 6 nodes: `drafter`, `reviewer_academic`, `reviewer_emotional`, `consensus_check`, `adv_rewrite`, `plan_output`
- Rewired edges: `gather_intel` → `drafter` → parallel reviewers → `consensus_check` → conditional output/revise loop

## Verification

### Graph node check
```
python -c "from src.graph.builder import build_graph; g = build_graph(); print(sorted(g.nodes.keys()))"
```
**Result**: 17 nodes confirmed:
```
['academic_router', 'adv_rewrite', 'consensus_check', 'drafter', 'emotional_response',
 'evaluate_hallucination', 'gather_intel', 'generate_answer', 'handle_unknown',
 'plan_output', 'rag_retrieve', 'reviewer_academic', 'reviewer_emotional',
 'rewrite_query', 'search_policy', 'supervisor', 'web_search']
```

### Test results
```
pytest tests/test_builder.py tests/test_state.py -v
```
- `test_graph_has_all_nodes` — **FAILED (expected)** — test still asserts old `plan_adversarial` node; will be fixed in BE-07
- All other 6 tests — **PASSED**
