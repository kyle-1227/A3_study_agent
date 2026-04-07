# v0.3.0 Fix Design — Adversarial Validator Review

> **Reviewer**: Claude Opus 4.6 (Conservative Validator) | **Date**: 2026-04-04 | **Plan under review**: `agents_context/architecture/v0.3.0_fix_design.md`

---

## 1. Review Verdict

**APPROVE WITH CHANGES** — The plan's core architectural decisions (flatten SubGraph, `"text"` SSE event) are sound and correctly targeted at the root causes. However, two blockers must be resolved before handing to execution agents: (1) `interrupt()` crashes without a checkpointer, which the plan silently assumes is always present, and (2) the test breakage scope (~30 tests across 4 files) is critically under-specified in task BE-07, guaranteeing agent confusion.

---

## 2. Findings by Dimension

### D1: Factual Accuracy

The plan's code references are accurate. I verified every file path, function name, and line number cited.

| Severity | Plan Location | Code Location | Issue | Resolution |
|----------|--------------|---------------|-------|------------|
| **LOW** | BUG-08 "lines 69-104" | `planner.py:69-104` | The decorator `@traced_node` is on line 69; `async def generate_plan` starts at line 70. The function body ends at line 104. Trivially off-by-one on the start; the range is correct for deletion scope. | No action needed — the intent is clear. |

No other factual inaccuracies found. The plan correctly describes `ALLOWED_NODES` at `app.py:78`, reviewer returns at `plan_adversarial.py:152/163`, `consensus_check_node` revision_notes logic at lines 189-192, `retrieve()` calls at `academic.py:107` and `planner.py:156`, and all DAG-related code in `right-panel.tsx:218-247`.

---

### D2: Correctness of Root Cause Analysis

**BUG-01** — **CORRECT.** `sub_graph.ainvoke()` inside `plan_adversarial_node` (planner.py:234) runs the SubGraph as a standalone execution. The parent graph's `astream_events(version="v2")` cannot propagate events across a manual `ainvoke()` boundary. Flattening eliminates this boundary entirely. The fan-out/fan-in wiring (drafter → [reviewer_academic ∥ reviewer_emotional] → consensus_check) is mechanically correct for LangGraph — parallel branches write to distinct state keys (`academic_verdict`/`academic_reason` vs. `emotional_verdict`/`emotional_reason`), so no reducer collision occurs.

**BUG-02** — **CORRECT.** The `"text"` event approach is sound. After resume, `plan_output_node` returns `{"messages": [AIMessage(content=final_plan)]}`. The `on_chain_end` event fires synchronously after the node returns, with `data.output` containing this dict. No race condition — the text extraction happens in the same async iteration as the node completion event. The replacement semantics (not append) are necessary because the assistant message already contains the drafter's streamed tokens from the initial stream.

| Severity | Plan Location | Code Location | Issue | Resolution |
|----------|--------------|---------------|-------|------------|
| **LOW** | AC-02, "text" event flow | `app.py:179-189`, frontend `handleResume` | The plan does not explicitly confirm that `assistantMessageIdRef.current` persists from the initial stream into the resume flow. If it doesn't, the `"text"` event would target a stale/wrong message ID. | FE-03 is correctly listed as a verification task, but should be elevated to a **required** verification (not "likely no code change needed") — the plan should mandate the agent to assert this in a test. |

**BUG-03** — **CORRECT.** Adding `academic_reason`/`emotional_reason` fields and using them in `revision_notes` is the right fix. Fan-in behavior is safe because the two reviewer nodes write to entirely separate state keys. No reducer annotation is needed for simple `str` replacement.

**BUG-04 through BUG-10** — All root causes correctly identified.

---

### D3: Security and Input Validation

The plan does not address input validation at any system boundary. This is the weakest dimension.

| Severity | Plan Location | Code Location | Issue | Resolution |
|----------|--------------|---------------|-------|------------|
| **HIGH** | Not addressed | `src/schemas.py:9` `ChatRequest.query: str` | **No length limit on user query.** Pydantic `str` has no `max_length` by default. A user can POST 100KB+ of text that flows into `HumanMessage`, is stored in the PostgreSQL checkpointer, and is sent to the LLM API. Blast radius: cost spike, potential OOM in checkpointer serialization, downstream LLM context overflow. | Add `query: str = Field(max_length=4096)` (or appropriate limit) to `ChatRequest`. Add `edited_plan: str = Field(max_length=16384)` to `ResumeRequest`. Add to BE-05 or create a new BE task. |
| **HIGH** | Not addressed | `src/schemas.py:18` `ResumeRequest.edited_plan: str` | **User-supplied text flows directly into `AIMessage` content and is stored in state.** The edited plan is not sanitized. While it doesn't currently feed into another LLM call, any future feature that processes the plan (e.g., plan execution, plan summarization) would be a prompt injection vector. Defensive validation now prevents future exposure. | Add `max_length` constraint. Consider a warning comment: `# User-supplied — do not feed into LLM prompts without sanitization`. |
| **MEDIUM** | Not addressed | `src/database/checkpointer.py:42`, `app.py:240` | **`thread_id` has no format validation.** Any string is accepted — a user could supply another user's thread_id to `/resume` and hijack their interrupted session. The plan's REQ-10 auth (token gate) provides session-level auth, but `thread_id` itself is not scoped to a user/token. | Validate `thread_id` as UUID format (`uuid.UUID(thread_id)` in `make_thread_config`). For multi-user scenarios, bind thread_id to the access token. |
| **MEDIUM** | AC-05, REQ-10-B | Not yet implemented | **`X-Access-Token` comparison method unspecified.** Plain `==` comparison is vulnerable to timing attacks. The plan says "check header" but doesn't specify `hmac.compare_digest()` or similar constant-time comparison. | Specify `hmac.compare_digest(token, expected)` in the REQ-10-B task description. Specify middleware-level implementation (not per-endpoint) to prevent accidental bypass. |
| **LOW** | Not addressed | `app.py:71` | **CORS origin parsing doesn't strip whitespace.** `"http://a.com, http://b.com".split(",")` produces `["http://a.com", " http://b.com"]` — the second origin has a leading space and won't match. | Use `[o.strip() for o in origins.split(",") if o.strip()]`. Trivial fix, add to BE-05. |

---

### D4: Over-Engineering and Scope Creep

| Severity | Plan Location | Issue | Assessment |
|----------|--------------|-------|------------|
| **No issue** | AC-02, `"text"` SSE event | Considered whether a simulated `"token"` event could replace the new `"text"` event type. | The `"text"` event's replacement semantics (set, not append) are genuinely necessary: after drafter token streaming, the assistant message already has content. A single `"token"` event would *append* to the draft, producing duplicated text. `"text"` is a clean, justified protocol extension. |
| **No issue** | AC-01, TutorState +9 fields | State pollution from flattening. | Acceptable trade-off. All fields use `state.get(key, default)` access pattern, so they're inert for non-planning paths. The project has a single SubGraph; encapsulation overhead is not worth the event propagation risk. |
| **LOW** | AC-01 Step 3 | `gather_intel` returns 9 adversarial init fields alongside its 3 intel fields. | This couples `gather_intel` to the adversarial loop's init requirements. An alternative: let the graph's state defaults handle initialization (TypedDict fields that aren't set remain absent, and `.get()` with defaults handles access). However, explicit initialization is safer and more readable. Accept as-is. |

No scope creep detected. All changes trace back to a bug or stated requirement.

---

### D5: Regression Risk

This is the plan's most critical gap. The flattening deletes `PlanAdversarialState` and `build_adversarial_subgraph()`, which are directly referenced in **~30 tests** across **4 test files**. Task BE-07 says "Write/update tests for all changes" without enumerating the casualties.

| Severity | Plan Location | Code Location | Issue | Resolution |
|----------|--------------|---------------|-------|------------|
| **BLOCKER** | BE-07 (vague) | `tests/test_plan_adversarial.py` (entire file) | **All 25 tests break.** Every test constructs `PlanAdversarialState` dicts and/or calls `build_adversarial_subgraph()`. The imports at line 17-26 reference symbols that will be deleted. The `TestSubGraphIntegration` class (3 tests) calls `build_adversarial_subgraph()` directly. All `TestConsensusCheckNode`, `TestDrafterNode`, `TestReviewerAcademicNode`, `TestReviewerEmotionalNode`, `TestRewriteNode`, `TestOutputNode` tests pass `PlanAdversarialState` dicts. | BE-07 must explicitly list this file and specify: (1) change all `PlanAdversarialState` → `TutorState` dicts (add missing required fields), (2) delete `TestBuildSubGraph` class, (3) rewrite `TestSubGraphIntegration` to test the flattened nodes individually or via the parent graph, (4) rename `TestOutputNode` → test `plan_output_node` with interrupt mock. |
| **BLOCKER** | BE-07 (vague) | `tests/test_hil_interrupt.py:47-110` | **3 tests in `TestPlanAdversarialNodeInterrupt` break.** They import `plan_adversarial_node` from `src.graph.planner` (line 60, 81, 102) and mock `build_adversarial_subgraph`. Both symbols are deleted. | Rewrite to test `plan_output_node` from `src.graph.plan_adversarial`. Mock `interrupt()` on the new import path. Update state dicts to `TutorState`. |
| **HIGH** | BE-03 (BUG-08) | `tests/test_planner.py:10, 47-88` | **`TestGeneratePlan` class (2 tests) breaks.** Imports `generate_plan` at line 10. Deleting the function from `planner.py` makes this a hard import error — the entire test module fails. | Delete `TestGeneratePlan` class from `test_planner.py`. Update the import line to remove `generate_plan`. |
| **HIGH** | BE-03 (BUG-08) | `tests/test_llm_fallback.py:294-340` | **2 tests break.** Lines 314 and 338 import `generate_plan` from `src.graph.planner`. | Delete the `generate_plan` fallback test methods. |
| **HIGH** | BE-04 | `tests/test_builder.py:21-36` | **`test_graph_has_all_nodes` breaks.** The expected node set at lines 22-35 includes `"plan_adversarial"` and excludes the 6 flattened nodes. | Update expected set: remove `"plan_adversarial"`, add `"drafter"`, `"reviewer_academic"`, `"reviewer_emotional"`, `"consensus_check"`, `"adv_rewrite"`, `"plan_output"`. |
| **HIGH** | BE-05 | `tests/test_sse_lifecycle.py:319-329` | **`ALL_NODES` list in `TestSSEAllGraphNodes` is missing the new nodes.** Currently has 9 nodes; after flattening needs 16 (remove `plan_adversarial`, add 6 new nodes, add `rewrite_query`). Parametrized tests will miss the new nodes. | Update `ALL_NODES` list to match the new `GRAPH_NODES` set from `app.py`. |
| **MEDIUM** | BE-05 | `app.py:_stream_graph_events` | Adding `TEXT_EMIT_NODES` logic inside the `on_chain_end` handler. Risk of breaking existing node_event emission if the conditional is mis-placed. | The plan's code placement is correct (AFTER yielding the node_event end payload). Existing academic and emotional paths are unaffected — they're not in `TEXT_EMIT_NODES`. |
| **LOW** | FE-01, FE-02 | `frontend/app/page.tsx`, `right-panel.tsx` | Frontend changes don't touch `fetchWithErrorHandling` (429/401 handlers) or `consumeSSEStream` (SSE parser). No regression risk to FE-REQ-01. | None needed. |

---

### D6: Engineering Standards and Maintainability

| Severity | Plan Location | Code Location | Issue | Resolution |
|----------|--------------|---------------|-------|------------|
| **No issue** | BUG-06 | `academic.py:107`, `planner.py:156` | `asyncio.to_thread(retrieve, ...)` is the correct wrapping pattern. No deadlock risk — `to_thread` uses the default `ThreadPoolExecutor`, not the event loop. Matches existing `asyncio.to_thread(web_search_fn, ...)` pattern at `planner.py:49`, `academic.py:131`. | N/A |
| **No issue** | AC-01 | Test structure | Plan preserves pytest + per-module test files. New tests go in existing files or new files following the naming convention `test_*.py`. | N/A |
| **LOW** | AC-01 Step 2a | `plan_adversarial.py` (new), `planner.py:109`, `academic.py:43` | **`_last_human_query()` duplicated in 3 files** after the plan is implemented. | Accept for now — it's a 4-line utility. Extracting to a shared module for a single helper is premature. |
| **No issue** | BUG-09 | `app.py:_stream_graph_events` | `"done"` event is backward-compatible. The current frontend `consumeSSEStream` silently ignores unrecognized event types (no handler → falls through `processSSEEvent` with no effect). Adding the `"done"` event doesn't break anything. Frontend can add a handler later. | N/A |
| **No issue** | Agent separation | BE tasks / FE tasks | Backend and frontend tasks are correctly scoped. The SSE `"text"` event contract is the only cross-boundary dependency, and it's fully specified in AC-02. No simultaneous coordination required. | N/A |
| **No issue** | BUG-10 | `src/utils/` | Confirmed deleted in commit `9038727`. Directory does not exist. | N/A |

---

### D7: Missing Issues

Issues found in the codebase that the plan does NOT address:

| Severity | Code Location | Issue | Recommended Resolution |
|----------|---------------|-------|------------------------|
| **BLOCKER** | `plan_adversarial.py` (proposed `plan_output_node`), `app.py:37-55` | **`interrupt()` requires a checkpointer. Running without one (no `DB_URI`) raises `ValueError`.** The lifespan at `app.py:40-55` explicitly handles the no-checkpointer case (`checkpointer = None`). When no DB is configured, `get_compiled_graph(checkpointer=None)` produces a graph without persistence. Any planning query that reaches `plan_output_node` will crash with `ValueError: interrupt() requires a checkpointer`. This is a runtime crash, not a type error. | **Option A (recommended)**: Guard the interrupt call: `if graph_has_checkpointer: edited = interrupt(plan_text); else: edited = plan_text` (skip HIL when stateless). **Option B**: Make checkpointer mandatory for the planning path and return an error SSE event when a planning query is received without a checkpointer. Either way, this MUST be addressed — it's a crash path in the default dev configuration. |
| **MEDIUM** | `app.py:96-189` | **No try/except around `_stream_graph_events` generator.** If any node raises an unhandled exception, the SSE stream breaks silently — the frontend sees ReadableStream end without a `"done"` or `"error"` event. The SSE contract defines `{"type":"error","message":"..."}` but this event is never emitted anywhere in `app.py`. | Wrap the `async for event in graph.astream_events(...)` loop in a try/except. On exception, yield `{"type": "error", "message": str(e)}` before the generator exits. |
| **MEDIUM** | `app.py:240`, `src/database/checkpointer.py:42` | **No protection against concurrent requests on the same `thread_id`.** Two simultaneous `/resume` calls with the same `thread_id` could corrupt checkpointer state. LangGraph's PostgreSQL checkpointer may or may not be safe under concurrent writes. | Document this as a known limitation. The semaphore from REQ-10-A (`MAX_CONCURRENT_STREAMS=3`) reduces but doesn't eliminate the risk. Consider per-thread_id locking if concurrent access is expected. |
| **MEDIUM** | `frontend/app/page.tsx:274-317` | **`handleResume` doesn't create a new assistant message placeholder.** It relies on `assistantMessageIdRef.current` persisting from the initial stream. If the user sends a new message (which resets `assistantMessageIdRef`) before resuming, the resume stream's events would target the wrong message. The interleaving is unlikely (PlanReview UI blocks input) but not impossible via direct API calls. | FE-03 verification should confirm this ref integrity. Add a defensive check: if `assistantMessageIdRef.current` is empty when `handleResume` fires, create a new placeholder. |
| **LOW** | `frontend/components/plan-review.tsx:14` | **`PlanReview` uses `useState(draft)` for initial state.** If the parent re-renders with a different `draft` prop (e.g., on a second interrupt in the same session), the textarea won't update because `useState` initial value is only used on mount. | Use a `useEffect` keyed on `draft` to reset `editedPlan` when the draft prop changes. Low priority — second interrupt in the same session is unlikely. |
| **LOW** | `src/graph/supervisor.py:50-51` | **`user_text` extraction assumes `last_msg.content` is a string.** If `last_msg` is a multi-modal message or has a list content (LangChain supports list-of-blocks content), `.content` could be a list. | Defensive: `user_text = str(last_msg.content) if hasattr(last_msg, "content") else str(last_msg)`. Low risk in current system (all messages are text). |

---

## 3. Blocking Issues

These MUST be resolved before the plan is handed to execution agents:

### BLOCKER-1: `interrupt()` crashes without checkpointer

**Location**: Proposed `plan_output_node` in `plan_adversarial.py`, runtime interaction with `app.py:57`

**Problem**: The default development configuration runs without `DB_URI` (no PostgreSQL). `get_compiled_graph(checkpointer=None)` compiles the graph without persistence. When a user sends a planning query, execution reaches `plan_output_node` which calls `interrupt(plan_text)`. LangGraph raises `ValueError` because `interrupt()` requires a checkpointer to persist the suspended state.

**Fix**: Add to BE-02 or BE-05: in `plan_output_node`, check if the graph has a checkpointer before calling `interrupt()`. If no checkpointer, skip HIL and return the plan directly. Alternatively, add a runtime check in `app.py` that rejects planning queries when running stateless.

### BLOCKER-2: Test breakage scope unspecified in BE-07

**Location**: Task BE-07 in the Agent Task Breakdown

**Problem**: The task says "Write/update tests for all changes" but does not enumerate the ~30 tests that will break. Execution agents operating in isolation need explicit guidance. Without it, the backend agent will implement BE-01 through BE-06, run the test suite, see 30 failures, and either waste time investigating or make incorrect assumptions about which tests to delete vs. rewrite.

**Fix**: Replace BE-07's description with an explicit test migration plan:

| Test File | Action | Details |
|-----------|--------|---------|
| `test_plan_adversarial.py` | **Rewrite entirely** | All `PlanAdversarialState` → `TutorState` dicts. Delete `TestBuildSubGraph`. Rewrite `TestSubGraphIntegration` as individual node tests. Rename `TestOutputNode` → test `plan_output_node` with `interrupt()` mock. |
| `test_hil_interrupt.py` | **Rewrite `TestPlanAdversarialNodeInterrupt`** (3 tests) | Import `plan_output_node` from `src.graph.plan_adversarial`. Mock `interrupt` at new path. Keep `TestHILIntegration` and `TestSSE*` classes unchanged. |
| `test_planner.py` | **Delete `TestGeneratePlan`** class (2 tests), update import line | Remove `generate_plan` from import. `TestSearchPolicy` unchanged. |
| `test_llm_fallback.py` | **Delete 2 `generate_plan` fallback tests** | Lines ~294-340. All other fallback tests unchanged. |
| `test_builder.py` | **Update `test_graph_has_all_nodes`** | Remove `"plan_adversarial"`, add 6 new nodes to expected set. |
| `test_sse_lifecycle.py` | **Update `ALL_NODES`** list | Add 6 new nodes, remove `plan_adversarial`, add `rewrite_query`. Should match `GRAPH_NODES` from `app.py`. |
| `test_gather_intel.py` | **Update assertions** | Verify `gather_intel` returns 9 adversarial init fields in addition to existing 3 intel fields. |

---

## 4. Approved Elements

These parts of the plan are correct, well-designed, and should be kept as-is:

1. **AC-01: Flatten SubGraph decision** — Sound rationale. Guaranteed event propagation. Acceptable state trade-off. Implementation steps are precise and correctly ordered.

2. **AC-02: `"text"` SSE event** — Clean protocol extension. Replacement semantics are correct. Solves both BUG-02 and BUG-04 with one mechanism. Backend extraction logic (`on_chain_end` → `output.messages` → `msg.content`) is mechanically correct.

3. **AC-03: Reviewer reason preservation** — Minimal state extension. Correct `consensus_check_node` update. `adv_rewrite_node` correctly clears reason fields.

4. **AC-04: `handle_unknown` via `TEXT_EMIT_NODES`** — Zero additional code beyond AC-02. Elegant reuse.

5. **AC-05: Phase 3 decomposition** — Atomic tasks with clear acceptance criteria. Correct ordering constraints.

6. **BUG-06 fix** — `asyncio.to_thread(retrieve, ...)` is the correct pattern. Both call sites identified.

7. **BUG-07 fix** — Trivial but necessary. Correctly sequenced before AC-01.

8. **BUG-09 fix** — Correct placement (after interrupt check, before generator exit). Backward-compatible.

9. **BUG-10 resolution** — Correctly identified as already resolved by commit `9038727`.

10. **Task ordering constraints** — Critical path `BE-01 → BE-02 → BE-04 → BE-05 → BE-07` is correct. Parallel tracks are correctly identified.

11. **`app.py` constant updates** — `ALLOWED_NODES`, `TEXT_EMIT_NODES`, and `GRAPH_NODES` are all correct. The addition of `"rewrite_query"` to `GRAPH_NODES` (missing from both the bug list and the old code) is a good catch.

---

## 5. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `interrupt()` ValueError in stateless mode | **High** (default dev config has no DB) | **Critical** (crash on any planning query) | Guard `interrupt()` with checkpointer presence check (BLOCKER-1) |
| R2 | Agent rewrites wrong tests or deletes too many | **High** (BE-07 is vague) | **High** (broken test suite, false confidence) | Specify exact test file migration plan (BLOCKER-2) |
| R3 | Large user input causes cost spike / OOM | **Medium** (public deployment planned) | **High** (unbounded LLM cost) | Add `max_length` to Pydantic schemas (D3 finding) |
| R4 | Concurrent same-thread_id corrupts state | **Low** (requires deliberate API abuse or race) | **High** (session corruption) | Document as limitation; per-thread locking in future |
| R5 | `X-Access-Token` timing attack | **Low** (requires local network access) | **Medium** (token extraction) | Use `hmac.compare_digest()` in REQ-10-B |
| R6 | Unhandled exception kills SSE stream silently | **Medium** (any node can throw) | **Medium** (user sees frozen UI) | Wrap generator in try/except, emit `"error"` SSE event |
| R7 | `assistantMessageIdRef` mismatch on resume | **Low** (UI blocks input during interrupt) | **Medium** (plan text targets wrong message) | FE-03 verification + defensive ref check |
| R8 | CORS misconfigured due to whitespace in env var | **Low** | **Low** (CORS rejection) | Strip whitespace in CORS origin parsing |
