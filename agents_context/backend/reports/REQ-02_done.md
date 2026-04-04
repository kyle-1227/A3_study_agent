# REQ-02: Retry loop fix — context clearing + query rewrite — DONE

## Summary
Fixed the hallucination retry loop: (1) context now clears on retry instead of accumulating, (2) a new `rewrite_query` node rewrites the user's question using hallucination feedback before re-retrieval.

## Files Changed

### Source
| File | Change |
|------|--------|
| `src/graph/state.py` | Added `rewritten_query: str` and `hallucination_reason: str` fields. Replaced `operator.add` with custom `context_reducer` that supports `CONTEXT_CLEAR` sentinel for resetting context on retry. |
| `src/graph/academic.py` | `academic_router`: returns `CONTEXT_CLEAR` when `retry_count > 0`. New `rewrite_query` async node: calls supervisor LLM to rewrite query using `hallucination_reason`. `rag_retrieve` + `web_search`: use `rewritten_query` if non-empty. `evaluate_hallucination`: stores `hallucination_reason` on detection. |
| `src/graph/builder.py` | Added `rewrite_query` node. Retry path: `evaluate_hallucination → rewrite_query → academic_router` (was `evaluate_hallucination → academic_router`). |
| `config/prompts/rewrite_query.xml` | **NEW** — Prompt template for query rewriting with `{original_query}` and `{hallucination_reason}` variables. |

### Tests
| File | Change |
|------|--------|
| `tests/test_academic.py` | Added `TestAcademicRouterRetry` (3 tests: first run, retry clear, second retry). Added `TestRewriteQuery` (3 tests: produces rewritten query, uses reason in prompt, fallback on failure). Added `TestRagRetrieveWithRewrittenQuery` (2 tests). Added `TestWebSearchWithRewrittenQuery` (2 tests). |
| `tests/test_hallucination.py` | Added 2 tests in `TestEvaluateHallucinationNode`: `test_stores_hallucination_reason`, `test_no_reason_when_faithful`. |
| `tests/test_parallel_retrieval.py` | Replaced `operator.add` tests with `context_reducer` tests including clear signal test. Added `TestRewriteQueryNodeInGraph` (2 tests for graph topology). |
| `tests/conftest.py` | `sample_state` fixture: added `rewritten_query` and `hallucination_reason` fields. |

## Verification

```bash
# Run tests (295 passed, 0 failed, 1 skipped)
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# Verify context_reducer replaces operator.add
grep -r "operator.add" src/graph/state.py  # returns zero results

# Verify rewrite_query node exists in graph
grep -rn "rewrite_query" src/graph/builder.py
```

## Retry Flow (before → after)

**Before:** `evaluate_hallucination → academic_router → [rag_retrieve, web_search] → generate_answer`
- Context accumulated across retries (stale + new mixed)
- Same query used on every attempt

**After:** `evaluate_hallucination → rewrite_query → academic_router → [rag_retrieve, web_search] → generate_answer`
- `academic_router` clears context via `CONTEXT_CLEAR` sentinel
- `rewrite_query` rewrites user query using `hallucination_reason`
- `rag_retrieve` / `web_search` use `rewritten_query` for improved retrieval

## Suggested commit

```
git add src/graph/state.py src/graph/academic.py src/graph/builder.py config/prompts/rewrite_query.xml tests/test_academic.py tests/test_hallucination.py tests/test_parallel_retrieval.py tests/conftest.py

git commit -m "feat(retry): fix context clearing + add query rewrite on retry (REQ-02)

- Custom context_reducer with CONTEXT_CLEAR sentinel replaces operator.add
- academic_router clears context when retry_count > 0
- New rewrite_query node rewrites query using hallucination_reason
- rag_retrieve/web_search use rewritten_query when available
- evaluate_hallucination stores hallucination_reason in state
- Retry path: evaluate_hallucination → rewrite_query → academic_router
- All 295 tests pass"
```
