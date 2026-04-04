# REQ-01: Full async node refactoring — DONE

## Summary
All graph node functions converted from synchronous `def` to `async def`. All `llm.invoke()` replaced with `await llm.ainvoke()`. `ThreadPoolExecutor` in web search nodes replaced with `asyncio.to_thread()`. `@traced_node` decorator updated to support both sync and async functions.

## Files Changed

### Source
| File | Change |
|------|--------|
| `src/graph/supervisor.py` | `supervisor_node` → `async def`, `llm.invoke()` → `await llm.ainvoke()` |
| `src/graph/academic.py` | All 5 nodes → `async def`; `invoke_with_fallback` → `await async_invoke_with_fallback`; `ThreadPoolExecutor` → `asyncio.to_thread()` + `asyncio.wait_for()` in `web_search` |
| `src/graph/planner.py` | Both nodes → `async def`; `invoke_with_fallback` → `await async_invoke_with_fallback`; `ThreadPoolExecutor` → `asyncio.to_thread()` + `asyncio.wait_for()` in `search_policy` |
| `src/graph/emotional.py` | `emotional_response` → `async def`; `invoke_with_fallback` → `await async_invoke_with_fallback` |
| `src/graph/llm.py` | Added `async_invoke_with_fallback()` — async equivalent using `ainvoke()` throughout. Sync `invoke_with_fallback()` preserved for backward compatibility. |
| `src/tracing/decorators.py` | `@traced_node` now detects `asyncio.iscoroutinefunction()` and wraps with async/sync appropriately. Shared `_record_result()` helper extracted to eliminate duplication. |
| `pyproject.toml` | **NEW** — `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` |
| `requirements.txt` | Added `pytest-asyncio>=0.23.0` |

### Tests
| File | Change |
|------|--------|
| `tests/test_supervisor.py` | All node tests → `async def`; mocks use `ainvoke = AsyncMock(...)` |
| `tests/test_academic.py` | All node tests → `async def`; mocks use `ainvoke = AsyncMock(...)` |
| `tests/test_planner.py` | All node tests → `async def`; mocks use `ainvoke = AsyncMock(...)` |
| `tests/test_emotional.py` | All node tests → `async def`; mocks use `ainvoke = AsyncMock(...)` |
| `tests/test_hallucination.py` | All node tests → `async def`; mocks use `ainvoke = AsyncMock(...)` |
| `tests/test_llm_fallback.py` | Node-level fallback tests → `async def`; added `TestAsyncInvokeWithFallback` class for new async helper. Sync `TestInvokeWithFallback` kept. |
| `tests/test_parallel_retrieval.py` | Node-calling tests → `async def` |
| `tests/test_tracing.py` | Added `TestTracedNodeDecoratorAsync` class (6 tests for async decorator path) |
| `tests/test_config.py` | 2 node integration tests → `async def` |

## Verification

```bash
# Run tests (280 passed, 0 failed, 1 skipped)
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# Verify no sync invoke() remains in src/
grep -r "llm\.invoke()" src/   # returns zero results

# Verify all node functions are async def
grep -rn "^async def\|^def" src/graph/supervisor.py src/graph/academic.py src/graph/planner.py src/graph/emotional.py
```

## Suggested commit

```
git add pyproject.toml requirements.txt src/graph/supervisor.py src/graph/academic.py src/graph/planner.py src/graph/emotional.py src/graph/llm.py src/tracing/decorators.py tests/test_supervisor.py tests/test_academic.py tests/test_planner.py tests/test_emotional.py tests/test_hallucination.py tests/test_llm_fallback.py tests/test_parallel_retrieval.py tests/test_tracing.py tests/test_config.py

git commit -m "feat(async): convert all graph nodes to async def + ainvoke (REQ-01)

- All node functions: def → async def, llm.invoke() → await llm.ainvoke()
- Added async_invoke_with_fallback() to llm.py
- ThreadPoolExecutor → asyncio.to_thread() in web_search/search_policy
- @traced_node decorator: supports both sync and async nodes
- Added pytest-asyncio + pyproject.toml with asyncio_mode=auto
- All 280 tests pass"
```
