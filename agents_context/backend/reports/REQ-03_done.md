# REQ-03: Supervisor structured output + few-shot fix — Done

## Summary

Replaced manual `json.loads()` parsing in `supervisor.py` with Pydantic `with_structured_output()`.
Added "unknown" intent for off-topic queries. Added few-shot examples for historical exam queries.

## Files Changed

| File | Change |
|------|--------|
| `src/graph/supervisor.py` | Added `SupervisorOutput` Pydantic model; replaced `json.loads()` with `llm.with_structured_output(SupervisorOutput).ainvoke()`; added `handle_unknown` async node; removed `import json` |
| `src/graph/builder.py` | Imported `handle_unknown`; added `handle_unknown` node; added `"unknown": "handle_unknown"` route; added `handle_unknown → END` edge |
| `config/prompts/supervisor_system.xml` | Added "unknown" intent description; added few-shot examples for historical exam queries (2024高考作文题, 历届真题, etc.); added `confidence` field in output format; added off-topic examples |
| `config/settings.yaml` | Added `unknown` to `supervisor.valid_intents` list |
| `tests/test_supervisor.py` | Rewrote all tests to mock `with_structured_output` chain; added `TestHandleUnknown` (2 tests); added `TestSupervisorOutput` (3 tests); added tests for unknown intent, historical exam routing, keywords→keypoints mapping |
| `tests/test_builder.py` | Added `handle_unknown` and `rewrite_query` to expected node set |
| `tests/test_config.py` | Updated `test_supervisor_valid_intents` to include "unknown" |

## Verification

- `supervisor.py` contains **zero** `json.loads()` calls
- Test for "2024高考作文题" routes to `academic` intent
- Test for off-topic queries routes to `unknown` intent
- **304 passed, 1 skipped, 0 failed**

## How to Run Tests

```bash
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short
```

## Suggested Commit

```
git add src/graph/supervisor.py src/graph/builder.py config/prompts/supervisor_system.xml config/settings.yaml tests/test_supervisor.py tests/test_builder.py tests/test_config.py
git commit -m "feat(supervisor): structured output + unknown intent + few-shot examples (REQ-03)"
```

## Design Notes

- `SupervisorOutput.keywords` maps to `state["keypoints"]` in the node return dict — the Pydantic field name follows the task spec while the state field name stays backward-compatible
- Subject detection uses keyword matching on the user text (math/chinese keywords) since structured output focuses on intent + keywords + confidence
- `handle_unknown` returns a static friendly redirect message (no LLM call needed for off-topic)
- Fallback on any structured output exception defaults to `academic/other/[]` (safe default)
