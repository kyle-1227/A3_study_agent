# Context Engineering Phase 0 Audit

Date: 2026-06-29

This report captures the repository state before Phase 0 cleanup. It is intentionally evidence-bearing: raw findings are preserved so later phases can see what was present before remediation.

## Scan Scope

Command:

```powershell
rg -n --hidden --glob '!.git/**' --glob '!__pycache__/**' --glob '!*.pdf' "fallback|dummy|default|_default|fail_fast|get_setting\(|token_budget|embedding_provider|classify_error|default_classification|except Exception|return \[\]|return None|logger\.warning|logger\.exception"
```

Scanned keywords:

- `fallback`
- `dummy`
- `default`
- `_default`
- `fail_fast`
- `get_setting(`
- `token_budget`
- `embedding_provider`
- `classify_error`
- `default_classification`
- `except Exception`
- `return []`
- `return None`
- `logger.warning`
- `logger.exception`

## Raw Findings And Classification

| File / raw finding | Classification | Current behavior before cleanup | Context Engineering risk | Phase 0 status | Reason |
|---|---|---|---|---|---|
| `src/assessment/classifier.py:103` `logger.warning("Error classification LLM returned no valid output, using default")` | fake success fallback | Invalid structured classification returns a default business classification. | A failed diagnosis looks like a successful concept error. | fixed | Target path. |
| `src/assessment/classifier.py:105` `return _default_classification(quiz_result)` | fake success fallback | Produces `concept` classification on invalid LLM output. | Hides LLM/schema failure from caller. | fixed | Target path. |
| `src/assessment/classifier.py:118` `logger.exception("Error classification LLM call failed: %s", exc)` + `return _default_classification(...)` | swallowed runtime failure | Provider/runtime failures become default classification. | Masks classification pipeline failure. | fixed | Target path. |
| `src/assessment/classifier.py:123` `_default_classification(...)` | production default object | Helper fabricates conservative concept classification. | Makes false positives indistinguishable from real LLM result. | fixed | Target path. |
| `src/assessment/types.py:37-50` `ErrorClassificationStrict` fields have Pydantic defaults | structured output drift | Missing LLM fields can pass with defaults. | Invalid LLM output can look valid. | fixed | Target path. |
| `src/memory/embeddings.py:8` `DummyEmbeddingProvider: zero-vector fallback` | dummy embedding fallback | Dummy provider is documented as production fallback. | Pollutes retrieval scoring and hides provider failure. | fixed | Target path. |
| `src/memory/embeddings.py:21-26` `DEFAULT_EMBEDDING_MODEL`, `DEFAULT_EMBEDDING_BASE_URL`, `DEFAULT_DUMMY_DIM` | hardcoded/silent defaults | Embedding config defaults to OpenRouter-style values and dummy dimension. | Production embedding behavior not explicit in config. | fixed | Target path. |
| `src/memory/embeddings.py:87-89` missing API key returns zero vectors | runtime failure to fake vectors | Missing key becomes vector of zeros. | Retrieval degrades silently to keyword-only. | fixed | Target path. |
| `src/memory/embeddings.py:122-124` embedding API exception returns zero vectors | runtime failure to fake vectors | API failure becomes zero-vector result. | Makes provider outage look like successful embedding. | fixed | Target path. |
| `src/memory/embeddings.py:127-143` `embedding_dim()` falls back to default | silent default | Unknown dimension returns dummy dimension. | Hides missing provider state. | fixed | Target path. |
| `src/memory/embeddings.py:147-161` `DummyEmbeddingProvider` | production dummy provider | Zero-vector provider available from production module. | Enables production dummy config/path. | fixed | Target path. |
| `src/memory/embeddings.py:170-184` factory falls back to dummy | provider fallback | Provider construction failure creates dummy provider. | Provider misconfig becomes fake success. | fixed | Target path. |
| `config/settings.yaml:548` `embedding_provider: "deepseek" # "deepseek" | "dummy"` | production dummy config surface | Dummy is advertised as valid production provider. | fixed | Target path. |
| `src/memory/retrieval.py:176-185` DB fetch exceptions logged then continue | runtime failure to empty/partial result | Episodic/semantic DB failures produce partial or empty retrieval. | DB failure is indistinguishable from no memories. | fixed | Target path. |
| `src/memory/retrieval.py:193-198` embedding failure logged then retrieval continues | runtime failure to keyword-only result | Query embedding failure produces keyword-only scoring. | Embedding failure is hidden as low-quality success. | fixed | Target path. |
| `src/memory/retrieval.py:246` `reasons.append("fallback")` | misleading fallback label | Low-signal match reason is labeled fallback. | Confuses retrieval semantics. | fixed | Target path. |
| `src/graph/academic.py:6861-6882` `episodic_memory_retriever` catches retrieval exceptions and returns empty results | runtime failure to empty result | Retrieval failure becomes `episodic_memory_results=[]`, `semantic_memory_results=[]`. | Runtime failure is indistinguishable from valid empty memory. | fixed | Strongly related memory retrieval path. |
| `src/context/token_manager.py:29-31` `deepseek-v4-pro with 128k context` comment | stale model-window statement | Legacy memory budget comment claims 128k context. | Misleads Context Engineering sizing. | fixed | Target path. |
| `src/context/token_manager.py:47-55` `get_setting(..., default)` | silent config fallback | Missing token budget values use code defaults. | Missing production config appears valid. | fixed | Target path. |
| `config/settings.yaml:91-96` `context_budget.model_limits.deepseek-v4-pro: 128000` | hardcoded wrong model window | Context usage telemetry treats `deepseek-v4-pro` as 128k. | Wrong context window can drive bad budget decisions. | fixed | Deleted wrong value; no 1M hardcode added. |
| `config/settings.yaml:569` `memory.token_budget` lacks legacy warning | misleading config semantics | Memory-only local budget can be mistaken for model context. | Pollutes future Context Engineering model-limit design. | fixed | Target path. |
| `src/memory/episodic.py:90-95` embedding exceptions are swallowed | runtime failure to partial success | Episodic memory can be persisted without embedding. | Embedding failure is hidden. | fixed | Target embedding path. |
| `src/memory/episodic.py:104-109` persist failure can be non-fatal by config | memory write fallback | Failed memory write may return a record. | Fake persistence success. | deferred | Broader memory-write policy; reported for Phase 1 or a separate memory-write spec. |
| `src/memory/semantic.py:96-107` consolidation LLM failure returns `None` | structured LLM failure to empty result | Consolidation failure looks like no summary created. | Hides memory summarization failure. | deferred | Not in this Phase 0 target surface; requires consolidation behavior spec. |
| `src/memory/semantic.py:120-126` summary embedding failure is swallowed | runtime failure to partial success | Semantic summary can be saved without embedding. | Retrieval quality degrades silently. | deferred | Not in this Phase 0 target surface; requires consolidation behavior spec. |
| `src/memory/semantic.py:150-154` persistence failure returns `None` | memory write failure to empty result | Save failure looks like no summary. | Hides DB failure. | deferred | Not in this Phase 0 target surface; requires consolidation behavior spec. |
| `src/memory/consolidation.py:107-109`, `153-155` forgetting exceptions logged and swallowed | maintenance failure swallowed | Background maintenance can partially fail. | Less critical than live context path, but should remain visible. | deferred | Background maintenance is non-request-critical; audit only in Phase 0. |
| `app.py:1057-1104` profile/input memory recording exceptions are logged and non-fatal | request-side observability/memory failure swallowed | Startup request memory/profile failures do not block `/stream`. | Could hide memory write failures. | deferred | Requires app-level runtime policy decision; not changed to avoid broad graph/API behavior churn. |
| `src/graph/llm.py:63`, `119`, `137` provider/model/env defaults and fallback helper residue | legacy provider fallback/hardcode | LLM bridge still has default model/provider/fallback env behavior. | Protected legacy risk for future cleanup. | deferred | Existing governance report already flags this protected area; not in Phase 0 implementation surface. |
| `src/llm/structured_output.py` `_output_setting(..., default)`, fallback mode machinery | legacy structured-output fallback surface | Runtime has configurable fallback/retry concepts, though current config disables fallback. | High-risk protected contract. | deferred | Existing governance report flags it; do not alter protected runtime beyond target path. |
| `src/observability/context_usage.py` emits `context_usage_error` for missing/unknown model windows | explicit warning, not fallback | Missing model window is visible as warning event. | Acceptable Phase 0 behavior. | verified | No code change needed after deleting wrong 128k config. |
| `tests/test_context_builder.py:62` `TokenBudget()` default test | test depends on default budget | Test validates direct default construction. | Conflicts with strict config intent. | fixed | Updated tests to explicit construction / strict `from_settings`. |
| `tests/test_context_builder.py:106-111` imports `DummyEmbeddingProvider` | test depends on production dummy provider | Test validates production dummy provider export. | Conflicts with no production dummy provider. | fixed | Updated tests to production exports and tests fake injection. |
| `tests/test_run_control.py:280-284` hardcoded `max_context_tokens: 128000` in synthetic event | test fixture stale value | Test emits synthetic context usage with old model window. | Not runtime config, but confusing after cleanup. | fixed | Changed regression coverage to unknown-window path where applicable. |

## Valid Empty Result Policy

Phase 0 preserves valid empty results:

- No historical episodic or semantic memories for a user may return `[]`.
- Empty input list to an embedding provider may return `[]`.
- Empty memory context sections may produce an empty `MemoryContextInjection`.

Phase 0 fails fast on runtime failure:

- Missing `memory.embedding_provider`
- Unsupported `memory.embedding_provider`, including `dummy`, `fake`, or `test`
- Missing embedding model/base URL/API key env config
- Missing API key when constructing a real embedding provider
- Embedding API HTTP/JSON/schema failures
- DB query failures during memory retrieval
- Structured classifier output failures
- Missing or invalid `memory.token_budget.*` config

## Deferred Findings

Deferred findings are intentionally not deleted or rewritten in Phase 0. They remain visible here and in the existing governance reports so future work can address them with separate specs.

## Phase 0 Delivery Confirmation

Date: 2026-06-29

### Ruff Check Remaining Blockers

Command:

```powershell
ruff check . --output-format=json
```

Result: 81 remaining Ruff findings across 32 files. They are treated as pre-existing blockers for Phase 0 because none are in the new Phase 0 files or the touched Phase 0 lines. The only file in both the Phase 0 diff and the remaining Ruff output is `src/graph/academic.py`; the Phase 0 diff touches only lines 6877 and 6999, while Ruff reports lines 5250, 7070, 7113, and 7189.

| File | Ruff code(s) and locations | Pre-existing reason |
|---|---|---|
| `app.py` | `E402` at 24, 30, 31, 32, 33, 34, 35, 50, 51, 52, 53, 54, 147 | Module import ordering around dotenv/app instrumentation predates Phase 0; file was not touched by this cleanup. |
| `scripts/debug_evidence_judge_schema_probe.py` | `E402` at 19 | Script path/bootstrap import style predates Phase 0; file was not touched. |
| `scripts/debug_rag.py` | `E402` at 23, 24 | Script path/bootstrap import style predates Phase 0; file was not touched. |
| `scripts/demo_profile.py` | `E402` at 27, 29; `F401` at 27, 32, 33; `F541` at 110 | Existing demo-script import and unused-symbol issues; file was not touched. |
| `scripts/doctor_rag_env.py` | `F541` at 94; `F841` at 141 | Existing diagnostic-script formatting/unused-variable issues; file was not touched. |
| `scripts/e2e_test_evidence_judge.py` | `E401` at 2; `F401` at 2; `E402` at 21, 22; `F541` at 65, 77 | Existing E2E script style issues; file was not touched. |
| `scripts/inspect_chunks.py` | `E402` at 21 | Existing script path/bootstrap import style; file was not touched. |
| `src/analytics/cognitive_graph.py` | `F401` at 19 | Existing unused import; file was not touched. |
| `src/analytics/explainability_engine.py` | `F841` at 89, 124 | Existing unused exception variables; file was not touched. |
| `src/analytics/growth_analyzer.py` | `F841` at 53 | Existing unused exception variable; file was not touched. |
| `src/analytics/memory_dashboard.py` | `F401` at 17 | Existing unused analytics type imports; file was not touched. |
| `src/curriculum/knowledge_graph.py` | `F401` at 14 | Existing unused import; file was not touched. |
| `src/curriculum/path_planner.py` | `F841` at 90, 286; `F541` at 107 | Existing unused locals and static f-string; file was not touched. |
| `src/graph/academic.py` | `F821` at 5250; `F841` at 7070, 7113, 7189 | Existing graph issues outside Phase 0 diff lines. Phase 0 changed only memory retriever/writer exception propagation at 6877 and 6999. |
| `src/graph/llm.py` | `F841` at 62 | Existing unused `provider_name` in legacy LLM bridge; file was not touched in Phase 0. |
| `src/memory/semantic.py` | `F841` at 96, 150 | Existing deferred semantic-memory exception handling findings; file was not touched in Phase 0. |
| `src/memory/storage.py` | `F401` at 18 | Existing unused import; file was not touched. |
| `src/profile/extractor.py` | `F401` at 14 | Existing unused import; file was not touched. |
| `src/profile/manager.py` | `F401` at 21 | Existing unused import; file was not touched. |
| `src/profile/scorer.py` | `F401` at 18 | Existing unused import; file was not touched. |
| `src/profile/storage.py` | `F401` at 17 | Existing unused import; file was not touched. |
| `src/profile/updater.py` | `F401` at 20, 23, 30 | Existing unused imports; file was not touched. |
| `tests/conftest.py` | `F401` at 11 | Existing unused import; file was not touched. |
| `tests/test_app.py` | `F401` at 5 | Existing unused import; file was not touched. |
| `tests/test_checkpointer.py` | `F401` at 15, 19 | Existing unused imports; file was not touched. |
| `tests/test_config.py` | `F401` at 11 | Existing unused import; file was not touched. |
| `tests/test_emotional.py` | `F401` at 7 | Existing unused import; file was not touched. |
| `tests/test_memory_storage.py` | `F401` at 5 | Existing unused import; file was not touched. |
| `tests/test_profile.py` | `F401` at 11, 36, 37; `F841` at 256, 1162, 1206, 1314 | Existing unused imports/locals; file was not touched. |
| `tests/test_profile_manager.py` | `F401` at 11, 22; `F841` at 234, 758 | Existing unused imports/locals; file was not touched. |
| `tests/test_section_splitter.py` | `F401` at 16, 17 | Existing unused imports; file was not touched. |
| `tests/test_security.py` | `F401` at 9 | Existing unused import; file was not touched. |
| `tests/test_tracing.py` | `F401` at 19, 20; `F841` at 548 | Existing unused imports/local; file was not touched. |

Scoped Ruff checks for the Phase 0 touched files pass after this cleanup. Full-repo `ruff check .` remains blocked by the table above.

### Context Usage Unknown Window Confirmation

Commands:

```powershell
python -m pytest tests/test_no_context_phase0_regression.py -q --basetemp .pytest_tmp_phase0_confirm
```

```powershell
@'
from src.config import clear_cache
from src.observability.context_usage import build_context_usage_payload

clear_cache()
stage, payload = build_context_usage_payload(
    node_name="study_plan_agent",
    llm_node="study_plan",
    provider="deepseek_official",
    model="deepseek-v4-pro",
    messages=[],
)
print(stage)
print(payload)
'@ | python -
```

Observed result:

- `tests/test_no_context_phase0_regression.py`: `5 passed`
- Direct payload stage: `context_usage_error`
- Direct payload reason: `model_window_unknown`

Conclusion: after `context_budget.model_limits` was emptied, existing context usage telemetry emits an explicit `context_usage_error` payload with `reason="model_window_unknown"`. The `/stream` and `/resume` regression tests continue to emit normal SSE events and do not crash.

### Provider Branch Confirmation

Production provider branch scan:

```powershell
rg -n -i "\b(dummy|fake|zero_vector)\b|zero-vector|keyword-only" src app.py config/settings.yaml
rg -n -i "memory\.embedding_provider.*(dummy|fake|test|zero_vector)|provider_name.*(dummy|fake|test|zero_vector)|if .*provider.*(dummy|fake|test|zero_vector)|elif .*provider.*(dummy|fake|test|zero_vector)|case .*(dummy|fake|test|zero_vector)" src app.py config/settings.yaml
```

Observed result: no production matches for forbidden provider names or provider-name branches.

The production memory embedding factory currently has only one provider branch:

- `src/memory/embeddings.py`: `if provider_name != "deepseek": raise MemoryEmbeddingConfigError(...)`

Test fake scan:

```powershell
rg -n "DeterministicFakeEmbeddingProvider|tests\.fakes" tests src config/settings.yaml
```

Observed result:

- Fake implementation exists at `tests/fakes/embeddings.py`.
- Fake imports/usages are limited to tests.
- Retrieval tests pass the fake via `embedding_provider=DeterministicFakeEmbeddingProvider()`.
- Fail-fast tests use monkeypatch to prove the production factory is not called when a fake is explicitly injected.

Conclusion: production code has no `dummy`, `fake`, `test`, or `zero_vector` provider branch. The deterministic fake embedder is test-only and is used only through monkeypatch or dependency injection.
