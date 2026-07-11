# Hardcoding Scan Report

Initial governance report created on 2026-06-20. This is a lightweight text-scan baseline from repository inspection, not a complete semantic scan.

## Policy

- Do not hardcode provider/model/base_url/api_key in business nodes.
- Route provider configuration through `src/config`.
- Do not reintroduce OpenRouter DeepSeek routing.
- Do not commit secrets or raw trace bodies.

## Observed Legacy Risk Areas

### 2026-07-10 Parent-child RAG implementation baseline

- `scripts/build_index.py` contains a fixed five-subject directory map while
  query and evaluation paths use separate dynamic discovery implementations.
- `src/rag/indexer.py` contains legacy collection, persistence, embedding
  provider, model, base URL, API-key environment, batching, and retry defaults.
- `src/rag/reranker.py` contains legacy provider/model/base URL/API-key
  environment defaults.
- New parent-child generation code must load these values only from the strict
  `src.config.rag_*` resolvers. This report does not approve changing or
  deleting the legacy baseline paths in the same diff.

Implementation update (2026-07-11): the fixed five-subject map was removed
from `scripts/build_index.py`. The legacy flat-baseline CLI now requires an
explicit pipeline, paths, subject list, embedding identity, key environment
name, timeout, batch size, and retry count. The defaults inside
`src/rag/indexer.py` remain reported legacy behavior; the new parent-child path
does not import or use them.

The parent-child mode of `scripts/doctor_rag_env.py` now requires explicit
index, benchmark, and rollout config paths. It discovers subjects through the
same strict `SubjectCatalog`, checks configured secret environment names
without logging values, and performs no network call. The former fixed course
directory list and fixed OpenRouter environment-variable inventory were
removed from this diagnostic boundary.

### 2026-06-29 Context Engineering Phase 0

- Removed the incorrect `context_budget.model_limits.deepseek-v4-pro` window value from `config/settings.yaml`.
- Did not add a replacement hardcoded model window; current context usage telemetry should surface unknown windows explicitly.
- Phase 0 hardcoding findings and deferred items are recorded in `docs/reports/context_engineering_phase0_audit.md`.

### 2026-06-28 Run Control implementation note

- Added `context_budget.model_limits` in `config/settings.yaml` for context telemetry.
- No provider/model/base_url/api_key literal was added to graph business nodes.
- Missing or unknown model context windows are surfaced as telemetry warnings instead of code-level hardcoded defaults.

### `src/graph/llm.py`

- Contains `ChatOpenAI` provider construction.
- Contains DeepSeek-oriented env/default names such as `DEEPSEEK_MODEL`, `DEEPSEEK_API_KEY`, and `DEEPSEEK_BASE_URL`.
- Contains OpenRouter header handling for `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE`.
- Contains literal base URL defaults.

### `src/tools/search_tool.py`

- Uses Tavily API key env lookup.
- This should remain explicit and redacted in diagnostics.

### `src/tools/document_tool.py` and `src/tools/mindmap_tool.py`

- Use env-configured artifact directories. These are not provider/model hardcoding but should remain bounded and test-covered.

## Semgrep Rule

The first local rule file is:

```text
semgrep_rules/a3_no_fallback_no_hardcode.yml
```

Run:

```powershell
semgrep --config semgrep_rules/a3_no_fallback_no_hardcode.yml src tests app.py
```

Findings in old code should be classified and reported. Findings introduced in a new diff must stop the change.

## Follow-Up

1. Establish a reviewed baseline for legacy provider defaults.
2. Move remediation into small specs.
3. Add targeted tests before changing provider/config behavior.
