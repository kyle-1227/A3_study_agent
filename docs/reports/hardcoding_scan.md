# Hardcoding Scan Report

Initial governance report created on 2026-06-20. This is a lightweight text-scan baseline from repository inspection, not a complete semantic scan.

## Policy

- Do not hardcode provider/model/base_url/api_key in business nodes.
- Route provider configuration through `src/config`.
- Do not reintroduce OpenRouter DeepSeek routing.
- Do not commit secrets or raw trace bodies.

## Observed Legacy Risk Areas

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
