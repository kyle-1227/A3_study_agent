---
name: no_fallback_no_hardcode_guard
description: Repo-scoped A3 Study Agent guard for detecting and preventing hidden fallback, silent defaults, provider/model/base_url/api_key hardcoding, OpenRouter DeepSeek residue, alias auto-normalization, and swallowed validation errors.
---

# no_fallback_no_hardcode_guard

## Purpose

Protect A3_study_agent from maintainability decay caused by hidden fallback paths, hardcoded LLM configuration, schema-drift shortcuts, and exception handlers that return plausible but invalid business results.

## When to use

Use for any change touching LLM calls, provider configuration, model selection, base_url, api_key handling, settings.yaml, env vars, structured-output modes, graph nodes, profile extraction, tools, retries, error handling, or tests around those areas.

## Required inputs

- User request and spec_first_change output.
- Diff or planned edit locations.
- Current configuration entry points, especially src/config and settings files.
- Any related tests under tests/test_config.py, tests/test_deepseek_structured_output.py, tests/test_structured_retry.py, tests/test_llm_fallback.py, and tests/test_security.py.
- Existing reports in docs/reports/fallback_paths.md and docs/reports/hardcoding_scan.md.

## Procedure

1. Read the relevant runtime files before editing.
2. Identify whether the change adds or changes:
   - fallback behavior,
   - default values,
   - provider/model/base_url/api_key lookup,
   - exception handling,
   - structured-output validation or parsing,
   - OpenRouter or DeepSeek-specific behavior.
3. Route all provider/model/base_url/api_key decisions through the config resolver. Do not put provider literals in business nodes.
4. Treat existing fallback as legacy unless the user explicitly asks to preserve or expose it. Do not add a new fallback path.
5. For validation failures, fail loudly with a typed error or explicit diagnostic. Do not return a default business object.
6. Run the Semgrep guard when code changes touch Python:
   `semgrep --config semgrep_rules/a3_no_fallback_no_hardcode.yml src tests app.py`
7. Update docs/reports/fallback_paths.md or docs/reports/hardcoding_scan.md when old fallback or hardcoding is discovered during the work.
8. Hand off to static_quality_gate after edits.

## Forbidden actions

- Do not add fallback modes, fallback models, fallback providers, fallback env vars, fallback return objects, or deterministic fallback logic.
- Do not use `or "default"`, `dict.get(..., "default")`, `os.getenv(..., "default")`, or `get_setting(..., "default")` for provider/model/base_url/api_key in business logic.
- Do not reintroduce OpenRouter DeepSeek routing, `openrouter.ai`, `OPENROUTER_*`, or `sk-or-v1` assumptions.
- Do not catch ValidationError, business validation errors, provider protocol errors, or parsing errors and return a default business result.
- Do not add alias normalization, key rewriting, or fuzzy mapping so Pydantic accepts drifted payloads.
- Do not hide failures behind logs only.
- Do not weaken tests that assert fail-fast behavior.

## Required output

When used, include:

```text
Fallback/hardcode review:
- New fallback added: no
- New silent defaults added: no
- Provider/model/base_url/api_key routed through config: yes/no/not applicable
- Validation/business validation swallowed: no
- OpenRouter DeepSeek residue introduced: no
- Reports updated: yes/no
```

If any item cannot be answered "no" or "yes" safely, stop before editing.

## Commands

```powershell
Select-String -Path 'src\**\*.py','tests\*.py','app.py' -Pattern 'fallback','default','provider','model','base_url','api_key','OPENROUTER','openrouter.ai','ValidationError','model_construct','validation_alias','alias_generator'
semgrep --config semgrep_rules/a3_no_fallback_no_hardcode.yml src tests app.py
python -m pytest tests/test_config.py tests/test_deepseek_structured_output.py tests/test_structured_retry.py tests/test_llm_fallback.py -q
```

If Semgrep is not installed, do not invent an alternate pass/fail result. Record that the guard was not run.

## Stop conditions

- A requested change requires new fallback or silent default behavior.
- The only way to pass tests is to loosen validation or delete business logic.
- Provider/model/base_url/api_key cannot be routed through config without broader design work.
- Existing legacy fallback is discovered in the touched path and the requested change would make it harder to remove later.
- The Semgrep guard reports a new finding in the diff.

## A3_study_agent-specific rules

- Business nodes include src/graph/*.py, src/profile/*.py, and src/tools/*.py.
- Configuration must flow through src/config and the established resolver APIs.
- Existing fallback in src/graph/llm.py, src/llm/structured_output.py, or graph evidence paths is legacy risk. Report it; do not extend it.
- Do not modify src/llm/structured_output.py in the first governance round.
- Do not modify src/graph/*.py or src/profile/*.py in the first governance round.
