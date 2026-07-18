---
name: static_quality_gate
description: Repo-scoped A3 Study Agent skill for post-change quality gates. Use after edits to run Ruff check, Ruff format check, Python compile checks, and related pytest without hiding failures.
---

# static_quality_gate

## Purpose

Make every change leave a visible quality signal. The gate favors fast, focused checks first, then broader checks when risk or blast radius justifies them.

## When to use

Use after every code, test, config, CI, script, or process change. For docs-only changes, run file/content sanity checks and any relevant configuration validation.

## Required inputs

- spec_first_change output.
- List of changed files from `git status --short`.
- Touched Python modules and related tests.
- Relevant skills used during the change.
- Tool availability in the local environment.

## Procedure

1. Inspect changed files.
2. For Python changes, run syntax/compile checks on changed Python files or the package.
3. Run Ruff lint and format checks when Ruff is available:
   - `ruff check .`
   - `ruff format --check .`
4. Run related pytest, choosing the narrowest meaningful set first.
5. For structured output, provider config, architecture, type, or security work, run the extra gates from those skills.
6. Do not reinterpret a failed gate as success. Fix the issue or report it clearly.
7. Report commands run and results.

## Forbidden actions

- Do not skip tests because they are inconvenient.
- Do not delete or weaken tests to pass the gate.
- Do not add or broaden fallback merely to satisfy tests; any authorized fallback must remain explicit, finite-budget, typed, observable, identity-preserving, and pass the normal Pydantic and business validation path.
- Do not add silent defaults to satisfy tests.
- Do not run a different command and claim the required command passed.
- Do not hide missing tools as "passed".
- Do not leave generated caches or unrelated formatting churn.

## Required output

At completion, include:

```text
Quality gate:
- py_compile/compileall:
- ruff check:
- ruff format --check:
- related pytest:
- extra gates:
```

Use `not run` only with a reason.

## Commands

```powershell
git status --short
python -m py_compile <changed-python-files>
python -m compileall -q src tests app.py
ruff check .
ruff format --check .
python -m pytest <related tests> -q
```

For docs/config-only changes:

```powershell
git diff --check
python - <<'PY'
from pathlib import Path
for path in [
    Path("AGENTS.md"),
    Path(".pre-commit-config.yaml"),
    Path(".importlinter"),
    Path("semgrep_rules/a3_no_fallback_no_hardcode.yml"),
]:
    if path.exists():
        path.read_text(encoding="utf-8")
print("text files readable")
PY
```

## Stop conditions

- A required gate fails and the fix would exceed the approved scope.
- Ruff or pytest reveals unrelated failures that cannot be separated from the change.
- Tests require real secrets or external services.
- A missing tool is required for the risk level and cannot be installed within the user's constraints.

## A3_study_agent-specific rules

- Related tests often live in tests/test_config.py, tests/test_deepseek_structured_output.py, tests/test_structured_retry.py, tests/test_security.py, tests/test_profile.py, and graph-specific tests.
- For LLM/provider/config changes, run `semgrep_rules/a3_no_fallback_no_hardcode.yml` when Semgrep is available; use it to detect unsafe fallback and hardcoding patterns without treating a separate skill as required.
- For structured-output changes, include structured_output_contract checks.
- For import changes, include architecture_boundary checks.
- For protected modules, include type_contract checks when type tools are available.
