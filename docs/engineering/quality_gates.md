# Quality Gates

Quality gates are layered. Use the smallest meaningful set for the change, then expand when risk increases.

## Reproducible Python Tooling

Install the repository-pinned Python quality group in an isolated environment:

```powershell
python -m pip install -e ".[quality]"
```

The group pins Bandit, import-linter, mypy, Ruff, and Vulture. Semgrep and
Gitleaks remain separate platform tools; report either one as unavailable when
it is not installed instead of treating another scanner as an equivalent pass.

## Always

Before editing:

```powershell
git status --short
Get-Content -Raw .\AGENTS.md
Get-Content -Raw .\skills\spec_first_change\SKILL.md
```

After editing:

```powershell
git status --short
git diff --check
```

## Python Static Gate

For Python code changes:

```powershell
python -m py_compile <changed-python-files>
python -m compileall -q src tests app.py
ruff check .
ruff format --check .
python -m pytest <related tests> -q
```

If Ruff is missing, report it as missing. Do not call another command "Ruff passed."

## Structured Output Gate

Use when touching Pydantic models, structured output, graph outputs, profile extraction, or business validators:

```powershell
python -m pytest tests/test_deepseek_structured_output.py tests/test_structured_retry.py tests/test_structured_output_contract.py -q
semgrep --config semgrep_rules/a3_no_fallback_no_hardcode.yml src tests app.py
```

Add or preserve negative tests for validation failures and schema drift.

## Fallback and Hardcoding Gate

Use when touching LLM provider/model/base_url/api_key/config, retries, fallback, or error handling:

```powershell
semgrep --config semgrep_rules/a3_no_fallback_no_hardcode.yml src tests app.py
python -m pytest tests/test_config.py tests/test_llm_fallback.py tests/test_deepseek_structured_output.py -q
```

Findings in legacy code are reported under `docs/reports/`; new findings in the diff must stop the change.

## Architecture Gate

Use when changing imports or package boundaries:

```powershell
lint-imports --config .importlinter
```

The initial import-linter contracts protect:

- `src.llm` from importing graph business nodes.
- graph/profile/tools business nodes from importing provider SDKs.
- `src.config` from importing runtime business layers.

## Type Gate

Priority protected paths:

- `src/llm`
- `src/config`
- `src/graph/web_research.py`
- `src/graph/evidence.py`
- `src/profile`

Run available type checkers:

```powershell
mypy src/llm src/config src/graph/web_research.py src/graph/evidence.py src/profile
pyright src/llm src/config src/graph/web_research.py src/graph/evidence.py src/profile
ty check src/llm src/config src/graph/web_research.py src/graph/evidence.py src/profile
```

Missing tools are reported as missing. Do not replace precise types with `Any` to pass.

## Security Gate

Use when touching secrets, traces, auth, HTTP, DB URIs, subprocesses, or file handling:

```powershell
gitleaks detect --source . --redact
bandit -r src -x tests
python -m pytest tests/test_security.py -q
```

Reports must not contain real secret values.

## Dead Code Gate

Dead-code analysis is report-only unless the user explicitly approves cleanup:

```powershell
vulture src tests --min-confidence 80
```

Summarize candidates in `docs/reports/dead_code_candidates.md`. Do not automatically delete old fallback/helper/dead code.

## Pre-Commit

`.pre-commit-config.yaml` uses local lightweight hooks only:

- Python syntax compile.
- Block secret-bearing `.env` files.
- Conflict marker scan.

It intentionally avoids heavy first-round gates and third-party hook repositories.

## CodeQL Medium-Term Plan

Do not wire CodeQL in the first governance round. A later phase can add a dedicated GitHub Actions workflow after:

1. Baseline Semgrep, Bandit, Gitleaks, import-linter, Ruff, and related pytest are stable.
2. False-positive ownership is assigned.
3. Query scope is documented.
4. CI runtime budget is agreed.

Candidate future workflow:

- GitHub CodeQL analysis for Python.
- Pull request annotations.
- Scheduled weekly deep scan.
- Security review owner for triage.
