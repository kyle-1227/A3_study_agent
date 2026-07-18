# A3 Study Agent Codex Instructions

These instructions apply to Codex work in this repository.

## Repo-Scoped Skills

Before modifying this repository, Codex must read the relevant repo-scoped skills under `./skills`. These skills are version-controlled governance rules for A3_study_agent and take precedence over ad hoc coding habits.

Required skill routing:

- Use `skills/spec_first_change/SKILL.md` before every code, test, config, CI, script, or process modification.
- Use `skills/static_quality_gate/SKILL.md` after every modification.
- Use `skills/structured_output_contract/SKILL.md` for structured output, Pydantic schemas, LLM JSON parsing, business validators, graph outputs, and profile extraction.
- Use `skills/architecture_boundary/SKILL.md` for imports, package boundaries, helper placement, and dependency direction.
- Use `skills/type_contract/SKILL.md` for `src/llm`, `src/config`, `src/graph/web_research.py`, `src/graph/evidence.py`, and `src/profile`.
- Use `skills/security_secret/SKILL.md` for secrets, trace bodies, auth headers, DB URIs, network calls, subprocesses, and security tests.
- Use `skills/dead_code_and_diff_risk/SKILL.md` before deleting code, moving helpers, or reporting stale fallback/dead code.

## Hard Rules

- Production fallback is allowed only when its mini-spec and strict configuration define a finite call/time/retry budget, typed reason and status, sanitized observability, and the same provider/model/runtime identity.
- A fallback result may be treated as success only after the normal Pydantic and business validators pass; partial, empty, guessed, or stale results must remain explicitly degraded or blocked.
- Do not add silent defaults, cross-provider/model routing, legacy-chain switching, validation bypasses, or fallback-derived success without validated evidence.
- Do not bypass Pydantic validation.
- Do not bypass business validation.
- Do not automatically alias-normalize structured output to make Pydantic pass.
- Do not reintroduce OpenRouter DeepSeek calls or assumptions.
- Do not hardcode provider/model/base_url/api_key in business nodes.
- Do not delete real business logic to make tests pass.
- Do not perform unplanned large-scale refactors.
- If old fallback/helper/dead code is discovered, write a report under `docs/reports/` and do not delete it without explicit approval.

## Protected Runtime Areas

The following areas are high-risk and require the relevant skill before any runtime edit:

- `src/llm/structured_output.py`
- `src/llm/`
- `src/config/`
- `src/graph/*.py`
- `src/profile/*.py`
- `src/tools/*.py`

First prefer small, explicit changes with focused tests. If a request implies broad architecture work, stop and propose a plan before editing.

## Third-Party Skill and Script Policy

- Do not blindly install third-party skills from GitHub.
- Do not execute unknown third-party skill scripts.
- Do not install third-party skills globally into Codex for this repo.
- When using open-source projects as references, adapt only their design patterns into repo-scoped, auditable files.
- Any scripts added for governance must be generated inside this project, readable, and version-controlled.

## Quality Gates

After changes, run the relevant gate from `skills/static_quality_gate/SKILL.md`. At minimum, report:

- Python compile or docs/config sanity check.
- Ruff check and Ruff format check when Ruff is installed.
- Related pytest for code changes.
- Semgrep, import-linter, type checker, Gitleaks, Bandit, or Vulture only when relevant to the touched area and available.

Missing tools must be reported as missing, not treated as passing.
