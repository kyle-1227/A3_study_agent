# Anti Vibe Coding Governance

This repository uses repo-scoped Codex skills to keep AI-assisted changes explicit, testable, and reviewable.

## Why

A3_study_agent has high-risk surfaces around LLM providers, structured output, graph orchestration, profile extraction, and evidence validation. These areas are easy to make "work" by adding fallback, defaults, aliases, or broad exception handling. That kind of work hides real failures and makes later maintenance harder.

## Core Workflow

1. Read `AGENTS.md`.
2. Read the relevant skill under `./skills`.
3. Use `spec_first_change` before editing.
4. Make the smallest change that satisfies the request.
5. Use focused tests and quality gates after editing.
6. Report legacy fallback/helper/dead-code findings instead of deleting them opportunistically.

## Strong Rules

- No new fallback.
- No new silent default.
- No Pydantic validation bypass.
- No business validation bypass.
- No automatic alias normalization.
- No OpenRouter DeepSeek reintroduction.
- No provider/model/base_url/api_key hardcoding in business nodes.
- No deletion of real business logic just to make tests pass.
- No unplanned large-scale refactor.

## Skill Map

- `skills/spec_first_change/SKILL.md`: pre-change mini-spec.
- `skills/static_quality_gate/SKILL.md`: post-change checks.
- `skills/no_fallback_no_hardcode_guard/SKILL.md`: fallback and hardcoding guard.
- `skills/structured_output_contract/SKILL.md`: schema and validation integrity.
- `skills/architecture_boundary/SKILL.md`: import and ownership boundaries.
- `skills/type_contract/SKILL.md`: protected type surfaces.
- `skills/security_secret/SKILL.md`: secret and Python security baseline.
- `skills/dead_code_and_diff_risk/SKILL.md`: report-only dead-code and cleanup risk.

## Reference Patterns

This governance layer adapts patterns from:

- Spec-first development and convergence checklists: https://github.com/github/spec-kit
- Python lint/format gates: https://github.com/astral-sh/ruff
- Commit-time local hooks: https://github.com/pre-commit/pre-commit
- Static type contracts: https://github.com/python/mypy, https://github.com/microsoft/pyright, https://github.com/astral-sh/ty
- Import contracts: https://github.com/seddonym/import-linter
- Semantic static analysis: https://github.com/semgrep/semgrep
- Secret scanning: https://github.com/gitleaks/gitleaks
- Python security baseline: https://github.com/PyCQA/bandit
- Dead-code reporting: https://github.com/jendrikseipp/vulture
- Medium-term deep security scanning: https://github.com/github/codeql

The repository does not install third-party skills from those projects. It adapts their design ideas into local, auditable files.
