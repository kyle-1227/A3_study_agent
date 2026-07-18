---
name: spec_first_change
description: Repo-scoped A3 Study Agent skill for spec-first changes. Use before any code, test, config, or workflow modification to state goal, scope, non-goals, risks, test plan, and acceptance criteria before editing.
---

# spec_first_change

## Purpose

Force every A3_study_agent change to start from an explicit mini-spec before touching files. The goal is to prevent vague vibe-coding edits, accidental broad refactors, hidden fallback additions, and untestable changes.

## When to use

Use before every repository modification, including code, tests, config, docs that define engineering process, CI, scripts, prompts, schemas, and agent instructions.

Do not skip this skill for "small" code changes. For pure read-only analysis, code review, or answering a question without edits, this skill is not required.

## Required inputs

- User request and latest clarification.
- Current working tree status.
- Files or subsystems likely to be touched.
- Known constraints from AGENTS.md and relevant repo-scoped skills.
- Whether the change touches structured output, LLM provider/model/config, graph nodes, profile logic, tools, security-sensitive code, or tests.

## Procedure

1. Read AGENTS.md and the relevant files under ./skills before planning.
2. Inspect the current worktree with `git status --short`.
3. Identify the smallest viable change that satisfies the request.
4. Write the mini-spec before editing:
   - Objective
   - Scope
   - Non-goals
   - Risks
   - Test plan
   - Acceptance criteria
5. Name any additional required skills:
   - structured_output_contract for schemas, Pydantic models, structured LLM parsing, retries, or business validators.
   - architecture_boundary for cross-package imports or dependency direction.
   - type_contract for src/llm, src/config, src/graph/web_research.py, src/graph/evidence.py, or src/profile.
   - security_secret for secrets, trace bodies, API keys, auth, shell execution, file paths, or network data.
6. Stop and ask the user before broad refactors, runtime behavior removals, or changes outside the request.
7. After edits, run static_quality_gate.

## Forbidden actions

- Do not edit files before producing the mini-spec.
- Do not expand scope because nearby code looks messy.
- Do not introduce unplanned, implicit, unbounded, cross-identity, or validation-bypassing fallback.
- Do not introduce silent defaults or alias normalization as a convenience.
- Do not remove real business logic to make tests pass.
- Do not change src/llm/structured_output.py, src/graph/*.py, or src/profile/*.py unless the user explicitly asks for that runtime change.
- Do not install third-party skills or execute unknown third-party scripts.

## Required output

Before edits, output this compact spec:

```text
Objective:
Scope:
Non-goals:
Risks:
Test plan:
Acceptance criteria:
Skills used:
```

If a required input is missing but a conservative assumption is safe, state the assumption. If the assumption could change runtime behavior or architecture, stop and ask.

## Commands

```powershell
git status --short
Get-ChildItem -Force .\skills
Get-Content -Raw .\AGENTS.md
```

Use focused file reads for touched areas, for example:

```powershell
Get-Content -Raw .\src\llm\structured_output.py
Get-Content -Raw .\src\config\config_manager.py
```

## Stop conditions

- The requested change conflicts with AGENTS.md or A3-specific rules.
- The user requests broad refactoring without an explicit plan.
- A proposed fallback cannot be explicitly configured, given a finite budget, typed and observed, kept on the same provider/model/runtime identity, and passed through the complete Pydantic and business validation path.
- The change requires silent defaults, provider hardcoding, or validation bypasses.
- The current worktree has user changes in files that must be edited and the safe merge path is unclear.
- Required tests or tools are unavailable and the change is high-risk.

## A3_study_agent-specific rules

- Production fallback must be explicitly configured, finite-budget, typed, observable, and identity-preserving.
- Fallback output may succeed only after normal Pydantic and business validation; partial, empty, guessed, or stale output stays degraded or blocked.
- No silent default, cross-provider/model fallback, legacy-chain switch, or validation bypass.
- No Pydantic validation bypass.
- No business validation bypass.
- No automatic alias normalization to make Pydantic pass.
- No reintroduction of OpenRouter DeepSeek calls.
- No provider/model/base_url/api_key hardcoding in business nodes.
- Do not delete real business logic to make tests pass.
- Do not perform one-shot large refactors.
- If old fallback/helper/dead code is discovered, report it under docs/reports and do not delete it without explicit approval.
