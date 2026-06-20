---
name: type_contract
description: Repo-scoped A3 Study Agent skill for static type contract review. Use for src/llm, src/config, src/graph/web_research.py, src/graph/evidence.py, src/profile, Pydantic contracts, and public helper signatures.
---

# type_contract

## Purpose

Protect high-risk contracts with static typing discipline. The first priority is to keep LLM, config, evidence, web research, and profile types explicit enough that schema drift and business-state drift are visible.

## When to use

Use for edits under src/llm, src/config, src/graph/web_research.py, src/graph/evidence.py, src/profile, shared schemas, public helper signatures, or tests that depend on those types.

## Required inputs

- Touched files and public signatures.
- Existing type annotations and Pydantic models.
- Related tests.
- Available type checker: mypy, pyright, or ty.
- Any known dynamic boundaries that need explicit narrowing.

## Procedure

1. Identify the contract surface: function parameters, return types, Pydantic fields, TypedDicts, Literals, and state dictionaries.
2. Preserve or improve annotations on changed public functions.
3. Avoid `Any` unless the boundary is genuinely untyped; narrow immediately after external input.
4. Do not silence type errors with ignores unless the reason is documented and tightly scoped.
5. Prefer explicit `Literal`, `TypedDict`, `Protocol`, or Pydantic model contracts for structured outputs.
6. Run the available type checker on protected paths.
7. Pair type checks with related pytest when behavior is affected.

## Forbidden actions

- Do not replace precise types with `Any` to make a checker pass.
- Do not add broad `# type: ignore`, `pyright: ignore`, or casts without a local explanation.
- Do not make optional values look required through unchecked assertions.
- Do not loosen Pydantic models to match invalid output.
- Do not use type changes to hide fallback or validation behavior.

## Required output

When used, include:

```text
Type contract review:
- Protected paths touched:
- Public signatures changed:
- Any/cast/ignore added: no
- Type checker run:
- Related tests:
```

## Commands

Run whichever tools are installed; do not claim missing tools passed.

```powershell
mypy src/llm src/config src/graph/web_research.py src/graph/evidence.py src/profile
pyright src/llm src/config src/graph/web_research.py src/graph/evidence.py src/profile
ty check src/llm src/config src/graph/web_research.py src/graph/evidence.py src/profile
python -m pytest tests/test_config.py tests/test_web_research_v2.py tests/test_evidence_judge_v2.py tests/test_profile.py tests/test_profile_manager.py -q
```

## Stop conditions

- The intended change requires weakening a public type contract without explicit approval.
- Type checker output shows a real contract mismatch in the touched path.
- A schema or state type is unclear enough that implementation would be guesswork.
- Fixing type errors would require broad refactoring outside the spec.

## A3_study_agent-specific rules

- Highest-priority protected paths: src/llm, src/config, src/graph/web_research.py, src/graph/evidence.py, src/profile.
- Structured-output types must align with business validators and tests.
- Config access types must not encourage hardcoded provider/model/base_url/api_key defaults.
- Profile schema changes need migration or compatibility analysis before runtime edits.
