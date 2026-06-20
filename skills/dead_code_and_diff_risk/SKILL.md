---
name: dead_code_and_diff_risk
description: Repo-scoped A3 Study Agent skill for dead-code reporting and diff-risk control. Use for Vulture reports, helper audits, stale fallback discovery, and preventing unplanned large refactors.
---

# dead_code_and_diff_risk

## Purpose

Make dead code and risky helpers visible without deleting them casually. This skill turns stale fallback/helper/dead-code findings into reports and keeps diffs small.

## When to use

Use when deleting code, moving helpers, touching legacy fallback paths, reviewing unused functions, preparing refactors, or running Vulture/dead-code analysis.

## Required inputs

- Current spec_first_change output.
- Changed files or candidate files.
- Existing docs/reports/dead_code_candidates.md.
- Vulture output if available.
- Related tests for any code proposed for deletion.

## Procedure

1. Treat dead-code detection as evidence, not proof.
2. Run Vulture in report-only mode when available.
3. Cross-check candidates against dynamic imports, FastAPI routes, LangGraph nodes, tests, CLI entry points, and prompt/config references.
4. Write candidates to docs/reports/dead_code_candidates.md with confidence and reason.
5. Do not delete old fallback/helper/dead code unless the user explicitly approves a separate cleanup.
6. Keep current-task diffs narrow. If cleanup is tempting but out of scope, report it.
7. Pair any approved deletion with tests proving behavior remains intact.

## Forbidden actions

- Do not automatically delete Vulture findings.
- Do not delete legacy fallback/helper code in the same change that discovers it.
- Do not perform broad renames, moves, or file splits without an explicit plan.
- Do not remove real business logic to reduce warnings.
- Do not change runtime behavior during a governance-only pass.

## Required output

When used, include:

```text
Dead code/diff risk review:
- Vulture run:
- Candidates reported:
- Code deleted: no/yes with approval
- Diff remains scoped: yes/no
```

## Commands

```powershell
vulture src tests --min-confidence 80
git diff --stat
git diff --name-only
python -m pytest <related tests> -q
```

Redirect Vulture output to a temporary file for review, then summarize candidates in docs/reports/dead_code_candidates.md. Do not overwrite report context blindly.

## Stop conditions

- A deletion candidate might be used dynamically by LangGraph, FastAPI, tests, config, prompt loading, or external users.
- Cleanup would expand beyond the approved spec.
- Vulture is unavailable and no manual evidence exists.
- Tests are missing for behavior that deletion could affect.

## A3_study_agent-specific rules

- Graph nodes may be dynamically referenced; verify before deletion.
- Prompt/config keys may reference helpers indirectly.
- Existing fallback/helper code is report-only until explicitly approved.
- The first governance round must not delete any old code.
