---
name: architecture_boundary
description: Repo-scoped A3 Study Agent skill for Python architecture boundaries and import contracts. Use when changing imports, package ownership, graph/profile/tools boundaries, config resolver access, or dependency direction.
---

# architecture_boundary

## Purpose

Prevent architecture drift by keeping package ownership and import direction explicit. Use import-linter style contracts plus local review to stop business nodes from bypassing config or provider boundaries.

## When to use

Use when changing imports, moving code, introducing helpers, touching src/llm, src/config, src/graph, src/profile, src/tools, or adding a dependency between packages.

## Required inputs

- Current spec_first_change output.
- Proposed diff or touched modules.
- Existing .importlinter configuration.
- Current imports in the touched files.
- Any provider/config/security implications.

## Procedure

1. Identify the owner package for each new helper or dependency.
2. Keep src/config low-level: it must not import graph/profile/tools/llm runtime logic.
3. Keep src/llm from importing graph business nodes. The current src.graph.llm bridge is legacy coupling and must not spread.
4. Keep provider SDK imports out of graph business nodes, profile, and tools unless the module is explicitly the provider boundary.
5. Route provider/model/base_url/api_key decisions through config resolver APIs.
6. Update .importlinter only when the intended boundary changes are explicit and reviewed.
7. Run `lint-imports --config .importlinter` when import-linter is available.
8. If import-linter cannot express a rule, document the manual check in the work summary or docs/engineering/quality_gates.md.

## Forbidden actions

- Do not create convenience imports that make src/llm depend on graph business nodes.
- Do not put provider clients in arbitrary graph/profile/tools modules.
- Do not solve circular imports by moving code into unrelated packages.
- Do not add broad `try/except ImportError` fallback imports.
- Do not hide architecture violations by weakening .importlinter.
- Do not perform one-shot large refactors.

## Required output

When used, include:

```text
Architecture boundary review:
- New imports added:
- Boundary affected:
- Config resolver bypass introduced: no
- Provider SDK in business node introduced: no
- import-linter run: yes/no
```

## Commands

```powershell
Select-String -Path 'src\**\*.py' -Pattern '^from ','^import '
lint-imports --config .importlinter
```

If import-linter is unavailable, state that it was not run and perform a manual import review of touched files.

## Stop conditions

- The change introduces a cycle between src/llm and src/graph business modules.
- The change requires graph/profile/tools to instantiate provider clients directly.
- The change requires config to import runtime business logic.
- The intended ownership of a new helper is unclear.
- The boundary change is broad enough to require a separate architecture plan.

## A3_study_agent-specific rules

- src/config is the resolver boundary for settings.
- src/graph/*.py, src/profile/*.py, and src/tools/*.py must not hardcode provider/model/base_url/api_key.
- src/llm should not grow additional dependency on src/graph business nodes.
- .importlinter is an initial guard, not a substitute for semantic scans.
- Semgrep guards hardcoding and fallback patterns that import-linter cannot express.
