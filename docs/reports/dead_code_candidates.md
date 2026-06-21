# Dead Code Candidates

Initial governance report created on 2026-06-20. No dead code was deleted.

## Policy

- Vulture and text scans are report-only.
- Do not delete legacy fallback/helper/dead code without explicit approval.
- Verify dynamic references before cleanup, especially LangGraph nodes, FastAPI routes, config keys, prompts, and tests.

## Current Baseline

Vulture has not been run in this first governance round. No dead-code candidates are approved for deletion.

## How to Add Findings

Use this format:

```text
Candidate:
File:
Symbol:
Evidence:
Confidence:
Dynamic reference checks:
Related tests:
Recommended action:
```

## Follow-Up Command

```powershell
vulture src tests --min-confidence 80
```

Review output manually and update this report. Do not automatically apply deletions.
