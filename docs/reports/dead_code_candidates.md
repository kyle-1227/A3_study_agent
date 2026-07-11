# Dead Code Candidates

Initial governance report created on 2026-06-20. No dead code was deleted.

## Policy

- Vulture and text scans are report-only.
- Do not delete legacy fallback/helper/dead code without explicit approval.
- Verify dynamic references before cleanup, especially LangGraph nodes, FastAPI routes, config keys, prompts, and tests.

## Current Baseline

Vulture has not been run in this first governance round. No dead-code candidates are approved for deletion.

## 2026-07-10 Supervisor phrase-detector candidates

Candidate: Legacy resource-request phrase detectors
File: `src/graph/supervisor.py`
Symbol: `_READABLE_*_MARKERS`, `_detect_requested_resource_types`, `_detect_requested_resource_type`
Evidence: Repository reference scan finds runtime definitions and test imports only; `supervisor_node` uses strict structured output and does not call these helpers.
Confidence: High
Dynamic reference checks: LangGraph builder and prompt/config scans show no dynamic symbol lookup for these private helpers.
Related tests: `tests/test_supervisor.py`
Recommended action: Keep report-only until explicit deletion approval; do not use these helpers as a runtime routing source.

## 2026-07-10 RAG parent JSONL cleanup candidate

Candidate: `parent_chunks.jsonl` cleanup entry
File: `scripts/reset_index.py`
Symbol: cleanup target only; no corresponding writer or reader was found
Evidence: Repository reference scans find the filename only in the reset
script and a metadata-schema test. No production Parent Store implementation
uses it at the approved baseline.
Confidence: Medium; this may be an unfinished design reservation rather than
dead code.
Dynamic reference checks: No config, prompt, CLI, or runtime import constructs
the filename outside the reset path.
Related tests: `tests/test_metadata_schema.py`
Recommended action: Keep report-only. Do not delete the entry until the
generation-owned cleanup path is implemented and separately approved.

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
