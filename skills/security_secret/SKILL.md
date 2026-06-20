---
name: security_secret
description: Repo-scoped A3 Study Agent skill for secret scanning and Python security baseline. Use for API keys, env files, trace payloads, DB URIs, auth headers, network calls, file handling, Bandit, Gitleaks, and security tests.
---

# security_secret

## Purpose

Prevent credential leakage and obvious Python security regressions. Keep traces, reports, logs, and tests from exposing raw secrets or full provider bodies.

## When to use

Use when touching env handling, API keys, DB URIs, trace emission, HTTP requests, auth headers, logging, file paths, subprocesses, serialization, user uploads, or security tests.

## Required inputs

- Touched files and diff.
- Secret-bearing variables: DEEPSEEK_API_KEY, OPENROUTER_API_KEY, TAVILY_API_KEY, DB URIs, Authorization headers, cookies, raw HTTP bodies.
- Existing tests/test_security.py.
- Whether Gitleaks and Bandit are installed locally.

## Procedure

1. Identify every secret source and every sink: logs, traces, exceptions, reports, docs, test snapshots, and subprocess output.
2. Redact secrets before writing diagnostics.
3. Avoid storing raw provider request or response bodies unless explicitly needed, bounded, and redacted.
4. Do not print DB URIs with passwords or API tokens.
5. Run Gitleaks and Bandit when relevant and available.
6. Run tests/test_security.py for security-sensitive changes.
7. Record new security debt in docs/engineering/quality_gates.md or docs/reports when it is discovered but not fixed.

## Forbidden actions

- Do not commit real `.env` files or secrets.
- Do not put API keys, bearer tokens, DB passwords, or provider raw bodies in docs/reports.
- Do not log Authorization, Cookie, x-api-key, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, TAVILY_API_KEY, or full DB URIs.
- Do not add `verify=False`, unsafe deserialization, shell=True, or broad filesystem access without explicit review and tests.
- Do not skip redaction because a value is "only for tests".

## Required output

When used, include:

```text
Security/secret review:
- Secret sources touched:
- Redaction preserved:
- Raw body exposure added: no
- Gitleaks:
- Bandit:
- Security tests:
```

## Commands

```powershell
gitleaks detect --source . --redact
bandit -r src -x tests
python -m pytest tests/test_security.py -q
Select-String -Path 'src\**\*.py','tests\*.py','*.md','docs\**\*.md' -Pattern 'DEEPSEEK_API_KEY','OPENROUTER_API_KEY','TAVILY_API_KEY','Authorization','Bearer ','x-api-key','postgres://','postgresql://','sk-or-v1','sk-'
```

If Gitleaks or Bandit is missing, state that the tool was not run.

## Stop conditions

- A real secret appears in the diff or generated report.
- The change would expose raw provider request/response bodies.
- A security finding is new and cannot be fixed within scope.
- The change needs a risky file, subprocess, or network pattern without tests.

## A3_study_agent-specific rules

- Treat trace payloads as security-sensitive.
- OPENROUTER_API_KEY and OpenRouter DeepSeek residue must not return.
- DB URI scans must redact credentials.
- Security reports must contain patterns and file references, not secret values.
- Do not modify runtime security behavior in the first governance-only round.
