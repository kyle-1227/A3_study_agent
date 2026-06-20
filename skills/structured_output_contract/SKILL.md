---
name: structured_output_contract
description: Repo-scoped A3 Study Agent skill for structured LLM output and Pydantic schema integrity. Use for schemas, validators, structured_output runtime, graph outputs, profile extraction, JSON parsing, retries, and business validation.
---

# structured_output_contract

## Purpose

Keep structured LLM output strict, auditable, and fail-fast. Prevent schema drift from being hidden by alias normalization, JSON repair, default objects, or swallowed validation errors.

## When to use

Use whenever a change touches Pydantic models, structured LLM output modes, parsing, schema fields, business validators, graph node output contracts, profile extraction output, JSON serialization, retry behavior, or tests that assert structured-output behavior.

## Required inputs

- The target schema or model.
- The producer prompt or LLM call site.
- The consumer that relies on parsed output.
- The business validator, if any.
- Related tests and sample payloads.
- Current spec_first_change output.

## Procedure

1. Identify the exact contract: required fields, forbidden extras, field types, enums, bounds, and cross-field business rules.
2. Keep Pydantic validation strict. Prefer `extra="forbid"` where the schema is externally produced and drift matters.
3. Keep business validation separate and explicit. A Pydantic pass is not enough when the domain has invariants.
4. Make invalid output fail loudly with a typed error or explicit diagnostic.
5. Preserve raw-enough diagnostics for debugging while redacting secrets and avoiding full raw body leakage.
6. Update tests with negative cases for schema drift and business validation failures.
7. Run related structured-output tests and static_quality_gate.
8. If existing schema drift, alias mapping, or fallback parsing is discovered, document it in docs/reports/fallback_paths.md.

## Forbidden actions

- Do not bypass Pydantic with `model_construct`, unchecked casts, or direct dataclass/dict conversion for model output.
- Do not automatically rename, normalize, alias, or fuzzy-match keys to make invalid payloads pass.
- Do not set `populate_by_name=True`, `alias_generator`, `validation_alias`, or broad `AliasChoices` without explicit user approval and tests proving it is not masking drift.
- Do not catch `ValidationError` or business validation errors and return a default success object.
- Do not add JSON repair fallback that changes the schema contract.
- Do not weaken tests by changing assertions to accept missing or renamed fields.
- Do not modify src/llm/structured_output.py unless explicitly requested.

## Required output

When used, include:

```text
Structured output contract:
- Schema changed: yes/no
- Required fields preserved: yes/no
- Extra fields policy: forbid/allow/existing
- Business validator preserved or added: yes/no/not applicable
- Alias normalization added: no
- Validation errors swallowed: no
- Negative tests added or updated: yes/no
```

## Commands

```powershell
Select-String -Path 'src\**\*.py','tests\*.py' -Pattern 'BaseModel','ConfigDict','model_validate','model_validate_json','model_construct','ValidationError','business_validator','validation_alias','alias_generator','populate_by_name'
python -m pytest tests/test_deepseek_structured_output.py tests/test_structured_retry.py tests/test_json_output.py tests/test_profile.py tests/test_profile_manager.py -q
semgrep --config semgrep_rules/a3_no_fallback_no_hardcode.yml src tests app.py
```

Run only the tests relevant to the touched schema when the full set is not proportional, and state the reason.

## Stop conditions

- The requested behavior requires accepting multiple schema names for the same field without a documented migration plan.
- A business validation failure would be converted to success.
- The prompt, schema, and consumer disagree on required fields.
- Raw provider payloads would expose secrets, auth headers, or full trace bodies.
- Tests cannot demonstrate both valid and invalid structured-output behavior.

## A3_study_agent-specific rules

- src/llm/structured_output.py is a high-risk contract file.
- Graph nodes and profile extraction must not compensate for invalid structured output with local defaults.
- DeepSeek structured-output protocol errors must remain distinguishable from schema validation and business validation errors.
- Do not reintroduce OpenRouter DeepSeek structured-output routing.
- Keep structured output provider-neutral unless an explicitly named provider protocol requires otherwise.
