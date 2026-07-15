# Production cleanup evidence (2026-07-15)

This report records the evidence used for the final deadline cleanup. It is a
deletion record, not a generated test artifact.

## Deleted files

- `frontend-dev.log`: tracked, empty, and had no repository references.
- `frontend-dev.err.log`: tracked, empty, and had no repository references.
- `config/prompts/gather_emotional_intel.xml`: had no direct reference, no
  configuration reference, and no matching dynamic prompt-name producer. The
  active emotional graph loads `emotional_system`; study-plan prompt content is
  constructed in its runtime module.
- `frontend-dev.pid`: tracked a stale local process identifier and had no
  repository references.
- `tmp_debug_gold.py`: root-level diagnostic with no entry point or references;
  it also embedded one developer workstation path.
- `scripts/debug_structured_output_provider.py`: unreferenced local probe that
  loaded `.env` implicitly and printed raw provider output/error bodies.
- `route_by_intent`: obsolete supervisor router with no graph, registry, route,
  configuration, or dynamic reference. It also silently mapped a missing intent
  to academic. The served graph uses the strict `route_after_supervisor`.
- `study_plan_profile_gate`: backward-compatibility wrapper with no graph,
  registry, resource-runner, route, configuration, or dynamic reference. The
  only registered identity is `study_plan_profile_gate_main`; retained behavior
  tests now exercise that production node directly.

## Retained files

- The legacy flat RAG implementation and artifacts remain available as an
  explicit deployment rollback. The Parent-Child candidate is not proven safe
  for activation while retrieval quality is below the sealed baseline.
- Existing engineering and rollout reports remain because they provide audit
  evidence rather than temporary runtime output.
- Graph helpers identified only by static dead-code tools are retained unless a
  runtime, route, configuration, and dynamic-import audit proves deletion safe.

## Repository scan

- No tracked `prompt.md` file existed at cleanup time.
- The tracked TypeScript build cache is handled by the service integration
  change together with its ignore rule; it is not duplicated in this cleanup.
