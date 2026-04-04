# REQ-07 Completion Report: Adversarial Plan SubGraph

## Summary

Built the adversarial planning SubGraph and wired it into the main graph's planner branch.
The planner flow is now: `search_policy → gather_intel → plan_adversarial → END`.

- **Phase2a (gather_intel)**: Parallel fan-out — emotional LLM analysis + RAG/web search in parallel via `asyncio.gather()`
- **Phase2b (SubGraph)**: `drafter → [reviewer_academic ∥ reviewer_emotional] → consensus_check → (loop/output)`
- **Safety valve**: `max_rounds` (default 3) forces output even if reviewers never agree
- **ADR-001 compliant**: SubGraph with independent `PlanAdversarialState`, mounted via wrapper node
- **ADR-002 compliant**: Reviewers use `llm.with_structured_output(ReviewVerdict).ainvoke()`

## Files Changed

| File | Change |
|------|--------|
| `src/graph/plan_adversarial.py` | **NEW** — `PlanAdversarialState`, `ReviewVerdict`, 6 async nodes, `build_adversarial_subgraph()` |
| `src/graph/planner.py` | Added `gather_intel`, `plan_adversarial_node` (wrapper), `_gather_emotional_intel`, `_gather_resource_intel` |
| `src/graph/builder.py` | Replaced `generate_plan` with `gather_intel → plan_adversarial` in planner flow |
| `src/graph/state.py` | Added `emotional_intel`, `resource_intel`, `intel_summary` fields to TutorState |
| `config/settings.yaml` | Added `planner.reviewer_temperature` (0.0) and `planner.adversarial_max_rounds` (3) |
| `config/prompts/plan_drafter_system.xml` | **NEW** — Drafter system prompt |
| `config/prompts/plan_drafter.xml` | **NEW** — Drafter user prompt (first draft) |
| `config/prompts/plan_rewrite.xml` | **NEW** — Drafter user prompt (revision) |
| `config/prompts/plan_reviewer_academic_system.xml` | **NEW** — Academic reviewer system prompt |
| `config/prompts/plan_reviewer_emotional_system.xml` | **NEW** — Emotional reviewer system prompt |
| `config/prompts/gather_emotional_intel.xml` | **NEW** — Emotional intel gathering prompt |
| `tests/test_plan_adversarial.py` | **NEW** — 21 tests: model validation, all nodes, SubGraph integration (consensus/reject/max_rounds) |
| `tests/test_gather_intel.py` | **NEW** — 5 tests: intel fields, emotional LLM, resource RAG+web, error degradation |
| `tests/test_builder.py` | Updated expected node set (generate_plan → gather_intel + plan_adversarial) |
| `tests/conftest.py` | Added new TutorState fields to `sample_state` fixture |

## Design Details

### SubGraph Architecture
```
drafter → [reviewer_academic ∥ reviewer_emotional] → consensus_check
                                                        ├── consensus=True  → output → END
                                                        └── consensus=False → rewrite → drafter (loop)
```

### Key Decisions
- **Reviewer fallback**: If structured output fails, default to `approve` (safe fallback, avoids blocking the pipeline)
- **Consensus check at max_rounds**: Forces `consensus=True` regardless of verdicts, logs a warning
- **gather_intel parallelism**: `asyncio.gather()` runs emotional LLM + (RAG + web search) concurrently
- **Error degradation**: Both emotional and resource intel have fallback strings on failure

## Verification

```bash
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short
```

Result: **366 passed, 1 skipped, 0 failed** (26 new tests: 21 adversarial + 5 gather_intel)

## Suggested Commit	

```bash
git add src/graph/plan_adversarial.py src/graph/planner.py src/graph/builder.py src/graph/state.py \
        config/settings.yaml \
        config/prompts/plan_drafter_system.xml config/prompts/plan_drafter.xml 			  			  				config/prompts/plan_rewrite.xml \
        config/prompts/plan_reviewer_academic_system.xml config/prompts/plan_reviewer_emotional_system.xml \
        config/prompts/gather_emotional_intel.xml \
        tests/test_plan_adversarial.py tests/test_gather_intel.py tests/test_builder.py tests/conftest.py
git commit -m "feat(graph): 对抗式计划 SubGraph + 情报收集节点 (REQ-07)

- 新增 plan_adversarial.py: drafter→[reviewer×2]→consensus 循环子图
- planner.py: gather_intel 并行收集情绪+资源情报
- builder.py: 规划分支改为 search_policy→gather_intel→plan_adversarial
- 6 个新 prompt 模板 + settings.yaml 审查配置
- 26 项新测试全部通过（共 366 passed）"
```
