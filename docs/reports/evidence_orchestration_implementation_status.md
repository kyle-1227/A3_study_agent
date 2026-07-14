# Resource-aware evidence orchestration implementation status

Date: 2026-07-12 (updated 2026-07-14)
Branch baseline: `39df529`

## Outcome

The resource-aware evidence loop is implemented as a new, explicit Parent-Child candidate graph. It compiles resource/profile/subject evidence requirements before retrieval, executes requirement-bound local and Web tasks, judges coverage for every requirement, schedules at most two targeted supplement rounds, hydrates Parent-Child parents exactly once after terminal judging, derives readiness independently for each resource, and dispatches workers only for ready resources.

The served application still calls `get_compiled_graph()`. No rollout stage, active generation, registry pointer, provider configuration, or request-time fallback was changed. Production activation remains blocked until the existing Parent-Child data/index/gold prerequisites are satisfied and the new factorial evaluation gate passes on real gold cases.

## Candidate flow

1. `search_query_rewriter` supplies canonical subject branches.
2. The explicit resource branch enters `resource_evidence_planner` directly. The planner produces one strict requirement per requested resource, subject, and configured profile need and writes the runtime orchestration fingerprint.
3. `retrieval_round_router` validates task bindings, source policy, round budget, total budget, and no-repeat signatures.
4. `local_rag_search_batch` and `web_research_search_batch` execute in parallel. Empty task sets are explicit skips and make no provider call.
5. `retrieval_round_merge` is the sole owner of cumulative candidate, outcome, and child-only Parent-Child snapshots.
6. `requirement_evidence_judge` returns exactly one coverage row for every requirement. Code, not the LLM, derives resource readiness.
7. `evidence_repair_planner` schedules only blocked required gaps. `local_then_web_on_gap` must complete a successful-or-empty local attempt before it can schedule Web.
8. The loop stops on readiness, the two-round supplement limit, or one supplement round with no measurable progress.
9. `resource_parent_hydration` hydrates accepted resource child evidence once, after terminal judging; academic evidence uses the separate `academic_parent_hydration` node.
10. `resource_evidence_assignment` assigns accepted evidence per ready resource. Blocked resources never enter a worker and receive `blocked_insufficient_evidence`; no degraded artifact is produced.

## Strict runtime limits

| Limit | Value |
|---|---:|
| Supplement rounds | 2 |
| Search tasks per round | 6 |
| Total search tasks | 18 |
| Concurrent Web tasks | 4 |
| Results per task | 3 |
| Requirements per request | 12 |
| Evidence ledger entries | 54 |
| Evidence refs per requirement | 4 |
| Consecutive supplement rounds without progress | 1 |
| Required/supporting task priority | `high` / `medium` |
| High/medium/low adapter weights | `1.0` / `0.7` / `0.4` |

Progress is counted only when missing coverage decreases, complete coverage increases, or a new accepted evidence identity is bound. New text alone does not count. An exact requirement/source/subject/query signature cannot repeat across rounds.

## Configuration and contracts

- `config/rag/evidence_orchestration.yaml`: bounded execution and fail-fast policy.
- `config/rag/resource_evidence_profiles.yaml`: complete profiles for all seven canonical resources.
- `src/config/evidence_orchestration_config.py`: strict loaders; extra fields and incomplete inventories fail.
- `src/config/evidence_orchestration_contracts.py`: deterministic requirement/task/evidence/repair/assignment identities and business validators.
- `src/resource_contracts.py`: low-level canonical resource type ownership used by config, capability, planning, and generation.
- `config/prompts/resource_evidence_planner.xml` and `config/prompts/requirement_evidence_judge.xml`: versioned structured-output prompts.

No schema aliases, JSON repair, validation bypass, provider/model/base URL/API-key literal, candidate fallback, or silent resource omission is present in the new path.

## Trace and SSE

The strict trace family is `evidence_orchestration_trace_v1`:

- `evidence_orchestration.plan.accepted`
- `evidence_orchestration.round.started`
- `evidence_orchestration.source.completed|empty|failed`
- `evidence_orchestration.round.merged`
- `evidence_orchestration.coverage.judged`
- `evidence_orchestration.progress.evaluated`
- `evidence_orchestration.route.decided`
- `evidence_orchestration.resource.assigned`
- `evidence_orchestration.terminal|failed`

The trace contract rejects unknown fields before the sink is called. Query text, URL, evidence/provider bodies, headers, secrets, and full exception messages are not permitted. Existing app trace draining converts these events to `activity_event` SSE payloads with `kind=evidence_progress`; only counts, hashes, bounded timings, status, budget, and reason codes are exposed.

The joint candidate bundle fingerprint covers the pinned Parent-Child handoff/index policy, orchestration config, resource profiles, actual query/planner/judge/summarizer prompt contents, and the structured schemas. Prompt, schema, profile, policy, or index drift therefore changes the fingerprint.

## Evaluation and activation

`config/rag/evidence_benchmark.yaml` and `src/rag/parent_child/evidence_evaluation.py` implement a fail-closed paired 2x2 gate. Every case must contain all variants:

- P0: Parent-Child baseline, resource planning off, repair off.
- PG: resource-aware planning on, repair off.
- PR: planning off, bounded repair on.
- PGR: joint resource-aware planning and repair candidate.

The dataset is invalid if it lacks simple, multi-resource/multi-subject, or initially-sufficient cases. PGR must satisfy every gate:

- 100% bounded execution; zero forced-stop-as-sufficient; zero silent resource/subject omission; zero repeated query.
- Overall weighted coverage lift at least 8 percentage points; multi-case lift at least 10 points.
- Required gaps reduced by at least 25%; evidence precision loss at most 2 points; simple-case coverage regression at most 3 points.
- Premature-stop and over-search rates each at most 5%.
- Source-routing F1 at least 0.85; resource-subject recall at least 0.95; assignment precision at least 0.90.
- Claim support lift at least 5 points; ungrounded facts reduced by at least 20%.
- Average retrieval cost ratio at most 1.50; initially-sufficient cost ratio at most 1.10; p95 latency ratio at most 1.25.

`src/rag/evidence_observability.py` supplies versioned, content-free Shadow and health records. Candidate failures remain failures and never cause request-level rerouting to the primary graph.

## Verification snapshot

The candidate implementation has passed the following scoped gates on the current dirty worktree:

- Python compilation for `src`, `tests`, and `app.py`.
- Ruff check for all files changed by the evidence candidate.
- Mypy for the three new strict config/contract modules, the orchestration core, Shadow/health observability, and factorial evaluation modules.
- 38 focused evidence-orchestration tests, 227 related graph/RAG/resource/observability tests, 82 Academic/Web adapter tests, and 8 security tests.
- `git diff --check` with line-ending warnings only.

The final repository-wide test run completed with 1,847 passed, 5 skipped, and zero failures. The remaining warnings are the existing `pkg_resources` deprecation and `aiosqlite` worker-thread/event-loop shutdown warnings. Repository-wide Ruff remains blocked by 53 pre-existing findings outside this candidate. `academic.py` has 29 existing mypy findings outside the new direct Web adapter. Semgrep, import-linter, Pyright, ty, Gitleaks, Bandit, and Vulture are not installed and are reported as unavailable rather than passing.

Ruff format check passes 21 of the 22 scoped Python files. The shared `academic.py` file contains mixed-format concurrent/legacy regions; formatting the whole file would rewrite unrelated user work, so this change leaves those regions untouched. Ruff semantic checks pass for the file.

## Current production blockers

1. No valid production `config/rag/index.yaml` and activated immutable Parent-Child generation are available.
2. The current corpus/gold prerequisites in `rag_parent_child_implementation_status.md` remain unresolved: no human/historical gold set and insufficient independent sources for primary subjects.
3. No real-secret live run of the new planner, direct Web executor, requirement judge, and resource workers has been performed.
4. No P0/PG/PR/PGR gold result bundle has been collected, so the new activation decision cannot yet be eligible.
5. The app has not been wired to select `get_compiled_resource_evidence_parent_child_graph(...)`; rollout remains an explicit future control-plane action.

These blockers must not be bypassed with synthetic gold, a degraded candidate, a hidden fallback, or a request-time primary retry.

## 2026-07-14 superseded router cleanup

The 2026-07-12 initial candidate briefly used `rag_generation_router` and
`rag_generation_route` to select between a joint hydration node's academic and
resource behavior. Commit `5d6f51a` replaced that topology with a direct
`resource_evidence_planner` edge and distinct academic/resource hydration
nodes. The planner already writes the same
`evidence_orchestration_fingerprint` from its bound runtime.

After static, dynamic import, graph, registry, prompt, config, and test scans
confirmed no current consumer, the obsolete router factory/export, transient
state field, TypedDict field, and node metadata were removed. The graph test
now locks the absence of the old factory, state, and metadata while separately
proving the active planner still emits the runtime fingerprint. Existing
checkpoints are not rewritten by this code cleanup; an old pending candidate
node remains subject to the explicit checkpoint migration/unknown-node block
gate and is not claimed as migrated.

Focused verification: 51 evidence graph/trace/state/manifest tests passed;
compileall, touched Ruff check/format, three-file scoped mypy, diff check, and
active old-symbol scans passed. The final backend suite passed with
`2274 passed, 6 skipped`; frontend Vitest (69 tests), typecheck, ESLint, and
production build also passed. This cleanup does not activate the candidate or
change any production blocker above.
