"""Candidate-graph tests for bounded evidence planning, repair, and blocking."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage
import pytest

from src.config.evidence_orchestration_config import (
    load_evidence_orchestration_config,
    load_resource_evidence_profiles,
)
from src.config.evidence_orchestration_contracts import (
    DuplicateRetrievalSignatureError,
    EvidenceRequirementDraft,
    EvidenceRequirementDraftBatch,
    RequirementCoverage,
    RequirementCoverageBatch,
    ResourceReadiness,
    RetrievalTask,
    build_retrieval_task,
    compile_evidence_requirement_batch,
)
from src.graph import academic
from src.graph import evidence_orchestration as orchestration
from src.graph.builder import (
    build_graph,
    build_resource_evidence_parent_child_graph,
    route_after_candidate_query_rewrite,
)
from src.graph.parent_child_nodes import ParentChildGraphRuntime
from src.graph.resource_generation import (
    dispatch_resource_workers,
    resource_bundle_output,
    resource_orchestrator,
)
from src.graph.web_research import WebResearchTask

ROOT = Path(__file__).resolve().parents[1]


class _NoopRetriever:
    def retrieve_children_multi(self, request):
        raise AssertionError("graph construction test must not retrieve")

    def hydrate_kept_multi(self, result, kept_child_ids):
        raise AssertionError("empty hydration test must not call the retriever")


def _runtime() -> orchestration.EvidenceOrchestrationRuntime:
    parent_child = ParentChildGraphRuntime(
        generation_id="generation-evidence-test",
        available_subjects=("math",),
        retriever=_NoopRetriever(),
        retrieval_fingerprint="f" * 64,
        cross_branch_rrf_k=20,
        parent_top_k=3,
        preview_max_chars=600,
    )
    return orchestration.EvidenceOrchestrationRuntime(
        parent_child=parent_child,
        policy=load_evidence_orchestration_config(
            ROOT / "config" / "rag" / "evidence_orchestration.yaml"
        ),
        profiles=load_resource_evidence_profiles(
            ROOT / "config" / "rag" / "resource_evidence_profiles.yaml"
        ),
        web_timeout_seconds=10.0,
    )


def _quiz_draft_batch(runtime: orchestration.EvidenceOrchestrationRuntime):
    profile = runtime.profiles.profile_for("quiz")
    return EvidenceRequirementDraftBatch(
        schema_version="evidence_requirement_draft_batch_v1",
        requirements=tuple(
            EvidenceRequirementDraft(
                resource_type="quiz",
                subject="math",
                profile_need_id=need.need_id,
                evidence_kind=need.evidence_kind,
                scope=need.scope,
                criticality=need.criticality,
                source_policy=need.source_policy,
                acceptance_criteria=need.acceptance_criteria,
                query_intent=f"math {need.evidence_kind} initial evidence",
            )
            for need in profile.needs
        ),
    )


def _planner_state():
    return {
        "messages": [HumanMessage(content="请生成函数复习测验")],
        "request_id": "request-evidence-1",
        "session_id": "session-evidence-1",
        "thread_id": "thread-evidence-1",
        "response_mode": "resource",
        "requested_resource_type": "quiz",
        "requested_resource_types": ["quiz"],
        "learning_goal": "review mathematical functions",
        "retrieval_plan": [
            {
                "subject": "math",
                "role": "core_concept",
                "local_retrieval_query": "math functions course concepts",
                "web_research_seed_query": "math functions official tutorial",
                "purpose": "support the requested quiz",
                "priority": 1.0,
                "_parent_child_priority_explicit": True,
            }
        ],
    }


def _missing_coverage(requirements, *, round_index: int, suffix: str):
    rows = []
    for requirement in requirements:
        if requirement["source_policy"] == "local_only":
            local_query = f"math assessable facts {suffix}"
            web_query = ""
        elif requirement["source_policy"] == "web_only":
            local_query = ""
            web_query = f"math official evidence {suffix}"
        else:
            local_query = f"math misconception course evidence {suffix}"
            web_query = f"math misconception official evidence {suffix}"
        rows.append(
            RequirementCoverage(
                requirement_id=requirement["requirement_id"],
                resource_type=requirement["resource_type"],
                subject=requirement["subject"],
                round_index=round_index,
                coverage_state="missing",
                evidence_ids=(),
                confidence=0.0,
                reason="No candidate satisfies the configured acceptance criteria.",
                suggested_local_query=local_query,
                suggested_web_query=web_query,
            )
        )
    return RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=round_index,
        coverages=tuple(rows),
    )


def test_joint_candidate_graph_is_explicit_and_legacy_served_graph_is_unchanged():
    runtime = _runtime()
    legacy = build_graph()
    candidate = build_resource_evidence_parent_child_graph(runtime)

    assert "resource_evidence_planner" not in legacy.nodes
    assert {
        "rag_generation_router",
        "resource_evidence_planner",
        "retrieval_round_router",
        "local_rag_search_batch",
        "web_research_search_batch",
        "retrieval_round_merge",
        "requirement_evidence_judge",
        "evidence_repair_planner",
        "resource_evidence_assignment",
    }.issubset(candidate.nodes)
    assert (
        ("local_rag_search_batch", "web_research_search_batch"),
        "retrieval_round_merge",
    ) in candidate.waiting_edges
    assert candidate.compile() is not None


def test_candidate_query_route_requires_explicit_canonical_resources():
    assert (
        route_after_candidate_query_rewrite(
            {
                "response_mode": "resource",
                "requested_resource_types": ["quiz", "mindmap"],
            }
        )
        == "resource_evidence"
    )
    assert (
        route_after_candidate_query_rewrite(
            {"response_mode": "qa", "requested_resource_types": []}
        )
        == "academic"
    )
    with pytest.raises(ValueError, match="requires canonical resource types"):
        route_after_candidate_query_rewrite({"response_mode": "resource"})


def test_planner_compiles_profile_slots_and_first_repair_round(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(_planner_state())
    )

    assert len(planned["evidence_requirements"]) == 2
    initial_tasks = [
        RetrievalTask.model_validate(item) for item in planned["evidence_current_tasks"]
    ]
    assert len(initial_tasks) == 3
    assert {task.source_type for task in initial_tasks} == {"local_rag", "web"}
    assert planned["evidence_current_round"] == 0
    assert "degraded_generation" not in planned

    missing = _missing_coverage(
        planned["evidence_requirements"],
        round_index=0,
        suffix="repair one",
    )

    async def fake_judge(**_kwargs):
        return SimpleNamespace(parsed=missing)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_judge)
    outcomes = [
        {
            "round_index": 0,
            "task_id": task.task_id,
            "requirement_id": task.requirement_id,
            "source_type": task.source_type,
            "status": "empty",
            "candidate_count": 0,
        }
        for task in initial_tasks
    ]
    judged = asyncio.run(
        orchestration.make_requirement_evidence_judge_node(runtime)(
            {
                **_planner_state(),
                **planned,
                "evidence_candidate_records": [],
                "evidence_source_outcomes": outcomes,
            }
        )
    )
    assert judged["evidence_orchestration_route"] == "repair"
    assert judged["evidence_terminal_status"] == ""

    repaired = orchestration.make_evidence_repair_planner_node(runtime)(
        {**_planner_state(), **planned, **judged}
    )
    repair_tasks = [
        RetrievalTask.model_validate(item)
        for item in repaired["evidence_current_tasks"]
    ]
    assert repaired["evidence_current_round"] == 1
    assert len(repair_tasks) == 1
    assert repair_tasks[0].source_type == "local_rag"
    assert repair_tasks[0].requirement_id == next(
        item["requirement_id"]
        for item in planned["evidence_requirements"]
        if item["criticality"] == "required"
    )


def test_repair_can_schedule_third_search_round_and_reject_exact_repeat(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(_planner_state())
    )
    requirements = planned["evidence_requirements"]
    first_missing = _missing_coverage(
        requirements,
        round_index=0,
        suffix="repair one",
    )
    readiness = [
        {
            "resource_type": "quiz",
            "readiness_state": "blocked_insufficient_evidence",
            "required_requirement_ids": [requirements[0]["requirement_id"]],
            "complete_requirement_ids": [],
            "blocked_requirement_ids": [requirements[0]["requirement_id"]],
            "evidence_ids": [],
            "reason_code": "required_evidence_incomplete",
        }
    ]
    round_one_state = {
        **_planner_state(),
        **planned,
        "evidence_coverage": first_missing.model_dump(mode="json"),
        "resource_evidence_readiness": readiness,
    }
    round_one = orchestration.make_evidence_repair_planner_node(runtime)(
        round_one_state
    )
    second_missing = _missing_coverage(
        requirements,
        round_index=1,
        suffix="repair two",
    )
    round_two = orchestration.make_evidence_repair_planner_node(runtime)(
        {
            **round_one_state,
            **round_one,
            "evidence_coverage": second_missing.model_dump(mode="json"),
            "resource_evidence_readiness": readiness,
        }
    )

    assert round_two["evidence_current_round"] == 2
    assert all(
        RetrievalTask.model_validate(item).round_index == 2
        for item in round_two["evidence_current_tasks"]
    )

    repeated = _missing_coverage(
        requirements,
        round_index=1,
        suffix="repair one",
    )
    with pytest.raises(
        DuplicateRetrievalSignatureError,
        match="duplicate_retrieval_signature",
    ):
        orchestration.make_evidence_repair_planner_node(runtime)(
            {
                **round_one_state,
                **round_one,
                "evidence_coverage": repeated.model_dump(mode="json"),
                "resource_evidence_readiness": readiness,
            }
        )


def test_local_then_web_policy_never_skips_the_required_local_attempt():
    runtime = _runtime()
    need = next(
        item
        for item in runtime.profiles.profile_for("review_doc").needs
        if item.source_policy == "local_then_web_on_gap"
    )
    requirement = compile_evidence_requirement_batch(
        EvidenceRequirementDraftBatch(
            schema_version="evidence_requirement_draft_batch_v1",
            requirements=(
                EvidenceRequirementDraft(
                    resource_type="review_doc",
                    subject="math",
                    profile_need_id=need.need_id,
                    evidence_kind=need.evidence_kind,
                    scope=need.scope,
                    criticality=need.criticality,
                    source_policy=need.source_policy,
                    acceptance_criteria=need.acceptance_criteria,
                    query_intent="math authoritative concepts",
                ),
            ),
        )
    )[0]
    gap = SimpleNamespace(
        suggested_local_query="math authoritative local concepts",
        suggested_web_query="math authoritative official concepts",
    )

    assert orchestration._repair_source_queries(
        requirement,
        gap,
        local_attempted=False,
    ) == (("local_rag", "math authoritative local concepts"),)
    assert orchestration._repair_source_queries(
        requirement,
        gap,
        local_attempted=True,
    ) == (("web", "math authoritative official concepts"),)


def test_repair_planner_skips_a_prior_source_signature_and_keeps_unseen_source():
    runtime = _runtime()
    profile = runtime.profiles.profile_for("code_practice")
    need = next(item for item in profile.needs if item.need_id == "executable_patterns")
    requirement = compile_evidence_requirement_batch(
        EvidenceRequirementDraftBatch(
            schema_version="evidence_requirement_draft_batch_v1",
            requirements=(
                EvidenceRequirementDraft(
                    resource_type="code_practice",
                    subject="math",
                    profile_need_id=need.need_id,
                    evidence_kind=need.evidence_kind,
                    scope=need.scope,
                    criticality=need.criticality,
                    source_policy=need.source_policy,
                    acceptance_criteria=need.acceptance_criteria,
                    query_intent="math executable pattern evidence",
                ),
            ),
        )
    )[0]
    prior_local_task = build_retrieval_task(
        requirement=requirement,
        source_type="local_rag",
        query="math executable pattern evidence",
        purpose=requirement.acceptance_criteria,
        priority=runtime.policy.required_task_priority,
        round_index=0,
        result_limit=runtime.policy.max_results_per_task,
    )
    coverage = RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=0,
        coverages=(
            RequirementCoverage(
                requirement_id=requirement.requirement_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                round_index=0,
                coverage_state="missing",
                evidence_ids=(),
                confidence=0.2,
                reason="The Web half of the required evidence is still missing.",
                suggested_local_query="math executable pattern evidence",
                suggested_web_query="math official executable pattern evidence",
            ),
        ),
    )
    readiness = ResourceReadiness(
        resource_type=requirement.resource_type,
        readiness_state="blocked_insufficient_evidence",
        required_requirement_ids=(requirement.requirement_id,),
        complete_requirement_ids=(),
        blocked_requirement_ids=(requirement.requirement_id,),
        evidence_ids=(),
        reason_code="required_evidence_missing",
    )

    result = orchestration.make_evidence_repair_planner_node(runtime)(
        {
            "evidence_current_round": 0,
            "evidence_requirements": [requirement.model_dump(mode="json")],
            "evidence_coverage": coverage.model_dump(mode="json"),
            "resource_evidence_readiness": [readiness.model_dump(mode="json")],
            "evidence_all_tasks": [prior_local_task.model_dump(mode="json")],
            "evidence_source_outcomes": [],
            "evidence_repair_plans": [],
        }
    )

    repair_tasks = [
        RetrievalTask.model_validate(item) for item in result["evidence_current_tasks"]
    ]
    assert [(task.source_type, task.query) for task in repair_tasks] == [
        ("web", "math official executable pattern evidence")
    ]
    assert len(result["evidence_all_tasks"]) == 2


def test_direct_web_execution_distinguishes_empty_from_provider_failure(
    monkeypatch,
):
    task = WebResearchTask(
        task_id="task-web-1",
        subject="math",
        role="requirement_math",
        purpose="Find exact evidence.",
        search_query="math official evidence",
        reason="The requirement needs current evidence.",
        priority=1.0,
    )

    async def empty_executor(**_kwargs):
        return [], [{"status": "failed", "error_type": None}]

    monkeypatch.setattr(academic, "_execute_web_research_tasks", empty_executor)
    empty = asyncio.run(
        academic.execute_validated_web_research_tasks(
            state={},
            tasks=[task],
            original_user_query="math evidence",
            timeout=5.0,
            max_results_per_task=3,
            max_concurrent_tasks=2,
        )
    )
    assert empty["status"] == "empty"
    assert empty["candidates"] == []

    async def failed_executor(**_kwargs):
        return [], [{"status": "failed", "error_type": "ProviderTimeout"}]

    monkeypatch.setattr(academic, "_execute_web_research_tasks", failed_executor)
    with pytest.raises(
        academic.ValidatedWebResearchExecutionError,
        match="web_source_execution_failed",
    ):
        asyncio.run(
            academic.execute_validated_web_research_tasks(
                state={},
                tasks=[task],
                original_user_query="math evidence",
                timeout=5.0,
                max_results_per_task=3,
                max_concurrent_tasks=2,
            )
        )


def test_terminal_hydration_is_one_shot_even_when_no_parent_is_selected(
    monkeypatch,
):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(_planner_state())
    )
    missing = _missing_coverage(
        planned["evidence_requirements"],
        round_index=0,
        suffix="terminal",
    )
    hydrate = orchestration.make_terminal_parent_hydration_node(runtime)
    state = {
        **_planner_state(),
        **planned,
        "evidence_coverage": missing.model_dump(mode="json"),
        "evidence_ledger": [],
        "evidence_candidate_records": [],
        "evidence_parent_child_rounds": [],
    }
    hydrated = asyncio.run(hydrate(state))

    assert hydrated["evidence_hydration_count"] == 1
    assert hydrated["parent_child_hydration"]["parent_count"] == 0
    with pytest.raises(
        orchestration.EvidenceOrchestrationRuntimeError,
        match="parent_hydration_repeated",
    ):
        asyncio.run(hydrate({**state, "evidence_hydration_count": 1}))


def test_all_blocked_resources_skip_workers_and_return_explicit_bundle():
    state = {
        "request_id": "request-blocked",
        "session_id": "session-blocked",
        "thread_id": "thread-blocked",
        "evidence_requested_resource_types": ["quiz"],
        "requested_resource_type": "",
        "requested_resource_types": [],
        "blocked_resource_types": ["quiz"],
        "resource_evidence_readiness": [
            {
                "resource_type": "quiz",
                "readiness_state": "blocked_insufficient_evidence",
                "required_requirement_ids": ["requirement_1"],
                "complete_requirement_ids": [],
                "blocked_requirement_ids": ["requirement_1"],
                "evidence_ids": [],
                "reason_code": "required_evidence_incomplete",
            }
        ],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [],
    }
    orchestrated = asyncio.run(resource_orchestrator(state))
    sends = dispatch_resource_workers({**state, **orchestrated})
    bundled = asyncio.run(resource_bundle_output({**state, **orchestrated}))

    assert orchestrated["resource_generation_plan"]["tasks"] == []
    assert len(sends) == 1
    assert sends[0].node == "resource_bundle_output"
    assert bundled["resource_generation_status"] == ("blocked_insufficient_evidence")
    assert bundled["resource_bundle_artifact"]["blocked_count"] == 1
    assert "未生成降级内容" in bundled["messages"][0].content
