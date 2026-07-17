"""Candidate-graph tests for bounded evidence planning, repair, and blocking."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import inspect
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
    EvidenceLedgerEntry,
    EvidenceRequirementDraft,
    EvidenceRequirementDraftBatch,
    RequirementCoverage,
    RequirementCoverageBatch,
    RESOURCE_EVIDENCE_CONTRACT_VERSION,
    ResourceReadiness,
    RetrievalTask,
    build_retrieval_task,
    compile_evidence_requirement_batch,
    compile_requirement_coverage_batch,
    make_evidence_id,
)
from src.graph import academic
from src.graph import evidence_orchestration as orchestration
from src.graph.evidence import EvidenceCandidate
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
from src.graph.state import LearningState, initial_request_reset_transient_state
from src.learning_guidance.runtime import LearningGuidanceRuntime
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink
from src.observability.node_registry import get_node_runtime_metadata

ROOT = Path(__file__).resolve().parents[1]


class _NoopRetriever:
    def retrieve_children_multi(self, request):
        raise AssertionError("graph construction test must not retrieve")

    def hydrate_kept_multi(self, result, kept_child_ids):
        raise AssertionError("empty hydration test must not call the retriever")


async def _unexpected_guidance_dependency(*_args, **_kwargs):
    raise AssertionError("this evidence-orchestration test must not call guidance")


def _knowledge_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "test-v1",
            "subjects": [
                {
                    "subject_id": "math",
                    "title": "Mathematics",
                    "topics": [
                        {
                            "topic_id": "functions",
                            "title": "Functions",
                            "difficulty": 0.5,
                            "estimated_hours": 2.0,
                            "prerequisite_topic_ids": [],
                            "knowledge_points": ["Function definitions"],
                            "resources": [
                                {
                                    "resource_id": "functions_quiz",
                                    "resource_type": "quiz",
                                    "title": "Functions quiz",
                                }
                            ],
                        },
                        {
                            "topic_id": "limits",
                            "title": "Limits",
                            "difficulty": 0.6,
                            "estimated_hours": 2.0,
                            "prerequisite_topic_ids": ["functions"],
                            "knowledge_points": ["Limit definitions"],
                            "resources": [
                                {
                                    "resource_id": "limits_quiz",
                                    "resource_type": "quiz",
                                    "title": "Limits quiz",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    )


def _learning_guidance_runtime() -> LearningGuidanceRuntime:
    return LearningGuidanceRuntime(
        runtime_fingerprint="3" * 64,
        knowledge_graph=_knowledge_graph(),
        provider_projection_max_steps=50,
        provider_projection_max_chars=65_536,
        load_profile=_unexpected_guidance_dependency,
        load_history=_unexpected_guidance_dependency,
        plan_learning_path=_unexpected_guidance_dependency,
        recommend_resources=_unexpected_guidance_dependency,
    )


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
        learning_guidance=_learning_guidance_runtime(),
        web_timeout_seconds=10.0,
    )


def _multisubject_runtime() -> orchestration.EvidenceOrchestrationRuntime:
    base = _runtime()
    graph_payload = _knowledge_graph().model_dump(mode="python")
    subjects = list(graph_payload["subjects"])
    graph_payload["subjects"] = subjects
    subjects.append(
        {
            "subject_id": "computer",
            "title": "Computer Science",
            "topics": [
                {
                    "topic_id": "computer.systems",
                    "title": "Computer Systems",
                    "difficulty": 0.5,
                    "estimated_hours": 2.0,
                    "prerequisite_topic_ids": [],
                    "knowledge_points": ["Computer system foundations"],
                    "resources": [
                        {
                            "resource_id": "computer_systems_quiz",
                            "resource_type": "quiz",
                            "title": "Computer systems quiz",
                        }
                    ],
                }
            ],
        }
    )
    knowledge_graph = KnowledgeGraphV1.model_validate(graph_payload)
    return replace(
        base,
        parent_child=replace(
            base.parent_child,
            available_subjects=("computer", "math"),
        ),
        learning_guidance=replace(
            base.learning_guidance,
            knowledge_graph=knowledge_graph,
        ),
    )


def _quiz_draft_batch(runtime: orchestration.EvidenceOrchestrationRuntime):
    profile = runtime.profiles.profile_for("quiz")
    return EvidenceRequirementDraftBatch(
        schema_version="evidence_requirement_draft_batch_v1",
        requirements=[
            EvidenceRequirementDraft(
                resource_type="quiz",
                subject="math",
                topic_id="functions",
                profile_need_id=need.need_id,
                evidence_kind=need.evidence_kind,
                scope=need.scope,
                criticality=need.criticality,
                source_policy=need.source_policy,
                acceptance_criteria=need.acceptance_criteria,
                query_intent=f"math {need.evidence_kind} initial evidence",
            )
            for need in profile.needs
        ],
    )


def _multisubject_quiz_draft_batch(
    runtime: orchestration.EvidenceOrchestrationRuntime,
    *,
    include_computer: bool = True,
    computer_topic_id: str = "computer.systems",
) -> EvidenceRequirementDraftBatch:
    profile = runtime.profiles.profile_for("quiz")
    subject_topics = [("math", "functions")]
    if include_computer:
        subject_topics.append(("computer", computer_topic_id))
    return EvidenceRequirementDraftBatch(
        schema_version="evidence_requirement_draft_batch_v1",
        requirements=[
            EvidenceRequirementDraft(
                resource_type="quiz",
                subject=subject,
                topic_id=topic_id,
                profile_need_id=need.need_id,
                evidence_kind=need.evidence_kind,
                scope=need.scope,
                criticality=need.criticality,
                source_policy=need.source_policy,
                acceptance_criteria=need.acceptance_criteria,
                query_intent=f"{subject} {topic_id} {need.evidence_kind} evidence",
            )
            for subject, topic_id in subject_topics
            for need in profile.needs
        ],
    )


def _planner_state():
    guidance = _learning_guidance_runtime()
    return {
        "messages": [HumanMessage(content="请生成函数复习测验")],
        "request_id": "request-evidence-1",
        "session_id": "session-evidence-1",
        "thread_id": "thread-evidence-1",
        "user_id": "",
        "subject": "math",
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
        "learner_path_planner_output": {
            "schema_version": "learner_path_planner_output_v1",
            "runtime_fingerprint": guidance.runtime_fingerprint,
            "provider_projection_policy_fingerprint": (
                guidance.provider_projection_policy_fingerprint
            ),
            "provider_projection_max_steps": (guidance.provider_projection_max_steps),
            "provider_projection_max_chars": (guidance.provider_projection_max_chars),
            "request_id": "request-evidence-1",
            "status": "unavailable",
            "unavailable_reason": "missing_user_id",
            "user_id": None,
            "subject": "math",
            "plan": None,
        },
        "learner_path_provider_projection": {
            "schema_version": "learner_path_provider_projection_v1",
            "status": "unavailable",
            "unavailable_reason": "missing_user_id",
            "subject": "math",
            "summary": None,
            "steps": [],
        },
    }


def _multisubject_planner_state() -> dict:
    guidance = _learning_guidance_runtime()
    state = {
        **_planner_state(),
        "user_id": "learner-math-1",
        "learner_path_planner_output": {
            "schema_version": "learner_path_planner_output_v1",
            "runtime_fingerprint": guidance.runtime_fingerprint,
            "provider_projection_policy_fingerprint": (
                guidance.provider_projection_policy_fingerprint
            ),
            "provider_projection_max_steps": (guidance.provider_projection_max_steps),
            "provider_projection_max_chars": (guidance.provider_projection_max_chars),
            "request_id": "request-evidence-1",
            "status": "unavailable",
            "unavailable_reason": "unsupported_subject_scope",
            "user_id": "learner-math-1",
            "subject": "math",
            "plan": None,
        },
        "learner_path_provider_projection": {
            "schema_version": "learner_path_provider_projection_v1",
            "status": "unavailable",
            "unavailable_reason": "unsupported_subject_scope",
            "subject": "math",
            "summary": None,
            "steps": [],
        },
    }
    state["retrieval_plan"] = [
        {
            "subject": "math",
            "role": "core_concept",
            "local_retrieval_query": "math functions course concepts",
            "web_research_seed_query": "math functions official tutorial",
            "purpose": "support the requested quiz",
            "priority": 1.0,
            "_parent_child_priority_explicit": True,
        },
        {
            "subject": "computer",
            "role": "supporting_context",
            "local_retrieval_query": "computer systems course concepts",
            "web_research_seed_query": "computer systems official tutorial",
            "purpose": "support the requested quiz context",
            "priority": 0.7,
            "_parent_child_priority_explicit": True,
        },
    ]
    return state


def _available_math_path_output() -> dict:
    guidance = _learning_guidance_runtime()
    return {
        "schema_version": "learner_path_planner_output_v1",
        "runtime_fingerprint": guidance.runtime_fingerprint,
        "provider_projection_policy_fingerprint": (
            guidance.provider_projection_policy_fingerprint
        ),
        "provider_projection_max_steps": guidance.provider_projection_max_steps,
        "provider_projection_max_chars": guidance.provider_projection_max_chars,
        "request_id": "request-evidence-1",
        "status": "available",
        "unavailable_reason": None,
        "user_id": "learner-math-1",
        "subject": "math",
        "plan": {
            "schema_version": "learner_path_plan_v1",
            "user_id": "learner-math-1",
            "subject": "math",
            "generated_at": "2026-07-14T00:00:00Z",
            "steps": [
                {
                    "step_id": "path-functions-reinforce",
                    "position": 1,
                    "topic_id": "functions",
                    "subject": "math",
                    "title": "强化函数概念",
                    "status": "reinforce",
                    "estimated_hours": 2.0,
                    "reason": "最近学习记录显示函数概念仍需强化。",
                    "recommended_resource_types": ["quiz"],
                    "profile_signal_ids": ["skill-functions"],
                    "history_ids": ["history-functions-1"],
                }
            ],
            "summary": "先强化函数概念，再进入后续主题。",
        },
    }


def _available_math_path_provider_projection() -> dict:
    return {
        "schema_version": "learner_path_provider_projection_v1",
        "status": "available",
        "unavailable_reason": None,
        "subject": "math",
        "summary": "先强化函数概念，再进入后续主题。",
        "steps": [
            {
                "step_id": "path-functions-reinforce",
                "position": 1,
                "topic_id": "functions",
                "title": "强化函数概念",
                "status": "reinforce",
                "estimated_hours": 2.0,
                "reason": "最近学习记录显示函数概念仍需强化。",
                "recommended_resource_types": ["quiz"],
            }
        ],
    }


def _available_math_projection(
    runtime: orchestration.EvidenceOrchestrationRuntime,
):
    state = {
        **_planner_state(),
        "user_id": "learner-math-1",
        "learner_path_planner_output": _available_math_path_output(),
        "learner_path_provider_projection": (
            _available_math_path_provider_projection()
        ),
    }
    return orchestration.learner_path_provider_projection_for_runtime_from_state(
        state,
        runtime=runtime.learning_guidance,
    )


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
                evidence_ids=[],
                confidence=0.0,
                reason="No candidate satisfies the configured acceptance criteria.",
                suggested_local_query=local_query,
                suggested_web_query=web_query,
            )
        )
    return RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=round_index,
        coverages=rows,
    )


def _coverage_budget_validation_result(*, evidence_count: int, limit: int) -> str:
    requirement = compile_evidence_requirement_batch(_quiz_draft_batch(_runtime()))[0]
    entries: list[EvidenceLedgerEntry] = []
    for index in range(evidence_count):
        source_identity = f"{index + 1:064x}"
        content_fingerprint = f"{index + 101:064x}"
        evidence_id = make_evidence_id(
            requirement_id=requirement.requirement_id,
            source_type="local_rag",
            source_identity_fingerprint=source_identity,
            content_fingerprint=content_fingerprint,
        )
        entries.append(
            EvidenceLedgerEntry(
                round_index=0,
                task_id=f"task-{index}",
                requirement_id=requirement.requirement_id,
                evidence_id=evidence_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                source_type="local_rag",
                candidate_ref=f"candidate-{index}",
                candidate_snapshot_fingerprint=f"{index + 201:064x}",
                source_identity_fingerprint=source_identity,
                content_fingerprint=content_fingerprint,
                accepted=True,
                rejection_reason_code="",
            )
        )
    batch = RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=0,
        coverages=[
            RequirementCoverage(
                requirement_id=requirement.requirement_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                round_index=0,
                coverage_state="complete",
                evidence_ids=[entry.evidence_id for entry in entries],
                confidence=1.0,
                reason="The selected evidence satisfies the requirement.",
                suggested_local_query="",
                suggested_web_query="",
            )
        ],
    )
    return orchestration._coverage_business_validation(
        batch,
        round_index=0,
        max_evidence_per_requirement=limit,
        requirements=(requirement,),
        provisional_entries=tuple(entries),
        attempted_tasks=(),
        outcomes=(),
    )


def test_coverage_business_validation_accepts_evidence_limit_boundary() -> None:
    assert _coverage_budget_validation_result(evidence_count=4, limit=4) == ""


def test_coverage_business_validation_rejects_evidence_over_limit() -> None:
    error = _coverage_budget_validation_result(evidence_count=5, limit=4)

    assert "requirement_evidence_budget_exceeded" in error
    assert "max_evidence_per_requirement" in error


def test_staged_coverage_error_identifies_requirement_without_query_text() -> None:
    runtime = _runtime()
    need = next(
        item
        for item in runtime.profiles.profile_for("review_doc").needs
        if item.source_policy == "local_then_web_on_gap"
    )
    requirement = compile_evidence_requirement_batch(
        EvidenceRequirementDraftBatch(
            schema_version="evidence_requirement_draft_batch_v1",
            requirements=[
                EvidenceRequirementDraft(
                    resource_type="review_doc",
                    subject="math",
                    topic_id="functions",
                    profile_need_id=need.need_id,
                    evidence_kind=need.evidence_kind,
                    scope=need.scope,
                    criticality=need.criticality,
                    source_policy=need.source_policy,
                    acceptance_criteria=need.acceptance_criteria,
                    query_intent="math authoritative concepts",
                )
            ],
        )
    )[0]
    secret_query = "private staged web query"
    batch = RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=0,
        coverages=[
            RequirementCoverage(
                requirement_id=requirement.requirement_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                round_index=0,
                coverage_state="missing",
                evidence_ids=[],
                confidence=0.0,
                reason="The staged evidence is incomplete.",
                suggested_local_query="",
                suggested_web_query=secret_query,
            )
        ],
    )

    error = orchestration._coverage_business_validation(
        batch,
        round_index=0,
        max_evidence_per_requirement=runtime.policy.max_evidence_per_requirement,
        requirements=(requirement,),
        provisional_entries=(),
        attempted_tasks=(),
        outcomes=(),
    )

    assert "staged_gap_query_invalid" in error
    assert f"requirement_id={requirement.requirement_id}" in error
    assert secret_query not in error


def test_coverage_validation_reports_all_repeated_query_bindings() -> None:
    runtime = _runtime()
    requirements = compile_evidence_requirement_batch(_quiz_draft_batch(runtime))
    attempted_tasks: list[RetrievalTask] = []
    coverages: list[RequirementCoverage] = []
    expected_bindings: set[tuple[str, str]] = set()

    for index, requirement in enumerate(requirements):
        if requirement.source_policy == "local_only":
            source_types = ("local_rag",)
        elif requirement.source_policy == "web_only":
            source_types = ("web",)
        elif requirement.source_policy == "local_and_web":
            source_types = ("local_rag", "web")
        elif requirement.source_policy == "local_then_web_on_gap":
            source_types = ("local_rag",)
        else:
            raise AssertionError(
                f"unsupported source policy: {requirement.source_policy}"
            )

        queries = {
            source_type: f"math repeated evidence query {index} {source_type}"
            for source_type in source_types
        }
        for source_type, query in queries.items():
            attempted_tasks.append(
                build_retrieval_task(
                    requirement=requirement,
                    source_type=source_type,
                    query=query,
                    purpose=requirement.acceptance_criteria,
                    priority=runtime.policy.required_task_priority,
                    round_index=0,
                    result_limit=runtime.policy.max_results_per_task,
                )
            )
            expected_bindings.add((requirement.requirement_id, source_type))
        coverages.append(
            RequirementCoverage(
                requirement_id=requirement.requirement_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                round_index=0,
                coverage_state="missing",
                evidence_ids=[],
                confidence=0.0,
                reason="The configured acceptance criteria are not yet satisfied.",
                suggested_local_query=queries.get("local_rag", ""),
                suggested_web_query=queries.get("web", ""),
            )
        )

    error = orchestration._coverage_business_validation(
        RequirementCoverageBatch(
            schema_version="requirement_coverage_batch_v1",
            round_index=0,
            coverages=coverages,
        ),
        round_index=0,
        max_evidence_per_requirement=runtime.policy.max_evidence_per_requirement,
        requirements=requirements,
        provisional_entries=(),
        attempted_tasks=tuple(attempted_tasks),
        outcomes=(),
    )

    assert len(expected_bindings) >= 2
    assert "repeated_gap_query" in error
    for requirement_id, source_type in expected_bindings:
        assert f"{requirement_id}:{source_type}" in error


def test_candidate_records_are_stably_bounded_per_task() -> None:
    runtime = _runtime()
    requirements = compile_evidence_requirement_batch(_quiz_draft_batch(runtime))
    local_tasks = tuple(
        task
        for task in orchestration._build_initial_tasks(requirements, runtime)
        if task.source_type == "local_rag"
    )
    assert len(local_tasks) == 2
    assert all(task.result_limit == 3 for task in local_tasks)

    records: list[orchestration.EvidenceCandidateRecord] = []
    for candidate_index in range(5):
        for task_index, task in enumerate(local_tasks):
            raw_id = f"raw-local-{task_index}-{candidate_index}"
            records.append(
                orchestration._candidate_record(
                    candidate=EvidenceCandidate(
                        evidence_id=raw_id,
                        source_type="local_rag",
                        provider="chroma_parent_child",
                        subject=task.subject,
                        role=task.requirement_id,
                        title=f"Candidate {task_index}-{candidate_index}",
                        source=f"source-{task_index}.md",
                        content_preview=(
                            f"Evidence content {task_index}-{candidate_index}."
                        ),
                        metadata={"source_id": raw_id},
                    ),
                    original={"content": f"Original content for {raw_id}."},
                    task=task,
                )
            )

    bounded = orchestration._bound_candidate_records_to_task_limits(
        records,
        local_tasks,
    )

    assert len(bounded) == 6
    assert Counter(record.task_id for record in bounded) == {
        task.task_id: 3 for task in local_tasks
    }
    assert [record.candidate_ref for record in bounded] == [
        f"raw-local-{task_index}-{candidate_index}"
        for candidate_index in range(3)
        for task_index in range(2)
    ]


def test_judge_requirement_payload_has_exact_evidence_allowlists() -> None:
    runtime = _runtime()
    requirements = compile_evidence_requirement_batch(_quiz_draft_batch(runtime))
    tasks = orchestration._build_initial_tasks(requirements, runtime)
    records = tuple(
        orchestration._candidate_record(
            candidate=EvidenceCandidate(
                evidence_id=f"raw-candidate-{index}",
                source_type=task.source_type,
                provider="test-provider",
                subject=task.subject,
                role=task.requirement_id,
                title=f"Candidate {index}",
                source=f"source-{index}",
                content_preview=f"Evidence content {index}.",
                metadata={"source_id": f"source-{index}"},
            ),
            original={"content": f"Original content {index}."},
            task=task,
        )
        for index, task in enumerate(tasks)
    )

    payload = orchestration._judge_requirements_payload(requirements, records, tasks)
    payload_by_id = {str(item["requirement_id"]): item for item in payload}

    assert set(payload_by_id) == {item.requirement_id for item in requirements}
    for requirement in requirements:
        requirement_payload = payload_by_id[requirement.requirement_id]
        expected_evidence_ids = [
            record.evidence_id
            for record in records
            if record.requirement_id == requirement.requirement_id
        ]
        assert requirement_payload["eligible_evidence_ids"] == expected_evidence_ids
        bound_candidates = requirement_payload["bound_candidates"]
        assert isinstance(bound_candidates, list)
        assert [item["evidence_id"] for item in bound_candidates] == (
            expected_evidence_ids
        )
        assert {item["requirement_id"] for item in bound_candidates} == {
            requirement.requirement_id
        }
        expected_shape = (
            "both" if requirement.source_policy == "local_and_web" else "local_only"
        )
        assert requirement_payload["required_incomplete_query_shape"] == expected_shape


def test_coverage_binding_reask_discards_cross_requirement_references() -> None:
    runtime = _runtime()
    requirements = compile_evidence_requirement_batch(_quiz_draft_batch(runtime))
    tasks = orchestration._build_initial_tasks(requirements, runtime)
    records = tuple(
        orchestration._candidate_record(
            candidate=EvidenceCandidate(
                evidence_id=f"raw-candidate-{index}",
                source_type=task.source_type,
                provider="test-provider",
                subject=task.subject,
                role=task.requirement_id,
                title=f"Candidate {index}",
                source=f"source-{index}",
                content_preview=f"Evidence content {index}.",
                metadata={"source_id": f"source-{index}"},
            ),
            original={"content": f"Original content {index}."},
            task=task,
        )
        for index, task in enumerate(tasks)
    )
    first_requirement, second_requirement = requirements[:2]
    first_record = next(
        record
        for record in records
        if record.requirement_id == first_requirement.requirement_id
    )
    missing_batch = _missing_coverage(
        [requirement.model_dump(mode="json") for requirement in requirements],
        round_index=0,
        suffix="binding",
    )
    cross_bound_requirement_ids = {
        first_requirement.requirement_id,
        second_requirement.requirement_id,
    }
    coverage_rows: list[RequirementCoverage] = []
    for row in missing_batch.coverages:
        row_payload = row.model_dump(mode="json")
        if row.requirement_id in cross_bound_requirement_ids:
            row_payload["coverage_state"] = "partial"
            row_payload["evidence_ids"] = [first_record.evidence_id]
        coverage_rows.append(RequirementCoverage.model_validate(row_payload))
    batch = RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=0,
        coverages=coverage_rows,
    )

    error = orchestration._coverage_business_validation(
        batch,
        round_index=0,
        max_evidence_per_requirement=runtime.policy.max_evidence_per_requirement,
        requirements=requirements,
        provisional_entries=tuple(
            orchestration._ledger_entry(record, accepted=True) for record in records
        ),
        attempted_tasks=tasks,
        outcomes=(),
    )

    assert "invalid_coverage_evidence_ref" in error
    assert "discard all evidence_ids from the previous row" in error
    assert "nested bound_candidates" in error
    assert "Never retain an ID from another requirement" in error


def test_required_incomplete_query_shape_is_explicit_for_every_policy() -> None:
    assert (
        orchestration._required_incomplete_query_shape(
            source_policy="local_only",
            local_attempted=False,
        )
        == "local_only"
    )
    assert (
        orchestration._required_incomplete_query_shape(
            source_policy="web_only",
            local_attempted=False,
        )
        == "web_only"
    )
    assert (
        orchestration._required_incomplete_query_shape(
            source_policy="local_and_web",
            local_attempted=True,
        )
        == "both"
    )
    assert (
        orchestration._required_incomplete_query_shape(
            source_policy="local_then_web_on_gap",
            local_attempted=False,
        )
        == "local_only"
    )
    assert (
        orchestration._required_incomplete_query_shape(
            source_policy="local_then_web_on_gap",
            local_attempted=True,
        )
        == "web_only"
    )


def test_joint_candidate_graph_is_explicit_and_legacy_served_graph_is_unchanged():
    runtime = _runtime()
    legacy = build_graph(runtime.learning_guidance)
    candidate = build_resource_evidence_parent_child_graph(runtime)

    assert "resource_evidence_planner" not in legacy.nodes
    assert {
        "parent_child_retrieve",
        "web_research",
        "academic_parent_hydration",
        "resource_parent_hydration",
        "learner_path_planner",
        "resource_evidence_planner",
        "retrieval_round_router",
        "local_rag_search_batch",
        "web_research_search_batch",
        "retrieval_round_merge",
        "requirement_evidence_judge",
        "evidence_repair_planner",
        "resource_evidence_assignment",
        "resource_bundle_aggregator",
        "resource_recommendation_auto",
        "resource_recommendation_explicit",
        "recommendation_final_output",
    }.issubset(candidate.nodes)
    assert {
        "rag_retrieve",
        "web_search",
        "rag_generation_router",
        "parent_child_parent_hydration",
    }.isdisjoint(candidate.nodes)
    assert (
        ("local_rag_search_batch", "web_research_search_batch"),
        "retrieval_round_merge",
    ) in candidate.waiting_edges
    assert ("learner_path_planner", "resource_evidence_planner") in candidate.edges
    assert ("resource_worker", "resource_bundle_aggregator") in candidate.edges
    assert (
        "resource_bundle_aggregator",
        "resource_recommendation_auto",
    ) in candidate.edges
    assert (
        "resource_recommendation_auto",
        "resource_bundle_output",
    ) in candidate.edges
    assert (
        "resource_recommendation_explicit",
        "recommendation_final_output",
    ) in candidate.edges
    assert ("recommendation_final_output", "__end__") in candidate.edges
    assert "learner_path_planner" not in legacy.nodes
    assert "resource_bundle_aggregator" not in legacy.nodes
    assert "resource_recommendation_auto" not in legacy.nodes
    assert get_node_runtime_metadata("learner_path_planner") is not None
    assert get_node_runtime_metadata("resource_bundle_aggregator") is not None
    assert get_node_runtime_metadata("resource_recommendation_auto") is not None
    assert get_node_runtime_metadata("resource_recommendation_explicit") is not None
    assert get_node_runtime_metadata("recommendation_final_output") is not None
    assert candidate.compile() is not None


def test_candidate_graph_has_no_superseded_generation_router_contract():
    assert not hasattr(orchestration, "make_rag_generation_router_node")
    assert "make_rag_generation_router_node" not in orchestration.__all__
    assert "rag_generation_route" not in LearningState.__annotations__
    assert "rag_generation_route" not in initial_request_reset_transient_state()
    assert get_node_runtime_metadata("rag_generation_router") is None


def test_guidance_runtime_fingerprint_changes_candidate_identity():
    runtime = _runtime()
    changed = replace(
        runtime,
        learning_guidance=replace(
            runtime.learning_guidance,
            runtime_fingerprint="4" * 64,
        ),
    )

    assert runtime.orchestration_fingerprint != changed.orchestration_fingerprint


def test_guidance_projection_policy_changes_candidate_identity():
    runtime = _runtime()
    changed_steps = replace(
        runtime,
        learning_guidance=replace(
            runtime.learning_guidance,
            provider_projection_max_steps=49,
        ),
    )
    changed_chars = replace(
        runtime,
        learning_guidance=replace(
            runtime.learning_guidance,
            provider_projection_max_chars=65_535,
        ),
    )

    fingerprints = {
        runtime.orchestration_fingerprint,
        changed_steps.orchestration_fingerprint,
        changed_chars.orchestration_fingerprint,
    }
    assert len(fingerprints) == 3


def test_web_timeout_changes_candidate_identity():
    runtime = _runtime()
    changed = replace(runtime, web_timeout_seconds=11.0)

    assert runtime.orchestration_fingerprint != changed.orchestration_fingerprint


def test_downstream_rejects_missing_or_changed_orchestration_fingerprint():
    runtime = _runtime()
    changed = replace(
        runtime,
        learning_guidance=replace(
            runtime.learning_guidance,
            runtime_fingerprint="4" * 64,
        ),
    )
    router = orchestration.make_retrieval_round_router_node(changed)

    with pytest.raises(
        orchestration.EvidenceOrchestrationRuntimeError,
        match="missing_evidence_orchestration_fingerprint",
    ):
        router({})
    with pytest.raises(
        orchestration.EvidenceOrchestrationRuntimeError,
        match="evidence_orchestration_fingerprint_mismatch",
    ):
        router(
            {"evidence_orchestration_fingerprint": (runtime.orchestration_fingerprint)}
        )


def test_candidate_node_metadata_matches_path_then_evidence_topology():
    rewrite = get_node_runtime_metadata("search_query_rewriter")
    learner_path = get_node_runtime_metadata("learner_path_planner")
    evidence_plan = get_node_runtime_metadata("resource_evidence_planner")

    assert rewrite is not None
    assert learner_path is not None
    assert evidence_plan is not None
    assert rewrite.stage_rank < learner_path.stage_rank < evidence_plan.stage_rank
    assert rewrite.order < learner_path.order < evidence_plan.order


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


@pytest.mark.parametrize(
    ("requested", "singular", "expected_code"),
    (
        ([" quiz "], "quiz", "noncanonical_requested_resources"),
        (["quiz", "quiz"], "quiz", "noncanonical_requested_resources"),
        (["exercise"], "exercise", "noncanonical_requested_resources"),
        (["quiz"], "mindmap", "requested_resource_shadow_mismatch"),
    ),
)
def test_evidence_planner_rejects_resource_ingress_repair(
    requested,
    singular,
    expected_code,
):
    with pytest.raises(
        orchestration.EvidenceOrchestrationRuntimeError,
        match=expected_code,
    ):
        orchestration._requested_resources(
            {
                "requested_resource_types": requested,
                "requested_resource_type": singular,
            }
        )


@pytest.mark.parametrize(
    ("plan", "expected_code"),
    (
        ([{"subject": " math "}], "invalid_subject_plan"),
        ([{"subject": "math"}, {"subject": "math"}], "duplicate_subject_plan"),
        ([{"subject": "math"}, "bad-entry"], "invalid_subject_plan"),
    ),
)
def test_evidence_planner_rejects_subject_ingress_repair(plan, expected_code):
    with pytest.raises(
        orchestration.EvidenceOrchestrationRuntimeError,
        match=expected_code,
    ):
        orchestration._requested_subjects(
            {"retrieval_plan": plan},
            _runtime(),
        )


def test_planner_compiles_profile_slots_and_first_repair_round(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(_planner_state())
    )

    assert planned["evidence_orchestration_fingerprint"] == (
        runtime.orchestration_fingerprint
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


def test_requirement_evidence_judge_receives_attempted_query_text(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    base_state = _planner_state()
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(base_state)
    )
    attempted_tasks = [
        RetrievalTask.model_validate(item) for item in planned["evidence_current_tasks"]
    ]
    missing = _missing_coverage(
        planned["evidence_requirements"],
        round_index=0,
        suffix="focused repair",
    )
    captured: dict[str, object] = {}

    async def fake_judge(**kwargs):
        captured.update(kwargs)
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
        for task in attempted_tasks
    ]
    asyncio.run(
        orchestration.make_requirement_evidence_judge_node(runtime)(
            {
                **base_state,
                **planned,
                "evidence_candidate_records": [],
                "evidence_source_outcomes": outcomes,
            }
        )
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    prompt = messages[1].content
    for task in attempted_tasks:
        assert f'"requirement_id": "{task.requirement_id}"' in prompt
        assert f'"source_type": "{task.source_type}"' in prompt
        assert f'"query": "{task.query}"' in prompt
        assert f'"query_fingerprint": "{task.query_fingerprint}"' in prompt


def test_multisubject_planner_keeps_all_slots_and_scopes_available_path(monkeypatch):
    runtime = _multisubject_runtime()
    batch = _multisubject_quiz_draft_batch(runtime)
    captured: dict[str, object] = {}

    async def fake_planner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed=batch)

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(
            _multisubject_planner_state()
        )
    )

    requirements = planned["evidence_requirements"]
    assert {
        (
            requirement["subject"],
            requirement["topic_id"],
            requirement["profile_need_id"],
        )
        for requirement in requirements
    } == {
        ("math", "functions", need.need_id)
        for need in runtime.profiles.profile_for("quiz").needs
    } | {
        ("computer", "computer.systems", need.need_id)
        for need in runtime.profiles.profile_for("quiz").needs
    }
    available_projection = _available_math_projection(runtime)
    assert (
        orchestration._planner_business_validation(
            batch,
            resources=("quiz",),
            subjects=("math", "computer"),
            learner_path_projection=available_projection,
            runtime=runtime,
        )
        == ""
    )
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "constrains only requirements" in messages[0].content
    assert "never copy a path topic across subjects" in messages[0].content
    assert "Subject-scope rule" in messages[1].content
    assert "Every other selected subject is still mandatory" in messages[1].content


def test_planner_reask_diagnostic_names_missing_supporting_subject_slots():
    runtime = _multisubject_runtime()
    projection = _available_math_projection(runtime)
    incomplete = _multisubject_quiz_draft_batch(
        runtime,
        include_computer=False,
    )

    feedback = orchestration._planner_business_validation(
        incomplete,
        resources=("quiz",),
        subjects=("math", "computer"),
        learner_path_projection=projection,
        runtime=runtime,
    )

    assert feedback.startswith("requirement_inventory_mismatch:")
    for need in runtime.profiles.profile_for("quiz").needs:
        assert (
            f"resource_type=quiz|subject=computer|profile_need_id={need.need_id}"
        ) in feedback
    assert "unexpected_slots=[]" in feedback


def test_planner_rejects_copying_path_topic_to_supporting_subject():
    runtime = _multisubject_runtime()
    projection = _available_math_projection(runtime)
    invalid = _multisubject_quiz_draft_batch(
        runtime,
        computer_topic_id="functions",
    )

    feedback = orchestration._planner_business_validation(
        invalid,
        resources=("quiz",),
        subjects=("math", "computer"),
        learner_path_projection=projection,
        runtime=runtime,
    )

    assert feedback.startswith("requirement_topic_not_in_knowledge_graph:")


def test_evidence_planner_injects_only_provider_safe_learner_path(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)
    captured: dict[str, object] = {}

    async def fake_planner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed=batch)

    state = {
        **_planner_state(),
        "user_id": "learner-math-1",
        "learner_path_planner_output": _available_math_path_output(),
        "learner_path_provider_projection": (
            _available_math_path_provider_projection()
        ),
    }
    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)

    asyncio.run(orchestration.make_resource_evidence_planner_node(runtime)(state))

    prompt = captured["messages"][1].content
    assert '"step_id":"path-functions-reinforce"' in prompt
    assert '"status":"available"' in prompt
    assert '"schema_version":"learner_path_provider_projection_v1"' in prompt
    for forbidden_field in (
        "request_id",
        "user_id",
        "profile_signal_ids",
        "history_ids",
        "runtime_fingerprint",
        "provider_projection_policy_fingerprint",
    ):
        assert f'"{forbidden_field}"' not in prompt
    assert "request-evidence-1" not in prompt
    assert "learner-math-1" not in prompt
    assert "skill-functions" not in prompt
    assert "history-functions-1" not in prompt


def test_evidence_planner_rejects_resource_topic_swap_inside_available_path(
    monkeypatch,
):
    runtime = _runtime()
    payload = _quiz_draft_batch(runtime).model_dump(mode="python")
    for requirement in payload["requirements"]:
        requirement["topic_id"] = "limits"
    swapped = EvidenceRequirementDraftBatch.model_validate(payload)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=swapped)

    state = {
        **_planner_state(),
        "user_id": "learner-math-1",
        "learner_path_planner_output": _available_math_path_output(),
        "learner_path_provider_projection": (
            _available_math_path_provider_projection()
        ),
    }
    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)

    with pytest.raises(
        orchestration.EvidenceOrchestrationContractError,
        match="requirement_resource_topic_mismatch",
    ):
        asyncio.run(orchestration.make_resource_evidence_planner_node(runtime)(state))


def test_evidence_planner_rejects_stale_learner_path_before_llm(monkeypatch):
    runtime = _runtime()
    state = _planner_state()
    state["learner_path_planner_output"] = {
        **state["learner_path_planner_output"],
        "request_id": "stale-request",
    }

    async def forbidden_planner(**_kwargs):
        raise AssertionError("stale learner path must fail before provider dispatch")

    monkeypatch.setattr(orchestration, "invoke_structured_llm", forbidden_planner)

    with pytest.raises(RuntimeError, match="learner_path_request_mismatch"):
        asyncio.run(orchestration.make_resource_evidence_planner_node(runtime)(state))


def test_evidence_planner_rejects_path_from_changed_guidance_runtime(monkeypatch):
    runtime = _runtime()
    changed = replace(
        runtime,
        learning_guidance=replace(
            runtime.learning_guidance,
            runtime_fingerprint="4" * 64,
        ),
    )
    state = _planner_state()

    async def forbidden_planner(**_kwargs):
        raise AssertionError("stale guidance output must fail before provider dispatch")

    monkeypatch.setattr(orchestration, "invoke_structured_llm", forbidden_planner)

    with pytest.raises(RuntimeError, match="learning_guidance_runtime_mismatch"):
        asyncio.run(orchestration.make_resource_evidence_planner_node(changed)(state))


def test_repair_schedules_third_supplement_then_rejects_beyond_budget(monkeypatch):
    runtime = _runtime()
    assert runtime.policy.max_supplement_rounds == 3
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
        "evidence_coverage": compile_requirement_coverage_batch(
            first_missing
        ).model_dump(mode="json"),
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
            "evidence_coverage": compile_requirement_coverage_batch(
                second_missing
            ).model_dump(mode="json"),
            "resource_evidence_readiness": readiness,
        }
    )

    assert round_two["evidence_current_round"] == 2
    assert all(
        RetrievalTask.model_validate(item).round_index == 2
        for item in round_two["evidence_current_tasks"]
    )

    third_missing = _missing_coverage(
        requirements,
        round_index=2,
        suffix="repair three",
    )
    round_three_state = {
        **round_one_state,
        **round_one,
        **round_two,
        "evidence_coverage": compile_requirement_coverage_batch(
            third_missing
        ).model_dump(mode="json"),
        "resource_evidence_readiness": readiness,
    }
    round_three = orchestration.make_evidence_repair_planner_node(runtime)(
        round_three_state
    )

    assert round_three["evidence_current_round"] == 3
    assert all(
        RetrievalTask.model_validate(item).round_index == 3
        for item in round_three["evidence_current_tasks"]
    )

    with pytest.raises(orchestration.EvidenceBudgetExceededError) as exc_info:
        orchestration.make_evidence_repair_planner_node(runtime)(
            {
                **round_three_state,
                **round_three,
                "resource_evidence_readiness": readiness,
            }
        )

    assert exc_info.value.code == "repair_round_budget_exceeded"

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
                "evidence_coverage": compile_requirement_coverage_batch(
                    repeated
                ).model_dump(mode="json"),
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
            requirements=[
                EvidenceRequirementDraft(
                    resource_type="review_doc",
                    subject="math",
                    topic_id="functions",
                    profile_need_id=need.need_id,
                    evidence_kind=need.evidence_kind,
                    scope=need.scope,
                    criticality=need.criticality,
                    source_policy=need.source_policy,
                    acceptance_criteria=need.acceptance_criteria,
                    query_intent="math authoritative concepts",
                ),
            ],
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
            requirements=[
                EvidenceRequirementDraft(
                    resource_type="code_practice",
                    subject="math",
                    topic_id="functions",
                    profile_need_id=need.need_id,
                    evidence_kind=need.evidence_kind,
                    scope=need.scope,
                    criticality=need.criticality,
                    source_policy=need.source_policy,
                    acceptance_criteria=need.acceptance_criteria,
                    query_intent="math executable pattern evidence",
                ),
            ],
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
        coverages=[
            RequirementCoverage(
                requirement_id=requirement.requirement_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                round_index=0,
                coverage_state="missing",
                evidence_ids=[],
                confidence=0.2,
                reason="The Web half of the required evidence is still missing.",
                suggested_local_query="math executable pattern evidence",
                suggested_web_query="math official executable pattern evidence",
            ),
        ],
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
            "evidence_orchestration_fingerprint": runtime.orchestration_fingerprint,
            "evidence_current_round": 0,
            "evidence_requirements": [requirement.model_dump(mode="json")],
            "evidence_coverage": compile_requirement_coverage_batch(
                coverage
            ).model_dump(mode="json"),
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


def test_direct_web_execution_honors_single_task_concurrency(monkeypatch) -> None:
    tasks = [
        WebResearchTask(
            task_id=f"task-web-{index}",
            subject="math",
            role=f"requirement_math_{index}",
            purpose="Find exact evidence.",
            search_query=f"math official evidence {index}",
            reason="The requirement needs current evidence.",
            priority=1.0,
        )
        for index in range(3)
    ]
    active_count = 0
    peak_active_count = 0

    async def observed_executor(**_kwargs):
        nonlocal active_count, peak_active_count
        active_count += 1
        peak_active_count = max(peak_active_count, active_count)
        try:
            await asyncio.sleep(0.01)
            return [], [{"status": "failed", "error_type": None}]
        finally:
            active_count -= 1

    monkeypatch.setattr(academic, "_execute_web_research_tasks", observed_executor)

    result = asyncio.run(
        academic.execute_validated_web_research_tasks(
            state={},
            tasks=tasks,
            original_user_query="math evidence",
            timeout=5.0,
            max_results_per_task=3,
            max_concurrent_tasks=1,
        )
    )

    assert result["status"] == "empty"
    assert peak_active_count == 1


@pytest.mark.parametrize(
    ("factory", "reason_code", "source"),
    [
        (
            orchestration.make_retrieval_round_merge_node,
            "retrieval_round_merge_failed",
            "orchestration",
        ),
        (
            orchestration.make_evidence_repair_planner_node,
            "evidence_repair_planner_failed",
            "orchestration",
        ),
        (
            orchestration.make_resource_evidence_assignment_node,
            "resource_evidence_assignment_failed",
            "assignment",
        ),
    ],
)
def test_sync_orchestration_boundaries_emit_one_overall_failure(
    factory,
    reason_code: str,
    source: str,
):
    runtime = _runtime()
    state = {
        "evidence_orchestration_fingerprint": runtime.orchestration_fingerprint,
        "evidence_current_round": 0,
        "evidence_all_tasks": [],
    }
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        with pytest.raises(Exception):
            factory(runtime)(state)
    finally:
        reset_trace_event_sink(token)

    failures = [
        event for event in sink if event.get("stage") == "evidence_orchestration.failed"
    ]
    assert len(failures) == 1
    assert failures[0]["reason_code"] == reason_code
    assert failures[0]["source"] == source


def test_local_source_failure_is_followed_by_one_overall_failure(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    async def failed_parent_rag(_state):
        raise RuntimeError("fixture local retrieval failure")

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    monkeypatch.setattr(
        orchestration,
        "make_parent_child_rag_node",
        lambda _runtime: failed_parent_rag,
    )
    base_state = _planner_state()
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(base_state)
    )
    state = {**base_state, **planned}
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        with pytest.raises(RuntimeError, match="fixture local retrieval failure"):
            asyncio.run(orchestration.make_local_rag_search_batch_node(runtime)(state))
    finally:
        reset_trace_event_sink(token)

    stages = [
        event.get("stage")
        for event in sink
        if str(event.get("stage") or "").startswith("evidence_orchestration.")
    ]
    assert stages == [
        "evidence_orchestration.source.failed",
        "evidence_orchestration.failed",
    ]
    failures = [
        event for event in sink if event.get("stage") == "evidence_orchestration.failed"
    ]
    assert len(failures) == 1
    assert failures[0]["source"] == "local"


def test_judge_failure_emits_one_overall_failure(monkeypatch):
    runtime = _runtime()
    batch = _quiz_draft_batch(runtime)

    async def fake_planner(**_kwargs):
        return SimpleNamespace(parsed=batch)

    async def failed_judge(**_kwargs):
        raise RuntimeError("fixture judge failure")

    monkeypatch.setattr(orchestration, "invoke_structured_llm", fake_planner)
    base_state = _planner_state()
    planned = asyncio.run(
        orchestration.make_resource_evidence_planner_node(runtime)(base_state)
    )
    monkeypatch.setattr(orchestration, "invoke_structured_llm", failed_judge)
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        with pytest.raises(RuntimeError, match="fixture judge failure"):
            asyncio.run(
                orchestration.make_requirement_evidence_judge_node(runtime)(
                    {**base_state, **planned}
                )
            )
    finally:
        reset_trace_event_sink(token)

    failures = [
        event for event in sink if event.get("stage") == "evidence_orchestration.failed"
    ]
    assert len(failures) == 1
    assert failures[0]["source"] == "judge"
    assert failures[0]["reason_code"] == "coverage_judge_failed"


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
        "evidence_coverage": compile_requirement_coverage_batch(missing).model_dump(
            mode="json"
        ),
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


def test_terminal_parent_hydration_stays_on_sqlite_owner_thread():
    source = inspect.getsource(orchestration.make_terminal_parent_hydration_node)

    assert "asyncio.to_thread" not in source
    assert "retriever.hydrate_kept_multi" in source


def test_all_blocked_resources_skip_workers_and_return_explicit_bundle():
    state = {
        "request_id": "request-blocked",
        "session_id": "session-blocked",
        "thread_id": "thread-blocked",
        "evidence_requested_resource_types": ["quiz"],
        "requested_resource_type": "",
        "requested_resource_types": [],
        "blocked_resource_types": ["quiz"],
        "resource_evidence_contract_version": RESOURCE_EVIDENCE_CONTRACT_VERSION,
        "resource_evidence_assignments": [],
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
