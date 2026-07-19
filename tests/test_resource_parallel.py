"""Tests for parallel learning-resource generation orchestration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.assessment.attempt_contracts import AssessmentQuizSourceItemV1
from src.config.evidence_orchestration_contracts import (
    RESOURCE_EVIDENCE_CONTRACT_VERSION,
    make_resource_assignment_fingerprint,
)
from src.context_engineering.influence import (
    build_influence_entry,
    merge_context_influence_ledger,
)
from src.assessment.identity import stable_exercise_question_id
from src.graph import resource_generation as rg
from src.graph.assessment_quiz import build_assessment_quiz_projection_v1
from src.graph.resource_final_v3 import (
    ResourceFinalV3,
    ResourceFinalV3ResourceValidation,
)
from src.graph.resource_validation import ResourceValidationResultV1
from src.graph.resource_generation import (
    dispatch_resource_workers,
    dispatch_resource_workers_to_recommendation_aggregator,
    normalize_requested_resource_types,
    resource_bundle_aggregator,
    resource_bundle_output,
    resource_bundle_output_with_recommendations,
    resource_orchestrator,
    resource_preflight_router,
    resource_worker,
    route_after_resource_preflight,
)
from src.graph.state import LearningState, resource_branch_results_reducer
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.learning_guidance.runtime import (
    LearningGuidanceContractError,
    LearningGuidanceRuntime,
)
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink
from src.resource_contracts import ResourceType


FALLBACK_DELIVERY_TIMEOUTS_BY_RESOURCE: dict[ResourceType, float] = {
    "review_doc": 120.0,
    "mindmap": 120.0,
    "quiz": 120.0,
    "code_practice": 120.0,
    "video_script": 240.0,
    "video_animation": 120.0,
    "study_plan": 120.0,
}


async def _unexpected_guidance_dependency(*_args, **_kwargs):
    raise AssertionError("resource bundle tests must not invoke guidance dependencies")


def _knowledge_graph() -> KnowledgeGraphV1:
    return KnowledgeGraphV1.model_validate(
        {
            "schema_version": "knowledge_graph_v1",
            "data_version": "resource-parallel-test-v1",
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
                                    "resource_id": "functions.mindmap",
                                    "resource_type": "mindmap",
                                    "title": "Functions mindmap",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def _candidate_assignment(resource_type: ResourceType) -> dict[str, object]:
    subjects = ("math",)
    topic_ids = ("functions",)
    requirement_ids = (f"requirement:{resource_type}",)
    evidence_ids = (f"evidence:{resource_type}",)
    return {
        "resource_type": resource_type,
        "subjects": list(subjects),
        "topic_ids": list(topic_ids),
        "requirement_ids": list(requirement_ids),
        "evidence_ids": list(evidence_ids),
        "delivery_mode": "strict",
        "unmet_requirement_ids": [],
        "assignment_fingerprint": make_resource_assignment_fingerprint(
            resource_type=resource_type,
            subjects=subjects,
            topic_ids=topic_ids,
            requirement_ids=requirement_ids,
            evidence_ids=evidence_ids,
            delivery_mode="strict",
            unmet_requirement_ids=(),
        ),
    }


def _candidate_evidence_state(
    *requested_resource_types: ResourceType,
    assigned_resource_types: tuple[ResourceType, ...] | None = None,
) -> dict[str, object]:
    assigned = (
        requested_resource_types
        if assigned_resource_types is None
        else assigned_resource_types
    )
    return {
        "evidence_requested_resource_types": list(requested_resource_types),
        "resource_evidence_contract_version": RESOURCE_EVIDENCE_CONTRACT_VERSION,
        "resource_evidence_assignments": [
            _candidate_assignment(resource_type) for resource_type in assigned
        ],
    }


@pytest.mark.parametrize(
    "resource_type",
    (
        "review_doc",
        "mindmap",
        "quiz",
        "code_practice",
        "video_script",
        "video_animation",
        "study_plan",
    ),
)
def test_every_resource_type_schedules_a_bounded_fallback_task(
    resource_type: ResourceType,
) -> None:
    state = _candidate_evidence_state(resource_type)
    assignment = dict(state["resource_evidence_assignments"][0])
    requirement_ids = tuple(assignment["requirement_ids"])
    evidence_ids = tuple(assignment["evidence_ids"])
    subjects = tuple(assignment["subjects"])
    topic_ids = tuple(assignment["topic_ids"])
    assignment.update(
        {
            "delivery_mode": "fallback",
            "unmet_requirement_ids": list(requirement_ids),
            "assignment_fingerprint": make_resource_assignment_fingerprint(
                resource_type=resource_type,
                subjects=subjects,
                topic_ids=topic_ids,
                requirement_ids=requirement_ids,
                evidence_ids=evidence_ids,
                delivery_mode="fallback",
                unmet_requirement_ids=requirement_ids,
            ),
        }
    )
    state.update(
        {
            "resource_evidence_assignments": [assignment],
            "requested_resource_type": resource_type,
            "requested_resource_types": [resource_type],
            "resource_fallback_delivery_max_seconds_by_resource": dict(
                FALLBACK_DELIVERY_TIMEOUTS_BY_RESOURCE
            ),
        }
    )

    tasks = rg._resource_plan_from_state(state)

    assert tasks == [
        {
            "task_id": f"resource:{resource_type}",
            "resource_type": resource_type,
            "subjects": ["math"],
            "topic_ids": ["functions"],
            "delivery_mode": "fallback",
            "fallback_delivery_timeout_seconds": (
                FALLBACK_DELIVERY_TIMEOUTS_BY_RESOURCE[resource_type]
            ),
            "status": "pending",
        }
    ]


def test_fallback_result_is_revalidated_and_never_promoted_to_success() -> None:
    validation = ResourceValidationResultV1(
        schema_version="resource_validation_v1",
        resource_type="mindmap",
        valid=True,
        terminal_status="success",
        renderable_count=1,
        downloadable_count=0,
        verified_local_count=0,
        remote_unverified_count=0,
        failure_reason="",
        warnings=(),
    )

    result = rg._mark_fallback_result_partial_success(
        {"status": "success", "validation": validation.model_dump(mode="json")}
    )

    assert result["status"] == "partial_success"
    assert result["validation"]["terminal_status"] == "partial_success"
    assert result["validation"]["warnings"] == ["evidence_scope_limited"]


GUIDANCE_RUNTIME = LearningGuidanceRuntime(
    runtime_fingerprint="7" * 64,
    knowledge_graph=_knowledge_graph(),
    provider_projection_max_steps=50,
    provider_projection_max_chars=65_536,
    load_profile=_unexpected_guidance_dependency,
    load_history=_unexpected_guidance_dependency,
    plan_learning_path=_unexpected_guidance_dependency,
    recommend_resources=_unexpected_guidance_dependency,
)


def _public_quiz_item() -> dict:
    question = "Q"
    tags = ["testing"]
    return {
        "schema_version": "exercise_card_v1",
        "question_id": stable_exercise_question_id(
            level="basic",
            question_type="free_text",
            question=question,
            choices=(),
            tags=tags,
        ),
        "question_type": "free_text",
        "level": "basic",
        "question": question,
        "choices": [],
        "tags": tags,
    }


def _quiz_binding(*, thread_id: str, request_id: str) -> dict:
    public = _public_quiz_item()
    source = AssessmentQuizSourceItemV1(
        question_id=public["question_id"],
        question_type="free_text",
        level="basic",
        question="Q",
        choices=(),
        answer="SERVER_ONLY_ANSWER",
        explanation="Private explanation.",
        pitfall="Private pitfall.",
        tags=("testing",),
    )
    projection = build_assessment_quiz_projection_v1(
        thread_id=thread_id,
        request_id=request_id,
        title="Mock Quiz",
        summary="One validated public exercise card.",
        source_items=(source,),
        artifact_refs={},
        validation=ResourceFinalV3ResourceValidation(
            schema_version="resource_validation_v1",
            resource_type="quiz",
            valid=True,
            terminal_status="success",
            renderable_count=1,
            downloadable_count=0,
            verified_local_count=0,
            remote_unverified_count=0,
            failure_reason="",
            warnings=(),
        ),
        learning_guidance_binding=None,
    )
    return {
        "exercise_items": [public],
        "exercise_artifact": {
            "schema_version": "exercise_public_artifact_v1",
            "title": "Mock Quiz",
            "items": [public],
            "resource_id": projection.public_resource.resource_id,
            "payload_hash": projection.public_resource.payload_hash,
        },
        "exercise_resource_v3": projection.public_resource.model_dump(mode="json"),
        "assessment_checkpoint_resources": {
            "schema_version": "assessment_checkpoint_resources_v2",
            "thread_id": thread_id,
            "resources": [projection.checkpoint_resource.model_dump(mode="json")],
        },
    }


def _branch_validation(resource_type: str, *, status: str = "success") -> dict:
    return {
        "schema_version": "resource_validation_v1",
        "resource_type": resource_type,
        "valid": True,
        "terminal_status": status,
        "renderable_count": 1,
        "downloadable_count": 0,
        "verified_local_count": 0,
        "remote_unverified_count": 0,
        "failure_reason": "",
        "warnings": [],
    }


def _mindmap_branch_result(*, influence_entries: list[dict] | None = None) -> dict:
    tree = {
        "title": "Mock Map",
        "children": [{"title": "Functions"}],
    }
    result = {
        "resource_type": "mindmap",
        "subjects": ["math"],
        "topic_ids": ["functions"],
        "status": "success",
        "title": "Mock Map",
        "artifact": {"title": "Mock Map", "tree": tree, "xmind_url": "/m.xmind"},
        "artifacts": [],
        "state_updates": {
            "mindmap_artifact": {
                "title": "Mock Map",
                "tree": tree,
                "xmind_url": "/m.xmind",
            },
            "mindmap_tree": tree,
        },
        "message_content": "Generated mindmap: Mock Map",
        "message_preview": "Generated mindmap: Mock Map",
        "elapsed_ms": 10,
        "validation": _branch_validation("mindmap"),
    }
    if influence_entries is not None:
        result["context_influence_entries"] = influence_entries
    return result


def _failed_branch_result(resource_type: str) -> dict:
    return {
        "resource_type": resource_type,
        "status": "failed",
        "title": resource_type,
        "artifact": {},
        "artifacts": [],
        "state_updates": {},
        "message_content": "",
        "message_preview": "",
        "error_code": f"{resource_type}.generation_failed",
        "error_type": "RuntimeError",
        "error_message_sanitized": f"{resource_type} failed",
        "elapsed_ms": 5,
        "validation": None,
    }


def _automatic_recommendation_output(
    source_resource_id: str,
    *,
    request_id: str = "request-1",
) -> dict:
    return {
        "schema_version": "resource_recommendation_output_v1",
        "runtime_fingerprint": GUIDANCE_RUNTIME.runtime_fingerprint,
        "provider_projection_policy_fingerprint": (
            GUIDANCE_RUNTIME.provider_projection_policy_fingerprint
        ),
        "provider_projection_max_steps": (
            GUIDANCE_RUNTIME.provider_projection_max_steps
        ),
        "provider_projection_max_chars": (
            GUIDANCE_RUNTIME.provider_projection_max_chars
        ),
        "request_id": request_id,
        "mode": "automatic_after_generation",
        "status": "available",
        "unavailable_reason": None,
        "user_id": "learner-1",
        "subject": "math",
        "batch": {
            "schema_version": "resource_recommendation_batch_v1",
            "mode": "automatic_after_generation",
            "user_id": "learner-1",
            "subject": "math",
            "generated_at": "2026-07-14T00:00:00Z",
            "items": [
                {
                    "recommendation_id": "recommendation-math-mindmap",
                    "resource_id": source_resource_id,
                    "resource_type": "mindmap",
                    "subject": "math",
                    "topic_id": "functions",
                    "title": "Mock Map",
                    "rank": 1,
                    "score_factors": {
                        "weakness": 0.8,
                        "forgetting": 0.6,
                        "preference": 0.4,
                        "goal": 0.2,
                        "combined": 0.5,
                        "weights": {
                            "weakness": 0.25,
                            "forgetting": 0.25,
                            "preference": 0.25,
                            "goal": 0.25,
                        },
                    },
                    "reason": "学习画像与最近学习记录均支持继续强化函数知识结构。",
                    "profile_signal_ids": ["skill-math", "goal-math"],
                    "history_ids": ["history-math-1"],
                    "source_resource_ids": [source_resource_id],
                }
            ],
            "summary": "已基于真实生成资源给出一项个性化推荐。",
        },
    }


def _automatic_recommendation_unavailable(reason: str) -> dict:
    return {
        "schema_version": "resource_recommendation_output_v1",
        "runtime_fingerprint": GUIDANCE_RUNTIME.runtime_fingerprint,
        "provider_projection_policy_fingerprint": (
            GUIDANCE_RUNTIME.provider_projection_policy_fingerprint
        ),
        "provider_projection_max_steps": (
            GUIDANCE_RUNTIME.provider_projection_max_steps
        ),
        "provider_projection_max_chars": (
            GUIDANCE_RUNTIME.provider_projection_max_chars
        ),
        "request_id": "request-1",
        "mode": "automatic_after_generation",
        "status": "unavailable",
        "unavailable_reason": reason,
        "user_id": "learner-1",
        "subject": "math",
        "batch": None,
    }


def _compile_candidate_resource_test_graph(
    *,
    orchestrator,
    worker,
    aggregator,
    recommendation,
    finalizer,
):
    graph = StateGraph(LearningState)
    graph.add_node("resource_orchestrator", orchestrator)
    graph.add_node("resource_worker", worker)
    graph.add_node("resource_bundle_aggregator", aggregator)
    graph.add_node("resource_recommendation_auto", recommendation)
    graph.add_node("resource_bundle_output", finalizer)
    graph.set_entry_point("resource_orchestrator")
    graph.add_conditional_edges(
        "resource_orchestrator",
        dispatch_resource_workers_to_recommendation_aggregator,
    )
    graph.add_edge("resource_worker", "resource_bundle_aggregator")
    graph.add_edge("resource_bundle_aggregator", "resource_recommendation_auto")
    graph.add_edge("resource_recommendation_auto", "resource_bundle_output")
    graph.add_edge("resource_bundle_output", END)
    return graph.compile()


def test_normalize_requested_resource_types_dedupes_and_aliases():
    assert normalize_requested_resource_types(
        ["mindmap", "exercise"], "quiz", "roadmap"
    ) == [
        "mindmap",
        "quiz",
        "study_plan",
    ]


def test_resource_branch_results_reducer_merges_by_resource_type():
    existing = [{"resource_type": "quiz", "status": "failed"}]
    update = [
        {"resource_type": "quiz", "status": "success"},
        {"resource_type": "mindmap", "status": "success"},
    ]

    merged = resource_branch_results_reducer(existing, update)

    assert [item["resource_type"] for item in merged] == ["mindmap", "quiz"]
    assert (
        next(item for item in merged if item["resource_type"] == "quiz")["status"]
        == "success"
    )


async def test_resource_orchestrator_plans_dynamic_worker_tasks():
    result = await resource_orchestrator(
        {
            "request_id": "req-1",
            "requested_resource_types": ["mindmap", "quiz"],
            "requested_resource_type": "",
        }
    )

    assert result["requested_resource_type"] == "mindmap"
    assert result["requested_resource_types"] == ["mindmap", "quiz"]
    assert result["resource_generation_status"] == "running"
    assert result["resource_generation_plan"]["tasks"] == [
        {
            "task_id": "resource:mindmap",
            "resource_type": "mindmap",
            "status": "pending",
        },
        {"task_id": "resource:quiz", "resource_type": "quiz", "status": "pending"},
    ]


async def test_candidate_orchestrator_rejects_assignment_topic_fingerprint_tamper():
    state = {
        "request_id": "req-candidate-1",
        "requested_resource_types": ["mindmap"],
        "requested_resource_type": "",
        **_candidate_evidence_state("mindmap"),
    }
    state["resource_evidence_assignments"][0]["topic_ids"] = ["limits"]

    with pytest.raises(
        LearningGuidanceContractError,
        match="invalid_resource_evidence_assignments",
    ):
        await resource_orchestrator(state)


async def test_candidate_orchestrator_rejects_missing_version_marker():
    state = {
        "request_id": "req-candidate-1",
        "requested_resource_types": ["mindmap"],
        "requested_resource_type": "",
        **_candidate_evidence_state("mindmap"),
    }
    del state["resource_evidence_contract_version"]

    with pytest.raises(
        LearningGuidanceContractError,
        match="missing_resource_evidence_contract_version",
    ):
        await resource_orchestrator(state)


async def test_candidate_orchestrator_preserves_exact_assignment_binding():
    result = await resource_orchestrator(
        {
            "request_id": "req-candidate-success",
            "requested_resource_types": ["mindmap"],
            "requested_resource_type": "mindmap",
            **_candidate_evidence_state("mindmap"),
        }
    )

    assert result["resource_generation_plan"]["tasks"] == [
        {
            "task_id": "resource:mindmap",
            "resource_type": "mindmap",
            "subjects": ["math"],
            "topic_ids": ["functions"],
            "delivery_mode": "strict",
            "status": "pending",
        }
    ]


async def test_candidate_orchestrator_rejects_ready_resource_alias():
    with pytest.raises(
        LearningGuidanceContractError,
        match="invalid_candidate_ready_resources",
    ):
        await resource_orchestrator(
            {
                "request_id": "req-candidate-alias",
                "requested_resource_types": ["exercise"],
                "requested_resource_type": "exercise",
                **_candidate_evidence_state("quiz"),
            }
        )


async def test_candidate_worker_rejects_task_topic_mismatch_before_runner(
    monkeypatch,
):
    called = False

    async def forbidden_runner(_state):
        nonlocal called
        called = True
        raise AssertionError("mismatched task must not reach the resource runner")

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "mindmap", forbidden_runner)
    result = await resource_worker(
        {
            "request_id": "req-candidate-1",
            **_candidate_evidence_state("mindmap"),
            "context": [
                {
                    "evidence_id": "evidence:mindmap",
                    "content": "Approved functions evidence.",
                }
            ],
            "resource_task": {
                "task_id": "resource:mindmap",
                "resource_type": "mindmap",
                "subjects": ["math"],
                "topic_ids": ["limits"],
            },
        }
    )

    branch = result["resource_branch_results"][0]
    assert called is False
    assert branch["status"] == "failed"
    assert branch["error_type"] == "LearningGuidanceContractError"
    assert "resource_task_assignment_mismatch" in branch["error_message_sanitized"]


async def test_candidate_worker_rejects_resource_alias_before_runner(monkeypatch):
    called = False

    async def forbidden_runner(_state):
        nonlocal called
        called = True
        raise AssertionError("aliased task must not reach the resource runner")

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "quiz", forbidden_runner)
    result = await resource_worker(
        {
            "request_id": "req-candidate-2",
            **_candidate_evidence_state("quiz"),
            "context": [
                {
                    "evidence_id": "evidence:quiz",
                    "content": "Approved quiz evidence.",
                }
            ],
            "resource_task": {
                "task_id": "resource:quiz",
                "resource_type": "exercise",
                "subjects": ["math"],
                "topic_ids": ["functions"],
            },
        }
    )

    branch = result["resource_branch_results"][0]
    assert called is False
    assert branch["status"] == "failed"
    assert "noncanonical_resource_task_type" in branch["error_message_sanitized"]


async def test_resource_preflight_routes_study_plan_requests_to_profile_gate():
    result = await resource_preflight_router(
        {
            "request_id": "req-1",
            "requested_resource_types": ["mindmap", "study_plan"],
            "requested_resource_type": "",
        }
    )

    assert result["requested_resource_types"] == ["mindmap", "study_plan"]
    assert result["resource_generation_status"] == "preflight"
    assert route_after_resource_preflight(result) == "study_plan_profile_gate_main"


def test_resource_preflight_routes_non_study_plan_to_orchestrator():
    assert (
        route_after_resource_preflight(
            {
                "requested_resource_types": ["mindmap", "quiz"],
                "requested_resource_type": "",
            }
        )
        == "resource_orchestrator"
    )


def test_dispatch_resource_workers_returns_send_packets():
    sends = dispatch_resource_workers(
        {
            "resource_generation_plan": {
                "tasks": [
                    {"task_id": "resource:mindmap", "resource_type": "mindmap"},
                    {"task_id": "resource:quiz", "resource_type": "quiz"},
                ]
            }
        }
    )

    assert all(isinstance(send, Send) for send in sends)
    assert [send.node for send in sends] == ["resource_worker", "resource_worker"]
    assert [send.arg["resource_task"]["resource_type"] for send in sends] == [
        "mindmap",
        "quiz",
    ]


def test_candidate_dispatch_routes_empty_plan_to_recommendation_aggregator():
    sends = dispatch_resource_workers_to_recommendation_aggregator(
        {"resource_generation_plan": {"tasks": []}}
    )

    assert len(sends) == 1
    assert sends[0].node == "resource_bundle_aggregator"


async def test_candidate_compiled_graph_fans_in_two_sends_before_one_final():
    calls = {
        "worker": 0,
        "aggregator": 0,
        "recommendation": 0,
        "finalizer": 0,
    }
    worker_resource_types: list[str] = []
    emitted_resource_finals: list[dict] = []

    async def fake_orchestrator(_state: LearningState) -> dict:
        return {
            "resource_generation_plan": {
                "tasks": [
                    {
                        "task_id": "resource:mindmap",
                        "resource_type": "mindmap",
                        "subjects": ["math"],
                        "topic_ids": ["functions"],
                        "status": "pending",
                    },
                    {
                        "task_id": "resource:quiz",
                        "resource_type": "quiz",
                        "subjects": ["math"],
                        "topic_ids": ["functions"],
                        "status": "pending",
                    },
                ]
            },
            "resource_generation_status": "running",
        }

    async def fake_worker(state: LearningState) -> dict:
        calls["worker"] += 1
        resource_type = state["resource_task"]["resource_type"]
        worker_resource_types.append(resource_type)
        branch = (
            _mindmap_branch_result()
            if resource_type == "mindmap"
            else _failed_branch_result("quiz")
        )
        branch["subjects"] = list(state["resource_task"]["subjects"])
        branch["topic_ids"] = list(state["resource_task"]["topic_ids"])
        return {"resource_branch_results": [branch]}

    async def counting_aggregator(state: LearningState) -> dict:
        calls["aggregator"] += 1
        return await resource_bundle_aggregator(state)

    async def fake_recommendation(state: LearningState) -> dict:
        calls["recommendation"] += 1
        contexts = state["recommendation_resource_context"]
        assert len(contexts) == 1
        assert {
            key: contexts[0][key]
            for key in ("resource_type", "subject", "topic_id", "title")
        } == {
            "resource_type": "mindmap",
            "subject": "math",
            "topic_id": "functions",
            "title": "Mock Map",
        }
        return {
            "resource_recommendation_output": _automatic_recommendation_output(
                contexts[0]["resource_id"]
            )
        }

    async def counting_finalizer(state: LearningState) -> dict:
        calls["finalizer"] += 1
        output = await resource_bundle_output_with_recommendations(
            state,
            runtime=GUIDANCE_RUNTIME,
        )
        emitted_resource_finals.append(output["resource_final_v3"])
        return output

    compiled = _compile_candidate_resource_test_graph(
        orchestrator=fake_orchestrator,
        worker=fake_worker,
        aggregator=counting_aggregator,
        recommendation=fake_recommendation,
        finalizer=counting_finalizer,
    )
    result = await compiled.ainvoke(
        {
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "request_id": "request-1",
            "user_id": "learner-1",
            "subject": "math",
            "evidence_requested_subjects": ["math"],
            **_candidate_evidence_state("mindmap", "quiz"),
            "requested_resource_types": ["mindmap", "quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [],
        }
    )

    assert calls == {
        "worker": 2,
        "aggregator": 1,
        "recommendation": 1,
        "finalizer": 1,
    }
    assert sorted(worker_resource_types) == ["mindmap", "quiz"]
    assert len(emitted_resource_finals) == 1
    parsed = ResourceFinalV3.model_validate_json(
        json.dumps(result["resource_final_v3"], ensure_ascii=False)
    )
    assert parsed.terminal_status == "partial_success"
    assert [resource.kind for resource in parsed.resources] == ["mindmap"]
    assert [error.resource_type for error in parsed.errors] == ["quiz"]
    assert len(parsed.recommendations) == 1
    assert parsed.payload_hash == emitted_resource_finals[0]["payload_hash"]


async def test_candidate_compiled_graph_empty_plan_reaches_controlled_stop_once():
    calls = {
        "worker": 0,
        "aggregator": 0,
        "recommendation": 0,
        "finalizer": 0,
    }

    async def fake_orchestrator(_state: LearningState) -> dict:
        return {
            "resource_generation_plan": {"tasks": []},
            "resource_generation_status": "skipped",
        }

    async def forbidden_worker(_state: LearningState) -> dict:
        calls["worker"] += 1
        raise AssertionError("an empty resource plan must not execute a worker")

    async def counting_aggregator(state: LearningState) -> dict:
        calls["aggregator"] += 1
        return await resource_bundle_aggregator(state)

    async def unavailable_recommendation(state: LearningState) -> dict:
        calls["recommendation"] += 1
        assert state["recommendation_resource_context"] == []
        return {
            "resource_recommendation_output": _automatic_recommendation_unavailable(
                "generated_resources_unavailable"
            )
        }

    async def counting_finalizer(state: LearningState) -> dict:
        calls["finalizer"] += 1
        return await resource_bundle_output_with_recommendations(
            state,
            runtime=GUIDANCE_RUNTIME,
        )

    compiled = _compile_candidate_resource_test_graph(
        orchestrator=fake_orchestrator,
        worker=forbidden_worker,
        aggregator=counting_aggregator,
        recommendation=unavailable_recommendation,
        finalizer=counting_finalizer,
    )
    result = await compiled.ainvoke(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "user_id": "learner-1",
            "subject": "math",
            "evidence_requested_subjects": ["math"],
            **_candidate_evidence_state(
                "quiz",
                assigned_resource_types=(),
            ),
            "requested_resource_types": ["quiz"],
            "resource_evidence_readiness": [
                {
                    "resource_type": "quiz",
                    "readiness_state": "blocked_insufficient_evidence",
                    "blocked_requirement_ids": ["requirement-quiz"],
                    "reason_code": "required_evidence_incomplete",
                }
            ],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [],
        }
    )

    assert calls == {
        "worker": 0,
        "aggregator": 1,
        "recommendation": 1,
        "finalizer": 1,
    }
    parsed = ResourceFinalV3.model_validate_json(
        json.dumps(result["resource_final_v3"], ensure_ascii=False)
    )
    assert parsed.terminal_status == "controlled_stop"
    assert [item.resource_type for item in parsed.blocked_resources] == ["quiz"]
    assert parsed.recommendations == ()


async def test_candidate_compiled_graph_recommendation_error_is_fail_fast():
    calls = {
        "worker": 0,
        "aggregator": 0,
        "recommendation": 0,
        "finalizer": 0,
    }

    async def fake_orchestrator(_state: LearningState) -> dict:
        return {
            "resource_generation_plan": {
                "tasks": [
                    {
                        "task_id": "resource:mindmap",
                        "resource_type": "mindmap",
                        "subjects": ["math"],
                        "topic_ids": ["functions"],
                        "status": "pending",
                    }
                ]
            },
            "resource_generation_status": "running",
        }

    async def fake_worker(state: LearningState) -> dict:
        calls["worker"] += 1
        branch = _mindmap_branch_result()
        branch["subjects"] = list(state["resource_task"]["subjects"])
        branch["topic_ids"] = list(state["resource_task"]["topic_ids"])
        return {"resource_branch_results": [branch]}

    async def counting_aggregator(state: LearningState) -> dict:
        calls["aggregator"] += 1
        return await resource_bundle_aggregator(state)

    async def failing_recommendation(_state: LearningState) -> dict:
        calls["recommendation"] += 1
        raise RuntimeError("recommendation execution failed")

    async def forbidden_finalizer(_state: LearningState) -> dict:
        calls["finalizer"] += 1
        raise AssertionError("a recommendation error must not reach finalization")

    compiled = _compile_candidate_resource_test_graph(
        orchestrator=fake_orchestrator,
        worker=fake_worker,
        aggregator=counting_aggregator,
        recommendation=failing_recommendation,
        finalizer=forbidden_finalizer,
    )
    with pytest.raises(RuntimeError, match="recommendation execution failed"):
        await compiled.ainvoke(
            {
                "thread_id": "thread-1",
                "request_id": "request-1",
                "user_id": "learner-1",
                "subject": "math",
                "evidence_requested_subjects": ["math"],
                **_candidate_evidence_state("mindmap"),
                "requested_resource_types": ["mindmap"],
                "resource_generation_debug": {"stages": []},
                "resource_branch_results": [],
            }
        )

    assert calls == {
        "worker": 1,
        "aggregator": 1,
        "recommendation": 1,
        "finalizer": 0,
    }


async def test_candidate_aggregator_then_finalizer_emits_one_recommended_v3():
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("mindmap"),
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [_mindmap_branch_result()],
    }

    aggregate = await resource_bundle_aggregator(state)

    assert set(aggregate) == {"recommendation_resource_context"}
    assert "messages" not in aggregate
    assert "resource_final_v3" not in aggregate
    context = aggregate["recommendation_resource_context"][0]
    resource_id = context["resource_id"]
    assert {
        key: context[key] for key in ("resource_type", "subject", "topic_id", "title")
    } == {
        "resource_type": "mindmap",
        "subject": "math",
        "topic_id": "functions",
        "title": "Mock Map",
    }
    baseline = await resource_bundle_output(state)
    final = await resource_bundle_output_with_recommendations(
        {
            **state,
            **aggregate,
            "resource_recommendation_output": _automatic_recommendation_output(
                resource_id
            ),
        },
        runtime=GUIDANCE_RUNTIME,
    )
    parsed = ResourceFinalV3.model_validate_json(
        json.dumps(final["resource_final_v3"], ensure_ascii=False)
    )

    assert parsed.resources[0].resource_id == resource_id
    assert baseline["resource_final_v3"]["resources"][0]["resource_id"] == resource_id
    assert baseline["resource_final_v3"]["payload_hash"] != parsed.payload_hash
    assert len(parsed.recommendations) == 1
    assert parsed.recommendations[0].trigger == "automatic"
    assert parsed.recommendations[0].recommendation_id == (
        "recommendation-math-mindmap"
    )
    assert parsed.recommendations[0].resource_id == resource_id
    assert final["messages"][0].content.startswith("个性化推荐已生成（1 项）。")

    tampered_final = json.loads(
        json.dumps(final["resource_final_v3"], ensure_ascii=False)
    )
    tampered_final["recommendations"][0]["resource_id"] = "resource-not-generated"
    with pytest.raises(
        ValueError,
        match="automatic recommendation must target a generated resource",
    ):
        ResourceFinalV3.model_validate_json(
            json.dumps(tampered_final, ensure_ascii=False)
        )


async def test_candidate_finalizer_keeps_unavailable_status_public_and_empty():
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("quiz"),
        "requested_resource_types": ["quiz"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [_failed_branch_result("quiz")],
    }
    aggregate = await resource_bundle_aggregator(state)
    final = await resource_bundle_output_with_recommendations(
        {
            **state,
            **aggregate,
            "resource_recommendation_output": (
                _automatic_recommendation_unavailable("generated_resources_unavailable")
            ),
        },
        runtime=GUIDANCE_RUNTIME,
    )

    assert final["resource_final_v3"]["recommendations"] == []
    assert final["resource_final_v3"]["summary"].startswith(
        "个性化推荐暂不可用 [generated_resources_unavailable]"
    )


async def test_candidate_finalizer_rejects_tampered_recommendation_source():
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("mindmap"),
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [_mindmap_branch_result()],
    }
    aggregate = await resource_bundle_aggregator(state)

    with pytest.raises(
        RuntimeError,
        match="unknown_recommendation_resource_evidence",
    ):
        await resource_bundle_output_with_recommendations(
            {
                **state,
                **aggregate,
                "resource_recommendation_output": (
                    _automatic_recommendation_output("resource-not-in-bundle")
                ),
            },
            runtime=GUIDANCE_RUNTIME,
        )


async def test_candidate_finalizer_rejects_tampered_bundle_context():
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("mindmap"),
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [_mindmap_branch_result()],
    }
    aggregate = await resource_bundle_aggregator(state)
    aggregate["recommendation_resource_context"][0]["resource_id"] = "resource-tampered"

    with pytest.raises(
        RuntimeError,
        match="recommendation_resource_context_mismatch",
    ):
        await resource_bundle_output_with_recommendations(
            {
                **state,
                **aggregate,
                "resource_recommendation_output": (
                    _automatic_recommendation_output("resource-tampered")
                ),
            },
            runtime=GUIDANCE_RUNTIME,
        )


async def test_candidate_aggregator_rejects_branch_topic_tamper():
    branch = _mindmap_branch_result()
    branch["topic_ids"] = ["limits"]
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("mindmap"),
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [branch],
    }

    with pytest.raises(
        LearningGuidanceContractError,
        match="resource_branch_assignment_mismatch",
    ):
        await resource_bundle_aggregator(state)


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    (
        ("resource_type", "quiz"),
        ("topic_id", "limits"),
        ("title", "Tampered title"),
    ),
)
async def test_candidate_finalizer_rejects_recommendation_target_binding_tamper(
    field_name: str,
    tampered_value: str,
):
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("mindmap"),
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [_mindmap_branch_result()],
    }
    aggregate = await resource_bundle_aggregator(state)
    resource_id = aggregate["recommendation_resource_context"][0]["resource_id"]
    output = _automatic_recommendation_output(resource_id)
    output["batch"]["items"][0][field_name] = tampered_value

    with pytest.raises(
        LearningGuidanceContractError,
        match="recommendation_target_binding_mismatch",
    ):
        await resource_bundle_output_with_recommendations(
            {
                **state,
                **aggregate,
                "resource_recommendation_output": output,
            },
            runtime=GUIDANCE_RUNTIME,
        )


async def test_candidate_finalizer_rejects_changed_guidance_runtime():
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "user_id": "learner-1",
        "subject": "math",
        "evidence_requested_subjects": ["math"],
        **_candidate_evidence_state("mindmap"),
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
        "resource_branch_results": [_mindmap_branch_result()],
    }
    aggregate = await resource_bundle_aggregator(state)
    resource_id = aggregate["recommendation_resource_context"][0]["resource_id"]

    with pytest.raises(RuntimeError, match="learning_guidance_runtime_mismatch"):
        await resource_bundle_output_with_recommendations(
            {
                **state,
                **aggregate,
                "resource_recommendation_output": (
                    _automatic_recommendation_output(resource_id)
                ),
            },
            runtime=replace(GUIDANCE_RUNTIME, runtime_fingerprint="8" * 64),
        )


async def test_resource_worker_success_uses_runner_without_writing_messages(
    monkeypatch,
):
    async def fake_quiz_runner(local_state):
        local_state.update(
            _quiz_binding(
                thread_id=local_state["thread_id"],
                request_id=local_state["request_id"],
            )
        )
        return "## Mock Quiz\n\nQuestion body"

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "quiz", fake_quiz_runner)

    result = await resource_worker(
        {
            "messages": [HumanMessage(content="make a quiz")],
            "thread_id": "thread-1",
            "request_id": "request-1",
            "resource_task": {"task_id": "resource:quiz", "resource_type": "quiz"},
        }
    )

    branch = result["resource_branch_results"][0]
    assert branch["resource_type"] == "quiz"
    assert branch["status"] == "success"
    assert branch["artifact"]["title"] == "Mock Quiz"
    assert branch["state_updates"]["exercise_items"] == [_public_quiz_item()]
    assert "assessment_checkpoint_resources" not in branch["state_updates"]
    assert result["assessment_checkpoint_resources"]["thread_id"] == "thread-1"
    assert "SERVER_ONLY_ANSWER" not in str(branch)
    assert "messages" not in result


async def test_resource_worker_rejects_quiz_without_v3_checkpoint_binding(
    monkeypatch,
):
    async def incomplete_quiz_runner(local_state):
        public = _public_quiz_item()
        local_state["exercise_items"] = [public]
        local_state["exercise_artifact"] = {
            "title": "Incomplete Quiz",
            "items": [public],
        }
        return "## Incomplete Quiz"

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "quiz", incomplete_quiz_runner)

    result = await resource_worker(
        {
            "messages": [HumanMessage(content="make a quiz")],
            "thread_id": "thread-1",
            "request_id": "request-1",
            "resource_task": {"task_id": "resource:quiz", "resource_type": "quiz"},
        }
    )

    branch = result["resource_branch_results"][0]
    assert branch["status"] == "failed"
    assert branch["error_type"] == "AssessmentQuizProjectionError"
    assert "assessment_checkpoint_resources" not in result


async def test_resource_bundle_output_does_not_copy_private_quiz_checkpoint():
    binding = _quiz_binding(thread_id="thread-1", request_id="request-1")
    branch = {
        "resource_type": "quiz",
        "subjects": ["math"],
        "topic_ids": ["functions"],
        "status": "success",
        "title": "Mock Quiz",
        "artifact": binding["exercise_artifact"],
        "artifacts": [],
        "state_updates": {
            key: value
            for key, value in binding.items()
            if key != "assessment_checkpoint_resources"
        },
        "message_content": "Generated quiz: Mock Quiz",
        "message_preview": "Generated quiz: Mock Quiz",
        "error_type": None,
        "error_message_sanitized": None,
        "elapsed_ms": 10,
        "validation": {
            "schema_version": "resource_validation_v1",
            "resource_type": "quiz",
            "valid": True,
            "terminal_status": "success",
            "renderable_count": 1,
            "downloadable_count": 0,
            "verified_local_count": 0,
            "remote_unverified_count": 0,
            "failure_reason": "",
            "warnings": [],
        },
    }
    result = await resource_bundle_output(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "requested_resource_types": ["quiz"],
            "resource_generation_debug": {"stages": []},
            "assessment_checkpoint_resources": binding[
                "assessment_checkpoint_resources"
            ],
            "resource_branch_results": [branch],
        }
    )

    public_surface = {
        **result,
        "messages": [message.content for message in result.get("messages", [])],
    }
    public_json = json.dumps(public_surface, ensure_ascii=False, sort_keys=True)
    assert "SERVER_ONLY_ANSWER" not in public_json
    assert "assessment_checkpoint_resources" not in result

    aggregate = await resource_bundle_aggregator(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "user_id": "learner-1",
            "subject": "math",
            "evidence_requested_subjects": ["math"],
            **_candidate_evidence_state("quiz"),
            "requested_resource_types": ["quiz"],
            "assessment_checkpoint_resources": binding[
                "assessment_checkpoint_resources"
            ],
            "resource_branch_results": [branch],
        }
    )
    assert "SERVER_ONLY_ANSWER" not in json.dumps(aggregate, ensure_ascii=False)
    assert set(aggregate) == {"recommendation_resource_context"}
    context = aggregate["recommendation_resource_context"][0]
    assert {
        key: context[key] for key in ("resource_type", "subject", "topic_id", "title")
    } == {
        "resource_type": "quiz",
        "subject": "math",
        "topic_id": "functions",
        "title": "Mock Quiz",
    }


async def test_resource_worker_emits_safe_internal_subnode_traces(monkeypatch):
    async def fake_node(local_state):
        local_state["exercise_items"] = [_public_quiz_item()]
        local_state["exercise_artifact"] = {"title": "Mock Quiz"}
        return {}

    async def fake_output(local_state):
        return {
            **_quiz_binding(
                thread_id=local_state["thread_id"],
                request_id=local_state["request_id"],
            ),
            "messages": [AIMessage(content="## Mock Quiz")],
        }

    monkeypatch.setattr(rg, "exercise_planner", fake_node)
    monkeypatch.setattr(rg, "exercise_agent", fake_node)
    monkeypatch.setattr(rg, "exercise_reviewer", fake_node)
    monkeypatch.setattr(rg, "exercise_output", fake_output)
    monkeypatch.setattr(rg, "should_rewrite_exercise", lambda _state: "output")

    trace_events: list[dict] = []
    sink_token = set_trace_event_sink(trace_events)
    try:
        result = await resource_worker(
            {
                "messages": [HumanMessage(content="make a quiz")],
                "resource_task": {"task_id": "resource:quiz", "resource_type": "quiz"},
                "request_id": "req-1",
                "thread_id": "thread-1",
            }
        )
    finally:
        reset_trace_event_sink(sink_token)

    assert result["resource_branch_results"][0]["status"] == "success"
    subnode_events = [
        event
        for event in trace_events
        if str(event.get("stage", "")).startswith("resource_subnode.")
    ]
    assert [event["stage"] for event in subnode_events] == [
        "resource_subnode.start",
        "resource_subnode.end",
        "resource_subnode.start",
        "resource_subnode.end",
        "resource_subnode.start",
        "resource_subnode.end",
        "resource_subnode.start",
        "resource_subnode.end",
    ]
    assert {event["subnode"] for event in subnode_events} == {
        "exercise_planner",
        "exercise_agent",
        "exercise_reviewer",
        "exercise_output",
    }
    assert all(event["resource_type"] == "quiz" for event in subnode_events)
    assert all("messages" not in event for event in subnode_events)
    assert all("content" not in event for event in subnode_events)


async def test_resource_worker_failure_is_captured(monkeypatch):
    async def failing_runner(_local_state):
        raise RuntimeError("quiz failed")

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "quiz", failing_runner)

    result = await resource_worker(
        {
            "messages": [HumanMessage(content="make a quiz")],
            "resource_task": {"task_id": "resource:quiz", "resource_type": "quiz"},
        }
    )

    branch = result["resource_branch_results"][0]
    assert branch["resource_type"] == "quiz"
    assert branch["status"] == "failed"
    assert branch["error_type"] == "RuntimeError"
    assert "quiz failed" in branch["error_message_sanitized"]


async def test_empty_timeout_error_builds_typed_failed_resource_final(monkeypatch):
    async def timed_out_runner(_local_state):
        raise TimeoutError()

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "review_doc", timed_out_runner)

    worker_result = await resource_worker(
        {
            "messages": [HumanMessage(content="make a review document")],
            "thread_id": "thread-1",
            "request_id": "request-1",
            "resource_task": {
                "task_id": "resource:review_doc",
                "resource_type": "review_doc",
            },
        }
    )

    branch = worker_result["resource_branch_results"][0]
    assert branch["status"] == "failed"
    assert branch["error_type"] == "TimeoutError"
    assert branch["error_message_sanitized"] == (
        "Resource generation failed with TimeoutError."
    )

    final = await resource_bundle_output(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "requested_resource_types": ["review_doc"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [branch],
        }
    )

    assert final["resource_generation_status"] == "failed"
    assert final["resource_final_v3"]["terminal_status"] == "failed"
    assert final["resource_final_v3"]["errors"] == [
        {
            "resource_type": "review_doc",
            "error_code": "review_doc.generation_failed",
            "error_type": "TimeoutError",
            "message_sanitized": "Resource generation failed with TimeoutError.",
        }
    ]


async def test_study_plan_worker_fails_if_profile_gate_was_bypassed(monkeypatch):
    async def fake_study_plan_runner(_local_state):
        raise AssertionError("runner must not be called")

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "study_plan", fake_study_plan_runner)

    result = await resource_worker(
        {
            "messages": [HumanMessage(content="make a study plan")],
            "resource_task": {
                "task_id": "resource:study_plan",
                "resource_type": "study_plan",
            },
            "thread_id": "thread-1",
            "request_id": "request-1",
        }
    )

    branch = result["resource_branch_results"][0]
    assert branch["resource_type"] == "study_plan"
    assert branch["status"] == "failed"
    assert branch["error_type"] == "RuntimeError"
    assert (
        "study_plan_profile_missing_after_preflight"
        in branch["error_message_sanitized"]
    )


async def test_resource_worker_reraises_graph_interrupt(monkeypatch):
    async def interrupting_runner(_local_state):
        raise GraphInterrupt(())

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "study_plan", interrupting_runner)

    with pytest.raises(GraphInterrupt):
        await resource_worker(
            {
                "messages": [HumanMessage(content="make a study plan")],
                "resource_task": {
                    "task_id": "resource:study_plan",
                    "resource_type": "study_plan",
                },
                "learner_profile": {
                    "learning_goal": "Master ML",
                    "current_foundation": "Python",
                    "daily_study_time": "2 hours",
                },
            }
        )


async def test_resource_bundle_output_partial_success_sets_artifacts_and_message():
    result = await resource_bundle_output(
        {
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "request_id": "request-1",
            "subject": "math",
            "learning_goal": "review functions",
            "requested_resource_types": ["mindmap", "quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [
                _mindmap_branch_result(),
                _failed_branch_result("quiz"),
            ],
        }
    )

    assert result["resource_generation_status"] == "partial_success"
    assert result["resource_bundle_artifact"]["success_count"] == 1
    assert result["resource_bundle_artifact"]["failed_count"] == 1
    assert result["resource_final_v3"]["schema_version"] == "resource_final_v3"
    assert result["resource_final_v3"]["terminal_status"] == "partial_success"
    assert len(result["resource_final_v3"]["resources"]) == 1
    assert len(result["resource_final_v3"]["errors"]) == 1
    assert result["mindmap_artifact"]["title"] == "Mock Map"
    assert result["resource_artifacts_by_type"]["mindmap"]["title"] == "Mock Map"
    assert len(result["last_generated_artifacts"]) == 1
    assert len(result["task_workspace"]["artifacts_by_id"]) == 1
    assert result["task_workspace"]["latest_artifact_by_resource_type"]["mindmap"]
    assert isinstance(result["messages"][0], AIMessage)
    assert "部分资源已生成" in result["messages"][0].content


async def test_resource_bundle_output_all_failed_returns_failed_bundle():
    result = await resource_bundle_output(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "requested_resource_types": ["quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [_failed_branch_result("quiz")],
        }
    )

    assert result["resource_generation_status"] == "failed"
    assert result["resource_bundle_artifact"]["status"] == "failed"
    assert result["resource_final_v3"]["terminal_status"] == "failed"
    assert result["resource_final_v3"]["resources"] == []
    assert "所有请求的学习资源都生成失败" in result["messages"][0].content


async def test_resource_bundle_fan_in_persists_branch_influences_idempotently():
    state = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "subject": "math",
        "requested_resource_types": ["mindmap"],
        "resource_generation_debug": {"stages": []},
    }
    entry = build_influence_entry(
        state=state,
        kind="planner_output",
        source_node="mindmap_planner",
        title="Mindmap plan",
        preview="Functions and derivatives.",
        metadata={"workflow": "mindmap", "iteration": 1},
    )
    state["resource_branch_results"] = [
        _mindmap_branch_result(influence_entries=[entry, entry])
    ]

    result = await resource_bundle_output(state)
    once = merge_context_influence_ledger({}, result["context_influence_ledger"])
    twice = merge_context_influence_ledger(once, result["context_influence_ledger"])

    assert len(result["context_influence_ledger"]["entries"]) == 1
    assert once["ordered_ids"] == twice["ordered_ids"]
    assert twice["counts_by_kind"] == {"planner_output": 1}


def test_resource_summary_includes_review_doc_metrics():
    summary = rg._resource_summary(
        {
            "resource_type": "review_doc",
            "status": "success",
            "title": "Review",
            "artifact": {"markdown_url": "/r.md", "docx_url": "/r.docx"},
            "artifacts": [{"filename": "r.md"}, {"filename": "r.docx"}],
            "state_updates": {"review_doc_markdown": "hello"},
        }
    )

    assert summary["metrics"]["artifact_count"] == 2
    assert summary["metrics"]["markdown_chars"] == 5
    assert summary["metrics"]["has_markdown"] is True
    assert summary["metrics"]["has_docx"] is True


def test_resource_summary_includes_mindmap_metrics_and_safe_node_count():
    tree = {
        "title": "Root",
        "children": [
            {"title": "A", "children": [{"title": "A1"}]},
            {"title": "B"},
        ],
    }

    summary = rg._resource_summary(
        {
            "resource_type": "mindmap",
            "status": "success",
            "title": "Map",
            "artifact": {"tree": tree, "xmind_url": "/m.xmind"},
            "state_updates": {},
        }
    )

    assert summary["metrics"]["node_count"] == 4
    assert summary["metrics"]["has_xmind"] is True
    assert rg._count_mindmap_nodes(None) == 0
    assert (
        rg._count_mindmap_nodes([{"title": "A"}, {"children": [{"title": "B"}]}]) == 3
    )
    assert rg._count_mindmap_nodes("not a tree") == 0


def test_resource_summary_includes_quiz_metrics():
    summary = rg._resource_summary(
        {
            "resource_type": "quiz",
            "status": "success",
            "title": "Quiz",
            "artifact": {"markdown_url": "/q.md", "docx_filename": "q.docx"},
            "state_updates": {
                "exercise_items": [{"question": "Q1"}, {"question": "Q2"}]
            },
        }
    )

    assert summary["metrics"]["item_count"] == 2
    assert summary["metrics"]["has_markdown"] is True
    assert summary["metrics"]["has_docx"] is True


def test_resource_summary_includes_study_plan_metrics_without_fixed_shape():
    summary = rg._resource_summary(
        {
            "resource_type": "study_plan",
            "status": "success",
            "title": "Plan",
            "artifact": {
                "title": "Plan",
                "document": {
                    "title": "Plan",
                    "markdown_url": "/p.md",
                    "docx_filename": "p.docx",
                },
            },
            "state_updates": {"study_plan_markdown": "# Plan"},
        }
    )
    empty_summary = rg._resource_summary(
        {"resource_type": "study_plan", "artifact": {}, "state_updates": {}}
    )

    assert summary["metrics"]["has_document"] is True
    assert summary["metrics"]["has_markdown"] is True
    assert summary["metrics"]["has_docx"] is True
    assert empty_summary["metrics"] == {
        "has_markdown": False,
        "has_docx": False,
        "has_document": False,
    }


async def test_resource_bundle_output_stores_message_on_bundle_artifact():
    quiz_binding = _quiz_binding(thread_id="thread-1", request_id="request-1")
    result = await resource_bundle_output(
        {
            "thread_id": "thread-1",
            "request_id": "request-1",
            "requested_resource_types": ["mindmap", "quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [
                _mindmap_branch_result(),
                {
                    "resource_type": "quiz",
                    "status": "success",
                    "title": "Mock Quiz",
                    "artifact": quiz_binding["exercise_artifact"],
                    "artifacts": [],
                    "state_updates": quiz_binding,
                    "message_content": "Generated quiz: Mock Quiz",
                    "message_preview": "Generated quiz: Mock Quiz",
                    "elapsed_ms": 10,
                    "validation": quiz_binding["exercise_resource_v3"]["validation"],
                },
            ],
        }
    )

    message = result["resource_bundle_artifact"]["message"]
    assert "# 已生成多类学习资源" in message
    assert "### 思维导图" in message
    assert "### 练习题" in message
    assert result["messages"][0].content == message
    assert result["resource_final_v3"]["terminal_status"] == "success"
    assert [
        resource["kind"] for resource in result["resource_final_v3"]["resources"]
    ] == ["mindmap", "quiz"]


def test_compose_bundle_message_single_success_preserves_resource_message():
    message = rg._compose_bundle_message(
        "success",
        [
            {
                "resource_type": "quiz",
                "status": "success",
                "message_content": "single quiz message",
            }
        ],
        [],
    )

    assert message == "single quiz message"


def test_no_legacy_multi_resource_runtime_references():
    project_root = Path(__file__).resolve().parent.parent
    runtime_files = [
        project_root / "app.py",
        project_root / "src" / "graph" / "builder.py",
        project_root / "src" / "graph" / "state.py",
        project_root / "src" / "graph" / "supervisor.py",
        project_root / "src" / "graph" / "resource_generation.py",
        project_root / "src" / "graph" / "__init__.py",
    ]
    legacy_tokens = [
        "multi_resource_runner",
        "multi_resource_mode",
        "multi_resource_results",
        "multi_resource_summary",
        '"multi_resource"',
    ]

    assert not (project_root / "src" / "graph" / "multi_resource.py").exists()
    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        for token in legacy_tokens:
            assert token not in text, f"{token} remains in {path}"
