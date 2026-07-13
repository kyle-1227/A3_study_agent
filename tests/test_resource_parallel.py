"""Tests for parallel learning-resource generation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Send

from src.assessment.attempt_contracts import AssessmentQuizSourceItemV1
from src.context_engineering.influence import (
    build_influence_entry,
    merge_context_influence_ledger,
)
from src.assessment.identity import stable_exercise_question_id
from src.graph import resource_generation as rg
from src.graph.assessment_quiz import build_assessment_quiz_projection_v1
from src.graph.resource_final_v3 import ResourceFinalV3ResourceValidation
from src.graph.resource_generation import (
    dispatch_resource_workers,
    normalize_requested_resource_types,
    resource_bundle_output,
    resource_orchestrator,
    resource_preflight_router,
    resource_worker,
    route_after_resource_preflight,
)
from src.graph.state import resource_branch_results_reducer
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


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
            "schema_version": "assessment_checkpoint_resources_v1",
            "thread_id": thread_id,
            "resources": [projection.checkpoint_resource.model_dump(mode="json")],
        },
    }


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
                {
                    "resource_type": "mindmap",
                    "status": "success",
                    "title": "Mock Map",
                    "artifact": {
                        "title": "Mock Map",
                        "tree": {"title": "Mock Map"},
                        "xmind_url": "/m.xmind",
                    },
                    "artifacts": [],
                    "state_updates": {
                        "mindmap_artifact": {
                            "title": "Mock Map",
                            "tree": {"title": "Mock Map"},
                            "xmind_url": "/m.xmind",
                        },
                        "mindmap_tree": {"title": "Mock Map"},
                    },
                    "message_content": "Generated mindmap: Mock Map",
                    "message_preview": "Generated mindmap: Mock Map",
                    "elapsed_ms": 10,
                },
                {
                    "resource_type": "quiz",
                    "status": "failed",
                    "title": "quiz",
                    "artifact": {},
                    "artifacts": [],
                    "state_updates": {},
                    "message_content": "",
                    "message_preview": "",
                    "error_type": "RuntimeError",
                    "error_message_sanitized": "quiz failed",
                    "elapsed_ms": 5,
                },
            ],
        }
    )

    assert result["resource_generation_status"] == "partial_success"
    assert result["resource_bundle_artifact"]["success_count"] == 1
    assert result["resource_bundle_artifact"]["failed_count"] == 1
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
            "requested_resource_types": ["quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [
                {
                    "resource_type": "quiz",
                    "status": "failed",
                    "title": "quiz",
                    "artifact": {},
                    "artifacts": [],
                    "state_updates": {},
                    "message_content": "",
                    "message_preview": "",
                    "error_type": "RuntimeError",
                    "error_message_sanitized": "quiz failed",
                    "elapsed_ms": 5,
                },
            ],
        }
    )

    assert result["resource_generation_status"] == "failed"
    assert result["resource_bundle_artifact"]["status"] == "failed"
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
        {
            "resource_type": "mindmap",
            "status": "success",
            "title": "Mock Map",
            "artifact": {"title": "Mock Map", "tree": {"title": "Mock Map"}},
            "artifacts": [],
            "state_updates": {"mindmap_tree": {"title": "Mock Map"}},
            "message_content": "Generated map",
            "message_preview": "Generated map",
            "elapsed_ms": 1,
            "context_influence_entries": [entry, entry],
        }
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
    result = await resource_bundle_output(
        {
            "requested_resource_types": ["mindmap", "quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [
                {
                    "resource_type": "mindmap",
                    "status": "success",
                    "title": "Mock Map",
                    "artifact": {"title": "Mock Map", "tree": {"title": "Mock Map"}},
                    "artifacts": [],
                    "state_updates": {"mindmap_tree": {"title": "Mock Map"}},
                    "message_content": "Generated mindmap: Mock Map",
                    "message_preview": "Generated mindmap: Mock Map",
                    "elapsed_ms": 10,
                },
                {
                    "resource_type": "quiz",
                    "status": "success",
                    "title": "Mock Quiz",
                    "artifact": {"title": "Mock Quiz"},
                    "artifacts": [],
                    "state_updates": {"exercise_items": [{"question": "Q1"}]},
                    "message_content": "Generated quiz: Mock Quiz",
                    "message_preview": "Generated quiz: Mock Quiz",
                    "elapsed_ms": 10,
                },
            ],
        }
    )

    message = result["resource_bundle_artifact"]["message"]
    assert "# 已生成多类学习资源" in message
    assert "### 思维导图" in message
    assert "### 练习题" in message
    assert result["messages"][0].content == message


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
