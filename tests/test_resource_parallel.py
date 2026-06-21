"""Tests for parallel learning-resource generation orchestration."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send

from src.graph import resource_generation as rg
from src.graph.resource_generation import (
    dispatch_resource_workers,
    normalize_requested_resource_types,
    resource_bundle_output,
    resource_orchestrator,
    resource_worker,
)
from src.graph.state import resource_branch_results_reducer


def test_normalize_requested_resource_types_dedupes_and_aliases():
    assert normalize_requested_resource_types(["mindmap", "exercise"], "quiz", "roadmap") == [
        "mindmap",
        "quiz",
        "study_plan",
    ]


def test_resource_branch_results_reducer_merges_by_resource_type():
    existing = [{"resource_type": "quiz", "status": "failed"}]
    update = [{"resource_type": "quiz", "status": "success"}, {"resource_type": "mindmap", "status": "success"}]

    merged = resource_branch_results_reducer(existing, update)

    assert [item["resource_type"] for item in merged] == ["mindmap", "quiz"]
    assert next(item for item in merged if item["resource_type"] == "quiz")["status"] == "success"


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
        {"task_id": "resource:mindmap", "resource_type": "mindmap", "status": "pending"},
        {"task_id": "resource:quiz", "resource_type": "quiz", "status": "pending"},
    ]


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
    assert [send.arg["resource_task"]["resource_type"] for send in sends] == ["mindmap", "quiz"]


async def test_resource_worker_success_uses_runner_without_writing_messages(monkeypatch):
    async def fake_quiz_runner(local_state):
        local_state["exercise_items"] = [{"question": "Q", "answer": "A"}]
        local_state["exercise_artifact"] = {"title": "Mock Quiz"}
        return "## Mock Quiz\n\nQuestion body"

    monkeypatch.setitem(rg.RESOURCE_RUNNERS, "quiz", fake_quiz_runner)

    result = await resource_worker(
        {
            "messages": [HumanMessage(content="make a quiz")],
            "resource_task": {"task_id": "resource:quiz", "resource_type": "quiz"},
        }
    )

    branch = result["resource_branch_results"][0]
    assert branch["resource_type"] == "quiz"
    assert branch["status"] == "success"
    assert branch["artifact"]["title"] == "Mock Quiz"
    assert branch["state_updates"]["exercise_items"] == [{"question": "Q", "answer": "A"}]
    assert "messages" not in result


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


async def test_resource_bundle_output_partial_success_sets_artifacts_and_message():
    result = await resource_bundle_output(
        {
            "requested_resource_types": ["mindmap", "quiz"],
            "resource_generation_debug": {"stages": []},
            "resource_branch_results": [
                {
                    "resource_type": "mindmap",
                    "status": "success",
                    "title": "Mock Map",
                    "artifact": {"title": "Mock Map", "tree": {"title": "Mock Map"}, "xmind_url": "/m.xmind"},
                    "artifacts": [],
                    "state_updates": {
                        "mindmap_artifact": {"title": "Mock Map", "tree": {"title": "Mock Map"}, "xmind_url": "/m.xmind"},
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
