"""Tests for parallel learning-resource generation orchestration."""

from __future__ import annotations

from pathlib import Path

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
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


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
    assert branch["state_updates"]["exercise_items"] == [
        {"question": "Q", "answer": "A"}
    ]
    assert "messages" not in result


async def test_resource_worker_emits_safe_internal_subnode_traces(monkeypatch):
    async def fake_node(local_state):
        local_state["exercise_items"] = [{"question": "Q", "answer": "A"}]
        local_state["exercise_artifact"] = {"title": "Mock Quiz"}
        return {}

    async def fake_output(_local_state):
        return {"messages": [AIMessage(content="## Mock Quiz")]}

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
