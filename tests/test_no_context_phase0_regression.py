"""Phase 0 regression tests for existing SSE and context-usage behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.stream_draft_helpers import draft_payloads
from src.assessment.identity import stable_exercise_question_id


class AsyncIteratorMock:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _payloads(collected) -> list[dict]:
    return draft_payloads(collected)


def _snapshot(values: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(next=(), tasks=[], values=values or {})


def _node_start(node_name: str) -> dict:
    return {
        "event": "on_chain_start",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"input": {}},
    }


def _node_end(node_name: str, output: dict | None = None) -> dict:
    return {
        "event": "on_chain_end",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output or {}},
    }


def _usage(node_name: str) -> dict:
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {
            "output": SimpleNamespace(
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }
            )
        },
    }


def _public_exercise_card() -> dict:
    question = "What does a Python list store?"
    tags = ("python", "collections")
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
        "tags": list(tags),
    }


@pytest.mark.anyio
async def test_stream_keeps_node_usage_resource_final_events(monkeypatch):
    from uuid import UUID

    from app import generate_stream_drafts
    from src.graph.resource_final_v3 import (
        ResourceFinalV3ResourceValidation,
        ResourceFinalV3Validation,
        build_resource_final_v3,
        build_resource_final_v3_resource,
    )

    request_id = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setattr("app.uuid.uuid4", lambda: UUID(request_id))
    public_card = _public_exercise_card()
    resource_validation = ResourceFinalV3ResourceValidation(
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
    )
    quiz_resource = build_resource_final_v3_resource(
        thread_id="thread-1",
        request_id=request_id,
        kind="quiz",
        status="success",
        title="Python quiz",
        summary="Python quiz ready",
        payload={
            "exercise_artifact": {
                "schema_version": "exercise_public_artifact_v1",
                "title": "Python quiz",
                "items": [public_card],
            },
            "exercise_items": [public_card],
        },
        artifact_refs={},
        validation=resource_validation,
    )
    resource_final = build_resource_final_v3(
        thread_id="thread-1",
        request_id=request_id,
        terminal_status="success",
        resources=(quiz_resource,),
        recommendations=(),
        blocked_resources=(),
        errors=(),
        validation=ResourceFinalV3Validation(
            schema_version="resource_final_validation_v3",
            resource_count=1,
            success_count=1,
            partial_success_count=0,
            failed_count=0,
            blocked_count=0,
            renderable_count=1,
            downloadable_count=0,
        ),
        summary="Python quiz ready",
    )

    final_state = {
        "thread_id": "thread-1",
        "request_id": request_id,
        "requested_resource_type": "quiz",
        "messages": [SimpleNamespace(content="quiz ready")],
        "exercise_items": [public_card],
        "exercise_artifact": {"title": "Python quiz"},
        "resource_final_v3": resource_final.model_dump(mode="json"),
    }
    graph = MagicMock()
    graph._a3_activity_events_enabled = True
    graph._a3_node_ids = frozenset({"supervisor", "generate_answer"})
    graph.astream_events = MagicMock(
        return_value=AsyncIteratorMock(
            [
                _node_start("generate_answer"),
                _usage("generate_answer"),
                _node_end("generate_answer"),
            ]
        )
    )
    graph.aget_state = AsyncMock(return_value=_snapshot(final_state))
    graph.aupdate_state = AsyncMock()

    collected = []
    async for event in generate_stream_drafts("q", graph, thread_id="thread-1"):
        collected.append(event)

    payloads = _payloads(collected)
    assert any(
        payload.get("type") == "activity_event"
        and payload.get("kind") == "node"
        and payload.get("node") == "generate_answer"
        for payload in payloads
    )
    assert any(payload.get("type") == "usage" for payload in payloads)
    resource_events = [
        payload for payload in payloads if payload.get("type") == "resource_final"
    ]
    assert resource_events
    assert resource_events[0]["schema_version"] == "resource_final_v3"
    assert resource_events[0]["resources"][0]["kind"] == "quiz"
    assert payloads[-1]["type"] == "resource_final"


@pytest.mark.anyio
async def test_resume_keeps_basic_stream_path():
    from app import generate_resume_stream_drafts

    final_snapshot = _snapshot({"schema_version": "run_control_v1"})
    graph = MagicMock()
    graph._a3_activity_events_enabled = True
    graph._a3_node_ids = frozenset({"supervisor", "resource_bundle_output"})
    graph.astream_events = MagicMock(
        return_value=AsyncIteratorMock([_node_start("resource_bundle_output")])
    )
    graph.aget_state = AsyncMock(
        side_effect=[
            _snapshot({"schema_version": "run_control_v1"}),
            final_snapshot,
            final_snapshot,
        ]
    )
    graph.aupdate_state = AsyncMock()

    collected = []
    async for event in generate_resume_stream_drafts(
        "approved", None, graph, "thread-1"
    ):
        collected.append(event)

    payloads = _payloads(collected)
    assert payloads[0]["run_status"] == "continuing"
    assert any(
        payload.get("type") == "activity_event"
        and payload.get("kind") == "node"
        and payload.get("node") == "resource_bundle_output"
        and payload.get("status") == "running"
        for payload in payloads
    )
    assert not [payload for payload in payloads if payload.get("type") == "stream_done"]


def test_unknown_model_window_emits_error_when_context_engineering_is_non_strict(
    monkeypatch,
):
    import src.context_engineering.budget as budget
    from src.observability.context_usage import build_context_usage_payload

    def fake_get_setting(key, default=None):
        if key == "context_engineering":
            return {
                "enabled": True,
                "strict": False,
                "model_limits": {"deepseek-v4-pro": 1000000},
            }
        return default

    monkeypatch.setattr(budget, "get_setting", fake_get_setting)

    stage, payload = build_context_usage_payload(
        node_name="study_plan_agent",
        llm_node="study_plan",
        provider="deepseek_official",
        model="unknown-model",
        messages=[],
    )

    assert stage == "context_usage_error"
    assert payload is not None
    assert payload["reason"] == "model_window_unknown"
