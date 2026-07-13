from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.context_engineering.compaction import (
    ConversationSummaryV2,
    FullCompactionConfigV1,
    build_compact_boundary,
)
from src.llm.compaction import (
    ConversationCompactionError,
    invoke_conversation_compaction,
)


def _config() -> FullCompactionConfigV1:
    return FullCompactionConfigV1(
        schema_version=1,
        enabled=True,
        retain_recent_rounds=1,
        summary_llm_node="conversation_compactor",
        output_mode="deepseek_tool_call_strict",
        max_summary_input_chars=128000,
    )


def _fixture():
    messages = [
        HumanMessage(content="old query"),
        AIMessage(content="old answer"),
        HumanMessage(content="current query"),
    ]
    boundary = build_compact_boundary(
        messages,
        thread_id="thread-1",
        request_id="request-2",
        trigger_dispatch_id="dispatch-1",
        retain_recent_rounds=1,
    )
    assert boundary is not None
    return messages, boundary


@pytest.mark.anyio
async def test_compaction_llm_uses_one_strict_validated_call(monkeypatch):
    from src.llm import compaction as compaction_module

    messages, boundary = _fixture()
    captured: dict = {}
    expected = ConversationSummaryV2(
        schema_version=2,
        boundary_id=boundary.boundary_id,
        summary="Validated summary",
        learning_goals=["Learn mechanics"],
        preferences=[],
        facts=[],
        decisions=[],
        unfinished_tasks=[],
        evidence_ids=["evidence-1"],
        artifact_ids=["artifact-1"],
    )

    async def fake_invoke_one_mode(**kwargs):
        captured.update(kwargs)
        return expected, "raw output must not be traced by this wrapper", object()

    monkeypatch.setattr(compaction_module, "_invoke_one_mode", fake_invoke_one_mode)

    result = await invoke_conversation_compaction(
        boundary=boundary,
        messages=messages,
        state={
            "evidence_summary_memory": [{"memory_id": "evidence-1"}],
            "last_generated_artifacts": [{"artifact_id": "artifact-1"}],
        },
        config=_config(),
    )

    assert result == expected
    assert captured["node_name"] == "conversation_compactor"
    assert captured["mode"] == "deepseek_tool_call_strict"
    assert captured["state"] == {
        "request_id": "request-2",
        "thread_id": "thread-1",
        "session_id": "thread-1",
        "compaction_summary_call": True,
    }
    prompt = json.loads(captured["messages"][1].content)
    assert prompt["transcript"] == [
        {
            "role": "human",
            "content": "old query",
            "tool_call_ids": [],
            "tool_result_id": "",
        },
        {
            "role": "ai",
            "content": "old answer",
            "tool_call_ids": [],
            "tool_result_id": "",
        },
    ]


@pytest.mark.anyio
async def test_compaction_llm_blocks_hallucinated_reference_ids(monkeypatch):
    from src.llm import compaction as compaction_module

    messages, boundary = _fixture()
    invalid = ConversationSummaryV2(
        schema_version=2,
        boundary_id=boundary.boundary_id,
        summary="Invalid summary",
        learning_goals=[],
        preferences=[],
        facts=[],
        decisions=[],
        unfinished_tasks=[],
        evidence_ids=["hallucinated"],
        artifact_ids=[],
    )

    async def fake_invoke_one_mode(**_kwargs):
        return invalid, "", object()

    monkeypatch.setattr(compaction_module, "_invoke_one_mode", fake_invoke_one_mode)

    with pytest.raises(
        ConversationCompactionError, match="conversation compaction failed"
    ):
        await invoke_conversation_compaction(
            boundary=boundary,
            messages=messages,
            state={},
            config=_config(),
        )


@pytest.mark.anyio
async def test_compaction_llm_blocks_provider_failure_without_fallback(monkeypatch):
    from src.llm import compaction as compaction_module

    messages, boundary = _fixture()
    calls = 0

    async def fake_invoke_one_mode(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider body must not escape")

    monkeypatch.setattr(compaction_module, "_invoke_one_mode", fake_invoke_one_mode)

    with pytest.raises(
        ConversationCompactionError,
        match=r"conversation compaction failed \(RuntimeError\)",
    ):
        await invoke_conversation_compaction(
            boundary=boundary,
            messages=messages,
            state={},
            config=_config(),
        )

    assert calls == 1
