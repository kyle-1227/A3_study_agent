from __future__ import annotations

import copy
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.context_engineering.model_view import (
    ModelViewConfigV1,
    build_model_view_projection,
    get_model_view_config,
)
from src.context_engineering.schema import ContextConfigError


def _config(*, enabled: bool = True, rounds: int = 1) -> ModelViewConfigV1:
    return ModelViewConfigV1(
        schema_version=1,
        micro_compaction_enabled=enabled,
        retain_recent_rounds=rounds,
    )


def test_projection_deduplicates_exact_injected_context_without_mutating_input():
    context = "<INJECTED_CONTEXT>same memory</INJECTED_CONTEXT>"
    messages = [
        SystemMessage(content="authoritative system"),
        SystemMessage(content=context),
        SystemMessage(content=context),
        HumanMessage(content="current query"),
    ]
    before = copy.deepcopy(messages)

    result = build_model_view_projection(messages, config=_config())

    assert messages == before
    assert result.messages is not messages
    assert [message.content for message in result.messages] == [
        "authoritative system",
        context,
        "current query",
    ]
    assert result.projection.duplicate_context_messages_removed == 1
    assert result.projection.tool_results_compacted == 0
    assert result.projection.output_message_count == 3


def test_projection_compacts_only_old_tool_result_with_trusted_metadata():
    tool_result = ToolMessage(
        content="raw retrieval body that must remain outside persisted projection",
        tool_call_id="call-1",
        additional_kwargs={
            "model_view_compaction": {
                "trusted": True,
                "reference_id": "evidence:physics:1",
                "safe_summary": "Newton's second law evidence.",
            }
        },
    )
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old query"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "physics"}, "id": "call-1"}],
        ),
        tool_result,
        AIMessage(content="old answer"),
        HumanMessage(content="current query"),
    ]

    result = build_model_view_projection(messages, config=_config(rounds=1))

    projected_tool = result.messages[3]
    assert isinstance(projected_tool, ToolMessage)
    assert projected_tool.tool_call_id == "call-1"
    assert projected_tool.content == (
        "[COMPACTED_TOOL_RESULT]\n"
        "reference_id: evidence:physics:1\n"
        "Newton's second law evidence."
    )
    assert tool_result.content.startswith("raw retrieval body")
    assert result.projection.tool_results_compacted == 1
    operation = result.projection.operations[0]
    assert operation.reference_id == "evidence:physics:1"
    serialized = json.dumps(result.projection.model_dump(mode="json"))
    assert "raw retrieval body" not in serialized
    assert "Newton's second law evidence" not in serialized


def test_projection_retains_recent_tool_result_even_with_trusted_metadata():
    messages = [
        HumanMessage(content="current query"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "call-current"}],
        ),
        ToolMessage(
            content="current raw result",
            tool_call_id="call-current",
            additional_kwargs={
                "model_view_compaction": {
                    "trusted": True,
                    "reference_id": "evidence:current",
                    "safe_summary": "current summary",
                }
            },
        ),
    ]

    result = build_model_view_projection(messages, config=_config(rounds=1))

    assert result.messages[2].content == "current raw result"
    assert result.projection.tool_results_compacted == 0


def test_projection_retains_old_tool_result_without_explicit_trusted_replacement():
    messages = [
        HumanMessage(content="old query"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "call-old"}],
        ),
        ToolMessage(content="raw result", tool_call_id="call-old"),
        AIMessage(content="answer"),
        HumanMessage(content="current query"),
    ]

    result = build_model_view_projection(messages, config=_config(rounds=1))

    assert result.messages[2].content == "raw result"
    assert result.projection.tool_results_compacted == 0


def test_projection_rejects_malformed_replacement_instead_of_falling_back():
    messages = [
        {"role": "user", "content": "old query"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-old", "function": {"name": "search"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call-old",
            "content": "raw result",
            "model_view_compaction": {
                "trusted": True,
                "reference_id": "",
                "safe_summary": "",
            },
        },
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "current query"},
    ]

    with pytest.raises(
        ValueError,
        match="invalid trusted model-view replacement metadata",
    ):
        build_model_view_projection(messages, config=_config(rounds=1))


def test_disabled_micro_compaction_builds_identity_projection():
    context = "<INJECTED_CONTEXT>same</INJECTED_CONTEXT>"
    messages = [
        {"role": "system", "content": context},
        {"role": "system", "content": context},
        {"role": "user", "content": "query"},
    ]

    result = build_model_view_projection(messages, config=_config(enabled=False))

    assert result.messages == messages
    assert result.messages is not messages
    assert result.projection.input_message_count == 3
    assert result.projection.output_message_count == 3
    assert result.projection.operations == []
    assert (
        result.projection.source_message_fingerprint
        == result.projection.projected_message_fingerprint
    )


def test_projection_handles_fewer_user_rounds_than_retention_limit():
    messages = [{"role": "user", "content": "only query"}]

    result = build_model_view_projection(messages, config=_config(rounds=3))

    assert result.messages == messages
    assert result.projection.retained_recent_rounds == 3


def test_model_view_config_is_required(monkeypatch):
    from src.context_engineering import model_view

    monkeypatch.setattr(model_view, "get_setting", lambda _key: None)

    with pytest.raises(ContextConfigError) as exc_info:
        get_model_view_config()

    assert exc_info.value.reason == "model_view_config_missing"


def test_model_view_config_rejects_unknown_fields(monkeypatch):
    from src.context_engineering import model_view

    monkeypatch.setattr(
        model_view,
        "get_setting",
        lambda _key: {
            "schema_version": 1,
            "micro_compaction_enabled": True,
            "retain_recent_rounds": 3,
            "silent_default": True,
        },
    )

    with pytest.raises(ContextConfigError) as exc_info:
        get_model_view_config()

    assert exc_info.value.reason == "model_view_config_invalid"
