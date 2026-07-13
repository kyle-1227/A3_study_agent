from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from src.context_engineering.compaction import (
    CompactBoundaryV1,
    ConversationSummaryV2,
    FullCompactionConfigV1,
    ProviderBoundUsageV1,
    build_compact_boundary,
    collect_summary_reference_ids,
    evaluate_full_compaction,
    validate_conversation_summary,
)
from src.context_engineering.model_view import (
    ModelViewConfigV1,
    ModelViewProjectionError,
    build_model_view_projection,
)


def _full_config(*, enabled: bool = True, rounds: int = 3):
    return FullCompactionConfigV1(
        schema_version=1,
        enabled=enabled,
        retain_recent_rounds=rounds,
        summary_llm_node="conversation_compactor",
        output_mode="deepseek_tool_call_strict",
        max_summary_input_chars=128000,
    )


def _model_view_config(*, rounds: int = 3):
    return ModelViewConfigV1(
        schema_version=1,
        micro_compaction_enabled=True,
        retain_recent_rounds=rounds,
    )


def _usage(*, input_tokens: int) -> ProviderBoundUsageV1:
    return ProviderBoundUsageV1(
        dispatch_id="dispatch-1",
        call_id="call-1",
        request_id="request-1",
        thread_id="thread-1",
        attempt=1,
        provider="deepseek_official",
        model="deepseek-v4-pro",
        input_tokens=input_tokens,
        tokenizer_mode="estimated_mixed",
        estimated=True,
        trigger_eligible=True,
        dispatched_at=datetime.now(timezone.utc),
    )


def _summary(boundary: CompactBoundaryV1) -> ConversationSummaryV2:
    return ConversationSummaryV2(
        schema_version=2,
        boundary_id=boundary.boundary_id,
        summary="The learner is continuing a physics study task.",
        learning_goals=["Understand Newtonian mechanics"],
        preferences=["Use worked examples"],
        facts=["The learner has reviewed force diagrams"],
        decisions=["Continue with acceleration problems"],
        unfinished_tasks=["Complete the practice set"],
        evidence_ids=[],
        artifact_ids=[],
    )


def test_full_compaction_decision_uses_actual_dispatch_ratio():
    first_request = evaluate_full_compaction(None, config=_full_config())
    below = evaluate_full_compaction(
        _usage(input_tokens=899_999).model_dump(mode="json"),
        config=_full_config(),
    )
    reached = evaluate_full_compaction(
        _usage(input_tokens=900_000).model_dump(mode="json"),
        config=_full_config(),
    )

    assert first_request.reason == "no_actual_provider_dispatch"
    assert below.reason == "below_threshold"
    assert below.eligible is False
    assert reached.reason == "threshold_reached"
    assert reached.eligible is True
    assert reached.observed_ratio == 0.9
    assert reached.context_window_limit_tokens == 1_000_000


def test_full_compaction_disabled_is_explicit():
    decision = evaluate_full_compaction(
        _usage(input_tokens=999_999).model_dump(mode="json"),
        config=_full_config(enabled=False),
    )

    assert decision.reason == "disabled"
    assert decision.eligible is False


def test_boundary_retains_recent_rounds_and_preserves_tool_pairs():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old query"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "tool-1"}],
        ),
        ToolMessage(content="old result", tool_call_id="tool-1"),
        AIMessage(content="old answer"),
        HumanMessage(content="round two"),
        AIMessage(content="answer two"),
        HumanMessage(content="round three"),
        AIMessage(content="answer three"),
        HumanMessage(content="current query"),
    ]

    boundary = build_compact_boundary(
        messages,
        thread_id="thread-1",
        request_id="request-2",
        trigger_dispatch_id="dispatch-1",
        retain_recent_rounds=3,
    )

    assert boundary is not None
    assert [item.original_index for item in boundary.compacted_messages] == [1, 2, 3, 4]
    assert boundary.compacted_messages[1].tool_call_ids == ["tool-1"]
    assert boundary.compacted_messages[2].tool_result_id == "tool-1"
    assert boundary.retained_message_count == 6


def test_boundary_returns_none_when_history_does_not_exceed_retention():
    messages = [
        HumanMessage(content="one"),
        AIMessage(content="answer"),
        HumanMessage(content="two"),
    ]

    boundary = build_compact_boundary(
        messages,
        thread_id="thread-1",
        request_id="request-2",
        trigger_dispatch_id="dispatch-1",
        retain_recent_rounds=3,
    )

    assert boundary is None


def test_boundary_rejects_incomplete_tool_pair():
    messages = [
        HumanMessage(content="old"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "tool-missing"}],
        ),
        HumanMessage(content="current"),
    ]

    with pytest.raises(ValueError, match="tool call/result pair is incomplete"):
        build_compact_boundary(
            messages,
            thread_id="thread-1",
            request_id="request-2",
            trigger_dispatch_id="dispatch-1",
            retain_recent_rounds=1,
        )


def test_model_view_applies_boundary_without_mutating_transcript():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old query " + "details " * 200),
        AIMessage(content="old answer " + "explanation " * 200),
        HumanMessage(content="current query"),
    ]
    before = copy.deepcopy(messages)
    boundary = build_compact_boundary(
        messages,
        thread_id="thread-1",
        request_id="request-2",
        trigger_dispatch_id="dispatch-1",
        retain_recent_rounds=1,
    )
    assert boundary is not None
    summary = _summary(boundary)

    result = build_model_view_projection(
        messages,
        config=_model_view_config(rounds=1),
        state={
            "thread_id": "thread-1",
            "compact_boundary": boundary.model_dump(mode="json"),
            "conversation_summary_v2": summary.model_dump(mode="json"),
        },
    )

    assert messages == before
    assert result.projection.full_compaction_boundary_id == boundary.boundary_id
    assert result.projection.compacted_history_messages_removed == 2
    assert result.projection.conversation_summary_injected is True
    contents = [message.content for message in result.messages]
    assert all("old query" not in content for content in contents)
    assert all("old answer" not in content for content in contents)
    assert any("COMPACTED_CONVERSATION_SUMMARY" in content for content in contents)
    assert contents[-1] == "current query"
    persisted = json.dumps(result.projection.model_dump(mode="json"))
    assert "old query" not in persisted
    assert summary.summary not in persisted


def test_model_view_blocks_when_boundary_no_longer_matches():
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
    summary = _summary(boundary)
    changed_messages = [*messages]
    changed_messages[0] = HumanMessage(content="changed old query")

    with pytest.raises(ModelViewProjectionError, match="did not match"):
        build_model_view_projection(
            changed_messages,
            config=_model_view_config(rounds=1),
            state={
                "thread_id": "thread-1",
                "compact_boundary": boundary.model_dump(mode="json"),
                "conversation_summary_v2": summary.model_dump(mode="json"),
            },
        )


def test_summary_validation_requires_exact_reference_ids():
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
    summary = _summary(boundary).model_copy(
        update={"evidence_ids": ["evidence-1"], "artifact_ids": ["artifact-1"]}
    )

    validate_conversation_summary(
        summary,
        boundary=boundary,
        required_evidence_ids={"evidence-1"},
        required_artifact_ids={"artifact-1"},
    )
    with pytest.raises(ValueError, match="evidence_ids mismatch"):
        validate_conversation_summary(
            summary,
            boundary=boundary,
            required_evidence_ids=set(),
            required_artifact_ids={"artifact-1"},
        )


def test_reference_collection_preserves_previous_summary_ids():
    previous = ConversationSummaryV2(
        schema_version=2,
        boundary_id="compact-boundary:v1:sha256:" + "a" * 64,
        summary="previous",
        learning_goals=[],
        preferences=[],
        facts=[],
        decisions=[],
        unfinished_tasks=[],
        evidence_ids=["evidence-old"],
        artifact_ids=["artifact-old"],
    )
    state = {
        "conversation_summary_v2": previous.model_dump(mode="json"),
        "evidence_summary_memory": [{"memory_id": "evidence-new"}],
        "last_generated_artifacts": [{"artifact_id": "artifact-new"}],
    }

    evidence_ids, artifact_ids = collect_summary_reference_ids(state)

    assert evidence_ids == {"evidence-old", "evidence-new"}
    assert artifact_ids == {"artifact-old", "artifact-new"}


def test_conversation_summary_requires_explicit_schema_version():
    with pytest.raises(ValidationError):
        ConversationSummaryV2.model_validate(
            {
                "boundary_id": "compact-boundary:v1:sha256:" + "a" * 64,
                "summary": "missing explicit schema version",
                "learning_goals": [],
                "preferences": [],
                "facts": [],
                "decisions": [],
                "unfinished_tasks": [],
                "evidence_ids": [],
                "artifact_ids": [],
            }
        )
