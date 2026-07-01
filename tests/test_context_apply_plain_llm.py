"""Plain LLM integration tests for Phase 3B-1 context apply."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextInjectionPolicy,
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.schema import ContextItem


def _policy(
    *,
    enabled: bool = True,
    nodes: tuple[str, ...] = ("plain_node",),
    fallback_on_error: bool = True,
    max_tokens: int = 10000,
) -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=enabled,
        apply_enabled_nodes=nodes,
        fallback_on_error=fallback_on_error,
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=max_tokens,
        injectable_sources=("memory", "evidence", "rules"),
    )


def _item(
    item_id: str = "memory-1",
    *,
    source_type: str = "memory",
    content: str = "useful memory",
) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=content,
        token_estimate=5,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
    )


def _packed(items: list[ContextItem] | None = None):
    return pack_context_items(
        node_name="plain_node",
        llm_node="llm",
        items=items or [_item()],
        max_context_block_tokens=10000,
    )


def _mock_llm(content: str = "answer"):
    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content=content))
    return mock_llm


@pytest.mark.anyio
async def test_apply_disabled_keeps_phase3a_order_and_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    order: list[str] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_context_injection_policy",
        lambda **_: _policy(enabled=False),
    )
    monkeypatch.setattr(llm_module, "apply_node_enabled", lambda policy, **_: False)
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(
        llm_module,
        "emit_context_usage_trace",
        lambda *_, **__: order.append("usage"),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: order.append("items") or [_item()],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: order.append("packing") or _packed(),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert order[:3] == ["usage", "items", "packing"]
    mock_llm.ainvoke.assert_awaited_once_with(messages)
    assert messages == [{"role": "user", "content": "question"}]


@pytest.mark.anyio
async def test_apply_enabled_but_node_miss_uses_original_messages(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_context_injection_policy",
        lambda **_: _policy(nodes=("other_node",)),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item()],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed(),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    mock_llm.ainvoke.assert_awaited_once_with(messages)


@pytest.mark.anyio
async def test_apply_enabled_node_uses_final_messages_and_context_usage(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    usage_messages: list[list] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_context_injection_policy",
        lambda **_: _policy(),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(
        llm_module,
        "emit_context_usage_trace",
        lambda *_, **kwargs: usage_messages.append(kwargs["messages"]),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item()],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed(),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    final_messages = mock_llm.ainvoke.await_args.args[0]
    assert final_messages is usage_messages[0]
    assert final_messages is not messages
    assert "<INJECTED_CONTEXT>" in final_messages[0]["content"]
    assert "useful memory" in final_messages[0]["content"]
    assert messages == [{"role": "user", "content": "question"}]


@pytest.mark.anyio
async def test_apply_error_with_fallback_uses_original_messages_and_usage(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    usage_messages: list[list] = []
    trace_payloads: list[dict] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_context_injection_policy",
        lambda **_: _policy(max_tokens=1, fallback_on_error=True),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(
        llm_module,
        "emit_context_usage_trace",
        lambda *_, **kwargs: usage_messages.append(kwargs["messages"]),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_a3_trace",
        lambda _logger, stage, payload, **_kwargs: trace_payloads.append(
            {"stage": stage, **payload}
        ),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item()],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed(),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    assert usage_messages == [messages]
    mock_llm.ainvoke.assert_awaited_once_with(messages)
    output_payload = next(
        payload for payload in trace_payloads if payload["stage"] == "plain_llm_output"
    )
    assert output_payload["fallback_used"] is False
    assert output_payload["context_apply_applied"] is False
    assert output_payload["context_apply_fallback_used"] is True


@pytest.mark.anyio
async def test_apply_error_without_fallback_raises_before_llm_and_emits_safe_error(
    monkeypatch,
):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    apply_errors: list[ContextApplyError] = []
    mock_llm = _mock_llm()

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_context_injection_policy",
        lambda **_: _policy(max_tokens=1, fallback_on_error=False),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 0)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [
            _item(content="api_key=sk-secret-value cookie=session must not leak")
        ],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed(
            [_item(content="api_key=sk-secret-value cookie=session must not leak")]
        ),
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_apply_error",
        lambda _logger, *, error, **_kwargs: apply_errors.append(error),
    )

    with pytest.raises(ContextApplyError) as exc_info:
        await invoke_plain_llm_fail_fast(
            node_name="plain_node",
            llm_node="llm",
            messages=messages,
            state={"request_id": "r1", "thread_id": "t1"},
        )

    assert exc_info.value.reason == "injected_context_over_budget"
    mock_llm.ainvoke.assert_not_awaited()
    assert len(apply_errors) == 1
    serialized_error = repr(
        {
            "reason": apply_errors[0].reason,
            "warning": apply_errors[0].warning,
            "fallback_used": apply_errors[0].fallback_used,
        }
    ).lower()
    assert apply_errors[0].fallback_used is False
    assert "api_key" not in serialized_error
    assert "cookie" not in serialized_error
    assert "sk-secret-value" not in serialized_error
    assert "must not leak" not in serialized_error


@pytest.mark.anyio
async def test_apply_retry_uses_same_final_messages_object(monkeypatch):
    from src.graph import llm as llm_module
    from src.graph.llm import invoke_plain_llm_fail_fast

    messages = [{"role": "user", "content": "question"}]
    mock_llm = MagicMock()
    mock_llm.model_name = "deepseek-v4-pro"
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            SimpleNamespace(content=""),
            SimpleNamespace(content="answer"),
        ]
    )

    monkeypatch.setattr(llm_module, "get_node_llm", lambda _node: mock_llm)
    monkeypatch.setattr(
        llm_module,
        "get_context_injection_policy",
        lambda **_: _policy(),
    )
    monkeypatch.setattr(llm_module, "get_llm_call_max_retries", lambda *_, **__: 1)
    monkeypatch.setattr(llm_module, "emit_context_usage_trace", lambda *_, **__: None)
    monkeypatch.setattr(
        llm_module,
        "emit_context_items_shadow",
        lambda *_, **__: [_item()],
    )
    monkeypatch.setattr(
        llm_module,
        "emit_context_packing_shadow",
        lambda *_, **__: _packed(),
    )

    result = await invoke_plain_llm_fail_fast(
        node_name="plain_node",
        llm_node="llm",
        messages=messages,
        state={"request_id": "r1", "thread_id": "t1"},
    )

    assert result == "answer"
    first_messages = mock_llm.ainvoke.await_args_list[0].args[0]
    second_messages = mock_llm.ainvoke.await_args_list[1].args[0]
    assert first_messages is second_messages
    assert "<INJECTED_CONTEXT>" in first_messages[0]["content"]
