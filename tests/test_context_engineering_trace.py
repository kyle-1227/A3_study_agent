"""Trace payload tests for Context Engineering telemetry."""

from __future__ import annotations

import logging

from src.context_engineering import trace
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def test_context_usage_event_allows_only_canonical_safe_fields():
    event = trace.build_context_usage_event(
        {
            "node_name": "node",
            "llm_node": "llm",
            "provider": "provider",
            "model": "model",
            "input_estimated_tokens": 10,
            "reserved_output_tokens": 5,
            "used_tokens": 15,
            "max_context_tokens": 1000,
            "available_tokens": 985,
            "used_ratio": 0.015,
            "warning_level": "ok",
            "estimated": True,
            "tokenizer_mode": "estimated_mixed",
            "message_count": 1,
            "schema_size_chars": 123,
            "breakdown": {"input_estimated_tokens": 10, "reserved_output_tokens": 5},
            "prompt": "do not expose",
            "schema": {"type": "object"},
            "messages": [{"content": "secret"}],
            "raw_output": "secret",
            "api_key": "sk-secret",
            "cookie": "session=secret",
            "db_uri": "postgres://secret",
            "prompt_tokens": 10,
            "usage_ratio": 0.015,
        }
    )

    assert event == {
        "node_name": "node",
        "llm_node": "llm",
        "provider": "provider",
        "model": "model",
        "input_estimated_tokens": 10,
        "reserved_output_tokens": 5,
        "used_tokens": 15,
        "max_context_tokens": 1000,
        "available_tokens": 985,
        "used_ratio": 0.015,
        "warning_level": "ok",
        "estimated": True,
        "tokenizer_mode": "estimated_mixed",
        "message_count": 1,
        "schema_size_chars": 123,
        "breakdown": {"input_estimated_tokens": 10, "reserved_output_tokens": 5},
    }


def test_context_usage_event_filters_malicious_breakdown_fields():
    event = trace.build_context_usage_event(
        {
            "node_name": "node",
            "llm_node": "llm",
            "provider": "provider",
            "model": "model",
            "input_estimated_tokens": 10,
            "reserved_output_tokens": 5,
            "used_tokens": 15,
            "max_context_tokens": 1000,
            "available_tokens": 985,
            "used_ratio": 0.015,
            "warning_level": "ok",
            "estimated": True,
            "tokenizer_mode": "estimated_mixed",
            "message_count": 1,
            "breakdown": {
                "input_estimated_tokens": 10,
                "reserved_output_tokens": 5,
                "schema_size_chars": 123,
                "prompt": "secret prompt",
                "messages": [{"content": "secret message"}],
                "schema": {"properties": {"secret": {"type": "string"}}},
                "raw_output": "secret raw output",
                "api_key": "sk-secret",
                "cookie": "session=secret",
                "db_uri": "postgres://secret",
                "bool_is_not_int": True,
                "string_is_not_int": "10",
            },
        }
    )

    assert event["breakdown"] == {
        "input_estimated_tokens": 10,
        "reserved_output_tokens": 5,
        "schema_size_chars": 123,
    }
    serialized = repr(event).lower()
    assert "secret prompt" not in serialized
    assert "secret message" not in serialized
    assert "secret raw output" not in serialized
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "db_uri" not in serialized


def test_context_usage_error_event_is_sanitized():
    event = trace.build_context_usage_error_event(
        node_name="node",
        llm_node="llm",
        provider="provider",
        model="model",
        reason="model_window_unknown",
        warning="model context window is unknown",
    )

    assert event == {
        "node_name": "node",
        "llm_node": "llm",
        "provider": "provider",
        "model": "model",
        "reason": "model_window_unknown",
        "warning": "model context window is unknown",
    }


def test_emit_context_usage_payload_does_not_leak_prompt_or_schema():
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        trace.emit_context_usage(
            logging.getLogger("test_context_engineering_trace"),
            node_name="node",
            llm_node="study_plan",
            provider="deepseek_official",
            model="deepseek-v4-pro",
            messages=[{"content": "prompt must only be counted"}],
            state={"request_id": "r1", "thread_id": "t1"},
            schema_size_chars=456,
        )
    finally:
        reset_trace_event_sink(token)

    assert sink
    event = sink[0]
    serialized = repr(event).lower()
    assert event["stage"] == "context_usage"
    assert event["schema_size_chars"] == 456
    assert "prompt must only be counted" not in serialized
    assert "messages" not in serialized
    assert "raw_output" not in serialized
    assert "api_key" not in serialized
    assert "cookie" not in serialized
    assert "db_uri" not in serialized
