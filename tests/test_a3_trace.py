"""Tests for temporary A3_TRACE diagnostic logs."""

from __future__ import annotations

import json
import logging

from src.observability.a3_trace import emit_a3_trace


def _a3_payload(record) -> dict:
    text = record.getMessage()
    assert text.startswith("A3_TRACE ")
    return json.loads(text.removeprefix("A3_TRACE "))


def test_emit_a3_trace_default_off(caplog, monkeypatch):
    monkeypatch.delenv("LOG_A3_TRACE", raising=False)
    monkeypatch.delenv("LOG_RAG_RESULT", raising=False)

    logger = logging.getLogger("tests.a3_trace.default_off")
    with caplog.at_level(logging.WARNING):
        emit_a3_trace(logger, "rag", {"query": "test"}, env_flag="LOG_RAG_RESULT")

    assert not caplog.records


def test_emit_a3_trace_master_switch_with_ids_and_truncation(caplog, monkeypatch):
    monkeypatch.setenv("LOG_A3_TRACE", "true")

    logger = logging.getLogger("tests.a3_trace.master")
    with caplog.at_level(logging.WARNING):
        emit_a3_trace(
            logger,
            "query_rewrite",
            {"long": "x" * 20},
            state={"request_id": "req-1", "session_id": "sess-1", "thread_id": "thread-1"},
            env_flag="LOG_QUERY_REWRITE_RESULT",
            max_chars=5,
        )

    payload = _a3_payload(caplog.records[0])
    assert payload["stage"] == "query_rewrite"
    assert payload["request_id"] == "req-1"
    assert payload["session_id"] == "sess-1"
    assert payload["thread_id"] == "thread-1"
    assert payload["long"] == "xxxxx..."


def test_emit_a3_trace_fine_grained_switch_keeps_query_rewrite_compat(caplog, monkeypatch):
    monkeypatch.delenv("LOG_A3_TRACE", raising=False)
    monkeypatch.setenv("LOG_QUERY_REWRITE_RESULT", "true")

    logger = logging.getLogger("tests.a3_trace.query")
    with caplog.at_level(logging.WARNING):
        emit_a3_trace(
            logger,
            "query_rewrite",
            {"result": "ok"},
            state={"thread_id": "thread-2"},
            env_flag="LOG_QUERY_REWRITE_RESULT",
        )

    payload = _a3_payload(caplog.records[0])
    assert payload["stage"] == "query_rewrite"
    assert payload["thread_id"] == "thread-2"
    assert payload["session_id"] == "thread-2"
    assert payload["request_id"] == "unknown"


def test_emit_a3_trace_never_raises(monkeypatch):
    monkeypatch.setenv("LOG_A3_TRACE", "true")

    class BadLogger:
        def warning(self, _line):
            raise RuntimeError("logging down")

        def debug(self, *_args, **_kwargs):
            pass

    emit_a3_trace(BadLogger(), "bad", {"x": "y"})
