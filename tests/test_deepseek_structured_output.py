import json
from typing import Annotated
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from src.config import get_setting
from src.graph.qa import QAResponse
from src.graph.supervisor import SupervisorOutput, validate_supervisor_output
from src.llm.structured_output import (
    StructuredOutputError,
    StructuredLLMResult,
    _build_reask_instruction,
    compile_pydantic_schema_for_deepseek_tool,
    get_fallback_modes,
    get_llm_output_mode,
    invoke_structured_llm,
)
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink
from src.streaming.provisional import (
    reset_provisional_event_sink,
    set_provisional_event_sink,
)


class NestedDeepSeekSchemaModel(BaseModel):
    label: str = Field("", max_length=16)


class DeepSeekSchemaModel(BaseModel):
    name: str = Field(..., max_length=32)
    tags: list[Annotated[str, Field(max_length=12)]] = Field(
        default_factory=list, max_length=4
    )
    nested: NestedDeepSeekSchemaModel = Field(default_factory=NestedDeepSeekSchemaModel)


class DeepSeekMapModel(BaseModel):
    values: dict[str, str]


class _FakeResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class _FakeAsyncClient:
    responses: list[_FakeResponse] = []
    stream_responses: list["_FakeStreamResponse"] = []
    requests: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None):
        self.__class__.requests.append(
            {"url": url, "headers": headers or {}, "json": json or {}}
        )
        if len(self.__class__.responses) > 1:
            return self.__class__.responses.pop(0)
        return self.__class__.responses[0]

    def stream(self, method, url, *, headers=None, json=None):
        self.__class__.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "json": json or {},
            }
        )
        if len(self.__class__.stream_responses) > 1:
            return self.__class__.stream_responses.pop(0)
        return self.__class__.stream_responses[0]


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self.lines = lines
        self.status_code = status_code
        self.text = "\n".join(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self):
        return self.text.encode("utf-8")


def _tool_response(
    arguments, *, tool_name: str = "supervisor_SupervisorOutput"
) -> _FakeResponse:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return _FakeResponse(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": arguments,
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )


def _json_response(content) -> _FakeResponse:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return _FakeResponse(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": content},
                }
            ]
        }
    )


def _supervisor_args(**overrides) -> dict:
    data = {
        "intent": "academic",
        "response_mode": "resource",
        "qa_scope": "",
        "requires_live_verification": False,
        "keywords": ["Python"],
        "confidence": 0.92,
        "subject_candidates": ["python"],
        "requested_resource_type": "quiz",
    }
    data.update(overrides)
    return data


def _assert_no_schema_key(schema: object, key: str) -> None:
    if isinstance(schema, dict):
        assert key not in schema
        for value in schema.values():
            _assert_no_schema_key(value, key)
    elif isinstance(schema, list):
        for value in schema:
            _assert_no_schema_key(value, key)


class TestDeepSeekSchemaCompiler:
    def test_removes_unsupported_validation_keywords_and_requires_all_properties(self):
        compiled = compile_pydantic_schema_for_deepseek_tool(DeepSeekSchemaModel)

        _assert_no_schema_key(compiled, "maxLength")
        _assert_no_schema_key(compiled, "maxItems")
        assert compiled["additionalProperties"] is False
        assert set(compiled["required"]) == set(compiled["properties"])

        nested = compiled["$defs"]["NestedDeepSeekSchemaModel"]
        assert nested["additionalProperties"] is False
        assert set(nested["required"]) == set(nested["properties"])

    def test_unsupported_map_schema_fails_fast(self):
        with pytest.raises(RuntimeError) as exc_info:
            compile_pydantic_schema_for_deepseek_tool(DeepSeekMapModel)

        assert (
            getattr(exc_info.value, "failure_phase", "")
            == "deepseek_schema_compile_error"
        )


@pytest.mark.anyio
class TestDeepSeekStrictRuntime:
    async def test_qa_tool_arguments_stream_only_provisional_answer(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        payload = {
            "answer": "先解释概念，再给出例子。",
            "uncertainty_note": "",
            "grounding_status": "general_knowledge",
            "suggestions": [],
        }
        arguments = json.dumps(payload, ensure_ascii=False)
        split_at = arguments.index("再给出")
        _FakeAsyncClient.stream_responses = [
            _FakeStreamResponse(
                [
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "finish_reason": None,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "name": "qa_agent_QAResponse",
                                                    "arguments": arguments[:split_at],
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "finish_reason": "tool_calls",
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "arguments": arguments[split_at:],
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "data: [DONE]",
                ]
            )
        ]
        _FakeAsyncClient.requests = []
        provisional: list[dict] = []
        sink_token = set_provisional_event_sink(provisional.append)
        try:
            result = await invoke_structured_llm(
                node_name="qa_agent",
                llm_node="qa_agent",
                schema=QAResponse,
                messages=[{"role": "user", "content": "解释这个概念"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={
                    "request_id": "00000000-0000-4000-8000-000000000001",
                    "thread_id": "thread-1",
                },
            )
        finally:
            reset_provisional_event_sink(sink_token)

        assert result.success is True
        assert isinstance(result.parsed, QAResponse)
        assert result.parsed.answer == payload["answer"]
        assert _FakeAsyncClient.requests[0]["json"]["stream"] is True
        assert [event["type"] for event in provisional] == [
            "qa_provisional_start",
            "qa_provisional_delta",
            "qa_provisional_delta",
            "qa_provisional_stop",
        ]
        assert (
            "".join(
                event["delta"]
                for event in provisional
                if event["type"] == "qa_provisional_delta"
            )
            == payload["answer"]
        )

    async def test_supervisor_tool_call_success_and_trace(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [_tool_response(_supervisor_args())]
        _FakeAsyncClient.requests = []
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            result = await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "给我一份 Python 的练习题"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                business_validator=validate_supervisor_output,
                state={"request_id": "r1", "thread_id": "t1"},
            )
        finally:
            reset_trace_event_sink(token)

        assert result.success is True
        assert isinstance(result.parsed, SupervisorOutput)
        assert (
            _FakeAsyncClient.requests[0]["url"]
            == "https://api.deepseek.com/beta/chat/completions"
        )
        assert _FakeAsyncClient.requests[0]["json"]["thinking"] == {"type": "disabled"}
        payload = next(
            event for event in events if event["stage"] == "structured_llm_output"
        )
        assert payload["provider"] == "deepseek_official"
        assert payload["provider_request_mode"] == "deepseek_tool_call_strict"
        assert payload["using_deepseek_official_http"] is True
        assert payload["using_direct_openrouter_http"] is False
        assert payload["fallback_used"] is False
        assert payload["tool_call_present"] is True

    async def test_missing_tool_call_fails_with_deepseek_phase(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [
            _FakeResponse({"choices": [{"finish_reason": "stop", "message": {}}]})
        ]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )

        assert exc_info.value.result.failure_phase == "deepseek_tool_call_missing"

    async def test_wrong_tool_name_fails_with_deepseek_phase(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [
            _tool_response(_supervisor_args(), tool_name="wrong_tool")
        ]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )

        assert exc_info.value.result.failure_phase == "deepseek_wrong_tool_name"

    async def test_empty_arguments_fails_with_deepseek_phase(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [_tool_response("")]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )

        assert exc_info.value.result.failure_phase == "deepseek_empty_tool_arguments"

    async def test_malformed_arguments_fails_as_parsing_error(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [_tool_response("{not-json")]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )

        assert exc_info.value.result.failure_phase == "parsing_error"

    async def test_malformed_arguments_reask_then_success(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [
            _tool_response("{not-json"),
            _tool_response(_supervisor_args()),
        ]
        _FakeAsyncClient.requests = []
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            result = await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                business_validator=validate_supervisor_output,
                state={},
            )
        finally:
            reset_trace_event_sink(token)

        assert result.success is True
        assert result.retry_count == 1
        assert len(result.attempts) == 2
        assert _FakeAsyncClient.requests[1]["json"]["messages"][-1]["role"] == "user"
        correction = _FakeAsyncClient.requests[1]["json"]["messages"][-1]["content"]
        assert "Structured output correction required" in correction
        assert "Previous failure_phase: parsing_error" in correction

        retry_event = next(
            event
            for event in events
            if event["stage"] == "structured_llm_retry_attempt"
        )
        reask_event = next(
            event
            for event in events
            if event["stage"] == "structured_llm_reask_attempt"
        )
        assert retry_event["reask_used"] is True
        assert reask_event["reask_reason"] == "parsing_error"
        final_payload = [
            event for event in events if event["stage"] == "structured_llm_output"
        ][-1]
        assert final_payload["reask_used"] is True
        assert final_payload["provider"] == "deepseek_official"
        assert final_payload["provider_request_mode"] == "deepseek_tool_call_strict"

    async def test_validation_error_reask_includes_field_path_then_success(
        self, monkeypatch
    ):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        invalid_args = _supervisor_args()
        invalid_args.pop("keywords")
        _FakeAsyncClient.responses = [
            _tool_response(invalid_args),
            _tool_response(_supervisor_args()),
        ]
        _FakeAsyncClient.requests = []

        result = await invoke_structured_llm(
            node_name="supervisor",
            llm_node="supervisor",
            schema=SupervisorOutput,
            messages=[{"role": "user", "content": "route"}],
            output_mode="deepseek_tool_call_strict",
            fallback_modes=[],
            business_validator=validate_supervisor_output,
            state={},
        )

        assert result.success is True
        assert result.retry_count == 1
        correction = _FakeAsyncClient.requests[1]["json"]["messages"][-1]["content"]
        assert "Previous failure_phase: validation_error" in correction
        assert "keywords" in correction

    async def test_business_validation_failure_keeps_business_phase(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        monkeypatch.setattr(
            "src.llm.structured_output._reask_business_validation_enabled",
            lambda _node_name: False,
        )
        _FakeAsyncClient.responses = [
            _tool_response(
                _supervisor_args(intent="emotional", requested_resource_type="quiz")
            )
        ]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                business_validator=validate_supervisor_output,
                state={},
            )

        assert exc_info.value.result.failure_phase == "business_validation_error"
        assert len(exc_info.value.result.attempts) == 1

    async def test_length_finish_reason_is_semantic_failure(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [
            _FakeResponse({"choices": [{"finish_reason": "length", "message": {}}]})
        ]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )

        assert exc_info.value.result.failure_phase == "deepseek_tool_call_truncated"
        assert len(exc_info.value.result.attempts) == 3

    async def test_content_filter_finish_reason_fails_fast(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [
            _FakeResponse(
                {"choices": [{"finish_reason": "content_filter", "message": {}}]}
            )
        ]
        _FakeAsyncClient.requests = []

        with pytest.raises(StructuredOutputError) as exc_info:
            await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )

        assert exc_info.value.result.failure_phase == "provider_content_filter"
        assert len(exc_info.value.result.attempts) == 1

    async def test_insufficient_system_resource_uses_transport_retry(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        from src.graph import llm as llm_module

        monkeypatch.setattr(
            llm_module, "_provider_transport_max_retries", lambda node_name=None: 2
        )
        monkeypatch.setattr(
            llm_module, "_provider_transport_delay_seconds", lambda _attempt: 0
        )
        monkeypatch.setattr(llm_module.asyncio, "sleep", AsyncMock())
        _FakeAsyncClient.responses = [
            _FakeResponse(
                {
                    "choices": [
                        {"finish_reason": "insufficient_system_resource", "message": {}}
                    ]
                }
            ),
            _tool_response(_supervisor_args()),
        ]
        _FakeAsyncClient.requests = []
        events: list[dict] = []
        token = set_trace_event_sink(events)
        try:
            result = await invoke_structured_llm(
                node_name="supervisor",
                llm_node="supervisor",
                schema=SupervisorOutput,
                messages=[{"role": "user", "content": "route"}],
                output_mode="deepseek_tool_call_strict",
                fallback_modes=[],
                state={},
            )
        finally:
            reset_trace_event_sink(token)

        assert result.success is True
        assert len(_FakeAsyncClient.requests) == 2
        stages = [event["stage"] for event in events]
        assert "structured_llm_transport_retry_attempt" in stages

    async def test_deepseek_json_object_does_not_force_strict_thinking_override(
        self, monkeypatch
    ):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [_json_response(_supervisor_args())]
        _FakeAsyncClient.requests = []

        result = await invoke_structured_llm(
            node_name="supervisor",
            llm_node="academic",
            schema=SupervisorOutput,
            messages=[{"role": "user", "content": "return json route"}],
            output_mode="deepseek_json_object",
            fallback_modes=[],
            state={},
        )

        assert result.success is True
        request = _FakeAsyncClient.requests[0]
        assert request["url"] == "https://api.deepseek.com/chat/completions"
        assert request["json"]["response_format"] == {"type": "json_object"}
        assert "tool_choice" not in request["json"]
        assert "tools" not in request["json"]
        assert "thinking" not in request["json"]

    async def test_business_validation_reask_only_when_enabled(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(
            "src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient
        )
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        monkeypatch.setattr(
            "src.llm.structured_output._reask_business_validation_enabled",
            lambda _node_name: True,
        )
        _FakeAsyncClient.responses = [
            _tool_response(_supervisor_args(confidence=0.50)),
            _tool_response(_supervisor_args(confidence=0.99)),
        ]
        _FakeAsyncClient.requests = []

        def require_high_confidence(parsed):
            if parsed.confidence < 0.95:
                return "confidence must be at least 0.95 for this test"
            return ""

        result = await invoke_structured_llm(
            node_name="supervisor",
            llm_node="supervisor",
            schema=SupervisorOutput,
            messages=[{"role": "user", "content": "route"}],
            output_mode="deepseek_tool_call_strict",
            fallback_modes=[],
            business_validator=require_high_confidence,
            state={},
        )

        assert result.success is True
        assert result.retry_count == 1
        correction = _FakeAsyncClient.requests[1]["json"]["messages"][-1]["content"]
        assert "Previous failure_phase: business_validation_error" in correction
        assert "confidence must be at least 0.95" in correction


async def _fake_transport_retry(operation, **_kwargs):
    return await operation(), 0


class TestDeepSeekConfigScope:
    LLM_NODES = (
        "academic",
        "supervisor",
        "query_rewrite",
        "memory_use_decider",
        "hallucination_eval",
        "mindmap",
        "exercise",
        "web_research_planner",
        "web_source_summarizer",
        "evidence_judge",
        "emotional",
        "review_doc",
        "study_plan",
        "profile_extractor",
    )
    STRUCTURED_NODE_TO_LLM_NODE = {
        "supervisor": "supervisor",
        "memory_use_decider": "memory_use_decider",
        "search_query_rewriter": "query_rewrite",
        "hallucination_eval": "hallucination_eval",
        "web_research_planner": "web_research_planner",
        "web_source_summarizer": "web_source_summarizer",
        "evidence_item_grader": "evidence_judge",
        "evidence_sufficiency_judge": "evidence_judge",
        "exercise_agent": "exercise",
        "exercise_reviewer": "exercise",
        "mindmap_agent": "mindmap",
        "mindmap_reviewer": "mindmap",
        "study_plan_emotional_intel": "study_plan",
        "study_plan_agent": "study_plan",
        "study_plan_reviewer_academic": "study_plan",
        "study_plan_reviewer_emotional": "study_plan",
        "review_doc_reviewer": "review_doc",
        "profile_extractor": "profile_extractor",
    }
    STRICT_LLM_NODES = tuple(sorted(set(STRUCTURED_NODE_TO_LLM_NODE.values())))

    def test_all_deepseek_llm_nodes_resolve_to_official_api(self):
        for llm_node in self.LLM_NODES:
            assert get_setting(f"llm.{llm_node}.provider") == "deepseek_official"
            assert get_setting(f"llm.{llm_node}.model") == "deepseek-v4-pro"
            assert get_setting(f"llm.{llm_node}.base_url") == "https://api.deepseek.com"
            assert (
                get_setting(f"llm.{llm_node}.beta_base_url")
                == "https://api.deepseek.com/beta"
            )
            assert get_setting(f"llm.{llm_node}.api_key_env") == "DEEPSEEK_API_KEY"

    def test_structured_nodes_resolve_strict_mode_and_no_fallback(self):
        for node_name, llm_node in self.STRUCTURED_NODE_TO_LLM_NODE.items():
            assert get_setting(f"llm.{llm_node}.provider") == "deepseek_official"
            assert get_llm_output_mode(node_name) == "deepseek_tool_call_strict"
            assert get_fallback_modes(node_name) == []
            assert get_setting(f"llm_outputs.{node_name}.fallback_modes", []) == []
            max_retries = get_setting(f"llm_outputs.{node_name}.max_retries", 2)
            assert isinstance(max_retries, int)
            assert not isinstance(max_retries, bool)
            assert max_retries >= 1

        # Staged study-plan generation intentionally permits one strict repair.
        assert get_setting("llm_outputs.study_plan_agent.max_retries") == 1

    def test_strict_structured_llm_nodes_disable_thinking_in_config(self):
        for llm_node in self.STRICT_LLM_NODES:
            assert get_setting(f"llm.{llm_node}.thinking") == "disabled"

    def test_no_deepseek_chat_generation_node_uses_openrouter(self):
        for llm_node in self.LLM_NODES:
            provider = str(get_setting(f"llm.{llm_node}.provider", "")).lower()
            model = str(get_setting(f"llm.{llm_node}.model", "")).lower()
            base_url = str(get_setting(f"llm.{llm_node}.base_url", "")).lower()
            api_key_env = str(get_setting(f"llm.{llm_node}.api_key_env", "")).upper()
            if "deepseek" in model:
                assert provider != "openrouter"
                assert "openrouter.ai" not in base_url
                assert api_key_env != "OPENROUTER_API_KEY"

        assert get_setting("rag.reranker_base_url") == "https://openrouter.ai/api/v1"
        assert get_setting("llm.web_search.provider", None) is None

    def test_default_reask_config_is_conservative(self):
        assert get_setting("llm_outputs.default.reask_enabled") is True
        assert get_setting("llm_outputs.default.reask_business_validation") is True
        assert get_setting("llm_outputs.default.transport_max_retries") == 2

    def test_reask_instruction_does_not_use_empty_value_for_non_empty_business_rule(
        self,
    ):
        instruction = _build_reask_instruction(
            result=StructuredLLMResult(
                success=False,
                parsed=None,
                node_name="evidence_item_grader",
                llm_node="evidence_judge",
                schema_name="EvidenceGradeBatch",
                provider="unit",
                model="unit",
                output_mode="native_json_schema_pydantic",
                failure_phase="business_validation_error",
                error_type="BusinessValidationError",
                error_message="coverage_contribution must not be empty",
                business_validation_error="coverage_contribution must not be empty",
            ),
            schema_name="EvidenceGradeBatch",
            previous_error_summary="coverage_contribution must not be empty",
        )

        assert "validation error says non-empty" in instruction
        assert "business validation error says a field must be non-empty" in instruction
        assert (
            "Do not use an empty value to satisfy a field that failed a non-empty business rule"
            in instruction
        )
        assert (
            'If the previous error says "Extra inputs are not permitted"' in instruction
        )
        assert 'If the previous error says "Field required"' in instruction
        assert (
            "Do not fix one validation error by introducing extra fields" in instruction
        )
