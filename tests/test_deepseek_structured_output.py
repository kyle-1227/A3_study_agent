import json
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from src.config import get_setting
from src.graph.supervisor import SupervisorOutput, validate_supervisor_output
from src.llm.structured_output import (
    StructuredOutputError,
    compile_pydantic_schema_for_deepseek_tool,
    get_llm_output_mode,
    invoke_structured_llm,
)
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


class NestedDeepSeekSchemaModel(BaseModel):
    label: str = Field("", max_length=16)


class DeepSeekSchemaModel(BaseModel):
    name: str = Field(..., max_length=32)
    tags: list[Annotated[str, Field(max_length=12)]] = Field(default_factory=list, max_length=4)
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
    requests: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, headers=None, json=None):
        self.__class__.requests.append({"url": url, "headers": headers or {}, "json": json or {}})
        if len(self.__class__.responses) > 1:
            return self.__class__.responses.pop(0)
        return self.__class__.responses[0]


def _tool_response(arguments, *, tool_name: str = "supervisor_SupervisorOutput") -> _FakeResponse:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return _FakeResponse({
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
    })


def _supervisor_args(**overrides) -> dict:
    data = {
        "intent": "academic",
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

        assert getattr(exc_info.value, "failure_phase", "") == "deepseek_schema_compile_error"


@pytest.mark.anyio
class TestDeepSeekStrictRuntime:
    async def test_supervisor_tool_call_success_and_trace(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr("src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient)
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
        assert _FakeAsyncClient.requests[0]["url"] == "https://api.deepseek.com/beta/chat/completions"
        payload = next(event for event in events if event["stage"] == "structured_llm_output")
        assert payload["provider"] == "deepseek_official"
        assert payload["provider_request_mode"] == "deepseek_tool_call_strict"
        assert payload["using_deepseek_official_http"] is True
        assert payload["using_direct_openrouter_http"] is False
        assert payload["fallback_used"] is False
        assert payload["tool_call_present"] is True

    async def test_missing_tool_call_fails_with_deepseek_phase(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr("src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [_FakeResponse({"choices": [{"finish_reason": "stop", "message": {}}]})]
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
        monkeypatch.setattr("src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [_tool_response(_supervisor_args(), tool_name="wrong_tool")]
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
        monkeypatch.setattr("src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient)
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
        monkeypatch.setattr("src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient)
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

    async def test_business_validation_failure_keeps_business_phase(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr("src.llm.structured_output.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "src.llm.structured_output.invoke_with_provider_transport_retry",
            _fake_transport_retry,
        )
        _FakeAsyncClient.responses = [
            _tool_response(_supervisor_args(intent="emotional", requested_resource_type="quiz"))
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


async def _fake_transport_retry(operation, **_kwargs):
    return await operation(), 0


class TestDeepSeekConfigScope:
    def test_first_batch_nodes_use_deepseek_official_strict_mode(self):
        for node_name in (
            "supervisor",
            "memory_use_decider",
            "search_result_judge",
            "web_coverage_decision",
        ):
            assert get_setting(f"llm.{node_name}.provider") == "deepseek_official"
            assert get_setting(f"llm.{node_name}.api_key_env") == "DEEPSEEK_API_KEY"
            assert get_llm_output_mode(node_name) == "deepseek_tool_call_strict"

    def test_complex_nodes_keep_existing_modes(self):
        assert get_setting("llm.query_rewrite.provider") == "openrouter"
        assert get_llm_output_mode("search_query_rewriter") == "native_json_schema_pydantic"
        assert get_setting("llm.evidence_judge.provider") == "openrouter"
        assert get_llm_output_mode("evidence_judge") == "native_json_schema_pydantic"
        assert get_setting("llm.web_search.provider", None) is None
