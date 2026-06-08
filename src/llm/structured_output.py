"""Provider-neutral structured LLM output runtime.

The runtime separates model provider configuration from the second-layer
structured-output mechanism.  Output modes therefore describe how the response
is constrained and parsed, never which provider serves the model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.config import get_setting
from src.graph.llm import get_node_llm
from src.observability.a3_trace import emit_a3_trace

logger = logging.getLogger(__name__)

ALLOWED_OUTPUT_MODES = {
    "prompt_json_pydantic",
    "json_mode_pydantic",
    "tool_call_pydantic",
    "native_json_schema_pydantic",
    "constrained_decoding",
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^;\n]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"nvapi-[A-Za-z0-9_-]+"),
)


@dataclass
class StructuredLLMAttempt:
    output_mode: str
    success: bool = False
    failure_phase: str = ""
    error_type: str = ""
    error_message: str = ""
    status_code: Any = None
    provider_error_body: str = ""


@dataclass
class StructuredLLMResult:
    success: bool
    parsed: BaseModel | None
    node_name: str
    llm_node: str
    schema_name: str
    provider: str
    model: str
    output_mode: str
    fallback_modes: list[str] = field(default_factory=list)
    attempts: list[StructuredLLMAttempt] = field(default_factory=list)
    raw_output: str = ""
    provider_error_body: str = ""
    failure_phase: str = ""
    error_type: str = ""
    error_message: str = ""
    status_code: Any = None
    parsing_error: str = ""
    validation_error: str = ""
    business_validation_error: str = ""
    fail_fast: bool = False
    fallback_used: bool = False
    default_used: bool = False
    retry_count: int = 0
    failure_policy: str = ""

    def to_debug_payload(self, *, max_raw_chars: int = 4000) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "llm_node": self.llm_node,
            "schema_name": self.schema_name,
            "provider": self.provider,
            "model": self.model,
            "output_mode": self.output_mode,
            "fallback_modes": self.fallback_modes,
            "fail_fast": self.fail_fast,
            "fallback_used": self.fallback_used,
            "default_used": self.default_used,
            "retry_count": self.retry_count,
            "failure_policy": self.failure_policy,
            "success": self.success,
            "failure_phase": self.failure_phase,
            "error_type": self.error_type,
            "error_message": _sanitize(self.error_message, max_chars=2000),
            "status_code": self.status_code,
            "raw_output": _sanitize(self.raw_output, max_chars=max_raw_chars),
            "provider_error_body": _sanitize(self.provider_error_body, max_chars=max_raw_chars),
            "parsing_error": _sanitize(self.parsing_error, max_chars=2000),
            "validation_error": _sanitize(self.validation_error, max_chars=4000),
            "business_validation_error": _sanitize(self.business_validation_error, max_chars=4000),
            "attempts": [
                {
                    "output_mode": attempt.output_mode,
                    "success": attempt.success,
                    "failure_phase": attempt.failure_phase,
                    "error_type": attempt.error_type,
                    "error_message": _sanitize(attempt.error_message, max_chars=1200),
                    "status_code": attempt.status_code,
                    "provider_error_body": _sanitize(attempt.provider_error_body, max_chars=3000),
                }
                for attempt in self.attempts
            ],
        }


class StructuredOutputError(RuntimeError):
    """Raised when a structured-output call fails under fail-fast policy."""

    def __init__(self, result: StructuredLLMResult):
        self.result = result
        super().__init__(
            f"{result.node_name} failed to produce valid {result.schema_name}: "
            f"{result.failure_phase or result.error_type or 'structured_output_failed'}"
        )


def _setting(node_name: str, key: str, default: Any = None) -> Any:
    return get_setting(f"llm.{node_name}.{key}", default)


def _output_setting(node_name: str, key: str, default: Any = None) -> Any:
    value = get_setting(f"llm_outputs.{node_name}.{key}", None)
    if value is not None:
        return value
    return get_setting(f"llm_outputs.default.{key}", default)


def _fail_fast_enabled() -> bool:
    return bool(get_setting("development.fail_fast_structured_output", True))


def get_llm_output_mode(node_name: str) -> str:
    """Return the configured provider-neutral output mode for a structured node."""
    mode = str(_output_setting(node_name, "output_mode", "native_json_schema_pydantic") or "")
    _validate_mode(mode)
    return mode


def get_fallback_modes(node_name: str) -> list[str]:
    """Return configured fallback modes; always empty in fail-fast development mode."""
    if _fail_fast_enabled():
        return []
    raw = _output_setting(node_name, "fallback_modes", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"llm_outputs.{node_name}.fallback_modes must be a list[str]")
    modes = [str(mode) for mode in raw]
    for mode in modes:
        _validate_mode(mode)
    return modes


def get_max_raw_chars(node_name: str) -> int:
    return int(_output_setting(node_name, "max_raw_chars", 12000) or 12000)


def _failure_policy(node_name: str) -> str:
    return str(_output_setting(node_name, "failure_policy", "block") or "block")


def _provider(node_name: str) -> str:
    return str(_setting(node_name, "provider", "unknown") or "unknown")


def _model(node_name: str) -> str:
    return str(_setting(node_name, "model", "") or "")


def _sanitize(value: Any, *, max_chars: int = 4000) -> str:
    text = str(value or "").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.lower().startswith("sk-"):
            text = pattern.sub("sk-[REDACTED]", text)
        elif pattern.pattern.lower().startswith("sk-or"):
            text = pattern.sub("sk-or-v1-[REDACTED]", text)
        elif pattern.pattern.lower().startswith("nvapi"):
            text = pattern.sub("nvapi-[REDACTED]", text)
        else:
            text = pattern.sub(r"\1[REDACTED]", text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _raw_output_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    return _message_text(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Structured output JSON root must be an object")
    return parsed


def _extract_status_code(exc: Exception) -> Any:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _extract_provider_error_body(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        return str(response.text)
    except Exception:
        return ""


def _is_second_layer_unsupported(exc: Exception, mode: str) -> bool:
    text = f"{type(exc).__name__} {exc} {_extract_provider_error_body(exc)}".lower()
    if _extract_status_code(exc) in {400, 404, 422}:
        return True
    return any(term in text for term in ("unsupported", "not support", "response_format", "tool", "json_schema", "function"))


def _validate_mode(mode: str) -> None:
    if mode not in ALLOWED_OUTPUT_MODES:
        raise ValueError(
            f"Unsupported output_mode={mode!r}. Allowed modes: {sorted(ALLOWED_OUTPUT_MODES)}"
        )


def _json_output_contract(schema: type[BaseModel], node_name: str, mode: str) -> str:
    return (
        "Structured output contract for this call:\n"
        f"- Node: {node_name}\n"
        f"- Schema: {schema.__name__}\n"
        f"- Output mode: {mode}\n"
        "- Return exactly one valid JSON object matching the configured Pydantic schema.\n"
        "- Do not output markdown, code fences, comments, explanations, or extra text.\n"
        "- Do not omit required fields. Use only schema-compatible enum values.\n"
        "- If unsure, still return the best schema-valid JSON object; never answer in prose."
    )


def _inject_json_contract(messages: list, *, schema: type[BaseModel], node_name: str, mode: str) -> list:
    contract = _json_output_contract(schema, node_name, mode)
    if messages and isinstance(messages[0], dict):
        return [{"role": "system", "content": contract}, *messages]
    return [SystemMessage(content=contract), *messages]


def _tool_schema(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.__name__,
            "description": f"Return a {schema.__name__} object.",
            "parameters": schema.model_json_schema(),
        },
    }


async def _invoke_one_mode(
    *,
    node_name: str,
    llm_node: str,
    schema: type[BaseModel],
    messages: list,
    mode: str,
) -> tuple[BaseModel, str]:
    llm = get_node_llm(llm_node)
    messages = _inject_json_contract(messages, schema=schema, node_name=node_name, mode=mode)

    if mode == "constrained_decoding":
        raise NotImplementedError("constrained_decoding is reserved but not implemented")

    if mode == "native_json_schema_pydantic":
        runnable = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
        )
        response = await runnable.ainvoke(messages)
        raw_output = _raw_output_from_response(response)
        return schema.model_validate(_extract_json_object(raw_output)), raw_output

    if mode == "json_mode_pydantic":
        runnable = llm.bind(response_format={"type": "json_object"})
        response = await runnable.ainvoke(messages)
        raw_output = _raw_output_from_response(response)
        return schema.model_validate(_extract_json_object(raw_output)), raw_output

    if mode == "tool_call_pydantic":
        runnable = llm.bind(tools=[_tool_schema(schema)], tool_choice={"type": "function", "function": {"name": schema.__name__}})
        response = await runnable.ainvoke(messages)
        raw_output = _raw_output_from_response(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls and isinstance(response, AIMessage):
            raw_calls = (response.additional_kwargs or {}).get("tool_calls") or []
            for raw_call in raw_calls:
                function = raw_call.get("function") or {}
                args = function.get("arguments") or "{}"
                tool_calls.append({"name": function.get("name", ""), "args": args})
        if not tool_calls:
            raise ValueError("No tool call returned")
        args = tool_calls[0].get("args") if isinstance(tool_calls[0], dict) else getattr(tool_calls[0], "args", None)
        parsed_args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(parsed_args, dict):
            raise ValueError("Tool call arguments must be a JSON object")
        return schema.model_validate(parsed_args), raw_output or json.dumps(parsed_args, ensure_ascii=False)

    if mode == "prompt_json_pydantic":
        response = await llm.ainvoke(messages)
        raw_output = _raw_output_from_response(response)
        return schema.model_validate(_extract_json_object(raw_output)), raw_output

    raise ValueError(f"Unsupported output mode {mode!r}")


async def invoke_structured_llm(
    *,
    node_name: str,
    llm_node: str | None = None,
    schema: type[BaseModel],
    messages: list,
    output_mode: str,
    fallback_modes: list[str] | None = None,
    business_validator: Callable[[BaseModel], None | str | list[str]] | None = None,
    state: dict | None = None,
    max_raw_chars: int | None = None,
) -> StructuredLLMResult:
    """Invoke a structured-output LLM.

    In development fail-fast mode, the first failure raises
    :class:`StructuredOutputError`; fallback modes remain configuration-only
    and are not executed.
    """
    llm_node = llm_node or node_name
    fail_fast = _fail_fast_enabled()
    requested_fallback_modes = list(fallback_modes or [])
    effective_fallback_modes = [] if fail_fast else requested_fallback_modes
    modes = [output_mode, *effective_fallback_modes]
    provider = _provider(llm_node)
    model = _model(llm_node)
    schema_name = schema.__name__
    raw_limit = int(max_raw_chars or get_max_raw_chars(node_name))
    failure_policy = _failure_policy(node_name)
    attempts: list[StructuredLLMAttempt] = []

    def _emit_and_maybe_raise(result: StructuredLLMResult, *, exc: Exception | None = None) -> None:
        emit_a3_trace(
            logger,
            "structured_llm_output",
            result.to_debug_payload(max_raw_chars=raw_limit),
            state=state,
            env_flag="LOG_STRUCTURED_LLM_OUTPUT",
            max_chars=raw_limit,
        )
        if fail_fast and not result.success:
            if exc is not None:
                raise StructuredOutputError(result) from exc
            raise StructuredOutputError(result)

    for mode in modes:
        last_raw_output = ""
        try:
            _validate_mode(mode)
        except Exception as exc:
            attempt = StructuredLLMAttempt(
                output_mode=mode,
                success=False,
                failure_phase="invalid_output_mode",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            attempts.append(attempt)
            result = StructuredLLMResult(
                success=False,
                parsed=None,
                node_name=node_name,
                llm_node=llm_node,
                schema_name=schema_name,
                provider=provider,
                model=model,
                output_mode=mode,
                fallback_modes=effective_fallback_modes,
                attempts=attempts,
                failure_phase=attempt.failure_phase,
                error_type=attempt.error_type,
                error_message=attempt.error_message,
                fail_fast=fail_fast,
                fallback_used=False,
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
            )
            _emit_and_maybe_raise(result, exc=exc)
            continue

        try:
            parsed, raw_output = await _invoke_one_mode(
                node_name=node_name,
                llm_node=llm_node,
                schema=schema,
                messages=messages,
                mode=mode,
            )
            last_raw_output = raw_output
            business_error = ""
            if business_validator is not None:
                validation_result = business_validator(parsed)
                if isinstance(validation_result, list):
                    business_error = "; ".join(str(item) for item in validation_result if item)
                elif validation_result:
                    business_error = str(validation_result)
            if business_error:
                raise ValueError(f"business_validation_error: {business_error}")

            attempt = StructuredLLMAttempt(output_mode=mode, success=True)
            attempts.append(attempt)
            result = StructuredLLMResult(
                success=True,
                parsed=parsed,
                node_name=node_name,
                llm_node=llm_node,
                schema_name=schema_name,
                provider=provider,
                model=model,
                output_mode=mode,
                fallback_modes=effective_fallback_modes,
                attempts=attempts,
                raw_output=raw_output,
                fail_fast=fail_fast,
                fallback_used=(mode != output_mode),
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
            )
            _emit_and_maybe_raise(result)
            return result
        except Exception as exc:
            status_code = _extract_status_code(exc)
            provider_error_body = _extract_provider_error_body(exc)
            if isinstance(exc, NotImplementedError):
                phase = f"second_layer_{mode}_unsupported"
            elif _is_second_layer_unsupported(exc, mode):
                phase = f"second_layer_{mode}_unsupported"
            elif isinstance(exc, (json.JSONDecodeError, ValueError)) and "business_validation_error:" not in str(exc):
                phase = "parsing_error" if isinstance(exc, json.JSONDecodeError) else "validation_error"
            elif isinstance(exc, ValidationError):
                phase = "validation_error"
            elif "business_validation_error:" in str(exc):
                phase = "business_validation_error"
            else:
                phase = "llm_exception"

            attempt = StructuredLLMAttempt(
                output_mode=mode,
                success=False,
                failure_phase=phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=status_code,
                provider_error_body=provider_error_body,
            )
            attempts.append(attempt)
            result = StructuredLLMResult(
                success=False,
                parsed=None,
                node_name=node_name,
                llm_node=llm_node,
                schema_name=schema_name,
                provider=provider,
                model=model,
                output_mode=mode,
                fallback_modes=effective_fallback_modes,
                attempts=attempts,
                provider_error_body=provider_error_body,
                failure_phase=phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=status_code,
                raw_output=last_raw_output,
                parsing_error=str(exc) if phase == "parsing_error" else "",
                validation_error=str(exc) if phase == "validation_error" else "",
                business_validation_error=str(exc).replace("business_validation_error:", "").strip()
                if phase == "business_validation_error"
                else "",
                fail_fast=fail_fast,
                fallback_used=(mode != output_mode),
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
            )
            _emit_and_maybe_raise(result, exc=exc)
            continue

    last = attempts[-1] if attempts else StructuredLLMAttempt(output_mode=output_mode, failure_phase="no_attempts")
    result = StructuredLLMResult(
        success=False,
        parsed=None,
        node_name=node_name,
        llm_node=llm_node,
        schema_name=schema_name,
        provider=provider,
        model=model,
        output_mode=last.output_mode or output_mode,
        fallback_modes=effective_fallback_modes,
        attempts=attempts,
        failure_phase=last.failure_phase,
        error_type=last.error_type,
        error_message=last.error_message,
        status_code=last.status_code,
        provider_error_body=last.provider_error_body,
        fail_fast=fail_fast,
        fallback_used=False,
        default_used=False,
        retry_count=0,
        failure_policy=failure_policy,
    )
    _emit_and_maybe_raise(result)
    return result
