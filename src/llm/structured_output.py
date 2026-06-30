"""Provider-neutral structured LLM output runtime.

The runtime separates model provider configuration from the second-layer
structured-output mechanism.  Output modes therefore describe how the response
is constrained and parsed, never which provider serves the model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.config import get_setting
from src.graph.llm import (
    get_llm_call_max_retries,
    get_node_llm,
    invoke_with_provider_transport_retry,
)
from src.llm.http_messages import (
    _InvalidHttpMessageFormatError,
    normalize_openai_messages,
    preview_openai_messages,
    validate_openai_messages,
)
from src.llm.schema_drift import (
    analyze_schema_drift_trace_only,
    drift_report_summary,
    render_drift_report_text,
)
from src.llm.schema_manifest import (
    DriftGuardConfig,
    SchemaManifest,
    build_canonical_manifest,
    config_error_message,
    get_structured_output_manifest_config,
    load_drift_guard_config,
    manifest_summary,
    render_manifest_text,
)
from src.context_engineering.providers import emit_context_items_shadow
from src.context_engineering.tokenizer import count_schema_chars
from src.observability.context_usage import emit_context_usage_trace
from src.observability.a3_trace import emit_a3_trace

logger = logging.getLogger(__name__)

ALLOWED_OUTPUT_MODES = {
    "prompt_json_pydantic",
    "json_mode_pydantic",
    "tool_call_pydantic",
    "native_json_schema_pydantic",
    "constrained_decoding",
    "deepseek_tool_call_strict",
    "deepseek_json_object",
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


class _BusinessValidationError(Exception):
    """Raised when business_validator rejects a parsed result."""

    def __init__(self, message: str):
        super().__init__(message)


class _DeepSeekStructuredOutputError(RuntimeError):
    """Raised for DeepSeek official structured-output protocol failures."""

    def __init__(self, failure_phase: str, message: str):
        self.failure_phase = failure_phase
        super().__init__(message)


class _StructuredManifestError(RuntimeError):
    """Raised when prompt/debug manifest config cannot be built safely."""

    def __init__(self, failure_phase: str, message: str):
        self.failure_phase = failure_phase
        super().__init__(message)


class DeepSeekInsufficientSystemResourceError(RuntimeError):
    """Retryable DeepSeek provider capacity/infra failure."""

    retryable_provider_error = True

    def __init__(self, message: str, response: httpx.Response | None = None):
        super().__init__(message)
        self.response = response
        self.failure_phase = "provider_transport_error"


class DeepSeekProviderResponseJSONError(RuntimeError):
    """Retryable invalid provider envelope without model output."""

    retryable_provider_error = True

    def __init__(self, message: str, response: httpx.Response | None = None):
        super().__init__(message)
        self.response = response
        self.failure_phase = "provider_transport_error"


@dataclass
class _InvokeMetrics:
    """Diagnostics for one structured output attempt."""
    total_elapsed_ms: float = 0.0
    llm_elapsed_ms: float = 0.0
    json_pydantic_elapsed_ms: float = 0.0
    business_validate_elapsed_ms: float = 0.0
    parse_validate_elapsed_ms: float = 0.0
    prompt_chars: int = 0
    raw_output_chars: int = 0
    using_direct_openrouter_http: bool = False
    provider_request_mode: str = ""
    http_messages_preview: list[dict[str, Any]] = field(default_factory=list)
    extra_debug: dict[str, Any] = field(default_factory=dict)


class _InvokeOneModeError(Exception):
    """Wraps _invoke_one_mode exception with metrics + raw_output."""

    def __init__(self, cause: Exception, metrics: _InvokeMetrics, raw_output: str = ""):
        self.cause = cause
        self.metrics = metrics
        self.raw_output = raw_output
        super().__init__(str(cause))
        self.__cause__ = cause


@dataclass
class StructuredLLMAttempt:
    output_mode: str
    success: bool = False
    failure_phase: str = ""
    error_type: str = ""
    error_message: str = ""
    status_code: Any = None
    provider_error_body: str = ""
    # diagnostics
    total_elapsed_ms: float = 0.0
    llm_elapsed_ms: float = 0.0
    json_pydantic_elapsed_ms: float = 0.0
    business_validate_elapsed_ms: float = 0.0
    parse_validate_elapsed_ms: float = 0.0
    prompt_chars: int = 0
    raw_output_chars: int = 0
    schema_size_chars: int = 0
    using_direct_openrouter_http: bool = False
    provider_request_mode: str = ""
    http_messages_preview: list[dict[str, Any]] = field(default_factory=list)
    extra_debug: dict[str, Any] = field(default_factory=dict)


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
    # diagnostics
    total_elapsed_ms: float = 0.0
    llm_elapsed_ms: float = 0.0
    json_pydantic_elapsed_ms: float = 0.0
    business_validate_elapsed_ms: float = 0.0
    parse_validate_elapsed_ms: float = 0.0
    prompt_chars: int = 0
    raw_output_chars: int = 0
    schema_size_chars: int = 0
    using_direct_openrouter_http: bool = False
    provider_request_mode: str = ""
    http_messages_preview: list[dict[str, Any]] = field(default_factory=list)
    extra_debug: dict[str, Any] = field(default_factory=dict)

    def to_debug_payload(self, *, max_raw_chars: int = 4000) -> dict[str, Any]:
        using_direct_openrouter_http = (
            self.using_direct_openrouter_http
            or any(attempt.using_direct_openrouter_http for attempt in self.attempts)
        )
        provider_request_mode = self.provider_request_mode or next(
            (attempt.provider_request_mode for attempt in reversed(self.attempts) if attempt.provider_request_mode),
            "",
        )
        http_messages_preview = self.http_messages_preview or next(
            (attempt.http_messages_preview for attempt in reversed(self.attempts) if attempt.http_messages_preview),
            [],
        )
        extra_debug = self.extra_debug or next(
            (attempt.extra_debug for attempt in reversed(self.attempts) if attempt.extra_debug),
            {},
        )
        payload = {
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
            # diagnostics
            "total_elapsed_ms": self.total_elapsed_ms,
            "llm_elapsed_ms": self.llm_elapsed_ms,
            "json_pydantic_elapsed_ms": self.json_pydantic_elapsed_ms,
            "business_validate_elapsed_ms": self.business_validate_elapsed_ms,
            "parse_validate_elapsed_ms": self.parse_validate_elapsed_ms,
            "prompt_chars": self.prompt_chars,
            "raw_output_chars": self.raw_output_chars,
            "schema_size_chars": self.schema_size_chars,
            "using_direct_openrouter_http": using_direct_openrouter_http,
            "provider_request_mode": provider_request_mode,
            "http_messages_preview": http_messages_preview,
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
                    # diagnostics
                    "total_elapsed_ms": attempt.total_elapsed_ms,
                    "llm_elapsed_ms": attempt.llm_elapsed_ms,
                    "json_pydantic_elapsed_ms": attempt.json_pydantic_elapsed_ms,
                    "business_validate_elapsed_ms": attempt.business_validate_elapsed_ms,
                    "parse_validate_elapsed_ms": attempt.parse_validate_elapsed_ms,
                    "prompt_chars": attempt.prompt_chars,
                    "raw_output_chars": attempt.raw_output_chars,
                    "schema_size_chars": attempt.schema_size_chars,
                    "using_direct_openrouter_http": attempt.using_direct_openrouter_http,
                    "provider_request_mode": attempt.provider_request_mode,
                    "http_messages_preview": attempt.http_messages_preview,
                    **attempt.extra_debug,
                }
                for attempt in self.attempts
            ],
        }
        payload.update(extra_debug)
        return payload


class StructuredOutputError(RuntimeError):
    """Raised when a structured-output call fails under fail-fast policy."""

    def __init__(self, result: StructuredLLMResult):
        self.result = result
        super().__init__(
            f"{result.node_name} failed to produce valid {result.schema_name}: "
            f"{result.failure_phase or result.error_type or 'structured_output_failed'}"
        )


@dataclass
class _ReaskContext:
    instruction: str
    reason: str
    attempt_number: int
    previous_failure_phase: str
    previous_error_summary: str
    schema_manifest_summary: dict[str, Any] = field(default_factory=dict)
    schema_drift_report: dict[str, Any] = field(default_factory=dict)
    drift_guard_source: str = ""
    drift_guard_config_validated: bool = False

    def to_debug(self) -> dict[str, Any]:
        return {
            "reask_used": True,
            "reask_reason": self.reason,
            "reask_attempt_number": self.attempt_number,
            "previous_failure_phase": self.previous_failure_phase,
            "previous_error_summary": self.previous_error_summary,
            "schema_manifest": self.schema_manifest_summary,
            "schema_drift_report": self.schema_drift_report,
            "drift_guard_source": self.drift_guard_source,
            "drift_guard_config_validated": self.drift_guard_config_validated,
        }


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


def _reask_enabled(node_name: str) -> bool:
    return bool(_output_setting(node_name, "reask_enabled", True))


def _reask_business_validation_enabled(node_name: str) -> bool:
    return bool(_output_setting(node_name, "reask_business_validation", False))


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


def _compute_prompt_chars(messages: list) -> int:
    """Total characters in message list after _inject_json_contract. Never raise."""
    total = 0
    for msg in messages or []:
        try:
            if hasattr(msg, "model_dump"):
                total += len(json.dumps(msg.model_dump(), ensure_ascii=False, default=str))
            elif isinstance(msg, dict):
                total += len(json.dumps(msg, ensure_ascii=False, default=str))
            else:
                total += len(str(msg))
        except Exception:
            total += len(str(msg))
    return total


def _safe_schema_size_chars(schema: type[BaseModel]) -> int:
    """Return JSON Schema size for diagnostics via Context Engineering."""
    return count_schema_chars(schema)


_DEEPSEEK_SAFE_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "anyOf",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "type",
}

_DEEPSEEK_DROP_SCHEMA_KEYS = {
    "const",
    "default",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "multipleOf",
    "pattern",
    "title",
    "uniqueItems",
}

_DEEPSEEK_UNSUPPORTED_SCHEMA_KEYS = {
    "allOf",
    "contains",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "maxContains",
    "minContains",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "else",
    "unevaluatedItems",
    "unevaluatedProperties",
}


def compile_pydantic_schema_for_deepseek_tool(schema_model: type[BaseModel]) -> dict[str, Any]:
    """Compile a Pydantic JSON schema for DeepSeek strict tool calling.

    DeepSeek strict mode validates a narrower subset of JSON Schema than
    Pydantic emits.  We relax API-side validation keywords here while leaving
    Pydantic validation intact after the model returns tool arguments.
    """
    try:
        raw_schema = schema_model.model_json_schema()
        compiled = _compile_deepseek_schema_node(raw_schema, path=schema_model.__name__)
    except _DeepSeekStructuredOutputError:
        raise
    except Exception as exc:
        raise _DeepSeekStructuredOutputError(
            "deepseek_schema_compile_error",
            f"Failed to compile {schema_model.__name__} for DeepSeek strict tool calling: {exc}",
        ) from exc
    if not isinstance(compiled, dict):
        raise _DeepSeekStructuredOutputError(
            "deepseek_schema_compile_error",
            f"Compiled {schema_model.__name__} schema is not an object",
        )
    return compiled


def _compile_deepseek_schema_node(node: Any, *, path: str) -> Any:
    if isinstance(node, list):
        return [
            _compile_deepseek_schema_node(item, path=f"{path}[{index}]")
            for index, item in enumerate(node)
        ]
    if not isinstance(node, dict):
        return node

    unsupported = sorted(set(node) & _DEEPSEEK_UNSUPPORTED_SCHEMA_KEYS)
    if unsupported:
        raise _DeepSeekStructuredOutputError(
            "deepseek_schema_compile_error",
            f"DeepSeek strict schema does not support {unsupported} at {path}",
        )

    node_type = node.get("type")
    if node_type == "null" or (isinstance(node_type, list) and "null" in node_type):
        raise _DeepSeekStructuredOutputError(
            "deepseek_schema_compile_error",
            f"DeepSeek strict schema does not support nullable type at {path}",
        )

    compiled: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DEEPSEEK_DROP_SCHEMA_KEYS:
            continue
        if key == "additionalProperties":
            if isinstance(value, dict) and value:
                raise _DeepSeekStructuredOutputError(
                    "deepseek_schema_compile_error",
                    f"DeepSeek strict schema does not support map-like additionalProperties at {path}",
                )
            continue
        if key in {"properties", "$defs"}:
            if not isinstance(value, dict):
                raise _DeepSeekStructuredOutputError(
                    "deepseek_schema_compile_error",
                    f"DeepSeek strict schema expected object for {key} at {path}",
                )
            compiled[key] = {
                str(child_key): _compile_deepseek_schema_node(
                    child_value,
                    path=f"{path}.{key}.{child_key}",
                )
                for child_key, child_value in value.items()
            }
            continue
        if key not in _DEEPSEEK_SAFE_SCHEMA_KEYS:
            continue
        compiled[key] = _compile_deepseek_schema_node(value, path=f"{path}.{key}")

    properties = compiled.get("properties")
    if node_type == "object" or isinstance(properties, dict):
        if not isinstance(properties, dict):
            properties = {}
            compiled["properties"] = properties
        compiled["additionalProperties"] = False
        compiled["required"] = list(properties.keys())

    return compiled


def _round_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _refresh_parse_validate_total(metrics: _InvokeMetrics) -> None:
    """Unified recompute of parse_validate_elapsed_ms and total_elapsed_ms."""
    metrics.parse_validate_elapsed_ms = (
        metrics.json_pydantic_elapsed_ms + metrics.business_validate_elapsed_ms
    )
    metrics.total_elapsed_ms = (
        metrics.llm_elapsed_ms + metrics.parse_validate_elapsed_ms
    )


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


def _classify_failure_phase(exc: Exception, mode: str, _metrics: _InvokeMetrics | None = None) -> str:
    """Classify failure phase for a structured output attempt."""
    explicit_phase = getattr(exc, "failure_phase", "")
    if explicit_phase:
        return str(explicit_phase)
    if isinstance(exc, _InvalidHttpMessageFormatError):
        return "invalid_http_message_format"
    if isinstance(exc, _BusinessValidationError):
        return "business_validation_error"
    if isinstance(exc, json.JSONDecodeError):
        return "parsing_error"
    if isinstance(exc, ValidationError):
        return "validation_error"
    status_code = _extract_status_code(exc)
    if mode in {"deepseek_tool_call_strict", "deepseek_json_object"}:
        if _is_provider_transport_error(exc, status_code=status_code):
            return "provider_transport_error"
        if status_code is not None:
            return "provider_http_error"
    if isinstance(exc, NotImplementedError):
        return f"second_layer_{mode}_unsupported"
    if _is_second_layer_unsupported(exc, mode):
        return f"second_layer_{mode}_unsupported"
    if isinstance(exc, ValueError):
        return "validation_error"
    return "llm_exception"


def _is_semantic_retryable_failure(
    phase: str,
    *,
    include_business_validation: bool = False,
) -> bool:
    if phase in {
        "parsing_error",
        "validation_error",
        "deepseek_tool_call_missing",
        "deepseek_empty_tool_arguments",
        "deepseek_tool_call_truncated",
    }:
        return True
    return bool(include_business_validation and phase == "business_validation_error")


def _is_provider_transport_error(exc: Exception, *, status_code: Any = None) -> bool:
    if bool(getattr(exc, "retryable_provider_error", False)):
        return True
    try:
        numeric_status = int(status_code) if status_code is not None else None
    except Exception:
        numeric_status = None
    if numeric_status == 429 or (numeric_status is not None and 500 <= numeric_status <= 599):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.ConnectError, httpx.TimeoutException)):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def _semantic_retry_count(attempts: list[StructuredLLMAttempt]) -> int:
    attempts_by_mode: dict[str, int] = {}
    for attempt in attempts:
        attempts_by_mode[attempt.output_mode] = attempts_by_mode.get(attempt.output_mode, 0) + 1
    return sum(max(0, count - 1) for count in attempts_by_mode.values())


def _structured_error_summary(result: StructuredLLMResult) -> str:
    return _sanitize(
        result.business_validation_error
        or result.validation_error
        or result.parsing_error
        or result.error_message
        or result.provider_error_body
        or result.error_type
        or result.failure_phase,
        max_chars=1000,
    )


def _schema_manifest_context(
    *,
    schema: type[BaseModel],
    node_name: str,
    mode: str,
) -> tuple[SchemaManifest, DriftGuardConfig, str, dict[str, Any]]:
    try:
        manifest_config = get_structured_output_manifest_config()
        drift_guard = load_drift_guard_config(node_name)
    except Exception as exc:
        raise _StructuredManifestError(
            "drift_guard_config_error",
            f"Invalid structured output manifest/drift guard config: {config_error_message(exc)}",
        ) from exc

    try:
        manifest = build_canonical_manifest(schema, node_name=node_name, output_mode=mode)
    except Exception as exc:
        raise _StructuredManifestError(
            "schema_manifest_error",
            f"Failed to build schema manifest for {schema.__name__}: {config_error_message(exc)}",
        ) from exc

    full_text = render_manifest_text(
        manifest,
        max_chars=None,
        include_descriptions=manifest_config.include_descriptions,
        include_constraints=manifest_config.include_constraints,
        include_enum_values=manifest_config.include_enum_values,
    )
    manifest_text = ""
    truncated = False
    if manifest_config.enabled:
        manifest_text = render_manifest_text(
            manifest,
            max_chars=manifest_config.max_chars,
            include_descriptions=manifest_config.include_descriptions,
            include_constraints=manifest_config.include_constraints,
            include_enum_values=manifest_config.include_enum_values,
        )
        truncated = len(full_text) > manifest_config.max_chars

    debug = {
        "manifest_enabled": manifest_config.enabled,
        "manifest_injected": bool(manifest_config.enabled),
        "manifest_truncated": truncated,
        "manifest_max_chars": manifest_config.max_chars,
        "schema_manifest": manifest_summary(manifest),
        "drift_guard_source": manifest.drift_guard_source,
        "drift_guard_config_validated": manifest.drift_guard_config_validated,
    }
    return manifest, drift_guard, manifest_text, debug


def _schema_drift_report_debug(
    *,
    raw_output: str,
    manifest: SchemaManifest,
    drift_guard: DriftGuardConfig,
    node_name: str,
) -> tuple[str, dict[str, Any]]:
    report = analyze_schema_drift_trace_only(
        raw_output,
        manifest=manifest,
        drift_guard=drift_guard,
        node_name=node_name,
    )
    return render_drift_report_text(report), drift_report_summary(report)


def _attach_schema_drift_debug(
    metrics: _InvokeMetrics,
    *,
    schema: type[BaseModel],
    node_name: str,
    mode: str,
    raw_output: str,
) -> None:
    if "schema_drift_report" in metrics.extra_debug:
        return
    try:
        manifest, drift_guard, _manifest_text, manifest_debug = _schema_manifest_context(
            schema=schema,
            node_name=node_name,
            mode=mode,
        )
        _drift_text, drift_debug = _schema_drift_report_debug(
            raw_output=raw_output,
            manifest=manifest,
            drift_guard=drift_guard,
            node_name=node_name,
        )
        metrics.extra_debug.update(manifest_debug)
        metrics.extra_debug["schema_drift_report"] = drift_debug
    except Exception as exc:
        metrics.extra_debug["schema_drift_report_error"] = _sanitize(str(exc), max_chars=800)


def _build_reask_instruction(
    *,
    result: StructuredLLMResult,
    schema_name: str,
    previous_error_summary: str,
    node_name: str | None = None,
    schema_manifest_text: str = "",
    drift_report_text: str = "",
    drift_guard: DriftGuardConfig | None = None,
) -> str:
    phase = result.failure_phase or "structured_output_error"
    if phase == "parsing_error":
        issue = (
            "The previous response could not be parsed as valid JSON/tool arguments. "
            "Return one syntactically valid JSON object for the required schema."
        )
    elif phase == "validation_error":
        issue = (
            "The previous response parsed as JSON but failed Pydantic/schema validation. "
            "Fix the field paths, enum values, types, required fields, and length/item constraints named below."
        )
    elif phase == "business_validation_error":
        issue = (
            "The previous response passed schema validation but failed this node's business validation. "
            "Fix only the concrete business rule named below."
        )
    elif phase in {"deepseek_tool_call_missing", "deepseek_empty_tool_arguments"}:
        issue = (
            "The previous response did not provide the required tool call arguments. "
            "Call the required function exactly once with a complete schema-valid arguments object."
        )
    elif phase == "deepseek_tool_call_truncated":
        issue = (
            "The previous response was truncated before a complete tool arguments object was available. "
            "Return a shorter but complete schema-valid arguments object."
        )
    else:
        issue = "The previous response failed structured output compliance."

    if result.output_mode == "deepseek_tool_call_strict":
        channel_rule = "Call the required function exactly once and put the corrected object in function arguments."
    else:
        channel_rule = "Return exactly one corrected JSON object."

    manifest_section = ""
    if schema_manifest_text:
        manifest_section = f"\nCanonical schema manifest:\n{schema_manifest_text}\n"
    drift_section = ""
    if drift_report_text:
        drift_section = f"\nSchema drift report:\n{drift_report_text}\n"
    guard_section = ""
    if drift_guard is not None:
        guard_section = (
            "\nDrift guard:\n"
            f"- Forbidden output fields: {', '.join(drift_guard.forbidden_output_fields) or '<none>'}\n"
            f"- Canonical aliases are forbidden: {json.dumps(drift_guard.canonical_aliases, ensure_ascii=False, sort_keys=True)}\n"
        )

    return (
        "Structured output correction required.\n"
        f"- Node: {node_name or result.node_name or '<unknown>'}\n"
        f"- Schema: {schema_name}\n"
        f"- Previous failure_phase: {phase}\n"
        f"- Issue: {issue}\n"
        f"- Previous error summary: {previous_error_summary}\n"
        f"- {channel_rule}\n"
        "- Keep already-correct canonical fields unchanged.\n"
        "- Add missing required fields.\n"
        "- Remove extra fields not listed in the manifest.\n"
        "- Use canonical field names exactly.\n"
        "- Do not output aliases, translations, abbreviations, or wrapper keys.\n"
        "- If enum drift occurred, use one of the allowed enum values only.\n"
        "- If a non-empty business rule failed, provide a meaningful non-empty value.\n"
        "- Return exactly one structured tool/function result matching the schema.\n"
        "- Do not explain the error.\n"
        "- Do not output markdown, code fences, comments, or extra prose.\n"
        "- Do not change the schema, omit fields, invent fields, or use non-schema enum values.\n"
        "- If the previous error says \"Extra inputs are not permitted\", remove all fields not defined in the schema.\n"
        "- Do not include input-only metadata fields unless they are explicitly present in the schema.\n"
        "- If the previous error says \"Field required\", add that exact required field to every affected object.\n"
        "- Do not fix one validation error by introducing extra fields.\n"
        "- If a field is missing and the schema permits an empty value, you may use a "
        "schema-compatible empty value such as \"\", [], false, or 0. However, if the "
        "validation error says non-empty or the business validation error says a field "
        "must be non-empty, you must provide a valid non-empty value. Do not use an "
        "empty value to satisfy a field that failed a non-empty business rule."
        f"{manifest_section}"
        f"{drift_section}"
        f"{guard_section}"
    )


def _build_reask_context(
    *,
    node_name: str,
    schema_name: str,
    schema: type[BaseModel],
    result: StructuredLLMResult,
    attempt_number: int,
) -> _ReaskContext | None:
    phase = result.failure_phase or ""
    if not _reask_enabled(node_name):
        return None
    if phase not in {
        "parsing_error",
        "validation_error",
        "business_validation_error",
        "deepseek_tool_call_missing",
        "deepseek_empty_tool_arguments",
        "deepseek_tool_call_truncated",
    }:
        return None
    if phase == "business_validation_error" and not _reask_business_validation_enabled(node_name):
        return None

    previous_error_summary = _structured_error_summary(result)
    manifest, drift_guard, schema_manifest_text, manifest_debug = _schema_manifest_context(
        schema=schema,
        node_name=node_name,
        mode=result.output_mode,
    )
    drift_report_text, drift_debug = _schema_drift_report_debug(
        raw_output=result.raw_output,
        manifest=manifest,
        drift_guard=drift_guard,
        node_name=node_name,
    )
    return _ReaskContext(
        instruction=_build_reask_instruction(
            result=result,
            schema_name=schema_name,
            previous_error_summary=previous_error_summary,
            node_name=node_name,
            schema_manifest_text=schema_manifest_text,
            drift_report_text=drift_report_text,
            drift_guard=drift_guard,
        ),
        reason=phase,
        attempt_number=attempt_number,
        previous_failure_phase=phase,
        previous_error_summary=previous_error_summary,
        schema_manifest_summary=manifest_debug["schema_manifest"],
        schema_drift_report=drift_debug,
        drift_guard_source=manifest_debug["drift_guard_source"],
        drift_guard_config_validated=bool(manifest_debug["drift_guard_config_validated"]),
    )


def _append_reask_message(messages: list, instruction: str) -> list:
    base_messages = list(messages or [])
    if base_messages and isinstance(base_messages[0], dict):
        return [*base_messages, {"role": "user", "content": instruction}]
    return [*base_messages, HumanMessage(content=instruction)]


def _is_second_layer_unsupported(exc: Exception, mode: str) -> bool:
    text = f"{type(exc).__name__} {exc} {_extract_provider_error_body(exc)}".lower()
    return any(
        term in text
        for term in (
            "unsupported",
            "not support",
            "not supported",
            "response_format",
            "json_schema",
            "structured output",
            "structured outputs",
            "no endpoints found",
            "can handle the requested parameters",
        )
    )


def _validate_mode(mode: str) -> None:
    if mode not in ALLOWED_OUTPUT_MODES:
        raise ValueError(
            f"Unsupported output_mode={mode!r}. Allowed modes: {sorted(ALLOWED_OUTPUT_MODES)}"
        )


def _json_output_contract_with_debug(
    schema: type[BaseModel],
    node_name: str,
    mode: str,
) -> tuple[str, dict[str, Any]]:
    contract = (
        "Structured output contract for this call:\n"
        f"- Node: {node_name}\n"
        f"- Schema: {schema.__name__}\n"
        f"- Output mode: {mode}\n"
        "- Return exactly one valid JSON object matching the configured Pydantic schema.\n"
        "- Do not output markdown, code fences, comments, explanations, or extra text.\n"
        "- Do not omit required fields. Use only schema-compatible enum values.\n"
        "- Use canonical field names exactly. Do not output aliases, translations, abbreviations, wrapper keys, or extra fields.\n"
        "- If a business rule requires a non-empty value, provide a meaningful non-empty value.\n"
        "- If unsure, still return the best schema-valid JSON object; never answer in prose."
    )
    if mode == "deepseek_tool_call_strict":
        contract += (
            "\n- Do not omit any field. If empty, use \"\", [], false, or 0 according to the schema."
        )
    elif mode == "deepseek_json_object":
        contract += (
            "\n- The response_format is json_object, so the response must be valid json."
            f"\n- Example json object shape: {json.dumps(_minimal_json_example(schema), ensure_ascii=False)}"
        )
    _manifest, _drift_guard, manifest_text, debug = _schema_manifest_context(
        schema=schema,
        node_name=node_name,
        mode=mode,
    )
    if manifest_text:
        contract += f"\n\n{manifest_text}"
    elif not debug.get("manifest_enabled", True):
        contract += "\n\nCanonical schema manifest injection is disabled by configuration."
    return contract, debug


def _json_output_contract(schema: type[BaseModel], node_name: str, mode: str) -> str:
    contract, _debug = _json_output_contract_with_debug(schema, node_name, mode)
    return contract


def _inject_json_contract(messages: list, *, schema: type[BaseModel], node_name: str, mode: str) -> list:
    contract, _debug = _json_output_contract_with_debug(schema, node_name, mode)
    if messages and isinstance(messages[0], dict):
        return [{"role": "system", "content": contract}, *messages]
    return [SystemMessage(content=contract), *messages]


def _inject_json_contract_with_debug(
    messages: list,
    *,
    schema: type[BaseModel],
    node_name: str,
    mode: str,
) -> tuple[list, dict[str, Any]]:
    contract, debug = _json_output_contract_with_debug(schema, node_name, mode)
    if messages and isinstance(messages[0], dict):
        return [{"role": "system", "content": contract}, *messages], debug
    return [SystemMessage(content=contract), *messages], debug


def _tool_schema(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.__name__,
            "description": f"Return a {schema.__name__} object.",
            "parameters": schema.model_json_schema(),
        },
    }


def _minimal_json_example(schema: type[BaseModel]) -> dict[str, Any]:
    example: dict[str, Any] = {}
    fields = getattr(schema, "model_fields", {}) or {}
    for name, model_field in fields.items():
        annotation = getattr(model_field, "annotation", None)
        annotation_text = str(annotation)
        if "bool" in annotation_text:
            value: Any = False
        elif "int" in annotation_text:
            value = 0
        elif "float" in annotation_text:
            value = 0.0
        elif "list" in annotation_text or "List" in annotation_text:
            value = []
        else:
            value = ""
        example[str(name)] = value
        if len(example) >= 6:
            break
    return example or {"result": {}}


def _deepseek_tool_name(node_name: str, schema: type[BaseModel]) -> str:
    raw = f"{node_name}_{schema.__name__}"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return (cleaned or schema.__name__)[:64]


def _deepseek_api_key(llm_node: str) -> tuple[str, str]:
    api_key_env = str(_setting(llm_node, "api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY")
    return api_key_env, os.getenv(api_key_env, "").strip()


def _deepseek_base_url(llm_node: str, *, beta: bool) -> str:
    key = "beta_base_url" if beta else "base_url"
    default = "https://api.deepseek.com/beta" if beta else "https://api.deepseek.com"
    return str(_setting(llm_node, key, default) or default).rstrip("/")


def _deepseek_request_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _deepseek_temperature(llm_node: str) -> float:
    try:
        return float(_setting(llm_node, "temperature", 0.0) or 0.0)
    except Exception:
        return 0.0


def _deepseek_max_tokens(llm_node: str) -> int:
    try:
        return int(_setting(llm_node, "max_tokens", 1024) or 1024)
    except Exception:
        return 1024


def _deepseek_thinking(llm_node: str) -> dict[str, str] | None:
    value = str(_setting(llm_node, "thinking", "") or "").strip().lower()
    if value in {"enabled", "disabled"}:
        return {"type": value}
    return None


def _deepseek_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return {}
    return choices[0]


def _deepseek_message_from_choice(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message", {})
    return message if isinstance(message, dict) else {}


def _deepseek_raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    err = RuntimeError(f"Error code: {response.status_code} - {response.text}")
    err.response = response  # for _extract_status_code / _extract_provider_error_body
    raise err


async def _deepseek_chat_completion(
    *,
    payload: dict[str, Any],
    base_url: str,
    api_key: str,
    node_name: str,
    llm_node: str,
    state: dict | None,
) -> tuple[httpx.Response, int]:
    async def _post_request():
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=_deepseek_request_headers(api_key),
                json=payload,
            )
        _deepseek_raise_for_status(response)
        try:
            data = response.json()
        except Exception as exc:
            raise DeepSeekProviderResponseJSONError(
                f"DeepSeek provider returned a non-JSON response envelope: {exc}",
                response=response,
            ) from exc
        choice = _deepseek_choice(data)
        finish_reason = str(choice.get("finish_reason", "") or "")
        if finish_reason == "insufficient_system_resource":
            raise DeepSeekInsufficientSystemResourceError(
                "DeepSeek provider returned finish_reason=insufficient_system_resource",
                response=response,
            )
        return response

    return await invoke_with_provider_transport_retry(
        _post_request,
        node_name=node_name,
        llm_node=llm_node,
        provider=_provider(llm_node),
        model=_model(llm_node),
        output_mode="deepseek_tool_call_strict" if base_url.endswith("/beta") else "deepseek_json_object",
        trace_stage_prefix="structured_llm_transport",
        state=state or {},
    )


async def _invoke_openrouter_native(
    schema: type[BaseModel],
    messages: list,
    metrics: _InvokeMetrics,
    llm_node: str,
    node_name: str,
    state: dict | None = None,
) -> tuple[BaseModel, str, _InvokeMetrics]:
    """Direct httpx call for OpenRouter native json_schema.

    Bypasses ChatOpenAI defaults that break require_parameters=true.
    """
    metrics.using_direct_openrouter_http = True
    metrics.provider_request_mode = "openrouter_direct_http"
    base_url = str(_setting(llm_node, "base_url", "https://openrouter.ai/api/v1")).rstrip("/")
    api_key_env = str(_setting(llm_node, "api_key_env", "OPENROUTER_API_KEY"))
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise _InvokeOneModeError(
            RuntimeError(f"{api_key_env} is not configured"),
            metrics,
        )

    openai_messages = normalize_openai_messages(messages)
    metrics.http_messages_preview = preview_openai_messages(openai_messages)
    try:
        validate_openai_messages(openai_messages)
    except Exception as exc:
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

    payload: dict[str, Any] = {
        "model": _model(llm_node),
        "messages": openai_messages,
        "temperature": 0,
        "max_tokens": int(_setting(llm_node, "max_tokens", 4096) or 4096),
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        },
        "provider": {"require_parameters": True},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title

    llm_started = time.perf_counter()
    try:
        async def _post_request():
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            if response.status_code >= 400:
                err = RuntimeError(
                    f"Error code: {response.status_code} - {response.text}"
                )
                err.response = response  # for _extract_status_code / _extract_provider_error_body
                raise err
            return response

        response, _transport_retry_count = await invoke_with_provider_transport_retry(
            _post_request,
            node_name=node_name,
            llm_node=llm_node,
            provider=_provider(llm_node),
            model=_model(llm_node),
            state=state or {},
        )
        metrics.llm_elapsed_ms = _round_ms(llm_started)
    except Exception as exc:
        metrics.llm_elapsed_ms = _round_ms(llm_started)
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

    try:
        data = response.json()
    except Exception as exc:
        metrics.raw_output_chars = len(response.text or "")
        raise _InvokeOneModeError(exc, metrics, raw_output=response.text) from exc
    choices = data.get("choices", [])
    raw_output = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            raw_output = "\n".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            raw_output = str(content or "")
    metrics.raw_output_chars = len(raw_output or "")

    parse_started = time.perf_counter()
    try:
        parsed = schema.model_validate(_extract_json_object(raw_output))
        metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
    except Exception as exc:
        metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        _refresh_parse_validate_total(metrics)
        raise _InvokeOneModeError(exc, metrics, raw_output=raw_output) from exc

    _refresh_parse_validate_total(metrics)
    return parsed, raw_output, metrics


async def _invoke_deepseek_tool_call_strict(
    schema: type[BaseModel],
    messages: list,
    metrics: _InvokeMetrics,
    llm_node: str,
    node_name: str,
    state: dict | None = None,
) -> tuple[BaseModel, str, _InvokeMetrics]:
    """Direct DeepSeek official strict tool-call invocation."""
    metrics.using_direct_openrouter_http = False
    metrics.provider_request_mode = "deepseek_tool_call_strict"
    tool_name = _deepseek_tool_name(node_name, schema)
    base_url = _deepseek_base_url(llm_node, beta=True)
    metrics.extra_debug.update({
        "using_deepseek_official_http": True,
        "base_url_type": "beta",
        "resolved_base_url": base_url,
        "deepseek_schema_size_chars": 0,
        "tool_name": tool_name,
        "tool_call_present": False,
        "tool_arguments_chars": 0,
        "finish_reason": "",
    })

    api_key_env, api_key = _deepseek_api_key(llm_node)
    if not api_key:
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "provider_auth_error",
                f"{api_key_env} is not configured",
            ),
            metrics,
            raw_output="",
        )

    try:
        deepseek_schema = compile_pydantic_schema_for_deepseek_tool(schema)
    except Exception as exc:
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc
    metrics.extra_debug["deepseek_schema_size_chars"] = len(
        json.dumps(deepseek_schema, ensure_ascii=False, default=str)
    )

    openai_messages = normalize_openai_messages(messages)
    metrics.http_messages_preview = preview_openai_messages(openai_messages)
    try:
        validate_openai_messages(openai_messages)
    except Exception as exc:
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

    payload: dict[str, Any] = {
        "model": _model(llm_node),
        "messages": openai_messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Return a {schema.__name__} object for {node_name}.",
                    "strict": True,
                    "parameters": deepseek_schema,
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": tool_name},
        },
        "temperature": _deepseek_temperature(llm_node),
        "max_tokens": _deepseek_max_tokens(llm_node),
        "stream": False,
        "thinking": {"type": "disabled"},
    }

    llm_started = time.perf_counter()
    try:
        response, _transport_retry_count = await _deepseek_chat_completion(
            payload=payload,
            base_url=base_url,
            api_key=api_key,
            node_name=node_name,
            llm_node=llm_node,
            state=state,
        )
        metrics.llm_elapsed_ms = _round_ms(llm_started)
    except Exception as exc:
        metrics.llm_elapsed_ms = _round_ms(llm_started)
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

    try:
        data = response.json()
    except Exception as exc:
        metrics.raw_output_chars = len(response.text or "")
        raise _InvokeOneModeError(exc, metrics, raw_output=response.text) from exc

    choice = _deepseek_choice(data)
    finish_reason = str(choice.get("finish_reason", "") or "")
    metrics.extra_debug["finish_reason"] = finish_reason
    if finish_reason == "insufficient_system_resource":
        raise _InvokeOneModeError(
            DeepSeekInsufficientSystemResourceError(
                "DeepSeek provider returned finish_reason=insufficient_system_resource",
                response=response,
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )
    if finish_reason == "content_filter":
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "provider_content_filter",
                "DeepSeek response was blocked by provider content_filter",
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )
    if finish_reason == "length":
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "deepseek_tool_call_truncated",
                "DeepSeek response was truncated before complete tool arguments were returned",
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )
    message = _deepseek_message_from_choice(choice)
    tool_calls = message.get("tool_calls") or []
    metrics.extra_debug["tool_call_present"] = bool(tool_calls)
    if not tool_calls:
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "deepseek_tool_call_missing",
                "DeepSeek response did not include a tool call",
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )

    tool_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    returned_name = str(function.get("name") or "")
    if returned_name != tool_name:
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "deepseek_wrong_tool_name",
                f"Expected DeepSeek tool {tool_name!r}, got {returned_name!r}",
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )

    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        raw_arguments = json.dumps(arguments, ensure_ascii=False)
    else:
        raw_arguments = str(arguments or "")
    metrics.raw_output_chars = len(raw_arguments)
    metrics.extra_debug["tool_arguments_chars"] = len(raw_arguments)
    if not raw_arguments.strip():
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "deepseek_empty_tool_arguments",
                "DeepSeek tool call returned empty arguments",
            ),
            metrics,
            raw_output="",
        )

    parse_started = time.perf_counter()
    try:
        parsed_args = json.loads(raw_arguments)
        if not isinstance(parsed_args, dict):
            raise ValueError("DeepSeek tool arguments must be a JSON object")
        parsed = schema.model_validate(parsed_args)
        metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
    except Exception as exc:
        metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        _refresh_parse_validate_total(metrics)
        raise _InvokeOneModeError(exc, metrics, raw_output=raw_arguments) from exc

    _refresh_parse_validate_total(metrics)
    return parsed, raw_arguments, metrics


async def _invoke_deepseek_json_object(
    schema: type[BaseModel],
    messages: list,
    metrics: _InvokeMetrics,
    llm_node: str,
    node_name: str,
    state: dict | None = None,
) -> tuple[BaseModel, str, _InvokeMetrics]:
    """Direct DeepSeek official JSON object invocation."""
    metrics.using_direct_openrouter_http = False
    metrics.provider_request_mode = "deepseek_json_object"
    base_url = _deepseek_base_url(llm_node, beta=False)
    metrics.extra_debug.update({
        "using_deepseek_official_http": True,
        "base_url_type": "stable",
        "resolved_base_url": base_url,
        "deepseek_schema_size_chars": 0,
        "tool_name": "",
        "tool_call_present": False,
        "tool_arguments_chars": 0,
        "finish_reason": "",
    })
    try:
        deepseek_schema = compile_pydantic_schema_for_deepseek_tool(schema)
        metrics.extra_debug["deepseek_schema_size_chars"] = len(
            json.dumps(deepseek_schema, ensure_ascii=False, default=str)
        )
    except Exception:
        # JSON object mode does not send schema, so schema compile diagnostics are best effort.
        metrics.extra_debug["deepseek_schema_size_chars"] = 0

    api_key_env, api_key = _deepseek_api_key(llm_node)
    if not api_key:
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "provider_auth_error",
                f"{api_key_env} is not configured",
            ),
            metrics,
            raw_output="",
        )

    openai_messages = normalize_openai_messages(messages)
    metrics.http_messages_preview = preview_openai_messages(openai_messages)
    try:
        validate_openai_messages(openai_messages)
    except Exception as exc:
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

    payload: dict[str, Any] = {
        "model": _model(llm_node),
        "messages": openai_messages,
        "response_format": {"type": "json_object"},
        "temperature": _deepseek_temperature(llm_node),
        "max_tokens": _deepseek_max_tokens(llm_node),
        "stream": False,
    }
    thinking = _deepseek_thinking(llm_node)
    if thinking is not None:
        payload["thinking"] = thinking

    llm_started = time.perf_counter()
    try:
        response, _transport_retry_count = await _deepseek_chat_completion(
            payload=payload,
            base_url=base_url,
            api_key=api_key,
            node_name=node_name,
            llm_node=llm_node,
            state=state,
        )
        metrics.llm_elapsed_ms = _round_ms(llm_started)
    except Exception as exc:
        metrics.llm_elapsed_ms = _round_ms(llm_started)
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

    try:
        data = response.json()
    except Exception as exc:
        metrics.raw_output_chars = len(response.text or "")
        raise _InvokeOneModeError(exc, metrics, raw_output=response.text) from exc

    choice = _deepseek_choice(data)
    finish_reason = str(choice.get("finish_reason", "") or "")
    metrics.extra_debug["finish_reason"] = finish_reason
    if finish_reason == "insufficient_system_resource":
        raise _InvokeOneModeError(
            DeepSeekInsufficientSystemResourceError(
                "DeepSeek provider returned finish_reason=insufficient_system_resource",
                response=response,
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )
    if finish_reason == "content_filter":
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "provider_content_filter",
                "DeepSeek response was blocked by provider content_filter",
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )
    if finish_reason == "length":
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "deepseek_json_truncated",
                "DeepSeek JSON output was truncated by max_tokens",
            ),
            metrics,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
        )

    message = _deepseek_message_from_choice(choice)
    content = _message_text(message.get("content", ""))
    metrics.raw_output_chars = len(content or "")
    if not content.strip():
        raise _InvokeOneModeError(
            _DeepSeekStructuredOutputError(
                "deepseek_json_empty_content",
                "DeepSeek JSON output returned empty content",
            ),
            metrics,
            raw_output="",
        )

    parse_started = time.perf_counter()
    try:
        parsed = schema.model_validate(_extract_json_object(content))
        metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
    except Exception as exc:
        metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        _refresh_parse_validate_total(metrics)
        raise _InvokeOneModeError(exc, metrics, raw_output=content) from exc

    _refresh_parse_validate_total(metrics)
    return parsed, content, metrics


async def _invoke_one_mode(
    *,
    node_name: str,
    llm_node: str,
    schema: type[BaseModel],
    messages: list,
    mode: str,
    state: dict | None = None,
    reask_context: _ReaskContext | None = None,
) -> tuple[BaseModel, str, _InvokeMetrics]:
    metrics = _InvokeMetrics()
    try:
        messages, contract_debug = _inject_json_contract_with_debug(
            messages,
            schema=schema,
            node_name=node_name,
            mode=mode,
        )
    except _StructuredManifestError as exc:
        raise _InvokeOneModeError(exc, metrics, raw_output="") from exc
    except Exception as exc:
        wrapped = _StructuredManifestError(
            "schema_manifest_error",
            f"Failed to inject structured output schema manifest: {exc}",
        )
        raise _InvokeOneModeError(wrapped, metrics, raw_output="") from exc
    metrics.prompt_chars = _compute_prompt_chars(messages)
    metrics.extra_debug.update(contract_debug)
    if reask_context is not None:
        metrics.extra_debug.update(reask_context.to_debug())
    emit_context_usage_trace(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        provider=_provider(llm_node),
        model=_model(llm_node),
        messages=messages,
        state=state or {},
        schema_size_chars=_safe_schema_size_chars(schema),
    )
    emit_context_items_shadow(
        logger,
        node_name=node_name,
        llm_node=llm_node,
        messages=messages,
        state=state or {},
    )

    # ── constrained_decoding (reserved) ──
    if mode == "constrained_decoding":
        raise _InvokeOneModeError(
            NotImplementedError("constrained_decoding is reserved but not implemented"),
            metrics,
            raw_output="",
        )

    # ── native_json_schema_pydantic ──
    if mode == "deepseek_tool_call_strict":
        return await _invoke_deepseek_tool_call_strict(
            schema,
            messages,
            metrics,
            llm_node,
            node_name=node_name,
            state=state,
        )

    if mode == "deepseek_json_object":
        return await _invoke_deepseek_json_object(
            schema,
            messages,
            metrics,
            llm_node,
            node_name=node_name,
            state=state,
        )

    llm = get_node_llm(llm_node)

    if mode == "native_json_schema_pydantic":
        if _provider(llm_node) == "openrouter":
            # Direct httpx: bypass ChatOpenAI default params (frequency_penalty,
            # presence_penalty, top_p, n) that break require_parameters=true.
            return await _invoke_openrouter_native(
                schema,
                messages,
                metrics,
                llm_node,
                node_name=node_name,
                state=state,
            )

        metrics.using_direct_openrouter_http = False
        metrics.provider_request_mode = "langchain_bind_native_schema"
        runnable = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        )
        llm_started = time.perf_counter()
        try:
            response, _transport_retry_count = await invoke_with_provider_transport_retry(
                lambda: runnable.ainvoke(messages),
                node_name=node_name,
                llm_node=llm_node,
                provider=_provider(llm_node),
                model=_model(llm_node),
                state=state or {},
            )
            metrics.llm_elapsed_ms = _round_ms(llm_started)
        except Exception as exc:
            metrics.llm_elapsed_ms = _round_ms(llm_started)
            raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

        raw_output = _raw_output_from_response(response)
        metrics.raw_output_chars = len(raw_output or "")

        parse_started = time.perf_counter()
        try:
            parsed = schema.model_validate(_extract_json_object(raw_output))
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        except Exception as exc:
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
            _refresh_parse_validate_total(metrics)
            raise _InvokeOneModeError(exc, metrics, raw_output=raw_output) from exc

        _refresh_parse_validate_total(metrics)
        return parsed, raw_output, metrics

    # ── json_mode_pydantic ──
    if mode == "json_mode_pydantic":
        metrics.using_direct_openrouter_http = False
        metrics.provider_request_mode = "langchain_json_mode"
        runnable = llm.bind(response_format={"type": "json_object"})
        llm_started = time.perf_counter()
        try:
            response, _transport_retry_count = await invoke_with_provider_transport_retry(
                lambda: runnable.ainvoke(messages),
                node_name=node_name,
                llm_node=llm_node,
                provider=_provider(llm_node),
                model=_model(llm_node),
                state=state or {},
            )
            metrics.llm_elapsed_ms = _round_ms(llm_started)
        except Exception as exc:
            metrics.llm_elapsed_ms = _round_ms(llm_started)
            raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

        raw_output = _raw_output_from_response(response)
        metrics.raw_output_chars = len(raw_output or "")

        parse_started = time.perf_counter()
        try:
            parsed = schema.model_validate(_extract_json_object(raw_output))
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        except Exception as exc:
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
            _refresh_parse_validate_total(metrics)
            raise _InvokeOneModeError(exc, metrics, raw_output=raw_output) from exc

        _refresh_parse_validate_total(metrics)
        return parsed, raw_output, metrics

    # ── tool_call_pydantic ──
    if mode == "tool_call_pydantic":
        metrics.using_direct_openrouter_http = False
        metrics.provider_request_mode = "langchain_tool_call"
        runnable = llm.bind(tools=[_tool_schema(schema)], tool_choice={"type": "function", "function": {"name": schema.__name__}})
        llm_started = time.perf_counter()
        try:
            response, _transport_retry_count = await invoke_with_provider_transport_retry(
                lambda: runnable.ainvoke(messages),
                node_name=node_name,
                llm_node=llm_node,
                provider=_provider(llm_node),
                model=_model(llm_node),
                state=state or {},
            )
            metrics.llm_elapsed_ms = _round_ms(llm_started)
        except Exception as exc:
            metrics.llm_elapsed_ms = _round_ms(llm_started)
            raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

        raw_output = _raw_output_from_response(response)
        metrics.raw_output_chars = len(raw_output or "")

        parse_started = time.perf_counter()
        try:
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
            parsed = schema.model_validate(parsed_args)
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        except Exception as exc:
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
            _refresh_parse_validate_total(metrics)
            raise _InvokeOneModeError(exc, metrics, raw_output=raw_output or json.dumps(parsed_args if 'parsed_args' in dir() else {}, ensure_ascii=False)) from exc

        _refresh_parse_validate_total(metrics)
        return parsed, raw_output or json.dumps(parsed_args, ensure_ascii=False), metrics

    # ── prompt_json_pydantic ──
    if mode == "prompt_json_pydantic":
        metrics.using_direct_openrouter_http = False
        metrics.provider_request_mode = "langchain_prompt_json"
        llm_started = time.perf_counter()
        try:
            response, _transport_retry_count = await invoke_with_provider_transport_retry(
                lambda: llm.ainvoke(messages),
                node_name=node_name,
                llm_node=llm_node,
                provider=_provider(llm_node),
                model=_model(llm_node),
                state=state or {},
            )
            metrics.llm_elapsed_ms = _round_ms(llm_started)
        except Exception as exc:
            metrics.llm_elapsed_ms = _round_ms(llm_started)
            raise _InvokeOneModeError(exc, metrics, raw_output="") from exc

        raw_output = _raw_output_from_response(response)
        metrics.raw_output_chars = len(raw_output or "")

        parse_started = time.perf_counter()
        try:
            parsed = schema.model_validate(_extract_json_object(raw_output))
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
        except Exception as exc:
            metrics.json_pydantic_elapsed_ms = _round_ms(parse_started)
            _refresh_parse_validate_total(metrics)
            raise _InvokeOneModeError(exc, metrics, raw_output=raw_output) from exc

        _refresh_parse_validate_total(metrics)
        return parsed, raw_output, metrics

    raise _InvokeOneModeError(
        ValueError(f"Unsupported output mode {mode!r}"),
        metrics,
        raw_output="",
    )


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

    Retryable parsing/schema failures are retried before fail-fast raises.
    Fallback modes remain configuration-only in development fail-fast mode.
    """
    llm_node = llm_node or node_name
    fail_fast = _fail_fast_enabled()
    requested_fallback_modes = list(fallback_modes or [])
    effective_fallback_modes = [] if fail_fast else requested_fallback_modes
    base_modes = [output_mode, *effective_fallback_modes]
    max_retries = get_llm_call_max_retries(node_name)
    modes = [mode for mode in base_modes for _ in range(max_retries + 1)]
    provider = _provider(llm_node)
    model = _model(llm_node)
    schema_name = schema.__name__
    raw_limit = int(max_raw_chars or get_max_raw_chars(node_name))
    failure_policy = _failure_policy(node_name)
    schema_size_chars = _safe_schema_size_chars(schema)
    attempts: list[StructuredLLMAttempt] = []
    terminal_modes: set[str] = set()
    pending_reasks: dict[str, _ReaskContext] = {}

    def _make_metrics_from_attempt(attempt: StructuredLLMAttempt) -> _InvokeMetrics:
        return _InvokeMetrics(
            total_elapsed_ms=attempt.total_elapsed_ms,
            llm_elapsed_ms=attempt.llm_elapsed_ms,
            json_pydantic_elapsed_ms=attempt.json_pydantic_elapsed_ms,
            business_validate_elapsed_ms=attempt.business_validate_elapsed_ms,
            parse_validate_elapsed_ms=attempt.parse_validate_elapsed_ms,
            prompt_chars=attempt.prompt_chars,
            raw_output_chars=attempt.raw_output_chars,
            using_direct_openrouter_http=attempt.using_direct_openrouter_http,
            provider_request_mode=attempt.provider_request_mode,
            http_messages_preview=attempt.http_messages_preview,
            extra_debug=dict(attempt.extra_debug),
        )

    def _emit_and_maybe_raise(result: StructuredLLMResult, *, exc: Exception | None = None) -> None:
        result.retry_count = _semantic_retry_count(attempts)
        emit_a3_trace(
            logger,
            "structured_llm_output",
            result.to_debug_payload(max_raw_chars=raw_limit),
            state=state,
            env_flag="LOG_STRUCTURED_LLM_OUTPUT",
            max_chars=raw_limit,
        )
        attempts_for_mode = sum(1 for attempt in attempts if attempt.output_mode == result.output_mode)
        include_business_validation_retry = (
            _reask_enabled(node_name) and _reask_business_validation_enabled(node_name)
        )
        should_retry = (
            not result.success
            and _is_semantic_retryable_failure(
                result.failure_phase,
                include_business_validation=include_business_validation_retry,
            )
            and attempts_for_mode <= max_retries
        )
        reask_context = (
            _build_reask_context(
                node_name=node_name,
                schema_name=schema_name,
                schema=schema,
                result=result,
                attempt_number=attempts_for_mode,
            )
            if should_retry
            else None
        )
        if should_retry:
            reask_debug = reask_context.to_debug() if reask_context is not None else {
                "reask_used": False,
                "reask_reason": "",
                "reask_attempt_number": 0,
                "previous_failure_phase": "",
                "previous_error_summary": "",
            }
            if reask_context is not None:
                pending_reasks[result.output_mode] = reask_context
                emit_a3_trace(
                    logger,
                    "structured_llm_reask_attempt",
                    {
                        "node_name": node_name,
                        "llm_node": llm_node,
                        "schema_name": schema_name,
                        "provider": provider,
                        "model": model,
                        "output_mode": result.output_mode,
                        "failure_phase": result.failure_phase,
                        "error_type": result.error_type,
                        "error_message": _sanitize(result.error_message, max_chars=1200),
                        "max_retries": max_retries,
                        "next_attempt": attempts_for_mode + 1,
                        "fallback_used": result.fallback_used,
                        **reask_debug,
                    },
                    state=state,
                    env_flag="LOG_STRUCTURED_LLM_OUTPUT",
                    max_chars=raw_limit,
                )
            emit_a3_trace(
                logger,
                "structured_llm_retry_attempt",
                {
                    "node_name": node_name,
                    "llm_node": llm_node,
                    "schema_name": schema_name,
                    "provider": provider,
                    "model": model,
                    "output_mode": result.output_mode,
                    "failure_phase": result.failure_phase,
                    "error_type": result.error_type,
                    "error_message": _sanitize(result.error_message, max_chars=1200),
                    "retry_count": attempts_for_mode,
                    "max_retries": max_retries,
                    "next_attempt": attempts_for_mode + 1,
                    "fallback_used": result.fallback_used,
                    **reask_debug,
                },
                state=state,
                env_flag="LOG_STRUCTURED_LLM_OUTPUT",
                max_chars=raw_limit,
            )
            return
        if not result.success:
            terminal_modes.add(result.output_mode)
        if fail_fast and not result.success:
            if exc is not None:
                raise StructuredOutputError(result) from exc
            raise StructuredOutputError(result)

    for mode in modes:
        if mode in terminal_modes:
            continue
        reask_context = pending_reasks.pop(mode, None)
        attempt_messages = (
            _append_reask_message(messages, reask_context.instruction)
            if reask_context is not None
            else messages
        )
        attempt_started = time.perf_counter()
        metrics = _InvokeMetrics()
        try:
            _validate_mode(mode)
        except Exception as exc:
            metrics.total_elapsed_ms = _round_ms(attempt_started)
            attempt = StructuredLLMAttempt(
                output_mode=mode,
                success=False,
                failure_phase="invalid_output_mode",
                error_type=type(exc).__name__,
                error_message=str(exc),
                total_elapsed_ms=metrics.total_elapsed_ms,
                schema_size_chars=schema_size_chars,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
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
                total_elapsed_ms=metrics.total_elapsed_ms,
                schema_size_chars=schema_size_chars,
                fail_fast=fail_fast,
                fallback_used=False,
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
            )
            _emit_and_maybe_raise(result, exc=exc)
            continue

        # ── success path ──
        try:
            parsed, raw_output, metrics = await _invoke_one_mode(
                node_name=node_name,
                llm_node=llm_node,
                schema=schema,
                messages=attempt_messages,
                mode=mode,
                state=state,
                reask_context=reask_context,
            )
            # business validation with timing
            business_started = time.perf_counter()
            business_error = ""
            if business_validator is not None:
                try:
                    validation_result = business_validator(parsed)
                except Exception as exc:
                    validation_result = str(exc)
                if isinstance(validation_result, list):
                    business_error = "; ".join(str(item) for item in validation_result if item)
                elif validation_result:
                    business_error = str(validation_result)
            metrics.business_validate_elapsed_ms = _round_ms(business_started)
            _refresh_parse_validate_total(metrics)

            if business_error:
                _attach_schema_drift_debug(
                    metrics,
                    schema=schema,
                    node_name=node_name,
                    mode=mode,
                    raw_output=raw_output,
                )
                attempt = StructuredLLMAttempt(
                    output_mode=mode,
                    success=False,
                    failure_phase="business_validation_error",
                    error_type="BusinessValidationError",
                    error_message=business_error,
                    total_elapsed_ms=metrics.total_elapsed_ms,
                    llm_elapsed_ms=metrics.llm_elapsed_ms,
                    json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                    business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                    parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                    prompt_chars=metrics.prompt_chars,
                    raw_output_chars=metrics.raw_output_chars,
                    schema_size_chars=schema_size_chars,
                    using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                    provider_request_mode=metrics.provider_request_mode,
                    http_messages_preview=metrics.http_messages_preview,
                    extra_debug=dict(metrics.extra_debug),
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
                    raw_output=raw_output,
                    failure_phase="business_validation_error",
                    error_type="BusinessValidationError",
                    error_message=business_error,
                    business_validation_error=business_error,
                    total_elapsed_ms=metrics.total_elapsed_ms,
                    llm_elapsed_ms=metrics.llm_elapsed_ms,
                    json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                    business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                    parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                    prompt_chars=metrics.prompt_chars,
                    raw_output_chars=metrics.raw_output_chars,
                    schema_size_chars=schema_size_chars,
                    fail_fast=fail_fast,
                    fallback_used=(mode != output_mode),
                    default_used=False,
                    retry_count=0,
                    failure_policy=failure_policy,
                    using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                    provider_request_mode=metrics.provider_request_mode,
                    http_messages_preview=metrics.http_messages_preview,
                    extra_debug=dict(metrics.extra_debug),
                )
                _emit_and_maybe_raise(result, exc=_BusinessValidationError(business_error))
                continue

            # success
            attempt = StructuredLLMAttempt(
                output_mode=mode,
                success=True,
                total_elapsed_ms=metrics.total_elapsed_ms,
                llm_elapsed_ms=metrics.llm_elapsed_ms,
                json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                prompt_chars=metrics.prompt_chars,
                raw_output_chars=metrics.raw_output_chars,
                schema_size_chars=schema_size_chars,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
            )
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
                total_elapsed_ms=metrics.total_elapsed_ms,
                llm_elapsed_ms=metrics.llm_elapsed_ms,
                json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                prompt_chars=metrics.prompt_chars,
                raw_output_chars=metrics.raw_output_chars,
                schema_size_chars=schema_size_chars,
                fail_fast=fail_fast,
                fallback_used=(mode != output_mode),
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
            )
            _emit_and_maybe_raise(result)
            return result

        except _InvokeOneModeError as wrapper:
            exc = wrapper.cause
            metrics = wrapper.metrics
            last_raw_output = wrapper.raw_output
            if metrics.total_elapsed_ms <= 0:
                metrics.total_elapsed_ms = _round_ms(attempt_started)

            status_code = _extract_status_code(exc)
            provider_error_body = _extract_provider_error_body(exc)
            phase = _classify_failure_phase(exc, mode, metrics)
            if _is_semantic_retryable_failure(
                phase,
                include_business_validation=_reask_business_validation_enabled(node_name),
            ):
                _attach_schema_drift_debug(
                    metrics,
                    schema=schema,
                    node_name=node_name,
                    mode=mode,
                    raw_output=last_raw_output,
                )

            attempt = StructuredLLMAttempt(
                output_mode=mode,
                success=False,
                failure_phase=phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=status_code,
                provider_error_body=provider_error_body,
                total_elapsed_ms=metrics.total_elapsed_ms,
                llm_elapsed_ms=metrics.llm_elapsed_ms,
                json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                prompt_chars=metrics.prompt_chars,
                raw_output_chars=metrics.raw_output_chars,
                schema_size_chars=schema_size_chars,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
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
                business_validation_error="",
                total_elapsed_ms=metrics.total_elapsed_ms,
                llm_elapsed_ms=metrics.llm_elapsed_ms,
                json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                prompt_chars=metrics.prompt_chars,
                raw_output_chars=metrics.raw_output_chars,
                schema_size_chars=schema_size_chars,
                fail_fast=fail_fast,
                fallback_used=(mode != output_mode),
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
            )
            _emit_and_maybe_raise(result, exc=exc)
            continue

        except StructuredOutputError:
            raise

        except Exception as exc:
            metrics.total_elapsed_ms = _round_ms(attempt_started)
            status_code = _extract_status_code(exc)
            provider_error_body = _extract_provider_error_body(exc)
            phase = _classify_failure_phase(exc, mode, metrics)
            raw_output = getattr(exc, "raw_output", "")
            if _is_semantic_retryable_failure(
                phase,
                include_business_validation=_reask_business_validation_enabled(node_name),
            ):
                _attach_schema_drift_debug(
                    metrics,
                    schema=schema,
                    node_name=node_name,
                    mode=mode,
                    raw_output=raw_output,
                )

            attempt = StructuredLLMAttempt(
                output_mode=mode,
                success=False,
                failure_phase=phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=status_code,
                provider_error_body=provider_error_body,
                total_elapsed_ms=metrics.total_elapsed_ms,
                prompt_chars=metrics.prompt_chars,
                raw_output_chars=0,
                schema_size_chars=schema_size_chars,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
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
                raw_output=raw_output,
                parsing_error=str(exc) if phase == "parsing_error" else "",
                validation_error=str(exc) if phase == "validation_error" else "",
                business_validation_error="",
                total_elapsed_ms=metrics.total_elapsed_ms,
                llm_elapsed_ms=metrics.llm_elapsed_ms,
                json_pydantic_elapsed_ms=metrics.json_pydantic_elapsed_ms,
                business_validate_elapsed_ms=metrics.business_validate_elapsed_ms,
                parse_validate_elapsed_ms=metrics.parse_validate_elapsed_ms,
                prompt_chars=metrics.prompt_chars,
                raw_output_chars=metrics.raw_output_chars,
                schema_size_chars=schema_size_chars,
                fail_fast=fail_fast,
                fallback_used=(mode != output_mode),
                default_used=False,
                retry_count=0,
                failure_policy=failure_policy,
                using_direct_openrouter_http=metrics.using_direct_openrouter_http,
                provider_request_mode=metrics.provider_request_mode,
                http_messages_preview=metrics.http_messages_preview,
                extra_debug=dict(metrics.extra_debug),
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
        total_elapsed_ms=last.total_elapsed_ms,
        llm_elapsed_ms=last.llm_elapsed_ms,
        json_pydantic_elapsed_ms=last.json_pydantic_elapsed_ms,
        business_validate_elapsed_ms=last.business_validate_elapsed_ms,
        parse_validate_elapsed_ms=last.parse_validate_elapsed_ms,
        prompt_chars=last.prompt_chars,
        raw_output_chars=last.raw_output_chars,
        schema_size_chars=schema_size_chars,
        fail_fast=fail_fast,
        fallback_used=(last.output_mode != output_mode),
        default_used=False,
        retry_count=_semantic_retry_count(attempts),
        failure_policy=failure_policy,
        using_direct_openrouter_http=last.using_direct_openrouter_http,
        provider_request_mode=last.provider_request_mode,
        http_messages_preview=last.http_messages_preview,
        extra_debug=dict(last.extra_debug),
    )
    _emit_and_maybe_raise(result)
    return result
