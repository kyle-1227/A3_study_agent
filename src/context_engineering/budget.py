"""Context Engineering budget and usage calculations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from src.config import get_setting
from src.context_engineering.policies import (
    get_thresholds,
    resolve_reserved_output_tokens,
)
from src.context_engineering.schema import (
    ContextBudget,
    ContextConfigError,
    ContextUsageError,
    ContextUsageReport,
)
from src.context_engineering.tokenizer import count_messages_tokens


def get_context_engineering_config() -> dict[str, Any]:
    """Return validated top-level Context Engineering config."""
    config = get_setting("context_engineering")
    if not isinstance(config, dict):
        raise ContextConfigError(
            "context_engineering_missing",
            "context_engineering config is required",
        )
    enabled = config.get("enabled")
    if not isinstance(enabled, bool):
        raise ContextConfigError(
            "context_engineering_enabled_invalid",
            "context_engineering.enabled must be a boolean",
        )
    if enabled is False:
        return config
    strict = config.get("strict")
    if not isinstance(strict, bool):
        raise ContextConfigError(
            "context_engineering_strict_invalid",
            "context_engineering.strict must be a boolean",
        )
    return config


def get_model_context_limit(model: str) -> int:
    """Read a model context window from context_engineering.model_limits."""
    model_name = str(model or "").strip()
    if not model_name:
        raise ContextConfigError("model_missing", "model name is required")

    config = get_context_engineering_config()
    if config.get("enabled") is not True:
        raise ContextConfigError(
            "context_engineering_disabled",
            "context engineering telemetry is disabled",
        )
    model_limits = config.get("model_limits")
    if not isinstance(model_limits, dict):
        raise ContextConfigError(
            "model_limits_missing",
            "context_engineering.model_limits is required",
        )
    if model_name not in model_limits:
        raise ContextConfigError(
            "model_window_unknown",
            "model context window is unknown",
        )
    value = model_limits[model_name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContextConfigError(
            "model_window_invalid",
            "model context window must be a positive integer",
        )
    return value


def build_context_budget(
    *,
    node_name: str,
    llm_node: str,
    model: str,
    reserved_output_tokens: int | None = None,
) -> ContextBudget:
    """Build a per-call context budget from explicit config."""
    max_context_tokens = get_model_context_limit(model)
    reserved = (
        _non_negative_int(reserved_output_tokens, "reserved_output_tokens")
        if reserved_output_tokens is not None
        else resolve_reserved_output_tokens(llm_node)
    )
    if reserved >= max_context_tokens:
        raise ContextConfigError(
            "reserved_output_tokens_invalid",
            "reserved_output_tokens must be less than max_context_tokens",
        )
    warning_ratio, critical_ratio, compact_ratio = get_thresholds()
    try:
        return ContextBudget(
            node_name=node_name,
            llm_node=llm_node,
            model=str(model or "").strip(),
            max_context_tokens=max_context_tokens,
            reserved_output_tokens=reserved,
            max_input_tokens=max_context_tokens - reserved,
            warning_ratio=warning_ratio,
            critical_ratio=critical_ratio,
            compact_ratio=compact_ratio,
        )
    except ValidationError as exc:
        raise ContextConfigError(
            "context_budget_invalid",
            "context budget config is invalid",
        ) from exc


def compute_context_usage(
    *,
    messages: list[Any],
    budget: ContextBudget,
    schema_size_chars: int | None = None,
    provider: str = "",
) -> ContextUsageReport:
    """Compute a sanitized pre-call context usage report."""
    token_count = count_messages_tokens(messages or [])
    used_tokens = token_count.value + budget.reserved_output_tokens
    used_ratio = used_tokens / budget.max_context_tokens
    available_tokens = max(budget.max_context_tokens - used_tokens, 0)
    breakdown = {
        "input_estimated_tokens": token_count.value,
        "reserved_output_tokens": budget.reserved_output_tokens,
    }
    if schema_size_chars is not None:
        breakdown["schema_size_chars"] = _non_negative_int(
            schema_size_chars,
            "schema_size_chars",
        )
    try:
        return ContextUsageReport(
            node_name=budget.node_name,
            llm_node=budget.llm_node,
            provider=str(provider or ""),
            model=budget.model,
            input_estimated_tokens=token_count.value,
            reserved_output_tokens=budget.reserved_output_tokens,
            used_tokens=used_tokens,
            max_context_tokens=budget.max_context_tokens,
            available_tokens=available_tokens,
            used_ratio=round(used_ratio, 6),
            warning_level=_warning_level(used_ratio, budget=budget),
            estimated=token_count.estimated,
            tokenizer_mode=token_count.method,
            message_count=len(messages or []),
            schema_size_chars=schema_size_chars,
            breakdown=breakdown,
        )
    except ValidationError as exc:
        raise ContextUsageError(
            "context_usage_report_invalid",
            "context usage report validation failed",
        ) from exc


def build_context_usage_payload(
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    messages: list[Any],
    reserved_output_tokens: int | None = None,
    schema_size_chars: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Build a context_usage payload, error payload, or disabled no-op."""
    config = get_context_engineering_config()
    if config.get("enabled") is False:
        return "", None
    strict = bool(config.get("strict"))
    try:
        budget = build_context_budget(
            node_name=node_name,
            llm_node=llm_node,
            model=model,
            reserved_output_tokens=reserved_output_tokens,
        )
        report = compute_context_usage(
            messages=messages,
            budget=budget,
            schema_size_chars=schema_size_chars,
            provider=provider,
        )
    except (ContextConfigError, ContextUsageError) as exc:
        if strict:
            raise
        return "context_usage_error", _error_payload(
            node_name=node_name,
            llm_node=llm_node,
            provider=provider,
            model=model,
            reason=exc.reason,
            warning=exc.warning,
        )
    return "context_usage", report.model_dump()


def _warning_level(
    used_ratio: float, *, budget: ContextBudget
) -> Literal["ok", "warning", "critical", "overflow"]:
    if used_ratio > 1:
        return "overflow"
    if used_ratio >= budget.critical_ratio:
        return "critical"
    if used_ratio >= budget.warning_ratio:
        return "warning"
    return "ok"


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContextConfigError(
            f"{key}_invalid", f"{key} must be a non-negative integer"
        )
    if value < 0:
        raise ContextConfigError(f"{key}_invalid", f"{key} must be >= 0")
    return value


def _error_payload(
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    reason: str,
    warning: str,
) -> dict[str, Any]:
    return {
        "node_name": node_name,
        "llm_node": llm_node,
        "provider": provider,
        "model": model,
        "reason": reason,
        "warning": warning,
    }
