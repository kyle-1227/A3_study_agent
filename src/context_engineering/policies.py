"""Phase 1 Context Engineering policy readers."""

from __future__ import annotations

from src.config import get_setting
from src.context_engineering.schema import ContextConfigError


def get_thresholds() -> tuple[float, float, float]:
    """Return warning, critical, and compact ratios from explicit config."""
    config = _require_enabled_config()
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ContextConfigError(
            "context_thresholds_missing",
            "context_engineering.thresholds config is required",
        )
    warning_ratio = _required_ratio(thresholds, "warning_ratio")
    critical_ratio = _required_ratio(thresholds, "critical_ratio")
    compact_ratio = _required_ratio(thresholds, "compact_ratio")
    if warning_ratio >= critical_ratio:
        raise ContextConfigError(
            "context_thresholds_invalid",
            "warning_ratio must be less than critical_ratio",
        )
    if critical_ratio > compact_ratio:
        raise ContextConfigError(
            "context_thresholds_invalid",
            "critical_ratio must not exceed compact_ratio",
        )
    return warning_ratio, critical_ratio, compact_ratio


def get_default_reserved_output_tokens() -> int:
    """Return the configured default output reservation."""
    config = _require_enabled_config()
    value = config.get("default_reserved_output_tokens")
    return _positive_int(value, "context_engineering.default_reserved_output_tokens")


def resolve_reserved_output_tokens(llm_node: str) -> int:
    """Resolve output reservation from node max_tokens or explicit default."""
    nested_value = get_setting(f"llm.{llm_node}.max_tokens")
    if nested_value is None:
        legacy_value = get_setting(f"{llm_node}.max_tokens")
    else:
        legacy_value = None
    value = nested_value if nested_value is not None else legacy_value
    if value is not None:
        return _positive_int(value, f"llm.{llm_node}.max_tokens")
    return get_default_reserved_output_tokens()


def _require_enabled_config() -> dict:
    config = get_setting("context_engineering")
    if not isinstance(config, dict):
        raise ContextConfigError(
            "context_engineering_missing",
            "context_engineering config is required",
        )
    if config.get("enabled") is not True:
        raise ContextConfigError(
            "context_engineering_disabled",
            "context engineering telemetry is disabled",
        )
    return config


def _required_ratio(values: dict, key: str) -> float:
    raw = values.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ContextConfigError(
            "context_thresholds_invalid",
            f"context_engineering.thresholds.{key} must be a number",
        )
    value = float(raw)
    if value < 0 or value > 1:
        raise ContextConfigError(
            "context_thresholds_invalid",
            f"context_engineering.thresholds.{key} must be between 0 and 1",
        )
    return value


def _positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContextConfigError(f"{key}_invalid", f"{key} must be a positive integer")
    if value <= 0:
        raise ContextConfigError(f"{key}_invalid", f"{key} must be > 0")
    return value
