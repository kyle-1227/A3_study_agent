"""Strict configuration for server and browser performance observation."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import get_setting


class FrontendPerformanceIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    enabled: bool
    endpoint_path: str = Field(pattern=r"^/[a-z0-9/_-]{1,119}$", max_length=120)
    secret_env: str = Field(min_length=1, max_length=120)
    allowed_origins: list[str] = Field(min_length=1, max_length=16)
    token_ttl_seconds: int = Field(ge=30, le=3600)
    max_payload_bytes: int = Field(ge=1024, le=65_536)
    max_milestones_per_request: int = Field(ge=1, le=16)
    max_batches_per_request: int = Field(ge=1, le=1)

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            origin = value.strip().rstrip("/")
            if not origin.startswith(("http://", "https://")):
                raise ValueError("frontend performance origin must use http or https")
            if "?" in origin or "#" in origin or "@" in origin:
                raise ValueError(
                    "frontend performance origin contains forbidden components"
                )
            if origin not in normalized:
                normalized.append(origin)
        return normalized


class PerformanceObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    enabled: bool
    max_spans_per_request: int = Field(ge=32, le=4096)
    report_retention_count: int = Field(ge=16, le=4096)
    frontend_ingestion: FrontendPerformanceIngestionConfig

    @model_validator(mode="after")
    def validate_enabled_contract(self) -> "PerformanceObservabilityConfig":
        if self.frontend_ingestion.enabled and not self.enabled:
            raise ValueError(
                "frontend performance ingestion requires server observation"
            )
        return self


def load_performance_observability_config() -> PerformanceObservabilityConfig:
    raw = get_setting("observability.performance", None)
    if not isinstance(raw, dict):
        raise RuntimeError("observability.performance configuration is required")
    return PerformanceObservabilityConfig.model_validate(raw)


def resolve_frontend_performance_secret(
    config: PerformanceObservabilityConfig,
) -> bytes | None:
    frontend = config.frontend_ingestion
    if not frontend.enabled:
        return None
    value = os.getenv(frontend.secret_env, "")
    if not value:
        raise RuntimeError(
            f"{frontend.secret_env} is required when frontend performance ingestion is enabled"
        )
    encoded = value.encode("utf-8")
    if len(encoded) < 32:
        raise RuntimeError("frontend performance HMAC secret must be at least 32 bytes")
    return encoded


__all__ = [
    "FrontendPerformanceIngestionConfig",
    "PerformanceObservabilityConfig",
    "load_performance_observability_config",
    "resolve_frontend_performance_secret",
]
