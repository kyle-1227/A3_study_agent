"""Strict configuration for the agent streaming transport."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import get_setting
from src.streaming.contracts import StreamContractError


class StreamingRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retry_ms: int = Field(gt=0, le=60000)
    journal_max_events: int = Field(gt=3, le=100000)
    journal_max_bytes: int = Field(gt=8192, le=134217728)
    journal_ttl_seconds: int = Field(gt=0, le=86400)


def load_streaming_runtime_config() -> StreamingRuntimeConfig:
    raw = get_setting("streaming")
    if not isinstance(raw, dict):
        raise StreamContractError("streaming configuration is required")
    try:
        return StreamingRuntimeConfig.model_validate(raw)
    except ValidationError as exc:
        raise StreamContractError("streaming configuration is invalid") from exc


__all__ = ["StreamingRuntimeConfig", "load_streaming_runtime_config"]
