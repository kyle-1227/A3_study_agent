"""Public Context Window V3 projection from the durable injection ledger."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.context_engineering.budget import (
    get_context_engineering_config,
    get_model_context_limit,
)
from src.context_engineering.schema import ContextConfigError
from src.context_engineering.session_memory import (
    INJECTION_SOURCES,
    InjectionSource,
    SessionContextMemoryLedgerV1,
)


class ThreadContextInjectionTypeStatsV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retained_tokens: int = Field(ge=0)
    lifetime_injected_tokens: int = Field(ge=0)
    lifetime_unique_tokens: int = Field(ge=0)
    injection_count: int = Field(ge=0)
    repeat_injection_count: int = Field(ge=0)
    active_item_count: int = Field(ge=0)


class ThreadContextMeasurementV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_tokenizer_mode: str
    last_estimated: bool
    estimated_injection_count: int = Field(ge=0)


class ThreadContextMemorySummaryV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_item_count: int = Field(ge=0)
    active_unique_content_count: int = Field(ge=0)


class ThreadContextCompactionV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["never", "compacted"]
    boundary_id: str
    compacted_at: datetime | None
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)


class ThreadContextWindowV3(BaseModel):
    """Current retained memory plus monotonic session injection statistics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    thread_id: str = Field(min_length=1, max_length=160)
    updated_at: datetime
    updating: bool
    window_model: str = Field(min_length=1, max_length=160)
    context_window_limit_tokens: int = Field(gt=0)
    retained_memory_tokens: int = Field(ge=0)
    retained_ratio: float = Field(ge=0)
    lifetime_injected_tokens: int = Field(ge=0)
    lifetime_unique_tokens: int = Field(ge=0)
    request_count: int = Field(ge=0)
    injection_count: int = Field(ge=0)
    repeat_injection_count: int = Field(ge=0)
    injection_types: dict[InjectionSource, ThreadContextInjectionTypeStatsV3]
    measurement: ThreadContextMeasurementV3
    memory_summary: ThreadContextMemorySummaryV3
    compaction: ThreadContextCompactionV3

    @model_validator(mode="after")
    def _validate_totals(self) -> "ThreadContextWindowV3":
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must include a timezone")
        if set(self.injection_types) != set(INJECTION_SOURCES):
            raise ValueError("injection_types must contain every injection source")
        if sum(item.retained_tokens for item in self.injection_types.values()) != (
            self.retained_memory_tokens
        ):
            raise ValueError("retained type totals must equal retained_memory_tokens")
        if (
            sum(item.lifetime_injected_tokens for item in self.injection_types.values())
            != self.lifetime_injected_tokens
        ):
            raise ValueError("lifetime type totals must equal lifetime_injected_tokens")
        if (
            sum(item.lifetime_unique_tokens for item in self.injection_types.values())
            != self.lifetime_unique_tokens
        ):
            raise ValueError("unique type totals must equal lifetime_unique_tokens")
        if sum(item.injection_count for item in self.injection_types.values()) != (
            self.injection_count
        ):
            raise ValueError("type injection counts must equal injection_count")
        return self


def build_thread_context_window_v3(
    ledger: SessionContextMemoryLedgerV1,
    *,
    updating: bool = False,
) -> ThreadContextWindowV3:
    window_model = _session_window_model()
    context_window_limit_tokens = get_model_context_limit(window_model)
    injection_types = {
        source: ThreadContextInjectionTypeStatsV3.model_validate(
            ledger.source_stats[source].model_dump(mode="python")
        )
        for source in INJECTION_SOURCES
    }
    return ThreadContextWindowV3(
        thread_id=ledger.thread_id,
        updated_at=ledger.updated_at,
        updating=updating,
        window_model=window_model,
        context_window_limit_tokens=context_window_limit_tokens,
        retained_memory_tokens=ledger.retained_memory_tokens,
        retained_ratio=round(
            ledger.retained_memory_tokens / context_window_limit_tokens,
            8,
        ),
        lifetime_injected_tokens=ledger.lifetime_injected_tokens,
        lifetime_unique_tokens=ledger.lifetime_unique_tokens,
        request_count=ledger.request_count,
        injection_count=ledger.injection_count,
        repeat_injection_count=ledger.repeat_injection_count,
        injection_types=injection_types,
        measurement=ThreadContextMeasurementV3.model_validate(
            ledger.measurement.model_dump(mode="python")
        ),
        memory_summary=ThreadContextMemorySummaryV3.model_validate(
            ledger.memory_summary.model_dump(mode="python")
        ),
        compaction=ThreadContextCompactionV3.model_validate(
            ledger.compaction.model_dump(mode="python")
        ),
    )


def _session_window_model() -> str:
    config = get_context_engineering_config()
    session_memory = config.get("session_memory")
    if not isinstance(session_memory, dict):
        raise ContextConfigError(
            "session_memory_config_missing",
            "context_engineering.session_memory is required",
        )
    window_model = session_memory.get("window_model")
    if not isinstance(window_model, str) or not window_model.strip():
        raise ContextConfigError(
            "session_memory_window_model_invalid",
            "context_engineering.session_memory.window_model is required",
        )
    return window_model.strip()


__all__ = [
    "ThreadContextCompactionV3",
    "ThreadContextInjectionTypeStatsV3",
    "ThreadContextMeasurementV3",
    "ThreadContextMemorySummaryV3",
    "ThreadContextWindowV3",
    "build_thread_context_window_v3",
]
