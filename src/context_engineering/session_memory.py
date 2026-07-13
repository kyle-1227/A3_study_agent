"""Persistent, content-free accounting for context actually dispatched to providers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.context_engineering.schema import ContextItem

SESSION_CONTEXT_MEMORY_LEDGER_SCHEMA_VERSION = 1

InjectionSource = Literal[
    "profile",
    "memory",
    "evidence",
    "artifact",
    "rules",
    "curriculum",
    "trajectory",
    "pipeline",
]
INJECTION_SOURCES: tuple[InjectionSource, ...] = (
    "profile",
    "memory",
    "evidence",
    "artifact",
    "rules",
    "curriculum",
    "trajectory",
    "pipeline",
)


class ContextInjectionItemDescriptorV1(BaseModel):
    """Safe identity and measurement for one provider-bound CE item."""

    model_config = ConfigDict(extra="forbid")

    logical_item_id: str = Field(min_length=1, max_length=240)
    source_type: InjectionSource
    content_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    token_count: int = Field(ge=0)
    tokenizer_mode: str = Field(min_length=1, max_length=120)
    estimated: bool


class ContextInjectionRecordV1(BaseModel):
    """One item included in one real provider dispatch attempt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SESSION_CONTEXT_MEMORY_LEDGER_SCHEMA_VERSION
    record_id: str = Field(min_length=1, max_length=240)
    dispatch_id: str = Field(min_length=1, max_length=240)
    request_id: str = Field(min_length=1, max_length=160)
    call_id: str = Field(min_length=1, max_length=200)
    attempt: int = Field(ge=1)
    manifest_id: str = Field(min_length=1, max_length=240)
    thread_id: str = Field(min_length=1, max_length=160)
    item: ContextInjectionItemDescriptorV1
    dispatched_at: datetime

    @model_validator(mode="after")
    def _validate_dispatch_identity(self) -> "ContextInjectionRecordV1":
        if self.dispatched_at.tzinfo is None:
            raise ValueError("dispatched_at must include a timezone")
        return self


class ContextMemoryCompactionMutationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["apply_compaction"]
    boundary_id: str = Field(min_length=1, max_length=240)
    retained_logical_item_ids: list[str]
    compacted_at: datetime
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_compacted_at(self) -> "ContextMemoryCompactionMutationV1":
        if self.compacted_at.tzinfo is None:
            raise ValueError("compacted_at must include a timezone")
        if len(self.retained_logical_item_ids) != len(
            set(self.retained_logical_item_ids)
        ):
            raise ValueError("retained_logical_item_ids must be unique")
        return self


class SourceInjectionStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retained_tokens: int = Field(default=0, ge=0)
    lifetime_injected_tokens: int = Field(default=0, ge=0)
    lifetime_unique_tokens: int = Field(default=0, ge=0)
    injection_count: int = Field(default=0, ge=0)
    repeat_injection_count: int = Field(default=0, ge=0)
    active_item_count: int = Field(default=0, ge=0)


class LedgerMeasurementV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_tokenizer_mode: str = ""
    last_estimated: bool = True
    estimated_injection_count: int = Field(default=0, ge=0)


class LedgerMemorySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_item_count: int = Field(default=0, ge=0)
    active_unique_content_count: int = Field(default=0, ge=0)


class LedgerCompactionStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["never", "compacted"] = "never"
    boundary_id: str = ""
    compacted_at: datetime | None = None
    before_tokens: int = Field(default=0, ge=0)
    after_tokens: int = Field(default=0, ge=0)


class SessionContextMemoryLedgerV1(BaseModel):
    """Thread-scoped monotonic injection totals plus the currently retained set."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SESSION_CONTEXT_MEMORY_LEDGER_SCHEMA_VERSION
    thread_id: str = Field(min_length=1, max_length=160)
    updated_at: datetime
    retained_memory_tokens: int = Field(default=0, ge=0)
    lifetime_injected_tokens: int = Field(default=0, ge=0)
    lifetime_unique_tokens: int = Field(default=0, ge=0)
    request_count: int = Field(default=0, ge=0)
    injection_count: int = Field(default=0, ge=0)
    repeat_injection_count: int = Field(default=0, ge=0)
    active_items: dict[str, ContextInjectionItemDescriptorV1] = Field(
        default_factory=dict
    )
    seen_content: dict[str, ContextInjectionItemDescriptorV1] = Field(
        default_factory=dict
    )
    processed_record_ids: list[str] = Field(default_factory=list)
    request_ids: list[str] = Field(default_factory=list)
    source_stats: dict[InjectionSource, SourceInjectionStatsV1] = Field(
        default_factory=lambda: _empty_source_stats()
    )
    measurement: LedgerMeasurementV1 = Field(default_factory=LedgerMeasurementV1)
    memory_summary: LedgerMemorySummaryV1 = Field(default_factory=LedgerMemorySummaryV1)
    compaction: LedgerCompactionStatusV1 = Field(
        default_factory=LedgerCompactionStatusV1
    )

    @model_validator(mode="after")
    def _validate_invariants(self) -> "SessionContextMemoryLedgerV1":
        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must include a timezone")
        if set(self.source_stats) != set(INJECTION_SOURCES):
            raise ValueError("source_stats must contain every injection source")
        if self.request_count != len(self.request_ids):
            raise ValueError("request_count must equal unique request_ids")
        if self.injection_count != len(self.processed_record_ids):
            raise ValueError("injection_count must equal processed record count")
        if self.lifetime_unique_tokens > self.lifetime_injected_tokens:
            raise ValueError("lifetime_unique_tokens cannot exceed lifetime total")
        return self


def new_session_context_memory_ledger(thread_id: str) -> SessionContextMemoryLedgerV1:
    return SessionContextMemoryLedgerV1(
        thread_id=thread_id,
        updated_at=_utc_now(),
    )


def build_injection_descriptor(
    item: ContextItem,
) -> ContextInjectionItemDescriptorV1 | None:
    """Return a safe descriptor only for the eight session-memory CE sources."""

    if item.source_type not in INJECTION_SOURCES:
        return None
    return ContextInjectionItemDescriptorV1(
        logical_item_id=item.id,
        source_type=item.source_type,
        content_fingerprint=_content_fingerprint(item),
        token_count=item.token_estimate,
        tokenizer_mode=item.tokenizer_mode,
        estimated=item.estimated,
    )


def record_context_injection(
    ledger: SessionContextMemoryLedgerV1,
    record: ContextInjectionRecordV1,
) -> SessionContextMemoryLedgerV1:
    """Apply one real dispatch item idempotently without retaining its body."""

    if record.thread_id != ledger.thread_id:
        raise ValueError("context injection thread_id does not match ledger")
    if record.record_id in ledger.processed_record_ids:
        return ledger.model_copy(deep=True)

    payload = ledger.model_dump(mode="python")
    active_items = dict(ledger.active_items)
    seen_content = dict(ledger.seen_content)
    request_ids = list(ledger.request_ids)
    processed_record_ids = list(ledger.processed_record_ids)
    source_stats = {
        source: ledger.source_stats[source].model_copy(deep=True)
        for source in INJECTION_SOURCES
    }
    descriptor = record.item
    is_repeat = descriptor.content_fingerprint in seen_content

    processed_record_ids.append(record.record_id)
    if record.request_id not in request_ids:
        request_ids.append(record.request_id)
    if not is_repeat:
        seen_content[descriptor.content_fingerprint] = descriptor
    active_items[descriptor.logical_item_id] = descriptor

    source = source_stats[descriptor.source_type]
    source.lifetime_injected_tokens += descriptor.token_count
    source.injection_count += 1
    if is_repeat:
        source.repeat_injection_count += 1
    else:
        source.lifetime_unique_tokens += descriptor.token_count

    payload.update(
        {
            "updated_at": record.dispatched_at,
            "lifetime_injected_tokens": (
                ledger.lifetime_injected_tokens + descriptor.token_count
            ),
            "lifetime_unique_tokens": (
                ledger.lifetime_unique_tokens
                + (0 if is_repeat else descriptor.token_count)
            ),
            "request_count": len(request_ids),
            "injection_count": len(processed_record_ids),
            "repeat_injection_count": ledger.repeat_injection_count + int(is_repeat),
            "active_items": active_items,
            "seen_content": seen_content,
            "processed_record_ids": processed_record_ids,
            "request_ids": request_ids,
            "source_stats": source_stats,
            "measurement": LedgerMeasurementV1(
                last_tokenizer_mode=descriptor.tokenizer_mode,
                last_estimated=descriptor.estimated,
                estimated_injection_count=(
                    ledger.measurement.estimated_injection_count
                    + int(descriptor.estimated)
                ),
            ),
        }
    )
    _recompute_retained(payload)
    return SessionContextMemoryLedgerV1.model_validate(payload)


def apply_context_memory_compaction(
    ledger: SessionContextMemoryLedgerV1,
    *,
    boundary_id: str,
    retained_logical_item_ids: list[str],
    compacted_at: datetime,
    before_tokens: int,
    after_tokens: int,
) -> SessionContextMemoryLedgerV1:
    """Atomically replace the retained item set while preserving lifetime totals."""

    if compacted_at.tzinfo is None:
        raise ValueError("compacted_at must include a timezone")
    if not boundary_id.strip():
        raise ValueError("boundary_id is required")
    if before_tokens != ledger.retained_memory_tokens:
        raise ValueError("before_tokens does not match current retained total")
    unknown = set(retained_logical_item_ids) - set(ledger.active_items)
    if unknown:
        raise ValueError("compaction retained unknown logical items")
    if ledger.active_items and not retained_logical_item_ids:
        raise ValueError("compaction cannot clear all retained memory")

    payload = ledger.model_dump(mode="python")
    payload["active_items"] = {
        item_id: ledger.active_items[item_id] for item_id in retained_logical_item_ids
    }
    payload["updated_at"] = compacted_at
    _recompute_retained(payload)
    recomputed_after = int(payload["retained_memory_tokens"])
    if after_tokens != recomputed_after:
        raise ValueError("after_tokens does not match compacted retained total")
    if after_tokens > before_tokens:
        raise ValueError("compaction cannot increase retained tokens")
    payload["compaction"] = LedgerCompactionStatusV1(
        status="compacted",
        boundary_id=boundary_id,
        compacted_at=compacted_at,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
    )
    return SessionContextMemoryLedgerV1.model_validate(payload)


def clear_session_context_memory(
    ledger: SessionContextMemoryLedgerV1,
) -> SessionContextMemoryLedgerV1:
    return new_session_context_memory_ledger(ledger.thread_id)


def _recompute_retained(payload: dict[str, object]) -> None:
    active = {
        key: ContextInjectionItemDescriptorV1.model_validate(value)
        for key, value in dict(payload.get("active_items") or {}).items()
    }
    source_stats = {
        source: SourceInjectionStatsV1.model_validate(
            dict(payload.get("source_stats") or {}).get(source, {})
        )
        for source in INJECTION_SOURCES
    }
    for stats in source_stats.values():
        stats.retained_tokens = 0
        stats.active_item_count = 0
    for descriptor in active.values():
        source_stats[descriptor.source_type].active_item_count += 1

    grouped: dict[str, list[tuple[str, ContextInjectionItemDescriptorV1]]] = {}
    for logical_id, descriptor in active.items():
        grouped.setdefault(descriptor.content_fingerprint, []).append(
            (logical_id, descriptor)
        )
    retained_tokens = 0
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry[0])
        canonical = entries[0][1]
        token_count = max(entry[1].token_count for entry in entries)
        retained_tokens += token_count
        source_stats[canonical.source_type].retained_tokens += token_count

    payload["retained_memory_tokens"] = retained_tokens
    payload["source_stats"] = source_stats
    payload["memory_summary"] = LedgerMemorySummaryV1(
        active_item_count=len(active),
        active_unique_content_count=len(grouped),
    )


def _content_fingerprint(item: ContextItem) -> str:
    raw = item.content
    if not raw.strip():
        raw = json.dumps(
            {"title": item.title, "metadata": item.metadata},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    normalized = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _empty_source_stats() -> dict[InjectionSource, SourceInjectionStatsV1]:
    return {source: SourceInjectionStatsV1() for source in INJECTION_SOURCES}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "INJECTION_SOURCES",
    "ContextInjectionItemDescriptorV1",
    "ContextInjectionRecordV1",
    "ContextMemoryCompactionMutationV1",
    "LedgerCompactionStatusV1",
    "LedgerMeasurementV1",
    "LedgerMemorySummaryV1",
    "SessionContextMemoryLedgerV1",
    "SourceInjectionStatsV1",
    "apply_context_memory_compaction",
    "build_injection_descriptor",
    "clear_session_context_memory",
    "new_session_context_memory_ledger",
    "record_context_injection",
]
