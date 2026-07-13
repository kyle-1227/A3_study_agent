"""Strict full-compaction contracts and pure boundary construction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Final, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.config import get_setting
from src.context_engineering.budget import get_model_context_limit
from src.context_engineering.input_accounting import (
    build_llm_input_accounting,
    message_content,
    message_role,
)
from src.context_engineering.policies import get_thresholds
from src.context_engineering.schema import ContextConfigError

FULL_COMPACTION_SCHEMA_VERSION: Final[Literal[1]] = 1
CONVERSATION_SUMMARY_SCHEMA_VERSION: Final[Literal[2]] = 2


class FullCompactionConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    enabled: bool
    retain_recent_rounds: int = Field(ge=1, le=20)
    summary_llm_node: str = Field(min_length=1, max_length=120)
    output_mode: Literal["deepseek_tool_call_strict"]
    max_summary_input_chars: int = Field(ge=1000, le=256_000)


class ProviderBoundUsageV1(BaseModel):
    """Latest actual, trigger-eligible provider dispatch measurement."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = FULL_COMPACTION_SCHEMA_VERSION
    dispatch_id: str = Field(min_length=1, max_length=240)
    call_id: str = Field(min_length=1, max_length=240)
    request_id: str = Field(min_length=1, max_length=160)
    thread_id: str = Field(min_length=1, max_length=160)
    attempt: int = Field(ge=1)
    provider: str = Field(min_length=1, max_length=120)
    model: str = Field(min_length=1, max_length=160)
    input_tokens: int = Field(ge=0)
    tokenizer_mode: str = Field(min_length=1, max_length=120)
    estimated: bool
    trigger_eligible: Literal[True]
    dispatched_at: datetime

    @model_validator(mode="after")
    def _validate_time(self) -> "ProviderBoundUsageV1":
        if self.dispatched_at.tzinfo is None:
            raise ValueError("dispatched_at must include a timezone")
        return self


class FullCompactionDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = FULL_COMPACTION_SCHEMA_VERSION
    eligible: bool
    reason: Literal[
        "disabled",
        "no_actual_provider_dispatch",
        "below_threshold",
        "threshold_reached",
    ]
    dispatch_id: str = Field(default="", max_length=240)
    model: str = Field(default="", max_length=160)
    observed_input_tokens: int = Field(ge=0)
    context_window_limit_tokens: int = Field(ge=0)
    observed_ratio: float = Field(ge=0)
    compact_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_decision(self) -> "FullCompactionDecisionV1":
        if self.eligible != (self.reason == "threshold_reached"):
            raise ValueError("eligible must match threshold_reached")
        if self.reason == "no_actual_provider_dispatch" and self.dispatch_id:
            raise ValueError("missing dispatch decision cannot include dispatch_id")
        return self


class CompactMessageIdentityV1(BaseModel):
    """Content-free identity for one transcript message replaced by a summary."""

    model_config = ConfigDict(extra="forbid")

    original_index: int = Field(ge=0)
    role: str = Field(min_length=1, max_length=40)
    content_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    occurrence: int = Field(ge=1)
    estimated_tokens: int = Field(ge=0)
    tool_call_ids: list[str]
    tool_result_id: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def _validate_tool_ids(self) -> "CompactMessageIdentityV1":
        if len(self.tool_call_ids) != len(set(self.tool_call_ids)):
            raise ValueError("tool_call_ids must be unique")
        if any(not value.strip() for value in self.tool_call_ids):
            raise ValueError("tool_call_ids cannot contain blanks")
        return self


class CompactBoundaryV1(BaseModel):
    """Atomic boundary between full transcript and summarized model history."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = FULL_COMPACTION_SCHEMA_VERSION
    boundary_id: str = Field(pattern=r"^compact-boundary:v1:sha256:[0-9a-f]{64}$")
    thread_id: str = Field(min_length=1, max_length=160)
    request_id: str = Field(min_length=1, max_length=160)
    trigger_dispatch_id: str = Field(min_length=1, max_length=240)
    created_at: datetime
    source_message_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    compacted_messages: list[CompactMessageIdentityV1] = Field(min_length=1)
    retained_message_count: int = Field(ge=0)
    retained_recent_rounds: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _validate_boundary(self) -> "CompactBoundaryV1":
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        indices = [item.original_index for item in self.compacted_messages]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("compacted message indices must be sorted and unique")
        if any(
            item.role.lower() in {"system", "developer"}
            for item in self.compacted_messages
        ):
            raise ValueError("system and developer messages cannot be compacted")
        call_ids = {
            call_id
            for item in self.compacted_messages
            for call_id in item.tool_call_ids
        }
        result_ids = {
            item.tool_result_id
            for item in self.compacted_messages
            if item.tool_result_id
        }
        if call_ids != result_ids:
            raise ValueError("compaction boundary must preserve tool call/result pairs")
        return self


class ConversationSummaryV2(BaseModel):
    """Validated semantic memory replacing messages before a compact boundary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    boundary_id: str = Field(pattern=r"^compact-boundary:v1:sha256:[0-9a-f]{64}$")
    summary: str = Field(min_length=1, max_length=12_000)
    learning_goals: list[str] = Field(max_length=40)
    preferences: list[str] = Field(max_length=40)
    facts: list[str] = Field(max_length=80)
    decisions: list[str] = Field(max_length=60)
    unfinished_tasks: list[str] = Field(max_length=60)
    evidence_ids: list[str] = Field(max_length=80)
    artifact_ids: list[str] = Field(max_length=80)

    @model_validator(mode="after")
    def _validate_summary(self) -> "ConversationSummaryV2":
        if self.summary != self.summary.strip():
            raise ValueError("summary must be trimmed")
        for field_name in (
            "learning_goals",
            "preferences",
            "facts",
            "decisions",
            "unfinished_tasks",
            "evidence_ids",
            "artifact_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
            if any(not value or value != value.strip() for value in values):
                raise ValueError(f"{field_name} entries must be non-empty and trimmed")
            if any(len(value) > 1000 for value in values):
                raise ValueError(f"{field_name} entry is too long")
        return self


class CompactionResultV1(BaseModel):
    """Safe measurement and recovery descriptor for a committed compaction."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = FULL_COMPACTION_SCHEMA_VERSION
    status: Literal["compacted"]
    boundary_id: str = Field(pattern=r"^compact-boundary:v1:sha256:[0-9a-f]{64}$")
    trigger_dispatch_id: str = Field(min_length=1, max_length=240)
    compacted_at: datetime
    trigger_input_tokens: int = Field(ge=0)
    context_window_limit_tokens: int = Field(gt=0)
    trigger_ratio: float = Field(ge=0)
    compact_ratio: float = Field(ge=0, le=1)
    model_view_before_tokens: int = Field(ge=0)
    model_view_after_tokens: int = Field(ge=0)
    compacted_message_count: int = Field(ge=1)
    retained_message_count: int = Field(ge=0)
    summary_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ledger_before_tokens: int = Field(ge=0)
    ledger_after_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_result(self) -> "CompactionResultV1":
        if self.compacted_at.tzinfo is None:
            raise ValueError("compacted_at must include a timezone")
        if self.model_view_after_tokens >= self.model_view_before_tokens:
            raise ValueError("full compaction must reduce model-view tokens")
        if self.ledger_after_tokens > self.ledger_before_tokens:
            raise ValueError("compaction cannot increase retained ledger tokens")
        return self


def get_full_compaction_config() -> FullCompactionConfigV1:
    raw = get_setting("context_engineering.full_compaction")
    if not isinstance(raw, Mapping):
        raise ContextConfigError(
            "full_compaction_config_missing",
            "context_engineering.full_compaction is required",
        )
    try:
        config = FullCompactionConfigV1.model_validate(dict(raw))
    except ValidationError as exc:
        raise ContextConfigError(
            "full_compaction_config_invalid",
            "context_engineering.full_compaction is invalid",
        ) from exc
    model_view_rounds = get_setting(
        "context_engineering.model_view.retain_recent_rounds"
    )
    if isinstance(model_view_rounds, bool) or not isinstance(model_view_rounds, int):
        raise ContextConfigError(
            "model_view_retention_invalid",
            "context_engineering.model_view.retain_recent_rounds is required",
        )
    if config.retain_recent_rounds != model_view_rounds:
        raise ContextConfigError(
            "compaction_retention_mismatch",
            "full_compaction and model_view retention rounds must match",
        )
    return config


def provider_bound_usage_from_trace(event: Mapping[str, Any]) -> ProviderBoundUsageV1:
    if event.get("stage") != "provider_dispatch.started":
        raise ValueError("provider dispatch trace stage is invalid")
    return ProviderBoundUsageV1.model_validate(
        {
            "dispatch_id": event.get("dispatch_id"),
            "call_id": event.get("call_id"),
            "request_id": event.get("request_id"),
            "thread_id": event.get("thread_id") or event.get("session_id"),
            "attempt": event.get("attempt"),
            "provider": event.get("provider"),
            "model": event.get("model"),
            "input_tokens": event.get("input_tokens"),
            "tokenizer_mode": event.get("tokenizer_mode"),
            "estimated": event.get("estimated"),
            "trigger_eligible": event.get("trigger_eligible"),
            "dispatched_at": event.get("dispatched_at"),
        }
    )


def evaluate_full_compaction(
    last_dispatch: Mapping[str, Any] | None,
    *,
    config: FullCompactionConfigV1 | None = None,
) -> FullCompactionDecisionV1:
    resolved = config or get_full_compaction_config()
    _warning_ratio, _critical_ratio, compact_ratio = get_thresholds()
    if not resolved.enabled:
        return FullCompactionDecisionV1(
            eligible=False,
            reason="disabled",
            observed_input_tokens=0,
            context_window_limit_tokens=0,
            observed_ratio=0,
            compact_ratio=compact_ratio,
        )
    if not isinstance(last_dispatch, Mapping) or not last_dispatch:
        return FullCompactionDecisionV1(
            eligible=False,
            reason="no_actual_provider_dispatch",
            observed_input_tokens=0,
            context_window_limit_tokens=0,
            observed_ratio=0,
            compact_ratio=compact_ratio,
        )
    usage = ProviderBoundUsageV1.model_validate(last_dispatch)
    limit = get_model_context_limit(usage.model)
    ratio = usage.input_tokens / limit
    eligible = ratio >= compact_ratio
    return FullCompactionDecisionV1(
        eligible=eligible,
        reason="threshold_reached" if eligible else "below_threshold",
        dispatch_id=usage.dispatch_id,
        model=usage.model,
        observed_input_tokens=usage.input_tokens,
        context_window_limit_tokens=limit,
        observed_ratio=round(ratio, 8),
        compact_ratio=compact_ratio,
    )


def build_compact_boundary(
    messages: list[Any],
    *,
    thread_id: str,
    request_id: str,
    trigger_dispatch_id: str,
    retain_recent_rounds: int,
    created_at: datetime | None = None,
) -> CompactBoundaryV1 | None:
    """Build a boundary that never splits system or tool-call/result semantics."""

    user_indices = [
        index
        for index, message in enumerate(messages or [])
        if message_role(message).lower() in {"human", "user"}
    ]
    if len(user_indices) <= retain_recent_rounds:
        return None
    retained_start = user_indices[-retain_recent_rounds]
    accounting = build_llm_input_accounting(messages or [])
    occurrences: dict[tuple[str, str], int] = {}
    identities: list[CompactMessageIdentityV1] = []
    for index, message in enumerate(messages or []):
        role = message_role(message).lower()
        content_fingerprint = _content_fingerprint(message_content(message))
        key = (role, content_fingerprint)
        occurrences[key] = occurrences.get(key, 0) + 1
        if index >= retained_start or role in {"system", "developer"}:
            continue
        identities.append(
            CompactMessageIdentityV1(
                original_index=index,
                role=role,
                content_fingerprint=content_fingerprint,
                occurrence=occurrences[key],
                estimated_tokens=accounting.messages[index].estimated_tokens,
                tool_call_ids=list(_tool_call_ids(message)),
                tool_result_id=_tool_result_id(message),
            )
        )
    if not identities:
        return None
    compacted_indices = {identity.original_index for identity in identities}
    _validate_tool_pair_boundary(messages, compacted_indices)
    created = created_at or datetime.now(timezone.utc)
    identity = {
        "thread_id": thread_id,
        "request_id": request_id,
        "trigger_dispatch_id": trigger_dispatch_id,
        "source_message_fingerprint": accounting.message_fingerprint,
        "compacted_messages": [item.model_dump(mode="json") for item in identities],
        "retained_recent_rounds": retain_recent_rounds,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CompactBoundaryV1(
        boundary_id=f"compact-boundary:v1:sha256:{digest}",
        thread_id=thread_id,
        request_id=request_id,
        trigger_dispatch_id=trigger_dispatch_id,
        created_at=created,
        source_message_fingerprint=accounting.message_fingerprint,
        compacted_messages=identities,
        retained_message_count=len(messages or []) - len(identities),
        retained_recent_rounds=retain_recent_rounds,
    )


def validate_conversation_summary(
    summary: ConversationSummaryV2,
    *,
    boundary: CompactBoundaryV1,
    required_evidence_ids: set[str],
    required_artifact_ids: set[str],
) -> None:
    if summary.boundary_id != boundary.boundary_id:
        raise ValueError("conversation summary boundary_id mismatch")
    evidence_ids = set(summary.evidence_ids)
    artifact_ids = set(summary.artifact_ids)
    if evidence_ids != required_evidence_ids:
        raise ValueError("conversation summary evidence_ids mismatch")
    if artifact_ids != required_artifact_ids:
        raise ValueError("conversation summary artifact_ids mismatch")


def collect_summary_reference_ids(
    state: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    evidence_ids: set[str] = set()
    artifact_ids: set[str] = set()
    previous_summary = state.get("conversation_summary_v2")
    if isinstance(previous_summary, Mapping) and previous_summary:
        validated_previous = ConversationSummaryV2.model_validate(previous_summary)
        evidence_ids.update(validated_previous.evidence_ids)
        artifact_ids.update(validated_previous.artifact_ids)
    _collect_ids(
        state.get("evidence_summary_memory"),
        evidence_ids,
        keys=("evidence_id", "memory_id", "source_id", "doc_id"),
    )
    workspace = state.get("task_workspace")
    if isinstance(workspace, Mapping):
        _collect_ids(
            workspace.get("evidence_summaries"),
            evidence_ids,
            keys=("evidence_id", "memory_id", "source_id", "doc_id", "id"),
        )
        _collect_ids(
            workspace.get("artifacts"),
            artifact_ids,
            keys=("artifact_id", "resource_id", "id"),
        )
    _collect_ids(
        state.get("last_generated_artifacts"),
        artifact_ids,
        keys=("artifact_id", "resource_id", "id"),
    )
    resource_payload = state.get("last_resource_final_payload")
    if isinstance(resource_payload, Mapping):
        value = str(resource_payload.get("resource_id") or "").strip()
        if value:
            artifact_ids.add(value)
    return evidence_ids, artifact_ids


def summary_fingerprint(summary: ConversationSummaryV2) -> str:
    encoded = json.dumps(
        summary.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _collect_ids(
    value: Any,
    target: set[str],
    *,
    keys: tuple[str, ...],
) -> None:
    items = value if isinstance(value, list) else []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key in keys:
            text = str(item.get(key) or "").strip()
            if text:
                target.add(text[:240])
                break


def _validate_tool_pair_boundary(
    messages: list[Any], compacted_indices: set[int]
) -> None:
    call_indices: dict[str, int] = {}
    result_indices: dict[str, int] = {}
    for index, message in enumerate(messages or []):
        for call_id in _tool_call_ids(message):
            call_indices[call_id] = index
        result_id = _tool_result_id(message)
        if result_id:
            result_indices[result_id] = index
    all_ids = set(call_indices) | set(result_indices)
    for call_id in all_ids:
        call_index = call_indices.get(call_id)
        result_index = result_indices.get(call_id)
        if call_index is None or result_index is None:
            raise ValueError("tool call/result pair is incomplete")
        if (call_index in compacted_indices) != (result_index in compacted_indices):
            raise ValueError("compaction boundary splits a tool call/result pair")


def _tool_result_id(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("tool_call_id") or "").strip()
    return str(getattr(message, "tool_call_id", "") or "").strip()


def _tool_call_ids(message: Any) -> tuple[str, ...]:
    raw_calls: Any = None
    if isinstance(message, Mapping):
        raw_calls = message.get("tool_calls")
        additional = message.get("additional_kwargs")
        if raw_calls is None and isinstance(additional, Mapping):
            raw_calls = additional.get("tool_calls")
    else:
        raw_calls = getattr(message, "tool_calls", None)
        if not raw_calls:
            additional = getattr(message, "additional_kwargs", {}) or {}
            if isinstance(additional, Mapping):
                raw_calls = additional.get("tool_calls")
    if not isinstance(raw_calls, list):
        return ()
    result: list[str] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        call_id = str(raw_call.get("id") or "").strip()
        if call_id and call_id not in result:
            result.append(call_id)
    return tuple(result)


def _content_fingerprint(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


__all__ = [
    "CONVERSATION_SUMMARY_SCHEMA_VERSION",
    "FULL_COMPACTION_SCHEMA_VERSION",
    "CompactBoundaryV1",
    "CompactMessageIdentityV1",
    "CompactionResultV1",
    "ConversationSummaryV2",
    "FullCompactionConfigV1",
    "FullCompactionDecisionV1",
    "ProviderBoundUsageV1",
    "build_compact_boundary",
    "collect_summary_reference_ids",
    "evaluate_full_compaction",
    "get_full_compaction_config",
    "provider_bound_usage_from_trace",
    "summary_fingerprint",
    "validate_conversation_summary",
]
