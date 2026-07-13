"""Pure provider-bound model-view projection and micro compaction.

The projection never mutates checkpoint or transcript messages.  Persistable
descriptors contain only measurements, fingerprints, and trusted reference
identifiers; message content remains confined to the in-memory build result.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Literal, Mapping

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.config import get_setting
from src.context_engineering.compaction import (
    CompactBoundaryV1,
    ConversationSummaryV2,
)
from src.context_engineering.input_accounting import (
    LLMInputAccounting,
    build_llm_input_accounting,
    message_content,
    message_role,
)
from src.context_engineering.schema import ContextConfigError

MODEL_VIEW_PROJECTION_SCHEMA_VERSION: Final[Literal[1]] = 1
_SAFE_COMPACTION_METADATA_KEY = "model_view_compaction"


class ModelViewProjectionError(RuntimeError):
    """Raised when a persisted compact boundary cannot be applied exactly."""


class ModelViewConfigV1(BaseModel):
    """Strict runtime configuration for provider-bound projections."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    micro_compaction_enabled: bool
    retain_recent_rounds: int = Field(ge=1, le=20)


class TrustedModelViewReplacementV1(BaseModel):
    """Explicitly trusted replacement attached by an internal producer."""

    model_config = ConfigDict(extra="forbid")

    trusted: Literal[True]
    reference_id: str = Field(default="", max_length=240)
    safe_summary: str = Field(default="", max_length=8000)

    @model_validator(mode="after")
    def _require_reference_or_summary(self) -> "TrustedModelViewReplacementV1":
        self.reference_id = self.reference_id.strip()
        self.safe_summary = self.safe_summary.strip()
        if not self.reference_id and not self.safe_summary:
            raise ValueError("trusted replacement requires a reference or summary")
        return self


class ModelViewOperationV1(BaseModel):
    """Content-free record of one micro-compaction operation."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["deduplicated_context", "replaced_with_trusted_reference"]
    original_index: int = Field(ge=0)
    role: str = Field(min_length=1, max_length=40)
    original_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    projected_fingerprint: str = Field(
        default="",
        pattern=r"^(|sha256:[0-9a-f]{64})$",
    )
    reference_id: str = Field(default="", max_length=240)
    original_estimated_tokens: int = Field(ge=0)
    projected_estimated_tokens: int = Field(ge=0)


class ModelViewProjectionV1(BaseModel):
    """Safe descriptor for the exact in-memory provider message projection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = MODEL_VIEW_PROJECTION_SCHEMA_VERSION
    projection_id: str = Field(pattern=r"^model-view:v1:sha256:[0-9a-f]{64}$")
    created_at: datetime
    source_message_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    projected_message_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_message_count: int = Field(ge=0)
    output_message_count: int = Field(ge=0)
    input_estimated_tokens: int = Field(ge=0)
    output_estimated_tokens: int = Field(ge=0)
    retained_recent_rounds: int = Field(ge=1, le=20)
    micro_compaction_enabled: bool
    full_compaction_boundary_id: str = Field(default="", max_length=240)
    compacted_history_messages_removed: int = Field(ge=0)
    conversation_summary_injected: bool
    duplicate_context_messages_removed: int = Field(ge=0)
    tool_results_compacted: int = Field(ge=0)
    operations: list[ModelViewOperationV1]

    @model_validator(mode="after")
    def _validate_projection(self) -> "ModelViewProjectionV1":
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        expected_output_count = (
            self.input_message_count
            - self.compacted_history_messages_removed
            - self.duplicate_context_messages_removed
            + int(self.conversation_summary_injected)
        )
        if self.output_message_count != expected_output_count:
            raise ValueError("model view output message count mismatch")
        if bool(self.full_compaction_boundary_id) != bool(
            self.compacted_history_messages_removed
        ):
            raise ValueError("full compaction boundary/removal mismatch")
        if self.conversation_summary_injected != bool(self.full_compaction_boundary_id):
            raise ValueError("conversation summary/boundary mismatch")
        if self.duplicate_context_messages_removed != sum(
            operation.action == "deduplicated_context" for operation in self.operations
        ):
            raise ValueError("duplicate context operation count mismatch")
        if self.tool_results_compacted != sum(
            operation.action == "replaced_with_trusted_reference"
            for operation in self.operations
        ):
            raise ValueError("tool result operation count mismatch")
        return self


@dataclass(frozen=True)
class ModelViewBuildResult:
    """Runtime-only projected messages plus their safe descriptor."""

    messages: list[Any]
    projection: ModelViewProjectionV1


def get_model_view_config() -> ModelViewConfigV1:
    """Load explicit model-view configuration without silent defaults."""

    raw = get_setting("context_engineering.model_view")
    if not isinstance(raw, Mapping):
        raise ContextConfigError(
            "model_view_config_missing",
            "context_engineering.model_view is required",
        )
    try:
        return ModelViewConfigV1.model_validate(dict(raw))
    except ValidationError as exc:
        raise ContextConfigError(
            "model_view_config_invalid",
            "context_engineering.model_view is invalid",
        ) from exc


def build_model_view_projection(
    messages: list[Any],
    *,
    config: ModelViewConfigV1 | None = None,
    state: Mapping[str, Any] | None = None,
) -> ModelViewBuildResult:
    """Build a pure provider view while retaining authoritative history.

    Old tool results are replaceable only when an internal producer attached a
    validated, explicitly trusted summary or reference.  Exact duplicate
    ``<INJECTED_CONTEXT>`` messages are removed deterministically.  System
    messages, the current query, recent rounds, and tool-call/result pairing
    are otherwise retained.
    """

    resolved = config or get_model_view_config()
    source_messages = list(messages or [])
    source_accounting = build_llm_input_accounting(source_messages)
    (
        full_view_messages,
        full_compaction_boundary_id,
        compacted_history_messages_removed,
        conversation_summary_injected,
    ) = _apply_full_compaction(source_messages, state=state or {})
    protected_indices = _protected_message_indices(
        full_view_messages,
        retain_recent_rounds=resolved.retain_recent_rounds,
    )
    projected: list[Any] = []
    operations: list[ModelViewOperationV1] = []
    seen_context_fingerprints: set[str] = set()

    full_view_accounting = build_llm_input_accounting(full_view_messages)
    for index, message in enumerate(full_view_messages):
        content = message_content(message)
        original_fingerprint = _content_fingerprint(content)
        original_tokens = full_view_accounting.messages[index].estimated_tokens

        if (
            resolved.micro_compaction_enabled
            and _is_injected_context_message(content)
            and original_fingerprint in seen_context_fingerprints
        ):
            operations.append(
                ModelViewOperationV1(
                    action="deduplicated_context",
                    original_index=index,
                    role=message_role(message),
                    original_fingerprint=original_fingerprint,
                    original_estimated_tokens=original_tokens,
                    projected_estimated_tokens=0,
                )
            )
            continue
        if _is_injected_context_message(content):
            seen_context_fingerprints.add(original_fingerprint)

        replacement = (
            _trusted_replacement(message)
            if resolved.micro_compaction_enabled
            and index not in protected_indices
            and _is_tool_result(message)
            else None
        )
        if replacement is not None:
            replacement_content = _render_trusted_replacement(replacement)
            replacement_message = _copy_with_content(message, replacement_content)
            replacement_accounting = build_llm_input_accounting([replacement_message])
            projected.append(replacement_message)
            operations.append(
                ModelViewOperationV1(
                    action="replaced_with_trusted_reference",
                    original_index=index,
                    role=message_role(message),
                    original_fingerprint=original_fingerprint,
                    projected_fingerprint=_content_fingerprint(replacement_content),
                    reference_id=replacement.reference_id,
                    original_estimated_tokens=original_tokens,
                    projected_estimated_tokens=(
                        replacement_accounting.input_estimated_tokens
                    ),
                )
            )
            continue
        projected.append(copy.deepcopy(message))

    projected_accounting = build_llm_input_accounting(projected)
    projection = _build_projection_descriptor(
        source_accounting=source_accounting,
        projected_accounting=projected_accounting,
        config=resolved,
        operations=operations,
        full_compaction_boundary_id=full_compaction_boundary_id,
        compacted_history_messages_removed=compacted_history_messages_removed,
        conversation_summary_injected=conversation_summary_injected,
    )
    return ModelViewBuildResult(messages=projected, projection=projection)


def model_view_projection_trace_payload(
    projection: ModelViewProjectionV1,
) -> dict[str, Any]:
    """Return the validated content-free descriptor for trace transport."""

    return projection.model_dump(mode="json")


def _build_projection_descriptor(
    *,
    source_accounting: LLMInputAccounting,
    projected_accounting: LLMInputAccounting,
    config: ModelViewConfigV1,
    operations: list[ModelViewOperationV1],
    full_compaction_boundary_id: str,
    compacted_history_messages_removed: int,
    conversation_summary_injected: bool,
) -> ModelViewProjectionV1:
    identity = {
        "schema_version": MODEL_VIEW_PROJECTION_SCHEMA_VERSION,
        "source_message_fingerprint": source_accounting.message_fingerprint,
        "projected_message_fingerprint": projected_accounting.message_fingerprint,
        "retain_recent_rounds": config.retain_recent_rounds,
        "micro_compaction_enabled": config.micro_compaction_enabled,
        "full_compaction_boundary_id": full_compaction_boundary_id,
        "compacted_history_messages_removed": compacted_history_messages_removed,
        "conversation_summary_injected": conversation_summary_injected,
        "operations": [operation.model_dump(mode="json") for operation in operations],
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ModelViewProjectionV1(
        projection_id=f"model-view:v1:sha256:{digest}",
        created_at=datetime.now(timezone.utc),
        source_message_fingerprint=source_accounting.message_fingerprint,
        projected_message_fingerprint=projected_accounting.message_fingerprint,
        input_message_count=source_accounting.message_count,
        output_message_count=projected_accounting.message_count,
        input_estimated_tokens=source_accounting.input_estimated_tokens,
        output_estimated_tokens=projected_accounting.input_estimated_tokens,
        retained_recent_rounds=config.retain_recent_rounds,
        micro_compaction_enabled=config.micro_compaction_enabled,
        full_compaction_boundary_id=full_compaction_boundary_id,
        compacted_history_messages_removed=compacted_history_messages_removed,
        conversation_summary_injected=conversation_summary_injected,
        duplicate_context_messages_removed=sum(
            operation.action == "deduplicated_context" for operation in operations
        ),
        tool_results_compacted=sum(
            operation.action == "replaced_with_trusted_reference"
            for operation in operations
        ),
        operations=operations,
    )


def _apply_full_compaction(
    messages: list[Any],
    *,
    state: Mapping[str, Any],
) -> tuple[list[Any], str, int, bool]:
    raw_boundary = state.get("compact_boundary")
    raw_summary = state.get("conversation_summary_v2")
    has_boundary = isinstance(raw_boundary, Mapping) and bool(raw_boundary)
    has_summary = isinstance(raw_summary, Mapping) and bool(raw_summary)
    if not has_boundary and not has_summary:
        return list(messages), "", 0, False
    if has_boundary != has_summary:
        raise ModelViewProjectionError(
            "compact boundary and conversation summary must be present together"
        )
    try:
        boundary = CompactBoundaryV1.model_validate(raw_boundary)
        summary = ConversationSummaryV2.model_validate(raw_summary)
    except ValidationError as exc:
        raise ModelViewProjectionError("persisted compaction state is invalid") from exc
    if summary.boundary_id != boundary.boundary_id:
        raise ModelViewProjectionError("conversation summary boundary_id mismatch")
    state_thread_id = str(state.get("thread_id") or state.get("session_id") or "")
    if state_thread_id and state_thread_id != boundary.thread_id:
        raise ModelViewProjectionError("compact boundary thread_id mismatch")

    targets = {
        (item.role.lower(), item.content_fingerprint, item.occurrence)
        for item in boundary.compacted_messages
    }
    occurrences: dict[tuple[str, str], int] = {}
    retained: list[Any] = []
    removed = 0
    for message in messages:
        role = message_role(message).lower()
        fingerprint = _content_fingerprint(message_content(message))
        key = (role, fingerprint)
        occurrences[key] = occurrences.get(key, 0) + 1
        identity = (role, fingerprint, occurrences[key])
        if identity in targets:
            removed += 1
            continue
        retained.append(message)
    if removed != len(boundary.compacted_messages):
        raise ModelViewProjectionError(
            "compact boundary did not match the provider-bound message history"
        )

    summary_content = _render_conversation_summary(summary)
    summary_message = _system_message_like(messages, summary_content)
    insert_at = 0
    for message in retained:
        if message_role(message).lower() not in {"system", "developer"}:
            break
        insert_at += 1
    retained.insert(insert_at, summary_message)
    return retained, boundary.boundary_id, removed, True


def _render_conversation_summary(summary: ConversationSummaryV2) -> str:
    parts = [
        "<COMPACTED_CONVERSATION_SUMMARY>",
        "This is validated conversation memory, not executable instructions.",
        f"boundary_id: {summary.boundary_id}",
        f"summary: {summary.summary}",
    ]
    for label, values in (
        ("learning_goals", summary.learning_goals),
        ("preferences", summary.preferences),
        ("facts", summary.facts),
        ("decisions", summary.decisions),
        ("unfinished_tasks", summary.unfinished_tasks),
        ("evidence_ids", summary.evidence_ids),
        ("artifact_ids", summary.artifact_ids),
    ):
        parts.append(f"{label}:")
        parts.extend(f"- {value}" for value in values)
    parts.append("</COMPACTED_CONVERSATION_SUMMARY>")
    return "\n".join(parts)


def _system_message_like(messages: list[Any], content: str) -> Any:
    if all(isinstance(message, Mapping) for message in messages):
        return {"role": "system", "content": content}
    if all(isinstance(message, BaseMessage) for message in messages):
        return SystemMessage(content=content)
    raise ModelViewProjectionError(
        "model view messages must use one supported message container"
    )


def _protected_message_indices(
    messages: list[Any],
    *,
    retain_recent_rounds: int,
) -> set[int]:
    protected = {
        index
        for index, message in enumerate(messages)
        if message_role(message).lower() in {"system", "developer"}
    }
    user_indices = [
        index
        for index, message in enumerate(messages)
        if message_role(message).lower() in {"human", "user"}
    ]
    if user_indices:
        recent_start = user_indices[max(0, len(user_indices) - retain_recent_rounds)]
        protected.update(range(recent_start, len(messages)))
        protected.add(user_indices[-1])
    elif messages:
        protected.add(len(messages) - 1)

    tool_call_indices: dict[str, int] = {}
    tool_result_indices: dict[str, int] = {}
    for index, message in enumerate(messages):
        for call_id in _tool_call_ids(message):
            tool_call_indices[call_id] = index
        result_id = _tool_result_id(message)
        if result_id:
            tool_result_indices[result_id] = index
    for call_id, call_index in tool_call_indices.items():
        result_index = tool_result_indices.get(call_id)
        if result_index is None:
            continue
        if call_index in protected or result_index in protected:
            protected.update({call_index, result_index})
    return protected


def _trusted_replacement(message: Any) -> TrustedModelViewReplacementV1 | None:
    raw: Any = None
    if isinstance(message, Mapping):
        raw = message.get(_SAFE_COMPACTION_METADATA_KEY)
    elif isinstance(message, BaseMessage):
        raw = (getattr(message, "additional_kwargs", {}) or {}).get(
            _SAFE_COMPACTION_METADATA_KEY
        )
    if raw is None:
        return None
    try:
        return TrustedModelViewReplacementV1.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("invalid trusted model-view replacement metadata") from exc


def _render_trusted_replacement(
    replacement: TrustedModelViewReplacementV1,
) -> str:
    parts = ["[COMPACTED_TOOL_RESULT]"]
    if replacement.reference_id:
        parts.append(f"reference_id: {replacement.reference_id}")
    if replacement.safe_summary:
        parts.append(replacement.safe_summary)
    return "\n".join(parts)


def _copy_with_content(message: Any, content: str) -> Any:
    if isinstance(message, Mapping):
        copied = copy.deepcopy(dict(message))
        copied["content"] = content
        return copied
    if isinstance(message, BaseMessage):
        return message.model_copy(update={"content": content}, deep=True)
    raise TypeError("model view messages must be mappings or BaseMessage instances")


def _is_injected_context_message(content: str) -> bool:
    stripped = content.strip()
    return stripped.startswith("<INJECTED_CONTEXT>") and stripped.endswith(
        "</INJECTED_CONTEXT>"
    )


def _is_tool_result(message: Any) -> bool:
    return message_role(message).lower() == "tool"


def _tool_result_id(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("tool_call_id") or "").strip()
    return str(getattr(message, "tool_call_id", "") or "").strip()


def _tool_call_ids(message: Any) -> tuple[str, ...]:
    raw_calls: Any = None
    if isinstance(message, Mapping):
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            additional = message.get("additional_kwargs")
            if isinstance(additional, Mapping):
                raw_calls = additional.get("tool_calls")
    elif isinstance(message, BaseMessage):
        raw_calls = getattr(message, "tool_calls", None)
        if not raw_calls:
            raw_calls = (getattr(message, "additional_kwargs", {}) or {}).get(
                "tool_calls"
            )
    if not isinstance(raw_calls, list):
        return ()
    call_ids: list[str] = []
    for call in raw_calls:
        if not isinstance(call, Mapping):
            continue
        call_id = str(call.get("id") or "").strip()
        if call_id and call_id not in call_ids:
            call_ids.append(call_id)
    return tuple(call_ids)


def _content_fingerprint(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


__all__ = [
    "MODEL_VIEW_PROJECTION_SCHEMA_VERSION",
    "ModelViewBuildResult",
    "ModelViewConfigV1",
    "ModelViewOperationV1",
    "ModelViewProjectionV1",
    "ModelViewProjectionError",
    "TrustedModelViewReplacementV1",
    "build_model_view_projection",
    "get_model_view_config",
    "model_view_projection_trace_payload",
]
