"""Runtime capture boundary for the Context Influence Ledger.

This module records compact influence metadata during a graph node invocation.
It never changes provider-bound messages and never stores raw prompt bodies,
retrieved documents, or full generated artifacts.
"""

from __future__ import annotations

import inspect
import json
import logging
from contextvars import ContextVar, Token
from functools import wraps
from typing import Any, Awaitable, Callable, Iterable, Mapping, TypeVar, cast

from src.context_engineering.influence import (
    ContextInfluenceEntry,
    ContextInfluenceLedgerUpdate,
    build_influence_entry,
    build_influence_update,
    content_fingerprint,
    merge_context_influence_ledger,
)
from src.context_engineering.itemizer import sanitize_metadata
from src.context_engineering.tokenizer import message_content_to_text
from src.context_engineering.workspace import sanitize_workspace_text
from src.observability.node_registry import (
    InfluenceCaptureRule,
    NodeRuntimeMetadata,
    get_node_runtime_metadata,
)
from src.observability.a3_trace import emit_a3_trace

T = TypeVar("T")
_CAPTURE_BUFFER: ContextVar[list[ContextInfluenceEntry] | None] = ContextVar(
    "context_influence_capture_buffer",
    default=None,
)
logger = logging.getLogger(__name__)

_SAFE_VALUE_KEYS = (
    "artifact_id",
    "evidence_id",
    "gap_id",
    "resource_type",
    "title",
    "summary",
    "message_preview",
    "status",
    "verdict",
    "reason",
    "review_reason",
    "decision_summary",
    "revision_notes",
    "outline",
    "plan",
    "subject",
    "active_subject",
    "active_learning_goal",
    "purpose",
)
_IDENTITY_METADATA_KEYS = (
    "artifact_id",
    "evidence_id",
    "gap_id",
    "manifest_id",
    "resource_type",
    "workflow",
    "iteration",
    "schema_name",
    "output_mode",
)


def begin_influence_capture() -> Token[list[ContextInfluenceEntry] | None]:
    """Start an isolated capture scope and return its ContextVar token."""
    return _CAPTURE_BUFFER.set([])


def end_influence_capture(
    token: Token[list[ContextInfluenceEntry] | None],
) -> list[ContextInfluenceEntry]:
    """Return captured entries and restore the enclosing capture scope."""
    entries = list(_CAPTURE_BUFFER.get() or [])
    _CAPTURE_BUFFER.reset(token)
    return entries


def record_influence_entry(entry: Mapping[str, Any]) -> bool:
    """Record an already-sanitized entry when a capture scope is active."""
    buffer = _CAPTURE_BUFFER.get()
    if buffer is None:
        return False
    influence_id = str(entry.get("influence_id") or "").strip()
    if not influence_id:
        return False
    if any(item.get("influence_id") == influence_id for item in buffer):
        return True
    buffer.append(cast(ContextInfluenceEntry, dict(entry)))
    return True


def record_llm_input_influences(
    *,
    node_name: str,
    llm_node: str,
    messages: list[Any],
    state: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
    schema_name: str = "",
    output_mode: str = "",
) -> None:
    """Capture safe provider-input structure without persisting message bodies."""
    state_payload = state or {}
    metadata = get_node_runtime_metadata(node_name)
    message_stats = _message_stats(messages)
    common = _runtime_metadata(metadata, state_payload)
    manifest_id = _safe_text(manifest.get("manifest_id"), 180)
    prompt_entry = build_influence_entry(
        state=state_payload,
        kind="prompt_snapshot_metadata",
        source_node=node_name,
        target_stage="provider_input",
        title="Prompt snapshot metadata",
        preview="",
        metadata={
            **common,
            **message_stats,
            "llm_node": _safe_text(llm_node, 120),
            "manifest_id": manifest_id,
        },
        priority=40,
        injectable=False,
        fingerprint_source=json.dumps(message_stats, sort_keys=True),
    )
    record_influence_entry(prompt_entry)

    provider_entry = build_influence_entry(
        state=state_payload,
        kind="provider_bound_messages_metadata",
        source_node=node_name,
        target_stage="provider_input",
        title="Provider-bound message metadata",
        preview="",
        metadata={
            **common,
            "manifest_id": manifest_id,
            "llm_node": _safe_text(llm_node, 120),
            "provider": _safe_text(manifest.get("provider"), 80),
            "model": _safe_text(manifest.get("model"), 160),
            "message_count": message_stats["message_count"],
            "context_apply_applied": bool(manifest.get("context_apply_applied")),
            "provider_bound_messages_mutated": bool(
                manifest.get("provider_bound_messages_mutated")
            ),
        },
        priority=45,
        injectable=False,
        fingerprint_source=manifest_id or json.dumps(message_stats, sort_keys=True),
    )
    record_influence_entry(provider_entry)

    if schema_name:
        schema_entry = build_influence_entry(
            state=state_payload,
            kind="schema_contract",
            source_node=node_name,
            target_stage="provider_input",
            title="Structured output contract",
            preview="",
            metadata={
                **common,
                "manifest_id": manifest_id,
                "schema_name": _safe_text(schema_name, 160),
                "output_mode": _safe_text(output_mode, 80),
                "schema_size_chars": _safe_non_negative_int(
                    manifest.get("schema_size_chars")
                ),
            },
            priority=50,
            injectable=False,
            fingerprint_source=f"{schema_name}:{output_mode}",
        )
        record_influence_entry(schema_entry)


def record_structured_output_influence(
    *,
    node_name: str,
    output: Any,
    state: Mapping[str, Any] | None,
) -> None:
    """Capture compact validated structured output for a registered logical node."""
    for entry in build_node_output_influences(
        node_name=node_name,
        output=_mapping_from_value(output),
        state=state,
    ):
        record_influence_entry(entry)


def record_plain_output_influence(
    *,
    node_name: str,
    output: object,
    state: Mapping[str, Any] | None,
) -> None:
    metadata = get_node_runtime_metadata(node_name)
    if metadata is None or metadata.role not in {"agent", "reviewer", "consensus"}:
        return
    kind = {
        "reviewer": "reviewer_output",
        "consensus": "consensus_output",
    }.get(metadata.role, "agent_output")
    entry = build_influence_entry(
        state=state,
        kind=kind,
        source_node=node_name,
        target_stage="downstream",
        title=metadata.label,
        preview=_safe_text(output, 600),
        metadata=_runtime_metadata(metadata, state or {}),
        priority=65,
        fingerprint_source=output,
    )
    record_influence_entry(entry)


def build_node_output_influences(
    *,
    node_name: str,
    output: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
) -> list[ContextInfluenceEntry]:
    """Build compact output influences according to canonical node metadata."""
    metadata = get_node_runtime_metadata(node_name)
    if metadata is None or not isinstance(output, Mapping):
        return []
    entries: list[ContextInfluenceEntry] = []
    for rule in metadata.capture_rules:
        entries.extend(
            _entries_for_rule(
                metadata=metadata,
                rule=rule,
                output=output,
                state=state or {},
            )
        )
    return _dedupe_entries(entries)


def influence_entries_for_scope(
    ledger: Mapping[str, Any] | None,
    *,
    request_id: str,
    workflow: str = "",
) -> list[ContextInfluenceEntry]:
    """Read a deterministic subset for resource branch fan-in handoff."""
    normalized = merge_context_influence_ledger({}, ledger or {})
    entries_by_id = normalized.get("entries_by_id") or {}
    result: list[ContextInfluenceEntry] = []
    for influence_id in normalized.get("ordered_ids") or []:
        entry = entries_by_id.get(influence_id)
        if not isinstance(entry, Mapping):
            continue
        if request_id and str(entry.get("request_id") or "") != request_id:
            continue
        metadata = entry.get("metadata") or {}
        entry_workflow = (
            str(metadata.get("workflow") or "") if isinstance(metadata, Mapping) else ""
        )
        if workflow and entry_workflow != workflow:
            continue
        result.append(cast(ContextInfluenceEntry, dict(entry)))
    return result


def combine_influence_updates(
    *,
    state: Mapping[str, Any] | None,
    updates: Iterable[Mapping[str, Any] | None],
    entries: Iterable[Mapping[str, Any]] = (),
) -> ContextInfluenceLedgerUpdate:
    """Combine reducer updates without assigning persistent sequence numbers."""
    by_id: dict[str, Mapping[str, Any]] = {}
    diagnostics: list[str] = []
    for update in updates:
        if not isinstance(update, Mapping):
            continue
        raw_entries = update.get("entries")
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if isinstance(entry, Mapping) and entry.get("influence_id"):
                    by_id[str(entry["influence_id"])] = entry
        raw_by_id = update.get("entries_by_id")
        if isinstance(raw_by_id, Mapping):
            for entry in raw_by_id.values():
                if isinstance(entry, Mapping) and entry.get("influence_id"):
                    by_id[str(entry["influence_id"])] = entry
        raw_diagnostics = update.get("diagnostics")
        if isinstance(raw_diagnostics, list):
            diagnostics.extend(str(item) for item in raw_diagnostics)
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("influence_id"):
            by_id[str(entry["influence_id"])] = entry
    return build_influence_update(
        state=state,
        entries=by_id.values(),
        diagnostics=diagnostics,
    )


def emit_influence_capture_trace(
    *,
    node_name: str,
    entries: Iterable[Mapping[str, Any]],
    state: Mapping[str, Any] | None,
) -> None:
    """Emit counts and stable IDs only; entry previews never enter trace logs."""
    safe_entries = [entry for entry in entries if isinstance(entry, Mapping)]
    if not safe_entries:
        return
    counts: dict[str, int] = {}
    token_estimate = 0
    influence_ids: list[str] = []
    for entry in safe_entries:
        kind = _safe_text(entry.get("kind"), 80)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
        token_estimate += _safe_non_negative_int(entry.get("token_estimate"))
        influence_id = _safe_text(entry.get("influence_id"), 180)
        if influence_id and influence_id not in influence_ids:
            influence_ids.append(influence_id)
    emit_a3_trace(
        logger,
        "context_influence.captured",
        {
            "node_name": _safe_text(node_name, 120),
            "entry_count": len(safe_entries),
            "counts_by_kind": dict(sorted(counts.items())),
            "token_estimate": token_estimate,
            "influence_ids": influence_ids[:20],
        },
        state=dict(state or {}),
        env_flag="LOG_A3_TRACE",
    )


def wrap_context_influence_node(
    node_name: str,
    node_fn: Callable[..., Any],
) -> Callable[..., Awaitable[Any]]:
    """Wrap one LangGraph node with bounded influence capture."""

    @wraps(node_fn)
    async def _wrapped(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        token = begin_influence_capture()
        try:
            metadata = get_node_runtime_metadata(node_name)
            if metadata and metadata.capture_current_user_query:
                query = _current_user_query(state.get("messages") or [])
                if query:
                    record_influence_entry(
                        build_influence_entry(
                            state=state,
                            kind="original_user_query",
                            source_node=node_name,
                            target_stage="routing",
                            title="Current user request",
                            preview=query,
                            metadata=_runtime_metadata(metadata, state),
                            priority=100,
                            injectable=False,
                            fingerprint_source=query,
                        )
                    )
            result = node_fn(state, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            output = result if isinstance(result, Mapping) else None
            for entry in build_node_output_influences(
                node_name=node_name,
                output=output,
                state=state,
            ):
                record_influence_entry(entry)
            captured = end_influence_capture(token)
        except BaseException:
            end_influence_capture(token)
            raise
        if not isinstance(result, Mapping):
            return result
        emit_influence_capture_trace(
            node_name=node_name,
            entries=captured,
            state=state,
        )
        existing_update = result.get("context_influence_ledger")
        if not captured and not isinstance(existing_update, Mapping):
            return result
        return {
            **dict(result),
            "context_influence_ledger": combine_influence_updates(
                state=state,
                updates=(
                    existing_update if isinstance(existing_update, Mapping) else None,
                ),
                entries=captured,
            ),
        }

    return _wrapped


def _entries_for_rule(
    *,
    metadata: NodeRuntimeMetadata,
    rule: InfluenceCaptureRule,
    output: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[ContextInfluenceEntry]:
    result: list[ContextInfluenceEntry] = []
    common_metadata = _runtime_metadata(metadata, state)
    for field in rule.list_fields:
        raw_items = output.get(field)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items[:20]:
            preview, item_metadata, fingerprint_source = _compact_value(item, field)
            result.append(
                build_influence_entry(
                    state=state,
                    kind=rule.kind,
                    source_node=metadata.node_id,
                    target_stage="downstream",
                    title=_capture_title(metadata, item, field),
                    preview=preview,
                    metadata={**common_metadata, **item_metadata, "field": field},
                    priority=rule.priority,
                    injectable=rule.injectable,
                    fingerprint_source=fingerprint_source,
                )
            )

    present_fields = [field for field in rule.preview_fields if field in output]
    if present_fields:
        previews: list[str] = []
        identity: dict[str, Any] = {}
        fingerprints: list[object] = []
        for field in present_fields:
            preview, item_metadata, fingerprint_source = _compact_value(
                output.get(field), field
            )
            if preview:
                previews.append(f"{field}: {preview}")
            identity.update(item_metadata)
            fingerprints.append(fingerprint_source)
        if previews or identity:
            result.append(
                build_influence_entry(
                    state=state,
                    kind=rule.kind,
                    source_node=metadata.node_id,
                    target_stage="downstream",
                    title=metadata.label,
                    preview=" | ".join(previews),
                    metadata={
                        **common_metadata,
                        **identity,
                        "fields": present_fields,
                    },
                    priority=rule.priority,
                    injectable=rule.injectable,
                    fingerprint_source=json.dumps(
                        fingerprints,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                )
            )
    return result


def _compact_value(value: Any, field: str) -> tuple[str, dict[str, Any], object]:
    value = _plain_value(value)
    if field == "messages":
        messages = value if isinstance(value, list) else [value]
        last = messages[-1] if messages else ""
        text = message_content_to_text(last)
        return _safe_text(text, 600), {"message_count": len(messages)}, text
    if field == "task_workspace" and isinstance(value, Mapping):
        metadata = {
            "workspace_id": _safe_text(value.get("workspace_id"), 180),
            "active_subject": _safe_text(value.get("active_subject"), 160),
            "evidence_summary_count": _section_count(
                value, "evidence_summaries", "evidence_summaries_by_id"
            ),
            "coverage_gap_count": _section_count(
                value, "coverage_gaps", "coverage_gaps_by_id"
            ),
            "artifact_count": _section_count(value, "artifacts", "artifacts_by_id"),
        }
        preview = (
            f"subject={metadata['active_subject']}; "
            f"evidence={metadata['evidence_summary_count']}; "
            f"gaps={metadata['coverage_gap_count']}; "
            f"artifacts={metadata['artifact_count']}"
        )
        return preview, sanitize_metadata(metadata), metadata
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key in _SAFE_VALUE_KEYS:
            if key not in value:
                continue
            item = value.get(key)
            if isinstance(item, (str, int, float, bool)) or item is None:
                safe[key] = _safe_text(item, 240) if isinstance(item, str) else item
        if not safe:
            safe = {
                "field_count": len(value),
                "field_names": sorted(_safe_text(key, 80) for key in value)[:20],
            }
        metadata = {key: safe[key] for key in _IDENTITY_METADATA_KEYS if key in safe}
        return (
            _safe_text(
                json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str),
                600,
            ),
            sanitize_metadata(metadata),
            safe,
        )
    if isinstance(value, list):
        labels: list[str] = []
        for item in value[:8]:
            plain = _plain_value(item)
            if isinstance(plain, Mapping):
                label = next(
                    (
                        plain.get(key)
                        for key in ("title", "id", "source_id", "evidence_id", "gap_id")
                        if plain.get(key)
                    ),
                    "",
                )
                if label:
                    labels.append(_safe_text(label, 100))
            elif isinstance(plain, (str, int, float, bool)):
                labels.append(_safe_text(plain, 100))
        summary = {"count": len(value), "labels": labels}
        return (
            json.dumps(summary, ensure_ascii=False),
            {"item_count": len(value)},
            summary,
        )
    text = _safe_text(value, 600)
    return text, {}, value


def _runtime_metadata(
    metadata: NodeRuntimeMetadata | None,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    if metadata is None:
        return {}
    iteration = 0
    if metadata.iteration_field:
        iteration = _safe_non_negative_int(state.get(metadata.iteration_field))
    return sanitize_metadata(
        {
            "node_role": metadata.role,
            "node_operation": metadata.operation,
            "node_group": metadata.group,
            "stage_rank": metadata.stage_rank,
            "parent_node": metadata.parent,
            "workflow": metadata.workflow,
            "iteration": iteration,
        }
    )


def _message_stats(messages: list[Any]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    char_count = 0
    fingerprints: list[str] = []
    for message in messages or []:
        role = _message_role(message)
        roles[role] = roles.get(role, 0) + 1
        text = message_content_to_text(message)
        char_count += len(text)
        fingerprints.append(content_fingerprint(text))
    return {
        "message_count": len(messages or []),
        "message_role_counts": dict(sorted(roles.items())),
        "message_char_count": char_count,
        "message_fingerprint": content_fingerprint("|".join(fingerprints)),
    }


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        role = str(message.get("role") or "").strip().lower()
        return role or "unknown"
    message_type = str(getattr(message, "type", "") or "").strip().lower()
    if message_type:
        return message_type
    class_name = type(message).__name__.lower()
    if "human" in class_name:
        return "user"
    if "system" in class_name:
        return "system"
    if "ai" in class_name:
        return "assistant"
    return "unknown"


def _current_user_query(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if _message_role(message) in {"user", "human"}:
            return _safe_text(message_content_to_text(message), 600)
    return ""


def _capture_title(metadata: NodeRuntimeMetadata, item: Any, field: str) -> str:
    plain = _plain_value(item)
    if isinstance(plain, Mapping):
        for key in ("title", "subject", "resource_type", "evidence_id", "gap_id"):
            if plain.get(key):
                return _safe_text(plain.get(key), 160)
    return f"{metadata.label}: {field}"


def _plain_value(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="python")
    return value


def _mapping_from_value(value: Any) -> Mapping[str, Any]:
    plain = _plain_value(value)
    return plain if isinstance(plain, Mapping) else {}


def _section_count(value: Mapping[str, Any], list_key: str, map_key: str) -> int:
    mapped = value.get(map_key)
    if isinstance(mapped, Mapping):
        return len(mapped)
    listed = value.get(list_key)
    return len(listed) if isinstance(listed, list) else 0


def _dedupe_entries(
    entries: Iterable[ContextInfluenceEntry],
) -> list[ContextInfluenceEntry]:
    by_id = {
        entry["influence_id"]: entry for entry in entries if entry.get("influence_id")
    }
    return [by_id[key] for key in sorted(by_id)]


def _safe_text(value: object, max_chars: int) -> str:
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _safe_non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
