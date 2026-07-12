"""Pure thread-level context window projection and next-call estimation.

The estimator is deliberately read-only. It never invokes a model, provider,
retriever, search client, checkpointer, or graph node. Fingerprints are used
only to decide whether prior accounting can be reused; token counts always
come from current text accounting or a validated prior manifest report.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from src.context_engineering.input_accounting import build_llm_input_accounting
from src.context_engineering.tokenizer import estimate_text_tokens_mixed
from src.context_engineering.workspace import (
    sanitize_workspace_text,
    utc_now_iso,
    workspace_status_payload,
)

THREAD_CONTEXT_WINDOW_SCHEMA_VERSION = 2
THREAD_CONTEXT_WINDOW_CONTRACT = "thread_context_window_v2"
SECTION_LIMIT = 16
MESSAGE_LIMIT = 32
MESSAGE_CHAR_LIMIT = 96_000
SECTION_CHAR_LIMIT = 48_000

EstimateBasis = Literal["known_next_node", "thread_baseline"]
EstimateConfidence = Literal["high", "medium", "low"]


class ContextSectionEstimate(TypedDict):
    section: str
    source: str
    item_count: int
    message_count: int
    char_count: int
    estimated_tokens: int
    known: bool


class NextCallContextEstimate(TypedDict):
    basis: EstimateBasis
    confidence: EstimateConfidence
    estimated: bool
    estimated_at: str
    target_node: str
    unknown_sections: list[str]
    sections: list[ContextSectionEstimate]
    estimated_input_tokens: int
    estimated_output_reserved_tokens: int
    estimated_used_tokens: int
    max_context_tokens: int
    used_ratio: float
    tokenizer_mode: str
    state_fingerprint: str
    reused_manifest_statistics: bool
    known_section_ratio: float


class LastLLMCallUsage(TypedDict):
    present: bool
    report_id: str
    manifest_id: str
    created_at: str
    node_name: str
    llm_node: str
    model: str
    input_estimated_tokens: int
    output_reserved_tokens: int
    used_tokens: int
    max_context_tokens: int
    used_ratio: float
    estimated: bool
    tokenizer_mode: str
    sections: list[ContextSectionEstimate]


class BackgroundInventory(TypedDict):
    conversation_summary_present: bool
    selected_memory_count: int
    evidence_summary_count: int
    artifact_summary_count: int
    workspace_present: bool
    workspace_active_subject: str
    workspace_evidence_summary_count: int
    workspace_gap_count: int
    workspace_artifact_count: int
    workspace_updated_at: str
    manifest_count: int
    influence_entry_count: int


class ThreadContextWindowV2(TypedDict):
    schema_version: Literal[2]
    contract: Literal["thread_context_window_v2"]
    thread_id: str
    updated_at: str
    next_call_context_estimate: NextCallContextEstimate
    last_llm_call_usage: LastLLMCallUsage
    background_inventory: BackgroundInventory


def build_thread_context_window_v2(
    state: Mapping[str, Any] | None,
    *,
    target_node: str = "",
) -> ThreadContextWindowV2:
    """Build the additive v2 window without mutating or enriching state."""

    state_payload = state if isinstance(state, Mapping) else {}
    safe_target = _safe_text(target_node, 120)
    latest_report = _latest_mapping(
        state_payload.get("context_usage_report"),
        state_payload.get("context_usage_reports"),
    )
    manifests = _mapping_items(state_payload.get("llm_input_manifests"))
    current_manifest = state_payload.get("llm_input_manifest")
    if isinstance(current_manifest, Mapping):
        manifests = [current_manifest, *manifests]
    next_estimate = _next_call_estimate(
        state_payload,
        target_node=safe_target,
        latest_report=latest_report,
        manifests=manifests,
    )
    return {
        "schema_version": THREAD_CONTEXT_WINDOW_SCHEMA_VERSION,
        "contract": THREAD_CONTEXT_WINDOW_CONTRACT,
        "thread_id": _safe_text(
            state_payload.get("thread_id") or state_payload.get("session_id"),
            120,
        ),
        "updated_at": utc_now_iso(),
        "next_call_context_estimate": next_estimate,
        "last_llm_call_usage": _last_call_usage(latest_report),
        "background_inventory": _background_inventory(state_payload, manifests),
    }


def _next_call_estimate(
    state: Mapping[str, Any],
    *,
    target_node: str,
    latest_report: Mapping[str, Any],
    manifests: list[Mapping[str, Any]],
) -> NextCallContextEstimate:
    message_values = _value_list(state.get("messages"))[-MESSAGE_LIMIT:]
    message_accounting = build_llm_input_accounting(message_values)
    message_chars = min(message_accounting.prompt_chars, MESSAGE_CHAR_LIMIT)
    message_tokens = (
        message_accounting.input_estimated_tokens
        if message_accounting.prompt_chars <= MESSAGE_CHAR_LIMIT
        else _bounded_message_tokens(message_values)
    )
    sections = [
        _section(
            "recent_messages",
            source="current_thread_state",
            item_count=len(message_values),
            message_count=len(message_values),
            char_count=message_chars,
            estimated_tokens=message_tokens,
        ),
        _text_section(
            "conversation_summary",
            state.get("conversation_summary"),
            source="current_thread_state",
        ),
        _workspace_section(state.get("task_workspace")),
        _profile_memory_section(state),
    ]
    sections = [item for item in sections if item["item_count"] or item["char_count"]]
    state_fingerprint = _state_fingerprint(sections)
    matching_report = _report_for_node(
        target_node,
        latest_report=latest_report,
        state=state,
    )
    matching_manifest = _manifest_for_report(matching_report, manifests)
    exact_reuse = bool(
        target_node
        and matching_report
        and matching_manifest
        and _safe_text(matching_manifest.get("message_fingerprint"), 80)
        == message_accounting.message_fingerprint
    )

    basis: EstimateBasis = "known_next_node" if target_node else "thread_baseline"
    unknown_sections = [
        "fixed_system_and_business_prompt",
        "tool_definitions",
        "structured_output_contract",
        "ce_block",
    ]
    if message_accounting.prompt_chars > MESSAGE_CHAR_LIMIT:
        unknown_sections.append("recent_messages_overflow")

    if exact_reuse:
        input_tokens = _safe_int(matching_report.get("input_estimated_tokens"))
        output_reserved = _safe_int(matching_report.get("output_reserved_tokens"))
        max_tokens = _safe_int(matching_report.get("max_context_tokens"))
        report_sections = _report_sections(matching_report)
        if report_sections:
            sections = report_sections
        unknown_sections = []
        confidence: EstimateConfidence = "high"
    else:
        input_tokens = sum(item["estimated_tokens"] for item in sections)
        output_reserved = 0
        max_tokens = _safe_int(matching_report.get("max_context_tokens"))
        if not max_tokens:
            max_tokens = _safe_int(latest_report.get("max_context_tokens"))
            if latest_report:
                unknown_sections.append("target_model_context_limit")
        confidence = "medium" if target_node and matching_report else "low"
        unknown_sections.append("output_reserved_tokens")

    unknown_sections = _unique_text(unknown_sections, limit=SECTION_LIMIT)
    used_tokens = input_tokens + output_reserved
    known_ratio = (
        round(len(sections) / (len(sections) + len(unknown_sections)), 4)
        if sections or unknown_sections
        else 0.0
    )
    return {
        "basis": basis,
        "confidence": confidence,
        "estimated": True,
        "estimated_at": utc_now_iso(),
        "target_node": target_node,
        "unknown_sections": unknown_sections,
        "sections": sections[:SECTION_LIMIT],
        "estimated_input_tokens": input_tokens,
        "estimated_output_reserved_tokens": output_reserved,
        "estimated_used_tokens": used_tokens,
        "max_context_tokens": max_tokens,
        "used_ratio": round(used_tokens / max_tokens, 6) if max_tokens else 0.0,
        "tokenizer_mode": "estimated_mixed",
        "state_fingerprint": state_fingerprint,
        "reused_manifest_statistics": exact_reuse,
        "known_section_ratio": known_ratio,
    }


def _last_call_usage(report: Mapping[str, Any]) -> LastLLMCallUsage:
    if not report:
        return {
            "present": False,
            "report_id": "",
            "manifest_id": "",
            "created_at": "",
            "node_name": "",
            "llm_node": "",
            "model": "",
            "input_estimated_tokens": 0,
            "output_reserved_tokens": 0,
            "used_tokens": 0,
            "max_context_tokens": 0,
            "used_ratio": 0.0,
            "estimated": True,
            "tokenizer_mode": "",
            "sections": [],
        }
    return {
        "present": True,
        "report_id": _safe_text(report.get("report_id"), 180),
        "manifest_id": _safe_text(report.get("manifest_id"), 180),
        "created_at": _safe_text(report.get("created_at"), 80),
        "node_name": _safe_text(report.get("node_name"), 120),
        "llm_node": _safe_text(report.get("llm_node"), 120),
        "model": _safe_text(report.get("model"), 160),
        "input_estimated_tokens": _safe_int(report.get("input_estimated_tokens")),
        "output_reserved_tokens": _safe_int(report.get("output_reserved_tokens")),
        "used_tokens": _safe_int(report.get("used_tokens")),
        "max_context_tokens": _safe_int(report.get("max_context_tokens")),
        "used_ratio": _safe_ratio(report.get("used_ratio")),
        "estimated": bool(report.get("estimated", True)),
        "tokenizer_mode": _safe_text(report.get("tokenizer_mode"), 80),
        "sections": _report_sections(report),
    }


def _background_inventory(
    state: Mapping[str, Any],
    manifests: list[Mapping[str, Any]],
) -> BackgroundInventory:
    workspace = workspace_status_payload(
        state.get("task_workspace")
        if isinstance(state.get("task_workspace"), dict)
        else {}
    )
    influence = state.get("context_influence_ledger")
    influence_entries = (
        influence.get("entries") if isinstance(influence, Mapping) else []
    )
    return {
        "conversation_summary_present": bool(
            str(state.get("conversation_summary") or "").strip()
        ),
        "selected_memory_count": _list_len(
            state.get("selected_evidence_memory_summaries")
        )
        + _list_len(state.get("episodic_memory_results"))
        + _list_len(state.get("semantic_memory_results"))
        + int(bool(state.get("profile") or state.get("learner_profile"))),
        "evidence_summary_count": _evidence_count(state),
        "artifact_summary_count": _artifact_count(state),
        "workspace_present": bool(workspace.get("workspace_present")),
        "workspace_active_subject": _safe_text(
            workspace.get("workspace_active_subject"), 120
        ),
        "workspace_evidence_summary_count": _safe_int(
            workspace.get("workspace_evidence_summary_count")
        ),
        "workspace_gap_count": _safe_int(workspace.get("workspace_gap_count")),
        "workspace_artifact_count": _safe_int(
            workspace.get("workspace_artifact_count")
        ),
        "workspace_updated_at": _safe_text(workspace.get("workspace_updated_at"), 80),
        "manifest_count": len(
            {
                _safe_text(item.get("manifest_id"), 180)
                for item in manifests
                if _safe_text(item.get("manifest_id"), 180)
            }
        ),
        "influence_entry_count": _list_len(influence_entries),
    }


def _report_sections(report: Mapping[str, Any]) -> list[ContextSectionEstimate]:
    sections: list[ContextSectionEstimate] = []
    for item in report.get("detailed_categories") or []:
        if not isinstance(item, Mapping):
            continue
        name = _safe_text(item.get("category"), 80)
        if not name:
            continue
        sections.append(
            _section(
                name,
                source="last_llm_call_manifest",
                item_count=_safe_int(item.get("segment_count")),
                message_count=_safe_int(item.get("message_count")),
                char_count=0,
                estimated_tokens=_safe_int(item.get("estimated_tokens")),
            )
        )
    return sections[:SECTION_LIMIT]


def _workspace_section(value: Any) -> ContextSectionEstimate:
    workspace = value if isinstance(value, Mapping) else {}
    status = workspace_status_payload(dict(workspace))
    if not status.get("workspace_present"):
        return _section(
            "task_workspace",
            source="task_workspace",
            item_count=0,
            message_count=0,
            char_count=0,
            estimated_tokens=0,
        )
    projection: dict[str, Any] = {
        "active_subject": _safe_text(workspace.get("active_subject"), 160),
        "active_learning_goal": _safe_text(workspace.get("active_learning_goal"), 320),
        "evidence_summaries": _compact_items(
            workspace.get("evidence_summaries"),
            keys=("evidence_id", "title", "summary", "subject", "source_type"),
        ),
        "coverage_gaps": _compact_items(
            workspace.get("coverage_gaps"),
            keys=("gap_id", "title", "summary", "subject"),
        ),
        "artifacts": _workspace_artifacts(workspace),
    }
    text = json.dumps(projection, ensure_ascii=False, sort_keys=True, default=str)
    return _text_section("task_workspace", text, source="task_workspace")


def _profile_memory_section(state: Mapping[str, Any]) -> ContextSectionEstimate:
    memory_items = [
        *_value_list(state.get("selected_evidence_memory_summaries")),
        *_value_list(state.get("episodic_memory_results")),
        *_value_list(state.get("semantic_memory_results")),
    ]
    profile = state.get("profile") or state.get("learner_profile")
    if not profile and not memory_items:
        return _section(
            "selected_memory_profile",
            source="current_thread_state",
            item_count=0,
            message_count=0,
            char_count=0,
            estimated_tokens=0,
        )
    projection = {
        "profile": _compact_mapping(
            profile,
            keys=(
                "learning_goal",
                "current_foundation",
                "daily_study_time",
                "deadline",
                "preferred_learning_style",
                "weak_points",
            ),
        ),
        "memory": _compact_items(
            memory_items,
            keys=("memory_id", "summary", "subject", "purpose"),
        ),
    }
    text = json.dumps(projection, ensure_ascii=False, sort_keys=True, default=str)
    return _text_section("selected_memory_profile", text, source="current_thread_state")


def _text_section(section: str, value: Any, *, source: str) -> ContextSectionEstimate:
    text = str(value or "").strip()
    bounded = text[:SECTION_CHAR_LIMIT]
    return _section(
        section,
        source=source,
        item_count=1 if bounded else 0,
        message_count=0,
        char_count=len(bounded),
        estimated_tokens=estimate_text_tokens_mixed(bounded) if bounded else 0,
    )


def _section(
    section: str,
    *,
    source: str,
    item_count: int,
    message_count: int,
    char_count: int,
    estimated_tokens: int,
) -> ContextSectionEstimate:
    return {
        "section": _safe_text(section, 80),
        "source": _safe_text(source, 80),
        "item_count": max(item_count, 0),
        "message_count": max(message_count, 0),
        "char_count": max(char_count, 0),
        "estimated_tokens": max(estimated_tokens, 0),
        "known": True,
    }


def _bounded_message_tokens(messages: list[Any]) -> int:
    remaining = MESSAGE_CHAR_LIMIT
    tokens = 0
    for message in reversed(messages):
        if remaining <= 0:
            break
        content = _message_text(message)
        bounded = content[-remaining:]
        remaining -= len(bounded)
        tokens += estimate_text_tokens_mixed(bounded)
    return tokens


def _message_text(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _report_for_node(
    target_node: str,
    *,
    latest_report: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not target_node:
        return {}
    reports = _mapping_items(state.get("context_usage_reports"))
    if latest_report:
        reports = [latest_report, *reports]
    return next(
        (
            report
            for report in reports
            if _safe_text(report.get("node_name"), 120) == target_node
        ),
        {},
    )


def _manifest_for_report(
    report: Mapping[str, Any],
    manifests: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    manifest_id = _safe_text(report.get("manifest_id"), 180)
    if not manifest_id:
        return {}
    return next(
        (
            manifest
            for manifest in manifests
            if _safe_text(manifest.get("manifest_id"), 180) == manifest_id
        ),
        {},
    )


def _latest_mapping(primary: Any, history: Any) -> Mapping[str, Any]:
    if isinstance(primary, Mapping) and primary:
        return primary
    return next(iter(_mapping_items(history)), {})


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _compact_items(
    value: Any,
    *,
    keys: tuple[str, ...],
) -> list[dict[str, str]]:
    items = _mapping_items(value)[:24]
    return [_compact_mapping(item, keys=keys) for item in items]


def _compact_mapping(value: Any, *, keys: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _safe_text(value.get(key), 900)
        for key in keys
        if _safe_text(value.get(key), 900)
    }


def _workspace_artifacts(workspace: Mapping[str, Any]) -> list[dict[str, str]]:
    by_id = workspace.get("artifacts_by_id")
    if isinstance(by_id, Mapping):
        values = [item for item in by_id.values() if isinstance(item, Mapping)]
    else:
        values = _mapping_items(workspace.get("artifacts"))
    return [
        _compact_mapping(
            item,
            keys=("artifact_id", "resource_type", "title", "summary", "subject"),
        )
        for item in values[:24]
    ]


def _evidence_count(state: Mapping[str, Any]) -> int:
    workspace = state.get("task_workspace")
    workspace_count = _list_len(
        workspace.get("evidence_summaries") if isinstance(workspace, Mapping) else []
    )
    graded = [
        item
        for item in (state.get("graded_evidence") or [])
        if isinstance(item, Mapping) and item.get("keep") is True
    ]
    return workspace_count + len(graded)


def _artifact_count(state: Mapping[str, Any]) -> int:
    workspace = state.get("task_workspace")
    workspace_by_id = (
        workspace.get("artifacts_by_id") if isinstance(workspace, Mapping) else {}
    )
    workspace_count = (
        len(workspace_by_id) if isinstance(workspace_by_id, Mapping) else 0
    )
    return (
        workspace_count
        + _list_len(state.get("last_generated_artifacts"))
        + len(state.get("resource_artifacts_by_type") or {})
        if isinstance(state.get("resource_artifacts_by_type") or {}, Mapping)
        else workspace_count + _list_len(state.get("last_generated_artifacts"))
    )


def _state_fingerprint(sections: list[ContextSectionEstimate]) -> str:
    identity = [
        {
            "section": item["section"],
            "item_count": item["item_count"],
            "message_count": item["message_count"],
            "char_count": item["char_count"],
            "estimated_tokens": item["estimated_tokens"],
        }
        for item in sections
    ]
    return (
        "thread_context:v2:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )


def _unique_text(values: Sequence[str], *, limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _safe_text(value, 80)
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _value_list(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return list(value)


def _safe_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return max(value, 0)


def _safe_ratio(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    return round(max(float(value), 0.0), 6)


def _safe_text(value: Any, max_chars: int) -> str:
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


__all__ = [
    "THREAD_CONTEXT_WINDOW_CONTRACT",
    "THREAD_CONTEXT_WINDOW_SCHEMA_VERSION",
    "BackgroundInventory",
    "ContextSectionEstimate",
    "LastLLMCallUsage",
    "NextCallContextEstimate",
    "ThreadContextWindowV2",
    "build_thread_context_window_v2",
]
