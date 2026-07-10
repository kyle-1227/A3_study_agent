"""Safe LLM input manifests and thread-level context-window helpers.

This module is intentionally pure: it never calls providers, databases,
network APIs, subprocesses, or mutates global state.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, TypedDict

from langchain_core.messages import BaseMessage

from src.context_engineering.input_accounting import (
    LLMInputAccounting,
    build_llm_input_accounting,
)
from src.context_engineering.itemizer import sanitize_metadata
from src.context_engineering.influence import influence_status_payload
from src.context_engineering.schema import ContextItem
from src.context_engineering.workspace import (
    sanitize_workspace_text,
    utc_now_iso,
    workspace_status_payload,
)

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ID_PREFIX = "llm_input_manifest:v1"
MANIFEST_HISTORY_LIMIT = 80
BACKGROUND_CONTEXT_SCHEMA_VERSION = 1
THREAD_LEDGER_SCHEMA_VERSION = 1

_SECTION_LIMIT = 16
_SOURCE_ID_LIMIT = 20
_TEXT_MAX = 240


class LLMInputManifestError(RuntimeError):
    """Raised when an LLM provider call has no valid input manifest."""


class LLMInputManifestSection(TypedDict, total=False):
    section: str
    present: bool
    item_count: int
    message_count: int
    char_count: int
    estimated_tokens: int
    source_ids: list[str]
    metadata: dict[str, Any]


class LLMInputManifest(TypedDict, total=False):
    schema_version: int
    manifest_id: str
    created_at: str
    request_id: str
    thread_id: str
    node_name: str
    llm_node: str
    provider: str
    model: str
    call_purpose: str
    output_mode: str
    message_count: int
    input_estimated_tokens: int
    prompt_chars: int
    context_apply_applied: bool
    context_apply_status: str
    optional_sources_missing: list[str]
    provider_input_budget_tokens: int
    provider_input_tokens_before_context: int
    provider_remaining_input_tokens: int
    effective_context_budget_tokens: int
    schema_contract_first: bool
    provider_bound_messages_mutated: bool
    sections: list[LLMInputManifestSection]
    section_names: list[str]
    source_counts: dict[str, int]
    trace_call_id: str
    trace_seq: int
    diagnostics: list[str]
    message_fingerprint: str


class ThreadContextLedger(TypedDict, total=False):
    schema_version: int
    thread_id: str
    updated_at: str
    last_manifest_id: str
    manifest_count: int
    influence_entry_count: int
    influence_token_estimate: int
    influence_source_node_count: int
    node_count: int
    llm_node_count: int
    section_counts: dict[str, int]
    latest_node_manifest: dict[str, str]
    recent_manifest_ids: list[str]


class BackgroundContextWindow(TypedDict, total=False):
    schema_version: int
    thread_id: str
    updated_at: str
    last_manifest_id: str
    used_tokens: int
    max_context_tokens: int
    used_ratio: float
    message_count: int
    section_count: int
    section_names: list[str]
    workspace_present: bool
    workspace_active_subject: str
    workspace_evidence_summary_count: int
    workspace_gap_count: int
    workspace_artifact_count: int
    workspace_updated_at: str
    conversation_summary_present: bool
    selected_memory_count: int
    artifact_summary_count: int
    evidence_summary_count: int
    ce_block_present: bool
    structured_contract_present: bool
    manifest_count: int
    influence_entry_count: int
    influence_token_estimate: int
    influence_source_node_count: int


def build_llm_input_manifest(
    *,
    node_name: str,
    llm_node: str,
    provider: str,
    model: str,
    messages: list[Any],
    state: Mapping[str, Any] | None,
    call_purpose: str,
    output_mode: str = "",
    schema_name: str = "",
    schema_size_chars: int | None = None,
    context_apply_applied: bool = False,
    context_apply_status: str = "",
    optional_sources_missing: Iterable[str] = (),
    provider_input_budget_tokens: int = 0,
    provider_input_tokens_before_context: int = 0,
    provider_remaining_input_tokens: int = 0,
    effective_context_budget_tokens: int = 0,
    schema_contract_first: bool = False,
    provider_bound_messages_mutated: bool = False,
    trace_call_id: str = "",
    trace_seq: int = 0,
    accounting: LLMInputAccounting | None = None,
    context_items: Iterable[ContextItem] = (),
) -> LLMInputManifest:
    """Build a sanitized manifest for the exact provider-bound messages."""
    state_payload = state or {}
    safe_messages = messages or []
    input_accounting = accounting or build_llm_input_accounting(safe_messages)
    if input_accounting.message_count != len(safe_messages):
        raise LLMInputManifestError("llm_input_accounting_message_count_mismatch")
    safe_context_items = tuple(context_items)
    request_id = _safe_text(state_payload.get("request_id"), 120)
    thread_id = _safe_text(
        state_payload.get("thread_id") or state_payload.get("session_id"),
        120,
    )
    role_counts: dict[str, int] = {}
    for message in input_accounting.messages:
        role_counts[message.role] = role_counts.get(message.role, 0) + 1
    role_counts = dict(sorted(role_counts.items()))
    prompt_chars = input_accounting.prompt_chars
    input_tokens = input_accounting.input_estimated_tokens
    sections = _bounded_sections(
        [
            _message_section(role_counts, prompt_chars, input_tokens),
            _conversation_summary_section(state_payload),
            _task_workspace_section(state_payload),
            _evidence_section(state_payload),
            _artifact_section(state_payload),
            _memory_profile_section(state_payload),
            _context_influence_section(state_payload),
            _capability_context_section(input_accounting),
            _ce_block_section(
                input_accounting,
                context_apply_applied,
                context_items=safe_context_items,
            ),
            _structured_contract_section(
                schema_name=schema_name,
                schema_size_chars=schema_size_chars,
                schema_contract_first=schema_contract_first,
                output_mode=output_mode,
            ),
        ]
    )
    section_names = [section["section"] for section in sections]
    source_counts = {
        section["section"]: int(
            section.get("item_count") or section.get("message_count") or 0
        )
        for section in sections
    }
    safe_node_name = _safe_text(node_name, 120)
    safe_llm_node = _safe_text(llm_node, 120)
    safe_provider = _safe_text(provider, 120)
    safe_model = _safe_text(model, 160)
    safe_call_purpose = _safe_text(call_purpose, 120)
    safe_output_mode = _safe_text(output_mode, 120)
    safe_trace_call_id = _safe_text(trace_call_id, 120)
    safe_apply_status = _safe_text(context_apply_status, 80)
    safe_optional_sources = [
        _safe_text(source, 80)
        for source in optional_sources_missing
        if _safe_text(source, 80)
    ][:_SOURCE_ID_LIMIT]
    identity: dict[str, Any] = {
        "request_id": request_id,
        "thread_id": thread_id,
        "node_name": safe_node_name,
        "llm_node": safe_llm_node,
        "provider": safe_provider,
        "model": safe_model,
        "call_purpose": safe_call_purpose,
        "output_mode": safe_output_mode,
        "message_fingerprint": input_accounting.message_fingerprint,
        "sections": section_names,
        "trace_call_id": safe_trace_call_id,
        "trace_seq": int(trace_seq or 0),
    }
    manifest: LLMInputManifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": _stable_manifest_id(identity),
        "created_at": utc_now_iso(),
        "request_id": request_id,
        "thread_id": thread_id,
        "node_name": safe_node_name,
        "llm_node": safe_llm_node,
        "provider": safe_provider,
        "model": safe_model,
        "call_purpose": safe_call_purpose,
        "output_mode": safe_output_mode,
        "message_count": len(safe_messages),
        "input_estimated_tokens": input_tokens,
        "prompt_chars": prompt_chars,
        "context_apply_applied": bool(context_apply_applied),
        "context_apply_status": safe_apply_status,
        "optional_sources_missing": safe_optional_sources,
        "provider_input_budget_tokens": _safe_int(provider_input_budget_tokens),
        "provider_input_tokens_before_context": _safe_int(
            provider_input_tokens_before_context
        ),
        "provider_remaining_input_tokens": _safe_int(provider_remaining_input_tokens),
        "effective_context_budget_tokens": _safe_int(effective_context_budget_tokens),
        "schema_contract_first": bool(schema_contract_first),
        "provider_bound_messages_mutated": bool(provider_bound_messages_mutated),
        "sections": sections,
        "section_names": section_names,
        "source_counts": source_counts,
        "trace_call_id": safe_trace_call_id,
        "trace_seq": int(trace_seq or 0),
        "diagnostics": [],
        "message_fingerprint": input_accounting.message_fingerprint,
    }
    validate_llm_input_manifest(manifest)
    return manifest


def validate_llm_input_manifest(manifest: Mapping[str, Any] | None) -> None:
    """Fail fast when a provider call has no valid manifest."""
    if not isinstance(manifest, Mapping):
        raise LLMInputManifestError("llm_input_manifest_missing")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise LLMInputManifestError("llm_input_manifest_schema_version_invalid")
    for key in ("manifest_id", "node_name", "llm_node", "call_purpose"):
        if not _safe_text(manifest.get(key), 160):
            raise LLMInputManifestError(f"llm_input_manifest_{key}_missing")
    if not isinstance(manifest.get("sections"), list):
        raise LLMInputManifestError("llm_input_manifest_sections_invalid")


def llm_input_manifest_trace_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a compact A3_TRACE/SSE-safe manifest payload."""
    validate_llm_input_manifest(manifest)
    sections = [
        _safe_section(section)
        for section in manifest.get("sections", [])
        if isinstance(section, Mapping)
    ][:_SECTION_LIMIT]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": _safe_text(manifest.get("manifest_id"), 180),
        "created_at": _safe_text(manifest.get("created_at"), 80),
        "request_id": _safe_text(manifest.get("request_id"), 120),
        "thread_id": _safe_text(manifest.get("thread_id"), 120),
        "node_name": _safe_text(manifest.get("node_name"), 120),
        "llm_node": _safe_text(manifest.get("llm_node"), 120),
        "provider": _safe_text(manifest.get("provider"), 120),
        "model": _safe_text(manifest.get("model"), 160),
        "call_purpose": _safe_text(manifest.get("call_purpose"), 120),
        "output_mode": _safe_text(manifest.get("output_mode"), 120),
        "message_count": _safe_int(manifest.get("message_count")),
        "input_estimated_tokens": _safe_int(manifest.get("input_estimated_tokens")),
        "prompt_chars": _safe_int(manifest.get("prompt_chars")),
        "context_apply_applied": bool(manifest.get("context_apply_applied")),
        "context_apply_status": _safe_text(
            manifest.get("context_apply_status"),
            80,
        ),
        "optional_sources_missing": [
            _safe_text(item, 80)
            for item in (manifest.get("optional_sources_missing") or [])
            if _safe_text(item, 80)
        ][:_SOURCE_ID_LIMIT],
        "provider_input_budget_tokens": _safe_int(
            manifest.get("provider_input_budget_tokens")
        ),
        "provider_input_tokens_before_context": _safe_int(
            manifest.get("provider_input_tokens_before_context")
        ),
        "provider_remaining_input_tokens": _safe_int(
            manifest.get("provider_remaining_input_tokens")
        ),
        "effective_context_budget_tokens": _safe_int(
            manifest.get("effective_context_budget_tokens")
        ),
        "schema_contract_first": bool(manifest.get("schema_contract_first")),
        "provider_bound_messages_mutated": bool(
            manifest.get("provider_bound_messages_mutated")
        ),
        "section_names": [
            _safe_text(item, 80) for item in (manifest.get("section_names") or [])
        ][:_SECTION_LIMIT],
        "source_counts": _safe_int_dict(manifest.get("source_counts")),
        "sections": sections,
        "trace_call_id": _safe_text(manifest.get("trace_call_id"), 120),
        "trace_seq": _safe_int(manifest.get("trace_seq")),
        "diagnostics": [
            _safe_text(item, 120) for item in (manifest.get("diagnostics") or [])
        ][:8],
        "message_fingerprint": _safe_text(
            manifest.get("message_fingerprint"),
            80,
        ),
    }


def merge_llm_input_manifest_history(
    existing: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Idempotently merge manifest history by deterministic manifest_id."""
    merged: dict[str, dict[str, Any]] = {}
    for item in [*(existing or []), *(update or [])]:
        if not isinstance(item, dict):
            continue
        manifest_id = _safe_text(item.get("manifest_id"), 180)
        if not manifest_id:
            continue
        merged[manifest_id] = llm_input_manifest_trace_payload(item)
    ordered = sorted(
        merged.values(),
        key=lambda item: (
            _safe_text(item.get("created_at"), 80),
            _safe_text(item.get("manifest_id"), 180),
        ),
        reverse=True,
    )
    return ordered[:MANIFEST_HISTORY_LIMIT]


def build_thread_context_ledger_update(
    *,
    existing: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
) -> ThreadContextLedger:
    """Build an additive thread-level manifest ledger update."""
    payload = llm_input_manifest_trace_payload(manifest)
    prior = existing if isinstance(existing, Mapping) else {}
    section_counts = _safe_int_dict(prior.get("section_counts"))
    for section in payload.get("section_names") or []:
        section_counts[section] = section_counts.get(section, 0) + 1
    latest_node_manifest = _safe_str_dict(prior.get("latest_node_manifest"))
    node_name = _safe_text(payload.get("node_name"), 120)
    if node_name:
        latest_node_manifest[node_name] = payload["manifest_id"]
    recent_ids = [
        _safe_text(item, 180)
        for item in (prior.get("recent_manifest_ids") or [])
        if _safe_text(item, 180)
    ]
    manifest_id = payload["manifest_id"]
    recent_ids = [manifest_id, *[item for item in recent_ids if item != manifest_id]]
    return {
        "schema_version": THREAD_LEDGER_SCHEMA_VERSION,
        "thread_id": _safe_text(
            payload.get("thread_id") or prior.get("thread_id"),
            120,
        ),
        "updated_at": utc_now_iso(),
        "last_manifest_id": manifest_id,
        "manifest_count": len(recent_ids[:MANIFEST_HISTORY_LIMIT]),
        "node_count": len(latest_node_manifest),
        "llm_node_count": _safe_int(prior.get("llm_node_count")),
        "section_counts": section_counts,
        "latest_node_manifest": latest_node_manifest,
        "recent_manifest_ids": recent_ids[:MANIFEST_HISTORY_LIMIT],
    }


def merge_thread_context_ledger(existing: dict | None, update: dict | None) -> dict:
    if not isinstance(update, dict):
        return dict(existing or {})
    merged = dict(existing or {})
    merged.update(sanitize_metadata(update))
    return merged


def build_background_context_window(
    *,
    manifest: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None = None,
    manifest_count: int = 0,
    max_context_tokens: int = 0,
) -> BackgroundContextWindow:
    """Build a thread-level background context window from latest safe state."""
    state_payload = state or {}
    payload = llm_input_manifest_trace_payload(manifest or {})
    workspace = workspace_status_payload(
        state_payload.get("task_workspace")
        if isinstance(state_payload.get("task_workspace"), dict)
        else {}
    )
    manifest_workspace = _section_metadata(payload, "task_workspace")
    if not workspace.get("workspace_present") and manifest_workspace:
        workspace = {
            **workspace,
            "workspace_present": bool(manifest_workspace.get("workspace_present")),
            "workspace_active_subject": _safe_text(
                manifest_workspace.get("workspace_active_subject"),
                120,
            ),
            "workspace_evidence_summary_count": _safe_int(
                manifest_workspace.get("workspace_evidence_summary_count")
            ),
            "workspace_gap_count": _safe_int(
                manifest_workspace.get("workspace_gap_count")
            ),
            "workspace_artifact_count": _safe_int(
                manifest_workspace.get("workspace_artifact_count")
            ),
            "workspace_updated_at": _safe_text(
                manifest_workspace.get("workspace_updated_at"),
                80,
            ),
        }
    used_tokens = _safe_int(payload.get("input_estimated_tokens"))
    max_tokens = _safe_int(max_context_tokens)
    section_names = [
        _safe_text(item, 80) for item in (payload.get("section_names") or [])
    ][:_SECTION_LIMIT]
    influence = influence_status_payload(
        state_payload.get("context_influence_ledger")
        if isinstance(state_payload.get("context_influence_ledger"), Mapping)
        else {},
        include_recent_entries=False,
    )
    influence_tokens = sum(
        _safe_int(value)
        for value in (influence.get("token_estimates_by_kind") or {}).values()
    )
    return {
        "schema_version": BACKGROUND_CONTEXT_SCHEMA_VERSION,
        "thread_id": _safe_text(
            payload.get("thread_id")
            or state_payload.get("thread_id")
            or state_payload.get("session_id"),
            120,
        ),
        "updated_at": utc_now_iso(),
        "last_manifest_id": _safe_text(payload.get("manifest_id"), 180),
        "used_tokens": used_tokens,
        "max_context_tokens": max_tokens,
        "used_ratio": round(used_tokens / max_tokens, 4) if max_tokens > 0 else 0.0,
        "message_count": _safe_int(payload.get("message_count")),
        "section_count": len(section_names),
        "section_names": section_names,
        "conversation_summary_present": bool(
            str(state_payload.get("conversation_summary") or "").strip()
        ),
        "selected_memory_count": _safe_list_len(
            state_payload.get("selected_evidence_memory_summaries")
        )
        + _safe_list_len(state_payload.get("episodic_memory_results"))
        + _safe_list_len(state_payload.get("semantic_memory_results")),
        "artifact_summary_count": _artifact_count(state_payload),
        "evidence_summary_count": _evidence_count(state_payload),
        "ce_block_present": bool(payload.get("context_apply_applied")),
        "structured_contract_present": bool(payload.get("schema_contract_first")),
        "manifest_count": manifest_count,
        "influence_entry_count": _safe_int(influence.get("entry_count")),
        "influence_token_estimate": influence_tokens,
        "influence_source_node_count": len(influence.get("source_nodes") or []),
        **workspace,
    }


def background_context_status_payload(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "background_context_window_present": False,
            "background_context_window_used_tokens": 0,
            "background_context_window_max_tokens": 0,
            "background_context_window_used_ratio": 0.0,
            "background_context_window_updated_at": "",
            "llm_input_manifest_count": 0,
            "background_context_influence_entry_count": 0,
            "background_context_influence_token_estimate": 0,
        }
    return {
        "background_context_window_present": bool(value.get("last_manifest_id")),
        "background_context_window_used_tokens": _safe_int(value.get("used_tokens")),
        "background_context_window_max_tokens": _safe_int(
            value.get("max_context_tokens")
        ),
        "background_context_window_used_ratio": _safe_float(value.get("used_ratio")),
        "background_context_window_updated_at": _safe_text(
            value.get("updated_at"),
            80,
        ),
        "llm_input_manifest_count": _safe_int(value.get("manifest_count")),
        "background_context_influence_entry_count": _safe_int(
            value.get("influence_entry_count")
        ),
        "background_context_influence_token_estimate": _safe_int(
            value.get("influence_token_estimate")
        ),
    }


def _message_section(
    role_counts: dict[str, int],
    prompt_chars: int,
    estimated_tokens: int,
) -> LLMInputManifestSection:
    return {
        "section": "provider_bound_messages",
        "present": True,
        "message_count": sum(role_counts.values()),
        "char_count": prompt_chars,
        "estimated_tokens": estimated_tokens,
        "metadata": {"role_counts": role_counts},
    }


def _conversation_summary_section(state: Mapping[str, Any]) -> LLMInputManifestSection:
    text = str(state.get("conversation_summary") or "")
    return {
        "section": "conversation_summary",
        "present": bool(text.strip()),
        "item_count": 1 if text.strip() else 0,
        "char_count": len(text),
    }


def _task_workspace_section(state: Mapping[str, Any]) -> LLMInputManifestSection:
    status = workspace_status_payload(
        state.get("task_workspace")
        if isinstance(state.get("task_workspace"), dict)
        else {}
    )
    count = (
        int(status.get("workspace_evidence_summary_count") or 0)
        + int(status.get("workspace_gap_count") or 0)
        + int(status.get("workspace_artifact_count") or 0)
    )
    return {
        "section": "task_workspace",
        "present": bool(status.get("workspace_present")),
        "item_count": count,
        "metadata": status,
    }


def _evidence_section(state: Mapping[str, Any]) -> LLMInputManifestSection:
    count = _evidence_count(state)
    return {
        "section": "evidence_summaries",
        "present": count > 0,
        "item_count": count,
        "source_ids": _source_ids_from_items(
            _iter_dict_items(
                state.get("graded_evidence"),
                state.get("evidence_candidates"),
                (state.get("task_workspace") or {}).get("evidence_summaries")
                if isinstance(state.get("task_workspace"), dict)
                else [],
            ),
            id_keys=("evidence_id", "original_evidence_id"),
        ),
    }


def _artifact_section(state: Mapping[str, Any]) -> LLMInputManifestSection:
    count = _artifact_count(state)
    return {
        "section": "artifact_summaries",
        "present": count > 0,
        "item_count": count,
        "source_ids": _source_ids_from_items(
            _iter_dict_items(
                state.get("last_generated_artifacts"),
                state.get("resource_artifacts_by_type"),
                (state.get("task_workspace") or {}).get("artifacts")
                if isinstance(state.get("task_workspace"), dict)
                else [],
            ),
            id_keys=("artifact_id", "resource_type"),
        ),
    }


def _memory_profile_section(state: Mapping[str, Any]) -> LLMInputManifestSection:
    count = (
        _safe_list_len(state.get("selected_evidence_memory_summaries"))
        + _safe_list_len(state.get("episodic_memory_results"))
        + _safe_list_len(state.get("semantic_memory_results"))
        + (1 if state.get("profile") else 0)
    )
    return {
        "section": "selected_memory_profile",
        "present": count > 0,
        "item_count": count,
    }


def _context_influence_section(state: Mapping[str, Any]) -> LLMInputManifestSection:
    status = influence_status_payload(
        state.get("context_influence_ledger")
        if isinstance(state.get("context_influence_ledger"), Mapping)
        else {},
        include_recent_entries=False,
    )
    count = _safe_int(status.get("entry_count"))
    token_estimate = sum(
        _safe_int(value)
        for value in (status.get("token_estimates_by_kind") or {}).values()
    )
    return {
        "section": "context_influence_ledger",
        "present": count > 0,
        "item_count": count,
        "estimated_tokens": token_estimate,
        "source_ids": [
            _safe_text(item, 120) for item in (status.get("source_nodes") or [])
        ][:_SOURCE_ID_LIMIT],
        "metadata": {
            "counts_by_kind": status.get("counts_by_kind") or {},
            "source_node_count": len(status.get("source_nodes") or []),
        },
    }


def _ce_block_section(
    accounting: LLMInputAccounting,
    context_apply_applied: bool,
    *,
    context_items: tuple[ContextItem, ...],
) -> LLMInputManifestSection:
    ce_messages = [
        message for message in accounting.messages if message.contains_injected_context
    ]
    ce_chars = sum(message.char_count for message in ce_messages)
    source_ids = [
        _safe_text(item.id, 180) for item in context_items if _safe_text(item.id, 180)
    ][:_SOURCE_ID_LIMIT]
    return {
        "section": "ce_block",
        "present": bool(context_apply_applied or ce_chars),
        "item_count": 1 if context_apply_applied or ce_chars else 0,
        "char_count": ce_chars,
        "estimated_tokens": sum(message.estimated_tokens for message in ce_messages),
        "source_ids": source_ids,
    }


def _capability_context_section(
    accounting: LLMInputAccounting,
) -> LLMInputManifestSection:
    capability_messages = tuple(
        message
        for message in accounting.messages
        if message.contains_capability_context
    )
    capability_chars = sum(message.char_count for message in capability_messages)
    return {
        "section": "capability_context",
        "present": capability_chars > 0,
        "item_count": 1 if capability_chars > 0 else 0,
        "char_count": capability_chars,
        "estimated_tokens": sum(
            message.estimated_tokens for message in capability_messages
        ),
    }


def _structured_contract_section(
    *,
    schema_name: str,
    schema_size_chars: int | None,
    schema_contract_first: bool,
    output_mode: str,
) -> LLMInputManifestSection:
    present = bool(schema_name or schema_size_chars or schema_contract_first)
    return {
        "section": "structured_output_contract",
        "present": present,
        "item_count": 1 if present else 0,
        "char_count": _safe_int(schema_size_chars),
        "metadata": {
            "schema_name": _safe_text(schema_name, 120),
            "output_mode": _safe_text(output_mode, 120),
            "schema_contract_first": bool(schema_contract_first),
        },
    }


def _bounded_sections(
    sections: Iterable[LLMInputManifestSection],
) -> list[LLMInputManifestSection]:
    result: list[LLMInputManifestSection] = []
    for section in sections:
        safe = _safe_section(section)
        if safe.get("present"):
            result.append(safe)
    return result[:_SECTION_LIMIT]


def _safe_section(section: Mapping[str, Any]) -> LLMInputManifestSection:
    return {
        "section": _safe_text(section.get("section"), 80),
        "present": bool(section.get("present")),
        "item_count": _safe_int(section.get("item_count")),
        "message_count": _safe_int(section.get("message_count")),
        "char_count": _safe_int(section.get("char_count")),
        "estimated_tokens": _safe_int(section.get("estimated_tokens")),
        "source_ids": [
            _safe_text(item, 180) for item in (section.get("source_ids") or [])
        ][:_SOURCE_ID_LIMIT],
        "metadata": sanitize_metadata(
            section.get("metadata") if isinstance(section.get("metadata"), dict) else {}
        ),
    }


def _message_role_counts_and_chars(messages: list[Any]) -> tuple[dict[str, int], int]:
    role_counts: dict[str, int] = {}
    total_chars = 0
    for message in messages:
        role = _message_role(message)
        role_counts[role] = role_counts.get(role, 0) + 1
        total_chars += len(_message_content(message))
    return dict(sorted(role_counts.items())), total_chars


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return _safe_text(message.get("role") or message.get("type") or "unknown", 40)
    if isinstance(message, BaseMessage):
        return _safe_text(getattr(message, "type", "unknown"), 40)
    return "unknown"


def _message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        return _content_text(message.get("content"))
    if isinstance(message, BaseMessage):
        return _content_text(getattr(message, "content", ""))
    return str(message or "")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_fingerprint(messages: list[Any]) -> str:
    payload = [
        {
            "role": _message_role(message),
            "chars": len(_message_content(message)),
            "digest": hashlib.sha256(
                _message_content(message).encode("utf-8")
            ).hexdigest()[:16],
        }
        for message in messages
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]


def _stable_manifest_id(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]
    return f"{MANIFEST_ID_PREFIX}:{digest}"


def _evidence_count(state: Mapping[str, Any]) -> int:
    workspace_raw = state.get("task_workspace")
    workspace: Mapping[str, Any] = (
        workspace_raw if isinstance(workspace_raw, Mapping) else {}
    )
    return (
        _safe_list_len(state.get("graded_evidence"))
        + _safe_list_len(state.get("evidence_candidates"))
        + _safe_list_len(workspace.get("evidence_summaries"))
    )


def _artifact_count(state: Mapping[str, Any]) -> int:
    workspace_raw = state.get("task_workspace")
    workspace: Mapping[str, Any] = (
        workspace_raw if isinstance(workspace_raw, Mapping) else {}
    )
    workspace_artifacts = workspace.get("artifacts_by_id")
    resource_artifacts = state.get("resource_artifacts_by_type")
    resource_artifact_count = (
        len(resource_artifacts) if isinstance(resource_artifacts, Mapping) else 0
    )
    return (
        _safe_list_len(state.get("last_generated_artifacts"))
        + resource_artifact_count
        + (
            len(workspace_artifacts)
            if isinstance(workspace_artifacts, Mapping)
            else _safe_list_len(workspace.get("artifacts"))
        )
    )


def _iter_dict_items(*values: Any) -> Iterable[Mapping[str, Any]]:
    for value in values:
        if isinstance(value, Mapping):
            iterable: Iterable[Any] = value.values()
        elif isinstance(value, list):
            iterable = value
        else:
            continue
        for item in iterable:
            if isinstance(item, Mapping):
                yield item


def _source_ids_from_items(
    items: Iterable[Mapping[str, Any]],
    *,
    id_keys: tuple[str, ...],
) -> list[str]:
    seen: list[str] = []
    for item in items:
        for key in id_keys:
            text = _safe_text(item.get(key), 180)
            if text and text not in seen:
                seen.append(text)
                break
        if len(seen) >= _SOURCE_ID_LIMIT:
            break
    return seen


def _safe_text(value: Any, max_chars: int = _TEXT_MAX) -> str:
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        safe_key = _safe_text(key, 80)
        if safe_key:
            result[safe_key] = _safe_int(item)
    return dict(sorted(result.items()))


def _safe_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        safe_key = _safe_text(key, 120)
        safe_item = _safe_text(item, 180)
        if safe_key and safe_item:
            result[safe_key] = safe_item
    return dict(sorted(result.items()))


def _safe_list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _section_metadata(
    manifest_payload: Mapping[str, Any],
    section_name: str,
) -> dict[str, Any]:
    for section in manifest_payload.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        if section.get("section") != section_name:
            continue
        metadata = section.get("metadata")
        return dict(metadata) if isinstance(metadata, Mapping) else {}
    return {}
