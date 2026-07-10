"""Bounded, sanitized Context Influence Ledger helpers.

The helpers in this module are pure. They do not call providers, databases,
network APIs, subprocesses, or mutate global state.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, TypedDict

from src.context_engineering.itemizer import sanitize_metadata
from src.context_engineering.tokenizer import estimate_text_tokens_mixed
from src.context_engineering.workspace import sanitize_workspace_text

INFLUENCE_LEDGER_SCHEMA_VERSION = 1
INFLUENCE_ID_PREFIX = "influence:v1"
INFLUENCE_ENTRY_LIMIT = 160
INFLUENCE_PREVIEW_MAX_CHARS = 600
INFLUENCE_LEDGER_TEXT_BUDGET = 64_000
INFLUENCE_STATUS_ENTRY_LIMIT = 40

CONTEXT_INFLUENCE_LEDGER_CLEAR = {"__context_influence_ledger_clear__": True}

INFLUENCE_KINDS = frozenset(
    {
        "original_user_query",
        "query_rewrite",
        "retrieval_plan",
        "prompt_snapshot_metadata",
        "provider_bound_messages_metadata",
        "local_evidence",
        "web_evidence",
        "evidence_judge",
        "coverage_gap",
        "profile_completion",
        "learner_profile",
        "planner_output",
        "agent_output",
        "reviewer_output",
        "consensus_output",
        "workspace",
        "artifact",
        "schema_contract",
    }
)

_NON_INJECTABLE_KINDS = frozenset(
    {
        "original_user_query",
        "prompt_snapshot_metadata",
        "provider_bound_messages_metadata",
        "local_evidence",
        "web_evidence",
        "evidence_judge",
        "schema_contract",
    }
)


class ContextInfluenceEntry(TypedDict, total=False):
    influence_id: str
    sequence: int
    request_id: str
    thread_id: str
    kind: str
    source_node: str
    target_stage: str
    title: str
    preview: str
    content_fingerprint: str
    token_estimate: int
    priority: int
    injectable: bool
    created_at: str
    metadata: dict[str, Any]


class ContextInfluenceLedger(TypedDict, total=False):
    schema_version: int
    thread_id: str
    updated_at: str
    next_sequence: int
    entries_by_id: dict[str, ContextInfluenceEntry]
    ordered_ids: list[str]
    total_recorded: int
    counts_by_kind: dict[str, int]
    token_estimates_by_kind: dict[str, int]
    source_nodes: list[str]
    diagnostics: list[str]


class ContextInfluenceLedgerUpdate(TypedDict, total=False):
    schema_version: int
    thread_id: str
    entries: list[ContextInfluenceEntry]
    diagnostics: list[str]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def stable_influence_id(
    *,
    request_id: str,
    thread_id: str,
    kind: str,
    source_node: str,
    target_stage: str,
    content_fingerprint: str,
    identity_metadata: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "request_id": _safe_text(request_id, 120),
        "thread_id": _safe_text(thread_id, 120),
        "kind": _safe_text(kind, 80),
        "source_node": _safe_text(source_node, 120),
        "target_stage": _safe_text(target_stage, 80),
        "content_fingerprint": _safe_text(content_fingerprint, 80),
        "identity_metadata": sanitize_metadata(dict(identity_metadata or {})),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]
    return f"{INFLUENCE_ID_PREFIX}:{digest}"


def content_fingerprint(value: object) -> str:
    text = str(value or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def influence_kind_is_injectable(kind: object) -> bool:
    """Return whether a valid kind may enter the pipeline provider."""
    normalized = str(kind or "").strip()
    return normalized in INFLUENCE_KINDS and normalized not in _NON_INJECTABLE_KINDS


def build_influence_entry(
    *,
    state: Mapping[str, Any] | None,
    kind: str,
    source_node: str,
    target_stage: str = "downstream",
    title: str = "",
    preview: object = "",
    metadata: Mapping[str, Any] | None = None,
    priority: int = 50,
    injectable: bool | None = None,
    fingerprint_source: object | None = None,
) -> ContextInfluenceEntry:
    """Build one safe influence entry with deterministic identity."""
    if kind not in INFLUENCE_KINDS:
        raise ValueError(f"unsupported context influence kind: {kind}")
    state_payload = state or {}
    request_id = _safe_text(state_payload.get("request_id"), 120)
    thread_id = _safe_text(
        state_payload.get("thread_id") or state_payload.get("session_id"),
        120,
    )
    safe_preview = _safe_text(preview, INFLUENCE_PREVIEW_MAX_CHARS)
    fingerprint = content_fingerprint(
        fingerprint_source if fingerprint_source is not None else safe_preview
    )
    safe_metadata = sanitize_metadata(dict(metadata or {}))
    safe_priority = max(0, min(int(priority), 100))
    is_injectable = kind not in _NON_INJECTABLE_KINDS
    if injectable is not None:
        is_injectable = bool(injectable)
    influence_id = stable_influence_id(
        request_id=request_id,
        thread_id=thread_id,
        kind=kind,
        source_node=source_node,
        target_stage=target_stage,
        content_fingerprint=fingerprint,
        identity_metadata=_identity_metadata(safe_metadata),
    )
    return {
        "influence_id": influence_id,
        "sequence": 0,
        "request_id": request_id,
        "thread_id": thread_id,
        "kind": kind,
        "source_node": _safe_text(source_node, 120),
        "target_stage": _safe_text(target_stage, 80),
        "title": _safe_text(title or kind, 160),
        "preview": safe_preview,
        "content_fingerprint": fingerprint,
        "token_estimate": estimate_text_tokens_mixed(safe_preview),
        "priority": safe_priority,
        "injectable": is_injectable,
        "created_at": utc_now_iso(),
        "metadata": safe_metadata,
    }


def build_influence_update(
    *,
    state: Mapping[str, Any] | None,
    entries: Iterable[Mapping[str, Any]],
    diagnostics: Iterable[str] = (),
) -> ContextInfluenceLedgerUpdate:
    state_payload = state or {}
    safe_entries = [
        _coerce_entry(entry)
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("influence_id")
    ]
    return {
        "schema_version": INFLUENCE_LEDGER_SCHEMA_VERSION,
        "thread_id": _safe_text(
            state_payload.get("thread_id") or state_payload.get("session_id"),
            120,
        ),
        "entries": [entry for entry in safe_entries if entry],
        "diagnostics": _bounded_strings(diagnostics, limit=10, max_chars=120),
    }


def merge_context_influence_ledger(
    existing: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> ContextInfluenceLedger:
    """Merge stable entries, assign monotonic sequences, and enforce bounds."""
    if (
        isinstance(update, Mapping)
        and update.get("__context_influence_ledger_clear__") is True
    ):
        return _empty_ledger()

    current = _coerce_ledger(existing)
    diagnostics = list(current.get("diagnostics") or [])
    if not isinstance(update, Mapping):
        return current
    version = update.get("schema_version")
    if version not in (None, INFLUENCE_LEDGER_SCHEMA_VERSION):
        diagnostics.append("context_influence_schema_version_incompatible")
        current["diagnostics"] = _bounded_strings(diagnostics, limit=10, max_chars=120)
        return current
    if version is None and update:
        diagnostics.append("context_influence_schema_version_missing")

    entries_by_id = dict(current.get("entries_by_id") or {})
    next_sequence = max(1, int(current.get("next_sequence") or 1))
    incoming = _incoming_entries(update)
    for entry in sorted(
        incoming,
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("influence_id") or ""),
        ),
    ):
        influence_id = str(entry.get("influence_id") or "")
        if not influence_id:
            continue
        prior = entries_by_id.get(influence_id)
        if prior:
            entry["sequence"] = int(prior.get("sequence") or 0)
            entry["created_at"] = str(
                prior.get("created_at") or entry.get("created_at") or utc_now_iso()
            )
        else:
            entry["sequence"] = next_sequence
            next_sequence += 1
        entries_by_id[influence_id] = entry

    total_recorded = max(
        max(0, int(current.get("total_recorded") or 0)),
        len(entries_by_id),
    )
    retained = _bounded_entries(entries_by_id.values())
    entries_by_id = {entry["influence_id"]: entry for entry in retained}
    ordered_ids = [entry["influence_id"] for entry in retained]
    counts, tokens = _aggregate_by_kind(retained)
    thread_id = _safe_text(update.get("thread_id") or current.get("thread_id"), 120)
    source_nodes = sorted(
        {
            _safe_text(entry.get("source_node"), 120)
            for entry in retained
            if _safe_text(entry.get("source_node"), 120)
        }
    )[:80]
    diagnostics.extend(
        str(item) for item in (update.get("diagnostics") or []) if str(item).strip()
    )
    if len(entries_by_id) < len(_merge_entry_maps(current, incoming)):
        diagnostics.append("context_influence_ledger_trimmed")
    return {
        "schema_version": INFLUENCE_LEDGER_SCHEMA_VERSION,
        "thread_id": thread_id,
        "updated_at": utc_now_iso(),
        "next_sequence": next_sequence,
        "entries_by_id": entries_by_id,
        "ordered_ids": ordered_ids,
        "total_recorded": total_recorded,
        "counts_by_kind": counts,
        "token_estimates_by_kind": tokens,
        "source_nodes": source_nodes,
        "diagnostics": _bounded_strings(diagnostics, limit=10, max_chars=120),
    }


def influence_status_payload(
    value: Mapping[str, Any] | None,
    *,
    include_recent_entries: bool = True,
) -> dict[str, Any]:
    ledger = _coerce_ledger(value)
    entries_by_id = ledger.get("entries_by_id") or {}
    ordered_ids = ledger.get("ordered_ids") or []
    recent: list[dict[str, Any]] = []
    if include_recent_entries:
        for influence_id in reversed(ordered_ids[-INFLUENCE_STATUS_ENTRY_LIMIT:]):
            entry = entries_by_id.get(influence_id)
            if isinstance(entry, Mapping):
                recent.append(_trace_entry(entry))
    return {
        "schema_version": INFLUENCE_LEDGER_SCHEMA_VERSION,
        "present": bool(entries_by_id),
        "thread_id": _safe_text(ledger.get("thread_id"), 120),
        "updated_at": _safe_text(ledger.get("updated_at"), 80),
        "entry_count": len(entries_by_id),
        "total_recorded": max(0, int(ledger.get("total_recorded") or 0)),
        "counts_by_kind": _safe_int_dict(ledger.get("counts_by_kind")),
        "token_estimates_by_kind": _safe_int_dict(
            ledger.get("token_estimates_by_kind")
        ),
        "source_nodes": [
            _safe_text(item, 120) for item in (ledger.get("source_nodes") or [])
        ][:80],
        "recent_entries": recent,
        "diagnostics": _bounded_strings(
            ledger.get("diagnostics") or [], limit=10, max_chars=120
        ),
    }


def _empty_ledger() -> ContextInfluenceLedger:
    return {
        "schema_version": INFLUENCE_LEDGER_SCHEMA_VERSION,
        "thread_id": "",
        "updated_at": "",
        "next_sequence": 1,
        "entries_by_id": {},
        "ordered_ids": [],
        "total_recorded": 0,
        "counts_by_kind": {},
        "token_estimates_by_kind": {},
        "source_nodes": [],
        "diagnostics": [],
    }


def _coerce_ledger(value: Mapping[str, Any] | None) -> ContextInfluenceLedger:
    if not isinstance(value, Mapping) or not value:
        return _empty_ledger()
    version = value.get("schema_version")
    if version not in (None, INFLUENCE_LEDGER_SCHEMA_VERSION):
        ledger = _empty_ledger()
        ledger["diagnostics"] = ["context_influence_schema_version_incompatible"]
        return ledger
    entries = _incoming_entries(value)
    entries_by_id = {
        entry["influence_id"]: entry for entry in entries if entry.get("influence_id")
    }
    retained = _bounded_entries(entries_by_id.values())
    if any(int(entry.get("sequence") or 0) <= 0 for entry in retained):
        for sequence, entry in enumerate(retained, start=1):
            entry["sequence"] = sequence
    retained.sort(key=lambda item: int(item.get("sequence") or 0))
    counts, tokens = _aggregate_by_kind(retained)
    highest = max((int(item.get("sequence") or 0) for item in retained), default=0)
    return {
        "schema_version": INFLUENCE_LEDGER_SCHEMA_VERSION,
        "thread_id": _safe_text(value.get("thread_id"), 120),
        "updated_at": _safe_text(value.get("updated_at"), 80),
        "next_sequence": max(highest + 1, int(value.get("next_sequence") or 1)),
        "entries_by_id": {item["influence_id"]: item for item in retained},
        "ordered_ids": [item["influence_id"] for item in retained],
        "total_recorded": max(
            len(retained), int(value.get("total_recorded") or len(retained))
        ),
        "counts_by_kind": counts,
        "token_estimates_by_kind": tokens,
        "source_nodes": sorted(
            {
                str(item.get("source_node") or "")
                for item in retained
                if str(item.get("source_node") or "")
            }
        )[:80],
        "diagnostics": _bounded_strings(
            value.get("diagnostics") or [], limit=10, max_chars=120
        ),
    }


def _incoming_entries(value: Mapping[str, Any]) -> list[ContextInfluenceEntry]:
    raw_entries: list[Any] = []
    entries = value.get("entries")
    if isinstance(entries, list):
        raw_entries.extend(entries)
    entries_by_id = value.get("entries_by_id")
    if isinstance(entries_by_id, Mapping):
        raw_entries.extend(entries_by_id.values())
    result: list[ContextInfluenceEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            continue
        entry = _coerce_entry(raw)
        if entry:
            result.append(entry)
    return result


def _coerce_entry(value: Mapping[str, Any]) -> ContextInfluenceEntry:
    influence_id = _safe_text(value.get("influence_id"), 180)
    kind = _safe_text(value.get("kind"), 80)
    if (
        not influence_id.startswith(f"{INFLUENCE_ID_PREFIX}:")
        or kind not in INFLUENCE_KINDS
    ):
        return {}
    preview = _safe_text(value.get("preview"), INFLUENCE_PREVIEW_MAX_CHARS)
    token_estimate = value.get("token_estimate")
    if isinstance(token_estimate, bool) or not isinstance(token_estimate, int):
        token_estimate = estimate_text_tokens_mixed(preview)
    priority = value.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int):
        priority = 50
    created_at = _safe_iso(value.get("created_at")) or utc_now_iso()
    return {
        "influence_id": influence_id,
        "sequence": max(0, int(value.get("sequence") or 0)),
        "request_id": _safe_text(value.get("request_id"), 120),
        "thread_id": _safe_text(value.get("thread_id"), 120),
        "kind": kind,
        "source_node": _safe_text(value.get("source_node"), 120),
        "target_stage": _safe_text(value.get("target_stage"), 80),
        "title": _safe_text(value.get("title"), 160),
        "preview": preview,
        "content_fingerprint": _safe_text(
            value.get("content_fingerprint") or content_fingerprint(preview), 80
        ),
        "token_estimate": max(0, token_estimate),
        "priority": max(0, min(priority, 100)),
        "injectable": bool(value.get("injectable")),
        "created_at": created_at,
        "metadata": sanitize_metadata(
            dict(value.get("metadata") or {})
            if isinstance(value.get("metadata"), Mapping)
            else {}
        ),
    }


def _bounded_entries(
    entries: Iterable[Mapping[str, Any]],
) -> list[ContextInfluenceEntry]:
    ordered = sorted(
        (_coerce_entry(entry) for entry in entries if isinstance(entry, Mapping)),
        key=lambda item: (
            str(item.get("created_at") or ""),
            int(item.get("priority") or 0),
            str(item.get("influence_id") or ""),
        ),
        reverse=True,
    )
    retained: list[ContextInfluenceEntry] = []
    chars = 0
    for entry in ordered:
        if not entry:
            continue
        entry_chars = _json_chars(entry)
        if len(retained) >= INFLUENCE_ENTRY_LIMIT:
            break
        if retained and chars + entry_chars > INFLUENCE_LEDGER_TEXT_BUDGET:
            continue
        retained.append(entry)
        chars += entry_chars
    retained.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("influence_id") or ""),
        )
    )
    return retained


def _aggregate_by_kind(
    entries: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    tokens: dict[str, int] = {}
    for entry in entries:
        kind = str(entry.get("kind") or "")
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        tokens[kind] = tokens.get(kind, 0) + max(
            0, int(entry.get("token_estimate") or 0)
        )
    return dict(sorted(counts.items())), dict(sorted(tokens.items()))


def _trace_entry(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "influence_id": _safe_text(value.get("influence_id"), 180),
        "sequence": max(0, int(value.get("sequence") or 0)),
        "request_id": _safe_text(value.get("request_id"), 120),
        "kind": _safe_text(value.get("kind"), 80),
        "source_node": _safe_text(value.get("source_node"), 120),
        "target_stage": _safe_text(value.get("target_stage"), 80),
        "title": _safe_text(value.get("title"), 160),
        "preview": _safe_text(value.get("preview"), INFLUENCE_PREVIEW_MAX_CHARS),
        "token_estimate": max(0, int(value.get("token_estimate") or 0)),
        "injectable": bool(value.get("injectable")),
        "created_at": _safe_text(value.get("created_at"), 80),
        "metadata": sanitize_metadata(
            dict(value.get("metadata") or {})
            if isinstance(value.get("metadata"), Mapping)
            else {}
        ),
    }


def _identity_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "artifact_id",
        "evidence_id",
        "gap_id",
        "manifest_id",
        "resource_type",
        "workflow",
        "iteration",
        "schema_name",
        "output_mode",
    }
    return {key: value[key] for key in sorted(allowed & set(value))}


def _merge_entry_maps(
    existing: Mapping[str, Any], incoming: Iterable[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    current = existing.get("entries_by_id")
    if isinstance(current, Mapping):
        for key, item in current.items():
            if isinstance(item, Mapping):
                result[str(key)] = item
    for item in incoming:
        influence_id = str(item.get("influence_id") or "")
        if influence_id:
            result[influence_id] = item
    return result


def _safe_text(value: object, max_chars: int) -> str:
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _safe_iso(value: object) -> str:
    text = _safe_text(value, 80)
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        return ""
    return parsed.astimezone(timezone.utc).isoformat()


def _safe_int_dict(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            continue
        safe_key = _safe_text(key, 80)
        if safe_key:
            result[safe_key] = item
    return dict(sorted(result.items()))


def _bounded_strings(
    values: Iterable[object], *, limit: int, max_chars: int
) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _safe_text(value, max_chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _json_chars(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
