"""Versioned task workspace helpers for Context Engineering.

This module is intentionally pure: it does not call LLMs, databases, network
APIs, subprocesses, or mutate global state.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Literal, Mapping, TypedDict
from urllib.parse import urlsplit, urlunsplit

from src.context_engineering.itemizer import sanitize_metadata
from src.context_engineering.schema import sanitize_error_message
from src.rag.course_catalog import normalize_subject

WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_ID_PREFIX = "workspace:v1"
ARTIFACT_ID_PREFIX = "artifact:v1"
EVIDENCE_ID_PREFIX = "evidence:v1"
GAP_ID_PREFIX = "gap:v1"
CONSTRAINT_ID_PREFIX = "constraint:v1"
PROFILE_REQUIREMENT_ID_PREFIX = "profile_requirement:v1"

WORKSPACE_EVIDENCE_LIMIT = 20
WORKSPACE_GAP_LIMIT = 20
WORKSPACE_ARTIFACT_LIMIT = 20
WORKSPACE_CONSTRAINT_LIMIT = 12
WORKSPACE_PROFILE_REQUIREMENT_LIMIT = 12
WORKSPACE_EVENT_LIMIT = 120

WORKSPACE_TEXT_BUDGET = 18_000
WORKSPACE_ARTIFACT_TEXT_BUDGET = 18_000
WORKSPACE_EVIDENCE_TEXT_BUDGET = 14_000
WORKSPACE_GAP_TEXT_BUDGET = 8_000
WORKSPACE_CONSTRAINT_TEXT_BUDGET = 6_000
WORKSPACE_PROFILE_REQUIREMENT_TEXT_BUDGET = 6_000

TASK_WORKSPACE_CLEAR = {"__task_workspace_clear__": True}

_SAFE_REF_KEYS = {
    "filename",
    "markdown_filename",
    "docx_filename",
    "pdf_filename",
    "html_filename",
    "srt_filename",
    "mp4_filename",
    "xmind_filename",
    "json_filename",
    "source_filename",
    "url",
    "markdown_url",
    "docx_url",
    "pdf_url",
    "html_url",
    "srt_url",
    "mp4_url",
    "xmind_url",
    "json_url",
    "source_url",
    "python_url",
    "artifact_url",
    "artifact_path",
    "relative_path",
    "path",
}
_UNSAFE_URL_HINTS = {
    "signature",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "token",
    "access_token",
    "auth",
    "authorization",
    "api_key",
    "apikey",
    "key",
    "password",
    "secret",
    "cookie",
}
_RAW_CONTENT_KEYS = {
    "content",
    "markdown",
    "html",
    "srt",
    "tree",
    "items",
    "documents",
    "pages",
    "raw",
    "raw_output",
    "raw_messages",
    "message_content",
}
_GOAL_TOKEN_RE = re.compile(r"[0-9a-zA-Z_\u4e00-\u9fff]+")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class WorkspaceEvidenceSummary(TypedDict, total=False):
    evidence_id: str
    original_evidence_id: str
    title: str
    summary: str
    subject: str
    normalized_subject: str
    source_type: str
    support_score: float
    relevance_score: float
    purpose: Literal["factual_grounding"]
    created_at: str
    request_id: str
    thread_id: str
    usable_for: list[str]


class WorkspaceCoverageGap(TypedDict, total=False):
    gap_id: str
    subject: str
    normalized_subject: str
    role: str
    gap: str
    suggested_search_query: str
    purpose: str
    priority: float
    created_at: str
    request_id: str
    thread_id: str


class WorkspaceArtifactSummary(TypedDict, total=False):
    artifact_id: str
    resource_type: str
    title: str
    summary: str
    message_preview: str
    subject: str
    normalized_subject: str
    active_learning_goal: str
    active_learning_goal_present: bool
    normalized_learning_goal: str
    purpose: Literal["artifact_reference"]
    thread_id: str
    request_id: str
    created_at: str
    metrics: dict[str, Any]
    artifact_refs: dict[str, str]


class WorkspaceConstraint(TypedDict, total=False):
    constraint_id: str
    field: str
    label: str
    value_preview: str
    purpose: Literal["profile_completion"]
    created_at: str
    request_id: str
    thread_id: str


class WorkspaceProfileRequirement(TypedDict, total=False):
    requirement_id: str
    field: str
    label: str
    value_preview: str
    purpose: Literal["profile_completion"]
    created_at: str
    request_id: str
    thread_id: str


class WorkspaceUpdate(TypedDict, total=False):
    schema_version: int
    workspace_id: str
    scope: dict[str, str]
    thread_id: str
    request_id: str
    active_subject: str
    normalized_subject: str
    active_learning_goal: str
    normalized_learning_goal: str
    evidence_state: str
    updated_at: str
    updated_sources: list[str]
    evidence_summaries: list[WorkspaceEvidenceSummary]
    coverage_gaps: list[WorkspaceCoverageGap]
    artifacts_by_id: dict[str, WorkspaceArtifactSummary]
    latest_artifact_by_resource_type: dict[str, str]
    artifacts: list[WorkspaceArtifactSummary]
    constraints: list[WorkspaceConstraint]
    profile_requirements: list[WorkspaceProfileRequirement]
    diagnostics: list[str]


class WorkspaceTracePayload(TypedDict, total=False):
    workspace_id: str
    thread_id: str
    request_id: str
    active_subject: str
    active_learning_goal_present: bool
    evidence_summary_count: int
    coverage_gap_count: int
    artifact_count: int
    constraint_count: int
    updated_sources: list[str]
    rotation_action: str
    diagnostics: list[str]


class WorkspaceContinuationContext(TypedDict, total=False):
    can_continue: bool
    continuation_applied: bool
    skip_reason: str
    workspace_id: str
    thread_id: str
    current_thread_id: str
    workspace_thread_id: str
    request_id: str
    active_subject: str
    normalized_subject: str
    active_learning_goal: str
    normalized_learning_goal: str
    resource_types: list[str]
    diagnostics: list[str]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def stable_workspace_id(
    *,
    thread_id: str,
    normalized_subject: str,
    normalized_learning_goal: str,
) -> str:
    return _stable_id(
        WORKSPACE_ID_PREFIX,
        {
            "thread_id": thread_id,
            "normalized_subject": normalized_subject,
            "normalized_learning_goal": normalized_learning_goal,
        },
    )


def stable_artifact_id(
    *,
    resource_type: str,
    title: str,
    thread_id: str,
    request_id: str,
    normalized_subject: str,
    artifact_refs: dict[str, str],
) -> str:
    return _stable_id(
        ARTIFACT_ID_PREFIX,
        {
            "resource_type": resource_type,
            "title": title,
            "thread_id": thread_id,
            "request_id": request_id,
            "normalized_subject": normalized_subject,
            "artifact_refs": artifact_refs,
        },
    )


def stable_evidence_id(
    *,
    original_evidence_id: str,
    thread_id: str,
    normalized_subject: str,
    request_id: str,
) -> str:
    return _stable_id(
        EVIDENCE_ID_PREFIX,
        {
            "original_evidence_id": original_evidence_id,
            "thread_id": thread_id,
            "normalized_subject": normalized_subject,
            "request_id": request_id,
        },
    )


def stable_gap_id(
    *,
    gap: str,
    subject: str,
    role: str,
    thread_id: str,
) -> str:
    return _stable_id(
        GAP_ID_PREFIX,
        {
            "gap": gap,
            "subject": subject,
            "role": role,
            "thread_id": thread_id,
        },
    )


def stable_constraint_id(
    *,
    field: str,
    value_preview: str,
    thread_id: str,
    request_id: str,
) -> str:
    return _stable_id(
        CONSTRAINT_ID_PREFIX,
        {
            "field": field,
            "value_preview": value_preview,
            "thread_id": thread_id,
            "request_id": request_id,
        },
    )


def stable_profile_requirement_id(
    *,
    field: str,
    value_preview: str,
    thread_id: str,
    request_id: str,
) -> str:
    return _stable_id(
        PROFILE_REQUIREMENT_ID_PREFIX,
        {
            "field": field,
            "value_preview": value_preview,
            "thread_id": thread_id,
            "request_id": request_id,
        },
    )


def normalize_learning_goal(value: object, *, max_chars: int = 160) -> str:
    """Build a compact deterministic learning-goal key."""
    text = sanitize_workspace_text(value, max_chars=max_chars, fallback="")
    lowered = text.lower()
    tokens = _GOAL_TOKEN_RE.findall(lowered)
    return "_".join(tokens)[:max_chars]


def workspace_scope_from_state(state: dict[str, Any]) -> dict[str, str]:
    thread_id = sanitize_workspace_text(
        state.get("thread_id") or state.get("session_id"),
        max_chars=120,
        fallback="",
    )
    subject = sanitize_workspace_text(
        state.get("primary_subject") or state.get("subject"),
        max_chars=120,
        fallback="",
    )
    normalized_subject = normalize_subject(subject) if subject else ""
    learning_goal = sanitize_workspace_text(
        state.get("learning_goal"),
        max_chars=240,
        fallback="",
    )
    normalized_goal = normalize_learning_goal(learning_goal)
    workspace_id = stable_workspace_id(
        thread_id=thread_id,
        normalized_subject=normalized_subject,
        normalized_learning_goal=normalized_goal,
    )
    return {
        "workspace_id": workspace_id,
        "thread_id": thread_id,
        "subject": subject,
        "normalized_subject": normalized_subject,
        "learning_goal": learning_goal,
        "normalized_learning_goal": normalized_goal,
    }


def sanitize_workspace_text(
    value: object,
    *,
    max_chars: int,
    fallback: str = "",
) -> str:
    """Redact, normalize whitespace, and bound workspace text."""
    text = sanitize_error_message(value, max_chars=max(max_chars * 2, max_chars))
    text = " ".join(text.split())
    if not text and fallback:
        text = fallback
    return text[:max_chars]


def sanitize_workspace_metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return sanitize_metadata(value)


def compact_evidence_item(
    item: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
    max_chars: int = 900,
) -> WorkspaceEvidenceSummary:
    """Compact one kept Evidence Judge item into persistent workspace evidence."""
    state_payload = state or {}
    scope = workspace_scope_from_state(state_payload)
    original_evidence_id = sanitize_workspace_text(
        item.get("evidence_id") or item.get("original_evidence_id"),
        max_chars=180,
        fallback="evidence",
    )
    subject = sanitize_workspace_text(
        item.get("subject") or scope.get("subject"),
        max_chars=120,
        fallback="",
    )
    normalized_subject = (
        normalize_subject(subject)
        if subject
        else scope.get(
            "normalized_subject",
            "",
        )
    )
    request_id = sanitize_workspace_text(
        item.get("request_id") or state_payload.get("request_id"),
        max_chars=120,
        fallback="",
    )
    thread_id = sanitize_workspace_text(
        item.get("thread_id") or scope.get("thread_id"),
        max_chars=120,
        fallback="",
    )
    score = _safe_ratio(
        item.get("relevance_score")
        if item.get("relevance_score") is not None
        else item.get("evidence_score")
    )
    title = sanitize_workspace_text(
        item.get("title") or item.get("source") or original_evidence_id,
        max_chars=180,
        fallback="Evidence",
    )
    summary = sanitize_workspace_text(
        item.get("coverage_contribution")
        or item.get("score_reason")
        or item.get("judge_reason")
        or item.get("reason")
        or item.get("summary"),
        max_chars=max_chars,
        fallback=title,
    )
    evidence_id = stable_evidence_id(
        original_evidence_id=original_evidence_id,
        thread_id=thread_id,
        normalized_subject=normalized_subject,
        request_id=request_id,
    )
    result: WorkspaceEvidenceSummary = {
        "evidence_id": evidence_id,
        "original_evidence_id": original_evidence_id,
        "title": title,
        "summary": summary,
        "subject": subject,
        "normalized_subject": normalized_subject,
        "source_type": sanitize_workspace_text(
            item.get("source_type"),
            max_chars=80,
            fallback="",
        ),
        "purpose": "factual_grounding",
        "created_at": _safe_iso(item.get("created_at")) or utc_now_iso(),
        "request_id": request_id,
        "thread_id": thread_id,
        "usable_for": ["factual_grounding", "resource_generation"],
    }
    if score is not None:
        result["support_score"] = score
        result["relevance_score"] = score
    return result


def compact_artifact_result(
    result: dict[str, Any],
    state: dict[str, Any],
    *,
    max_preview_chars: int = 1200,
) -> WorkspaceArtifactSummary:
    """Compact one successful resource result into a durable artifact summary."""
    scope = workspace_scope_from_state(state)
    resource_type = sanitize_workspace_text(
        result.get("resource_type"),
        max_chars=80,
        fallback="artifact",
    )
    artifact_candidate = result.get("artifact")
    artifact: dict[str, Any] = (
        artifact_candidate if isinstance(artifact_candidate, dict) else {}
    )
    title = sanitize_workspace_text(
        result.get("title") or artifact.get("title") or resource_type,
        max_chars=180,
        fallback=resource_type,
    )
    refs = _safe_artifact_refs(result)
    request_id = sanitize_workspace_text(
        state.get("request_id"),
        max_chars=120,
        fallback="",
    )
    thread_id = scope["thread_id"]
    artifact_id = stable_artifact_id(
        resource_type=resource_type,
        title=title,
        thread_id=thread_id,
        request_id=request_id,
        normalized_subject=scope["normalized_subject"],
        artifact_refs=refs,
    )
    message_preview = sanitize_workspace_text(
        result.get("message_preview")
        or artifact.get("summary")
        or artifact.get("status")
        or title,
        max_chars=max_preview_chars,
        fallback=title,
    )
    summary = sanitize_workspace_text(
        artifact.get("summary") or result.get("summary") or message_preview,
        max_chars=700,
        fallback=message_preview,
    )
    return {
        "artifact_id": artifact_id,
        "resource_type": resource_type,
        "title": title,
        "summary": summary,
        "message_preview": message_preview,
        "subject": scope["subject"],
        "normalized_subject": scope["normalized_subject"],
        "active_learning_goal": scope["learning_goal"],
        "normalized_learning_goal": scope["normalized_learning_goal"],
        "purpose": "artifact_reference",
        "thread_id": thread_id,
        "request_id": request_id,
        "created_at": utc_now_iso(),
        "metrics": _safe_metrics(result),
        "artifact_refs": refs,
    }


def build_workspace_evidence_update(
    state: dict[str, Any],
    judged_output: dict[str, Any],
) -> WorkspaceUpdate:
    """Build a workspace update from kept Evidence Judge outputs only."""
    now = utc_now_iso()
    judge = judged_output.get("evidence_judge_output")
    if not isinstance(judge, dict):
        judge = judged_output
    judged_items = judge.get("judged_evidence") if isinstance(judge, dict) else []
    kept_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(judged_items, list):
        for raw_item in judged_items:
            if isinstance(raw_item, dict) and raw_item.get("keep") is True:
                evidence_id = sanitize_workspace_text(
                    raw_item.get("evidence_id"),
                    max_chars=180,
                    fallback="",
                )
                if evidence_id:
                    kept_by_id[evidence_id] = raw_item

    evidence_items: list[WorkspaceEvidenceSummary] = []
    graded = judged_output.get("graded_evidence") or judged_output.get("context_docs")
    if isinstance(graded, list):
        for raw_doc in graded:
            if not isinstance(raw_doc, dict):
                continue
            original_id = sanitize_workspace_text(
                raw_doc.get("evidence_id"),
                max_chars=180,
                fallback="",
            )
            judge_item = kept_by_id.get(original_id)
            if not judge_item and raw_doc.get("judge_keep") is not True:
                continue
            merged = {**raw_doc, **(judge_item or {})}
            merged["created_at"] = now
            evidence_items.append(compact_evidence_item(merged, state=state))

    if not evidence_items and kept_by_id:
        candidates = judged_output.get("evidence_candidates") or []
        candidates_by_id = {
            str(item.get("evidence_id") or ""): item
            for item in candidates
            if isinstance(item, dict)
        }
        for original_id, judge_item in kept_by_id.items():
            candidate = candidates_by_id.get(original_id, {})
            merged = {**candidate, **judge_item, "created_at": now}
            evidence_items.append(compact_evidence_item(merged, state=state))

    gaps = _compact_coverage_gaps(
        judged_output.get("coverage_gaps")
        or judge.get("coverage_gaps")
        or state.get("evidence_coverage_gaps")
        or [],
        state=state,
        created_at=now,
    )
    update: WorkspaceUpdate = _base_workspace_update(state, now=now)
    update.update(
        {
            "evidence_state": sanitize_workspace_text(
                judge.get("overall_evidence_state") if isinstance(judge, dict) else "",
                max_chars=80,
                fallback="",
            ),
            "evidence_summaries": evidence_items,
            "coverage_gaps": gaps,
            "updated_sources": ["evidence_judge"],
        }
    )
    return update


def build_workspace_artifact_update(
    state: dict[str, Any],
    successes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build graph state updates for successful resource artifacts."""
    now = utc_now_iso()
    artifacts = [
        compact_artifact_result(result, state)
        for result in successes
        if isinstance(result, dict) and result.get("status") == "success"
    ]
    workspace_update: WorkspaceUpdate = _base_workspace_update(state, now=now)
    artifacts_by_id = {
        artifact["artifact_id"]: artifact
        for artifact in artifacts
        if artifact.get("artifact_id")
    }
    latest_by_type: dict[str, str] = {}
    for artifact in artifacts:
        resource_type = artifact.get("resource_type", "")
        artifact_id = artifact.get("artifact_id", "")
        if resource_type and artifact_id:
            latest_by_type[resource_type] = artifact_id
    workspace_update.update(
        {
            "artifacts_by_id": artifacts_by_id,
            "latest_artifact_by_resource_type": latest_by_type,
            "artifacts": artifacts,
            "updated_sources": ["resource_bundle_output"],
        }
    )
    return {
        "task_workspace": workspace_update,
        "resource_artifacts_by_type": {
            artifact["resource_type"]: artifact
            for artifact in artifacts
            if artifact.get("resource_type")
        },
        "last_generated_artifacts": artifacts,
        "workspace_events": [
            {
                "stage": "resource_artifacts.indexed",
                **workspace_trace_payload(workspace_update),
            }
        ]
        if artifacts
        else [],
    }


def build_workspace_profile_completion_update(
    state: dict[str, Any],
    profile_completion: Mapping[str, Any],
    *,
    field_labels: Mapping[str, str] | None = None,
) -> WorkspaceUpdate:
    """Build a compact workspace update from user-supplied profile facts."""
    now = utc_now_iso()
    scope = workspace_scope_from_state(state)
    request_id = sanitize_workspace_text(state.get("request_id"), max_chars=120)
    thread_id = scope["thread_id"]
    labels = dict(field_labels or {})
    constraints: list[WorkspaceConstraint] = []
    requirements: list[WorkspaceProfileRequirement] = []
    if isinstance(profile_completion, Mapping):
        for field, value in sorted(profile_completion.items()):
            field_key = sanitize_workspace_text(field, max_chars=80)
            value_preview = sanitize_workspace_text(value, max_chars=360)
            if not field_key or not value_preview:
                continue
            label = sanitize_workspace_text(
                labels.get(field_key) or field_key,
                max_chars=120,
                fallback=field_key,
            )
            constraint_id = stable_constraint_id(
                field=field_key,
                value_preview=value_preview,
                thread_id=thread_id,
                request_id=request_id,
            )
            requirement_id = stable_profile_requirement_id(
                field=field_key,
                value_preview=value_preview,
                thread_id=thread_id,
                request_id=request_id,
            )
            constraints.append(
                {
                    "constraint_id": constraint_id,
                    "field": field_key,
                    "label": label,
                    "value_preview": value_preview,
                    "purpose": "profile_completion",
                    "created_at": now,
                    "request_id": request_id,
                    "thread_id": thread_id,
                }
            )
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "field": field_key,
                    "label": label,
                    "value_preview": value_preview,
                    "purpose": "profile_completion",
                    "created_at": now,
                    "request_id": request_id,
                    "thread_id": thread_id,
                }
            )

    update: WorkspaceUpdate = _base_workspace_update(state, now=now)
    update.update(
        {
            "constraints": constraints,
            "profile_requirements": requirements,
            "updated_sources": ["profile_completion"],
        }
    )
    return update


def merge_task_workspace(
    existing: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """Idempotently merge one workspace update into existing workspace state."""
    if isinstance(update, dict) and update.get("__task_workspace_clear__") is True:
        return {}
    existing_workspace = _coerce_workspace(existing)
    update_workspace = _coerce_workspace(update)
    if not update_workspace:
        return existing_workspace
    rotation_action = "merge"
    if _should_rotate_workspace(existing_workspace, update_workspace):
        existing_workspace = {}
        rotation_action = "rotate"

    merged = _base_workspace_from_scope(update_workspace, existing_workspace)
    merged["rotation_action"] = rotation_action
    merged["diagnostics"] = _bounded_strings(
        [
            *(existing_workspace.get("diagnostics") or []),
            *(update_workspace.get("diagnostics") or []),
        ],
        limit=10,
        max_chars=120,
    )

    evidence = _merge_list_by_id(
        existing_workspace.get("evidence_summaries"),
        update_workspace.get("evidence_summaries"),
        id_key="evidence_id",
        limit=WORKSPACE_EVIDENCE_LIMIT,
        text_budget=WORKSPACE_EVIDENCE_TEXT_BUDGET,
    )
    gaps = _merge_list_by_id(
        existing_workspace.get("coverage_gaps"),
        update_workspace.get("coverage_gaps"),
        id_key="gap_id",
        limit=WORKSPACE_GAP_LIMIT,
        text_budget=WORKSPACE_GAP_TEXT_BUDGET,
    )
    artifacts_by_id = _merge_artifact_maps(
        existing_workspace.get("artifacts_by_id"),
        update_workspace.get("artifacts_by_id"),
        existing_workspace.get("artifacts"),
        update_workspace.get("artifacts"),
    )
    artifact_items = _bounded_items(
        artifacts_by_id.values(),
        limit=WORKSPACE_ARTIFACT_LIMIT,
        text_budget=WORKSPACE_ARTIFACT_TEXT_BUDGET,
    )
    artifacts_by_id = {
        str(item.get("artifact_id")): item
        for item in artifact_items
        if isinstance(item, dict) and item.get("artifact_id")
    }
    latest_by_type = _latest_artifact_by_type(
        {
            **_safe_str_dict(
                existing_workspace.get("latest_artifact_by_resource_type")
            ),
            **_safe_str_dict(update_workspace.get("latest_artifact_by_resource_type")),
        },
        artifacts_by_id,
    )
    constraints = _merge_list_by_id(
        existing_workspace.get("constraints"),
        update_workspace.get("constraints"),
        id_key="constraint_id",
        limit=WORKSPACE_CONSTRAINT_LIMIT,
        text_budget=WORKSPACE_CONSTRAINT_TEXT_BUDGET,
    )
    profile_requirements = _merge_list_by_id(
        existing_workspace.get("profile_requirements"),
        update_workspace.get("profile_requirements"),
        id_key="requirement_id",
        limit=WORKSPACE_PROFILE_REQUIREMENT_LIMIT,
        text_budget=WORKSPACE_PROFILE_REQUIREMENT_TEXT_BUDGET,
    )

    merged.update(
        {
            "evidence_summaries": evidence,
            "coverage_gaps": gaps,
            "artifacts_by_id": artifacts_by_id,
            "latest_artifact_by_resource_type": latest_by_type,
            "artifacts": list(artifact_items),
            "constraints": constraints,
            "profile_requirements": profile_requirements,
        }
    )
    return _enforce_workspace_text_budget(merged)


def workspace_trace_payload(
    workspace_update: Mapping[str, Any],
) -> WorkspaceTracePayload:
    """Return counts/metadata only; never raw content."""
    workspace = _coerce_workspace(dict(workspace_update))
    artifacts = workspace.get("artifacts_by_id")
    artifact_count = (
        len(artifacts)
        if isinstance(artifacts, dict)
        else len(workspace.get("artifacts") or [])
    )
    return {
        "workspace_id": sanitize_workspace_text(
            workspace.get("workspace_id"),
            max_chars=160,
            fallback="",
        ),
        "thread_id": sanitize_workspace_text(
            workspace.get("thread_id")
            or (workspace.get("scope") or {}).get("thread_id"),
            max_chars=120,
            fallback="",
        ),
        "request_id": sanitize_workspace_text(
            workspace.get("request_id"),
            max_chars=120,
            fallback="",
        ),
        "active_subject": sanitize_workspace_text(
            workspace.get("active_subject"),
            max_chars=120,
            fallback="",
        ),
        "active_learning_goal_present": bool(
            sanitize_workspace_text(
                workspace.get("active_learning_goal"),
                max_chars=1,
                fallback="",
            )
        ),
        "evidence_summary_count": len(workspace.get("evidence_summaries") or []),
        "coverage_gap_count": len(workspace.get("coverage_gaps") or []),
        "artifact_count": artifact_count,
        "constraint_count": len(workspace.get("constraints") or []),
        "updated_sources": _bounded_strings(
            workspace.get("updated_sources") or [],
            limit=8,
            max_chars=80,
        ),
        "rotation_action": sanitize_workspace_text(
            workspace.get("rotation_action"),
            max_chars=80,
            fallback="merge",
        ),
        "diagnostics": _bounded_strings(
            workspace.get("diagnostics") or [],
            limit=8,
            max_chars=120,
        ),
    }


def workspace_status_payload(workspace: dict[str, Any] | None) -> dict[str, Any]:
    safe = _coerce_workspace(workspace)
    if not safe:
        return {
            "workspace_present": False,
            "workspace_active_subject": "",
            "workspace_evidence_summary_count": 0,
            "workspace_gap_count": 0,
            "workspace_artifact_count": 0,
            "workspace_updated_at": "",
        }
    artifacts = safe.get("artifacts_by_id")
    artifact_count = (
        len(artifacts)
        if isinstance(artifacts, dict)
        else len(safe.get("artifacts") or [])
    )
    evidence_count = len(safe.get("evidence_summaries") or [])
    gap_count = len(safe.get("coverage_gaps") or [])
    present = bool(
        safe.get("workspace_id") or evidence_count or gap_count or artifact_count
    )
    return {
        "workspace_present": present,
        "workspace_active_subject": sanitize_workspace_text(
            safe.get("active_subject"),
            max_chars=120,
            fallback="",
        ),
        "workspace_evidence_summary_count": evidence_count,
        "workspace_gap_count": gap_count,
        "workspace_artifact_count": artifact_count,
        "workspace_updated_at": sanitize_workspace_text(
            safe.get("updated_at"),
            max_chars=80,
            fallback="",
        ),
    }


def workspace_continuation_context(
    state: Mapping[str, Any],
) -> WorkspaceContinuationContext:
    """Return safe same-thread task continuation context without mutating state."""
    current_thread = sanitize_workspace_text(
        state.get("thread_id") or state.get("session_id"),
        max_chars=120,
        fallback="",
    )
    request_id = sanitize_workspace_text(
        state.get("request_id"),
        max_chars=120,
        fallback="",
    )
    resource_types = _requested_resource_types_from_state(state)
    is_recommendation_request = state.get("response_mode") == "recommendation"
    workspace = _coerce_workspace(
        state.get("task_workspace")
        if isinstance(state.get("task_workspace"), dict)
        else None
    )
    diagnostics = _bounded_strings(
        workspace.get("diagnostics") or [],
        limit=8,
        max_chars=120,
    )
    active_subject = sanitize_workspace_text(
        workspace.get("active_subject"),
        max_chars=120,
        fallback="",
    )
    normalized_subject = sanitize_workspace_text(
        workspace.get("normalized_subject")
        or (normalize_subject(active_subject) if active_subject else ""),
        max_chars=120,
        fallback="",
    )
    workspace_thread = sanitize_workspace_text(
        workspace.get("thread_id") or (workspace.get("scope") or {}).get("thread_id"),
        max_chars=120,
        fallback="",
    )
    context: WorkspaceContinuationContext = {
        "can_continue": False,
        "skip_reason": "",
        "workspace_id": sanitize_workspace_text(
            workspace.get("workspace_id"),
            max_chars=160,
            fallback="",
        ),
        "thread_id": current_thread,
        "current_thread_id": current_thread,
        "workspace_thread_id": workspace_thread,
        "request_id": request_id,
        "active_subject": active_subject,
        "normalized_subject": normalized_subject,
        "active_learning_goal": sanitize_workspace_text(
            workspace.get("active_learning_goal"),
            max_chars=240,
            fallback="",
        ),
        "normalized_learning_goal": sanitize_workspace_text(
            workspace.get("normalized_learning_goal"),
            max_chars=240,
            fallback="",
        ),
        "resource_types": resource_types,
        "diagnostics": diagnostics,
    }

    if not workspace or "workspace_schema_version_incompatible" in diagnostics:
        return _continuation_skip(context, "workspace_unavailable")
    if not current_thread or not workspace_thread:
        return _continuation_skip(context, "thread_missing")
    if current_thread != workspace_thread:
        return _continuation_skip(context, "thread_mismatch")
    if not resource_types and not is_recommendation_request:
        return _continuation_skip(context, "no_resource_request")
    if _has_explicit_current_subject(state):
        return _continuation_skip(context, "current_subject_present")
    if not normalized_subject or normalized_subject == "other":
        return _continuation_skip(context, "workspace_subject_unavailable")

    context["can_continue"] = True
    context["skip_reason"] = ""
    return context


def workspace_continuation_trace_payload(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact continuation diagnostics suitable for A3_TRACE."""
    current_thread = sanitize_workspace_text(
        context.get("current_thread_id") or context.get("thread_id"),
        max_chars=120,
        fallback="",
    )
    workspace_thread = sanitize_workspace_text(
        context.get("workspace_thread_id"),
        max_chars=120,
        fallback="",
    )
    return {
        "can_continue": bool(context.get("can_continue")),
        "continuation_applied": bool(context.get("continuation_applied")),
        "skip_reason": sanitize_workspace_text(
            context.get("skip_reason"),
            max_chars=120,
            fallback="",
        ),
        "workspace_id": sanitize_workspace_text(
            context.get("workspace_id"),
            max_chars=160,
            fallback="",
        ),
        "thread_id": sanitize_workspace_text(
            current_thread,
            max_chars=120,
            fallback="",
        ),
        "current_thread_id": current_thread,
        "workspace_thread_id": workspace_thread,
        "request_id": sanitize_workspace_text(
            context.get("request_id"),
            max_chars=120,
            fallback="",
        ),
        "active_subject": sanitize_workspace_text(
            context.get("active_subject"),
            max_chars=120,
            fallback="",
        ),
        "normalized_subject": sanitize_workspace_text(
            context.get("normalized_subject"),
            max_chars=120,
            fallback="",
        ),
        "active_learning_goal_present": bool(
            sanitize_workspace_text(
                context.get("active_learning_goal"),
                max_chars=1,
                fallback="",
            )
        ),
        "resource_types": _bounded_strings(
            context.get("resource_types") or [],
            limit=8,
            max_chars=80,
        ),
        "diagnostics": _bounded_strings(
            context.get("diagnostics") or [],
            limit=8,
            max_chars=120,
        ),
    }


def _continuation_skip(
    context: WorkspaceContinuationContext,
    reason: str,
) -> WorkspaceContinuationContext:
    context["can_continue"] = False
    context["skip_reason"] = sanitize_workspace_text(
        reason,
        max_chars=120,
        fallback="unknown",
    )
    return context


def _requested_resource_types_from_state(state: Mapping[str, Any]) -> list[str]:
    resource_types: list[str] = []

    def add(value: object) -> None:
        text = sanitize_workspace_text(value, max_chars=80, fallback="")
        if text and text not in resource_types:
            resource_types.append(text)

    raw_types = state.get("requested_resource_types")
    if isinstance(raw_types, (list, tuple)):
        for item in raw_types:
            add(item)
    add(state.get("requested_resource_type"))
    return resource_types[:8]


def _has_explicit_current_subject(state: Mapping[str, Any]) -> bool:
    raw_candidates = state.get("subject_candidates")
    if isinstance(raw_candidates, (list, tuple)):
        for candidate in raw_candidates:
            normalized = _normalize_workspace_subject(candidate)
            if normalized and normalized != "other":
                return True

    normalized_subject = _normalize_workspace_subject(state.get("subject"))
    return bool(normalized_subject and normalized_subject != "other")


def _normalize_workspace_subject(value: object) -> str:
    subject = sanitize_workspace_text(value, max_chars=120, fallback="")
    return normalize_subject(subject) if subject else ""


def _base_workspace_update(state: dict[str, Any], *, now: str) -> WorkspaceUpdate:
    scope = workspace_scope_from_state(state)
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": scope["workspace_id"],
        "scope": {
            "thread_id": scope["thread_id"],
            "normalized_subject": scope["normalized_subject"],
            "normalized_learning_goal": scope["normalized_learning_goal"],
        },
        "thread_id": scope["thread_id"],
        "request_id": sanitize_workspace_text(
            state.get("request_id"),
            max_chars=120,
            fallback="",
        ),
        "active_subject": scope["subject"],
        "normalized_subject": scope["normalized_subject"],
        "active_learning_goal": scope["learning_goal"],
        "normalized_learning_goal": scope["normalized_learning_goal"],
        "updated_at": now,
    }


def _base_workspace_from_scope(
    update: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": update.get("workspace_id")
        or existing.get("workspace_id")
        or "",
        "scope": update.get("scope") or existing.get("scope") or {},
        "thread_id": update.get("thread_id") or existing.get("thread_id") or "",
        "request_id": update.get("request_id") or existing.get("request_id") or "",
        "active_subject": update.get("active_subject")
        or existing.get("active_subject")
        or "",
        "normalized_subject": update.get("normalized_subject")
        or existing.get("normalized_subject")
        or "",
        "active_learning_goal": update.get("active_learning_goal")
        or existing.get("active_learning_goal")
        or "",
        "normalized_learning_goal": update.get("normalized_learning_goal")
        or existing.get("normalized_learning_goal")
        or "",
        "evidence_state": update.get("evidence_state")
        or existing.get("evidence_state")
        or "",
        "updated_at": update.get("updated_at") or existing.get("updated_at") or "",
        "updated_sources": _bounded_strings(
            [
                *(existing.get("updated_sources") or []),
                *(update.get("updated_sources") or []),
            ],
            limit=12,
            max_chars=80,
        ),
    }
    if not result["scope"]:
        result["scope"] = {
            "thread_id": result["thread_id"],
            "normalized_subject": result["normalized_subject"],
            "normalized_learning_goal": result["normalized_learning_goal"],
        }
    return result


def _coerce_workspace(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    diagnostics: list[str] = []
    version = value.get("schema_version")
    if version not in (None, WORKSPACE_SCHEMA_VERSION):
        diagnostics.append("workspace_schema_version_incompatible")
        return {"schema_version": WORKSPACE_SCHEMA_VERSION, "diagnostics": diagnostics}
    workspace = dict(value)
    workspace["schema_version"] = WORKSPACE_SCHEMA_VERSION
    if version is None:
        diagnostics.append("workspace_schema_version_missing")
    workspace["diagnostics"] = _bounded_strings(
        [*(workspace.get("diagnostics") or []), *diagnostics],
        limit=10,
        max_chars=120,
    )
    if "artifacts_by_id" not in workspace and isinstance(
        workspace.get("artifacts"), list
    ):
        artifacts_by_id = {}
        for item in workspace.get("artifacts") or []:
            if isinstance(item, dict) and item.get("artifact_id"):
                artifacts_by_id[str(item["artifact_id"])] = item
        workspace["artifacts_by_id"] = artifacts_by_id
    return workspace


def _should_rotate_workspace(existing: dict[str, Any], update: dict[str, Any]) -> bool:
    if not existing:
        return False
    old_thread = str(
        existing.get("thread_id")
        or (existing.get("scope") or {}).get("thread_id")
        or ""
    )
    new_thread = str(
        update.get("thread_id") or (update.get("scope") or {}).get("thread_id") or ""
    )
    if old_thread and new_thread and old_thread != new_thread:
        return True
    old_subject = str(existing.get("normalized_subject") or "")
    new_subject = str(update.get("normalized_subject") or "")
    if old_subject and new_subject and old_subject != new_subject:
        return True
    old_goal = str(existing.get("normalized_learning_goal") or "")
    new_goal = str(update.get("normalized_learning_goal") or "")
    if old_goal and new_goal and old_goal != new_goal:
        return _goal_overlap(old_goal, new_goal) < 0.25
    return False


def _goal_overlap(left: str, right: str) -> float:
    left_terms = set(_GOAL_TOKEN_RE.findall(left))
    right_terms = set(_GOAL_TOKEN_RE.findall(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)


def _compact_coverage_gaps(
    gaps: Any,
    *,
    state: dict[str, Any],
    created_at: str,
) -> list[WorkspaceCoverageGap]:
    if not isinstance(gaps, list):
        return []
    scope = workspace_scope_from_state(state)
    request_id = sanitize_workspace_text(state.get("request_id"), max_chars=120)
    thread_id = scope["thread_id"]
    result: list[WorkspaceCoverageGap] = []
    for raw_gap in gaps:
        if not isinstance(raw_gap, dict):
            continue
        gap_text = sanitize_workspace_text(raw_gap.get("gap"), max_chars=500)
        if not gap_text:
            continue
        subject = sanitize_workspace_text(
            raw_gap.get("subject") or scope["subject"],
            max_chars=120,
        )
        role = sanitize_workspace_text(raw_gap.get("role"), max_chars=80)
        result.append(
            {
                "gap_id": stable_gap_id(
                    gap=gap_text,
                    subject=subject,
                    role=role,
                    thread_id=thread_id,
                ),
                "subject": subject,
                "normalized_subject": normalize_subject(subject) if subject else "",
                "role": role,
                "gap": gap_text,
                "suggested_search_query": sanitize_workspace_text(
                    raw_gap.get("suggested_search_query"),
                    max_chars=220,
                ),
                "purpose": sanitize_workspace_text(
                    raw_gap.get("purpose"),
                    max_chars=80,
                ),
                "priority": _safe_ratio(raw_gap.get("priority")) or 0.5,
                "created_at": created_at,
                "request_id": request_id,
                "thread_id": thread_id,
            }
        )
    return result


def _safe_artifact_refs(result: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}

    def visit(value: Any, *, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = sanitize_workspace_text(key, max_chars=80)
                if key_text in _RAW_CONTENT_KEYS:
                    continue
                if key_text in _SAFE_REF_KEYS:
                    safe = _safe_ref_value(key_text, item)
                    if safe:
                        refs[key_text] = safe
                    continue
                if isinstance(item, dict):
                    visit(item, prefix=key_text)
                elif isinstance(item, list):
                    for index, child in enumerate(item[:5]):
                        visit(child, prefix=f"{key_text}.{index}")
        elif isinstance(value, list):
            for index, item in enumerate(value[:5]):
                visit(item, prefix=f"{prefix}.{index}")

    visit(result)
    return dict(sorted(refs.items()))


def _safe_ref_value(key: str, value: Any) -> str:
    text = sanitize_workspace_text(value, max_chars=500)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith(("postgres://", "postgresql://", "mysql://", "mongodb://")):
        return ""
    if "url" in key:
        return _safe_url(text)
    if "path" in key:
        return _safe_relative_path(text)
    return _safe_filename_or_relative(text)


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return ""
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return _safe_relative_path(value)
    if parts.username or parts.password or "@" in parts.netloc:
        return ""
    query_lower = parts.query.lower()
    if any(hint in query_lower for hint in _UNSAFE_URL_HINTS):
        return ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _safe_filename_or_relative(value: str) -> str:
    if "/" in value or "\\" in value:
        return _safe_relative_path(value)
    text = sanitize_workspace_text(value, max_chars=240)
    if not text or text in {".", ".."}:
        return ""
    return text


def _safe_relative_path(value: str) -> str:
    text = sanitize_workspace_text(value, max_chars=500)
    if not text:
        return ""
    if text.startswith(("~", "/", "\\")) or _WINDOWS_DRIVE_RE.match(text):
        return ""
    text = text.replace("\\", "/")
    path = PurePosixPath(text)
    if any(part in {"", ".", ".."} for part in path.parts):
        return ""
    if PureWindowsPath(text).is_absolute():
        return ""
    return str(path)


def _safe_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    for key in ("elapsed_ms", "message_chars"):
        if key in result:
            metrics = {**metrics, key: result[key]}
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        key_text = sanitize_workspace_text(key, max_chars=80)
        if not key_text:
            continue
        if isinstance(value, bool):
            safe[key_text] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            safe[key_text] = value
        elif isinstance(value, str):
            safe[key_text] = sanitize_workspace_text(value, max_chars=120)
    return safe


def _merge_list_by_id(
    existing: Any,
    update: Any,
    *,
    id_key: str,
    limit: int,
    text_budget: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*(existing or []), *(update or [])]:
        if not isinstance(item, dict):
            continue
        item_id = sanitize_workspace_text(item.get(id_key), max_chars=220)
        if not item_id:
            continue
        merged[item_id] = sanitize_workspace_metadata(item)
    return list(_bounded_items(merged.values(), limit=limit, text_budget=text_budget))


def _merge_artifact_maps(*values: Any) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for value in values:
        if isinstance(value, dict):
            iterable: Iterable[Any] = value.values()
        elif isinstance(value, list):
            iterable = value
        else:
            continue
        for item in iterable:
            if not isinstance(item, dict):
                continue
            artifact_id = sanitize_workspace_text(
                item.get("artifact_id"), max_chars=220
            )
            if artifact_id:
                merged[artifact_id] = sanitize_workspace_metadata(item)
    return merged


def _latest_artifact_by_type(
    previous: dict[str, str],
    artifacts_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    latest: dict[str, str] = {}
    ordered = _bounded_items(
        artifacts_by_id.values(),
        limit=WORKSPACE_ARTIFACT_LIMIT,
        text_budget=WORKSPACE_ARTIFACT_TEXT_BUDGET,
    )
    for item in reversed(list(ordered)):
        resource_type = sanitize_workspace_text(item.get("resource_type"), max_chars=80)
        artifact_id = sanitize_workspace_text(item.get("artifact_id"), max_chars=220)
        if resource_type and artifact_id:
            latest[resource_type] = artifact_id
    for resource_type, artifact_id in previous.items():
        if artifact_id in artifacts_by_id and resource_type not in latest:
            latest[resource_type] = artifact_id
    return dict(sorted(latest.items()))


def _bounded_items(
    values: Any,
    *,
    limit: int,
    text_budget: int,
) -> tuple[dict[str, Any], ...]:
    items = [item for item in values if isinstance(item, dict)]
    ordered = sorted(
        items,
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(
                item.get("artifact_id")
                or item.get("evidence_id")
                or item.get("gap_id")
                or ""
            ),
        ),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    total_chars = 0
    for item in ordered:
        item_chars = _json_chars(item)
        if len(result) >= limit:
            break
        if result and total_chars + item_chars > text_budget:
            continue
        result.append(item)
        total_chars += item_chars
    return tuple(result)


def _enforce_workspace_text_budget(workspace: dict[str, Any]) -> dict[str, Any]:
    if _json_chars(workspace) <= WORKSPACE_TEXT_BUDGET:
        return workspace
    workspace = dict(workspace)
    workspace["evidence_summaries"] = workspace.get("evidence_summaries", [])[:10]
    workspace["coverage_gaps"] = workspace.get("coverage_gaps", [])[:10]
    workspace["constraints"] = workspace.get("constraints", [])[:8]
    workspace["profile_requirements"] = workspace.get("profile_requirements", [])[:8]
    artifacts = list(workspace.get("artifacts", []))[:10]
    workspace["artifacts"] = artifacts
    workspace["artifacts_by_id"] = {
        item["artifact_id"]: item
        for item in artifacts
        if isinstance(item, dict) and item.get("artifact_id")
    }
    workspace["diagnostics"] = _bounded_strings(
        [*(workspace.get("diagnostics") or []), "workspace_text_budget_trimmed"],
        limit=10,
        max_chars=120,
    )
    return workspace


def _safe_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        safe_key = sanitize_workspace_text(key, max_chars=120)
        safe_item = sanitize_workspace_text(item, max_chars=220)
        if safe_key and safe_item:
            result[safe_key] = safe_item
    return result


def _bounded_strings(values: Any, *, limit: int, max_chars: int) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = sanitize_workspace_text(value, max_chars=max_chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _safe_ratio(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        return None
    return ratio


def _safe_iso(value: Any) -> str:
    text = sanitize_workspace_text(value, max_chars=80)
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        return ""
    return parsed.astimezone(timezone.utc).isoformat()


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _json_chars(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except TypeError:
        return len(str(value))
