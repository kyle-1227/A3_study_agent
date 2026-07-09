"""Stable resource_final event contract helpers.

This module is intentionally pure: it normalizes already-produced resource
artifacts into a bounded, sanitized SSE/status payload. It does not call LLMs,
storage, network APIs, or mutate graph state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from src.context_engineering.workspace import sanitize_workspace_text

RESOURCE_FINAL_SCHEMA_VERSION = 1
RESOURCE_ID_PREFIX = "resource:v1"
PAYLOAD_HASH_PREFIX = "payload:v1"
RESOURCE_TEXT_LIMIT = 12_000
RESOURCE_SHORT_TEXT_LIMIT = 1_200
RESOURCE_LIST_LIMIT = 80
RESOURCE_DICT_LIMIT = 80

_RENDER_FIELD_KEYS = {
    "answer",
    "markdown",
    "srt",
    "render_log",
    "content",
    "summary",
    "message",
}
_REF_HINTS = ("url", "filename", "path")
_UNSAFE_URL_HINTS = (
    "token",
    "signature",
    "credential",
    "secret",
    "key",
    "auth",
    "cookie",
    "password",
    "expires",
    "x-amz",
    "sig",
)
_UNSAFE_SCHEMES = ("postgres://", "postgresql://", "mysql://", "mongodb://")
_WINDOWS_DRIVE_PREFIXES = tuple(
    f"{chr(code)}:" for code in range(ord("a"), ord("z") + 1)
)


def normalize_resource_final_payload(
    legacy_payload: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a stable, sanitized resource_final payload.

    Existing legacy fields are retained for backward compatibility. New clients
    should consume the normalized ``resource`` object first.
    """
    if not isinstance(legacy_payload, Mapping) or not legacy_payload:
        return None

    state_payload = state or {}
    safe_payload = _sanitize_render_value(dict(legacy_payload))
    if not isinstance(safe_payload, dict):
        return None

    resource_type = sanitize_workspace_text(
        safe_payload.get("resource_type"),
        max_chars=80,
        fallback="resource",
    )
    thread_id = sanitize_workspace_text(
        safe_payload.get("thread_id")
        or state_payload.get("thread_id")
        or state_payload.get("session_id"),
        max_chars=120,
        fallback="",
    )
    request_id = sanitize_workspace_text(
        safe_payload.get("request_id") or state_payload.get("request_id"),
        max_chars=120,
        fallback="",
    )
    resource = _build_resource_object(safe_payload, resource_type=resource_type)
    payload_hash = stable_payload_hash(resource)
    resource_id = stable_resource_id(
        thread_id=thread_id,
        request_id=request_id,
        resource_type=resource_type,
        payload_hash=payload_hash,
    )

    safe_payload.update(
        {
            "type": "resource_final",
            "schema_version": RESOURCE_FINAL_SCHEMA_VERSION,
            "resource_type": resource_type,
            "thread_id": thread_id,
            "request_id": request_id,
            "resource_id": resource_id,
            "payload_hash": payload_hash,
            "resource": resource,
            "render_hints": resource.get("render_hints", {}),
        }
    )
    return safe_payload


def resource_run_was_requested(state: Mapping[str, Any] | None) -> bool:
    """Return true only for states that actually entered a resource workflow."""
    if not isinstance(state, Mapping):
        return False
    requested = _resource_types_from_state(state)
    if requested:
        return True
    status = sanitize_workspace_text(
        state.get("resource_generation_status"),
        max_chars=80,
        fallback="",
    )
    if status in {"success", "partial_success", "failed", "error"}:
        return True
    plan = state.get("resource_generation_plan")
    if isinstance(plan, Mapping) and plan.get("tasks"):
        return True
    return bool(state.get("resource_bundle_artifact"))


def completed_without_resource_payload(
    state: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build an additive diagnostic for a resource run with no renderable payload."""
    if not resource_run_was_requested(state):
        return None
    state_payload = state or {}
    return {
        "type": "resource_final_diagnostic",
        "status": "completed_without_resource",
        "thread_id": sanitize_workspace_text(
            state_payload.get("thread_id") or state_payload.get("session_id"),
            max_chars=120,
            fallback="",
        ),
        "request_id": sanitize_workspace_text(
            state_payload.get("request_id"),
            max_chars=120,
            fallback="",
        ),
        "requested_resource_type": sanitize_workspace_text(
            state_payload.get("requested_resource_type"),
            max_chars=80,
            fallback="",
        ),
        "requested_resource_types": _resource_types_from_state(state_payload),
        "resource_generation_status": sanitize_workspace_text(
            state_payload.get("resource_generation_status"),
            max_chars=80,
            fallback="",
        ),
    }


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{PAYLOAD_HASH_PREFIX}:{digest}"


def stable_resource_id(
    *,
    thread_id: str,
    request_id: str,
    resource_type: str,
    payload_hash: str,
) -> str:
    parts = {
        "thread_id": thread_id,
        "request_id": request_id,
        "resource_type": resource_type,
        "payload_hash": payload_hash,
    }
    body = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{RESOURCE_ID_PREFIX}:{digest}"


def _build_resource_object(
    payload: Mapping[str, Any], *, resource_type: str
) -> dict[str, Any]:
    render_payload = _extract_render_payload(payload, resource_type=resource_type)
    artifact_refs = _collect_artifact_refs(render_payload)
    title = _resource_title(render_payload, resource_type=resource_type)
    legacy_fields = sorted(
        key
        for key in (
            "mindmap",
            "review_doc",
            "review_doc_artifacts",
            "exercise_artifact",
            "exercise_items",
            "code_practice_artifact",
            "video_script_artifact",
            "video_animation_artifact",
            "study_plan",
            "resource_bundle",
        )
        if key in payload
    )
    return {
        "kind": resource_type,
        "title": title,
        "summary": sanitize_workspace_text(
            payload.get("answer")
            or _nested_get(render_payload, ("resource_bundle", "message"))
            or _nested_get(render_payload, ("resource_bundle", "status"))
            or title,
            max_chars=RESOURCE_SHORT_TEXT_LIMIT,
            fallback=title,
        ),
        "payload": render_payload,
        "artifact_refs": artifact_refs,
        "render_hints": {
            "primary_card": resource_type,
            "legacy_fields": legacy_fields,
            "has_downloads": bool(artifact_refs),
        },
    }


def _extract_render_payload(
    payload: Mapping[str, Any], *, resource_type: str
) -> dict[str, Any]:
    if isinstance(payload.get("resource"), Mapping):
        resource_payload = payload["resource"].get("payload")
        if isinstance(resource_payload, Mapping):
            sanitized = _sanitize_render_value(dict(resource_payload))
            return sanitized if isinstance(sanitized, dict) else {}

    keys_by_type = {
        "bundle": (
            "resource_bundle",
            "resources",
            "errors",
            "mindmap",
            "review_doc",
            "review_doc_artifacts",
            "exercise_items",
            "exercise_artifact",
            "code_practice_artifact",
            "video_script_artifact",
            "video_animation_artifact",
            "study_plan",
        ),
        "mindmap": ("mindmap", "mindmap_artifact", "mindmap_tree"),
        "quiz": ("exercise_artifact", "exercise_items"),
        "exercise": ("exercise_artifact", "exercise_items"),
        "review_doc": ("review_doc", "review_doc_artifacts"),
        "code_practice": ("code_practice_artifact",),
        "video_script": ("video_script_artifact",),
        "video_animation": ("video_animation_artifact",),
        "study_plan": ("study_plan",),
        "evidence_summary": ("answer", "controlled_stop", "controlled_stop_reason"),
    }
    keys = keys_by_type.get(resource_type, ())
    render_payload = {key: payload[key] for key in keys if key in payload}
    if not render_payload and payload.get("answer"):
        render_payload = {"answer": payload.get("answer")}
    sanitized = _sanitize_render_value(render_payload)
    return sanitized if isinstance(sanitized, dict) else {}


def _resource_title(payload: Mapping[str, Any], *, resource_type: str) -> str:
    for path in (
        ("mindmap", "title"),
        ("review_doc", "title"),
        ("study_plan", "title"),
        ("exercise_artifact", "title"),
        ("code_practice_artifact", "title"),
        ("video_script_artifact", "title"),
        ("video_animation_artifact", "title"),
        ("resource_bundle", "title"),
    ):
        value = _nested_get(payload, path)
        text = sanitize_workspace_text(value, max_chars=180, fallback="")
        if text:
            return text
    return sanitize_workspace_text(resource_type, max_chars=80, fallback="resource")


def _nested_get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _collect_artifact_refs(payload: Any) -> dict[str, str]:
    refs: dict[str, str] = {}

    def visit(value: Any, *, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                key_text = sanitize_workspace_text(
                    child_key, max_chars=120, fallback=""
                )
                visit(child_value, key=key_text)
        elif isinstance(value, list):
            for item in value[:RESOURCE_LIST_LIMIT]:
                visit(item, key=key)
        elif isinstance(value, str) and _is_ref_key(key):
            safe = _safe_resource_ref(key, value)
            if safe:
                refs.setdefault(key, safe)

    visit(payload)
    return dict(sorted(refs.items()))


def _sanitize_render_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > 8:
        return ""
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for index, (raw_key, raw_item) in enumerate(value.items()):
            if index >= RESOURCE_DICT_LIMIT:
                break
            key_text = sanitize_workspace_text(raw_key, max_chars=120, fallback="")
            if not key_text:
                continue
            safe_item = _sanitize_render_value(raw_item, key=key_text, depth=depth + 1)
            if safe_item not in ("", None, {}):
                safe[key_text] = safe_item
        return safe
    if isinstance(value, list):
        return [
            item
            for item in (
                _sanitize_render_value(item, key=key, depth=depth + 1)
                for item in value[:RESOURCE_LIST_LIMIT]
            )
            if item not in ("", None, [], {})
        ]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if _is_ref_key(key):
        return _safe_resource_ref(key, value)
    max_chars = (
        RESOURCE_TEXT_LIMIT if key in _RENDER_FIELD_KEYS else RESOURCE_SHORT_TEXT_LIMIT
    )
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _is_ref_key(key: str) -> bool:
    lowered = str(key or "").lower()
    return any(hint in lowered for hint in _REF_HINTS)


def _safe_resource_ref(key: str, value: Any) -> str:
    text = sanitize_workspace_text(value, max_chars=800, fallback="")
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith(_UNSAFE_SCHEMES):
        return ""
    if "url" in key.lower():
        return _safe_url(text)
    if "/" in text or "\\" in text:
        return _safe_relative_or_app_path(text)
    return sanitize_workspace_text(text, max_chars=240, fallback="")


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return ""
    if parts.scheme in {"http", "https"} and parts.netloc:
        if parts.username or parts.password or "@" in parts.netloc:
            return ""
        if any(hint in parts.query.lower() for hint in _UNSAFE_URL_HINTS):
            return ""
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if not parts.scheme and not parts.netloc:
        return _safe_relative_or_app_path(value)
    return ""


def _safe_relative_or_app_path(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text or text.startswith("~"):
        return ""
    lowered = text.lower()
    if (
        lowered.startswith(_WINDOWS_DRIVE_PREFIXES)
        or PureWindowsPath(text).is_absolute()
    ):
        return ""
    if text.startswith("//"):
        return ""
    app_relative = text.startswith("/")
    path_text = text[1:] if app_relative else text
    path = PurePosixPath(path_text)
    if any(part in {"", ".", ".."} for part in path.parts):
        return ""
    safe_path = str(path)
    return f"/{safe_path}" if app_relative else safe_path


def _resource_types_from_state(state: Mapping[str, Any]) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        text = sanitize_workspace_text(value, max_chars=80, fallback="")
        if text and text not in values:
            values.append(text)

    raw_types = state.get("requested_resource_types")
    if isinstance(raw_types, list):
        for item in raw_types:
            add(item)
    add(state.get("requested_resource_type"))
    return values
