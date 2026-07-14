"""Strict runtime projection from resource worker results to Resource Final V3."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.context_engineering.workspace import sanitize_workspace_text
from src.graph.resource_final_v3 import (
    JsonValue,
    ResourceFinalV3,
    ResourceFinalV3BlockedResource,
    ResourceFinalV3Error,
    ResourceFinalV3Quiz,
    ResourceFinalV3Recommendation,
    ResourceFinalV3Resource,
    ResourceFinalV3ResourceKind,
    ResourceFinalV3ResourceStatus,
    ResourceFinalV3ResourceValidation,
    ResourceFinalV3TerminalStatus,
    ResourceFinalV3Validation,
    build_resource_final_v3,
    build_resource_final_v3_resource,
)


class ResourceFinalRuntimeError(ValueError):
    """Raised when graph state cannot satisfy the authoritative V3 contract."""


_RESOURCE_ORDER: tuple[ResourceFinalV3ResourceKind, ...] = (
    "review_doc",
    "mindmap",
    "quiz",
    "code_practice",
    "video_script",
    "video_animation",
    "study_plan",
)
_RESOURCE_ORDER_INDEX = {value: index for index, value in enumerate(_RESOURCE_ORDER)}
_MAX_MAPPING_ITEMS = 80
_MAX_SEQUENCE_ITEMS = 80
_MAX_PAYLOAD_DEPTH = 8
_MAX_SHORT_TEXT = 1_200
_MAX_RENDER_TEXT = 12_000
_RENDER_TEXT_KEYS = frozenset(
    {"answer", "content", "markdown", "message", "render_log", "srt", "summary"}
)
_REFERENCE_HINTS = ("url", "filename", "path")
_UNSAFE_URL_HINTS = (
    "auth",
    "cookie",
    "credential",
    "expires",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "token",
    "x-amz",
)
_UNSAFE_SCHEMES = ("mongodb://", "mysql://", "postgres://", "postgresql://")
_WINDOWS_DRIVE_PREFIXES = tuple(
    f"{chr(code)}:" for code in range(ord("a"), ord("z") + 1)
)


class _DropValue:
    """Internal marker for a security-rejected render value."""


_DROP = _DropValue()


def build_resource_final_v3_from_bundle(
    *,
    thread_id: str,
    request_id: str,
    requested_resource_types: Sequence[str],
    terminal_status: ResourceFinalV3TerminalStatus,
    branch_results: Sequence[Mapping[str, Any]],
    blocked_resources: Sequence[Mapping[str, Any]],
    recommendations: Sequence[ResourceFinalV3Recommendation],
    summary: str,
) -> ResourceFinalV3:
    """Build the one authoritative final payload from completed fan-in state."""

    identity_thread = _bounded_required_text(thread_id, "thread_id", max_chars=160)
    identity_request = _bounded_required_text(
        request_id,
        "request_id",
        max_chars=160,
    )
    requested = tuple(_resource_kind(value) for value in requested_resource_types)
    if not requested:
        raise ResourceFinalRuntimeError(
            "Resource Final V3 requires at least one requested resource type"
        )
    if len(requested) != len(set(requested)):
        raise ResourceFinalRuntimeError(
            "requested_resource_types must not contain duplicates"
        )

    resources: list[ResourceFinalV3Resource] = []
    errors: list[ResourceFinalV3Error] = []
    result_types: list[ResourceFinalV3ResourceKind] = []
    for result in branch_results:
        resource_type = _resource_kind(result.get("resource_type"))
        result_types.append(resource_type)
        status = result.get("status")
        if status in {"success", "partial_success"}:
            resources.append(
                _resource_from_success_result(
                    thread_id=identity_thread,
                    request_id=identity_request,
                    resource_type=resource_type,
                    result=result,
                )
            )
        elif status == "failed":
            errors.append(
                _error_from_failed_result(
                    resource_type=resource_type,
                    result=result,
                )
            )
        else:
            raise ResourceFinalRuntimeError(
                f"resource branch {resource_type} has unsupported status {status!r}"
            )
    if len(result_types) != len(set(result_types)):
        raise ResourceFinalRuntimeError(
            "resource branch results must contain at most one result per resource type"
        )

    blocked = tuple(_blocked_resource(value) for value in blocked_resources)
    blocked_types = tuple(item.resource_type for item in blocked)
    if len(blocked_types) != len(set(blocked_types)):
        raise ResourceFinalRuntimeError(
            "blocked resources must contain at most one item per resource type"
        )
    if set(result_types) & set(blocked_types):
        raise ResourceFinalRuntimeError(
            "one resource type cannot be both executed and evidence-blocked"
        )
    if set(requested) != set(result_types) | set(blocked_types):
        raise ResourceFinalRuntimeError(
            "requested resources must exactly match executed and blocked resources"
        )

    ordered_resources = tuple(
        sorted(resources, key=lambda item: _RESOURCE_ORDER_INDEX[item.kind])
    )
    ordered_errors = tuple(
        sorted(errors, key=lambda item: _RESOURCE_ORDER_INDEX[item.resource_type])
    )
    ordered_blocked = tuple(
        sorted(blocked, key=lambda item: _RESOURCE_ORDER_INDEX[item.resource_type])
    )
    validation = ResourceFinalV3Validation(
        schema_version="resource_final_validation_v3",
        resource_count=len(ordered_resources),
        success_count=sum(item.status == "success" for item in ordered_resources),
        partial_success_count=sum(
            item.status == "partial_success" for item in ordered_resources
        ),
        failed_count=len(ordered_errors),
        blocked_count=len(ordered_blocked),
        renderable_count=sum(
            item.validation.renderable_count for item in ordered_resources
        ),
        downloadable_count=sum(
            item.validation.downloadable_count for item in ordered_resources
        ),
    )
    safe_summary = _bounded_required_text(summary, "summary", max_chars=1_200)
    return build_resource_final_v3(
        thread_id=identity_thread,
        request_id=identity_request,
        terminal_status=terminal_status,
        resources=ordered_resources,
        recommendations=tuple(recommendations),
        blocked_resources=ordered_blocked,
        errors=ordered_errors,
        validation=validation,
        summary=safe_summary,
    )


def _resource_from_success_result(
    *,
    thread_id: str,
    request_id: str,
    resource_type: ResourceFinalV3ResourceKind,
    result: Mapping[str, Any],
) -> ResourceFinalV3Resource:
    status = result.get("status")
    resource_status: ResourceFinalV3ResourceStatus
    if status == "success":
        resource_status = "success"
    elif status == "partial_success":
        resource_status = "partial_success"
    else:
        raise ResourceFinalRuntimeError(
            f"resource {resource_type} does not have a successful status"
        )
    validation = _resource_validation(
        resource_type=resource_type,
        status=resource_status,
        value=result.get("validation"),
    )

    state_updates = _required_mapping(result.get("state_updates"), "state_updates")
    if resource_type == "quiz":
        return _validated_quiz_resource(
            thread_id=thread_id,
            request_id=request_id,
            status=resource_status,
            validation=validation,
            title=_bounded_required_text(
                result.get("title"),
                "quiz.title",
                max_chars=240,
            ),
            state_updates=state_updates,
        )

    artifact = _required_mapping(result.get("artifact"), "artifact")
    artifacts = _mapping_sequence(result.get("artifacts"), "artifacts")
    payload = _payload_for_resource(
        resource_type=resource_type,
        artifact=artifact,
        artifacts=artifacts,
        state_updates=state_updates,
    )
    safe_payload = _sanitize_payload(payload)
    return build_resource_final_v3_resource(
        thread_id=thread_id,
        request_id=request_id,
        kind=resource_type,
        status=resource_status,
        title=_bounded_required_text(
            result.get("title"),
            f"{resource_type}.title",
            max_chars=240,
        ),
        summary=_bounded_required_text(
            result.get("message_content"),
            f"{resource_type}.message_content",
            max_chars=1_200,
        ),
        payload=safe_payload,
        artifact_refs=_collect_artifact_refs(safe_payload),
        validation=validation,
    )


def _validated_quiz_resource(
    *,
    thread_id: str,
    request_id: str,
    status: ResourceFinalV3ResourceStatus,
    validation: ResourceFinalV3ResourceValidation,
    title: str,
    state_updates: Mapping[str, Any],
) -> ResourceFinalV3Quiz:
    value = state_updates.get("exercise_resource_v3")
    if not isinstance(value, Mapping):
        raise ResourceFinalRuntimeError(
            "successful quiz requires state_updates.exercise_resource_v3"
        )
    try:
        resource = ResourceFinalV3Quiz.model_validate_json(
            _mapping_json(value),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResourceFinalRuntimeError(
            "state_updates.exercise_resource_v3 violates ResourceFinalV3Quiz"
        ) from exc
    if resource.status != status:
        raise ResourceFinalRuntimeError("quiz V3 status does not match branch status")
    if resource.title != title:
        raise ResourceFinalRuntimeError("quiz V3 title does not match branch title")
    if resource.validation != validation:
        raise ResourceFinalRuntimeError(
            "quiz V3 validation does not match branch validation"
        )
    expected_id = build_resource_final_v3_resource(
        thread_id=thread_id,
        request_id=request_id,
        kind="quiz",
        status=resource.status,
        title=resource.title,
        summary=resource.summary,
        payload=resource.payload,
        artifact_refs=resource.artifact_refs,
        validation=resource.validation,
    )
    if not isinstance(expected_id, ResourceFinalV3Quiz):
        raise ResourceFinalRuntimeError(
            "quiz V3 builder returned the wrong resource type"
        )
    if resource.resource_id != expected_id.resource_id:
        raise ResourceFinalRuntimeError(
            "quiz V3 resource_id does not match bundle request identity"
        )
    return resource


def _resource_validation(
    *,
    resource_type: ResourceFinalV3ResourceKind,
    status: ResourceFinalV3ResourceStatus,
    value: object,
) -> ResourceFinalV3ResourceValidation:
    if not isinstance(value, Mapping):
        raise ResourceFinalRuntimeError(
            f"successful {resource_type} branch requires validation"
        )
    try:
        validation = ResourceFinalV3ResourceValidation.model_validate_json(
            _mapping_json(value),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResourceFinalRuntimeError(
            f"{resource_type} branch validation cannot satisfy Resource Final V3"
        ) from exc
    if validation.resource_type != resource_type:
        raise ResourceFinalRuntimeError(
            f"{resource_type} branch validation.resource_type mismatch"
        )
    if validation.terminal_status != status:
        raise ResourceFinalRuntimeError(
            f"{resource_type} branch validation.terminal_status mismatch"
        )
    return validation


def _payload_for_resource(
    *,
    resource_type: ResourceFinalV3ResourceKind,
    artifact: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    state_updates: Mapping[str, Any],
) -> Mapping[str, Any]:
    if resource_type == "mindmap":
        payload: dict[str, Any] = {"mindmap": artifact}
        if "mindmap_tree" in state_updates:
            payload["mindmap_tree"] = state_updates["mindmap_tree"]
        return payload
    if resource_type == "review_doc":
        return {
            "review_doc": artifact,
            "review_doc_artifacts": list(artifacts),
        }
    if resource_type == "code_practice":
        return {"code_practice_artifact": artifact}
    if resource_type == "video_script":
        return {"video_script_artifact": artifact}
    if resource_type == "video_animation":
        return {"video_animation_artifact": artifact}
    if resource_type == "study_plan":
        study_plan = dict(artifact)
        if "study_plan_markdown" in state_updates:
            study_plan["markdown"] = state_updates["study_plan_markdown"]
        return {"study_plan": study_plan}
    raise ResourceFinalRuntimeError(
        f"resource payload projection is not registered for {resource_type}"
    )


def _error_from_failed_result(
    *,
    resource_type: ResourceFinalV3ResourceKind,
    result: Mapping[str, Any],
) -> ResourceFinalV3Error:
    return ResourceFinalV3Error(
        resource_type=resource_type,
        error_code=_bounded_required_text(
            result.get("error_code"),
            f"{resource_type}.error_code",
            max_chars=120,
        ),
        error_type=_bounded_required_text(
            result.get("error_type"),
            f"{resource_type}.error_type",
            max_chars=160,
        ),
        message_sanitized=_bounded_required_text(
            result.get("error_message_sanitized"),
            f"{resource_type}.error_message_sanitized",
            max_chars=1_200,
        ),
    )


def _blocked_resource(value: Mapping[str, Any]) -> ResourceFinalV3BlockedResource:
    resource_type = _resource_kind(value.get("resource_type"))
    requirement_ids = value.get("blocked_requirement_ids")
    if not isinstance(requirement_ids, Sequence) or isinstance(
        requirement_ids,
        (str, bytes),
    ):
        raise ResourceFinalRuntimeError(
            f"blocked {resource_type} requires blocked_requirement_ids"
        )
    normalized_ids = tuple(
        _bounded_required_text(
            item,
            f"{resource_type}.blocked_requirement_id",
            max_chars=160,
        )
        for item in requirement_ids
    )
    return ResourceFinalV3BlockedResource(
        resource_type=resource_type,
        status="blocked_insufficient_evidence",
        reason_code=_bounded_required_text(
            value.get("reason_code"),
            f"{resource_type}.reason_code",
            max_chars=120,
        ),
        blocked_requirement_ids=normalized_ids,
    )


def _resource_kind(value: object) -> ResourceFinalV3ResourceKind:
    if value == "mindmap":
        return "mindmap"
    if value == "quiz":
        return "quiz"
    if value == "review_doc":
        return "review_doc"
    if value == "code_practice":
        return "code_practice"
    if value == "video_script":
        return "video_script"
    if value == "video_animation":
        return "video_animation"
    if value == "study_plan":
        return "study_plan"
    raise ResourceFinalRuntimeError(f"unsupported resource type {value!r}")


def _required_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResourceFinalRuntimeError(f"{field_name} must be an object")
    return value


def _mapping_sequence(
    value: object,
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ResourceFinalRuntimeError(f"{field_name} must be an array")
    items: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ResourceFinalRuntimeError(f"{field_name} items must be objects")
        items.append(item)
    return tuple(items)


def _bounded_required_text(
    value: object,
    field_name: str,
    *,
    max_chars: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceFinalRuntimeError(f"{field_name} must be a non-blank string")
    return sanitize_workspace_text(
        value,
        max_chars=max_chars,
        fallback="",
    )


def _sanitize_payload(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    sanitized = _sanitize_json_value(value, depth=0, key="")
    if not isinstance(sanitized, dict) or not sanitized:
        raise ResourceFinalRuntimeError("resource payload is empty after sanitization")
    return sanitized


def _sanitize_json_value(
    value: object,
    *,
    depth: int,
    key: str,
) -> JsonValue | _DropValue:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ResourceFinalRuntimeError(
            "resource payload exceeds maximum nesting depth"
        )
    if isinstance(value, Mapping):
        if len(value) > _MAX_MAPPING_ITEMS:
            raise ResourceFinalRuntimeError(
                "resource payload object exceeds item limit"
            )
        output: dict[str, JsonValue] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ResourceFinalRuntimeError(
                    "resource payload keys must be non-blank strings"
                )
            if raw_key != raw_key.strip() or len(raw_key) > 120:
                raise ResourceFinalRuntimeError("resource payload key is not canonical")
            safe_item = _sanitize_json_value(
                raw_item,
                depth=depth + 1,
                key=raw_key,
            )
            if not isinstance(safe_item, _DropValue):
                output[raw_key] = safe_item
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > _MAX_SEQUENCE_ITEMS:
            raise ResourceFinalRuntimeError("resource payload array exceeds item limit")
        items: list[JsonValue] = []
        for item in value:
            safe_item = _sanitize_json_value(item, depth=depth + 1, key=key)
            if not isinstance(safe_item, _DropValue):
                items.append(safe_item)
        return items
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResourceFinalRuntimeError(
                "resource payload contains non-finite number"
            )
        return value
    if isinstance(value, str):
        if _is_reference_key(key):
            safe_reference = _safe_resource_reference(key, value)
            return safe_reference if safe_reference else _DROP
        max_chars = _MAX_RENDER_TEXT if key in _RENDER_TEXT_KEYS else _MAX_SHORT_TEXT
        return sanitize_workspace_text(value, max_chars=max_chars, fallback="")
    raise ResourceFinalRuntimeError(
        f"resource payload contains unsupported value type {type(value).__name__}"
    )


def _mapping_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _collect_artifact_refs(payload: Mapping[str, JsonValue]) -> dict[str, str]:
    refs: dict[str, str] = {}

    def visit(value: JsonValue, *, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, key=child_key)
        elif isinstance(value, list):
            for item in value:
                visit(item, key=key)
        elif isinstance(value, str) and _is_reference_key(key) and value:
            refs.setdefault(key, value)

    visit(dict(payload))
    return dict(sorted(refs.items()))


def _is_reference_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _REFERENCE_HINTS)


def _safe_resource_reference(key: str, value: str) -> str:
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


__all__ = [
    "ResourceFinalRuntimeError",
    "build_resource_final_v3_from_bundle",
]
