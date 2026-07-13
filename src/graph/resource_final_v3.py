"""Strict, migration-ready Resource Final V3 contract.

This module is deliberately not wired into the active SSE/runtime path yet.
It defines the target contract and deterministic builders so persisted V1/V2
payloads can be migrated and producers can be switched only after their
replacement gates pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

from src.assessment.checkpoint import validate_public_exercise_cards_v1


RESOURCE_FINAL_V3_SCHEMA_VERSION = "resource_final_v3"
RESOURCE_FINAL_V3_PAYLOAD_HASH_PREFIX = "payload:v3"
RESOURCE_FINAL_V3_RESOURCE_ID_PREFIX = "resource:v3"
RESOURCE_FINAL_V3_ID_PREFIX = "resource-final:v3"

ResourceFinalV3TerminalStatus: TypeAlias = Literal[
    "success",
    "partial_success",
    "failed",
    "controlled_stop",
]
ResourceFinalV3ResourceStatus: TypeAlias = Literal["success", "partial_success"]
ResourceFinalV3ResourceKind: TypeAlias = Literal[
    "mindmap",
    "quiz",
    "review_doc",
    "code_practice",
    "video_script",
    "video_animation",
    "study_plan",
]

_NONEMPTY_ID_PATTERN = r"^\S(?:.*\S)?$"
_ERROR_CODE_PATTERN = r"^[a-z][a-z0-9_.-]{0,119}$"
_HASH_PATTERN = rf"^{RESOURCE_FINAL_V3_PAYLOAD_HASH_PREFIX}:[a-f0-9]{{64}}$"
_RESOURCE_ID_PATTERN = rf"^{RESOURCE_FINAL_V3_RESOURCE_ID_PREFIX}:[a-f0-9]{{64}}$"
_FINAL_ID_PATTERN = rf"^{RESOURCE_FINAL_V3_ID_PREFIX}:[a-f0-9]{{64}}$"


class ResourceFinalV3ResourceValidation(BaseModel):
    """Validation truth attached to one renderable resource."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["resource_validation_v1"]
    resource_type: ResourceFinalV3ResourceKind
    valid: Literal[True]
    terminal_status: ResourceFinalV3ResourceStatus
    renderable_count: int = Field(ge=1, le=10_000)
    downloadable_count: int = Field(ge=0, le=10_000)
    verified_local_count: int = Field(ge=0, le=10_000)
    remote_unverified_count: int = Field(ge=0, le=10_000)
    failure_reason: Literal[""]
    warnings: tuple[str, ...] = Field(max_length=24)

    @model_validator(mode="after")
    def validate_reference_counts(self) -> ResourceFinalV3ResourceValidation:
        if (
            self.verified_local_count + self.remote_unverified_count
            > self.downloadable_count
        ):
            raise ValueError("reference counts exceed downloadable_count")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("validation warnings must not contain blank values")
        return self


class _ResourceFinalV3ResourceBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    expected_payload_keys: ClassVar[frozenset[str]]

    kind: ResourceFinalV3ResourceKind
    status: ResourceFinalV3ResourceStatus
    resource_id: str = Field(pattern=_RESOURCE_ID_PATTERN)
    payload_hash: str = Field(pattern=_HASH_PATTERN)
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=1_200)
    payload: dict[str, JsonValue]
    artifact_refs: dict[str, str] = Field(max_length=80)
    validation: ResourceFinalV3ResourceValidation

    @model_validator(mode="after")
    def validate_resource_truth(self) -> _ResourceFinalV3ResourceBase:
        if not self.title.strip() or not self.summary.strip():
            raise ValueError("resource title and summary must not be blank")
        populated_payload_keys = {
            key
            for key in self.expected_payload_keys
            if self.payload.get(key) not in (None, "", [], {})
        }
        if not populated_payload_keys:
            expected = ", ".join(sorted(self.expected_payload_keys))
            raise ValueError(
                f"{self.kind} payload requires at least one non-empty value for: "
                f"{expected}"
            )
        if self.validation.resource_type != self.kind:
            raise ValueError("resource kind must match validation.resource_type")
        if self.validation.terminal_status != self.status:
            raise ValueError("resource status must match validation.terminal_status")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.artifact_refs.items()
        ):
            raise ValueError("artifact_refs must contain non-blank keys and values")
        expected_hash = stable_resource_final_v3_resource_hash(self)
        if self.payload_hash != expected_hash:
            raise ValueError("resource payload_hash does not match canonical content")
        return self


class ResourceFinalV3Mindmap(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"mindmap", "mindmap_artifact", "mindmap_tree"})
    kind: Literal["mindmap"]


class ResourceFinalV3Quiz(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"exercise_artifact", "exercise_items"})
    kind: Literal["quiz"]

    @model_validator(mode="after")
    def validate_public_quiz_payload(self) -> ResourceFinalV3Quiz:
        if set(self.payload) != self.expected_payload_keys:
            raise ValueError(
                "quiz payload must contain only exercise_artifact and exercise_items"
            )
        artifact = self.payload.get("exercise_artifact")
        if not isinstance(artifact, dict):
            raise ValueError("quiz exercise_artifact must be an object")
        if set(artifact) != {"schema_version", "title", "items"}:
            raise ValueError("quiz exercise_artifact contains unsupported fields")
        if artifact.get("schema_version") != "exercise_public_artifact_v1":
            raise ValueError("quiz exercise_artifact schema_version is invalid")
        if artifact.get("title") != self.title:
            raise ValueError("quiz exercise_artifact title must match resource title")
        cards = validate_public_exercise_cards_v1(self.payload.get("exercise_items"))
        artifact_cards = validate_public_exercise_cards_v1(artifact.get("items"))
        if cards != artifact_cards:
            raise ValueError(
                "quiz exercise_artifact items must match exercise_items exactly"
            )
        return self


class ResourceFinalV3ReviewDocument(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"review_doc", "review_doc_artifacts"})
    kind: Literal["review_doc"]


class ResourceFinalV3CodePractice(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"code_practice_artifact"})
    kind: Literal["code_practice"]


class ResourceFinalV3VideoScript(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"video_script_artifact"})
    kind: Literal["video_script"]


class ResourceFinalV3VideoAnimation(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"video_animation_artifact"})
    kind: Literal["video_animation"]


class ResourceFinalV3StudyPlan(_ResourceFinalV3ResourceBase):
    expected_payload_keys = frozenset({"study_plan"})
    kind: Literal["study_plan"]


ResourceFinalV3Resource: TypeAlias = Annotated[
    ResourceFinalV3Mindmap
    | ResourceFinalV3Quiz
    | ResourceFinalV3ReviewDocument
    | ResourceFinalV3CodePractice
    | ResourceFinalV3VideoScript
    | ResourceFinalV3VideoAnimation
    | ResourceFinalV3StudyPlan,
    Field(discriminator="kind"),
]
_RESOURCE_ADAPTER = TypeAdapter(ResourceFinalV3Resource)


class ResourceFinalV3Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    recommendation_id: str = Field(pattern=_NONEMPTY_ID_PATTERN, max_length=160)
    resource_type: ResourceFinalV3ResourceKind
    trigger: Literal["automatic", "explicit_request"]
    rank: int = Field(ge=1, le=100)
    title: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=1_200)

    @model_validator(mode="after")
    def validate_non_blank_text(self) -> ResourceFinalV3Recommendation:
        if not self.title.strip() or not self.reason.strip():
            raise ValueError("recommendation title and reason must not be blank")
        return self


class ResourceFinalV3BlockedResource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    resource_type: ResourceFinalV3ResourceKind
    status: Literal["blocked_insufficient_evidence"]
    reason_code: str = Field(pattern=_ERROR_CODE_PATTERN)
    blocked_requirement_ids: tuple[str, ...] = Field(max_length=80)

    @model_validator(mode="after")
    def validate_requirement_ids(self) -> ResourceFinalV3BlockedResource:
        if any(not value.strip() for value in self.blocked_requirement_ids):
            raise ValueError("blocked_requirement_ids must not contain blank values")
        if len(set(self.blocked_requirement_ids)) != len(self.blocked_requirement_ids):
            raise ValueError("blocked_requirement_ids must be unique")
        return self


class ResourceFinalV3Error(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    resource_type: ResourceFinalV3ResourceKind
    error_code: str = Field(pattern=_ERROR_CODE_PATTERN)
    error_type: str = Field(pattern=_NONEMPTY_ID_PATTERN, max_length=160)
    message_sanitized: str = Field(min_length=1, max_length=1_200)

    @model_validator(mode="after")
    def validate_non_blank_message(self) -> ResourceFinalV3Error:
        if not self.error_type.strip() or not self.message_sanitized.strip():
            raise ValueError("error_type and message_sanitized must not be blank")
        return self


class ResourceFinalV3Validation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["resource_final_validation_v3"]
    resource_count: int = Field(ge=0, le=10_000)
    success_count: int = Field(ge=0, le=10_000)
    partial_success_count: int = Field(ge=0, le=10_000)
    failed_count: int = Field(ge=0, le=10_000)
    blocked_count: int = Field(ge=0, le=10_000)
    renderable_count: int = Field(ge=0, le=10_000)
    downloadable_count: int = Field(ge=0, le=10_000)


class ResourceFinalV3Content(BaseModel):
    """Canonical Resource Final V3 body before its derived id/hash."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["resource_final_v3"]
    type: Literal["resource_final"]
    thread_id: str = Field(pattern=_NONEMPTY_ID_PATTERN, max_length=160)
    request_id: str = Field(pattern=_NONEMPTY_ID_PATTERN, max_length=160)
    terminal_status: ResourceFinalV3TerminalStatus
    resources: tuple[ResourceFinalV3Resource, ...] = Field(max_length=80)
    recommendations: tuple[ResourceFinalV3Recommendation, ...] = Field(max_length=80)
    blocked_resources: tuple[ResourceFinalV3BlockedResource, ...] = Field(max_length=80)
    errors: tuple[ResourceFinalV3Error, ...] = Field(max_length=80)
    validation: ResourceFinalV3Validation
    summary: str = Field(min_length=1, max_length=1_200)

    @model_validator(mode="after")
    def validate_terminal_truth(self) -> ResourceFinalV3Content:
        if not self.thread_id.strip() or not self.request_id.strip():
            raise ValueError("thread_id and request_id must not be blank")
        if not self.summary.strip():
            raise ValueError("summary must not be blank")

        success_count = sum(item.status == "success" for item in self.resources)
        partial_count = sum(item.status == "partial_success" for item in self.resources)
        expected_counts = {
            "resource_count": len(self.resources),
            "success_count": success_count,
            "partial_success_count": partial_count,
            "failed_count": len(self.errors),
            "blocked_count": len(self.blocked_resources),
            "renderable_count": sum(
                item.validation.renderable_count for item in self.resources
            ),
            "downloadable_count": sum(
                item.validation.downloadable_count for item in self.resources
            ),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self.validation, field_name) != expected:
                raise ValueError(
                    f"validation.{field_name} must equal observed value {expected}"
                )

        if self.terminal_status == "success":
            if not self.resources:
                raise ValueError("success requires at least one real resource")
            if partial_count or self.errors or self.blocked_resources:
                raise ValueError(
                    "success cannot contain partial, failed, or blocked work"
                )
        elif self.terminal_status == "partial_success":
            if not self.resources:
                raise ValueError("partial_success requires at least one real resource")
            if not (partial_count or self.errors or self.blocked_resources):
                raise ValueError(
                    "partial_success requires partial, failed, or blocked work"
                )
        elif self.terminal_status == "failed":
            if self.resources:
                raise ValueError("failed cannot contain renderable resources")
            if not self.errors:
                raise ValueError("failed requires at least one typed error")
        elif self.terminal_status == "controlled_stop":
            if self.resources or self.errors:
                raise ValueError("controlled_stop cannot contain resources or errors")
            if not self.blocked_resources:
                raise ValueError("controlled_stop requires blocked resource evidence")

        resource_ids = [item.resource_id for item in self.resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource_id values must be unique")
        recommendation_ids = [item.recommendation_id for item in self.recommendations]
        if len(set(recommendation_ids)) != len(recommendation_ids):
            raise ValueError("recommendation_id values must be unique")
        recommendation_ranks = [item.rank for item in self.recommendations]
        if len(set(recommendation_ranks)) != len(recommendation_ranks):
            raise ValueError("recommendation ranks must be unique")

        for resource in self.resources:
            expected_id = stable_resource_final_v3_resource_id(
                thread_id=self.thread_id,
                request_id=self.request_id,
                resource_type=resource.kind,
                payload_hash=resource.payload_hash,
            )
            if resource.resource_id != expected_id:
                raise ValueError(
                    "resource_id does not match request identity and payload"
                )
        return self


class ResourceFinalV3(ResourceFinalV3Content):
    """Authoritative Resource Final V3 payload with verified derived identity."""

    resource_final_id: str = Field(pattern=_FINAL_ID_PATTERN)
    payload_hash: str = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_derived_identity(self) -> ResourceFinalV3:
        expected_hash = stable_resource_final_v3_hash(self)
        if self.payload_hash != expected_hash:
            raise ValueError("payload_hash does not match canonical Resource Final V3")
        expected_id = stable_resource_final_v3_id(
            thread_id=self.thread_id,
            request_id=self.request_id,
            payload_hash=self.payload_hash,
        )
        if self.resource_final_id != expected_id:
            raise ValueError(
                "resource_final_id does not match request identity and payload"
            )
        return self


def build_resource_final_v3_resource(
    *,
    thread_id: str,
    request_id: str,
    kind: ResourceFinalV3ResourceKind,
    status: ResourceFinalV3ResourceStatus,
    title: str,
    summary: str,
    payload: Mapping[str, JsonValue],
    artifact_refs: Mapping[str, str],
    validation: ResourceFinalV3ResourceValidation,
) -> ResourceFinalV3Resource:
    """Build one discriminated V3 resource and derive its stable identity."""

    unsigned = {
        "kind": kind,
        "status": status,
        "title": title,
        "summary": summary,
        "payload": dict(payload),
        "artifact_refs": dict(artifact_refs),
        "validation": validation,
    }
    payload_hash = _stable_hash(
        RESOURCE_FINAL_V3_PAYLOAD_HASH_PREFIX,
        _jsonable(unsigned),
    )
    resource_id = stable_resource_final_v3_resource_id(
        thread_id=thread_id,
        request_id=request_id,
        resource_type=kind,
        payload_hash=payload_hash,
    )
    return _RESOURCE_ADAPTER.validate_python(
        {
            **unsigned,
            "resource_id": resource_id,
            "payload_hash": payload_hash,
        },
        strict=True,
    )


def build_resource_final_v3(
    *,
    thread_id: str,
    request_id: str,
    terminal_status: ResourceFinalV3TerminalStatus,
    resources: Sequence[ResourceFinalV3Resource],
    recommendations: Sequence[ResourceFinalV3Recommendation],
    blocked_resources: Sequence[ResourceFinalV3BlockedResource],
    errors: Sequence[ResourceFinalV3Error],
    validation: ResourceFinalV3Validation,
    summary: str,
) -> ResourceFinalV3:
    """Validate terminal truth and build a stable Resource Final V3 payload."""

    content = ResourceFinalV3Content(
        schema_version=RESOURCE_FINAL_V3_SCHEMA_VERSION,
        type="resource_final",
        thread_id=thread_id,
        request_id=request_id,
        terminal_status=terminal_status,
        resources=tuple(resources),
        recommendations=tuple(recommendations),
        blocked_resources=tuple(blocked_resources),
        errors=tuple(errors),
        validation=validation,
        summary=summary,
    )
    payload_hash = stable_resource_final_v3_hash(content)
    resource_final_id = stable_resource_final_v3_id(
        thread_id=thread_id,
        request_id=request_id,
        payload_hash=payload_hash,
    )
    return ResourceFinalV3(
        **content.model_dump(),
        resource_final_id=resource_final_id,
        payload_hash=payload_hash,
    )


def stable_resource_final_v3_resource_hash(
    resource: _ResourceFinalV3ResourceBase,
) -> str:
    payload = resource.model_dump(
        mode="json",
        exclude={"resource_id", "payload_hash"},
    )
    return _stable_hash(RESOURCE_FINAL_V3_PAYLOAD_HASH_PREFIX, payload)


def stable_resource_final_v3_hash(
    value: ResourceFinalV3Content | Mapping[str, Any],
) -> str:
    if isinstance(value, ResourceFinalV3Content):
        payload = value.model_dump(
            mode="json",
            exclude={"resource_final_id", "payload_hash"},
        )
    else:
        content = ResourceFinalV3Content.model_validate(value, strict=True)
        payload = content.model_dump(mode="json")
    return _stable_hash(RESOURCE_FINAL_V3_PAYLOAD_HASH_PREFIX, payload)


def stable_resource_final_v3_resource_id(
    *,
    thread_id: str,
    request_id: str,
    resource_type: ResourceFinalV3ResourceKind,
    payload_hash: str,
) -> str:
    return _stable_hash(
        RESOURCE_FINAL_V3_RESOURCE_ID_PREFIX,
        {
            "thread_id": thread_id,
            "request_id": request_id,
            "resource_type": resource_type,
            "payload_hash": payload_hash,
        },
    )


def stable_resource_final_v3_id(
    *, thread_id: str, request_id: str, payload_hash: str
) -> str:
    return _stable_hash(
        RESOURCE_FINAL_V3_ID_PREFIX,
        {
            "thread_id": thread_id,
            "request_id": request_id,
            "payload_hash": payload_hash,
        },
    )


def _jsonable(value: Any) -> JsonValue:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=_dump)
    )


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _stable_hash(prefix: str, payload: JsonValue) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


__all__ = [
    "RESOURCE_FINAL_V3_SCHEMA_VERSION",
    "ResourceFinalV3",
    "ResourceFinalV3BlockedResource",
    "ResourceFinalV3CodePractice",
    "ResourceFinalV3Content",
    "ResourceFinalV3Error",
    "ResourceFinalV3Mindmap",
    "ResourceFinalV3Quiz",
    "ResourceFinalV3Recommendation",
    "ResourceFinalV3Resource",
    "ResourceFinalV3ResourceKind",
    "ResourceFinalV3ResourceStatus",
    "ResourceFinalV3ResourceValidation",
    "ResourceFinalV3ReviewDocument",
    "ResourceFinalV3StudyPlan",
    "ResourceFinalV3TerminalStatus",
    "ResourceFinalV3Validation",
    "ResourceFinalV3VideoAnimation",
    "ResourceFinalV3VideoScript",
    "build_resource_final_v3",
    "build_resource_final_v3_resource",
    "stable_resource_final_v3_hash",
    "stable_resource_final_v3_id",
    "stable_resource_final_v3_resource_hash",
    "stable_resource_final_v3_resource_id",
]
