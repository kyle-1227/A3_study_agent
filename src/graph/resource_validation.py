"""Strict validation for renderable resource branch results.

The validation registry is intentionally independent from resource generation
and capability routing. It validates already-produced in-memory artifacts and
never calls LLMs, retrieval systems, or remote URLs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.assessment.checkpoint import (
    AssessmentCheckpointError,
    validate_public_exercise_cards_v1,
)
from src.config import get_setting
from src.graph.study_plan import StudyPlanArtifact, validate_study_plan_artifact
from src.tools.document_tool import (
    get_code_practice_artifact_dir,
    get_exercise_artifact_dir,
    get_review_doc_artifact_dir,
    get_video_script_artifact_dir,
)
from src.tools.mindmap_tool import get_mindmap_artifact_dir
from src.tools.video_animation_tool import get_video_animation_artifact_dir


ResourceTerminalStatus = Literal["success", "partial_success", "failed"]
ReferenceVerification = Literal[
    "verified_local",
    "remote_unverified",
    "invalid",
]
_PRIVATE_ASSESSMENT_KEYS = frozenset(
    {"answer", "explanation", "pitfall", "answer_key", "accepted_answers"}
)


class ResourceValidationResultV1(BaseModel):
    """Content-free result used by workers, bundle aggregation, and traces."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["resource_validation_v1"]
    resource_type: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    valid: bool
    terminal_status: ResourceTerminalStatus
    renderable_count: int = Field(ge=0, le=10_000)
    downloadable_count: int = Field(ge=0, le=10_000)
    verified_local_count: int = Field(ge=0, le=10_000)
    remote_unverified_count: int = Field(ge=0, le=10_000)
    failure_reason: str = Field(default="", pattern=r"^(|[a-z][a-z0-9_.-]{0,79})$")
    warnings: tuple[str, ...] = Field(default_factory=tuple, max_length=24)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "ResourceValidationResultV1":
        if self.valid != (self.terminal_status != "failed"):
            raise ValueError(
                "valid must be false exactly when terminal_status is failed"
            )
        if self.terminal_status == "failed" and not self.failure_reason:
            raise ValueError("failed validation requires failure_reason")
        if self.terminal_status != "failed" and self.renderable_count <= 0:
            raise ValueError("successful validation requires a renderable result")
        if (
            self.verified_local_count + self.remote_unverified_count
            > self.downloadable_count
        ):
            raise ValueError("reference counts exceed downloadable_count")
        return self


class _ReferenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    verified_local_count: int = 0
    remote_unverified_count: int = 0
    invalid_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def downloadable_count(self) -> int:
        return self.verified_local_count + self.remote_unverified_count


ResourceValidator = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]],
    ResourceValidationResultV1,
]


def validate_renderable_resource_result(
    resource_type: str,
    artifact: Mapping[str, Any] | None,
    artifacts: Sequence[Mapping[str, Any]] | None,
    state_updates: Mapping[str, Any] | None,
) -> ResourceValidationResultV1:
    """Validate one resource through the independent validator registry."""

    validator = RESOURCE_VALIDATORS.get(str(resource_type or ""))
    if validator is None:
        raise ValueError(f"resource validator is not registered: {resource_type}")
    return validator(
        artifact or {},
        tuple(item for item in (artifacts or ()) if isinstance(item, Mapping)),
        state_updates or {},
    )


def _mindmap_validator(
    artifact: Mapping[str, Any],
    _artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    tree = artifact.get("tree") or state.get("mindmap_tree")
    node_count = _valid_mindmap_node_count(tree)
    references = _validate_references(
        (artifact,),
        root_resolver=get_mindmap_artifact_dir,
        field_pairs=(("xmind_url", "filename"),),
        allowed_suffixes=_allowed_suffixes("mindmap"),
    )
    inline_count = 1 if node_count >= 2 else 0
    warnings = list(references.warnings)
    if node_count == 1:
        warnings.append("mindmap.root_only")
    return _result(
        "mindmap",
        inline_count=inline_count,
        references=references,
        degraded=node_count == 1,
        warnings=warnings,
    )


def _quiz_validator(
    artifact: Mapping[str, Any],
    _artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    items = state.get("exercise_items") or artifact.get("items") or []
    references = _validate_references(
        (artifact,),
        root_resolver=get_exercise_artifact_dir,
        field_pairs=(("markdown_url", "filename"), ("docx_url", "docx_filename")),
        allowed_suffixes=_allowed_suffixes("quiz"),
    )
    if _contains_private_assessment_key(artifact):
        return _validation_model(
            "quiz",
            terminal_status="failed",
            renderable_count=0,
            references=references,
            failure_reason="quiz.private_answer_exposed",
            warnings=("quiz.private_answer_exposed",),
        )
    try:
        cards = validate_public_exercise_cards_v1(items)
        artifact_cards = validate_public_exercise_cards_v1(artifact.get("items"))
    except AssessmentCheckpointError:
        return _validation_model(
            "quiz",
            terminal_status="failed",
            renderable_count=0,
            references=references,
            failure_reason="quiz.public_cards_invalid",
            warnings=("quiz.public_cards_invalid",),
        )
    public_items = [card.model_dump(mode="json") for card in cards]
    public_artifact_items = [card.model_dump(mode="json") for card in artifact_cards]
    if public_items != public_artifact_items:
        return _validation_model(
            "quiz",
            terminal_status="failed",
            renderable_count=0,
            references=references,
            failure_reason="quiz.public_cards_mismatch",
            warnings=("quiz.public_cards_mismatch",),
        )
    return _result(
        "quiz",
        inline_count=1,
        references=references,
    )


def _contains_private_assessment_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _PRIVATE_ASSESSMENT_KEYS
            or _contains_private_assessment_key(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_contains_private_assessment_key(item) for item in value)
    return False


def _review_doc_validator(
    artifact: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    documents = tuple(artifacts) or (artifact,)
    references = _validate_references(
        documents,
        root_resolver=get_review_doc_artifact_dir,
        field_pairs=(("markdown_url", "filename"), ("docx_url", "docx_filename")),
        allowed_suffixes=_allowed_suffixes("review_doc"),
    )
    valid_document_count = sum(
        1 for item in documents if _nonempty_text(item.get("markdown"))
    )
    if not valid_document_count and _nonempty_text(state.get("review_doc_markdown")):
        valid_document_count = 1
    invalid_document_count = sum(
        1
        for item in documents
        if not _nonempty_text(item.get("markdown")) and not _has_reference(item)
    )
    return _result(
        "review_doc",
        inline_count=valid_document_count,
        references=references,
        degraded=invalid_document_count > 0,
        warnings=("review_doc.invalid_documents",) if invalid_document_count else (),
    )


def _code_practice_validator(
    artifact: Mapping[str, Any],
    _artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    references = _validate_references(
        (artifact,),
        root_resolver=get_code_practice_artifact_dir,
        field_pairs=(
            ("markdown_url", "filename"),
            ("docx_url", "docx_filename"),
            ("python_url", "python_filename"),
        ),
        allowed_suffixes=_allowed_suffixes("code_practice"),
    )
    return _result(
        "code_practice",
        inline_count=1
        if _nonempty_text(
            state.get("code_practice_markdown") or artifact.get("markdown")
        )
        else 0,
        references=references,
    )


def _video_script_validator(
    artifact: Mapping[str, Any],
    _artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    references = _validate_references(
        (artifact,),
        root_resolver=get_video_script_artifact_dir,
        field_pairs=(
            ("markdown_url", "filename"),
            ("docx_url", "docx_filename"),
            ("srt_url", "srt_filename"),
        ),
        allowed_suffixes=_allowed_suffixes("video_script"),
    )
    return _result(
        "video_script",
        inline_count=1
        if _nonempty_text(
            state.get("video_script_markdown") or artifact.get("markdown")
        )
        else 0,
        references=references,
    )


def _video_animation_validator(
    artifact: Mapping[str, Any],
    _artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    references = _validate_references(
        (artifact,),
        root_resolver=get_video_animation_artifact_dir,
        field_pairs=(
            ("mp4_url", "mp4_filename"),
            ("html_url", "html_filename"),
            ("json_url", "json_filename"),
            ("srt_url", "srt_filename"),
        ),
        allowed_suffixes=_allowed_suffixes("video_animation"),
    )
    verified_mp4 = _verified_local_reference(
        artifact,
        root_resolver=get_video_animation_artifact_dir,
        url_field="mp4_url",
        filename_field="mp4_filename",
        allowed_suffixes=frozenset({".mp4"}),
    )
    full_duration = int(artifact.get("full_duration_seconds") or 0)
    render_duration = int(artifact.get("render_duration_seconds") or 0)
    formal_full_video = bool(
        artifact.get("render_success") is True
        and verified_mp4
        and artifact.get("render_mode") == "production"
        and artifact.get("is_preview_video") is False
        and artifact.get("video_valid_for_teaching") is True
        and full_duration > 0
        and render_duration == full_duration
    )
    if formal_full_video:
        return _validation_model(
            "video_animation",
            terminal_status="success",
            renderable_count=max(1, references.downloadable_count),
            references=references,
        )
    preview_inline = 1 if _nonempty_text(state.get("video_animation_html")) else 0
    preview_references = references.downloadable_count
    if preview_inline or preview_references:
        return _validation_model(
            "video_animation",
            terminal_status="partial_success",
            renderable_count=preview_inline + preview_references,
            references=references,
            warnings=(*references.warnings, "video_animation.mp4_unavailable"),
        )
    return _validation_model(
        "video_animation",
        terminal_status="failed",
        renderable_count=0,
        references=references,
        failure_reason="video_animation.no_renderable_artifact",
        warnings=references.warnings,
    )


def _study_plan_validator(
    artifact: Mapping[str, Any],
    _artifacts: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> ResourceValidationResultV1:
    document = artifact.get("document")
    document_artifact = document if isinstance(document, Mapping) else {}
    references = _validate_references(
        (document_artifact,),
        root_resolver=get_review_doc_artifact_dir,
        field_pairs=(("markdown_url", "filename"), ("docx_url", "docx_filename")),
        allowed_suffixes=_allowed_suffixes("study_plan"),
    )
    try:
        plan_payload = {
            key: value for key, value in artifact.items() if key != "document"
        }
        parsed = StudyPlanArtifact.model_validate(plan_payload)
    except Exception:
        parsed = None
    business_error = (
        validate_study_plan_artifact(parsed) if parsed is not None else "invalid"
    )
    markdown_present = _nonempty_text(state.get("study_plan_markdown"))
    if business_error or not markdown_present:
        return _validation_model(
            "study_plan",
            terminal_status="failed",
            renderable_count=0,
            references=references,
            failure_reason="study_plan.business_validation_failed",
            warnings=references.warnings,
        )
    return _result(
        "study_plan",
        inline_count=1,
        references=references,
    )


RESOURCE_VALIDATORS: dict[str, ResourceValidator] = {
    "mindmap": _mindmap_validator,
    "quiz": _quiz_validator,
    "review_doc": _review_doc_validator,
    "code_practice": _code_practice_validator,
    "video_script": _video_script_validator,
    "video_animation": _video_animation_validator,
    "study_plan": _study_plan_validator,
}


def _result(
    resource_type: str,
    *,
    inline_count: int,
    references: _ReferenceSummary,
    degraded: bool = False,
    warnings: Sequence[str] = (),
) -> ResourceValidationResultV1:
    renderable_count = inline_count + references.downloadable_count
    all_warnings = tuple(dict.fromkeys((*warnings, *references.warnings)))
    if renderable_count <= 0:
        return _validation_model(
            resource_type,
            terminal_status="failed",
            renderable_count=0,
            references=references,
            failure_reason=f"{resource_type}.no_renderable_artifact",
            warnings=all_warnings,
        )
    remote_only = inline_count == 0 and references.verified_local_count == 0
    terminal_status: ResourceTerminalStatus = (
        "partial_success"
        if degraded or remote_only or references.invalid_count
        else "success"
    )
    return _validation_model(
        resource_type,
        terminal_status=terminal_status,
        renderable_count=renderable_count,
        references=references,
        warnings=all_warnings,
    )


def _validation_model(
    resource_type: str,
    *,
    terminal_status: ResourceTerminalStatus,
    renderable_count: int,
    references: _ReferenceSummary,
    failure_reason: str = "",
    warnings: Sequence[str] = (),
) -> ResourceValidationResultV1:
    return ResourceValidationResultV1(
        schema_version="resource_validation_v1",
        resource_type=resource_type,
        valid=terminal_status != "failed",
        terminal_status=terminal_status,
        renderable_count=renderable_count,
        downloadable_count=references.downloadable_count,
        verified_local_count=references.verified_local_count,
        remote_unverified_count=references.remote_unverified_count,
        failure_reason=failure_reason,
        warnings=tuple(dict.fromkeys(warnings))[:24],
    )


def _validate_references(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    root_resolver: Callable[[], Path],
    field_pairs: Sequence[tuple[str, str]],
    allowed_suffixes: frozenset[str],
) -> _ReferenceSummary:
    verified = 0
    remote = 0
    invalid = 0
    warnings: list[str] = []
    for artifact in artifacts:
        for url_field, filename_field in field_pairs:
            if not artifact.get(url_field) and not artifact.get(filename_field):
                continue
            verification = _validate_reference(
                artifact,
                root_resolver=root_resolver,
                url_field=url_field,
                filename_field=filename_field,
                allowed_suffixes=allowed_suffixes,
            )
            if verification == "verified_local":
                verified += 1
            elif verification == "remote_unverified":
                remote += 1
                warnings.append("artifact.remote_unverified")
            else:
                invalid += 1
                warnings.append("artifact.invalid_reference")
    return _ReferenceSummary(
        verified_local_count=verified,
        remote_unverified_count=remote,
        invalid_count=invalid,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_reference(
    artifact: Mapping[str, Any],
    *,
    root_resolver: Callable[[], Path],
    url_field: str,
    filename_field: str,
    allowed_suffixes: frozenset[str],
) -> ReferenceVerification:
    if _verified_local_reference(
        artifact,
        root_resolver=root_resolver,
        url_field=url_field,
        filename_field=filename_field,
        allowed_suffixes=allowed_suffixes,
    ):
        return "verified_local"
    raw_url = str(artifact.get(url_field) or "").strip()
    if not raw_url:
        return "invalid"
    parsed = urlsplit(raw_url)
    allowed_schemes = _allowed_remote_schemes()
    if parsed.scheme.lower() not in allowed_schemes or not parsed.hostname:
        return "invalid"
    if parsed.username or parsed.password:
        return "invalid"
    sensitive_keys = _sensitive_query_keys()
    if any(
        any(marker in key.casefold() for marker in sensitive_keys)
        for key, _ in parse_qsl(parsed.query)
    ):
        return "invalid"
    suffix = Path(parsed.path).suffix.casefold()
    return "remote_unverified" if suffix in allowed_suffixes else "invalid"


def _verified_local_reference(
    artifact: Mapping[str, Any],
    *,
    root_resolver: Callable[[], Path],
    url_field: str,
    filename_field: str,
    allowed_suffixes: frozenset[str],
) -> bool:
    artifact_id = str(artifact.get("artifact_id") or "").strip()
    filename = str(artifact.get(filename_field) or "").strip()
    raw_url = str(artifact.get(url_field) or "").strip()
    if not filename and raw_url:
        parsed = urlsplit(raw_url)
        if not parsed.scheme:
            filename = Path(parsed.path).name
    if not artifact_id or not filename or Path(filename).name != filename:
        return False
    if Path(filename).suffix.casefold() not in allowed_suffixes:
        return False
    root = root_resolver().resolve()
    candidate = (root / artifact_id / filename).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    if not candidate.is_file() or candidate.is_symlink():
        return False
    size = candidate.stat().st_size
    return 0 < size <= _max_file_bytes()


def _allowed_suffixes(resource_type: str) -> frozenset[str]:
    value = get_setting(f"resource_validation.allowed_suffixes.{resource_type}", None)
    if not isinstance(value, list) or not value:
        raise RuntimeError(
            f"missing resource validation suffix config: {resource_type}"
        )
    normalized = frozenset(
        str(item).strip().casefold() for item in value if str(item).strip()
    )
    if not normalized or any(not item.startswith(".") for item in normalized):
        raise RuntimeError(
            f"invalid resource validation suffix config: {resource_type}"
        )
    return normalized


def _allowed_remote_schemes() -> frozenset[str]:
    value = get_setting("resource_validation.allowed_remote_schemes", None)
    if not isinstance(value, list) or not value:
        raise RuntimeError("missing resource_validation.allowed_remote_schemes")
    schemes = frozenset(
        str(item).strip().casefold() for item in value if str(item).strip()
    )
    if not schemes.issubset({"http", "https"}):
        raise RuntimeError("resource validation remote schemes must be http/https")
    return schemes


def _sensitive_query_keys() -> frozenset[str]:
    value = get_setting("resource_validation.sensitive_query_keys", None)
    if not isinstance(value, list) or not value:
        raise RuntimeError("missing resource_validation.sensitive_query_keys")
    return frozenset(
        str(item).strip().casefold() for item in value if str(item).strip()
    )


def _max_file_bytes() -> int:
    value = get_setting("resource_validation.max_file_bytes", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(
            "resource_validation.max_file_bytes must be a positive integer"
        )
    return value


def _valid_mindmap_node_count(value: Any) -> int:
    if not isinstance(value, Mapping) or not _nonempty_text(value.get("title")):
        return 0
    children = value.get("children")
    if not isinstance(children, Sequence) or isinstance(children, str | bytes):
        return 1
    return 1 + sum(_valid_mindmap_node_count(child) for child in children)


def _has_reference(value: Mapping[str, Any]) -> bool:
    return any(
        _nonempty_text(value.get(key))
        for key in (
            "markdown_url",
            "docx_url",
            "xmind_url",
            "python_url",
            "srt_url",
            "html_url",
            "json_url",
            "mp4_url",
        )
    )


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "RESOURCE_VALIDATORS",
    "ResourceTerminalStatus",
    "ResourceValidationResultV1",
    "validate_renderable_resource_result",
]
