"""Study-plan resource-generation nodes."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Literal, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import get_setting
from src.context_engineering.workspace import (
    build_workspace_profile_completion_update,
    sanitize_workspace_text,
    workspace_trace_payload,
)
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.learning_guidance import learner_path_provider_projection_from_state
from src.graph.state import LearningState
from src.llm.structured_output import (
    StructuredLLMResult,
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import create_markdown_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)

PROFILE_COMPLETION_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "key": "learning_goal",
        "label": "学习目标",
        "required": True,
        "max_chars": 400,
    },
    {
        "key": "current_foundation",
        "label": "当前基础",
        "required": True,
        "max_chars": 400,
    },
    {
        "key": "daily_study_time",
        "label": "每天可学习时间",
        "required": True,
        "max_chars": 200,
    },
    {
        "key": "deadline",
        "label": "考试/截止时间",
        "required": False,
        "max_chars": 200,
    },
    {
        "key": "preferred_learning_style",
        "label": "偏好的学习方式",
        "required": False,
        "max_chars": 400,
    },
    {
        "key": "weak_points",
        "label": "薄弱点",
        "required": False,
        "max_chars": 500,
    },
)
PROFILE_COMPLETION_LABELS = {
    str(field["key"]): str(field["label"]) for field in PROFILE_COMPLETION_FIELDS
}
PROFILE_COMPLETION_REQUIRED_KEYS = tuple(
    str(field["key"]) for field in PROFILE_COMPLETION_FIELDS if field["required"]
)


class StudyPlanEmotionalProfile(BaseModel):
    """Learner emotional and workload context for study-plan generation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str = Field(..., min_length=1)
    workload_risk: Literal["low", "medium", "high"]
    motivation_state: str = Field(..., min_length=1)
    support_suggestions: list[str] = Field(default_factory=list)


class StudyPlanPhase(BaseModel):
    """A phase in a personalized study plan."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(..., min_length=1)
    duration: str = Field(..., min_length=1)
    goals: list[str] = Field(..., min_length=1)
    tasks: list[str] = Field(..., min_length=1)
    resources: list[str] = Field(..., min_length=1)
    practice: list[str] = Field(..., min_length=1)
    checkpoints: list[str] = Field(..., min_length=1)


class StudyPlanArtifact(BaseModel):
    """Structured personalized study-plan artifact."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(..., min_length=1)
    learner_profile_summary: str = Field(..., min_length=1)
    overall_goal: str = Field(..., min_length=1)
    phases: list[StudyPlanPhase] = Field(..., min_length=2)
    weekly_schedule: list[str] = Field(..., min_length=1)
    milestones: list[str] = Field(..., min_length=1)
    practice_tasks: list[str] = Field(..., min_length=1)
    risk_warnings: list[str] = Field(default_factory=list)
    evidence_usage: list[str] = Field(..., min_length=1)


class StudyPlanPhasesArtifact(BaseModel):
    """First-stage study-plan phase structure."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(..., min_length=1)
    learner_profile_summary: str = Field(..., min_length=1)
    overall_goal: str = Field(..., min_length=1)
    phases: list[StudyPlanPhase] = Field(..., min_length=2)
    evidence_usage: list[str] = Field(..., min_length=1)


class StudyPlanScheduleArtifact(BaseModel):
    """Second-stage study-plan execution schedule."""

    model_config = ConfigDict(extra="forbid", strict=True)

    weekly_schedule: list[str] = Field(..., min_length=1)
    milestones: list[str] = Field(..., min_length=1)
    practice_tasks: list[str] = Field(..., min_length=1)
    risk_warnings: list[str] = Field(default_factory=list)


class StudyPlanReviewVerdict(BaseModel):
    """Structured study-plan reviewer verdict."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["approve", "reject"]
    reason: str = Field(..., min_length=1)


class StudyPlanContractError(ValueError):
    """Raised when study-plan state violates the authoritative graph contract."""


class StudyPlanApprovalError(RuntimeError):
    """Raised when an unapproved study plan reaches authoritative output."""


class StudyPlanArtifactWriteError(RuntimeError):
    """Raised when an approved study plan cannot be persisted safely."""


def _last_human_query(state: LearningState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _model_to_dict(model: BaseModel) -> dict:
    return model.model_dump(mode="python")


def _format_keypoints(state: LearningState) -> str:
    keypoints = state.get("keypoints", [])
    return (
        ", ".join(str(item) for item in keypoints if str(item).strip())
        or "No explicit keypoints extracted."
    )


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations or course materials."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = (
            item.get("source")
            or item.get("title")
            or item.get("url")
            or "learning material"
        )
        content = str(
            item.get("content") or item.get("snippet") or item.get("text") or ""
        )[:900]
        if content:
            parts.append(f"[{idx}] Source: {source}\n{content}")
    return (
        "\n\n".join(parts)
        or "Judged evidence exists but has no readable body. Use only general learning-planning guidance."
    )


def _format_profile_context(state: LearningState) -> str:
    confirmed = _confirmed_profile_values(state)
    inferred = dict(state.get("learner_profile_inferred") or {})
    if not inferred:
        inferred = _inferred_profile_values(state)
    lines: list[str] = []
    for field in PROFILE_COMPLETION_FIELDS:
        key = str(field["key"])
        label = str(field["label"])
        if confirmed.get(key):
            lines.append(f"- {label} (user_confirmed): {confirmed[key]}")
        elif inferred.get(key):
            lines.append(f"- {label} (inferred_current_request): {inferred[key]}")
    return "\n".join(lines) or "No structured learner profile fields are available."


def _profile_field_by_key() -> dict[str, dict[str, Any]]:
    return {str(field["key"]): dict(field) for field in PROFILE_COMPLETION_FIELDS}


def _profile_text(value: object, *, max_chars: int = 512) -> str:
    if isinstance(value, Mapping):
        value = value.get("value") or value.get("text") or value.get("content")
    return sanitize_workspace_text(value, max_chars=max_chars, fallback="")


def _confirmed_profile_values(state: LearningState) -> dict[str, str]:
    """Collect user-confirmed profile facts only."""
    fields = _profile_field_by_key()
    result: dict[str, str] = {}
    profile = state.get("learner_profile")
    if isinstance(profile, Mapping):
        for key, field in fields.items():
            text = _profile_text(profile.get(key), max_chars=int(field["max_chars"]))
            if text:
                result[key] = text
    workspace = state.get("task_workspace")
    requirements = (
        workspace.get("profile_requirements") if isinstance(workspace, Mapping) else []
    )
    if isinstance(requirements, list):
        for item in requirements:
            if not isinstance(item, Mapping):
                continue
            key = sanitize_workspace_text(item.get("field"), max_chars=80)
            if key not in fields or result.get(key):
                continue
            text = _profile_text(
                item.get("value_preview"),
                max_chars=int(fields[key]["max_chars"]),
            )
            if text:
                result[key] = text
    return result


def _inferred_profile_values(state: LearningState) -> dict[str, str]:
    """Collect current-request inferred facts without making them persistent."""
    fields = _profile_field_by_key()
    result: dict[str, str] = {}
    raw_inferred = state.get("learner_profile_inferred")
    if isinstance(raw_inferred, Mapping):
        for key, field in fields.items():
            text = _profile_text(
                raw_inferred.get(key),
                max_chars=int(field["max_chars"]),
            )
            if text:
                result[key] = text
    learning_goal = _profile_text(
        state.get("learning_goal"),
        max_chars=int(fields["learning_goal"]["max_chars"]),
    )
    if learning_goal:
        result.setdefault("learning_goal", learning_goal)
    return result


def missing_profile_fields_for_resource(
    state: LearningState,
    resource_type: str,
) -> dict[str, Any]:
    """Return profile requirement status for the requested resource."""
    if resource_type != "study_plan":
        return {
            "missing_required_fields": [],
            "confirmed_values": {},
            "inferred_values": {},
            "field_sources": {},
        }
    confirmed = _confirmed_profile_values(state)
    inferred = _inferred_profile_values(state)
    field_sources: dict[str, str] = {}
    missing: list[dict[str, Any]] = []
    for field in PROFILE_COMPLETION_FIELDS:
        key = str(field["key"])
        if confirmed.get(key):
            field_sources[key] = "user_confirmed"
        elif inferred.get(key):
            field_sources[key] = "inferred"
        elif field.get("required") is True:
            missing.append(dict(field))
            field_sources[key] = "missing"
    return {
        "missing_required_fields": missing,
        "confirmed_values": confirmed,
        "inferred_values": inferred,
        "field_sources": field_sources,
    }


def _profile_completion_request_payload(
    state: LearningState,
    missing_required_fields: list[dict[str, Any]],
    *,
    node_name: str,
) -> dict[str, Any]:
    field_keys = {str(field["key"]) for field in missing_required_fields}
    fields = [
        dict(field)
        for field in PROFILE_COMPLETION_FIELDS
        if str(field["key"]) in field_keys or field.get("required") is not True
    ]
    return {
        "type": "profile_completion_required",
        "title": "生成学习计划前需要补充学习信息",
        "fields": fields,
        "missing_required_keys": sorted(field_keys),
        "node": node_name,
        "resource_type": "study_plan",
        "thread_id": sanitize_workspace_text(
            state.get("thread_id") or state.get("session_id"),
            max_chars=120,
            fallback="",
        ),
        "request_id": sanitize_workspace_text(
            state.get("request_id"),
            max_chars=120,
            fallback="",
        ),
        "resume_available": True,
    }


def _profile_completion_from_resume(
    value: object,
    *,
    required_keys: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("profile_completion resume payload must be an object")
    if value.get("type") != "profile_completion_required":
        raise ValueError("profile_completion resume payload has wrong type")
    raw_completion = value.get("profile_completion")
    if not isinstance(raw_completion, Mapping):
        raise ValueError(
            "profile_completion resume payload is missing profile_completion"
        )

    completion: dict[str, str] = {}
    field_by_key = _profile_field_by_key()
    for key, field in field_by_key.items():
        max_chars = int(field.get("max_chars") or 400)
        text = sanitize_workspace_text(
            raw_completion.get(key),
            max_chars=max_chars,
            fallback="",
        )
        if text:
            completion[key] = text

    missing = [key for key in required_keys if key not in completion]
    if missing:
        raise ValueError(
            "profile_completion missing required fields: " + ", ".join(missing)
        )
    return completion


def _merge_confirmed_profile(
    existing: Mapping[str, Any] | None,
    completion: Mapping[str, str],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    fields = _profile_field_by_key()
    if isinstance(existing, Mapping):
        for key, field in fields.items():
            text = _profile_text(existing.get(key), max_chars=int(field["max_chars"]))
            if text:
                merged[key] = text
    for key, value in completion.items():
        if key in fields and value:
            merged[key] = value
    return merged


def _profile_completion_summary(completion: Mapping[str, str]) -> str:
    lines = []
    for field in PROFILE_COMPLETION_FIELDS:
        key = str(field["key"])
        value = completion.get(key, "")
        if value:
            lines.append(f"{PROFILE_COMPLETION_LABELS[key]}: {value}")
    return "\n".join(lines)


def _emit_study_plan_repair_trace(
    state: LearningState,
    *,
    stage: str,
    result: StructuredLLMResult,
    status: Literal["succeeded", "failed"],
) -> None:
    if int(result.retry_count or 0) <= 0:
        return
    base_payload = {
        "node_name": "study_plan_agent",
        "stage": stage,
        "schema_name": result.schema_name,
        "retry_count": int(result.retry_count or 0),
        "failure_phase": sanitize_workspace_text(
            result.failure_phase,
            max_chars=120,
            fallback="",
        ),
        "error_type": sanitize_workspace_text(
            result.error_type,
            max_chars=120,
            fallback="",
        ),
    }
    emit_a3_trace(
        logger,
        "study_plan.schema_repair_started",
        base_payload,
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    emit_a3_trace(
        logger,
        f"study_plan.schema_repair_{status}",
        base_payload,
        state=state,
        env_flag="LOG_A3_TRACE",
    )


async def _invoke_study_plan_stage(
    state: LearningState,
    *,
    stage: str,
    schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    business_validator: Callable[[BaseModel], str],
) -> BaseModel:
    emit_a3_trace(
        logger,
        "study_plan.stage_started",
        {
            "node_name": "study_plan_agent",
            "stage": stage,
            "schema_name": schema.__name__,
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    try:
        structured_result = await invoke_structured_llm(
            node_name="study_plan_agent",
            llm_node="study_plan",
            schema=schema,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            output_mode=get_llm_output_mode("study_plan_agent"),
            business_validator=business_validator,
            state=state,
            max_raw_chars=get_max_raw_chars("study_plan_agent"),
        )
    except StructuredOutputError as exc:
        _emit_study_plan_repair_trace(
            state,
            stage=stage,
            result=exc.result,
            status="failed",
        )
        raise
    if structured_result.success is not True:
        _emit_study_plan_repair_trace(
            state,
            stage=stage,
            result=structured_result,
            status="failed",
        )
        raise StructuredOutputError(structured_result)
    if structured_result.parsed is None:
        raise TypeError(f"{stage} returned no parsed result")
    _emit_study_plan_repair_trace(
        state,
        stage=stage,
        result=structured_result,
        status="succeeded",
    )
    emit_a3_trace(
        logger,
        "study_plan.stage_succeeded",
        {
            "node_name": "study_plan_agent",
            "stage": stage,
            "schema_name": schema.__name__,
            "retry_count": int(structured_result.retry_count or 0),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    return structured_result.parsed


def validate_emotional_profile(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanEmotionalProfile):
        return "root expected StudyPlanEmotionalProfile"
    if not parsed.summary.strip():
        return "summary must be non-empty"
    if not parsed.motivation_state.strip():
        return "motivation_state must be non-empty"
    for idx, suggestion in enumerate(parsed.support_suggestions):
        if not suggestion.strip():
            return f"support_suggestions.{idx} must be non-empty"
    return ""


def _validate_string_items(
    field_name: str,
    items: list[str],
    *,
    required: bool = True,
) -> str:
    if required and not items:
        return f"{field_name} must be non-empty"
    for idx, item in enumerate(items):
        if not item.strip():
            return f"{field_name}.{idx} must be non-empty"
    return ""


def validate_study_plan_phases(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanPhasesArtifact):
        return "root expected StudyPlanPhasesArtifact"
    if len(parsed.phases or []) < 2:
        return "phases must contain at least 2 items"
    for field_name, value in (
        ("title", parsed.title),
        ("learner_profile_summary", parsed.learner_profile_summary),
        ("overall_goal", parsed.overall_goal),
    ):
        if not value.strip():
            return f"{field_name} must be non-empty"
    for idx, phase in enumerate(parsed.phases or []):
        prefix = f"phases.{idx}"
        if not phase.title.strip():
            return f"{prefix}.title must be non-empty"
        if not phase.duration.strip():
            return f"{prefix}.duration must be non-empty"
        for field_name, items in (
            ("goals", phase.goals),
            ("tasks", phase.tasks),
            ("resources", phase.resources),
            ("practice", phase.practice),
            ("checkpoints", phase.checkpoints),
        ):
            item_error = _validate_string_items(f"{prefix}.{field_name}", items)
            if item_error:
                return item_error
    if not parsed.evidence_usage:
        return "evidence_usage must be non-empty"
    evidence_error = _validate_string_items("evidence_usage", parsed.evidence_usage)
    if evidence_error:
        return evidence_error
    return ""


def validate_study_plan_schedule(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanScheduleArtifact):
        return "root expected StudyPlanScheduleArtifact"
    for field_name, items in (
        ("weekly_schedule", parsed.weekly_schedule),
        ("milestones", parsed.milestones),
        ("practice_tasks", parsed.practice_tasks),
    ):
        item_error = _validate_string_items(field_name, items)
        if item_error:
            return item_error
    risk_error = _validate_string_items(
        "risk_warnings",
        parsed.risk_warnings,
        required=False,
    )
    if risk_error:
        return risk_error
    return ""


def validate_study_plan_artifact(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanArtifact):
        return "root expected StudyPlanArtifact"
    for field_name, value in (
        ("title", parsed.title),
        ("learner_profile_summary", parsed.learner_profile_summary),
        ("overall_goal", parsed.overall_goal),
    ):
        if not value.strip():
            return f"{field_name} must be non-empty"
    if len(parsed.phases or []) < 2:
        return "phases must contain at least 2 items"
    for idx, phase in enumerate(parsed.phases or []):
        prefix = f"phases.{idx}"
        if not phase.title.strip():
            return f"{prefix}.title must be non-empty"
        if not phase.duration.strip():
            return f"{prefix}.duration must be non-empty"
        for field_name, items in (
            ("goals", phase.goals),
            ("tasks", phase.tasks),
            ("resources", phase.resources),
            ("practice", phase.practice),
            ("checkpoints", phase.checkpoints),
        ):
            item_error = _validate_string_items(f"{prefix}.{field_name}", items)
            if item_error:
                return item_error
    for field_name, items in (
        ("weekly_schedule", parsed.weekly_schedule),
        ("milestones", parsed.milestones),
        ("practice_tasks", parsed.practice_tasks),
        ("risk_warnings", parsed.risk_warnings),
        ("evidence_usage", parsed.evidence_usage),
    ):
        item_error = _validate_string_items(
            field_name,
            items,
            required=field_name != "risk_warnings",
        )
        if item_error:
            return item_error
    return ""


def validate_study_plan_review(parsed: BaseModel) -> str:
    if not isinstance(parsed, StudyPlanReviewVerdict):
        return "root expected StudyPlanReviewVerdict"
    if parsed.verdict not in {"approve", "reject"}:
        return "verdict must be approve or reject"
    if not parsed.reason.strip():
        return "reason must be non-empty"
    return ""


def _validated_study_plan_artifact(artifact: object) -> StudyPlanArtifact:
    if not isinstance(artifact, Mapping):
        raise StudyPlanContractError("study_plan artifact must be an object")
    try:
        validated = StudyPlanArtifact.model_validate(dict(artifact), strict=True)
    except ValidationError as exc:
        raise StudyPlanContractError(
            "study_plan artifact violates StudyPlanArtifact"
        ) from exc
    business_error = validate_study_plan_artifact(validated)
    if business_error:
        raise StudyPlanContractError(
            f"study_plan artifact failed business validation: {business_error}"
        )
    return validated


def _study_plan_model_name() -> str:
    configured_model = get_setting("llm.study_plan.model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError("llm.study_plan.model must be explicitly configured")
    return configured_model.strip()


def _study_plan_temperature() -> float:
    configured_temperature = get_setting("llm.study_plan.temperature", None)
    if isinstance(configured_temperature, bool) or not isinstance(
        configured_temperature,
        (int, float),
    ):
        raise ValueError("llm.study_plan.temperature must be explicitly configured")
    temperature = float(configured_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.study_plan.temperature must be between 0 and 2")
    return temperature


def _study_plan_max_generation_rounds() -> int:
    configured_rounds = get_setting(
        "llm.study_plan.max_generation_rounds",
        None,
    )
    if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
        raise ValueError(
            "llm.study_plan.max_generation_rounds must be explicitly configured"
        )
    if configured_rounds < 1:
        raise ValueError("llm.study_plan.max_generation_rounds must be positive")
    return configured_rounds


def _study_plan_planner_max_raw_chars() -> int:
    configured_chars = get_setting(
        "llm_outputs.study_plan_planner.max_raw_chars",
        None,
    )
    if isinstance(configured_chars, bool) or not isinstance(configured_chars, int):
        raise ValueError(
            "llm_outputs.study_plan_planner.max_raw_chars must be explicitly configured"
        )
    if configured_chars < 1:
        raise ValueError(
            "llm_outputs.study_plan_planner.max_raw_chars must be positive"
        )
    return configured_chars


def _study_plan_round(state: Mapping[str, object]) -> int:
    value = state.get("study_plan_round")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StudyPlanContractError(
            "study_plan_round must be an explicit non-negative integer"
        )
    return value


def _study_plan_review_state(
    state: Mapping[str, object],
) -> tuple[Literal["approve", "reject"], str, Literal["approve", "reject"], str]:
    raw_academic_verdict = state.get("study_plan_academic_verdict")
    raw_emotional_verdict = state.get("study_plan_emotional_verdict")
    if raw_academic_verdict == "approve":
        academic_verdict: Literal["approve", "reject"] = "approve"
    elif raw_academic_verdict == "reject":
        academic_verdict = "reject"
    else:
        raise StudyPlanContractError(
            "study_plan academic reviewer requires an explicit approve or reject verdict"
        )
    if raw_emotional_verdict == "approve":
        emotional_verdict: Literal["approve", "reject"] = "approve"
    elif raw_emotional_verdict == "reject":
        emotional_verdict = "reject"
    else:
        raise StudyPlanContractError(
            "study_plan emotional reviewer requires an explicit approve or reject verdict"
        )
    academic_reason = state.get("study_plan_academic_reason")
    emotional_reason = state.get("study_plan_emotional_reason")
    if not isinstance(academic_reason, str) or not academic_reason.strip():
        raise StudyPlanContractError(
            "study_plan academic reviewer reason must be non-empty"
        )
    if not isinstance(emotional_reason, str) or not emotional_reason.strip():
        raise StudyPlanContractError(
            "study_plan emotional reviewer reason must be non-empty"
        )
    return (
        academic_verdict,
        academic_reason.strip(),
        emotional_verdict,
        emotional_reason.strip(),
    )


def _require_study_plan_output_approval(state: Mapping[str, object]) -> None:
    if _study_plan_round(state) < 1:
        raise StudyPlanApprovalError(
            "study_plan output requires at least one completed generation round"
        )
    consensus = state.get("study_plan_consensus")
    if consensus is not True:
        raise StudyPlanApprovalError(
            "study_plan output requires authoritative consensus=true"
        )
    academic_verdict, _, emotional_verdict, _ = _study_plan_review_state(state)
    if academic_verdict != "approve" or emotional_verdict != "approve":
        raise StudyPlanApprovalError(
            "study_plan output requires exact approval from both reviewers"
        )


def _validated_study_plan_document(document: object) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise StudyPlanArtifactWriteError(
            "study_plan document writer returned a non-object artifact"
        )
    normalized = dict(document)
    for key in ("artifact_id", "filename", "markdown_url", "docx_url"):
        value = normalized.get(key)
        if not isinstance(value, str) or not value.strip():
            raise StudyPlanArtifactWriteError(
                f"study_plan document artifact missing required {key}"
            )
        normalized[key] = value.strip()
    return normalized


def _render_artifact_markdown(artifact: StudyPlanArtifact) -> str:
    lines = [
        f"# {artifact.title}",
        "",
        "## Learner Profile",
        artifact.learner_profile_summary,
        "",
        "## Overall Goal",
        artifact.overall_goal,
        "",
        "## Phases",
    ]
    for idx, phase in enumerate(artifact.phases, 1):
        lines.extend(
            [
                "",
                f"### {idx}. {phase.title}",
                f"- Duration: {phase.duration}",
                "- Goals:",
                *[f"  - {item}" for item in phase.goals],
                "- Tasks:",
                *[f"  - {item}" for item in phase.tasks],
                "- Resources:",
                *[f"  - {item}" for item in phase.resources],
                "- Practice:",
                *[f"  - {item}" for item in phase.practice],
                "- Checkpoints:",
                *[f"  - {item}" for item in phase.checkpoints],
            ]
        )
    lines.extend(
        [
            "",
            "## Weekly Schedule",
            *[f"- {item}" for item in artifact.weekly_schedule],
        ]
    )
    lines.extend(
        [
            "",
            "## Milestones",
            *[f"- {item}" for item in artifact.milestones],
        ]
    )
    lines.extend(
        [
            "",
            "## Practice Tasks",
            *[f"- {item}" for item in artifact.practice_tasks],
        ]
    )
    if artifact.risk_warnings:
        lines.extend(
            [
                "",
                "## Risk Warnings",
                *[f"- {item}" for item in artifact.risk_warnings],
            ]
        )
    lines.extend(
        [
            "",
            "## Evidence Usage",
            *[f"- {item}" for item in artifact.evidence_usage],
        ]
    )
    return "\n".join(lines).strip()


@traced_node
async def study_plan_emotional_intel(state: LearningState) -> dict:
    """Analyze learner workload and emotional context for study-plan generation."""
    query = _last_human_query(state)
    history = "\n".join(
        str(getattr(msg, "content", msg)) for msg in state.get("messages", [])[-8:]
    )
    prompt = (
        "Analyze the learner's emotional state and workload risk for a personalized university study plan.\n"
        "Do not provide therapy or medical diagnosis. Focus on study burden, motivation, pacing, and support needs.\n\n"
        f"User query:\n{query}\n\nConversation excerpt:\n{history}"
    )
    model_name = _study_plan_model_name()
    temperature = _study_plan_temperature()
    with traced_llm_call(
        model_name=model_name,
        node_name="study_plan_emotional_intel",
        temperature=temperature,
    ):
        structured_result = await invoke_structured_llm(
            node_name="study_plan_emotional_intel",
            llm_node="study_plan",
            schema=StudyPlanEmotionalProfile,
            messages=[
                SystemMessage(
                    content="You analyze learner workload for a study-plan agent. Return only JSON."
                ),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("study_plan_emotional_intel"),
            business_validator=validate_emotional_profile,
            state=state,
            max_raw_chars=get_max_raw_chars("study_plan_emotional_intel"),
        )
    if structured_result.success is not True:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, StudyPlanEmotionalProfile):
        raise TypeError(
            "study_plan_emotional_intel parsed result is not StudyPlanEmotionalProfile"
        )
    profile = _model_to_dict(result)
    emit_a3_trace(
        logger,
        "study_plan_emotional_intel",
        {
            "success": True,
            "workload_risk": profile.get("workload_risk"),
            "summary_chars": len(result.summary),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "study_plan_emotional_intel": result.summary,
        "study_plan_emotional_profile": profile,
    }


@traced_node
async def study_plan_planner(state: LearningState) -> dict:
    """Create a non-empty outline for the study-plan artifact."""
    temperature = _study_plan_temperature()
    planner_max_raw_chars = _study_plan_planner_max_raw_chars()
    query = _last_human_query(state)
    context = state.get("context", [])
    curriculum_context = state.get("curriculum_context", "")

    curriculum_section = ""
    if state.get("learner_path_planner_output") or state.get(
        "learner_path_provider_projection"
    ):
        learner_path_projection = learner_path_provider_projection_from_state(state)
        learner_path_json = json.dumps(
            learner_path_projection.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        curriculum_section = (
            "\n\n[Provider-safe Learner Path Projection V1 — use the available "
            "plan or respect "
            "the explicit unavailable reason]\n"
            f"{learner_path_json}\n"
        )
    elif curriculum_context:
        curriculum_section = (
            f"\n\n[Curriculum Engine Context — use this to adjust topic ordering, "
            f"skip mastered topics, reinforce weak topics, and respect prerequisites]\n"
            f"{curriculum_context}\n"
        )

    prompt = (
        "Create a concise outline for a personalized university learning plan.\n"
        "Use the judged evidence only as support. Do not invent course materials or citations.\n\n"
        f"User query:\n{query}\n\nKeypoints:\n{_format_keypoints(state)}\n\n"
        f"Emotional/workload intel:\n{state.get('study_plan_emotional_intel', '')}"
        f"{curriculum_section}\n\n"
        f"Judged evidence:\n{_format_context(context)}"
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="study_plan_planner",
        llm_node="study_plan",
        messages=[
            SystemMessage(
                content="You plan personalized university study resources. Return a concrete outline only."
            ),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=temperature,
        max_raw_chars=planner_max_raw_chars,
    )
    if not outline.strip():
        raise ValueError("study_plan_planner produced empty outline")
    return {
        "study_plan_outline": outline,
        "study_plan_artifact": {},
        "study_plan_markdown": "",
        "study_plan_round": 0,
        "study_plan_academic_verdict": "",
        "study_plan_academic_reason": "",
        "study_plan_emotional_verdict": "",
        "study_plan_emotional_reason": "",
        "study_plan_consensus": False,
        "study_plan_revision_notes": "",
        "study_plan_document_artifact": {},
    }


async def _study_plan_profile_gate(
    state: LearningState,
    *,
    node_name: str,
) -> dict:
    """Pause study-plan generation until minimum learner profile facts exist."""
    requirement_status = missing_profile_fields_for_resource(state, "study_plan")
    missing_required_fields = list(
        requirement_status.get("missing_required_fields") or []
    )
    inferred_values = dict(requirement_status.get("inferred_values") or {})
    if not missing_required_fields:
        return {
            "profile_completion_request": {},
            "learner_profile_inferred": inferred_values,
        }

    request_payload = _profile_completion_request_payload(
        state,
        missing_required_fields,
        node_name=node_name,
    )
    safe_profile_request = {
        "title": request_payload["title"],
        "fields": request_payload["fields"],
        "missing_required_keys": request_payload["missing_required_keys"],
    }
    emit_a3_trace(
        logger,
        "profile_completion.required",
        {
            "node_name": node_name,
            "resource_type": "study_plan",
            "field_count": len(request_payload.get("fields") or []),
            "required_field_count": len(missing_required_fields),
            "inferred_field_count": len(inferred_values),
            "profile_completion_request": safe_profile_request,
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    resume_value = interrupt(
        {
            **request_payload,
            "profile_completion_request": safe_profile_request,
        }
    )
    required_keys = tuple(str(field["key"]) for field in missing_required_fields)
    completion = _profile_completion_from_resume(
        resume_value,
        required_keys=required_keys,
    )
    confirmed_profile = _merge_confirmed_profile(
        state.get("learner_profile"),
        completion,
    )
    summary = _profile_completion_summary(confirmed_profile)
    workspace_update = build_workspace_profile_completion_update(
        dict(state),
        completion,
        field_labels=PROFILE_COMPLETION_LABELS,
    )
    trace_payload = workspace_trace_payload(workspace_update)
    emit_a3_trace(
        logger,
        "profile_completion.completed",
        {
            "node_name": node_name,
            "resource_type": "study_plan",
            "field_count": len(completion),
            "workspace_id": trace_payload.get("workspace_id", ""),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    return {
        "profile_completion_request": {},
        "learner_profile": confirmed_profile,
        "learner_profile_inferred": inferred_values,
        "learner_profile_summary": summary,
        "profile_summary": summary,
        "task_workspace": workspace_update,
        "workspace_events": [
            {
                "stage": "profile_completion.indexed",
                **trace_payload,
            }
        ],
    }


@traced_node
async def study_plan_profile_gate_main(state: LearningState) -> dict:
    """Graph-level resumable profile-completion gate for study-plan resources."""
    return await _study_plan_profile_gate(
        state,
        node_name="study_plan_profile_gate_main",
    )


@traced_node
async def study_plan_agent(state: LearningState) -> dict:
    """Generate a structured study-plan artifact from the outline."""
    outline = state.get("study_plan_outline", "")
    if not outline.strip():
        raise ValueError("study_plan outline is empty")
    round_no = _study_plan_round(state) + 1
    model_name = _study_plan_model_name()
    temperature = _study_plan_temperature()
    with traced_llm_call(
        model_name=model_name,
        node_name="study_plan_agent",
        temperature=temperature,
    ):
        phases_model = await _invoke_study_plan_stage(
            state,
            stage="phases",
            schema=StudyPlanPhasesArtifact,
            system_prompt=(
                "You design the phase structure for a personalized study plan. "
                "Return only JSON matching the schema."
            ),
            user_prompt=(
                "Generate the phase structure for this study plan.\n"
                "Use only judged evidence and clearly available learner facts.\n\n"
                f"User query:\n{_last_human_query(state)}\n\n"
                f"Outline:\n{outline}\n\n"
                f"Revision notes:\n{state.get('study_plan_revision_notes', '') or 'None'}\n\n"
                f"Learner profile facts:\n{_format_profile_context(state)}\n\n"
                f"Emotional/workload intel:\n{state.get('study_plan_emotional_intel', '')}\n\n"
                f"Judged evidence:\n{_format_context(state.get('context', []))}"
            ),
            business_validator=validate_study_plan_phases,
        )
        phases = _model_to_dict(phases_model)
        schedule_model = await _invoke_study_plan_stage(
            state,
            stage="schedule",
            schema=StudyPlanScheduleArtifact,
            system_prompt=(
                "You design the execution schedule for a personalized study plan. "
                "Return only JSON matching the schema."
            ),
            user_prompt=(
                "Generate weekly schedule, milestones, practice tasks, and risk warnings "
                "for the phase structure below.\n\n"
                f"Learner profile facts:\n{_format_profile_context(state)}\n\n"
                f"Phase structure:\n{phases}"
            ),
            business_validator=validate_study_plan_schedule,
        )
        schedule = _model_to_dict(schedule_model)
        final_model = await _invoke_study_plan_stage(
            state,
            stage="final_artifact",
            schema=StudyPlanArtifact,
            system_prompt=(
                "You assemble the final structured personalized study plan. "
                "Return only JSON matching the schema."
            ),
            user_prompt=(
                "Assemble a final StudyPlanArtifact from the staged outputs. "
                "Preserve the phase structure and schedule facts; do not invent unavailable resources.\n\n"
                f"Learner profile facts:\n{_format_profile_context(state)}\n\n"
                f"Phase structure:\n{phases}\n\n"
                f"Schedule structure:\n{schedule}\n\n"
                f"Judged evidence:\n{_format_context(state.get('context', []))}"
            ),
            business_validator=validate_study_plan_artifact,
        )
    result = final_model
    if not isinstance(result, StudyPlanArtifact):
        raise TypeError("study_plan_agent parsed result is not StudyPlanArtifact")
    return {
        "study_plan_artifact": _model_to_dict(result),
        "study_plan_round": round_no,
        "study_plan_academic_verdict": "",
        "study_plan_academic_reason": "",
        "study_plan_emotional_verdict": "",
        "study_plan_emotional_reason": "",
    }


async def _review_study_plan(state: LearningState, *, reviewer_kind: str) -> dict:
    artifact = _validated_study_plan_artifact(state.get("study_plan_artifact"))
    focus = (
        "academic soundness, phase progression, evidence consistency, and avoiding fabricated resources"
        if reviewer_kind == "academic"
        else "workload, pacing, review/rest balance, and fit with emotional intel"
    )
    prompt = (
        f"Review this study plan for {focus}. Return approve or reject with a concise reason.\n\n"
        f"Plan:\n{artifact.model_dump(mode='json')}\n\n"
        f"Emotional intel:\n{state.get('study_plan_emotional_intel', '')}"
    )
    node_name = f"study_plan_reviewer_{reviewer_kind}"
    model_name = _study_plan_model_name()
    temperature = _study_plan_temperature()
    with traced_llm_call(
        model_name=model_name,
        node_name=node_name,
        temperature=temperature,
    ):
        structured_result = await invoke_structured_llm(
            node_name=node_name,
            llm_node="study_plan",
            schema=StudyPlanReviewVerdict,
            messages=[
                SystemMessage(
                    content="You are a strict study-plan quality reviewer. Return only JSON."
                ),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode(node_name),
            business_validator=validate_study_plan_review,
            state=state,
            max_raw_chars=get_max_raw_chars(node_name),
        )
    if structured_result.success is not True:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, StudyPlanReviewVerdict):
        raise TypeError(f"{node_name} parsed result is not StudyPlanReviewVerdict")
    return result.model_dump(mode="python")


@traced_node
async def study_plan_reviewer_academic(state: LearningState) -> dict:
    result = await _review_study_plan(state, reviewer_kind="academic")
    return {
        "study_plan_academic_verdict": result["verdict"],
        "study_plan_academic_reason": result["reason"],
    }


@traced_node
async def study_plan_reviewer_emotional(state: LearningState) -> dict:
    result = await _review_study_plan(state, reviewer_kind="emotional")
    return {
        "study_plan_emotional_verdict": result["verdict"],
        "study_plan_emotional_reason": result["reason"],
    }


@traced_node
async def study_plan_consensus(state: LearningState) -> dict:
    current_round = _study_plan_round(state)
    if current_round < 1:
        raise StudyPlanContractError(
            "study_plan consensus requires at least one generation round"
        )
    academic_verdict, academic_reason, emotional_verdict, emotional_reason = (
        _study_plan_review_state(state)
    )
    if academic_verdict == "approve" and emotional_verdict == "approve":
        return {"study_plan_consensus": True, "study_plan_revision_notes": ""}
    max_rounds = _study_plan_max_generation_rounds()
    notes = "\n".join(
        reason
        for verdict, reason in [
            (academic_verdict, academic_reason),
            (emotional_verdict, emotional_reason),
        ]
        if verdict == "reject"
    )
    if not notes:
        raise StudyPlanContractError(
            "study_plan rejection requires explicit reviewer revision reasons"
        )
    if current_round >= max_rounds:
        raise RuntimeError(f"study_plan rejected after max rounds: {notes}")
    return {"study_plan_consensus": False, "study_plan_revision_notes": notes}


def route_after_study_plan_consensus(state: LearningState) -> str:
    consensus = state.get("study_plan_consensus")
    academic_verdict, _, emotional_verdict, _ = _study_plan_review_state(state)
    both_approved = academic_verdict == "approve" and emotional_verdict == "approve"
    if consensus is True:
        if not both_approved:
            raise StudyPlanContractError(
                "study_plan consensus=true conflicts with reviewer verdicts"
            )
        return "output"
    if consensus is False:
        if both_approved:
            raise StudyPlanContractError(
                "study_plan consensus=false conflicts with reviewer approvals"
            )
        return "rewrite"
    raise StudyPlanContractError(
        "study_plan routing requires an explicit boolean consensus"
    )


@traced_node
async def study_plan_rewrite(state: LearningState) -> dict:
    notes = state.get("study_plan_revision_notes", "")
    if not notes.strip():
        raise ValueError("study_plan rewrite requested without revision notes")
    return {
        "study_plan_revision_notes": f"Revise the study plan according to reviewer feedback:\n{notes}"
    }


@traced_node
async def study_plan_output(state: LearningState) -> dict:
    _require_study_plan_output_approval(state)
    artifact = _validated_study_plan_artifact(state.get("study_plan_artifact"))
    markdown = _render_artifact_markdown(artifact)
    if not markdown.strip():
        raise ValueError("study_plan markdown is empty")
    try:
        written_document = create_markdown_artifact(markdown, artifact.title)
    except Exception as exc:
        raise StudyPlanArtifactWriteError(
            "study_plan document artifact write failed"
        ) from exc
    document = _validated_study_plan_document(written_document)
    return {
        "study_plan_markdown": markdown,
        "study_plan_document_artifact": document,
        "messages": [AIMessage(content=markdown)],
    }
