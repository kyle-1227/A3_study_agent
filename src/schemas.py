"""Data structures that can be reused across modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from src.context_engineering.thread_window_v3 import ThreadContextWindowV3
from src.learning_guidance.recommendation_final import RecommendationFinalV1
from src.learning_guidance.profile_writer import (
    CompiledLearningGuidanceProfileWriteV1,
    LearningGuidanceProfileWriteSourceV1,
    LearningGuidanceProfileWriteRequestV1,
    build_profile_write_source_v1,
    compile_profile_write_request_v1,
)
from src.user_identity import UserIdentityError, validate_user_id


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(max_length=4096)
    request_id: UUID
    thread_id: str | None = None
    user_id: str | None = None

    @field_validator("user_id")
    @classmethod
    def validate_optional_user_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_user_id(value)
        except UserIdentityError as exc:
            raise ValueError(exc.code) from None


class ResumeRequest(BaseModel):
    """Resume a graph interrupted by Human-in-the-loop."""

    thread_id: str
    request_id: UUID
    edited_plan: str = Field(default="", max_length=16384)
    feedback: str | None = Field(default=None, max_length=4096)
    memory_use_choice: Literal["use", "ignore"] | None = None
    profile_completion: "ProfileCompletionSubmission | None" = None


class ProfileCompletionSubmission(BaseModel):
    """User-supplied learner facts for resuming study-plan generation."""

    learning_goal: str = Field(default="", max_length=512)
    current_foundation: str = Field(default="", max_length=512)
    daily_study_time: str = Field(default="", max_length=256)
    deadline: str = Field(default="", max_length=256)
    preferred_learning_style: str = Field(default="", max_length=512)
    weak_points: str = Field(default="", max_length=768)


class StopRequest(BaseModel):
    """Request a safe stop at the next LangGraph node boundary."""

    reason: str = Field(default="user_stop", max_length=512)


class ContinueRequest(BaseModel):
    """Idempotency identity for continuing one stopped graph."""

    request_id: UUID


class ThreadStatusResponse(BaseModel):
    """Run-control status for a LangGraph thread checkpoint."""

    thread_id: str
    schema_version: Literal["run_control_v1", "legacy"]
    run_status: str
    resume_available: bool
    pending_interrupt_type: str = ""
    current_node: str = ""
    last_completed_node: str = ""
    stopped_at: str = ""
    stop_reason: str = ""
    context_usage: dict[str, Any] = Field(default_factory=dict)
    context_usage_history: list[dict[str, Any]] = Field(default_factory=list)
    context_usage_report: dict[str, Any] = Field(default_factory=dict)
    context_usage_report_count: int = 0
    activity_timeline: list[dict[str, Any]] = Field(default_factory=list)
    activity_timeline_count: int = 0
    graph_version: str = ""
    last_llm_input_manifest: dict[str, Any] = Field(default_factory=dict)
    llm_input_manifest_count: int = 0
    background_context_window: dict[str, Any] = Field(default_factory=dict)
    request_context_window: dict[str, Any] = Field(default_factory=dict)
    thread_context_window: dict[str, Any] = Field(default_factory=dict)
    thread_context_window_v3: ThreadContextWindowV3
    context_influence_ledger: dict[str, Any] = Field(default_factory=dict)
    last_resource_final_payload: dict[str, Any] = Field(default_factory=dict)
    last_recommendation_final_payload: RecommendationFinalV1 | None = None
    last_qa_response: dict[str, Any] = Field(default_factory=dict)
    profile_completion_request: dict[str, Any] = Field(default_factory=dict)
    missing_run_control_fields: list[str] = Field(default_factory=list)
    message: str = ""


class OnboardRequest(BaseModel):
    """Strict V2 onboarding payload with explicit KG topic ownership."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    schema_version: Literal["onboard_v2"]
    profile: LearningGuidanceProfileWriteRequestV1
    nickname: str = Field(max_length=120)
    grade: str = Field(min_length=1, max_length=120)
    dislikes: list[Annotated[str, Field(max_length=500)]] = Field(
        strict=True,
        max_length=50,
    )

    @field_validator("nickname", "grade")
    @classmethod
    def validate_direct_text(cls, value: str, info: ValidationInfo) -> str:
        if value != value.strip() or (info.field_name == "grade" and not value):
            raise ValueError("onboarding text fields must be normalized")
        return value

    @field_validator("dislikes")
    @classmethod
    def validate_dislikes(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or value != value.strip() for value in values):
            raise ValueError("dislikes must contain normalized non-blank strings")
        if len(values) != len(set(values)):
            raise ValueError("dislikes must be unique")
        return values


@dataclass(frozen=True, slots=True)
class CompiledOnboardRequestV2:
    """Deeply immutable internal projection of a validated onboarding request."""

    schema_version: Literal["onboard_v2"]
    profile: CompiledLearningGuidanceProfileWriteV1
    profile_write_source: LearningGuidanceProfileWriteSourceV1
    nickname: str
    grade: str
    dislikes: tuple[str, ...]


def compile_onboard_request_v2(request: OnboardRequest) -> CompiledOnboardRequestV2:
    """Strictly revalidate the Python boundary before freezing its collections."""

    if not isinstance(request, OnboardRequest):
        raise TypeError("request must be OnboardRequest")
    validated = OnboardRequest.model_validate(
        dict(vars(request)),
        strict=True,
    )
    return CompiledOnboardRequestV2(
        schema_version=validated.schema_version,
        profile=compile_profile_write_request_v1(validated.profile),
        profile_write_source=build_profile_write_source_v1(
            source_kind="onboard_v2",
            payload=validated.model_dump(mode="json"),
        ),
        nickname=validated.nickname,
        grade=validated.grade,
        dislikes=tuple(validated.dislikes),
    )


class _StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def reject_unnormalized_text(cls, value: object) -> object:
        if isinstance(value, str) and (not value.strip() or value != value.strip()):
            raise ValueError("API text fields must be normalized and non-blank")
        return value


class LearningGuidanceCatalogTopicV1(_StrictApiModel):
    topic_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)


class HealthLiveV1(_StrictApiModel):
    schema_version: Literal["health_live_v1"]
    status: Literal["live"]


class HealthReadyV3(_StrictApiModel):
    schema_version: Literal["health_ready_v3"]
    status: Literal["ready"]
    checkpointer_type: Literal["postgres"]
    graph_version: str = Field(min_length=1, max_length=160)
    knowledge_graph_data_version: str = Field(min_length=1, max_length=160)
    knowledge_graph_artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_child_generation_id: str = Field(min_length=1, max_length=160)
    parent_child_generation_manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_orchestration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    deployment_mode: Literal["active"]
    rollout_activation_enabled: Literal[True]
    rollout_shadow_enabled: Literal[False]


class HealthReadyV4(_StrictApiModel):
    """Ready only when the one active Parent--Child primary is verified."""

    schema_version: Literal["health_ready_v4"]
    status: Literal["ready"]
    checkpointer_type: Literal["postgres"]
    graph_version: str = Field(min_length=1, max_length=160)
    knowledge_graph_data_version: str = Field(min_length=1, max_length=160)
    knowledge_graph_artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_child_primary_revision: int = Field(gt=0)
    parent_child_primary_updated_at: datetime
    parent_child_primary_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_orchestration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("parent_child_primary_updated_at")
    @classmethod
    def validate_primary_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("primary update time must be timezone-aware")
        return value


class LearningGuidanceCatalogSubjectV1(_StrictApiModel):
    subject_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=240)
    topics: list[LearningGuidanceCatalogTopicV1] = Field(
        strict=True,
        min_length=1,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_unique_topics(self) -> "LearningGuidanceCatalogSubjectV1":
        topic_ids = tuple(topic.topic_id for topic in self.topics)
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("catalog topic IDs must be unique within one subject")
        return self


class LearningGuidanceCatalogV1(_StrictApiModel):
    schema_version: Literal["learning_guidance_catalog_v1"]
    data_version: str = Field(min_length=1, max_length=160)
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    subjects: list[LearningGuidanceCatalogSubjectV1] = Field(
        strict=True,
        min_length=1,
        max_length=200,
    )

    @model_validator(mode="after")
    def validate_unique_subjects_and_topics(self) -> "LearningGuidanceCatalogV1":
        subject_ids = tuple(subject.subject_id for subject in self.subjects)
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("catalog subject IDs must be globally unique")
        topic_ids = tuple(
            topic.topic_id for subject in self.subjects for topic in subject.topics
        )
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("catalog topic IDs must be globally unique")
        return self


class OnboardResultV2(_StrictApiModel):
    schema_version: Literal["onboard_result_v2"]
    status: Literal["created", "replayed"]
    request_id: str = Field(min_length=1, max_length=160)
    user_id: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=65_536)
    skills_count: int = Field(ge=1, le=200)
    goals_count: int = Field(ge=1, le=50)
    preferences_count: int = Field(ge=0, le=200)


class ProfileResponse(BaseModel):
    """Profile data returned to the frontend."""

    user_id: str
    has_profile: bool
    summary: str | None = None
