"""Strict persisted contracts shared by guidance history readers and writers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.memory.retention import LEARNING_GUIDANCE_HISTORY_ID_PREFIX


LEARNING_GUIDANCE_HISTORY_ID_PATTERN = r"^learning-guidance-history:v1:[0-9a-f]{64}$"


class _StrictHistoryContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class LearningGuidanceHistoryBindingV1(_StrictHistoryContract):
    """JSON-native marker stored under episodic metadata.learning_guidance_v1."""

    schema_version: Literal["learning_guidance_history_event_v1"]
    topic_id: str = Field(min_length=1, max_length=160)
    event_type: Literal[
        "practice",
        "assessment",
        "resource_completion",
        "study_session",
    ]
    observed_at: str = Field(min_length=1, max_length=80)
    outcome_score: float | None = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("topic_id", "observed_at")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("text fields must be normalized and non-blank")
        return value

    def parsed_observed_at(self) -> datetime:
        try:
            parsed = datetime.fromisoformat(self.observed_at)
        except ValueError:
            raise ValueError("observed_at must be ISO 8601") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return parsed


class AssessmentHistorySourceV1(_StrictHistoryContract):
    """Content-free authority receipt for one completed assessment journal entry."""

    schema_version: Literal["assessment_history_source_v1"]
    source_kind: Literal["assessment_attempt_v1"]
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$")
    assessment_request_id: str = Field(
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        )
    )
    assessment_final_payload_hash: str = Field(
        pattern=r"^assessment-final:v1:[0-9a-f]{64}$"
    )
    resource_id: str = Field(pattern=r"^resource:v3:[0-9a-f]{64}$")
    question_id: str = Field(pattern=r"^question:v1:[0-9a-f]{64}$")
    generation_request_id: str = Field(
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        )
    )
    assignment_contract_version: Literal[
        "resource_evidence_assignment_v1",
        "resource_evidence_assignment_v2",
    ]
    assignment_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class LearningGuidanceHistoryWriteResultV1(_StrictHistoryContract):
    schema_version: Literal["learning_guidance_history_write_result_v1"]
    status: Literal["inserted", "replayed"]
    history_id: str = Field(pattern=LEARNING_GUIDANCE_HISTORY_ID_PATTERN)


__all__ = [
    "AssessmentHistorySourceV1",
    "LEARNING_GUIDANCE_HISTORY_ID_PATTERN",
    "LEARNING_GUIDANCE_HISTORY_ID_PREFIX",
    "LearningGuidanceHistoryBindingV1",
    "LearningGuidanceHistoryWriteResultV1",
]
