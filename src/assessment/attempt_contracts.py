"""Strict contracts for assessment attempt submission and terminal results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ASSESSMENT_ATTEMPT_SCHEMA_VERSION = "assessment_attempt_v1"
ASSESSMENT_FINAL_SCHEMA_VERSION = "assessment_final_v1"

_REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$"
_THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$"
_RESOURCE_ID_PATTERN = r"^resource:v3:[0-9a-f]{64}$"
_QUESTION_ID_PATTERN = r"^question:v1:[0-9a-f]{64}$"
_ATTEMPT_HASH_PATTERN = r"^assessment-attempt:v1:[0-9a-f]{64}$"
_FINAL_HASH_PATTERN = r"^assessment-final:v1:[0-9a-f]{64}$"


class AssessmentAttemptV1(BaseModel):
    """Strict request body for one exercise-card answer submission."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_attempt_v1"]
    request_id: str = Field(pattern=_REQUEST_ID_PATTERN, max_length=160)
    resource_id: str = Field(pattern=_RESOURCE_ID_PATTERN)
    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    answer: str = Field(min_length=1, max_length=10_000)
    time_spent_seconds: float = Field(ge=0.0, le=86_400.0)

    @model_validator(mode="after")
    def validate_non_blank_answer(self) -> AssessmentAttemptV1:
        if not self.answer.strip():
            raise ValueError("answer must not be blank")
        return self


class PublicExerciseCardV1(BaseModel):
    """Client-safe exercise card; answer material is intentionally absent."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["exercise_card_v1"]
    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    question_type: Literal["free_text", "single_choice"]
    level: Literal["basic", "intermediate", "application", "self_check"]
    question: str = Field(min_length=1, max_length=10_000)
    choices: tuple[str, ...] = Field(max_length=20)
    tags: tuple[str, ...] = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_public_card(self) -> PublicExerciseCardV1:
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if any(not value.strip() for value in self.choices):
            raise ValueError("choices must not contain blank values")
        if any(not value.strip() for value in self.tags):
            raise ValueError("tags must not contain blank values")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("choices must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must be unique")
        if self.question_type == "free_text" and self.choices:
            raise ValueError("free_text questions must not define choices")
        if self.question_type == "single_choice" and len(self.choices) < 2:
            raise ValueError("single_choice questions require at least two choices")
        return self


class PrivateExerciseAnswerKeyV1(BaseModel):
    """Server-bound answer key kept outside the public exercise card."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["exercise_answer_key_v1"]
    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    accepted_answers: tuple[str, ...] = Field(min_length=1, max_length=50)
    match_mode: Literal["exact", "trimmed_casefold"]
    answer_explanation: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_answer_key(self) -> PrivateExerciseAnswerKeyV1:
        if any(not value.strip() for value in self.accepted_answers):
            raise ValueError("accepted_answers must not contain blank values")
        if not self.answer_explanation.strip():
            raise ValueError("answer_explanation must not be blank")
        normalized = tuple(
            _normalize_answer(value, match_mode=self.match_mode)
            for value in self.accepted_answers
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("accepted_answers must be unique under match_mode")
        return self


class AssessmentQuestionRecordV1(BaseModel):
    """Checkpoint-owned public card paired with its private answer key."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_question_record_v1"]
    card: PublicExerciseCardV1
    answer_key: PrivateExerciseAnswerKeyV1

    @model_validator(mode="after")
    def validate_question_identity(self) -> AssessmentQuestionRecordV1:
        if self.card.question_id != self.answer_key.question_id:
            raise ValueError("card and answer_key question_id must match")
        return self


class AssessmentResourceRecordV1(BaseModel):
    """Assessment-capable resource as stored in checkpoint data."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_resource_record_v1"]
    resource_id: str = Field(pattern=_RESOURCE_ID_PATTERN)
    questions: tuple[AssessmentQuestionRecordV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def validate_unique_questions(self) -> AssessmentResourceRecordV1:
        question_ids = [item.card.question_id for item in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("resource question_id values must be unique")
        return self


class AssessmentCheckpointResourcesV1(BaseModel):
    """Narrow, injected view of assessment resources in a thread checkpoint."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_checkpoint_resources_v1"]
    thread_id: str = Field(pattern=_THREAD_ID_PATTERN, max_length=160)
    resources: tuple[AssessmentResourceRecordV1, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def validate_unique_resources(self) -> AssessmentCheckpointResourcesV1:
        resource_ids = [item.resource_id for item in self.resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("checkpoint resource_id values must be unique")
        return self


class AssessmentQuizSourceItemV1(BaseModel):
    """Private validated quiz source used to build public and checkpoint views."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    question_type: Literal["free_text", "single_choice"]
    level: Literal["basic", "intermediate", "application", "self_check"]
    question: str = Field(min_length=1, max_length=10_000)
    choices: tuple[str, ...] = Field(max_length=20)
    answer: str = Field(min_length=1, max_length=10_000)
    explanation: str = Field(min_length=1, max_length=10_000)
    pitfall: str = Field(min_length=1, max_length=10_000)
    tags: tuple[str, ...] = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_quiz_source(self) -> AssessmentQuizSourceItemV1:
        text_values = (
            self.question,
            self.answer,
            self.explanation,
            self.pitfall,
        )
        if any(not value.strip() for value in text_values):
            raise ValueError("quiz source text fields must not be blank")
        if any(not value.strip() for value in (*self.choices, *self.tags)):
            raise ValueError("quiz source choices and tags must not contain blanks")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("quiz source choices must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("quiz source tags must be unique")
        if self.question_type == "free_text":
            if self.choices:
                raise ValueError("free_text quiz source cannot define choices")
        else:
            if len(self.choices) < 2:
                raise ValueError(
                    "single_choice quiz source requires at least two choices"
                )
            if self.answer not in self.choices:
                raise ValueError("single_choice answer must exactly match one choice")
        return self


class AssessmentErrorClassificationV1(BaseModel):
    """Validated output required from the injected error classifier."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_error_classification_v1"]
    error_type: Literal["concept", "logic", "implementation"]
    concept_gap: str = Field(min_length=1, max_length=1_000)
    suggestion: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_non_blank_text(self) -> AssessmentErrorClassificationV1:
        if not self.concept_gap.strip() or not self.suggestion.strip():
            raise ValueError("classification text fields must not be blank")
        return self


class AdaptivePracticeTaskV1(BaseModel):
    """Complete adaptive task returned after an incorrect answer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["adaptive_practice_task_v1"]
    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    task_type: Literal["similar", "harder", "review"]
    question: str = Field(min_length=1, max_length=10_000)
    answer: str = Field(min_length=1, max_length=10_000)
    explanation: str = Field(min_length=1, max_length=10_000)
    reason: str = Field(min_length=1, max_length=2_000)
    tags: tuple[str, ...] = Field(min_length=1, max_length=80)
    difficulty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_complete_task(self) -> AdaptivePracticeTaskV1:
        required_text = (self.question, self.answer, self.explanation, self.reason)
        if any(not value.strip() for value in required_text):
            raise ValueError(
                "adaptive task question, answer, explanation, and reason "
                "must not be blank"
            )
        if any(not value.strip() for value in self.tags):
            raise ValueError("adaptive task tags must not contain blank values")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("adaptive task tags must be unique")
        return self


class AdaptivePracticeBatchV1(BaseModel):
    """Validated non-empty output required from the adaptive generator."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["adaptive_practice_batch_v1"]
    tasks: tuple[AdaptivePracticeTaskV1, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> AdaptivePracticeBatchV1:
        question_ids = [item.question_id for item in self.tasks]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("adaptive task question_id values must be unique")
        return self


class AssessmentEvaluationInputV1(BaseModel):
    """Private classifier input; never serialize this as a public response."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_evaluation_input_v1"]
    thread_id: str = Field(pattern=_THREAD_ID_PATTERN, max_length=160)
    attempt: AssessmentAttemptV1
    card: PublicExerciseCardV1
    canonical_correct_answer: str = Field(min_length=1, max_length=10_000)
    answer_explanation: str = Field(min_length=1, max_length=10_000)


class AdaptivePracticeInputV1(BaseModel):
    """Private adaptive-generator input for one classified wrong answer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["adaptive_practice_input_v1"]
    evaluation: AssessmentEvaluationInputV1
    classification: AssessmentErrorClassificationV1


class AssessmentFinalV1(BaseModel):
    """Authoritative public terminal result for one assessment request."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["assessment_final_v1"]
    type: Literal["assessment_final"]
    thread_id: str = Field(pattern=_THREAD_ID_PATTERN, max_length=160)
    request_id: str = Field(pattern=_REQUEST_ID_PATTERN, max_length=160)
    resource_id: str = Field(pattern=_RESOURCE_ID_PATTERN)
    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    terminal_status: Literal["correct", "incorrect"]
    is_correct: bool
    time_spent_seconds: float = Field(ge=0.0, le=86_400.0)
    error_classification: AssessmentErrorClassificationV1 | None
    adaptive_tasks: tuple[AdaptivePracticeTaskV1, ...] = Field(max_length=20)
    payload_hash: str = Field(pattern=_FINAL_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_terminal_truth_and_hash(self) -> AssessmentFinalV1:
        if self.terminal_status == "correct":
            if not self.is_correct:
                raise ValueError("correct terminal status requires is_correct=true")
            if self.error_classification is not None or self.adaptive_tasks:
                raise ValueError(
                    "correct result cannot contain error classification or adaptive tasks"
                )
        else:
            if self.is_correct:
                raise ValueError("incorrect terminal status requires is_correct=false")
            if self.error_classification is None:
                raise ValueError("incorrect result requires error classification")
            if not self.adaptive_tasks:
                raise ValueError("incorrect result requires adaptive tasks")

        expected_hash = stable_assessment_final_hash(self)
        if self.payload_hash != expected_hash:
            raise ValueError("payload_hash does not match assessment final content")
        return self


def answer_matches(
    *, submitted_answer: str, answer_key: PrivateExerciseAnswerKeyV1
) -> bool:
    """Evaluate an answer using the explicit server-bound comparison policy."""

    candidate = _normalize_answer(submitted_answer, match_mode=answer_key.match_mode)
    return any(
        candidate == _normalize_answer(accepted, match_mode=answer_key.match_mode)
        for accepted in answer_key.accepted_answers
    )


def stable_assessment_attempt_hash(
    *,
    thread_id: str,
    attempt: AssessmentAttemptV1,
) -> str:
    """Bind idempotency to the route thread and the complete strict request."""

    return _stable_hash(
        "assessment-attempt:v1",
        {
            "thread_id": thread_id,
            "attempt": attempt.model_dump(mode="json"),
        },
    )


def build_assessment_final_v1(
    *,
    thread_id: str,
    attempt: AssessmentAttemptV1,
    is_correct: bool,
    error_classification: AssessmentErrorClassificationV1 | None,
    adaptive_tasks: tuple[AdaptivePracticeTaskV1, ...],
) -> AssessmentFinalV1:
    """Build a final payload and derive its stable content hash."""

    unsigned = {
        "schema_version": ASSESSMENT_FINAL_SCHEMA_VERSION,
        "type": "assessment_final",
        "thread_id": thread_id,
        "request_id": attempt.request_id,
        "resource_id": attempt.resource_id,
        "question_id": attempt.question_id,
        "terminal_status": "correct" if is_correct else "incorrect",
        "is_correct": is_correct,
        "time_spent_seconds": attempt.time_spent_seconds,
        "error_classification": error_classification,
        "adaptive_tasks": adaptive_tasks,
    }
    payload_hash = _stable_hash(
        "assessment-final:v1",
        _assessment_final_json_payload(unsigned),
    )
    return AssessmentFinalV1(**unsigned, payload_hash=payload_hash)


def stable_assessment_final_hash(value: AssessmentFinalV1) -> str:
    """Recompute the canonical hash of a validated final payload."""

    return _stable_hash(
        "assessment-final:v1",
        value.model_dump(mode="json", exclude={"payload_hash"}),
    )


def _assessment_final_json_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return json.loads(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=_dump_model,
        )
    )


def _dump_model(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _normalize_answer(
    value: str,
    *,
    match_mode: Literal["exact", "trimmed_casefold"],
) -> str:
    if match_mode == "exact":
        return value
    return value.strip().casefold()


def _stable_hash(prefix: str, payload: object) -> str:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prefix}:{hashlib.sha256(body.encode('utf-8')).hexdigest()}"


__all__ = [
    "ASSESSMENT_ATTEMPT_SCHEMA_VERSION",
    "ASSESSMENT_FINAL_SCHEMA_VERSION",
    "AdaptivePracticeBatchV1",
    "AdaptivePracticeInputV1",
    "AdaptivePracticeTaskV1",
    "AssessmentAttemptV1",
    "AssessmentCheckpointResourcesV1",
    "AssessmentErrorClassificationV1",
    "AssessmentEvaluationInputV1",
    "AssessmentFinalV1",
    "AssessmentQuestionRecordV1",
    "AssessmentQuizSourceItemV1",
    "AssessmentResourceRecordV1",
    "PrivateExerciseAnswerKeyV1",
    "PublicExerciseCardV1",
    "answer_matches",
    "build_assessment_final_v1",
    "stable_assessment_attempt_hash",
    "stable_assessment_final_hash",
]
