"""Bridge validated quiz content into public Resource Final V3 and private state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from src.assessment.attempt_contracts import (
    AssessmentQuestionRecordV1,
    AssessmentQuizSourceItemV1,
    AssessmentResourceRecordV1,
    PrivateExerciseAnswerKeyV1,
    PublicExerciseCardV1,
)
from src.graph.resource_final_v3 import (
    JsonValue,
    ResourceFinalV3Quiz,
    ResourceFinalV3ResourceValidation,
    build_resource_final_v3_resource,
)


@dataclass(frozen=True, slots=True)
class AssessmentQuizProjectionV1:
    """One public quiz resource paired with its checkpoint-only answer keys."""

    public_resource: ResourceFinalV3Quiz
    checkpoint_resource: AssessmentResourceRecordV1


class AssessmentQuizProjectionError(ValueError):
    """Raised when validated quiz content cannot produce both bound views."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def build_assessment_quiz_projection_v1(
    *,
    thread_id: str,
    request_id: str,
    title: str,
    summary: str,
    source_items: Sequence[AssessmentQuizSourceItemV1 | Mapping[str, object]],
    artifact_refs: Mapping[str, str],
    downloadable_count: int,
    verified_local_count: int,
    remote_unverified_count: int,
    warnings: tuple[str, ...],
) -> AssessmentQuizProjectionV1:
    """Build public cards and private keys without serializing answers publicly."""

    if not thread_id.strip() or not request_id.strip():
        raise AssessmentQuizProjectionError(
            code="quiz_projection_identity_missing",
            message="thread_id and request_id must not be blank",
        )
    if not title.strip() or not summary.strip():
        raise AssessmentQuizProjectionError(
            code="quiz_projection_text_missing",
            message="title and summary must not be blank",
        )
    if not source_items:
        raise AssessmentQuizProjectionError(
            code="quiz_projection_items_missing",
            message="source_items must not be empty",
        )

    parsed_items = tuple(_validate_source_item(item) for item in source_items)
    question_ids = tuple(item.question_id for item in parsed_items)
    if len(question_ids) != len(set(question_ids)):
        raise AssessmentQuizProjectionError(
            code="quiz_projection_duplicate_question_id",
            message="source_items must contain unique question_id values",
        )

    public_cards = tuple(_public_card(item) for item in parsed_items)
    public_items: list[JsonValue] = [
        card.model_dump(mode="json") for card in public_cards
    ]
    validation = ResourceFinalV3ResourceValidation(
        schema_version="resource_validation_v1",
        resource_type="quiz",
        valid=True,
        terminal_status="success",
        renderable_count=1,
        downloadable_count=downloadable_count,
        verified_local_count=verified_local_count,
        remote_unverified_count=remote_unverified_count,
        failure_reason="",
        warnings=warnings,
    )
    resource = build_resource_final_v3_resource(
        thread_id=thread_id,
        request_id=request_id,
        kind="quiz",
        status="success",
        title=title,
        summary=summary,
        payload={
            "exercise_artifact": {
                "schema_version": "exercise_public_artifact_v1",
                "title": title,
                "items": public_items,
            },
            "exercise_items": public_items,
        },
        artifact_refs=artifact_refs,
        validation=validation,
    )
    if not isinstance(resource, ResourceFinalV3Quiz):
        raise AssessmentQuizProjectionError(
            code="quiz_projection_resource_type_mismatch",
            message="Resource Final V3 builder did not return a quiz resource",
        )

    checkpoint_resource = AssessmentResourceRecordV1(
        schema_version="assessment_resource_record_v1",
        resource_id=resource.resource_id,
        questions=tuple(
            AssessmentQuestionRecordV1(
                schema_version="assessment_question_record_v1",
                card=card,
                answer_key=_private_answer_key(item),
            )
            for item, card in zip(parsed_items, public_cards, strict=True)
        ),
    )
    return AssessmentQuizProjectionV1(
        public_resource=resource,
        checkpoint_resource=checkpoint_resource,
    )


def _validate_source_item(
    value: AssessmentQuizSourceItemV1 | Mapping[str, object],
) -> AssessmentQuizSourceItemV1:
    try:
        return AssessmentQuizSourceItemV1.model_validate(value, strict=True)
    except (TypeError, ValidationError) as exc:
        raise AssessmentQuizProjectionError(
            code="quiz_projection_source_item_invalid",
            message="source item violates AssessmentQuizSourceItemV1",
        ) from exc


def _public_card(item: AssessmentQuizSourceItemV1) -> PublicExerciseCardV1:
    return PublicExerciseCardV1(
        schema_version="exercise_card_v1",
        question_id=item.question_id,
        question_type=item.question_type,
        level=item.level,
        question=item.question,
        choices=item.choices,
        tags=item.tags,
    )


def _private_answer_key(
    item: AssessmentQuizSourceItemV1,
) -> PrivateExerciseAnswerKeyV1:
    return PrivateExerciseAnswerKeyV1(
        schema_version="exercise_answer_key_v1",
        question_id=item.question_id,
        accepted_answers=(item.answer,),
        match_mode=(
            "exact" if item.question_type == "single_choice" else "trimmed_casefold"
        ),
        answer_explanation=item.explanation,
    )


__all__ = [
    "AssessmentQuizProjectionError",
    "AssessmentQuizProjectionV1",
    "build_assessment_quiz_projection_v1",
]
