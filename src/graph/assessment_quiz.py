"""Bridge validated quiz content into public Resource Final V3 and private state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.assessment.attempt_contracts import (
    AssessmentLearningGuidanceBindingV1,
    AssessmentQuestionRecordV1,
    AssessmentQuizSourceItemV1,
    AssessmentResourceRecordV2,
    PrivateExerciseAnswerKeyV1,
    PublicExerciseCardV1,
)
from src.assessment.checkpoint import (
    AssessmentCheckpointError,
    validate_assessment_checkpoint_resources_v2,
    validate_public_exercise_cards_v1,
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
    checkpoint_resource: AssessmentResourceRecordV2


class AssessmentQuizProjectionError(ValueError):
    """Raised when validated quiz content cannot produce both bound views."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def build_public_exercise_cards_v1(
    source_items: Sequence[AssessmentQuizSourceItemV1 | Mapping[str, object]],
) -> tuple[PublicExerciseCardV1, ...]:
    """Validate private source items before deriving their client-safe cards."""

    parsed_items = _validated_source_items(source_items)
    return tuple(_public_card(item) for item in parsed_items)


def build_assessment_quiz_projection_v1(
    *,
    thread_id: str,
    request_id: str,
    title: str,
    summary: str,
    source_items: Sequence[AssessmentQuizSourceItemV1 | Mapping[str, object]],
    artifact_refs: Mapping[str, str],
    validation: ResourceFinalV3ResourceValidation,
    learning_guidance_binding: AssessmentLearningGuidanceBindingV1 | None,
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
    if validation.resource_type != "quiz":
        raise AssessmentQuizProjectionError(
            code="quiz_projection_validation_type_mismatch",
            message="validation.resource_type must be quiz",
        )
    parsed_items = _validated_source_items(source_items)
    public_cards = tuple(_public_card(item) for item in parsed_items)
    public_items: list[JsonValue] = [
        card.model_dump(mode="json") for card in public_cards
    ]
    resource = build_resource_final_v3_resource(
        thread_id=thread_id,
        request_id=request_id,
        kind="quiz",
        status=validation.terminal_status,
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

    checkpoint_resource = AssessmentResourceRecordV2(
        schema_version="assessment_resource_record_v2",
        resource_id=resource.resource_id,
        learning_guidance_binding=learning_guidance_binding,
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


def validate_assessment_quiz_runtime_binding_v1(
    *,
    thread_id: object,
    exercise_items: object,
    exercise_artifact: object,
    exercise_resource_v3: object,
    assessment_checkpoint_resources: object,
) -> None:
    """Require one public Quiz resource to be bound to its private answer keys."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_thread_missing",
            message="successful quiz requires a non-blank thread_id",
        )
    cards = validate_public_exercise_cards_v1(exercise_items)
    if not isinstance(exercise_artifact, Mapping):
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_artifact_invalid",
            message="successful quiz requires a public exercise artifact",
        )
    artifact_cards = validate_public_exercise_cards_v1(exercise_artifact.get("items"))
    if cards != artifact_cards:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_artifact_mismatch",
            message="exercise artifact cards must match public exercise_items",
        )
    if not isinstance(exercise_resource_v3, Mapping):
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_resource_v3_invalid",
            message="successful quiz requires a strict Resource Final V3 resource",
        )
    try:
        resource = ResourceFinalV3Quiz.model_validate_json(
            _mapping_json(exercise_resource_v3),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_resource_v3_invalid",
            message="successful quiz requires a strict Resource Final V3 resource",
        ) from exc
    if exercise_artifact.get("resource_id") != resource.resource_id:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_resource_id_mismatch",
            message="exercise artifact and Resource Final V3 resource_id must match",
        )
    if exercise_artifact.get("payload_hash") != resource.payload_hash:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_payload_hash_mismatch",
            message="exercise artifact and Resource Final V3 payload_hash must match",
        )
    if not isinstance(assessment_checkpoint_resources, Mapping):
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_checkpoint_invalid",
            message="successful quiz requires a strict private checkpoint resource",
        )
    try:
        checkpoint = validate_assessment_checkpoint_resources_v2(
            assessment_checkpoint_resources
        )
    except AssessmentCheckpointError as exc:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_checkpoint_invalid",
            message="successful quiz requires a strict private checkpoint resource",
        ) from exc
    if checkpoint.thread_id != thread_id:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_checkpoint_thread_mismatch",
            message="quiz checkpoint thread_id must match runtime thread_id",
        )
    checkpoint_resource = next(
        (
            item
            for item in checkpoint.resources
            if item.resource_id == resource.resource_id
        ),
        None,
    )
    if checkpoint_resource is None:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_checkpoint_resource_missing",
            message="quiz checkpoint does not contain the public resource_id",
        )
    checkpoint_cards = tuple(item.card for item in checkpoint_resource.questions)
    if checkpoint_cards != cards:
        raise AssessmentQuizProjectionError(
            code="quiz_runtime_checkpoint_cards_mismatch",
            message="quiz checkpoint cards must match the public resource cards",
        )


def _validate_source_item(
    value: AssessmentQuizSourceItemV1 | Mapping[str, object],
) -> AssessmentQuizSourceItemV1:
    try:
        payload: Mapping[str, object] = (
            value.model_dump(mode="json")
            if isinstance(value, AssessmentQuizSourceItemV1)
            else value
        )
        return AssessmentQuizSourceItemV1.model_validate_json(
            _mapping_json(payload),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise AssessmentQuizProjectionError(
            code="quiz_projection_source_item_invalid",
            message="source item violates AssessmentQuizSourceItemV1",
        ) from exc


def _validated_source_items(
    source_items: Sequence[AssessmentQuizSourceItemV1 | Mapping[str, object]],
) -> tuple[AssessmentQuizSourceItemV1, ...]:
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
    return parsed_items


def _mapping_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
    "build_public_exercise_cards_v1",
    "validate_assessment_quiz_runtime_binding_v1",
]
