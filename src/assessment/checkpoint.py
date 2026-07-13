"""Strict JSON-bound validation and merging for assessment checkpoint state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from src.assessment.attempt_contracts import (
    AssessmentCheckpointResourcesV1,
    AssessmentResourceRecordV1,
    PublicExerciseCardV1,
)
from src.assessment.identity import stable_exercise_question_id


class AssessmentCheckpointError(ValueError):
    """Raised when public cards or private checkpoint state violate identity."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def validate_public_exercise_cards_v1(
    value: object,
) -> tuple[PublicExerciseCardV1, ...]:
    """Validate the exact JSON-safe public card list without private fields."""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise AssessmentCheckpointError(
            code="public_exercise_cards_missing",
            message="public exercise cards must be a non-empty array",
        )
    cards: list[PublicExerciseCardV1] = []
    for index, item in enumerate(value):
        try:
            payload: object = (
                item.model_dump(mode="json")
                if isinstance(item, PublicExerciseCardV1)
                else item
            )
            card = PublicExerciseCardV1.model_validate_json(
                _mapping_json(payload),
                strict=True,
            )
        except (TypeError, ValueError) as exc:
            raise AssessmentCheckpointError(
                code="public_exercise_card_invalid",
                message=f"public exercise card {index + 1} violates its strict schema",
            ) from exc
        expected_question_id = stable_exercise_question_id(
            level=card.level,
            question_type=card.question_type,
            question=card.question,
            choices=card.choices,
            tags=card.tags,
        )
        if card.question_id != expected_question_id:
            raise AssessmentCheckpointError(
                code="public_exercise_question_id_mismatch",
                message=f"public exercise card {index + 1} question_id is invalid",
            )
        cards.append(card)
    question_ids = [card.question_id for card in cards]
    if len(question_ids) != len(set(question_ids)):
        raise AssessmentCheckpointError(
            code="public_exercise_question_id_duplicate",
            message="public exercise question_id values must be unique",
        )
    return tuple(cards)


def validate_assessment_checkpoint_resources_v1(
    value: AssessmentCheckpointResourcesV1 | Mapping[str, object],
) -> AssessmentCheckpointResourcesV1:
    """Validate a live or JSON-restored private checkpoint snapshot."""

    try:
        payload: object = (
            value.model_dump(mode="json")
            if isinstance(value, AssessmentCheckpointResourcesV1)
            else value
        )
        return AssessmentCheckpointResourcesV1.model_validate_json(
            _mapping_json(payload),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise AssessmentCheckpointError(
            code="assessment_checkpoint_invalid",
            message="assessment checkpoint violates its strict JSON schema",
        ) from exc


def merge_assessment_checkpoint_resource_v1(
    *,
    thread_id: str,
    existing: AssessmentCheckpointResourcesV1 | Mapping[str, object] | None,
    resource: AssessmentResourceRecordV1,
) -> AssessmentCheckpointResourcesV1:
    """Append one private resource idempotently and reject identity conflicts."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise AssessmentCheckpointError(
            code="assessment_checkpoint_thread_missing",
            message="thread_id must not be blank",
        )
    current = (
        AssessmentCheckpointResourcesV1(
            schema_version="assessment_checkpoint_resources_v1",
            thread_id=thread_id,
            resources=(),
        )
        if existing is None or existing == {}
        else validate_assessment_checkpoint_resources_v1(existing)
    )
    return _merge_resource_models(current, resource, thread_id=thread_id)


def assessment_checkpoint_resources_reducer(
    existing: dict,
    update: dict,
) -> dict:
    """Merge parallel checkpoint updates without copying them into public results."""

    if not update:
        return dict(existing or {})
    incoming = validate_assessment_checkpoint_resources_v1(update)
    if not existing:
        return incoming.model_dump(mode="json")
    current = validate_assessment_checkpoint_resources_v1(existing)
    merged = current
    for resource in incoming.resources:
        merged = _merge_resource_models(
            merged,
            resource,
            thread_id=incoming.thread_id,
        )
    return merged.model_dump(mode="json")


def _merge_resource_models(
    current: AssessmentCheckpointResourcesV1,
    resource: AssessmentResourceRecordV1,
    *,
    thread_id: str,
) -> AssessmentCheckpointResourcesV1:
    if current.thread_id != thread_id:
        raise AssessmentCheckpointError(
            code="assessment_checkpoint_thread_mismatch",
            message="assessment checkpoint belongs to another thread",
        )
    retained: list[AssessmentResourceRecordV1] = []
    for item in current.resources:
        if item.resource_id != resource.resource_id:
            retained.append(item)
            continue
        if item != resource:
            raise AssessmentCheckpointError(
                code="assessment_checkpoint_resource_conflict",
                message="resource_id is bound to different private answer keys",
            )
        return current
    retained.append(resource)
    try:
        return AssessmentCheckpointResourcesV1(
            schema_version="assessment_checkpoint_resources_v1",
            thread_id=thread_id,
            resources=tuple(retained),
        )
    except ValidationError as exc:
        raise AssessmentCheckpointError(
            code="assessment_checkpoint_capacity_exhausted",
            message="assessment checkpoint cannot accept another resource",
        ) from exc


def _mapping_json(value: object) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("value must be an object")
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "AssessmentCheckpointError",
    "assessment_checkpoint_resources_reducer",
    "merge_assessment_checkpoint_resource_v1",
    "validate_assessment_checkpoint_resources_v1",
    "validate_public_exercise_cards_v1",
]
