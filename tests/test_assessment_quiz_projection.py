from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.assessment.attempt_contracts import AssessmentQuizSourceItemV1
from src.graph.assessment_quiz import (
    AssessmentQuizProjectionError,
    build_assessment_quiz_projection_v1,
)


QUESTION_ONE = "question:v1:" + "1" * 64
QUESTION_TWO = "question:v1:" + "2" * 64


def _free_text_item() -> AssessmentQuizSourceItemV1:
    return AssessmentQuizSourceItemV1(
        question_id=QUESTION_ONE,
        question_type="free_text",
        level="basic",
        question="Explain gradient descent.",
        choices=(),
        answer="An iterative optimization method.",
        explanation="It follows the negative gradient to reduce loss.",
        pitfall="A learning rate that is too large can diverge.",
        tags=("optimization",),
    )


def _single_choice_item() -> AssessmentQuizSourceItemV1:
    return AssessmentQuizSourceItemV1(
        question_id=QUESTION_TWO,
        question_type="single_choice",
        level="intermediate",
        question="Which split is used for model selection?",
        choices=("training", "validation", "test"),
        answer="validation",
        explanation="Validation data guides model selection.",
        pitfall="The test split must remain untouched until final evaluation.",
        tags=("evaluation",),
    )


def _projection():
    return build_assessment_quiz_projection_v1(
        thread_id="thread-1",
        request_id="request-1",
        title="Machine learning checkpoint",
        summary="Two validated assessment cards.",
        source_items=(_free_text_item(), _single_choice_item()),
        artifact_refs={"markdown_url": "/artifacts/exercises/a/public.md"},
        downloadable_count=1,
        verified_local_count=1,
        remote_unverified_count=0,
        warnings=(),
    )


def test_projection_separates_public_cards_from_private_answer_keys():
    projection = _projection()

    public_json = json.dumps(
        projection.public_resource.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )
    assert '"answer"' not in public_json
    assert '"explanation"' not in public_json
    assert '"pitfall"' not in public_json
    assert projection.public_resource.resource_id.startswith("resource:v3:")

    checkpoint = projection.checkpoint_resource
    assert checkpoint.resource_id == projection.public_resource.resource_id
    assert checkpoint.questions[0].answer_key.accepted_answers == (
        "An iterative optimization method.",
    )
    assert checkpoint.questions[0].answer_key.match_mode == "trimmed_casefold"
    assert checkpoint.questions[1].answer_key.match_mode == "exact"


def test_projection_identity_is_stable_for_identical_public_content():
    first = _projection()
    second = _projection()

    assert first.public_resource.resource_id == second.public_resource.resource_id
    assert first.public_resource.payload_hash == second.public_resource.payload_hash


def test_projection_rejects_duplicate_question_ids():
    duplicated = _free_text_item().model_copy(
        update={"question": "A different question with the same identity."}
    )

    with pytest.raises(
        AssessmentQuizProjectionError,
        match="unique question_id",
    ) as exc_info:
        build_assessment_quiz_projection_v1(
            thread_id="thread-1",
            request_id="request-1",
            title="Quiz",
            summary="Validated quiz.",
            source_items=(_free_text_item(), duplicated),
            artifact_refs={},
            downloadable_count=0,
            verified_local_count=0,
            remote_unverified_count=0,
            warnings=(),
        )

    assert exc_info.value.code == "quiz_projection_duplicate_question_id"


def test_single_choice_source_requires_answer_to_match_a_choice():
    with pytest.raises(ValidationError, match="exactly match one choice"):
        AssessmentQuizSourceItemV1(
            question_id=QUESTION_TWO,
            question_type="single_choice",
            level="basic",
            question="Choose one.",
            choices=("A", "B"),
            answer="C",
            explanation="Only one listed option can be correct.",
            pitfall="Do not invent another option.",
            tags=("choice",),
        )


def test_projection_rejects_untyped_source_mapping_without_inference():
    raw = _free_text_item().model_dump()
    raw.pop("question_type")

    with pytest.raises(AssessmentQuizProjectionError) as exc_info:
        build_assessment_quiz_projection_v1(
            thread_id="thread-1",
            request_id="request-1",
            title="Quiz",
            summary="Validated quiz.",
            source_items=(raw,),
            artifact_refs={},
            downloadable_count=0,
            verified_local_count=0,
            remote_unverified_count=0,
            warnings=(),
        )

    assert exc_info.value.code == "quiz_projection_source_item_invalid"
