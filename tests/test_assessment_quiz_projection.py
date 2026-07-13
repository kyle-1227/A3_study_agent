from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import ValidationError

from src.assessment.attempt_contracts import (
    AssessmentQuestionRecordV1,
    AssessmentQuizSourceItemV1,
    AssessmentResourceRecordV1,
    PrivateExerciseAnswerKeyV1,
)
from src.assessment.checkpoint import (
    AssessmentCheckpointError,
    assessment_checkpoint_resources_reducer,
    merge_assessment_checkpoint_resource_v1,
    validate_assessment_checkpoint_resources_v1,
)
from src.assessment.identity import stable_exercise_question_id
from src.graph.assessment_quiz import (
    AssessmentQuizProjectionError,
    build_assessment_quiz_projection_v1,
    build_public_exercise_cards_v1,
)
from src.graph.resource_final_v3 import (
    ResourceFinalV3ResourceValidation,
    build_resource_final_v3_resource,
)


QUESTION_ONE = stable_exercise_question_id(
    level="basic",
    question_type="free_text",
    question="Explain gradient descent.",
    choices=(),
    tags=("optimization",),
)
QUESTION_TWO = stable_exercise_question_id(
    level="intermediate",
    question_type="single_choice",
    question="Which split is used for model selection?",
    choices=("training", "validation", "test"),
    tags=("evaluation",),
)


def _validation(downloadable_count: int = 1) -> ResourceFinalV3ResourceValidation:
    return ResourceFinalV3ResourceValidation(
        schema_version="resource_validation_v1",
        resource_type="quiz",
        valid=True,
        terminal_status="success",
        renderable_count=1 + downloadable_count,
        downloadable_count=downloadable_count,
        verified_local_count=downloadable_count,
        remote_unverified_count=0,
        failure_reason="",
        warnings=(),
    )


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
        validation=_validation(),
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
    duplicated = _free_text_item()

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
            validation=_validation(0),
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
            validation=_validation(0),
        )

    assert exc_info.value.code == "quiz_projection_source_item_invalid"


def test_public_card_builder_never_carries_private_fields():
    cards = build_public_exercise_cards_v1((_free_text_item(), _single_choice_item()))
    serialized = json.dumps(
        [item.model_dump(mode="json") for item in cards],
        ensure_ascii=False,
        sort_keys=True,
    )

    assert '"answer"' not in serialized
    assert '"explanation"' not in serialized
    assert '"pitfall"' not in serialized


def test_resource_final_v3_quiz_rejects_private_fields_at_schema_boundary():
    public = build_public_exercise_cards_v1((_free_text_item(),))[0].model_dump(
        mode="json"
    )
    leaked = {**public, "answer": "server-only answer"}

    with pytest.raises(ValidationError):
        build_resource_final_v3_resource(
            thread_id="thread-1",
            request_id="request-1",
            kind="quiz",
            status="success",
            title="Quiz",
            summary="One validated card.",
            payload={
                "exercise_artifact": {
                    "schema_version": "exercise_public_artifact_v1",
                    "title": "Quiz",
                    "items": [leaked],
                },
                "exercise_items": [leaked],
            },
            artifact_refs={},
            validation=_validation(0),
        )


def test_checkpoint_merge_preserves_prior_quizzes_and_is_idempotent():
    first = _projection()
    second = build_assessment_quiz_projection_v1(
        thread_id="thread-1",
        request_id="request-2",
        title="Second checkpoint",
        summary="One validated assessment card.",
        source_items=(_free_text_item(),),
        artifact_refs={"markdown_url": "/artifacts/exercises/b/public.md"},
        validation=_validation(),
    )

    checkpoint = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=None,
        resource=first.checkpoint_resource,
    )
    checkpoint = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=checkpoint,
        resource=second.checkpoint_resource,
    )
    repeated = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=checkpoint,
        resource=second.checkpoint_resource,
    )

    assert [item.resource_id for item in checkpoint.resources] == [
        first.public_resource.resource_id,
        second.public_resource.resource_id,
    ]
    assert repeated == checkpoint


def test_checkpoint_merge_rejects_thread_mismatch():
    projection = _projection()
    checkpoint = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=None,
        resource=projection.checkpoint_resource,
    )

    with pytest.raises(AssessmentCheckpointError) as exc_info:
        merge_assessment_checkpoint_resource_v1(
            thread_id="thread-2",
            existing=checkpoint,
            resource=projection.checkpoint_resource,
        )

    assert exc_info.value.code == "assessment_checkpoint_thread_mismatch"


def test_checkpoint_merge_rejects_private_key_conflict_for_same_resource_id():
    projection = _projection()
    original = projection.checkpoint_resource
    first_question = original.questions[0]
    conflicting_key = PrivateExerciseAnswerKeyV1(
        schema_version="exercise_answer_key_v1",
        question_id=first_question.answer_key.question_id,
        accepted_answers=("A conflicting private answer.",),
        match_mode=first_question.answer_key.match_mode,
        answer_explanation=first_question.answer_key.answer_explanation,
    )
    conflicting_question = AssessmentQuestionRecordV1(
        schema_version="assessment_question_record_v1",
        card=first_question.card,
        answer_key=conflicting_key,
    )
    conflicting_resource = AssessmentResourceRecordV1(
        schema_version="assessment_resource_record_v1",
        resource_id=original.resource_id,
        questions=(conflicting_question, *original.questions[1:]),
    )
    checkpoint = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=None,
        resource=original,
    )

    with pytest.raises(AssessmentCheckpointError) as exc_info:
        merge_assessment_checkpoint_resource_v1(
            thread_id="thread-1",
            existing=checkpoint,
            resource=conflicting_resource,
        )

    assert exc_info.value.code == "assessment_checkpoint_resource_conflict"


def test_checkpoint_merge_accepts_real_jsonplus_round_trip_shape():
    projection = _projection()
    checkpoint = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=None,
        resource=projection.checkpoint_resource,
    )
    serializer = JsonPlusSerializer()
    encoded = serializer.dumps_typed(checkpoint.model_dump(mode="json"))
    restored = serializer.loads_typed(encoded)

    assert isinstance(restored["resources"], list)
    assert isinstance(restored["resources"][0]["questions"], list)
    merged = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=restored,
        resource=projection.checkpoint_resource,
    )

    assert merged == checkpoint


def test_checkpoint_reducer_merges_parallel_resource_snapshots_without_conflict():
    first = _projection()
    second = build_assessment_quiz_projection_v1(
        thread_id="thread-1",
        request_id="request-2",
        title="Second checkpoint",
        summary="One validated assessment card.",
        source_items=(_single_choice_item(),),
        artifact_refs={"markdown_url": "/artifacts/exercises/b/public.md"},
        validation=_validation(),
    )
    first_snapshot = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=None,
        resource=first.checkpoint_resource,
    ).model_dump(mode="json")
    second_snapshot = merge_assessment_checkpoint_resource_v1(
        thread_id="thread-1",
        existing=None,
        resource=second.checkpoint_resource,
    ).model_dump(mode="json")

    merged_json = assessment_checkpoint_resources_reducer(
        first_snapshot,
        second_snapshot,
    )
    merged = validate_assessment_checkpoint_resources_v1(merged_json)

    assert {item.resource_id for item in merged.resources} == {
        first.public_resource.resource_id,
        second.public_resource.resource_id,
    }
