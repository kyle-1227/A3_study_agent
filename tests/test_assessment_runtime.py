"""Strict provider-runtime tests for assessment diagnosis and practice."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from src.assessment.attempt_contracts import (
    AdaptivePracticeDraftBatchV1,
    AdaptivePracticeInputV1,
    AdaptivePracticeTaskDraftV1,
    AssessmentAttemptV1,
    AssessmentErrorClassificationV1,
    AssessmentEvaluationInputV1,
    PublicExerciseCardV1,
)
from src.assessment.identity import stable_adaptive_practice_question_id
from src.assessment.runtime import (
    AssessmentRuntimeError,
    classify_assessment_error_v1,
    generate_adaptive_practice_v1,
    validate_adaptive_practice_draft_v1,
)
from src.llm.http_messages import normalize_openai_messages, preview_openai_messages

THREAD_ID = "thread-assessment-runtime-1"
REQUEST_ID = "00000000-0000-4000-8000-000000000201"
RESOURCE_ID = f"resource:v3:{'a' * 64}"
QUESTION_ID = f"question:v1:{'b' * 64}"


def _evaluation() -> AssessmentEvaluationInputV1:
    return AssessmentEvaluationInputV1(
        schema_version="assessment_evaluation_input_v1",
        thread_id=THREAD_ID,
        attempt=AssessmentAttemptV1(
            schema_version="assessment_attempt_v1",
            request_id=REQUEST_ID,
            resource_id=RESOURCE_ID,
            question_id=QUESTION_ID,
            answer="5",
            time_spent_seconds=4.0,
        ),
        card=PublicExerciseCardV1(
            schema_version="exercise_card_v1",
            question_id=QUESTION_ID,
            question_type="free_text",
            level="basic",
            question="What is 2 + 2?",
            choices=(),
            tags=("arithmetic",),
        ),
        canonical_correct_answer="4",
        answer_explanation="Two plus two equals four.",
    )


def _classification(*, error_type: str = "concept") -> AssessmentErrorClassificationV1:
    return AssessmentErrorClassificationV1.model_validate(
        {
            "schema_version": "assessment_error_classification_v1",
            "error_type": error_type,
            "concept_gap": "The learner has not stabilized number composition.",
            "suggestion": "Review smaller addition facts before retrying.",
            "confidence": 0.93,
        },
        strict=True,
    )


def _request(*, error_type: str = "concept") -> AdaptivePracticeInputV1:
    return AdaptivePracticeInputV1(
        schema_version="adaptive_practice_input_v1",
        evaluation=_evaluation(),
        classification=_classification(error_type=error_type),
    )


def _draft(
    *,
    task_type: str = "review",
    question: str = "What is 1 + 2?",
) -> AdaptivePracticeDraftBatchV1:
    return AdaptivePracticeDraftBatchV1.model_validate(
        {
            "schema_version": "adaptive_practice_draft_batch_v1",
            "tasks": [
                {
                    "schema_version": "adaptive_practice_task_draft_v1",
                    "task_type": task_type,
                    "question": question,
                    "answer": "3",
                    "explanation": "One plus two equals three.",
                    "reason": "Review a simpler addition fact after a concept error.",
                    "tags": ["arithmetic"],
                    "difficulty": 0.2,
                },
            ],
        },
        strict=True,
    )


@pytest.mark.anyio
async def test_classifier_uses_strict_runtime_and_identity_only_state():
    parsed = _classification()
    invoke = AsyncMock(return_value=SimpleNamespace(success=True, parsed=parsed))

    with patch("src.assessment.runtime.invoke_structured_llm", invoke):
        result = await classify_assessment_error_v1(_evaluation())

    assert result == parsed
    kwargs = invoke.await_args.kwargs
    assert kwargs["node_name"] == "error_classifier"
    assert kwargs["llm_node"] == "error_classifier"
    assert kwargs["schema"] is AssessmentErrorClassificationV1
    assert kwargs["sensitive_trace"] is True
    assert kwargs["business_validator"](parsed) == ""
    assert kwargs["state"] == {
        "thread_id": THREAD_ID,
        "session_id": THREAD_ID,
        "request_id": REQUEST_ID,
    }
    provider_envelope = json.loads(kwargs["messages"][1].content)
    assert provider_envelope["schema_version"] == (
        "assessment_private_provider_envelope_v1"
    )
    assert provider_envelope["payload"]["attempt"]["answer"] == "5"
    assert provider_envelope["payload"]["canonical_correct_answer"] == "4"
    preview = preview_openai_messages(
        normalize_openai_messages(kwargs["messages"]),
    )
    preview_json = json.dumps(preview, ensure_ascii=False)
    assert "Two plus two equals four" not in preview_json
    assert '"answer":"5"' not in preview_json
    assert "assessment_error_classification_v1" in kwargs["messages"][0].content


@pytest.mark.anyio
async def test_adaptive_runtime_derives_stable_ids_and_complete_tasks():
    draft = _draft()
    invoke = AsyncMock(return_value=SimpleNamespace(success=True, parsed=draft))

    with patch("src.assessment.runtime.invoke_structured_llm", invoke):
        first = await generate_adaptive_practice_v1(_request())
        second = await generate_adaptive_practice_v1(_request())

    assert first == second
    assert len(first.tasks) == 1
    task = first.tasks[0]
    assert task.question != _evaluation().card.question
    assert task.answer == "3"
    assert task.explanation
    assert task.reason
    assert task.question_id == stable_adaptive_practice_question_id(
        task_type=task.task_type,
        question=task.question,
        tags=task.tags,
        difficulty=task.difficulty,
    )
    kwargs = invoke.await_args.kwargs
    assert kwargs["node_name"] == "practice_generator"
    assert kwargs["llm_node"] == "practice_generator"
    assert kwargs["schema"] is AdaptivePracticeDraftBatchV1
    assert kwargs["sensitive_trace"] is True
    assert kwargs["business_validator"](draft) == ""
    assert "adaptive_practice_draft_batch_v1" in kwargs["messages"][0].content


@pytest.mark.parametrize(
    ("attempt_request", "draft", "message"),
    [
        (
            _request(),
            _draft(question="What is 2 + 2?"),
            "must not repeat",
        ),
        (
            _request(error_type="logic"),
            _draft(task_type="review"),
            "task_type=similar",
        ),
        (
            _request(error_type="implementation"),
            _draft(task_type="similar"),
            "task_type=harder",
        ),
    ],
)
def test_adaptive_business_validation_rejects_non_adaptive_output(
    attempt_request: AdaptivePracticeInputV1,
    draft: AdaptivePracticeDraftBatchV1,
    message: str,
):
    assert message in validate_adaptive_practice_draft_v1(
        draft,
        request=attempt_request,
    )


@pytest.mark.anyio
async def test_runtime_failure_is_content_free_and_has_no_default_result():
    invoke = AsyncMock(
        side_effect=RuntimeError(
            "Authorization: Bearer private-token answer=private-student-answer"
        )
    )
    with patch("src.assessment.runtime.invoke_structured_llm", invoke):
        with pytest.raises(AssessmentRuntimeError) as exc_info:
            await classify_assessment_error_v1(_evaluation())

    message = str(exc_info.value)
    assert exc_info.value.stage == "error_classification"
    assert "private-token" not in message
    assert "private-student-answer" not in message


@pytest.mark.anyio
async def test_runtime_rejects_wrong_parsed_model_without_alias_or_default():
    class WrongModel(BaseModel):
        value: str

    invoke = AsyncMock(
        return_value=SimpleNamespace(success=True, parsed=WrongModel(value="wrong"))
    )
    with patch("src.assessment.runtime.invoke_structured_llm", invoke):
        with pytest.raises(AssessmentRuntimeError) as exc_info:
            await generate_adaptive_practice_v1(_request())

    assert exc_info.value.exception_type == "AdaptivePracticeDraftContractError"


@pytest.mark.anyio
async def test_classifier_blocks_private_canary_echo_from_public_result():
    evaluation = _evaluation().model_copy(
        update={
            "attempt": _evaluation().attempt.model_copy(
                update={"answer": "private-student-answer-canary-91"},
            ),
            "answer_explanation": "private-answer-explanation-canary-37",
        },
    )
    leaked = _classification().model_copy(
        update={
            "concept_gap": ("The response repeated private-student-answer-canary-91.")
        },
    )
    invoke = AsyncMock(return_value=SimpleNamespace(success=True, parsed=leaked))

    with patch("src.assessment.runtime.invoke_structured_llm", invoke):
        with pytest.raises(AssessmentRuntimeError) as exc_info:
            await classify_assessment_error_v1(evaluation)

    assert exc_info.value.exception_type == (
        "AssessmentErrorClassificationContractError"
    )


def test_adaptive_draft_forbids_schema_drift_and_blank_content():
    payload = _draft().tasks[0].model_dump()
    payload["question_id"] = f"question:v1:{'f' * 64}"
    with pytest.raises(ValidationError):
        AdaptivePracticeTaskDraftV1.model_validate(payload, strict=True)

    payload = _draft().tasks[0].model_dump()
    payload["answer"] = "   "
    with pytest.raises(ValidationError):
        AdaptivePracticeTaskDraftV1.model_validate(payload, strict=True)

    payload = _draft().tasks[0].model_dump()
    payload["tags"] = ["arithmetic", " arithmetic "]
    with pytest.raises(ValidationError):
        AdaptivePracticeTaskDraftV1.model_validate(payload, strict=True)


def test_adaptive_draft_requires_json_array_shapes_without_tuple_coercion():
    parsed = _draft()

    assert isinstance(parsed.tasks, list)
    assert isinstance(parsed.tasks[0].tags, list)

    batch_payload = parsed.model_dump(mode="python")
    batch_payload["tasks"] = tuple(batch_payload["tasks"])
    with pytest.raises(ValidationError):
        AdaptivePracticeDraftBatchV1.model_validate(batch_payload, strict=True)

    task_payload = parsed.tasks[0].model_dump(mode="python")
    task_payload["tags"] = tuple(task_payload["tags"])
    with pytest.raises(ValidationError):
        AdaptivePracticeTaskDraftV1.model_validate(task_payload, strict=True)


def test_adaptive_draft_batch_rejects_more_than_three_tasks():
    tasks = []
    for index in range(4):
        payload = _draft().tasks[0].model_dump()
        payload["question"] = f"What is {index} + 1?"
        tasks.append(payload)
    with pytest.raises(ValidationError):
        AdaptivePracticeDraftBatchV1.model_validate(
            {
                "schema_version": "adaptive_practice_draft_batch_v1",
                "tasks": tasks,
            },
            strict=True,
        )


def test_assessment_structured_nodes_apply_required_rules_in_active_rollout():
    from src.llm.structured_output import _prepare_structured_messages_with_context

    for node_name in ("error_classifier", "practice_generator"):
        result = _prepare_structured_messages_with_context(
            node_name=node_name,
            llm_node=node_name,
            messages=[
                {"role": "system", "content": "structured contract"},
                {"role": "user", "content": "private provider envelope"},
            ],
            state={
                "request_id": "00000000-0000-4000-8000-000000000901",
                "thread_id": "thread-assessment-ce-1",
            },
        )

        assert result.debug["structured_context_apply_status"] == "applied"
        assert result.debug["context_apply_applied"] is True
        assert any(
            "<INJECTED_CONTEXT>" in str(message.get("content", ""))
            for message in result.messages
            if isinstance(message, dict)
        )
