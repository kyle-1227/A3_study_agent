"""Focused contract, idempotency, and security tests for assessment attempts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.assessment.attempt_contracts import (
    AdaptivePracticeBatchV1,
    AdaptivePracticeTaskV1,
    AssessmentAttemptV1,
    AssessmentCheckpointResourcesV2,
    AssessmentErrorClassificationV1,
    AssessmentFinalV1,
    AssessmentQuestionRecordV1,
    AssessmentResourceRecordV2,
    PrivateExerciseAnswerKeyV1,
    PublicExerciseCardV1,
    answer_matches,
)
from src.assessment.attempt_service import (
    AssessmentAttemptService,
    AssessmentDependencyFailed,
    AssessmentIdentityError,
    AssessmentOperation,
    AssessmentRequestConflict,
)
from src.assessment.identity import stable_adaptive_practice_question_id

THREAD_ID = "thread-assessment-1"
REQUEST_ID = "00000000-0000-4000-8000-000000000101"
RESOURCE_ID = f"resource:v3:{'a' * 64}"
OTHER_RESOURCE_ID = f"resource:v3:{'b' * 64}"
QUESTION_ID = f"question:v1:{'c' * 64}"
OTHER_QUESTION_ID = f"question:v1:{'d' * 64}"
ADAPTIVE_QUESTION_ID = stable_adaptive_practice_question_id(
    task_type="review",
    question="What is 1 + 2?",
    tags=("arithmetic",),
    difficulty=0.2,
)


@dataclass(frozen=True)
class _StoredResult:
    request_hash: str
    final: AssessmentFinalV1


class _AtomicMemoryIdempotency:
    """Test implementation of the execute-once protocol."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], _StoredResult] = {}
        self.locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.operation_count = 0

    async def execute_once(
        self,
        *,
        thread_id: str,
        request_id: str,
        request_hash: str,
        operation: AssessmentOperation,
    ) -> object:
        key = (thread_id, request_id)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            stored = self.records.get(key)
            if stored is not None:
                if stored.request_hash != request_hash:
                    raise AssessmentRequestConflict()
                return stored.final

            self.operation_count += 1
            final = await operation()
            self.records[key] = _StoredResult(
                request_hash=request_hash,
                final=final,
            )
            return final


def _card() -> PublicExerciseCardV1:
    return PublicExerciseCardV1(
        schema_version="exercise_card_v1",
        question_id=QUESTION_ID,
        question_type="free_text",
        level="basic",
        question="What is 2 + 2?",
        choices=(),
        tags=("arithmetic",),
    )


def _answer_key(
    *,
    match_mode: str = "exact",
) -> PrivateExerciseAnswerKeyV1:
    return PrivateExerciseAnswerKeyV1.model_validate(
        {
            "schema_version": "exercise_answer_key_v1",
            "question_id": QUESTION_ID,
            "accepted_answers": ("4",),
            "match_mode": match_mode,
            "answer_explanation": "Adding two and two gives four.",
        },
        strict=True,
    )


def _checkpoint() -> AssessmentCheckpointResourcesV2:
    return AssessmentCheckpointResourcesV2(
        schema_version="assessment_checkpoint_resources_v2",
        thread_id=THREAD_ID,
        resources=(
            AssessmentResourceRecordV2(
                schema_version="assessment_resource_record_v2",
                resource_id=RESOURCE_ID,
                learning_guidance_binding=None,
                questions=(
                    AssessmentQuestionRecordV1(
                        schema_version="assessment_question_record_v1",
                        card=_card(),
                        answer_key=_answer_key(),
                    ),
                ),
            ),
        ),
    )


def _attempt(
    *,
    answer: str = "4",
    request_id: str = REQUEST_ID,
    resource_id: str = RESOURCE_ID,
    question_id: str = QUESTION_ID,
) -> AssessmentAttemptV1:
    return AssessmentAttemptV1(
        schema_version="assessment_attempt_v1",
        request_id=request_id,
        resource_id=resource_id,
        question_id=question_id,
        answer=answer,
        time_spent_seconds=12.5,
    )


def _classification() -> AssessmentErrorClassificationV1:
    return AssessmentErrorClassificationV1(
        schema_version="assessment_error_classification_v1",
        error_type="concept",
        concept_gap="Addition facts are not yet stable.",
        suggestion="Review number composition before retrying.",
        confidence=0.94,
    )


def _adaptive_batch() -> AdaptivePracticeBatchV1:
    return AdaptivePracticeBatchV1(
        schema_version="adaptive_practice_batch_v1",
        tasks=(
            AdaptivePracticeTaskV1(
                schema_version="adaptive_practice_task_v1",
                question_id=ADAPTIVE_QUESTION_ID,
                task_type="review",
                question="What is 1 + 2?",
                answer="3",
                explanation="One plus two is three.",
                reason="Review a simpler addition fact after the concept error.",
                tags=("arithmetic",),
                difficulty=0.2,
            ),
        ),
    )


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_nested_keys(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_nested_keys(item) for item in value))
    return set()


def _service(
    *,
    idempotency: _AtomicMemoryIdempotency | None = None,
    classifier: AsyncMock | None = None,
    generator: AsyncMock | None = None,
) -> tuple[
    AssessmentAttemptService,
    _AtomicMemoryIdempotency,
    AsyncMock,
    AsyncMock,
]:
    journal = idempotency or _AtomicMemoryIdempotency()
    classifier_mock = classifier or AsyncMock(return_value=_classification())
    generator_mock = generator or AsyncMock(return_value=_adaptive_batch())
    return (
        AssessmentAttemptService(
            idempotency=journal,
            error_classifier=classifier_mock,
            adaptive_generator=generator_mock,
        ),
        journal,
        classifier_mock,
        generator_mock,
    )


@pytest.mark.anyio
async def test_correct_attempt_is_deterministic_and_never_calls_adaptive_dependencies():
    service, journal, classifier, generator = _service()

    final = await service.submit(
        thread_id=THREAD_ID,
        attempt=_attempt(),
        checkpoint=_checkpoint(),
    )

    assert final.terminal_status == "correct"
    assert final.is_correct is True
    assert final.error_classification is None
    assert final.adaptive_tasks == ()
    assert journal.operation_count == 1
    classifier.assert_not_awaited()
    generator.assert_not_awaited()
    assert AssessmentFinalV1.model_validate(final.model_dump(), strict=True) == final
    assert _nested_keys(final.model_dump()).isdisjoint(
        {
            "answer_key",
            "accepted_answers",
            "canonical_correct_answer",
            "answer_explanation",
        }
    )


@pytest.mark.anyio
async def test_wrong_attempt_requires_classification_and_complete_adaptive_tasks():
    service, _, classifier, generator = _service()

    final = await service.submit(
        thread_id=THREAD_ID,
        attempt=_attempt(answer="5"),
        checkpoint=_checkpoint(),
    )

    assert final.terminal_status == "incorrect"
    assert final.is_correct is False
    assert final.error_classification == _classification()
    assert final.adaptive_tasks == _adaptive_batch().tasks
    classifier.assert_awaited_once()
    generator.assert_awaited_once()
    evaluation = classifier.await_args.args[0]
    assert evaluation.card == _card()
    assert evaluation.canonical_correct_answer == "4"
    adaptive_input = generator.await_args.args[0]
    assert adaptive_input.classification == _classification()
    assert _nested_keys(final.model_dump()).isdisjoint(
        {
            "answer_key",
            "accepted_answers",
            "canonical_correct_answer",
            "answer_explanation",
        }
    )


@pytest.mark.anyio
async def test_same_request_is_replayed_once_even_when_submitted_concurrently():
    service, journal, classifier, generator = _service()
    attempt = _attempt(answer="5")
    checkpoint = _checkpoint()

    first, second = await asyncio.gather(
        service.submit(
            thread_id=THREAD_ID,
            attempt=attempt,
            checkpoint=checkpoint,
        ),
        service.submit(
            thread_id=THREAD_ID,
            attempt=attempt,
            checkpoint=checkpoint,
        ),
    )

    assert first == second
    assert journal.operation_count == 1
    classifier.assert_awaited_once()
    generator.assert_awaited_once()


@pytest.mark.anyio
async def test_request_id_reuse_with_different_answer_is_a_conflict():
    service, journal, classifier, generator = _service()
    checkpoint = _checkpoint()
    await service.submit(
        thread_id=THREAD_ID,
        attempt=_attempt(answer="5"),
        checkpoint=checkpoint,
    )

    with pytest.raises(AssessmentRequestConflict):
        await service.submit(
            thread_id=THREAD_ID,
            attempt=_attempt(answer="6"),
            checkpoint=checkpoint,
        )

    assert journal.operation_count == 1
    classifier.assert_awaited_once()
    generator.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("thread_id", "attempt", "reason"),
    [
        ("different-thread", _attempt(), "thread_mismatch"),
        (
            THREAD_ID,
            _attempt(resource_id=OTHER_RESOURCE_ID),
            "resource_not_found",
        ),
        (
            THREAD_ID,
            _attempt(question_id=OTHER_QUESTION_ID),
            "question_not_found",
        ),
    ],
)
async def test_thread_resource_and_question_identity_are_checkpoint_bound(
    thread_id: str,
    attempt: AssessmentAttemptV1,
    reason: str,
):
    service, journal, classifier, generator = _service()

    with pytest.raises(AssessmentIdentityError) as exc_info:
        await service.submit(
            thread_id=thread_id,
            attempt=attempt,
            checkpoint=_checkpoint(),
        )

    assert exc_info.value.reason == reason
    assert journal.records == {}
    classifier.assert_not_awaited()
    generator.assert_not_awaited()


@pytest.mark.anyio
async def test_dependency_failure_is_sanitized_and_not_cached():
    classifier = AsyncMock(
        side_effect=[
            RuntimeError(
                "provider failed Authorization: Bearer secret-token answer=private"
            ),
            _classification(),
        ]
    )
    service, journal, _, generator = _service(classifier=classifier)
    attempt = _attempt(answer="private")

    with pytest.raises(AssessmentDependencyFailed) as exc_info:
        await service.submit(
            thread_id=THREAD_ID,
            attempt=attempt,
            checkpoint=_checkpoint(),
        )

    message = str(exc_info.value)
    assert exc_info.value.stage == "error_classification"
    assert "secret-token" not in message
    assert "private" not in message
    assert journal.records == {}

    final = await service.submit(
        thread_id=THREAD_ID,
        attempt=attempt,
        checkpoint=_checkpoint(),
    )
    assert final.terminal_status == "incorrect"
    assert classifier.await_count == 2
    generator.assert_awaited_once()


@pytest.mark.anyio
async def test_invalid_adaptive_output_fails_without_default_or_cached_terminal():
    invalid_batch = {
        "schema_version": "adaptive_practice_batch_v1",
        "tasks": (
            {
                "schema_version": "adaptive_practice_task_v1",
                "question_id": ADAPTIVE_QUESTION_ID,
                "task_type": "review",
                "question": "What is 1 + 1?",
                "answer": "2",
                "explanation": "One and one make two.",
                "reason": "   ",
                "tags": ("arithmetic",),
                "difficulty": 0.1,
            },
        ),
    }
    generator = AsyncMock(side_effect=[invalid_batch, _adaptive_batch()])
    service, journal, classifier, _ = _service(generator=generator)
    attempt = _attempt(answer="5")

    with pytest.raises(AssessmentDependencyFailed) as exc_info:
        await service.submit(
            thread_id=THREAD_ID,
            attempt=attempt,
            checkpoint=_checkpoint(),
        )

    assert exc_info.value.stage == "adaptive_practice"
    assert journal.records == {}

    final = await service.submit(
        thread_id=THREAD_ID,
        attempt=attempt,
        checkpoint=_checkpoint(),
    )
    assert final.adaptive_tasks == _adaptive_batch().tasks
    assert classifier.await_count == 2
    assert generator.await_count == 2


def test_request_and_public_card_forbid_schema_drift_and_answer_material():
    attempt_payload = _attempt().model_dump()
    attempt_payload["unexpected"] = True
    with pytest.raises(ValidationError):
        AssessmentAttemptV1.model_validate(attempt_payload, strict=True)

    public_payload = _card().model_dump()
    assert "answer" not in public_payload
    assert "answer_key" not in public_payload
    assert "accepted_answers" not in public_payload
    public_payload["answer"] = "4"
    with pytest.raises(ValidationError):
        PublicExerciseCardV1.model_validate(public_payload, strict=True)


def test_answer_matching_uses_only_the_explicit_server_policy():
    exact_key = _answer_key(match_mode="exact")
    folded_key = PrivateExerciseAnswerKeyV1(
        schema_version="exercise_answer_key_v1",
        question_id=QUESTION_ID,
        accepted_answers=("Four",),
        match_mode="trimmed_casefold",
        answer_explanation="The expected word is four.",
    )

    assert answer_matches(submitted_answer="4", answer_key=exact_key)
    assert not answer_matches(submitted_answer=" 4 ", answer_key=exact_key)
    assert answer_matches(submitted_answer="  FOUR ", answer_key=folded_key)


def test_adaptive_tasks_reject_blank_required_content():
    payload = _adaptive_batch().tasks[0].model_dump()
    for field in ("question", "answer", "explanation", "reason"):
        invalid = {**payload, field: "   "}
        with pytest.raises(ValidationError):
            AdaptivePracticeTaskV1.model_validate(invalid, strict=True)
