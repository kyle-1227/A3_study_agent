"""Fail-fast domain service for strict assessment attempt submission."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from src.assessment.attempt_contracts import (
    AdaptivePracticeBatchV1,
    AdaptivePracticeInputV1,
    AssessmentAttemptV1,
    AssessmentCheckpointResourcesV2,
    AssessmentErrorClassificationV1,
    AssessmentEvaluationInputV1,
    AssessmentFinalV1,
    AssessmentQuestionRecordV1,
    AssessmentResourceRecordV2,
    answer_matches,
    build_assessment_final_v1,
    stable_assessment_attempt_hash,
)

AssessmentOperation = Callable[[], Awaitable[AssessmentFinalV1]]
ModelT = TypeVar("ModelT", bound=BaseModel)


class AssessmentErrorClassifier(Protocol):
    """Injected, provider-independent classifier boundary."""

    async def __call__(self, evaluation: AssessmentEvaluationInputV1, /) -> object:
        """Return data that strictly validates as AssessmentErrorClassificationV1."""


class AdaptivePracticeGenerator(Protocol):
    """Injected, provider-independent adaptive-practice boundary."""

    async def __call__(self, request: AdaptivePracticeInputV1, /) -> object:
        """Return data that strictly validates as AdaptivePracticeBatchV1."""


class AssessmentIdempotencyExecutor(Protocol):
    """Atomic durable execution boundary for one assessment request.

    Implementations must serialize ``(thread_id, request_id)`` operations, replay
    the stored final for an identical request hash, raise
    ``AssessmentRequestConflict`` for a different hash, persist a content-free
    claim before ``operation``, and persist a safe failed terminal when the
    operation raises. An unresolved claim must block automatic redispatch. The API
    adapter binds this protocol to the durable checkpoint journal and the
    checkpointer-specific execution lock.
    """

    async def execute_once(
        self,
        *,
        thread_id: str,
        request_id: str,
        request_hash: str,
        operation: AssessmentOperation,
    ) -> object:
        """Execute or replay one request atomically."""


class AssessmentAttemptServiceError(RuntimeError):
    """Base typed error for assessment submission failures."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AssessmentIdentityError(AssessmentAttemptServiceError):
    """Raised when route, checkpoint, resource, or question identity disagrees."""

    def __init__(
        self,
        reason: Literal[
            "thread_mismatch",
            "resource_not_found",
            "question_not_found",
        ],
    ) -> None:
        self.reason = reason
        super().__init__(
            code=f"assessment_{reason}",
            message=f"assessment identity validation failed: {reason}",
        )


class AssessmentRequestConflict(AssessmentAttemptServiceError):
    """Raised when one request_id is reused with different request content."""

    def __init__(self) -> None:
        super().__init__(
            code="assessment_request_id_conflict",
            message="assessment request_id was already used with different content",
        )


class AssessmentRecoveryRequired(AssessmentAttemptServiceError):
    """Raised when a durable pre-dispatch claim has no recorded terminal."""

    def __init__(self) -> None:
        super().__init__(
            code="assessment_request_recovery_required",
            message="assessment request has an unresolved durable execution claim",
        )


class AssessmentRecordedFailure(AssessmentAttemptServiceError):
    """Replay one content-free failure terminal without executing dependencies."""

    def __init__(
        self,
        *,
        code: str,
        stage: Literal[
            "assessment",
            "idempotency",
            "error_classification",
            "adaptive_practice",
        ],
        exception_type: str,
    ) -> None:
        self.stage = stage
        self.exception_type = exception_type
        super().__init__(
            code=code,
            message=(
                "assessment request previously failed: "
                f"stage={stage}, exception_type={exception_type}"
            ),
        )


class AssessmentDependencyFailed(AssessmentAttemptServiceError):
    """Raised when an injected dependency fails or violates its strict contract."""

    def __init__(
        self,
        *,
        stage: Literal[
            "idempotency",
            "error_classification",
            "adaptive_practice",
        ],
        exception_type: str,
    ) -> None:
        self.stage = stage
        self.exception_type = exception_type
        super().__init__(
            code=f"assessment_{stage}_failed",
            message=(
                f"assessment dependency failed: stage={stage}, "
                f"exception_type={exception_type}"
            ),
        )


class AssessmentAttemptService:
    """Validate, evaluate, and finalize one exercise-card answer."""

    def __init__(
        self,
        *,
        idempotency: AssessmentIdempotencyExecutor,
        error_classifier: AssessmentErrorClassifier,
        adaptive_generator: AdaptivePracticeGenerator,
    ) -> None:
        self._idempotency = idempotency
        self._error_classifier = error_classifier
        self._adaptive_generator = adaptive_generator

    async def submit(
        self,
        *,
        thread_id: str,
        attempt: AssessmentAttemptV1,
        checkpoint: AssessmentCheckpointResourcesV2,
    ) -> AssessmentFinalV1:
        """Submit one strict attempt through the injected atomic journal."""

        request_hash = stable_assessment_attempt_hash(
            thread_id=thread_id,
            attempt=attempt,
        )
        if checkpoint.thread_id != thread_id:
            raise AssessmentIdentityError("thread_mismatch")
        resource = _find_resource(checkpoint, resource_id=attempt.resource_id)
        question = _find_question(resource, question_id=attempt.question_id)

        async def operation() -> AssessmentFinalV1:
            return await self._evaluate(
                thread_id=thread_id,
                attempt=attempt,
                question=question,
            )

        try:
            raw_final = await self._idempotency.execute_once(
                thread_id=thread_id,
                request_id=attempt.request_id,
                request_hash=request_hash,
                operation=operation,
            )
        except AssessmentAttemptServiceError:
            raise
        except Exception as exc:
            raise AssessmentDependencyFailed(
                stage="idempotency",
                exception_type=type(exc).__name__,
            ) from exc

        final = _validate_dependency_output(
            model=AssessmentFinalV1,
            value=raw_final,
            stage="idempotency",
        )
        if (
            final.thread_id != thread_id
            or final.request_id != attempt.request_id
            or final.resource_id != attempt.resource_id
            or final.question_id != attempt.question_id
        ):
            raise AssessmentDependencyFailed(
                stage="idempotency",
                exception_type="AssessmentFinalIdentityMismatch",
            )
        return final

    async def _evaluate(
        self,
        *,
        thread_id: str,
        attempt: AssessmentAttemptV1,
        question: AssessmentQuestionRecordV1,
    ) -> AssessmentFinalV1:
        is_correct = answer_matches(
            submitted_answer=attempt.answer,
            answer_key=question.answer_key,
        )
        if is_correct:
            return build_assessment_final_v1(
                thread_id=thread_id,
                attempt=attempt,
                is_correct=True,
                error_classification=None,
                adaptive_tasks=(),
            )

        evaluation = AssessmentEvaluationInputV1(
            schema_version="assessment_evaluation_input_v1",
            thread_id=thread_id,
            attempt=attempt,
            card=question.card,
            canonical_correct_answer=question.answer_key.accepted_answers[0],
            answer_explanation=question.answer_key.answer_explanation,
        )
        try:
            raw_classification = await self._error_classifier(evaluation)
        except AssessmentAttemptServiceError:
            raise
        except Exception as exc:
            raise AssessmentDependencyFailed(
                stage="error_classification",
                exception_type=type(exc).__name__,
            ) from exc
        classification = _validate_dependency_output(
            model=AssessmentErrorClassificationV1,
            value=raw_classification,
            stage="error_classification",
        )

        adaptive_input = AdaptivePracticeInputV1(
            schema_version="adaptive_practice_input_v1",
            evaluation=evaluation,
            classification=classification,
        )
        try:
            raw_batch = await self._adaptive_generator(adaptive_input)
        except AssessmentAttemptServiceError:
            raise
        except Exception as exc:
            raise AssessmentDependencyFailed(
                stage="adaptive_practice",
                exception_type=type(exc).__name__,
            ) from exc
        batch = _validate_dependency_output(
            model=AdaptivePracticeBatchV1,
            value=raw_batch,
            stage="adaptive_practice",
        )

        return build_assessment_final_v1(
            thread_id=thread_id,
            attempt=attempt,
            is_correct=False,
            error_classification=classification,
            adaptive_tasks=batch.tasks,
        )


def _find_resource(
    checkpoint: AssessmentCheckpointResourcesV2,
    *,
    resource_id: str,
) -> AssessmentResourceRecordV2:
    for resource in checkpoint.resources:
        if resource.resource_id == resource_id:
            return resource
    raise AssessmentIdentityError("resource_not_found")


def _find_question(
    resource: AssessmentResourceRecordV2,
    *,
    question_id: str,
) -> AssessmentQuestionRecordV1:
    for question in resource.questions:
        if question.card.question_id == question_id:
            return question
    raise AssessmentIdentityError("question_not_found")


def _validate_dependency_output(
    *,
    model: type[ModelT],
    value: object,
    stage: Literal[
        "idempotency",
        "error_classification",
        "adaptive_practice",
    ],
) -> ModelT:
    try:
        return model.model_validate(value, strict=True)
    except (TypeError, ValidationError) as exc:
        raise AssessmentDependencyFailed(
            stage=stage,
            exception_type=type(exc).__name__,
        ) from exc


__all__ = [
    "AdaptivePracticeGenerator",
    "AssessmentAttemptService",
    "AssessmentAttemptServiceError",
    "AssessmentDependencyFailed",
    "AssessmentErrorClassifier",
    "AssessmentIdentityError",
    "AssessmentIdempotencyExecutor",
    "AssessmentOperation",
    "AssessmentRecordedFailure",
    "AssessmentRecoveryRequired",
    "AssessmentRequestConflict",
]
