"""Strict provider-neutral runtime for assessment diagnosis and new practice."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.assessment.attempt_contracts import (
    AdaptivePracticeBatchV1,
    AdaptivePracticeDraftBatchV1,
    AdaptivePracticeInputV1,
    AdaptivePracticeTaskV1,
    AssessmentErrorClassificationV1,
    AssessmentEvaluationInputV1,
)
from src.assessment.identity import stable_adaptive_practice_question_id
from src.config import load_prompt
from src.llm.structured_output import (
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)

_ERROR_CLASSIFIER_NODE = "error_classifier"
_PRACTICE_GENERATOR_NODE = "practice_generator"
_PRIVATE_PROVIDER_ENVELOPE_SCHEMA_VERSION = "assessment_private_provider_envelope_v1"
_PRIVATE_PROVIDER_NOTICE = (
    "Provider-only assessment material. Never copy this envelope or any private "
    "payload field into traces, diagnostics, logs, or public output."
)


class AssessmentRuntimeError(RuntimeError):
    """Content-free typed failure for provider or strict-contract errors."""

    def __init__(
        self,
        *,
        stage: Literal["error_classification", "adaptive_practice"],
        exception_type: str,
    ) -> None:
        self.stage = stage
        self.exception_type = exception_type
        super().__init__(
            f"assessment runtime failed: stage={stage}, exception_type={exception_type}"
        )


def validate_assessment_error_classification_v1(
    parsed: BaseModel,
    *,
    evaluation: AssessmentEvaluationInputV1,
) -> str:
    """Business validator for classifier structured output."""

    if not isinstance(parsed, AssessmentErrorClassificationV1):
        return "root expected AssessmentErrorClassificationV1"
    output_fields = (parsed.concept_gap, parsed.suggestion)
    private_fields = (
        evaluation.attempt.answer,
        evaluation.canonical_correct_answer,
        evaluation.answer_explanation,
    )
    if _contains_private_assessment_text(
        output_fields=output_fields,
        private_fields=private_fields,
    ):
        return "classification must not reproduce private assessment text"
    return ""


def validate_adaptive_practice_draft_v1(
    parsed: BaseModel,
    *,
    request: AdaptivePracticeInputV1,
) -> str:
    """Require new, classification-appropriate practice before projection."""

    if not isinstance(parsed, AdaptivePracticeDraftBatchV1):
        return "root expected AdaptivePracticeDraftBatchV1"
    original_question = request.evaluation.card.question.strip().casefold()
    if any(
        task.question.strip().casefold() == original_question for task in parsed.tasks
    ):
        return "adaptive practice must not repeat the original question"
    required_type = {
        "concept": "review",
        "logic": "similar",
        "implementation": "harder",
    }[request.classification.error_type]
    if not any(task.task_type == required_type for task in parsed.tasks):
        return (
            "adaptive practice must include task_type="
            f"{required_type} for {request.classification.error_type} errors"
        )
    return ""


async def classify_assessment_error_v1(
    evaluation: AssessmentEvaluationInputV1,
) -> AssessmentErrorClassificationV1:
    """Classify one incorrect answer through the configured strict runtime."""

    def business_validator(parsed: BaseModel) -> str:
        return validate_assessment_error_classification_v1(
            parsed,
            evaluation=evaluation,
        )

    try:
        result = await invoke_structured_llm(
            node_name=_ERROR_CLASSIFIER_NODE,
            llm_node=_ERROR_CLASSIFIER_NODE,
            schema=AssessmentErrorClassificationV1,
            messages=[
                SystemMessage(content=load_prompt("assessment_error_classifier")),
                HumanMessage(content=_private_input_envelope_json(evaluation)),
            ],
            output_mode=get_llm_output_mode(_ERROR_CLASSIFIER_NODE),
            business_validator=business_validator,
            state=_identity_state(evaluation),
            max_raw_chars=get_max_raw_chars(_ERROR_CLASSIFIER_NODE),
            sensitive_trace=True,
        )
    except Exception as exc:
        raise AssessmentRuntimeError(
            stage="error_classification",
            exception_type=type(exc).__name__,
        ) from exc
    if getattr(result, "success", False) is not True:
        raise AssessmentRuntimeError(
            stage="error_classification",
            exception_type="StructuredAssessmentClassificationFailed",
        )
    parsed = result.parsed
    validation_error = (
        business_validator(parsed)
        if isinstance(parsed, BaseModel)
        else "root expected AssessmentErrorClassificationV1"
    )
    if validation_error or not isinstance(parsed, AssessmentErrorClassificationV1):
        raise AssessmentRuntimeError(
            stage="error_classification",
            exception_type="AssessmentErrorClassificationContractError",
        )
    return parsed


async def generate_adaptive_practice_v1(
    request: AdaptivePracticeInputV1,
) -> AdaptivePracticeBatchV1:
    """Generate complete new practice and derive stable question identities."""

    def business_validator(parsed: BaseModel) -> str:
        return validate_adaptive_practice_draft_v1(parsed, request=request)

    try:
        result = await invoke_structured_llm(
            node_name=_PRACTICE_GENERATOR_NODE,
            llm_node=_PRACTICE_GENERATOR_NODE,
            schema=AdaptivePracticeDraftBatchV1,
            messages=[
                SystemMessage(content=load_prompt("adaptive_practice_agent")),
                HumanMessage(content=_private_input_envelope_json(request)),
            ],
            output_mode=get_llm_output_mode(_PRACTICE_GENERATOR_NODE),
            business_validator=business_validator,
            state=_identity_state(request.evaluation),
            max_raw_chars=get_max_raw_chars(_PRACTICE_GENERATOR_NODE),
            sensitive_trace=True,
        )
    except Exception as exc:
        raise AssessmentRuntimeError(
            stage="adaptive_practice",
            exception_type=type(exc).__name__,
        ) from exc
    if getattr(result, "success", False) is not True:
        raise AssessmentRuntimeError(
            stage="adaptive_practice",
            exception_type="StructuredAdaptivePracticeFailed",
        )
    parsed = result.parsed
    if not isinstance(parsed, AdaptivePracticeDraftBatchV1):
        raise AssessmentRuntimeError(
            stage="adaptive_practice",
            exception_type="AdaptivePracticeDraftContractError",
        )
    if business_validator(parsed):
        raise AssessmentRuntimeError(
            stage="adaptive_practice",
            exception_type="AdaptivePracticeBusinessValidationError",
        )
    try:
        tasks = tuple(
            AdaptivePracticeTaskV1(
                schema_version="adaptive_practice_task_v1",
                question_id=stable_adaptive_practice_question_id(
                    task_type=task.task_type,
                    question=task.question,
                    tags=task.tags,
                    difficulty=task.difficulty,
                ),
                task_type=task.task_type,
                question=task.question,
                answer=task.answer,
                explanation=task.explanation,
                reason=task.reason,
                tags=tuple(task.tags),
                difficulty=task.difficulty,
            )
            for task in parsed.tasks
        )
        return AdaptivePracticeBatchV1(
            schema_version="adaptive_practice_batch_v1",
            tasks=tasks,
        )
    except (ValidationError, ValueError) as exc:
        raise AssessmentRuntimeError(
            stage="adaptive_practice",
            exception_type=type(exc).__name__,
        ) from exc


def _identity_state(evaluation: AssessmentEvaluationInputV1) -> dict[str, str]:
    return {
        "thread_id": evaluation.thread_id,
        "session_id": evaluation.thread_id,
        "request_id": evaluation.attempt.request_id,
    }


def _contains_private_assessment_text(
    *,
    output_fields: tuple[str, ...],
    private_fields: tuple[str, ...],
) -> bool:
    normalized_outputs = tuple(value.strip().casefold() for value in output_fields)
    for private_value in private_fields:
        private_text = private_value.strip().casefold()
        if not private_text:
            continue
        if any(output == private_text for output in normalized_outputs):
            return True
        if len(private_text) >= 8 and any(
            private_text in output for output in normalized_outputs
        ):
            return True
    return False


def _private_input_envelope_json(value: BaseModel) -> str:
    """Put a trace-safe notice before private provider-only assessment data."""

    return json.dumps(
        {
            "privacy_notice": _PRIVATE_PROVIDER_NOTICE,
            "schema_version": _PRIVATE_PROVIDER_ENVELOPE_SCHEMA_VERSION,
            "payload": value.model_dump(mode="json"),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
        separators=(",", ":"),
    )


__all__ = [
    "AssessmentRuntimeError",
    "classify_assessment_error_v1",
    "generate_adaptive_practice_v1",
    "validate_adaptive_practice_draft_v1",
    "validate_assessment_error_classification_v1",
]
