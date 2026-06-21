"""
Error Classifier — LLM-based classification of quiz errors.

Classifies each incorrect answer into one of three root causes:
- concept: User doesn't understand the underlying concept
- logic: User understands the concept but applied wrong reasoning
- implementation: User has right logic but wrong syntax, details, or calculation

Uses invoke_structured_llm with ErrorClassificationStrict schema.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.assessment.types import (
    ErrorClassification,
    ErrorClassificationStrict,
    QuizAttemptResult,
)
from src.llm.structured_output import (
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)

logger = logging.getLogger(__name__)

_LLM_NODE = "error_classifier"

ERROR_CLASSIFIER_SYSTEM_PROMPT = """\
You are an expert learning diagnostician. Your task is to analyze a student's \
incorrect answer and classify the root cause of the error.

Classification categories:
1. **concept** — The student fundamentally misunderstands the concept or theory. \
They don't know WHAT the correct approach is, not just how to apply it.
2. **logic** — The student understands the concept but applied the wrong reasoning, \
chose the wrong approach, or made a logical error in their solution pathway.
3. **implementation** — The student has correct understanding and logic, but made \
a mistake in syntax, calculation, or specific implementation details.

For each classification:
- Provide a specific concept_gap describing exactly what knowledge is missing
- Provide a concrete suggestion for how to address this error
- Give a confidence score (0.0–1.0) for your classification

IMPORTANT: Be precise and specific. Don't just say "needs more practice" — \
say exactly what concept or skill needs work.
"""


async def classify_error(
    quiz_result: QuizAttemptResult,
) -> ErrorClassification:
    """Classify a failed quiz attempt's error type using LLM.

    Args:
        quiz_result: The failed quiz attempt with question, user_answer,
                     correct_answer, and knowledge points.

    Returns:
        ErrorClassification with error_type, concept_gap, suggestion, and confidence.
    """
    if quiz_result.is_correct:
        return ErrorClassification(
            error_type="implementation",
            concept_gap="",
            suggestion="Answer was correct; no error to classify.",
            confidence=1.0,
            quiz_topic=quiz_result.topic,
            quiz_question=quiz_result.question,
            quiz_knowledge_points=list(quiz_result.knowledge_points),
        )

    # Build the prompt
    user_prompt = (
        f"Topic: {quiz_result.topic or 'unknown'}\n"
        f"Difficulty: {quiz_result.difficulty_level}\n"
        f"Knowledge Points: {', '.join(quiz_result.knowledge_points) if quiz_result.knowledge_points else 'unknown'}\n\n"
        f"Question:\n{quiz_result.question[:500]}\n\n"
        f"Student's Answer:\n{quiz_result.user_answer[:500]}\n\n"
        f"Correct Answer:\n{quiz_result.correct_answer[:500]}"
    )

    try:
        result = await invoke_structured_llm(
            node_name=_LLM_NODE,
            llm_node=_LLM_NODE,
            schema=ErrorClassificationStrict,
            messages=[
                SystemMessage(content=ERROR_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
            output_mode=get_llm_output_mode(_LLM_NODE),
            fallback_modes=[],
            state={"thread_id": quiz_result.user_id},
            max_raw_chars=get_max_raw_chars(_LLM_NODE),
        )

        if not result.success or result.parsed is None:
            logger.warning("Error classification LLM returned no valid output, using default")
            return _default_classification(quiz_result)

        parsed = result.parsed
        return ErrorClassification(
            error_type=parsed.error_type,
            concept_gap=parsed.concept_gap,
            suggestion=parsed.suggestion,
            confidence=parsed.confidence,
            quiz_topic=quiz_result.topic,
            quiz_question=quiz_result.question,
            quiz_knowledge_points=list(quiz_result.knowledge_points),
        )

    except Exception as exc:
        logger.exception("Error classification LLM call failed: %s", exc)
        return _default_classification(quiz_result)


def _default_classification(quiz_result: QuizAttemptResult) -> ErrorClassification:
    """Return a conservative default classification when LLM is unavailable."""
    return ErrorClassification(
        error_type="concept",
        concept_gap=f"需要复习 {quiz_result.topic or 'this topic'} 的相关概念",
        suggestion=f"建议重新学习 {', '.join(quiz_result.knowledge_points) if quiz_result.knowledge_points else quiz_result.topic}",
        confidence=0.3,
        quiz_topic=quiz_result.topic,
        quiz_question=quiz_result.question,
        quiz_knowledge_points=list(quiz_result.knowledge_points),
    )
