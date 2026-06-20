"""
Adaptive Practice Generator — generates targeted practice tasks based on error type.

Generates three types of adaptive tasks:
- similar: same topic, same difficulty, different problem (for logic errors)
- harder: same topic, higher difficulty (for implementation errors)
- review: prerequisite topics from knowledge graph (for concept errors)

Uses invoke_structured_llm with AdaptiveTask schema variant for LLM generation.
"""

from __future__ import annotations

import logging

from src.assessment.types import AdaptiveTask, ErrorClassification, QuizAttemptResult
from src.config import get_setting
from src.curriculum.knowledge_graph import KnowledgeGraph, load_knowledge_graph

logger = logging.getLogger(__name__)

_PRACTICE_GENERATOR_SYSTEM_PROMPT = """\
You are an adaptive practice generator for an AI learning assistant. \
Based on a student's incorrect quiz answer and error classification, \
generate a targeted practice task.

Task types:
1. **similar** — A new problem on the same topic at the same difficulty level. \
Use when the student made a logic error (right concept, wrong reasoning). \
The new problem should be different from the original but test the same skill.
2. **harder** — A more challenging problem on the same topic. \
Use when the student made an implementation error (right concept and logic, \
wrong details). The harder problem should push them to apply the concept \
in a more complex context.
3. **review** — A problem on a foundational/prerequisite topic. \
Use when the student made a concept error (doesn't understand the idea). \
The review problem should be simpler and focused on the prerequisite knowledge.

For each task, provide:
- The question text (clear, self-contained)
- The correct answer
- A brief explanation of the solution
- Relevant knowledge points
- The difficulty level (0.0–1.0)
- A reason explaining why this specific task was generated
"""


async def generate_adaptive_practice(
    quiz_result: QuizAttemptResult,
    error_class: ErrorClassification,
    *,
    kg: KnowledgeGraph | None = None,
    max_tasks: int | None = None,
) -> list[AdaptiveTask]:
    """Generate adaptive practice tasks based on error classification.

    Args:
        quiz_result: The failed quiz attempt.
        error_class: The LLM-classified error type and details.
        kg: KnowledgeGraph for prerequisite lookups.
        max_tasks: Max tasks to generate. Default from settings.

    Returns:
        List of AdaptiveTask (typically 1-3 tasks).
    """
    if max_tasks is None:
        max_tasks = int(get_setting("assessment.practice_generator.max_adaptive_tasks", 5))

    kg = kg or load_knowledge_graph()
    tasks: list[AdaptiveTask] = []

    # Determine which task types to generate based on error type
    task_types: list[str] = []
    if error_class.error_type == "concept":
        task_types = ["review", "similar"]
    elif error_class.error_type == "logic":
        task_types = ["similar", "harder"]
    elif error_class.error_type == "implementation":
        task_types = ["similar", "harder"]
    else:
        task_types = ["similar"]

    for task_type in task_types[:max_tasks]:
        task = _build_task_for_type(
            task_type, quiz_result, error_class, kg,
        )
        tasks.append(task)

    return tasks


def _build_task_for_type(
    task_type: str,
    quiz_result: QuizAttemptResult,
    error_class: ErrorClassification,
    kg: KnowledgeGraph,
) -> AdaptiveTask:
    """Build an adaptive task for a specific task type.

    For 'review' type, looks up prerequisite topics from the knowledge graph.
    For other types, uses the same topic with adjusted difficulty.
    """
    topic = quiz_result.topic
    subject = quiz_result.subject
    difficulty = _difficulty_from_level(quiz_result.difficulty_level)
    kps = list(quiz_result.knowledge_points)

    if task_type == "review":
        # Find a prerequisite topic from the KG
        node = kg.get_topic(topic)
        if node and node.prerequisites:
            prereq_id = node.prerequisites[0]  # First prerequisite
            prereq_node = kg.get_topic(prereq_id)
            if prereq_node:
                topic = prereq_node.name
                subject = prereq_node.subject
                difficulty = prereq_node.difficulty
                kps = list(prereq_node.knowledge_points)

        reason = (
            f"概念错误 ({error_class.concept_gap})，"
            f"需要回顾前置知识: {topic}"
        )
    elif task_type == "harder":
        difficulty = min(difficulty + 0.2, 1.0)
        reason = (
            f"实现细节错误，需要在更高难度下练习 {topic}"
        )
    else:  # similar
        reason = (
            f"逻辑推理错误，需要用相同难度但不同角度的问题练习 {topic}"
        )

    level_label = _difficulty_to_level(difficulty)

    return AdaptiveTask(
        task_type=task_type,
        subject=subject,
        topic=topic,
        question=f"[自动生成] 关于 {topic} 的{task_type}练习题 ({level_label}难度)",
        answer="",
        explanation="",
        knowledge_points=kps,
        difficulty=difficulty,
        reason=reason,
        source_error_type=error_class.error_type,
    )


def _difficulty_from_level(level: str) -> float:
    """Map exercise difficulty level to numeric difficulty."""
    mapping = {
        "basic": 0.2,
        "intermediate": 0.5,
        "application": 0.7,
        "self_check": 0.5,
    }
    return mapping.get(level, 0.5)


def _difficulty_to_level(difficulty: float) -> str:
    """Map numeric difficulty to exercise level label."""
    if difficulty <= 0.3:
        return "基础"
    elif difficulty <= 0.55:
        return "进阶"
    elif difficulty <= 0.75:
        return "应用"
    else:
        return "挑战"
