"""
Path Planner — dynamic learning path computation with skip/reinforce/repeat logic.

Integrates user profile skills, semantic memory weak points, and episodic
memory recent failures to classify each knowledge graph topic and produce
a personalized, ordered LearningPath.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.config import get_setting
from src.curriculum.knowledge_graph import KnowledgeGraph, load_knowledge_graph
from src.curriculum.types import KnowledgeNode, LearningPath, PathStep
from src.memory.storage import MemoryStore, create_memory_store
from src.profile.schema import UserProfile

logger = logging.getLogger(__name__)


# ── Primary API ────────────────────────────────────────────────────────────


async def compute_learning_path(
    user_id: str,
    goal_subject: str,
    goal_topics: list[str] | None = None,
    *,
    profile: UserProfile | None = None,
    store: MemoryStore | None = None,
    kg: KnowledgeGraph | None = None,
) -> LearningPath:
    """Compute a personalized learning path for a user.

    For each topic in the goal subject (and its prerequisite chain), determines
    whether the user should skip (mastered), reinforce (weak), repeat (recent failure),
    or learn (ready/blocked).

    Args:
        user_id: The user identifier.
        goal_subject: Target subject, e.g. "python", "machine_learning".
        goal_topics: Optional subset of topics to focus on. If None, uses all topics
                     in the subject that have no unsatisfied prerequisites outside.
        profile: UserProfile with skills, learning style, goals. Loaded if None.
        store: MemoryStore for recent failure lookup.
        kg: KnowledgeGraph instance. Loads singleton if None.

    Returns:
        LearningPath with ordered steps and status annotations.
    """
    kg = kg or load_knowledge_graph()
    store = store or create_memory_store()
    now = datetime.now(timezone.utc).isoformat()

    # Resolve goal topics
    if goal_topics is None:
        subject_nodes = kg.get_subject_topics(goal_subject)
        goal_topics = [n.topic_id for n in subject_nodes]

    if not goal_topics:
        logger.warning("No goal topics found for subject=%s", goal_subject)
        return LearningPath(user_id=user_id, generated_at=now, goal_subject=goal_subject)

    # Build the full topic set: goal topics + all transitive prerequisites
    all_topic_ids: set[str] = set()
    for tid in goal_topics:
        all_topic_ids.add(tid)
        for prereq in kg.get_all_prerequisites(tid):
            all_topic_ids.add(prereq.topic_id)

    # Topologically sort the full set
    sorted_topics = kg.topological_sort(list(all_topic_ids))

    # Get user skill data from profile
    skill_levels = _get_skill_levels(profile)
    skill_confidences = _get_skill_confidences(profile)

    # Get weak points from semantic memory
    weak_points = await _get_weak_points(user_id, store)

    # Get recent failures from episodic memory
    recent_failures = await _get_recent_failures(user_id, store)

    # Settings
    skip_threshold = float(get_setting("curriculum.path_planner.skip_threshold", 0.8))
    confidence_threshold = float(get_setting("curriculum.path_planner.confidence_threshold", 0.7))
    reinforce_min = float(get_setting("curriculum.path_planner.reinforce_min", 0.3))
    repeat_window_days = int(get_setting("curriculum.path_planner.repeat_window_days", 7))

    # Classify each topic
    steps: list[PathStep] = []
    for node in sorted_topics:
        level = skill_levels.get(node.topic_id, 0.0)
        conf = skill_confidences.get(node.topic_id, 0.0)
        is_mastered = level > skip_threshold and conf > confidence_threshold
        is_weak = (reinforce_min <= level <= skip_threshold) or _is_weak_point(node, weak_points)
        is_recent_failure = node.topic_id in recent_failures

        # Determine missing prerequisites
        missing_prereqs = _missing_prerequisites(node, steps)

        # Classify
        if is_recent_failure and not is_mastered:
            status = "repeat"
            reason = f"最近7天内有失败记录"
        elif is_mastered:
            status = "skip"
            reason = f"已掌握 (level={level:.2f}, confidence={conf:.2f})"
        elif is_weak:
            if missing_prereqs:
                status = "blocked"
                reason = f"薄弱知识点，但缺少前置: {', '.join(missing_prereqs)}"
            else:
                status = "reinforce"
                reason = f"需要巩固 (level={level:.2f})"
        elif missing_prereqs:
            status = "blocked"
            reason = f"前置知识未完成: {', '.join(missing_prereqs)}"
        else:
            status = "ready"
            reason = "前置已满足，可以开始学习"

        # Build recommended resources
        resources = _build_resource_recommendations(node, status, profile)

        # Compute priority: higher for blocks closer to goal, reinforced topics
        priority = _compute_priority(node, status, level, goal_topics)

        steps.append(PathStep(
            topic_id=node.topic_id,
            name=node.name,
            subject=node.subject,
            status=status,
            user_skill_level=level,
            skill_confidence=conf,
            topic_difficulty=node.difficulty,
            estimated_hours=node.estimated_hours,
            recommended_resources=resources,
            priority=priority,
            status_reason=reason,
            missing_prerequisites=missing_prereqs,
        ))

    # Build summary
    skip_count = sum(1 for s in steps if s.status == "skip")
    reinforce_count = sum(1 for s in steps if s.status == "reinforce")
    repeat_count = sum(1 for s in steps if s.status == "repeat")
    ready_count = sum(1 for s in steps if s.status == "ready")
    blocked_count = sum(1 for s in steps if s.status == "blocked")
    total_hours = sum(s.estimated_hours for s in steps if s.status != "skip")

    summary = (
        f"为 {goal_subject} 生成了包含 {len(steps)} 个主题的学习路径。"
        f"已掌握可跳过: {skip_count}, 可开始: {ready_count}, "
        f"需巩固: {reinforce_count}, 需重学: {repeat_count}, "
        f"暂不可达: {blocked_count}。"
        f"预计总学时: {total_hours:.0f}h (不含已跳过)"
    )

    logger.info(
        "Learning path for user=%s subject=%s: %d steps, %d skip, %d ready, %d reinforce, %d repeat, %d blocked",
        user_id, goal_subject, len(steps), skip_count, ready_count, reinforce_count, repeat_count, blocked_count,
    )

    return LearningPath(
        user_id=user_id,
        generated_at=now,
        goal_subject=goal_subject,
        goal_topics=goal_topics,
        steps=steps,
        estimated_total_hours=total_hours,
        summary=summary,
        skip_count=skip_count,
        reinforce_count=reinforce_count,
        repeat_count=repeat_count,
        ready_count=ready_count,
        blocked_count=blocked_count,
    )


# ── Context Builder for study_plan_planner ─────────────────────────────────


def build_curriculum_context(learning_path: LearningPath) -> str:
    """Build a natural-language context string for the study_plan_planner.

    This is injected into the planner's prompt so it can make KG-aware decisions
    about phase ordering, task assignment, and difficulty progression.
    """
    parts: list[str] = [f"[课程引擎上下文] 目标学科: {learning_path.goal_subject}"]

    # Skippable topics
    skipped = [s for s in learning_path.steps if s.status == "skip"]
    if skipped:
        names = ", ".join(s.name for s in skipped)
        parts.append(f"已掌握可跳过: {names}")

    # Actionable topics (ordered)
    actionable = learning_path.actionable_steps
    if actionable:
        parts.append("当前可学习的主题 (按优先级排序):")
        for s in actionable:
            status_label = {
                "ready": "✅ 可开始",
                "reinforce": "🔧 需巩固",
                "repeat": "🔄 需重学",
            }.get(s.status, s.status)
            parts.append(
                f"  - {status_label} {s.name} (难度={s.topic_difficulty:.1f}, "
                f"当前水平={s.user_skill_level:.1f}, 预计{s.estimated_hours:.0f}h)"
            )
            if s.recommended_resources:
                res_str = ", ".join(
                    f"{r.get('type', '')}:{r.get('name', '')}"
                    for r in s.recommended_resources[:3]
                )
                parts.append(f"    推荐资源: {res_str}")

    # Blocked topics
    blocked = [s for s in learning_path.steps if s.status == "blocked"]
    if blocked:
        parts.append("暂不可达的主题 (前置未满足):")
        for s in blocked:
            parts.append(f"  - 🚫 {s.name}: 缺少 {', '.join(s.missing_prerequisites)}")

    return "\n".join(parts)


# ── Helper functions ───────────────────────────────────────────────────────


def _get_skill_levels(profile: UserProfile | None) -> dict[str, float]:
    """Extract topic-level skill levels from profile.

    Maps profile skill keys (e.g., 'python', 'algorithm') to topic IDs.
    Uses substring matching for flexible mapping.
    """
    if profile is None:
        return {}
    levels: dict[str, float] = {}
    for skill_name, entry in profile.skills.items():
        levels[skill_name] = entry.level
    return levels


def _get_skill_confidences(profile: UserProfile | None) -> dict[str, float]:
    if profile is None:
        return {}
    confs: dict[str, float] = {}
    for skill_name, entry in profile.skills.items():
        confs[skill_name] = entry.confidence
    return confs


async def _get_weak_points(user_id: str, store: MemoryStore) -> list[str]:
    """Get weak knowledge points from the user's semantic memory summaries."""
    try:
        summaries = await store.get_semantic_for_user(user_id, limit=10)
        weak_points: list[str] = []
        for s in summaries:
            weak_points.extend(s.weak_knowledge_points or [])
        return weak_points
    except Exception as exc:
        logger.debug("Failed to get weak points for user=%s: %s", user_id, exc)
        return []


async def _get_recent_failures(user_id: str, store: MemoryStore) -> set[str]:
    """Get topic IDs with recent quiz failures (last N days)."""
    repeat_window_days = int(get_setting("curriculum.path_planner.repeat_window_days", 7))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=repeat_window_days)).isoformat()

    try:
        records = await store.query_episodic(
            user_id,
            memory_type="error",
            start_time=cutoff,
            importance_min=0.7,
            limit=100,
        )
        # Extract subject/topic from content — crude but effective
        failed_topics: set[str] = set()
        for rec in records:
            subject = rec.subject or ""
            # Try to match topics from the content
            content_lower = rec.content.lower()
            # Look for common topic patterns in content
            if "recursion" in content_lower or "递归" in content_lower:
                failed_topics.add("python_algorithms")
            if "oop" in content_lower or "面向对象" in content_lower:
                failed_topics.add("python_oop")
            if "list comprehension" in content_lower or "列表推导" in content_lower:
                failed_topics.add("python_data_structures")
            if "algorithm" in content_lower or "算法" in content_lower:
                failed_topics.add("python_algorithms")
        return failed_topics
    except Exception as exc:
        logger.debug("Failed to get recent failures for user=%s: %s", user_id, exc)
        return set()


def _is_weak_point(node: KnowledgeNode, weak_points: list[str]) -> bool:
    """Check if the topic's knowledge points overlap with user's weak points."""
    if not weak_points:
        return False
    kp_lower = {kp.lower() for kp in node.knowledge_points}
    wp_lower = {wp.lower() for wp in weak_points}
    return bool(kp_lower & wp_lower)


def _missing_prerequisites(node: KnowledgeNode, completed_steps: list[PathStep]) -> list[str]:
    """Find prerequisites that are not yet skipped/ready/reinforce."""
    satisfied = {s.topic_id for s in completed_steps if s.status != "blocked"}
    missing = [pid for pid in node.prerequisites if pid not in satisfied]
    return missing


def _build_resource_recommendations(
    node: KnowledgeNode,
    status: str,
    profile: UserProfile | None,
) -> list[dict]:
    """Build recommended resource list based on status and learning style."""
    resources: list[dict] = []
    prefs = profile.learning_style if profile else None

    # Determine preferred order
    type_order = ["quiz", "mindmap", "doc", "case"]
    if prefs:
        if prefs.prefer_practice > 0.6:
            type_order = ["quiz", "case", "mindmap", "doc"]
        if prefs.prefer_visual > 0.6:
            type_order = ["mindmap", "doc", "quiz", "case"]

    # Add resources based on status
    if status == "repeat":
        # Focus on review docs and foundational quizzes
        for rtype in ["doc", "quiz", "mindmap"]:
            for name in node.resources.get(rtype, []):
                resources.append({"type": rtype, "name": name, "reason": "重学: 巩固基础"})
    elif status == "reinforce":
        for rtype in type_order:
            for name in node.resources.get(rtype, []):
                resources.append({"type": rtype, "name": name, "reason": "巩固: 强化薄弱点"})
    elif status == "ready":
        for rtype in type_order:
            for name in node.resources.get(rtype, []):
                resources.append({"type": rtype, "name": name, "reason": "学习: 新知识掌握"})

    return resources[:5]  # Limit to 5 recommendations per step


def _compute_priority(
    node: KnowledgeNode,
    status: str,
    user_level: float,
    goal_topics: list[str],
) -> float:
    """Compute priority score for ordering steps.

    Higher priority for:
    - Repeat/reinforce (urgent)
    - Topics closer to the goal
    - Higher difficulty with lower user level (bigger gap)
    """
    base = 0.5

    # Status boost
    status_boost = {
        "repeat": 0.3,
        "reinforce": 0.2,
        "ready": 0.1,
        "blocked": 0.0,
        "skip": 0.0,
    }
    base += status_boost.get(status, 0.0)

    # Gap boost (bigger gap = higher priority)
    gap = max(0.0, node.difficulty - user_level)
    base += gap * 0.2

    # Goal proximity boost
    if node.topic_id in goal_topics:
        base += 0.1

    return min(base, 1.0)
