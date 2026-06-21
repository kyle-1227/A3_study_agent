"""
Recommendation Engine — orchestrates multi-factor scoring to produce ranked,
explainable learning resource recommendations.

Integrates user profile skills, long-term memory (episodic + semantic),
and knowledge graph topics to generate personalized recommendations.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.config import get_setting
from src.curriculum.knowledge_graph import KnowledgeGraph, load_knowledge_graph
from src.memory.storage import MemoryStore, create_memory_store
from src.profile.scorer import weakest_skills
from src.profile.schema import UserProfile
from src.recommendation.scorer import (
    build_reason,
    compute_combined_score,
    forgetting_score,
    goal_score,
    preference_score,
    weakness_score,
)
from src.recommendation.types import Recommendation, RecommendationList, ScoreBreakdown

logger = logging.getLogger(__name__)


async def generate_recommendations(
    user_id: str,
    profile: UserProfile,
    *,
    store: MemoryStore | None = None,
    kg: KnowledgeGraph | None = None,
    top_n: int | None = None,
    subject_filter: str | None = None,
) -> RecommendationList:
    """Generate ranked, explainable learning resource recommendations.

    Flow:
    1. Load knowledge graph topics
    2. Identify user's weakest skills (from profile)
    3. For each topic, find matching KG nodes and resources
    4. Compute 4-factor scores (weakness + forgetting + preference + goal)
    5. Combine, sort, return top-N with reasons

    Args:
        user_id: The user identifier.
        profile: UserProfile with skills, style, goals.
        store: MemoryStore for episodic history lookup.
        kg: KnowledgeGraph instance.
        top_n: Max recommendations to return. Default from settings.
        subject_filter: Optional subject to filter by.

    Returns:
        RecommendationList with ranked items and generation context.
    """
    start_time = time.monotonic()
    store = store or create_memory_store()
    kg = kg or load_knowledge_graph()

    if top_n is None:
        top_n = int(get_setting("recommendation.top_n", 10))

    # Get weights
    weights = {
        "weakness": float(get_setting("recommendation.weights.weakness", 0.35)),
        "forgetting": float(get_setting("recommendation.weights.forgetting", 0.25)),
        "preference": float(get_setting("recommendation.weights.preference", 0.15)),
        "goal": float(get_setting("recommendation.weights.goal", 0.25)),
    }

    # Get weakest skills sorted by level*confidence ascending
    weak_skills = weakest_skills(profile, n=20)

    # Get last practice times from episodic memory
    last_practice_map = await _get_last_practice_times(user_id, store)

    # Build candidate recommendations
    candidates: list[Recommendation] = []
    total_candidates = 0

    for skill_name, skill_entry in weak_skills:
        # Find matching KG topics
        matching_nodes = _find_matching_topics(kg, skill_name, subject_filter)
        if not matching_nodes:
            continue

        for node in matching_nodes:
            # Get resources for this topic (all types)
            for rtype, names in node.resources.items():
                if not names:
                    continue
                for name in names:
                    total_candidates += 1

                    # Compute four-factor scores
                    weakness = weakness_score(skill_entry)
                    last_practice = last_practice_map.get(node.topic_id)
                    forgetting = forgetting_score(last_practice)
                    preference = preference_score(rtype, profile.learning_style)
                    goal = goal_score(profile.goals, node.subject)

                    combined = compute_combined_score(
                        weakness, forgetting, preference, goal, weights,
                    )

                    # Calculate days since practice
                    days_since = None
                    if last_practice:
                        try:
                            dt = datetime.fromisoformat(last_practice.replace("Z", "+00:00"))
                            days_since = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
                        except (ValueError, TypeError):
                            pass

                    # Build recommendation
                    rec = Recommendation(
                        resource_type=rtype,
                        subject=node.subject,
                        topic=node.name,
                        title=name,
                        priority=combined,
                        reason=build_reason(
                            weakness=weakness,
                            forgetting=forgetting,
                            preference=preference,
                            goal=goal,
                            subject=node.subject,
                            resource_type=rtype,
                            topic_name=node.name,
                            skill_level=skill_entry.level if skill_entry else None,
                            days_since_practice=days_since,
                        ),
                        score_breakdown=ScoreBreakdown(
                            weakness_score=weakness,
                            forgetting_score=forgetting,
                            preference_score=preference,
                            goal_score=goal,
                            combined_score=combined,
                            weights=weights,
                        ),
                        knowledge_points=list(node.knowledge_points),
                        user_skill_gap=weakness,
                    )
                    candidates.append(rec)

    # Filter below threshold
    min_threshold = float(get_setting("recommendation.min_score_threshold", 0.2))
    candidates = [c for c in candidates if c.priority >= min_threshold]

    # Sort by priority descending
    candidates.sort(key=lambda r: r.priority, reverse=True)

    # Deduplicate by title
    seen_titles: set[str] = set()
    deduped: list[Recommendation] = []
    for rec in candidates:
        key = f"{rec.resource_type}:{rec.subject}:{rec.title}"
        if key not in seen_titles:
            seen_titles.add(key)
            deduped.append(rec)

    top_results = deduped[:top_n]

    generation_time_ms = (time.monotonic() - start_time) * 1000

    # Build context summary
    context_parts: list[str] = []
    if weak_skills:
        top_weak = [f"{name}({entry.level:.0%})" for name, entry in weak_skills[:5]]
        context_parts.append(f"薄弱技能: {', '.join(top_weak)}")
    if profile.goals:
        top_goals = [
            f"{g.goal}(importance={g.importance:.0%})"
            for g in sorted(profile.goals, key=lambda g: g.importance, reverse=True)[:3]
        ]
        context_parts.append(f"学习目标: {', '.join(top_goals)}")
    context_summary = "; ".join(context_parts) if context_parts else "基于用户画像和记忆的个性化推荐"

    logger.info(
        "Generated %d recommendations for user=%s from %d candidates (%.0fms)",
        len(top_results), user_id, total_candidates, generation_time_ms,
    )

    return RecommendationList(
        user_id=user_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        items=top_results,
        context_summary=context_summary,
        total_candidates_considered=total_candidates,
        generation_time_ms=generation_time_ms,
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def _find_matching_topics(
    kg: KnowledgeGraph,
    skill_name: str,
    subject_filter: str | None = None,
) -> list:
    """Find KG topics that match a profile skill name.

    Uses substring matching between skill_name and topic names/KPs.
    """
    from src.curriculum.types import KnowledgeNode

    skill_lower = (skill_name or "").lower()
    matches: list[KnowledgeNode] = []

    for node in kg.get_all_topics():
        if subject_filter and node.subject != subject_filter:
            continue
        # Match on topic name
        if skill_lower in node.name.lower():
            matches.append(node)
            continue
        # Match on knowledge points
        for kp in node.knowledge_points:
            if skill_lower in kp.lower():
                matches.append(node)
                break

    return matches


async def _get_last_practice_times(
    user_id: str,
    store: MemoryStore,
) -> dict[str, str]:
    """Get the last practice time for each topic from episodic memory.

    Returns dict of topic_id → ISO timestamp.
    """
    try:
        records = await store.query_episodic(
            user_id,
            memory_type="learning_behavior",
            limit=200,
        )
        last_times: dict[str, str] = {}
        for rec in records:
            subject = rec.subject or ""
            if subject and rec.created_at:
                if subject not in last_times or rec.created_at > last_times[subject]:
                    last_times[subject] = rec.created_at
        return last_times
    except Exception as exc:
        logger.debug("Failed to get last practice times for user=%s: %s", user_id, exc)
        return {}
