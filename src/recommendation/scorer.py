"""
Recommendation Scorer — four-factor scoring for learning resource recommendations.

Factors:
1. WeaknessScore: how weak/uncertain the user's skill is
2. ForgettingScore: how long since last practice (Ebbinghaus sigmoid)
3. PreferenceScore: how well the resource type matches learning style
4. GoalScore: how important and incomplete the matching goal is

All scores are normalized to [0, 1].
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.config import get_setting
from src.profile.schema import Goal, LearningStyle, SkillEntry

# Resource type → learning style dimension mapping
RESOURCE_STYLE_MAP: dict[str, str] = {
    "mindmap": "prefer_visual",
    "quiz": "prefer_practice",
    "doc": "prefer_theory",
    "review_doc": "prefer_step_by_step",
    "case": "prefer_examples",
}


# ── Individual scoring functions ───────────────────────────────────────────


def weakness_score(skill: SkillEntry | None) -> float:
    """Compute weakness score: higher for weak or uncertain skills.

    Formula: 1.0 - level * confidence

    - Unknown skill (None) → 0.8 (assume weak, not maximally weak)
    - Strong skill (level=0.9, confidence=0.95) → 0.145
    - Weak skill (level=0.2, confidence=0.9) → 0.82
    - Uncertain skill (level=0.6, confidence=0.2) → 0.88

    Args:
        skill: SkillEntry from user profile, or None if unknown.

    Returns:
        Weakness score in [0, 1].
    """
    if skill is None:
        return 0.8  # Unknown → assume moderately weak
    return max(0.0, min(1.0, 1.0 - skill.level * skill.confidence))


def forgetting_score(
    last_practiced_at: str | None,
    decay_days: float | None = None,
) -> float:
    """Compute forgetting score: higher for stale knowledge.

    Uses a sigmoid function over days since last practice:
        score = 1.0 / (1.0 + exp(-(days - decay_days) / steepness))

    - Never practiced (None) → 0.7 (moderate forgetting pressure)
    - Practiced today (0 days) → ~0.06 (very fresh)
    - Practiced 14 days ago → 0.5 (at half-life)
    - Practiced 30 days ago → ~0.96 (mostly forgotten)

    Args:
        last_practiced_at: ISO timestamp of last practice, or None.
        decay_days: Days at which the score reaches 0.5. Default from settings.

    Returns:
        Forgetting score in [0, 1].
    """
    if decay_days is None:
        decay_days = float(get_setting("recommendation.decay_days", 14.0))
    steepness = 5.0  # Controls the sigmoid steepness

    if last_practiced_at is None or not last_practiced_at.strip():
        return 0.7  # Never practiced → moderate forgetting

    try:
        last_dt = datetime.fromisoformat(last_practiced_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days = (now - last_dt).total_seconds() / 86400.0
        days = max(0.0, days)
    except (ValueError, TypeError):
        return 0.7  # Unparseable → assume moderate

    # Sigmoid: ~0 at days=0, 0.5 at days=decay_days, ~1 at large days
    return 1.0 / (1.0 + math.exp(-(days - decay_days) / steepness))


def preference_score(
    resource_type: str,
    learning_style: LearningStyle | None,
) -> float:
    """Compute preference score: how well resource type matches learning style.

    Mapping:
        mindmap → prefer_visual
        quiz → prefer_practice
        doc → prefer_theory
        review_doc → prefer_step_by_step
        case → prefer_examples

    Args:
        resource_type: One of "mindmap", "quiz", "doc", "review_doc", "case".
        learning_style: User's LearningStyle from profile.

    Returns:
        Preference score in [0, 1].
    """
    if learning_style is None:
        return 0.5  # Neutral

    dim = RESOURCE_STYLE_MAP.get(resource_type, "")
    if not dim:
        return 0.5

    return float(getattr(learning_style, dim, 0.5))


def goal_score(goals: list[Goal], subject: str) -> float:
    """Compute goal alignment score: higher for important, incomplete goals.

    Finds goals whose description contains the subject keyword, then uses
    the most important incomplete goal's score.

    Args:
        goals: User's learning goals from profile.
        subject: Academic subject being scored.

    Returns:
        Goal score in [0, 1].
    """
    if not goals:
        return 0.3  # No goals → low priority

    subject_lower = (subject or "").lower()
    relevant = [
        g for g in goals
        if subject_lower in (g.goal or "").lower() or subject_lower == ""
    ]

    if not relevant:
        return 0.3  # No matching goal → neutral-low

    # Pick the most important incomplete goal
    best = max(relevant, key=lambda g: g.importance * (1.0 - g.progress))
    return best.importance * (1.0 - best.progress)


# ── Combined scoring ───────────────────────────────────────────────────────


def compute_combined_score(
    weakness: float,
    forgetting: float,
    preference: float,
    goal: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted combined score from four factors.

    Default weights from settings.yaml:
        weakness=0.35, forgetting=0.25, preference=0.15, goal=0.25

    Args:
        weakness: WeaknessScore output.
        forgetting: ForgettingScore output.
        preference: PreferenceScore output.
        goal: GoalScore output.
        weights: Optional weight overrides.

    Returns:
        Combined score in [0, 1].
    """
    if weights is None:
        weights = {
            "weakness": float(get_setting("recommendation.weights.weakness", 0.35)),
            "forgetting": float(get_setting("recommendation.weights.forgetting", 0.25)),
            "preference": float(get_setting("recommendation.weights.preference", 0.15)),
            "goal": float(get_setting("recommendation.weights.goal", 0.25)),
        }

    combined = (
        weights.get("weakness", 0.35) * weakness
        + weights.get("forgetting", 0.25) * forgetting
        + weights.get("preference", 0.15) * preference
        + weights.get("goal", 0.25) * goal
    )
    return max(0.0, min(1.0, combined))


def build_reason(
    weakness: float,
    forgetting: float,
    preference: float,
    goal: float,
    subject: str,
    resource_type: str,
    topic_name: str,
    skill_level: float | None = None,
    days_since_practice: float | None = None,
) -> str:
    """Build a human-readable recommendation reason.

    The reason explains WHY this recommendation was made, decomposing
    the dominant scoring factors into natural language.
    """
    reasons: list[str] = []

    # Weakness explanation
    if weakness > 0.6:
        if skill_level is not None and skill_level < 0.3:
            reasons.append(f"{topic_name}是明显的薄弱点 (水平={skill_level:.0%})")
        else:
            reasons.append(f"{topic_name}掌握不够牢固")
    elif weakness > 0.3:
        reasons.append(f"{topic_name}有提升空间")

    # Forgetting explanation
    if forgetting > 0.5:
        if days_since_practice is not None:
            reasons.append(f"已有{days_since_practice:.0f}天未练习，需要复习")
        else:
            reasons.append("长期未练习，可能存在遗忘")

    # Preference explanation
    if preference > 0.6:
        reasons.append(f"{resource_type}格式匹配你的学习偏好")

    # Goal explanation
    if goal > 0.5:
        reasons.append(f"与你的学习目标 '{subject}' 高度相关")

    if not reasons:
        reasons.append(f"推荐学习 {topic_name} 以保持进度")

    return "；".join(reasons)
