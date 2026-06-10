"""
Profile updater — incremental, non-destructive profile merging.

Design principles:
- NEVER replace old data wholesale — always merge
- Skills evolve via the scorer (Bayesian-style weighted update)
- Style preferences shift gradually
- Goals accumulate with deduplication
- Observations are append-only with a cap
- Every update returns a change log for transparency
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.profile.schema import (
    AgentObservation,
    BehaviorProfile,
    ExtractedProfileInfo,
    Goal,
    LearningStyle,
    ProfileUpdateResult,
    UserProfile,
)
from src.profile.scorer import (
    compute_skill_score,
    compute_style_score,
    MAX_SINGLE_UPDATE_DELTA,
)

logger = logging.getLogger(__name__)

# Maximum number of agent observations to retain (FIFO)
_MAX_OBSERVATIONS = 50


def update_profile(
    profile: UserProfile,
    extracted: ExtractedProfileInfo,
) -> ProfileUpdateResult:
    """Merge newly extracted profile information into the existing profile.

    This is the core "merge" function. It incrementally updates every
    dimension of the user profile without destroying old data.

    Args:
        profile: Current user profile (modified IN PLACE).
        extracted: Newly extracted signals from a conversation turn.

    Returns:
        ProfileUpdateResult with the updated profile and change log.
    """
    changes: list[str] = []

    # 1. Update skills
    if extracted.skills_observed:
        skill_changes = _update_skills(profile, extracted)
        changes.extend(skill_changes)

    # 2. Update learning style
    if extracted.style_signals:
        style_changes = _update_learning_style(profile, extracted)
        changes.extend(style_changes)

    # 3. Update goals
    if extracted.goals_observed:
        goal_changes = _update_goals(profile, extracted)
        changes.extend(goal_changes)

    # 4. Update behavior
    if extracted.behavior_update:
        behavior_changes = _update_behavior(profile, extracted)
        changes.extend(behavior_changes)

    # 5. Record observations
    obs_count = 0
    if extracted.observations:
        obs_count = _add_observations(profile, extracted)

    # 6. Update dislikes
    if extracted.dislikes_observed:
        _update_dislikes(profile, extracted)

    # 7. Touch timestamp
    profile.touch()

    return ProfileUpdateResult(
        profile=profile,
        changes=changes,
        new_observations=obs_count,
    )


# ── Internal update helpers ────────────────────────────────────────────────


def _update_skills(profile: UserProfile, extracted: ExtractedProfileInfo) -> list[str]:
    """Merge skill observations using the scorer."""
    changes: list[str] = []
    for skill_name, observed_level in extracted.skills_observed.items():
        current = profile.skills.get(skill_name)
        observation_confidence = 0.5  # Default; could be tuned per evidence quality
        new_entry, score_update = compute_skill_score(
            current, observed_level, observation_confidence
        )
        profile.skills[skill_name] = new_entry

        if current is None:
            changes.append(
                f"技能 [{skill_name}] 首次评估: level={new_entry.level:.2f}, "
                f"confidence={new_entry.confidence:.2f}"
            )
        elif abs(score_update.delta) > 0.02:
            changes.append(
                f"技能 [{skill_name}]: {score_update.old_level:.2f} → "
                f"{score_update.new_level:.2f} (Δ={score_update.delta:+.3f}, "
                f"{score_update.reason})"
            )
    return changes


def _update_learning_style(profile: UserProfile, extracted: ExtractedProfileInfo) -> list[str]:
    """Merge learning style signals with gradual adaptation."""
    changes: list[str] = []
    style = profile.learning_style

    valid_dims = {
        "prefer_examples", "prefer_visual", "prefer_step_by_step",
        "prefer_concise", "prefer_theory", "prefer_practice", "prefer_analogy",
    }

    for dim, signal in extracted.style_signals.items():
        if dim not in valid_dims:
            # Use extra dict for custom dimensions
            old_val = style.extra.get(dim, 0.5)
            new_val = compute_style_score(old_val, signal)
            style.extra[dim] = new_val
            if abs(new_val - old_val) > 0.03:
                changes.append(f"偏好 [{dim}]: {old_val:.2f} → {new_val:.2f}")
            continue

        old_val = getattr(style, dim, 0.5)
        new_val = compute_style_score(old_val, signal)
        setattr(style, dim, new_val)
        if abs(new_val - old_val) > 0.03:
            dim_label = _dim_label(dim)
            changes.append(f"偏好 [{dim_label}]: {old_val:.2f} → {new_val:.2f}")
    return changes


def _update_goals(profile: UserProfile, extracted: ExtractedProfileInfo) -> list[str]:
    """Add or update goals with deduplication."""
    changes: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    existing_texts = {g.goal for g in profile.goals}

    for g in extracted.goals_observed:
        goal_text = str(g.get("goal", "")).strip()
        if not goal_text:
            continue
        importance = float(g.get("importance", 0.5))

        if goal_text in existing_texts:
            # Update importance of existing goal
            for existing_goal in profile.goals:
                if existing_goal.goal == goal_text:
                    old_imp = existing_goal.importance
                    # Weighted average toward new importance
                    existing_goal.importance = round(old_imp * 0.7 + importance * 0.3, 3)
                    changes.append(f"目标 [{goal_text}] 重要性更新: {old_imp:.2f} → {existing_goal.importance:.2f}")
                    break
        else:
            profile.goals.append(Goal(
                goal=goal_text,
                importance=importance,
                created_at=now,
            ))
            existing_texts.add(goal_text)
            changes.append(f"新目标: {goal_text} (重要性: {importance:.2f})")
    return changes


def _update_behavior(profile: UserProfile, extracted: ExtractedProfileInfo) -> list[str]:
    """Update behavioral metrics with running averages."""
    changes: list[str] = []
    b = profile.behavior
    bu = extracted.behavior_update

    if "avg_session_minutes" in bu:
        old = b.avg_session_minutes
        if old == 0:
            b.avg_session_minutes = bu["avg_session_minutes"]
        else:
            b.avg_session_minutes = round(old * 0.7 + bu["avg_session_minutes"] * 0.3, 1)
        changes.append(f"平均会话时长: {old:.0f} → {b.avg_session_minutes:.0f}min")

    if "quiz_accuracy" in bu:
        old = b.quiz_accuracy
        if old == 0:
            b.quiz_accuracy = bu["quiz_accuracy"]
        else:
            b.quiz_accuracy = round(old * 0.6 + bu["quiz_accuracy"] * 0.4, 3)
        changes.append(f"答题正确率: {old:.2f} → {b.quiz_accuracy:.2f}")

    if "questions_asked" in bu:
        b.questions_asked += int(bu["questions_asked"])

    return changes


def _add_observations(profile: UserProfile, extracted: ExtractedProfileInfo) -> int:
    """Add agent observations with FIFO cap."""
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for obs_text in extracted.observations:
        if not obs_text.strip():
            continue
        profile.agent_observations.append(AgentObservation(
            content=obs_text,
            importance=0.5,
            created_at=now,
        ))
        count += 1

    # FIFO cap
    if len(profile.agent_observations) > _MAX_OBSERVATIONS:
        profile.agent_observations = profile.agent_observations[-_MAX_OBSERVATIONS:]

    return count


def _update_dislikes(profile: UserProfile, extracted: ExtractedProfileInfo) -> None:
    """Add new dislikes, deduplicate."""
    for d in extracted.dislikes_observed:
        if d and d not in profile.dislikes:
            profile.dislikes.append(d)


def _dim_label(dim: str) -> str:
    """Human-readable label for a style dimension."""
    labels = {
        "prefer_examples": "喜欢案例",
        "prefer_visual": "喜欢可视化",
        "prefer_step_by_step": "喜欢分步讲解",
        "prefer_concise": "喜欢简洁",
        "prefer_theory": "喜欢理论",
        "prefer_practice": "喜欢实践",
        "prefer_analogy": "喜欢类比",
    }
    return labels.get(dim, dim)
