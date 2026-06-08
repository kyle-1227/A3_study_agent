"""
Profile schema — AI-native dynamic user profile data model.

Design principles:
- Skills use continuous scores (0.0–1.0), not static labels
- Every fact carries a confidence weight
- Agent observations are first-class citizens
- Schema is extensible via extra fields, not fragile enums
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ── Atomic profile components ──────────────────────────────────────────────


class SkillEntry(BaseModel):
    """A single skill with a dynamic score and confidence.

    Scores evolve over time as the agent observes more evidence.
    Confidence captures how certain we are about the score.

    Examples:
        {"level": 0.35, "confidence": 0.6}  → weak signal, still learning
        {"level": 0.82, "confidence": 0.95} → strong signal, well-established
    """

    level: float = Field(default=0.0, ge=0.0, le=1.0, description="Current skill level 0→1")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="How certain we are 0→1")
    last_observed: str = Field(default="", description="ISO timestamp of last observation")
    evidence_count: int = Field(default=0, ge=0, description="Number of observations contributing")


class LearningStyle(BaseModel):
    """How the user prefers to learn — not fixed, can evolve.

    Each dimension is a 0→1 float representing strength of preference.
    """

    prefer_examples: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes code / concrete examples")
    prefer_visual: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes diagrams / visuals")
    prefer_step_by_step: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes sequential breakdowns")
    prefer_concise: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes short / direct answers")
    prefer_theory: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes theoretical depth")
    prefer_practice: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes hands-on exercises")
    prefer_analogy: float = Field(default=0.5, ge=0.0, le=1.0, description="Likes analogies and metaphors")
    extra: dict[str, float] = Field(default_factory=dict, description="Extensible preference dimensions")


class Goal(BaseModel):
    """A learning goal with importance weight."""

    goal: str = Field(..., description="Goal description, e.g. '准备408考研'")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="How important is this goal")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated progress toward goal")
    created_at: str = Field(default="", description="ISO timestamp")


class BehaviorProfile(BaseModel):
    """Aggregated behavioral signals."""

    avg_session_minutes: float = Field(default=0.0, ge=0.0)
    total_sessions: int = Field(default=0, ge=0)
    quiz_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    daily_active_days: int = Field(default=0, ge=0)
    questions_asked: int = Field(default=0, ge=0)
    avg_question_length: float = Field(default=0.0, ge=0.0)
    follow_up_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="How often user asks follow-ups")


class AgentObservation(BaseModel):
    """A single observation the agent has made about the user.

    These are the building blocks of the dynamic profile.
    Each observation feeds into skill scores, preferences, and goals.
    """

    content: str = Field(..., description="What the agent observed")
    category: str = Field(default="general", description="skill | preference | goal | behavior | general")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="How important this observation is")
    evidence: str = Field(default="", description="What triggered this observation (quote or summary)")
    created_at: str = Field(default="", description="ISO timestamp")


# ── Main profile ───────────────────────────────────────────────────────────


class UserProfile(BaseModel):
    """AI-native dynamic user profile.

    This is NOT a static CRM record — it's a living model that evolves
    with every interaction. Every field carries confidence metadata so
    the agent can decide how much to trust each signal.
    """

    user_id: str = Field(..., description="Unique user identifier")

    # Dynamic skill map — keyed by skill name (e.g. "python", "algorithm", "math")
    skills: dict[str, SkillEntry] = Field(default_factory=dict)

    # Learning style preferences (continuous, not binary)
    learning_style: LearningStyle = Field(default_factory=LearningStyle)

    # Active learning goals
    goals: list[Goal] = Field(default_factory=list)

    # Aggregated behavioral metrics
    behavior: BehaviorProfile = Field(default_factory=BehaviorProfile)

    # Raw agent observations — the evidence behind every score
    agent_observations: list[AgentObservation] = Field(default_factory=list)

    # Denylist: topics/subjects the user dislikes or finds unhelpful
    dislikes: list[str] = Field(default_factory=list)

    # Freeform notes for future extensions
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_at: str = Field(default="", description="ISO timestamp of first profile creation")
    updated_at: str = Field(default="", description="ISO timestamp of last update")

    def touch(self) -> None:
        """Update the timestamp to now."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

    def has_skill(self, name: str) -> bool:
        return name in self.skills

    def get_skill_level(self, name: str) -> float:
        entry = self.skills.get(name)
        return entry.level if entry else 0.0

    def get_skill_confidence(self, name: str) -> float:
        entry = self.skills.get(name)
        return entry.confidence if entry else 0.0

    def to_summary(self) -> str:
        """Generate a compact natural-language summary for prompt injection."""
        return profile_to_summary(self)


# ── Extraction / update DTOs ───────────────────────────────────────────────


class ExtractedProfileInfo(BaseModel):
    """What the LLM extracted from a conversation turn.

    All fields are optional — only return what was actually observed.
    """

    skills_observed: dict[str, float] = Field(
        default_factory=dict,
        description="Skill name → observed level (0→1). E.g. {'python': 0.4, 'algorithm': 0.2}",
    )
    skill_evidence: str = Field(default="", description="What evidence supports the skill assessment")

    style_signals: dict[str, float] = Field(
        default_factory=dict,
        description="Preference dimension → signal strength. E.g. {'prefer_examples': 0.9}",
    )
    style_evidence: str = Field(default="")

    goals_observed: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{'goal': '准备408', 'importance': 0.9}]",
    )

    behavior_update: dict[str, float] = Field(
        default_factory=dict,
        description="Behavioral metrics from this turn",
    )

    observations: list[str] = Field(
        default_factory=list,
        description="Free-text agent observations about the user",
    )

    dislikes_observed: list[str] = Field(default_factory=list)


class ProfileUpdateResult(BaseModel):
    """Result of a profile update operation."""

    profile: UserProfile
    changes: list[str] = Field(default_factory=list, description="Human-readable list of what changed")
    new_observations: int = Field(default=0)


# ── Helpers ────────────────────────────────────────────────────────────────


def profile_to_summary(profile: UserProfile, max_skills: int = 8, max_goals: int = 5) -> str:
    """Build a concise natural-language summary suitable for prompt injection.

    This is intentionally NOT a full JSON dump — it selects the most
    relevant signals to keep token usage low.
    """
    parts: list[str] = []

    # Skills (sorted by confidence-weighted level, most relevant first)
    if profile.skills:
        ranked = sorted(
            profile.skills.items(),
            key=lambda kv: kv[1].level * kv[1].confidence,
            reverse=True,
        )
        skill_lines = []
        for name, entry in ranked[:max_skills]:
            if entry.confidence < 0.3:
                continue  # skip low-confidence skills
            level_label = _level_label(entry.level)
            certainty = _certainty_label(entry.confidence)
            skill_lines.append(f"  - {name}: {level_label} ({certainty})")
        if skill_lines:
            parts.append("技能水平:\n" + "\n".join(skill_lines))

    # Learning style (only strong preferences)
    style = profile.learning_style
    strong_prefs = []
    for dim, label in [
        ("prefer_examples", "喜欢具体案例"),
        ("prefer_visual", "喜欢可视化"),
        ("prefer_step_by_step", "喜欢分步讲解"),
        ("prefer_concise", "喜欢简洁回答"),
        ("prefer_theory", "喜欢理论深度"),
        ("prefer_practice", "喜欢动手实践"),
        ("prefer_analogy", "喜欢类比讲解"),
    ]:
        val = getattr(style, dim, 0.5)
        if val > 0.7:
            strong_prefs.append(f"  - {label} (强偏好)")
        elif val < 0.3:
            strong_prefs.append(f"  - 不太喜欢{label.replace('喜欢', '')}")
    if strong_prefs:
        parts.append("学习偏好:\n" + "\n".join(strong_prefs))

    # Goals
    if profile.goals:
        ranked_goals = sorted(profile.goals, key=lambda g: g.importance, reverse=True)
        goal_lines = []
        for g in ranked_goals[:max_goals]:
            if g.importance < 0.3:
                continue
            goal_lines.append(f"  - {g.goal} (重要度: {g.importance:.0%})")
        if goal_lines:
            parts.append("学习目标:\n" + "\n".join(goal_lines))

    # Dislikes
    if profile.dislikes:
        parts.append(f"不喜欢/回避: {', '.join(profile.dislikes)}")

    # Key observations (most important ones)
    if profile.agent_observations:
        important = sorted(profile.agent_observations, key=lambda o: o.importance, reverse=True)
        top_obs = [f"  - {o.content}" for o in important[:5] if o.importance > 0.5]
        if top_obs:
            parts.append("关键观察:\n" + "\n".join(top_obs))

    if not parts:
        return "（暂无用户画像数据）"

    return "\n\n".join(parts)


def _level_label(level: float) -> str:
    if level < 0.2:
        return "入门"
    elif level < 0.4:
        return "初级"
    elif level < 0.6:
        return "中等"
    elif level < 0.8:
        return "熟练"
    else:
        return "精通"


def _certainty_label(confidence: float) -> str:
    if confidence < 0.3:
        return "低置信度"
    elif confidence < 0.7:
        return "中等置信度"
    else:
        return "高置信度"
