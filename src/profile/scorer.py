"""
Dynamic scoring system — Bayesian-style score evolution for user skills.

Design principles:
- Scores are continuous (0.0–1.0), NOT discrete labels
- Every score update has a confidence component
- New evidence incrementally adjusts old scores (not replaces them)
- Low-confidence scores have higher plasticity (change more)
- Evidence accumulation increases confidence

This enables the agent to answer questions like:
  "How confident are we that this user is a Python beginner?"
  → "Level 0.35, confidence 0.40 — we've only seen 2 interactions"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from src.profile.schema import SkillEntry


# ── Scoring constants ──────────────────────────────────────────────────────

# Learning rate: how much a single observation can move the score.
# Higher = faster adaptation, lower = more stable.
DEFAULT_LEARNING_RATE = 0.15

# Minimum confidence after any update (we're never 100% sure)
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.98

# Confidence gained per observation (diminishing returns)
CONFIDENCE_PER_OBSERVATION = 0.15

# How much confidence decays when contradictory evidence appears
CONTRADICTION_DECAY = 0.2

# Maximum score change in one update (safety cap)
MAX_SINGLE_UPDATE_DELTA = 0.3


@dataclass
class ScoreUpdate:
    """Result of a scoring operation."""

    old_level: float
    new_level: float
    old_confidence: float
    new_confidence: float
    delta: float
    reason: str


def compute_skill_score(
    current: SkillEntry | None,
    observed_level: float,
    observation_confidence: float = 0.5,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tuple[SkillEntry, ScoreUpdate]:
    """Compute a new skill score by merging current knowledge with new observation.

    Uses a weighted exponential moving average with confidence tracking:

        new_level = old_level + lr * confidence_delta * (observed - old_level)
        new_conf  = min(MAX, old_conf + conf_gain)

    Where:
    - confidence_delta incorporates both old and new confidence
    - conf_gain diminishes as confidence approaches MAX_CONFIDENCE
    - Contradictory evidence reduces confidence before updating

    Args:
        current: Existing skill entry (or None for first observation).
        observed_level: New observed skill level (0→1).
        observation_confidence: How confident we are in this observation (0→1).
        learning_rate: Base adaptation speed.

    Returns:
        (new_skill_entry, score_update_metadata)
    """
    now = datetime.now(timezone.utc).isoformat()

    if current is None:
        # First observation — initialize with moderate confidence
        conf = min(MAX_CONFIDENCE, observation_confidence * 0.5)
        entry = SkillEntry(
            level=observed_level,
            confidence=conf,
            last_observed=now,
            evidence_count=1,
        )
        update = ScoreUpdate(
            old_level=0.0,
            new_level=observed_level,
            old_confidence=0.0,
            new_confidence=conf,
            delta=observed_level,
            reason="Initial observation",
        )
        return entry, update

    old_level = current.level
    old_confidence = current.confidence

    # Detect contradiction: if observed is very different from current with high confidence
    gap = abs(observed_level - old_level)
    is_contradiction = gap > 0.4 and old_confidence > 0.5

    if is_contradiction:
        # Reduce old confidence — the world may have changed
        effective_old_conf = max(MIN_CONFIDENCE, old_confidence - CONTRADICTION_DECAY)
        reason = f"Contradictory evidence detected (gap={gap:.2f}), decaying old confidence"
    else:
        effective_old_conf = old_confidence
        reason = "Consistent evidence"

    # Weighted update: trust the signal proportionally to observation confidence,
    # resist change proportionally to old confidence
    weight = learning_rate * observation_confidence * (1.0 - effective_old_conf * 0.5)
    delta = weight * (observed_level - old_level)

    # Safety cap
    delta = max(-MAX_SINGLE_UPDATE_DELTA, min(MAX_SINGLE_UPDATE_DELTA, delta))

    new_level = max(0.0, min(1.0, old_level + delta))

    # Confidence update with diminishing returns
    conf_gain = CONFIDENCE_PER_OBSERVATION * (1.0 - old_confidence / MAX_CONFIDENCE) * observation_confidence
    if is_contradiction:
        conf_gain *= 0.3  # Contradiction slows confidence growth
    new_confidence = min(MAX_CONFIDENCE, effective_old_conf + conf_gain)

    entry = SkillEntry(
        level=round(new_level, 4),
        confidence=round(new_confidence, 4),
        last_observed=now,
        evidence_count=current.evidence_count + 1,
    )

    update = ScoreUpdate(
        old_level=old_level,
        new_level=new_level,
        old_confidence=old_confidence,
        new_confidence=new_confidence,
        delta=delta,
        reason=reason,
    )

    return entry, update


def compute_style_score(
    current_value: float,
    observed_signal: float,
    observation_confidence: float = 0.5,
    learning_rate: float = 0.2,
) -> float:
    """Update a learning style preference dimension.

    Style preferences shift more slowly than skills because they
    represent stable personality traits.
    """
    weight = learning_rate * observation_confidence * 0.5
    delta = weight * (observed_signal - current_value)
    delta = max(-0.2, min(0.2, delta))  # Slower update for style
    return round(max(0.0, min(1.0, current_value + delta)), 4)


def score_snapshot(profile) -> dict[str, dict[str, float]]:
    """Export all skill scores with confidence as a simple dict.

    Useful for analytics dashboards or quick inspection.
    """
    return {
        name: {"level": entry.level, "confidence": entry.confidence}
        for name, entry in profile.skills.items()
    }


def top_skills(profile, min_confidence: float = 0.3, top_n: int = 5) -> list[tuple[str, float]]:
    """Return the user's strongest skills (above confidence threshold)."""
    ranked = sorted(
        [(name, entry.level) for name, entry in profile.skills.items()
         if entry.confidence >= min_confidence],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def weakest_skills(profile, min_confidence: float = 0.3, top_n: int = 5) -> list[tuple[str, float]]:
    """Return the user's weakest skills (above confidence threshold)."""
    ranked = sorted(
        [(name, entry.level) for name, entry in profile.skills.items()
         if entry.confidence >= min_confidence],
        key=lambda x: x[1],
    )
    return ranked[:top_n]
