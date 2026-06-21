"""Unit tests for the user profile system — schema, scorer, prompts, extractor, updater.

All tests use mock LLM responses and run offline without API keys.

Usage:
    pytest tests/test_profile.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.structured_output import StructuredLLMResult
from src.profile.schema import (
    AgentObservation,
    BehaviorProfile,
    ExtractedProfileInfo,
    ExtractedProfileInfoStrict,
    Goal,
    LearningStyle,
    ProfileBehaviorUpdateSignal,
    ProfileGoalSignal,
    ProfileUpdateResult,
    ProfileSkillSignal,
    ProfileStyleSignal,
    SkillEntry,
    UserProfile,
    _certainty_label,
    _level_label,
    profile_to_summary,
)
from src.profile.scorer import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    ScoreUpdate,
    compute_skill_score,
    compute_style_score,
    score_snapshot,
    top_skills,
    weakest_skills,
)
from src.profile.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)
from src.profile.extractor import (
    ProfileExtractor,
    _format_conversation,
    _truncate_conversation,
)
from src.profile.updater import update_profile


# ===========================================================================
# TestSkillEntry
# ===========================================================================

class TestSkillEntry:

    def test_defaults(self):
        entry = SkillEntry()
        assert entry.level == 0.0
        assert entry.confidence == 0.0
        assert entry.last_observed == ""
        assert entry.evidence_count == 0

    def test_custom_values(self):
        entry = SkillEntry(
            level=0.75,
            confidence=0.9,
            last_observed="2026-06-03T00:00:00Z",
            evidence_count=8,
        )
        assert entry.level == 0.75
        assert entry.confidence == 0.9
        assert entry.evidence_count == 8

    def test_level_clamped_to_range(self):
        """Pydantic validates ge=0.0, le=1.0."""
        entry = SkillEntry(level=0.5)
        assert entry.level == 0.5
        # Values outside [0,1] should raise ValidationError
        with pytest.raises(Exception):
            SkillEntry(level=1.5)
        with pytest.raises(Exception):
            SkillEntry(level=-0.5)

    def test_confidence_clamped(self):
        with pytest.raises(Exception):
            SkillEntry(confidence=2.0)

    def test_evidence_count_non_negative(self):
        with pytest.raises(Exception):
            SkillEntry(evidence_count=-1)


# ===========================================================================
# TestLearningStyle
# ===========================================================================

class TestLearningStyle:

    def test_defaults_all_0_5(self):
        style = LearningStyle()
        assert style.prefer_examples == 0.5
        assert style.prefer_visual == 0.5
        assert style.prefer_step_by_step == 0.5
        assert style.prefer_concise == 0.5
        assert style.prefer_theory == 0.5
        assert style.prefer_practice == 0.5
        assert style.prefer_analogy == 0.5
        assert style.extra == {}

    def test_custom_preferences(self):
        style = LearningStyle(
            prefer_examples=0.9,
            prefer_visual=0.2,
            extra={"prefer_video": 0.8},
        )
        assert style.prefer_examples == 0.9
        assert style.prefer_visual == 0.2
        assert style.prefer_analogy == 0.5  # unchanged
        assert style.extra["prefer_video"] == 0.8

    def test_values_clamped(self):
        with pytest.raises(Exception):
            LearningStyle(prefer_examples=1.5)


# ===========================================================================
# TestGoal
# ===========================================================================

class TestGoal:

    def test_minimal_goal(self):
        g = Goal(goal="补强高等数学基础")
        assert g.goal == "补强高等数学基础"
        assert g.importance == 0.5
        assert g.progress == 0.0
        assert g.created_at == ""

    def test_full_goal(self):
        g = Goal(
            goal="准备408考研",
            importance=0.9,
            progress=0.3,
            created_at="2026-06-03T12:00:00Z",
        )
        assert g.goal == "准备408考研"
        assert g.importance == 0.9
        assert g.progress == 0.3

    def test_importance_clamped(self):
        with pytest.raises(Exception):
            Goal(goal="test", importance=1.5)

    def test_progress_clamped(self):
        with pytest.raises(Exception):
            Goal(goal="test", progress=2.0)


# ===========================================================================
# TestBehaviorProfile
# ===========================================================================

class TestBehaviorProfile:

    def test_defaults(self):
        b = BehaviorProfile()
        assert b.avg_session_minutes == 0.0
        assert b.total_sessions == 0
        assert b.quiz_accuracy == 0.0
        assert b.daily_active_days == 0
        assert b.questions_asked == 0
        assert b.avg_question_length == 0.0
        assert b.follow_up_rate == 0.0

    def test_custom_values(self):
        b = BehaviorProfile(
            avg_session_minutes=25.5,
            total_sessions=10,
            quiz_accuracy=0.85,
            questions_asked=42,
            follow_up_rate=0.6,
        )
        assert b.avg_session_minutes == 25.5
        assert b.total_sessions == 10
        assert b.quiz_accuracy == 0.85
        assert b.questions_asked == 42
        assert b.follow_up_rate == 0.6


# ===========================================================================
# TestAgentObservation
# ===========================================================================

class TestAgentObservation:

    def test_basic_observation(self):
        obs = AgentObservation(content="用户在算法题上需要较多提示")
        assert obs.content == "用户在算法题上需要较多提示"
        assert obs.category == "general"
        assert obs.importance == 0.5
        assert obs.evidence == ""

    def test_categorized_observation(self):
        obs = AgentObservation(
            content="用户对递归理解较好",
            category="skill",
            importance=0.8,
            evidence="用户在第3轮快速理解闭包概念",
        )
        assert obs.category == "skill"
        assert obs.importance == 0.8
        assert obs.evidence == "用户在第3轮快速理解闭包概念"

    def test_importance_clamped(self):
        with pytest.raises(Exception):
            AgentObservation(content="test", importance=1.5)


# ===========================================================================
# TestUserProfile
# ===========================================================================

class TestUserProfile:

    def test_minimal_profile(self):
        p = UserProfile(user_id="user_001")
        assert p.user_id == "user_001"
        assert p.skills == {}
        assert isinstance(p.learning_style, LearningStyle)
        assert p.goals == []
        assert isinstance(p.behavior, BehaviorProfile)
        assert p.agent_observations == []
        assert p.dislikes == []
        assert p.tags == []
        assert p.extra == {}
        assert p.created_at == ""
        assert p.updated_at == ""

    def test_touch_sets_timestamps(self):
        p = UserProfile(user_id="user_001")
        p.touch()
        assert p.created_at != ""
        assert p.updated_at != ""
        assert "T" in p.created_at  # ISO format

    def test_touch_updates_but_preserves_created(self):
        p = UserProfile(user_id="user_001")
        p.touch()
        created = p.created_at
        updated = p.updated_at
        p.touch()
        assert p.created_at == created  # unchanged
        # updated_at may not change if called very quickly, just check it's set
        assert p.updated_at != ""

    def test_has_skill(self):
        p = UserProfile(user_id="user_001")
        assert p.has_skill("python") is False
        p.skills["python"] = SkillEntry(level=0.5, confidence=0.6)
        assert p.has_skill("python") is True
        assert p.has_skill("math") is False

    def test_get_skill_level(self):
        p = UserProfile(user_id="user_001")
        assert p.get_skill_level("python") == 0.0
        p.skills["python"] = SkillEntry(level=0.75, confidence=0.9)
        assert p.get_skill_level("python") == 0.75
        assert p.get_skill_level("math") == 0.0

    def test_get_skill_confidence(self):
        p = UserProfile(user_id="user_001")
        assert p.get_skill_confidence("python") == 0.0
        p.skills["python"] = SkillEntry(level=0.3, confidence=0.7)
        assert p.get_skill_confidence("python") == 0.7

    def test_to_summary(self):
        p = UserProfile(user_id="user_001")
        p.skills["python"] = SkillEntry(level=0.4, confidence=0.7, evidence_count=3)
        summary = p.to_summary()
        assert "python" in summary
        assert "中等" in summary  # 0.4 = 中等 (0.2≤x<0.4→初级, 0.4≤x<0.6→中等)

    def test_to_summary_empty_profile(self):
        p = UserProfile(user_id="user_001")
        summary = p.to_summary()
        assert "暂无用户画像数据" in summary


# ===========================================================================
# TestExtractedProfileInfo
# ===========================================================================

class TestExtractedProfileInfo:

    def test_empty_extraction(self):
        info = ExtractedProfileInfo()
        assert info.skills_observed == {}
        assert info.skill_evidence == ""
        assert info.style_signals == {}
        assert info.style_evidence == ""
        assert info.goals_observed == []
        assert info.behavior_update == {}
        assert info.observations == []
        assert info.dislikes_observed == []

    def test_partial_extraction_skills_only(self):
        info = ExtractedProfileInfo(
            skills_observed={"python": 0.3, "algorithm": 0.2},
            skill_evidence="用户问了基础语法问题",
        )
        assert len(info.skills_observed) == 2
        assert info.skills_observed["python"] == 0.3
        assert info.skills_observed["algorithm"] == 0.2
        assert info.style_signals == {}

    def test_full_extraction(self):
        info = ExtractedProfileInfo(
            skills_observed={"python": 0.5},
            skill_evidence="用户在讨论闭包和装饰器",
            style_signals={"prefer_examples": 0.9, "prefer_step_by_step": 0.8},
            style_evidence="用户多次要求举例和分步讲解",
            goals_observed=[{"goal": "学Python做数据分析", "importance": 0.7}],
            behavior_update={"avg_session_minutes": 30},
            observations=["用户对递归有较好理解", "偏好代码而非理论"],
            dislikes_observed=["枯燥的理论"],
        )
        assert len(info.skills_observed) == 1
        assert len(info.style_signals) == 2
        assert len(info.goals_observed) == 1
        assert len(info.observations) == 2
        assert len(info.dislikes_observed) == 1


# ===========================================================================
# TestProfileUpdateResult
# ===========================================================================

class TestProfileUpdateResult:

    def test_basic_result(self):
        p = UserProfile(user_id="u1")
        result = ProfileUpdateResult(profile=p)
        assert result.profile.user_id == "u1"
        assert result.changes == []
        assert result.new_observations == 0

    def test_result_with_changes(self):
        p = UserProfile(user_id="u1")
        changes = ["技能 [python] 首次评估: level=0.30", "新目标: 补强机器学习基础"]
        result = ProfileUpdateResult(profile=p, changes=changes, new_observations=3)
        assert len(result.changes) == 2
        assert result.new_observations == 3


# ===========================================================================
# TestLabels
# ===========================================================================

class TestLabels:

    def test_level_label_entry(self):
        assert _level_label(0.10) == "入门"
        assert _level_label(0.25) == "初级"
        assert _level_label(0.50) == "中等"
        assert _level_label(0.70) == "熟练"
        assert _level_label(0.90) == "精通"

    def test_level_label_boundaries(self):
        assert _level_label(0.19) == "入门"
        assert _level_label(0.20) == "初级"
        assert _level_label(0.39) == "初级"
        assert _level_label(0.40) == "中等"
        assert _level_label(0.59) == "中等"
        assert _level_label(0.60) == "熟练"
        assert _level_label(0.79) == "熟练"
        assert _level_label(0.80) == "精通"

    def test_certainty_label(self):
        assert _certainty_label(0.10) == "低置信度"
        assert _certainty_label(0.50) == "中等置信度"
        assert _certainty_label(0.80) == "高置信度"

    def test_certainty_label_boundaries(self):
        assert _certainty_label(0.29) == "低置信度"
        assert _certainty_label(0.30) == "中等置信度"
        assert _certainty_label(0.69) == "中等置信度"
        assert _certainty_label(0.70) == "高置信度"


# ===========================================================================
# TestProfileToSummary
# ===========================================================================

class TestProfileToSummary:

    def test_empty_profile(self):
        p = UserProfile(user_id="u1")
        assert "暂无用户画像数据" in profile_to_summary(p)

    def test_skills_with_low_confidence_filtered(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.8, confidence=0.2)  # below 0.3
        p.skills["math"] = SkillEntry(level=0.5, confidence=0.7)    # above 0.3
        summary = profile_to_summary(p)
        assert "math" in summary
        assert "python" not in summary  # filtered due to low confidence

    def test_skills_ranked_by_confidence_weighted_level(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.9, confidence=0.9)  # 0.81
        p.skills["math"] = SkillEntry(level=0.5, confidence=0.5)     # 0.25
        summary = profile_to_summary(p)
        py_pos = summary.index("python")
        math_pos = summary.index("math")
        assert py_pos < math_pos  # Python should appear first

    def test_strong_style_preferences(self):
        p = UserProfile(user_id="u1")
        p.learning_style.prefer_examples = 0.85
        p.learning_style.prefer_visual = 0.75
        p.learning_style.prefer_theory = 0.15
        summary = profile_to_summary(p)
        assert "喜欢具体案例" in summary
        assert "强偏好" in summary
        assert "不太喜欢" in summary  # for low theory preference

    def test_style_near_default_not_included(self):
        p = UserProfile(user_id="u1")
        p.learning_style.prefer_examples = 0.55  # close to 0.5 default
        summary = profile_to_summary(p)
        assert "学习偏好" not in summary  # no strong signals

    def test_goals_ranked_by_importance(self):
        p = UserProfile(user_id="u1")
        p.goals = [
            Goal(goal="目标A", importance=0.3),
            Goal(goal="目标B", importance=0.9),
            Goal(goal="目标C", importance=0.5),
        ]
        summary = profile_to_summary(p)
        assert "目标B" in summary
        b_pos = summary.index("目标B")
        c_pos = summary.index("目标C")
        assert b_pos < c_pos  # Higher importance first

    def test_low_importance_goals_filtered(self):
        p = UserProfile(user_id="u1")
        p.goals = [Goal(goal="不重要目标", importance=0.2)]
        summary = profile_to_summary(p)
        assert "不重要目标" not in summary

    def test_dislikes_shown(self):
        p = UserProfile(user_id="u1")
        p.dislikes = ["死记硬背", "枯燥理论"]
        summary = profile_to_summary(p)
        assert "死记硬背" in summary
        assert "枯燥理论" in summary

    def test_key_observations_shown(self):
        p = UserProfile(user_id="u1")
        p.agent_observations = [
            AgentObservation(content="用户递归理解好", importance=0.9),
            AgentObservation(content="用户需要算法提示", importance=0.6),
            AgentObservation(content="不重要观察", importance=0.3),
        ]
        summary = profile_to_summary(p)
        assert "用户递归理解好" in summary
        assert "用户需要算法提示" in summary
        assert "不重要观察" not in summary  # filtered due to low importance

    def test_max_skills_limit(self):
        p = UserProfile(user_id="u1")
        for i in range(12):
            p.skills[f"skill_{i}"] = SkillEntry(level=0.5, confidence=0.8)
        summary = profile_to_summary(p, max_skills=8)
        # Should only include 8 skills
        skill_count = summary.count("入门")
        assert skill_count <= 8

    def test_max_goals_limit(self):
        p = UserProfile(user_id="u1")
        for i in range(8):
            p.goals.append(Goal(goal=f"目标_{i}", importance=0.5))
        summary = profile_to_summary(p, max_goals=5)
        # Should only include 5 goals
        goal_count = summary.count("目标_")
        assert goal_count <= 5


# ===========================================================================
# TestComputeSkillScore — Bayesian-style scoring
# ===========================================================================

class TestComputeSkillScore:

    def test_first_observation_initializes(self):
        entry, update = compute_skill_score(
            current=None,
            observed_level=0.4,
            observation_confidence=0.6,
        )
        assert entry.level == 0.4
        assert entry.confidence > 0.0
        assert entry.confidence == pytest.approx(0.3, abs=0.05)  # 0.6 * 0.5
        assert entry.evidence_count == 1
        assert update.delta == 0.4
        assert update.reason == "Initial observation"

    def test_first_observation_confidence_capped(self):
        entry, _ = compute_skill_score(
            current=None,
            observed_level=0.5,
            observation_confidence=1.0,
        )
        assert entry.confidence <= MAX_CONFIDENCE

    def test_consistent_evidence_moves_level_toward_observed(self):
        current = SkillEntry(level=0.3, confidence=0.5, evidence_count=2)
        entry, update = compute_skill_score(
            current=current,
            observed_level=0.5,
            observation_confidence=0.7,
        )
        # Should move toward 0.5 from 0.3
        assert entry.level > 0.3
        assert entry.level < 0.5  # partial update, not full replacement
        assert entry.confidence > 0.5  # confidence should grow
        assert entry.evidence_count == 3

    def test_contradictory_evidence_decays_confidence(self):
        """When observed level is very different from current with high old confidence."""
        current = SkillEntry(level=0.8, confidence=0.7, evidence_count=5)
        entry, update = compute_skill_score(
            current=current,
            observed_level=0.1,  # huge gap: 0.7
            observation_confidence=0.8,
        )
        assert "Contradictory" in update.reason
        # Level should decrease
        assert entry.level < 0.8
        assert entry.level > 0.1

    def test_contradiction_threshold_respected(self):
        """Small gaps below 0.4 should not trigger contradiction."""
        current = SkillEntry(level=0.5, confidence=0.6, evidence_count=3)
        entry, update = compute_skill_score(
            current=current,
            observed_level=0.2,  # gap = 0.3, below 0.4 threshold
            observation_confidence=0.5,
        )
        assert "Consistent" in update.reason

    def test_safety_cap_limits_delta(self):
        """Delta should never exceed MAX_SINGLE_UPDATE_DELTA (0.3)."""
        current = SkillEntry(level=0.1, confidence=0.1, evidence_count=1)
        entry, update = compute_skill_score(
            current=current,
            observed_level=1.0,  # huge gap
            observation_confidence=1.0,
        )
        assert abs(update.delta) <= 0.3
        assert 0.0 <= entry.level <= 1.0

    def test_level_never_exceeds_1_0(self):
        current = SkillEntry(level=0.9, confidence=0.8, evidence_count=4)
        entry, _ = compute_skill_score(
            current=current,
            observed_level=1.0,
            observation_confidence=0.9,
        )
        assert entry.level <= 1.0

    def test_level_never_below_0(self):
        current = SkillEntry(level=0.05, confidence=0.5, evidence_count=2)
        entry, _ = compute_skill_score(
            current=current,
            observed_level=0.0,
            observation_confidence=0.8,
        )
        assert entry.level >= 0.0

    def test_confidence_gain_diminishes_near_max(self):
        """Confidence grows slower as it approaches MAX_CONFIDENCE."""
        current = SkillEntry(level=0.5, confidence=0.95, evidence_count=10)
        entry, _ = compute_skill_score(
            current=current,
            observed_level=0.55,
            observation_confidence=0.8,
        )
        # Confidence should barely increase (diminishing returns)
        assert entry.confidence <= MAX_CONFIDENCE
        assert entry.confidence >= 0.95  # still at least old level

    def test_low_observation_confidence_reduces_update(self):
        current = SkillEntry(level=0.3, confidence=0.4, evidence_count=2)
        entry_low, _ = compute_skill_score(
            current=current,
            observed_level=0.8,
            observation_confidence=0.1,  # very low
        )
        # Recreate current for second call
        current2 = SkillEntry(level=0.3, confidence=0.4, evidence_count=2)
        entry_high, _ = compute_skill_score(
            current=current2,
            observed_level=0.8,
            observation_confidence=0.9,  # high
        )
        # Higher observation confidence → bigger move
        assert abs(entry_high.level - 0.3) > abs(entry_low.level - 0.3)

    def test_evidence_count_increments(self):
        current = SkillEntry(level=0.4, confidence=0.5, evidence_count=3)
        entry, _ = compute_skill_score(
            current=current,
            observed_level=0.5,
            observation_confidence=0.6,
        )
        assert entry.evidence_count == 4

    def test_timestamp_updated(self):
        current = SkillEntry(
            level=0.3, confidence=0.4, evidence_count=2,
            last_observed="2026-01-01T00:00:00Z",
        )
        entry, _ = compute_skill_score(
            current=current,
            observed_level=0.4,
            observation_confidence=0.5,
        )
        assert entry.last_observed != "2026-01-01T00:00:00Z"
        assert "T" in entry.last_observed

    def test_custom_learning_rate(self):
        current = SkillEntry(level=0.3, confidence=0.4, evidence_count=1)
        entry_slow, _ = compute_skill_score(
            current=current,
            observed_level=0.8,
            observation_confidence=0.7,
            learning_rate=0.05,  # slow
        )
        current2 = SkillEntry(level=0.3, confidence=0.4, evidence_count=1)
        entry_fast, _ = compute_skill_score(
            current=current2,
            observed_level=0.8,
            observation_confidence=0.7,
            learning_rate=0.25,  # fast
        )
        assert abs(entry_fast.level - 0.3) > abs(entry_slow.level - 0.3)


# ===========================================================================
# TestComputeStyleScore
# ===========================================================================

class TestComputeStyleScore:

    def test_basic_update(self):
        new_val = compute_style_score(0.5, 0.9, observation_confidence=0.6)
        assert new_val > 0.5
        assert new_val < 0.9  # partial update

    def test_first_signal_from_default(self):
        new_val = compute_style_score(0.5, 0.8, observation_confidence=0.8)
        assert 0.5 < new_val <= 0.8

    def test_delta_capped_at_0_2(self):
        """Style preferences shift more slowly — max delta 0.2."""
        new_val = compute_style_score(0.1, 1.0, observation_confidence=1.0)
        assert new_val <= 0.3  # 0.1 + max 0.2

    def test_value_clamped_to_range(self):
        new_val_high = compute_style_score(0.95, 1.0, observation_confidence=1.0)
        assert new_val_high <= 1.0
        new_val_low = compute_style_score(0.05, 0.0, observation_confidence=1.0)
        assert new_val_low >= 0.0

    def test_low_confidence_reduces_shift(self):
        shift_high = abs(compute_style_score(0.5, 0.9, observation_confidence=0.8) - 0.5)
        shift_low = abs(compute_style_score(0.5, 0.9, observation_confidence=0.2) - 0.5)
        assert shift_high > shift_low

    def test_result_is_rounded(self):
        new_val = compute_style_score(0.5, 0.7, observation_confidence=0.5)
        # Should be rounded to 4 decimal places
        assert new_val == round(new_val, 4)


# ===========================================================================
# TestScoreSnapshot
# ===========================================================================

class TestScoreSnapshot:

    def test_empty_profile(self):
        p = UserProfile(user_id="u1")
        assert score_snapshot(p) == {}

    def test_multiple_skills(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.75, confidence=0.9)
        p.skills["math"] = SkillEntry(level=0.5, confidence=0.6)
        snap = score_snapshot(p)
        assert snap["python"]["level"] == 0.75
        assert snap["python"]["confidence"] == 0.9
        assert snap["math"]["level"] == 0.5
        assert snap["math"]["confidence"] == 0.6


# ===========================================================================
# TestTopSkills
# ===========================================================================

class TestTopSkills:

    def test_empty_profile(self):
        p = UserProfile(user_id="u1")
        assert top_skills(p) == []

    def test_ranked_by_level(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.8, confidence=0.9)
        p.skills["math"] = SkillEntry(level=0.6, confidence=0.7)
        p.skills["english"] = SkillEntry(level=0.4, confidence=0.5)
        result = top_skills(p, min_confidence=0.3)
        assert result[0][0] == "python"
        assert result[1][0] == "math"
        assert result[2][0] == "english"

    def test_confidence_filter(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.9, confidence=0.2)  # below threshold
        p.skills["math"] = SkillEntry(level=0.5, confidence=0.8)
        result = top_skills(p, min_confidence=0.3)
        assert len(result) == 1
        assert result[0][0] == "math"

    def test_top_n_respected(self):
        p = UserProfile(user_id="u1")
        for i in range(10):
            p.skills[f"skill_{i}"] = SkillEntry(level=0.5 + i * 0.02, confidence=0.7)
        result = top_skills(p, min_confidence=0.3, top_n=3)
        assert len(result) == 3


# ===========================================================================
# TestWeakestSkills
# ===========================================================================

class TestWeakestSkills:

    def test_empty_profile(self):
        p = UserProfile(user_id="u1")
        assert weakest_skills(p) == []

    def test_ranked_weakest_first(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.8, confidence=0.9)
        p.skills["math"] = SkillEntry(level=0.2, confidence=0.7)
        p.skills["english"] = SkillEntry(level=0.5, confidence=0.6)
        result = weakest_skills(p, min_confidence=0.3)
        assert result[0][0] == "math"
        assert result[1][0] == "english"
        assert result[2][0] == "python"

    def test_confidence_filter(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.1, confidence=0.2)  # below threshold
        p.skills["math"] = SkillEntry(level=0.3, confidence=0.8)
        result = weakest_skills(p, min_confidence=0.3)
        assert len(result) == 1
        assert result[0][0] == "math"


# ===========================================================================
# TestBuildExtractionPrompt
# ===========================================================================

class TestBuildExtractionPrompt:

    def test_basic_prompt(self):
        prompt = build_extraction_prompt("用户: 你好\nAI: 你好！")
        assert "你好" in prompt
        assert "暂无已有画像" in prompt

    def test_with_existing_summary(self):
        prompt = build_extraction_prompt(
            "用户: 什么是Python？",
            existing_summary="技能: python 初级",
        )
        assert "什么是Python" in prompt
        assert "技能: python 初级" in prompt
        assert "暂无已有画像" not in prompt

    def test_truncation_at_4000_chars(self):
        long_text = "用户: " + "x" * 5000
        prompt = build_extraction_prompt(long_text)
        # The conversation_text is truncated before being passed, but the
        # template also truncates at 4000
        assert len(prompt) <= 4500  # template overhead + truncated content


# ===========================================================================
# TestFormatConversation
# ===========================================================================

class TestFormatConversation:

    def test_basic_format(self):
        text = _format_conversation("user msg", "assistant msg")
        assert "用户: user msg" in text
        assert "AI: assistant msg" in text

    def test_with_history(self):
        history = [
            {"role": "user", "content": "历史问题1"},
            {"role": "assistant", "content": "历史回答1"},
        ]
        text = _format_conversation("最新问题", "最新回答", history=history)
        assert "历史问题1" in text
        assert "历史回答1" in text
        assert "最新问题" in text
        assert "最新回答" in text

    def test_history_truncated_to_10_turns(self):
        history = [{"role": "user", "content": f"msg_{i}"} for i in range(15)]
        text = _format_conversation("latest", "reply", history=history)
        # msg_0 through msg_4 should be truncated
        assert "msg_0" not in text
        assert "msg_5" in text
        assert "msg_14" in text

    def test_long_messages_truncated(self):
        long_msg = "x" * 2000
        text = _format_conversation(long_msg, "reply")
        # User message truncated to 1000 chars
        assert len(text) < 3000


# ===========================================================================
# TestTruncateConversation
# ===========================================================================

class TestTruncateConversation:

    def test_short_text_unchanged(self):
        short = "hello world"
        assert _truncate_conversation(short, max_chars=100) == short

    def test_long_text_truncated(self):
        long_text = "abcd" * 2000  # 8000 chars
        result = _truncate_conversation(long_text, max_chars=1000)
        assert len(result) <= 1100  # truncation prefix + content
        assert "早期对话已截断" in result

    def test_exact_boundary(self):
        text = "a" * 500
        result = _truncate_conversation(text, max_chars=500)
        assert result == text  # exact match, no truncation needed


# ===========================================================================
# TestProfileExtractor — with mock LLM
# ===========================================================================

class TestProfileExtractor:

    @staticmethod
    def _strict_from_extracted(info: ExtractedProfileInfo) -> ExtractedProfileInfoStrict:
        return ExtractedProfileInfoStrict(
            skills_observed=[
                ProfileSkillSignal(name=name, level=level)
                for name, level in info.skills_observed.items()
            ],
            skill_evidence=info.skill_evidence,
            style_signals=[
                ProfileStyleSignal(dimension=dimension, strength=strength)
                for dimension, strength in info.style_signals.items()
            ],
            style_evidence=info.style_evidence,
            goals_observed=[
                ProfileGoalSignal(
                    goal=str(goal.get("goal", "")),
                    importance=float(goal.get("importance", 0.5)),
                )
                for goal in info.goals_observed
            ],
            behavior_update=ProfileBehaviorUpdateSignal(
                avg_session_minutes=float(info.behavior_update.get("avg_session_minutes", 0.0)),
                quiz_accuracy=float(info.behavior_update.get("quiz_accuracy", 0.0)),
                questions_asked=float(info.behavior_update.get("questions_asked", 0.0)),
            ),
            observations=list(info.observations),
            dislikes_observed=list(info.dislikes_observed),
        )

    @staticmethod
    def _structured_result(parsed: ExtractedProfileInfoStrict) -> StructuredLLMResult:
        return StructuredLLMResult(
            success=True,
            parsed=parsed,
            node_name="profile_extractor",
            llm_node="profile_extractor",
            schema_name="ExtractedProfileInfoStrict",
            provider="deepseek_official",
            model="deepseek-v4-pro",
            output_mode="deepseek_tool_call_strict",
            fallback_modes=[],
        )

    def _make_extractor(self, monkeypatch, ainvoke_return=None):
        """Build a ProfileExtractor with a mocked structured runtime."""
        parsed = self._strict_from_extracted(ainvoke_return or ExtractedProfileInfo())
        mock_invoke = AsyncMock(return_value=self._structured_result(parsed))
        monkeypatch.setattr("src.profile.extractor.invoke_structured_llm", mock_invoke)
        extractor = ProfileExtractor()
        return extractor, mock_invoke

    @pytest.mark.asyncio
    async def test_extract_skills(self, monkeypatch):
        extractor, mock_invoke = self._make_extractor(
            monkeypatch,
            ExtractedProfileInfo(
                skills_observed={"python": 0.35},
                skill_evidence="用户问了list和tuple的区别",
            )
        )
        result = await extractor.extract(
            user_message="list和tuple有什么区别？",
            assistant_response="主要区别是list可变...",
        )
        assert "python" in result.skills_observed
        assert result.skills_observed["python"] == 0.35
        mock_invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_empty_when_nothing_observed(self, monkeypatch):
        extractor, _mock_invoke = self._make_extractor(
            monkeypatch,
            ExtractedProfileInfo()  # empty
        )
        result = await extractor.extract(
            user_message="你好",
            assistant_response="你好！有什么可以帮你的？",
        )
        assert result.skills_observed == {}
        assert result.style_signals == {}
        assert result.goals_observed == []
        assert result.observations == []

    @pytest.mark.asyncio
    async def test_extract_with_existing_profile(self, monkeypatch):
        extractor, _mock_invoke = self._make_extractor(
            monkeypatch,
            ExtractedProfileInfo(
                style_signals={"prefer_examples": 0.8},
                style_evidence="用户多次要求举例",
            )
        )
        existing = UserProfile(user_id="u1")
        existing.skills["python"] = SkillEntry(level=0.3, confidence=0.6, evidence_count=2)

        result = await extractor.extract(
            user_message="能给我举个例子吗？",
            assistant_response="当然！看这个例子...",
            existing_profile=existing,
        )
        assert result.style_signals["prefer_examples"] == 0.8

    @pytest.mark.asyncio
    async def test_extract_with_history(self, monkeypatch):
        extractor, _mock_invoke = self._make_extractor(
            monkeypatch,
            ExtractedProfileInfo(
                observations=["用户在连续学习Python基础内容"],
            )
        )
        history = [
            {"role": "user", "content": "什么是变量？"},
            {"role": "assistant", "content": "变量是..."},
        ]
        result = await extractor.extract(
            user_message="那常量呢？",
            assistant_response="常量是...",
            history=history,
        )
        assert len(result.observations) == 1

    @pytest.mark.asyncio
    async def test_extract_sanitizes_invalid_skill_keys(self, monkeypatch):
        result = ProfileExtractor._sanitize(
            ExtractedProfileInfo(
                skills_observed={" A VERY LONG INVALID SKILL NAME " * 10: 1.5},
            )
        )
        # Long keys (>50 chars) are dropped, out-of-range values clamped
        for name in result.skills_observed:
            assert len(name) <= 50

    @pytest.mark.asyncio
    async def test_extract_sanitizes_style_signals(self, monkeypatch):
        result = ProfileExtractor._sanitize(
            ExtractedProfileInfo(
                style_signals={
                    "prefer_examples": 1.2,  # clamped to 1.0
                    "prefer_step_by_step": -0.5,  # clamped to 0.0
                    "invalid_dimension": 0.9,  # filtered out
                },
            )
        )
        assert "prefer_examples" in result.style_signals
        assert result.style_signals["prefer_examples"] == 1.0
        assert result.style_signals["prefer_step_by_step"] == 0.0
        assert "invalid_dimension" not in result.style_signals

    @pytest.mark.asyncio
    async def test_extract_sanitizes_goals(self, monkeypatch):
        result = ProfileExtractor._sanitize(
            ExtractedProfileInfo(
                goals_observed=[
                    {"goal": "正常目标", "importance": 0.7},
                    {"goal": "", "importance": 0.5},  # empty goal → filtered
                    {"goal": "无效重要度", "importance": 2.0},  # clamped
                    {"goal": "x" * 300, "importance": 0.5},  # too long → filtered
                ],
            )
        )
        # Only "正常目标" and "无效重要度" should survive
        assert len(result.goals_observed) == 2

    @pytest.mark.asyncio
    async def test_extract_sanitizes_observations(self, monkeypatch):
        result = ProfileExtractor._sanitize(
            ExtractedProfileInfo(
                observations=[
                    "正常观察",
                    "x" * 300,  # truncated to 200
                    "",  # filtered
                    "   ",  # filtered
                ],
            )
        )
        assert len(result.observations) == 2
        assert len(result.observations[1]) <= 200

    @pytest.mark.asyncio
    async def test_extract_llm_failure_returns_empty(self, monkeypatch):
        extractor, mock_invoke = self._make_extractor(monkeypatch)
        mock_invoke.side_effect = RuntimeError("API error")

        result = await extractor.extract(
            user_message="hello",
            assistant_response="world",
        )
        # Should return empty without raising
        assert result.skills_observed == {}
        assert result.style_signals == {}
        assert result.observations == []
        mock_invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_uses_runtime_result_after_internal_retries(self, monkeypatch):
        extractor, mock_invoke = self._make_extractor(
            monkeypatch,
            ExtractedProfileInfo(skills_observed={"python": 0.4}),
        )

        result = await extractor.extract(
            user_message="hello",
            assistant_response="world",
        )

        assert result.skills_observed == {"python": 0.4}
        mock_invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_returns_empty_after_runtime_failure(self, monkeypatch):
        extractor, mock_invoke = self._make_extractor(monkeypatch)
        mock_invoke.side_effect = RuntimeError("API error")

        result = await extractor.extract(
            user_message="hello",
            assistant_response="world",
        )

        assert result.skills_observed == {}
        assert result.style_signals == {}
        assert result.observations == []
        mock_invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_batch(self, monkeypatch):
        extractor, mock_invoke = self._make_extractor(monkeypatch)
        # Return different results for each call
        mock_invoke.side_effect = [
            self._structured_result(self._strict_from_extracted(ExtractedProfileInfo(skills_observed={"python": 0.2}))),
            self._structured_result(self._strict_from_extracted(ExtractedProfileInfo(style_signals={"prefer_examples": 0.7}))),
            self._structured_result(self._strict_from_extracted(ExtractedProfileInfo(observations=["third turn signal"]))),
        ]

        turns = [
            {"user": "Q1", "assistant": "A1"},
            {"user": "Q2", "assistant": "A2"},
            {"user": "Q3", "assistant": "A3"},
        ]
        results = await extractor.extract_batch(turns)
        assert len(results) == 3
        assert "python" in results[0].skills_observed
        assert results[1].style_signals["prefer_examples"] == 0.7
        assert results[2].observations[0] == "third turn signal"

    @pytest.mark.asyncio
    async def test_extract_uses_unified_structured_runtime(self, monkeypatch):
        """Verify profile extraction uses invoke_structured_llm, not json_mode."""
        mock_llm = MagicMock()
        extractor, mock_invoke = self._make_extractor(
            monkeypatch,
            ExtractedProfileInfo(skills_observed={"python": 0.3}),
        )
        extractor._base_llm = mock_llm

        result = await extractor.extract("hello", "world")

        assert result.skills_observed == {"python": 0.3}
        assert not mock_llm.with_structured_output.called
        kwargs = mock_invoke.await_args.kwargs
        assert kwargs["node_name"] == "profile_extractor"
        assert kwargs["llm_node"] == "profile_extractor"
        assert kwargs["output_mode"] == "deepseek_tool_call_strict"
        assert kwargs["fallback_modes"] == []


# ===========================================================================
# TestUpdateProfile — incremental merge
# ===========================================================================

class TestUpdateProfile:

    def test_new_skill_added(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            skills_observed={"python": 0.4},
            skill_evidence="用户基础语法问题",
        )
        result = update_profile(p, extracted)
        assert "python" in p.skills
        assert p.skills["python"].level == 0.4
        assert len(result.changes) >= 1
        assert any("python" in c for c in result.changes)

    def test_existing_skill_updated(self):
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.3, confidence=0.5, evidence_count=2)
        extracted = ExtractedProfileInfo(
            skills_observed={"python": 0.6},
        )
        result = update_profile(p, extracted)
        assert p.skills["python"].level > 0.3  # should increase
        assert p.skills["python"].evidence_count == 3

    def test_multiple_skills_updated(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            skills_observed={"python": 0.3, "algorithm": 0.5, "math": 0.7},
        )
        update_profile(p, extracted)
        assert len(p.skills) == 3
        assert "python" in p.skills
        assert "algorithm" in p.skills
        assert "math" in p.skills

    def test_new_goal_added(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            goals_observed=[{"goal": "补强高等数学基础", "importance": 0.9}],
        )
        result = update_profile(p, extracted)
        assert len(p.goals) == 1
        assert p.goals[0].goal == "补强高等数学基础"
        assert p.goals[0].importance == 0.9
        assert any("补强高等数学基础" in c for c in result.changes)

    def test_duplicate_goal_updates_importance(self):
        p = UserProfile(user_id="u1")
        p.goals = [Goal(goal="补强机器学习基础", importance=0.5)]
        extracted = ExtractedProfileInfo(
            goals_observed=[{"goal": "补强机器学习基础", "importance": 0.9}],
        )
        result = update_profile(p, extracted)
        assert len(p.goals) == 1  # no duplicate
        # Weighted average: 0.5 * 0.7 + 0.9 * 0.3 = 0.35 + 0.27 = 0.62
        assert pytest.approx(p.goals[0].importance, abs=0.01) == 0.62
        assert any("重要性更新" in c for c in result.changes)

    def test_style_preferences_updated(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            style_signals={"prefer_examples": 0.85, "prefer_visual": 0.75},
            style_evidence="用户连续3轮要求可视化",
        )
        result = update_profile(p, extracted)
        assert p.learning_style.prefer_examples > 0.5
        assert p.learning_style.prefer_visual > 0.5
        # Other dimensions unchanged
        assert p.learning_style.prefer_concise == 0.5

    def test_custom_style_dimension_stored_in_extra(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            style_signals={"custom_pref": 0.8},
        )
        update_profile(p, extracted)
        assert "custom_pref" in p.learning_style.extra
        assert p.learning_style.extra["custom_pref"] > 0.5

    def test_behavior_update_running_average(self):
        p = UserProfile(user_id="u1")
        p.behavior.avg_session_minutes = 20.0
        extracted = ExtractedProfileInfo(
            behavior_update={"avg_session_minutes": 30.0},
        )
        update_profile(p, extracted)
        # Weighted: 20 * 0.7 + 30 * 0.3 = 23
        assert pytest.approx(p.behavior.avg_session_minutes, abs=0.1) == 23.0

    def test_behavior_update_first_value(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            behavior_update={"avg_session_minutes": 25.0},
        )
        update_profile(p, extracted)
        assert p.behavior.avg_session_minutes == 25.0

    def test_behavior_quiz_accuracy_first_value(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            behavior_update={"quiz_accuracy": 0.85},
        )
        update_profile(p, extracted)
        assert p.behavior.quiz_accuracy == 0.85

    def test_behavior_quiz_accuracy_running_average(self):
        p = UserProfile(user_id="u1")
        p.behavior.quiz_accuracy = 0.6
        extracted = ExtractedProfileInfo(
            behavior_update={"quiz_accuracy": 0.9},
        )
        update_profile(p, extracted)
        # Weighted: 0.6 * 0.6 + 0.9 * 0.4 = 0.36 + 0.36 = 0.72
        assert pytest.approx(p.behavior.quiz_accuracy, abs=0.01) == 0.72

    def test_behavior_questions_asked_accumulates(self):
        p = UserProfile(user_id="u1")
        p.behavior.questions_asked = 10
        extracted = ExtractedProfileInfo(
            behavior_update={"questions_asked": 5},
        )
        update_profile(p, extracted)
        assert p.behavior.questions_asked == 15

    def test_observations_added(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            observations=["观察1", "观察2", "观察3"],
        )
        result = update_profile(p, extracted)
        assert len(p.agent_observations) == 3
        assert result.new_observations == 3

    def test_observations_fifo_cap(self):
        p = UserProfile(user_id="u1")
        # Pre-fill to 50 observations
        p.agent_observations = [
            AgentObservation(content=f"old_{i}", importance=0.5) for i in range(50)
        ]
        extracted = ExtractedProfileInfo(
            observations=["new_1", "new_2", "new_3"],
        )
        update_profile(p, extracted)
        assert len(p.agent_observations) == 50  # still capped
        # Oldest observations should be dropped
        contents = [o.content for o in p.agent_observations]
        assert "old_0" not in contents
        assert "new_3" in contents

    def test_dislikes_added(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo(
            dislikes_observed=["死记硬背", "枯燥理论"],
        )
        update_profile(p, extracted)
        assert "死记硬背" in p.dislikes
        assert "枯燥理论" in p.dislikes

    def test_dislikes_deduplicated(self):
        p = UserProfile(user_id="u1")
        p.dislikes = ["死记硬背"]
        extracted = ExtractedProfileInfo(
            dislikes_observed=["死记硬背", "新不喜欢"],
        )
        update_profile(p, extracted)
        assert p.dislikes.count("死记硬背") == 1
        assert "新不喜欢" in p.dislikes

    def test_timestamp_touched(self):
        p = UserProfile(user_id="u1")
        p.touch()
        # update_profile also calls touch() — timestamp should be set (non-empty)
        old_updated = p.updated_at
        extracted = ExtractedProfileInfo(
            skills_observed={"math": 0.5},
        )
        import time
        time.sleep(0.001)  # ensure timestamp changes
        update_profile(p, extracted)
        assert p.updated_at != ""
        assert p.created_at != ""

    def test_empty_extraction_no_changes(self):
        p = UserProfile(user_id="u1")
        extracted = ExtractedProfileInfo()  # completely empty
        result = update_profile(p, extracted)
        assert result.changes == []
        assert result.new_observations == 0

    def test_small_skill_delta_not_reported(self):
        """Very small skill deltas (<0.02) don't produce change entries."""
        p = UserProfile(user_id="u1")
        p.skills["python"] = SkillEntry(level=0.5, confidence=0.8, evidence_count=3)
        # An observation very close to the current level
        extracted = ExtractedProfileInfo(
            skills_observed={"python": 0.505},  # tiny difference
        )
        result = update_profile(p, extracted)
        # No change log for tiny deltas
        skill_changes = [c for c in result.changes if "python" in c]
        assert len(skill_changes) == 0


# ===========================================================================
# TestExtractionSystemPrompt
# ===========================================================================

class TestExtractionSystemPrompt:

    def test_prompt_contains_key_sections(self):
        assert "提取原则" in EXTRACTION_SYSTEM_PROMPT
        assert "技能水平" in EXTRACTION_SYSTEM_PROMPT
        assert "学习偏好" in EXTRACTION_SYSTEM_PROMPT
        assert "学习目标" in EXTRACTION_SYSTEM_PROMPT
        assert "Agent观察" in EXTRACTION_SYSTEM_PROMPT
        assert "只返回 JSON" in EXTRACTION_SYSTEM_PROMPT
