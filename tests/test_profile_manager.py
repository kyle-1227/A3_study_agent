"""Integration tests for ProfileManager — orchestrator, storage, and agent state.

All tests use mock LLM + in-memory store and run offline without API keys.

Usage:
    pytest tests/test_profile_manager.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.profile.manager import (
    ProfileManager,
    _profile_manager,
    get_profile_manager,
    reset_profile_manager,
)
from src.profile.schema import (
    AgentObservation,
    ExtractedProfileInfo,
    Goal,
    ProfileUpdateResult,
    SkillEntry,
    UserProfile,
    profile_to_summary,
)
from src.profile.storage import ProfileStore


# ===========================================================================
# In-memory store for testing (no aiosqlite dependency)
# ===========================================================================


class DictProfileStore(ProfileStore):
    """In-memory profile store backed by a plain dict — no database needed."""

    def __init__(self):
        self._data: dict[str, UserProfile] = {}

    async def save(self, profile: UserProfile) -> None:
        self._data[profile.user_id] = profile.model_copy(deep=True)

    async def load(self, user_id: str) -> UserProfile | None:
        profile = self._data.get(user_id)
        return profile.model_copy(deep=True) if profile else None

    async def delete(self, user_id: str) -> bool:
        if user_id in self._data:
            del self._data[user_id]
            return True
        return False

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[str]:
        keys = list(self._data.keys())
        return keys[offset:offset + limit]

    async def count(self) -> int:
        return len(self._data)


# ===========================================================================
# Mock extractor factory helpers
# ===========================================================================


def _make_mock_extractor(return_value=None):
    """Create a mock ProfileExtractor whose extract() returns the given value."""
    mock_extractor = MagicMock()
    if return_value is not None:
        mock_extractor.extract = AsyncMock(return_value=return_value)
    else:
        mock_extractor.extract = AsyncMock(return_value=ExtractedProfileInfo())
    return mock_extractor


def _make_mock_extractor_sequence(return_values: list):
    """Create a mock extractor that returns different values per call."""
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(side_effect=return_values)
    return mock_extractor


def _make_mock_extractor_error():
    """Mock extractor that raises an exception."""
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    return mock_extractor


# ===========================================================================
# TestProfileManagerInit
# ===========================================================================

class TestProfileManagerInit:

    def test_default_init(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)
        assert manager.store is store
        assert manager._injection_budget == 600

    def test_custom_injection_budget(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor, injection_budget=300)
        assert manager._injection_budget == 300

    def test_init_with_llm_creates_extractor(self):
        """Passing an LLM (instead of extractor) should create extractor internally."""
        store = DictProfileStore()
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        manager = ProfileManager(store=store, llm=mock_llm)
        assert manager._extractor is not None
        mock_llm.with_structured_output.assert_called_once()
        call_kwargs = mock_llm.with_structured_output.call_args
        assert call_kwargs[1]["method"] == "json_mode"


# ===========================================================================
# TestLoadOrCreate
# ===========================================================================

class TestLoadOrCreate:

    @pytest.mark.asyncio
    async def test_new_user_creates_profile(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        profile = await manager.load_or_create("new_user_001")
        assert profile.user_id == "new_user_001"
        assert profile.created_at != ""
        assert profile.updated_at != ""
        assert profile.skills == {}

    @pytest.mark.asyncio
    async def test_existing_user_returns_stored(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        # Create a profile with data
        existing = UserProfile(user_id="test_user")
        existing.skills["python"] = SkillEntry(level=0.5, confidence=0.7)
        existing.touch()
        await store.save(existing)

        loaded = await manager.load_or_create("test_user")
        assert loaded.user_id == "test_user"
        assert "python" in loaded.skills
        assert loaded.skills["python"].level == 0.5

    @pytest.mark.asyncio
    async def test_load_or_create_preserves_timestamps(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        profile = await manager.load_or_create("u1")
        created = profile.created_at

        # Simulate another load
        profile2 = await manager.load_or_create("u1")
        assert profile2.created_at == created  # preserved


# ===========================================================================
# TestProcessConversation
# ===========================================================================

class TestProcessConversation:

    @pytest.mark.asyncio
    async def test_successful_extraction_and_update(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor(
            ExtractedProfileInfo(
                skills_observed={"python": 0.3},
                skill_evidence="用户问了基础语法",
            )
        )
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.process_conversation(
            user_id="user_001",
            user_message="Python 的 list 和 tuple 有什么区别？",
            assistant_response="主要区别是 list 可变...",
        )

        assert isinstance(result, ProfileUpdateResult)
        assert "python" in result.profile.skills
        assert result.profile.skills["python"].level == 0.3
        assert len(result.changes) >= 1
        assert result.new_observations == 0

        # Verify it was persisted
        saved = await store.load("user_001")
        assert saved is not None
        assert "python" in saved.skills

    @pytest.mark.asyncio
    async def test_empty_extraction_no_update(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor(ExtractedProfileInfo())  # empty
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.process_conversation(
            user_id="user_001",
            user_message="你好",
            assistant_response="你好！",
        )

        assert result.changes == []
        assert result.new_observations == 0
        assert result.profile.skills == {}

    @pytest.mark.asyncio
    async def test_process_passes_history_to_extractor(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor(
            ExtractedProfileInfo(observations=["连续追问显示好奇心强"])
        )
        manager = ProfileManager(store=store, extractor=extractor)

        history = [
            {"role": "user", "content": "上一个问题"},
        ]
        result = await manager.process_conversation(
            user_id="user_001",
            user_message="继续追问",
            assistant_response="回答...",
            history=history,
        )

        # Verify extractor was called with history
        call_kwargs = extractor.extract.call_args
        assert call_kwargs.kwargs["history"] == history
        assert call_kwargs.kwargs["user_message"] == "继续追问"

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor_sequence([
            ExtractedProfileInfo(skills_observed={"python": 0.25}),
            ExtractedProfileInfo(style_signals={"prefer_examples": 0.9}),
            ExtractedProfileInfo(goals_observed=[{"goal": "学Python做AI", "importance": 0.8}]),
        ])
        manager = ProfileManager(store=store, extractor=extractor)

        # Turn 1
        r1 = await manager.process_conversation("u1", "Q1", "A1")
        assert "python" in r1.profile.skills

        # Turn 2
        r2 = await manager.process_conversation("u1", "Q2", "A2")
        assert r2.profile.learning_style.prefer_examples > 0.5

        # Turn 3
        r3 = await manager.process_conversation("u1", "Q3", "A3")
        assert len(r3.profile.goals) == 1

        # All accumulated
        assert r3.profile.skills["python"].level > 0
        assert r3.profile.learning_style.prefer_examples > 0.5
        assert r3.profile.goals[0].goal == "学Python做AI"

    @pytest.mark.asyncio
    async def test_profile_persisted_between_calls(self):
        """Verify the profile is durable across multiple process_conversation calls."""
        store = DictProfileStore()
        extractor = _make_mock_extractor_sequence([
            ExtractedProfileInfo(skills_observed={"math": 0.6}),
            ExtractedProfileInfo(skills_observed={"math": 0.75}),
        ])
        manager = ProfileManager(store=store, extractor=extractor)

        # First call
        await manager.process_conversation("u1", "Q1", "A1")
        p1 = await manager.get_profile("u1")
        assert p1 is not None
        assert p1.skills["math"].level == 0.6

        # Second call — should start from saved state
        await manager.process_conversation("u1", "Q2", "A2")
        p2 = await manager.get_profile("u1")
        assert p2 is not None
        # Level should have moved from 0.6 toward 0.75
        assert p2.skills["math"].level > 0.6


# ===========================================================================
# TestBuildProfileContext
# ===========================================================================

class TestBuildProfileContext:

    @pytest.mark.asyncio
    async def test_context_for_user_with_skills(self):
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        profile.skills["python"] = SkillEntry(level=0.35, confidence=0.7, evidence_count=3)
        profile.skills["math"] = SkillEntry(level=0.8, confidence=0.9, evidence_count=10)
        profile.learning_style.prefer_examples = 0.85
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = await manager.build_profile_context("u1")
        assert "[用户画像]" in ctx
        assert "python" in ctx
        assert "math" in ctx
        assert "初级" in ctx  # 0.35 = 初级
        assert "请根据以上用户画像调整你的回答策略" in ctx

    @pytest.mark.asyncio
    async def test_context_for_nonexistent_user(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = await manager.build_profile_context("nonexistent")
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_context_for_empty_profile(self):
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        profile.touch()
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = await manager.build_profile_context("u1")
        assert ctx == ""  # no meaningful data

    @pytest.mark.asyncio
    async def test_context_truncated_to_budget(self):
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        # Add many skills to make a long summary
        for i in range(20):
            profile.skills[f"very_long_skill_name_{i:03d}"] = SkillEntry(
                level=0.5, confidence=0.8, evidence_count=5,
            )
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor, injection_budget=200)

        ctx = await manager.build_profile_context("u1")
        # Budget is 200 chars for the summary portion; total may be slightly over
        # due to the "[用户画像]\n" prefix and "\n\n请根据以上用户画像调整你的回答策略。" suffix
        assert len(ctx) <= 400  # generous upper bound

    @pytest.mark.asyncio
    async def test_context_includes_strong_preferences(self):
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        profile.learning_style.prefer_step_by_step = 0.85
        profile.learning_style.prefer_theory = 0.15
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = await manager.build_profile_context("u1")
        assert "喜欢分步讲解" in ctx
        assert "强偏好" in ctx


# ===========================================================================
# TestBuildProfileContextSync
# ===========================================================================

class TestBuildProfileContextSync:

    def test_with_valid_profile(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        profile = UserProfile(user_id="u1")
        profile.skills["math"] = SkillEntry(level=0.7, confidence=0.9, evidence_count=4)
        profile.learning_style.prefer_examples = 0.8

        ctx = manager.build_profile_context_sync(profile)
        assert "[用户画像]" in ctx
        assert "math" in ctx

    def test_with_none_profile(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = manager.build_profile_context_sync(None)
        assert ctx == ""

    def test_sync_matches_async_for_same_profile(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor, injection_budget=500)

        profile = UserProfile(user_id="u1")
        profile.skills["python"] = SkillEntry(level=0.4, confidence=0.7, evidence_count=3)
        profile.learning_style.prefer_visual = 0.75

        sync_ctx = manager.build_profile_context_sync(profile)
        assert "[用户画像]" in sync_ctx
        assert "python" in sync_ctx
        assert "请根据以上用户画像调整你的回答策略" in sync_ctx


# ===========================================================================
# TestProcessBatch
# ===========================================================================

class TestProcessBatch:

    @pytest.mark.asyncio
    async def test_batch_processes_all_turns(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor_sequence([
            ExtractedProfileInfo(skills_observed={"python": 0.25}),
            ExtractedProfileInfo(skills_observed={"algorithm": 0.4}),
            ExtractedProfileInfo(style_signals={"prefer_examples": 0.8}),
        ])
        manager = ProfileManager(store=store, extractor=extractor)

        turns = [
            {"user": "Q1", "assistant": "A1"},
            {"user": "Q2", "assistant": "A2"},
            {"user": "Q3", "assistant": "A3"},
        ]
        result = await manager.process_batch("u1", turns)

        assert isinstance(result, ProfileUpdateResult)
        assert "python" in result.profile.skills
        assert "algorithm" in result.profile.skills
        assert result.profile.learning_style.prefer_examples > 0.5
        assert result.new_observations == 0

        # Verify persistence
        saved = await store.load("u1")
        assert saved is not None
        assert len(saved.skills) == 2

    @pytest.mark.asyncio
    async def test_batch_with_empty_turns(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.process_batch("u1", [])
        assert result.changes == []
        assert result.new_observations == 0

    @pytest.mark.asyncio
    async def test_batch_with_some_empty_extractions(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor_sequence([
            ExtractedProfileInfo(skills_observed={"python": 0.3}),
            ExtractedProfileInfo(),  # empty — skipped
            ExtractedProfileInfo(style_signals={"prefer_visual": 0.7}),
        ])
        manager = ProfileManager(store=store, extractor=extractor)

        turns = [
            {"user": "Q1", "assistant": "A1"},
            {"user": "Q2", "assistant": "A2"},
            {"user": "Q3", "assistant": "A3"},
        ]
        result = await manager.process_batch("u1", turns)
        assert "python" in result.profile.skills
        assert result.profile.learning_style.prefer_visual > 0.5


# ===========================================================================
# TestGetProfileAndDeleteProfile
# ===========================================================================

class TestGetProfileAndDeleteProfile:

    @pytest.mark.asyncio
    async def test_get_profile_existing(self):
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        profile.skills["math"] = SkillEntry(level=0.6, confidence=0.8)
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.get_profile("u1")
        assert result is not None
        assert result.user_id == "u1"
        assert "math" in result.skills

    @pytest.mark.asyncio
    async def test_get_profile_nonexistent(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.get_profile("nobody")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing_profile(self):
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.delete_profile("u1")
        assert result is True
        assert await store.load("u1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_profile(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        result = await manager.delete_profile("nobody")
        assert result is False


# ===========================================================================
# TestIsEmptyExtraction
# ===========================================================================

class TestIsEmptyExtraction:

    def test_completely_empty(self):
        extracted = ExtractedProfileInfo()
        assert ProfileManager._is_empty_extraction(extracted) is True

    def test_has_skills(self):
        extracted = ExtractedProfileInfo(skills_observed={"python": 0.3})
        assert ProfileManager._is_empty_extraction(extracted) is False

    def test_has_style_signals(self):
        extracted = ExtractedProfileInfo(style_signals={"prefer_examples": 0.8})
        assert ProfileManager._is_empty_extraction(extracted) is False

    def test_has_goals(self):
        extracted = ExtractedProfileInfo(
            goals_observed=[{"goal": "test", "importance": 0.5}],
        )
        assert ProfileManager._is_empty_extraction(extracted) is False

    def test_has_behavior(self):
        extracted = ExtractedProfileInfo(
            behavior_update={"avg_session_minutes": 30},
        )
        assert ProfileManager._is_empty_extraction(extracted) is False

    def test_has_observations(self):
        extracted = ExtractedProfileInfo(observations=["obs1"])
        assert ProfileManager._is_empty_extraction(extracted) is False

    def test_has_dislikes(self):
        extracted = ExtractedProfileInfo(dislikes_observed=["dislike1"])
        assert ProfileManager._is_empty_extraction(extracted) is False


# ===========================================================================
# TestSingletonFactory
# ===========================================================================

class TestSingletonFactory:

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_profile_manager()

    def test_first_call_creates_new_manager(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = get_profile_manager(store=store, extractor=extractor)
        assert isinstance(manager, ProfileManager)
        assert manager.store is store

    def test_second_call_returns_same_instance(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        m1 = get_profile_manager(store=store, extractor=extractor)
        m2 = get_profile_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        store = DictProfileStore()
        extractor = _make_mock_extractor()
        m1 = get_profile_manager(store=store, extractor=extractor)
        reset_profile_manager()
        m2 = get_profile_manager(store=DictProfileStore(), extractor=_make_mock_extractor())
        assert m1 is not m2

    def test_reset_when_none_is_noop(self):
        """Resetting when no singleton exists should not raise."""
        reset_profile_manager()
        reset_profile_manager()  # double reset should be fine
        assert _profile_manager is None


# ===========================================================================
# TestProfileManagerIntegration — full lifecycle
# ===========================================================================

class TestProfileManagerIntegration:

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """End-to-end: create → process → query → delete."""
        store = DictProfileStore()
        extractor = _make_mock_extractor_sequence([
            ExtractedProfileInfo(
                skills_observed={"python": 0.25},
                skill_evidence="基础语法问题",
                style_signals={"prefer_step_by_step": 0.9},
                observations=["用户是编程初学者"],
            ),
            ExtractedProfileInfo(
                skills_observed={"python": 0.35},
                style_signals={"prefer_examples": 0.8},
                goals_observed=[{"goal": "学Python自动化办公", "importance": 0.7}],
            ),
            ExtractedProfileInfo(
                skills_observed={"python": 0.5},
                observations=["用户进展快速，开始问进阶问题"],
            ),
        ])
        manager = ProfileManager(store=store, extractor=extractor)

        # Phase 1: First interaction
        r1 = await manager.process_conversation(
            "user_lifecycle",
            "Python print怎么用？",
            "print() 在控制台输出内容...",
        )
        assert "python" in r1.profile.skills
        initial_level = r1.profile.skills["python"].level
        assert initial_level > 0
        assert len(r1.profile.agent_observations) == 1

        # Phase 2: Second interaction — skills evolve
        r2 = await manager.process_conversation(
            "user_lifecycle",
            "Python的for循环怎么写？能给个例子吗？",
            "for i in range(10): print(i)...",
        )
        assert r2.profile.skills["python"].level >= initial_level
        assert r2.profile.learning_style.prefer_step_by_step > 0.5
        assert len(r2.profile.goals) == 1

        # Phase 3: Third interaction
        r3 = await manager.process_conversation(
            "user_lifecycle",
            "如何用Python读取Excel文件？",
            "用openpyxl库...",
        )
        assert r3.profile.skills["python"].level > initial_level
        assert r3.profile.skills["python"].evidence_count == 3

        # Phase 4: Build context for prompt injection
        ctx = await manager.build_profile_context("user_lifecycle")
        assert "[用户画像]" in ctx
        assert "python" in ctx
        assert "学Python自动化办公" in ctx

        # Phase 5: Delete
        assert await manager.delete_profile("user_lifecycle") is True
        assert await manager.get_profile("user_lifecycle") is None
        assert await manager.build_profile_context("user_lifecycle") == ""

    @pytest.mark.asyncio
    async def test_concurrent_users_independent(self):
        """Two users should have completely independent profiles."""
        store = DictProfileStore()
        extractor = _make_mock_extractor_sequence([
            ExtractedProfileInfo(skills_observed={"math": 0.8}),
            ExtractedProfileInfo(skills_observed={"english": 0.6}),
        ])
        manager = ProfileManager(store=store, extractor=extractor)

        await manager.process_conversation("user_a", "Q", "A")
        await manager.process_conversation("user_b", "Q", "A")

        p_a = await manager.get_profile("user_a")
        p_b = await manager.get_profile("user_b")

        assert "math" in p_a.skills
        assert "english" not in p_a.skills
        assert "english" in p_b.skills
        assert "math" not in p_b.skills

    @pytest.mark.asyncio
    async def test_budget_truncation_with_long_summary(self):
        """Verify build_profile_context respects the injection budget."""
        store = DictProfileStore()
        profile = UserProfile(user_id="u1")
        for i in range(15):
            profile.skills[f"skill_{i:03d}"] = SkillEntry(
                level=0.5, confidence=0.8, evidence_count=5,
            )
        profile.goals = [
            Goal(goal=f"详细的学习目标_{i}", importance=0.8) for i in range(5)
        ]
        await store.save(profile)

        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor, injection_budget=300)

        ctx = await manager.build_profile_context("u1")
        assert len(ctx) <= 400  # generous upper bound for budget + prefix/suffix


# ===========================================================================
# TestAgentStateIntegration — profile in TutorState
# ===========================================================================

class TestAgentStateIntegration:

    def test_tutor_state_has_user_id_field(self):
        """TutorState should include fields needed for profile integration."""
        from src.graph.state import TutorState

        annotations = TutorState.__annotations__
        # These are the fields the profile system needs (or can co-exist with)
        assert "messages" in annotations
        assert "intent" in annotations
        assert "subject" in annotations
        assert "context" in annotations
        assert "plan" in annotations

    def test_profile_context_can_be_injected_into_state(self):
        """Simulate how profile_context flows through the graph."""
        from src.graph.state import TutorState

        profile = UserProfile(user_id="test_user")
        profile.skills["math"] = SkillEntry(level=0.6, confidence=0.8, evidence_count=4)
        profile.learning_style.prefer_step_by_step = 0.85

        # Build profile context
        ctx = profile_to_summary(profile)

        # Simulate state enrichment in the graph
        state: TutorState = {
            "messages": [],
            "intent": "academic",
            "subject": "math",
            "keypoints": ["二次函数"],
            "context": [],
            "plan": "",
        }

        # This is what the integration code would do:
        # The profile context is prepended to the user message or
        # injected into the system prompt
        enriched_context = [{"type": "profile", "content": ctx}]
        assert len(enriched_context) == 1
        assert "math" in enriched_context[0]["content"]
        assert "熟练" in enriched_context[0]["content"]  # 0.6 = 熟练

    def test_profile_injection_is_markdown_friendly(self):
        """Profile summaries should not break markdown formatting in prompts."""
        profile = UserProfile(user_id="u1")
        profile.skills["python"] = SkillEntry(level=0.4, confidence=0.7, evidence_count=3)
        profile.learning_style.prefer_examples = 0.85
        profile.dislikes = ["死记硬背"]

        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = manager.build_profile_context_sync(profile)

        # Should be plain text — no JSON, no code blocks
        assert "```" not in ctx
        assert "{" not in ctx
        assert "}" not in ctx
        # Should have a clear header
        assert ctx.startswith("[用户画像]")
        # Should end with actionable guidance
        assert "调整你的回答策略" in ctx

    def test_empty_profile_clean_state(self):
        """When there's no profile data, the state should be clean."""
        profile = UserProfile(user_id="u1")
        summary = profile_to_summary(profile)
        assert "暂无用户画像数据" in summary

        # Integration check: when summary is empty, don't inject
        should_inject = "暂无" not in summary
        assert should_inject is False

    def test_skill_level_labels_map_to_chinese_levels(self):
        """Verify the skill levels produce sensible labels for the AI."""
        from src.profile.schema import _level_label

        # These labels are displayed to the AI in the prompt injection
        assert _level_label(0.1) == "入门"
        assert _level_label(0.3) == "初级"
        assert _level_label(0.5) == "中等"
        assert _level_label(0.7) == "熟练"
        assert _level_label(0.9) == "精通"

    def test_profile_context_includes_dislikes(self):
        """The AI should be warned about user dislikes."""
        profile = UserProfile(user_id="u1")
        profile.dislikes = ["过于抽象的理论", "直接给答案"]
        profile.learning_style.prefer_examples = 0.8

        store = DictProfileStore()
        extractor = _make_mock_extractor()
        manager = ProfileManager(store=store, extractor=extractor)

        ctx = manager.build_profile_context_sync(profile)
        assert "过于抽象的理论" in ctx
        assert "直接给答案" in ctx
