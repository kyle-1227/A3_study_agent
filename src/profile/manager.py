"""
ProfileManager — unified orchestrator for the user profile lifecycle.

This is the primary entry point. It coordinates:
  extraction → update → persist → prompt injection

Usage::

    manager = ProfileManager(llm=llm, store=SQLiteProfileStore())

    # After each conversation turn:
    await manager.process_conversation(user_id, user_message, assistant_response)

    # Before generating a response:
    prompt_prefix = await manager.build_profile_context(user_id)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.profile.extractor import ProfileExtractor, extractor_from_env
from src.profile.schema import (
    ExtractedProfileInfo,
    ProfileUpdateResult,
    UserProfile,
    profile_to_summary,
)
from src.profile.storage import ProfileStore, SQLiteProfileStore
from src.profile.updater import update_profile

logger = logging.getLogger(__name__)

# Default profile injection budget — max chars injected into the prompt
_DEFAULT_INJECTION_BUDGET = 600


class ProfileManager:
    """Unified orchestrator for the user profile system.

    Responsibilities:
    1. Load/save profiles via the storage layer
    2. Extract new signals from conversations via the extractor
    3. Merge extracted signals into the existing profile
    4. Build a prompt injection string for the agent
    """

    def __init__(
        self,
        store: ProfileStore | None = None,
        extractor: ProfileExtractor | None = None,
        llm=None,
        injection_budget: int = _DEFAULT_INJECTION_BUDGET,
    ):
        """Initialize the profile manager.

        Args:
            store: Profile storage backend. Defaults to SQLite at data/profile.db.
            extractor: LLM-based extractor. Created from env if not provided.
            llm: LLM instance for the extractor (if extractor not provided).
            injection_budget: Max characters for prompt injection.
        """
        self._store = store or SQLiteProfileStore()
        if extractor:
            self._extractor = extractor
        elif llm:
            self._extractor = ProfileExtractor(llm)
        else:
            self._extractor = extractor_from_env()
        self._injection_budget = injection_budget

    @property
    def store(self) -> ProfileStore:
        return self._store

    # ── Core API ───────────────────────────────────────────────────────────

    async def process_conversation(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        history: list[dict[str, str]] | None = None,
    ) -> ProfileUpdateResult:
        """Process a single conversation turn end-to-end.

        Flow:
            1. Load existing profile (or create new)
            2. Extract new signals from the conversation
            3. Merge signals into profile
            4. Persist updated profile

        Args:
            user_id: Unique user identifier.
            user_message: What the user said.
            assistant_response: What the assistant replied.
            history: Optional recent conversation history.

        Returns:
            ProfileUpdateResult with the updated profile and change summary.
        """
        # 1. Load or create
        profile = await self.load_or_create(user_id)

        # 2. Extract
        extracted = await self._extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
            existing_profile=profile,
            history=history,
        )

        if self._is_empty_extraction(extracted):
            logger.debug("No new profile signals extracted for user=%s", user_id)
            return ProfileUpdateResult(profile=profile, changes=[], new_observations=0)

        # 3. Update
        result = update_profile(profile, extracted)

        # 4. Persist
        await self._store.save(result.profile)

        if result.changes:
            logger.info(
                "Profile updated for user=%s: %d changes, %d new observations",
                user_id, len(result.changes), result.new_observations,
            )
            for change in result.changes:
                logger.debug("  %s", change)

        return result

    async def load_or_create(self, user_id: str) -> UserProfile:
        """Load an existing profile or create a fresh one."""
        profile = await self._store.load(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            profile.touch()
            logger.info("Created new profile for user=%s", user_id)
        return profile

    async def get_profile(self, user_id: str) -> UserProfile | None:
        """Get the current profile, or None."""
        return await self._store.load(user_id)

    async def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile completely."""
        return await self._store.delete(user_id)

    # ── Prompt injection ──────────────────────────────────────────────────

    async def build_profile_context(self, user_id: str) -> str:
        """Build a natural-language profile summary for prompt injection.

        Returns an empty string if the user has no profile or no meaningful data.

        Usage::

            ctx = await manager.build_profile_context(user_id)
            system_prompt = f"{base_prompt}\n\n{ctx}" if ctx else base_prompt
        """
        profile = await self._store.load(user_id)
        if profile is None:
            return ""

        summary = profile_to_summary(profile)
        if summary == "（暂无用户画像数据）":
            return ""

        # Truncate to budget
        if len(summary) > self._injection_budget:
            summary = summary[:self._injection_budget - 3] + "..."

        return f"[用户画像]\n{summary}\n\n请根据以上用户画像调整你的回答策略。"

    def build_profile_context_sync(self, profile: UserProfile | None) -> str:
        """Synchronous version for use when you already have the profile object.

        Useful in graph nodes where the profile is passed via state.
        """
        if profile is None:
            return ""
        summary = profile_to_summary(profile)
        if summary == "（暂无用户画像数据）":
            return ""
        if len(summary) > self._injection_budget:
            summary = summary[:self._injection_budget - 3] + "..."
        return f"[用户画像]\n{summary}\n\n请根据以上用户画像调整你的回答策略。"

    # ── Batch operations ──────────────────────────────────────────────────

    async def process_batch(
        self,
        user_id: str,
        turns: list[dict[str, str]],
    ) -> ProfileUpdateResult:
        """Process multiple conversation turns in batch.

        Each turn should be: {"user": "...", "assistant": "..."}
        """
        profile = await self.load_or_create(user_id)
        total_changes: list[str] = []
        total_obs = 0

        for i, turn in enumerate(turns):
            history = [
                {"role": "user", "content": t["user"]}
                for t in turns[:i]
            ]
            extracted = await self._extractor.extract(
                user_message=turn["user"],
                assistant_response=turn["assistant"],
                existing_profile=profile,
                history=history,
            )
            if self._is_empty_extraction(extracted):
                continue
            result = update_profile(profile, extracted)
            total_changes.extend(result.changes)
            total_obs += result.new_observations

        if total_changes:
            await self._store.save(profile)
            logger.info(
                "Batch profile update for user=%s: %d changes across %d turns",
                user_id, len(total_changes), len(turns),
            )

        return ProfileUpdateResult(
            profile=profile,
            changes=total_changes,
            new_observations=total_obs,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_empty_extraction(extracted: ExtractedProfileInfo) -> bool:
        """Check if the extraction contains any meaningful data."""
        return (
            not extracted.skills_observed
            and not extracted.style_signals
            and not extracted.goals_observed
            and not extracted.behavior_update
            and not extracted.observations
            and not extracted.dislikes_observed
        )


# ── Singleton factory ──────────────────────────────────────────────────────


_profile_manager: ProfileManager | None = None


def get_profile_manager(
    store: ProfileStore | None = None,
    extractor: ProfileExtractor | None = None,
) -> ProfileManager:
    """Get or create a singleton ProfileManager.

    On first call, creates the manager from environment / defaults.
    Subsequent calls return the same instance.
    """
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager(store=store, extractor=extractor)
    return _profile_manager


def reset_profile_manager() -> None:
    """Reset the singleton (useful for testing)."""
    global _profile_manager
    _profile_manager = None
