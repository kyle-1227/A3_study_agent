"""
AI-native user profile system for the AI Study Agent.

Public API::

    from src.profile import ProfileManager, SQLiteProfileStore, UserProfile

    manager = ProfileManager()

    # After each conversation turn:
    await manager.process_conversation(user_id, user_msg, assistant_msg)

    # Before generating a response:
    context = await manager.build_profile_context(user_id)
"""

from src.profile.manager import ProfileManager, get_profile_manager, reset_profile_manager
from src.profile.schema import (
    AgentObservation,
    BehaviorProfile,
    ExtractedProfileInfo,
    Goal,
    LearningStyle,
    ProfileUpdateResult,
    SkillEntry,
    UserProfile,
    profile_to_summary,
)
from src.profile.storage import ProfileStore, SQLiteProfileStore, create_store
from src.profile.extractor import ProfileExtractor, extractor_from_env
from src.profile.scorer import (
    compute_skill_score,
    compute_style_score,
    score_snapshot,
    top_skills,
    weakest_skills,
)
from src.profile.updater import update_profile

__all__ = [
    # Manager
    "ProfileManager",
    "get_profile_manager",
    "reset_profile_manager",
    # Schema
    "UserProfile",
    "SkillEntry",
    "LearningStyle",
    "Goal",
    "BehaviorProfile",
    "AgentObservation",
    "ExtractedProfileInfo",
    "ProfileUpdateResult",
    "profile_to_summary",
    # Storage
    "ProfileStore",
    "SQLiteProfileStore",
    "create_store",
    # Extractor
    "ProfileExtractor",
    "extractor_from_env",
    # Scorer
    "compute_skill_score",
    "compute_style_score",
    "score_snapshot",
    "top_skills",
    "weakest_skills",
    # Updater
    "update_profile",
]
