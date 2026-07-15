"""Versioned production adapters for strict learner guidance."""

from src.learning_guidance.adapters.history import (
    HISTORY_ADAPTER_VERSION,
    HistoryAdapterError,
    HistorySnapshotAdapterV1,
)
from src.learning_guidance.adapters.path import (
    PATH_ENGINE_VERSION,
    LearnerPathEngineV1,
    PathEngineError,
)
from src.learning_guidance.adapters.profile import (
    PROFILE_ADAPTER_VERSION,
    ProfileAdapterError,
    ProfileSnapshotAdapterV1,
    profile_goal_fingerprint,
)
from src.learning_guidance.adapters.recommendation import (
    RECOMMENDATION_ENGINE_VERSION,
    RecommendationEngineError,
    ResourceRecommendationEngineV1,
)

__all__ = [
    "HISTORY_ADAPTER_VERSION",
    "HistoryAdapterError",
    "HistorySnapshotAdapterV1",
    "LearnerPathEngineV1",
    "PATH_ENGINE_VERSION",
    "PROFILE_ADAPTER_VERSION",
    "PathEngineError",
    "ProfileAdapterError",
    "ProfileSnapshotAdapterV1",
    "profile_goal_fingerprint",
    "RECOMMENDATION_ENGINE_VERSION",
    "RecommendationEngineError",
    "ResourceRecommendationEngineV1",
]
