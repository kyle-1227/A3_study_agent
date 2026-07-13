"""Injected runtime boundary for strict learner-guidance graph nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from src.learning_guidance.contracts import (
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
    LearnerPathPlanV1,
    LearnerProfileSnapshotV1,
    ResourceRecommendationBatchV1,
    ResourceRecommendationEngineRequestV1,
)


ProfileSnapshotLoader = Callable[
    [str],
    Awaitable[LearnerProfileSnapshotV1 | None],
]
HistorySnapshotLoader = Callable[
    [str, str],
    Awaitable[LearnerHistorySnapshotV1 | None],
]
LearnerPathEngine = Callable[
    [LearnerPathEngineRequestV1],
    Awaitable[LearnerPathPlanV1],
]
ResourceRecommendationEngine = Callable[
    [ResourceRecommendationEngineRequestV1],
    Awaitable[ResourceRecommendationBatchV1],
]


class LearningGuidanceContractError(RuntimeError):
    """An injected dependency returned data that violates request binding."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


@dataclass(frozen=True, slots=True)
class LearningGuidanceRuntime:
    """All dependencies required by the strict path and recommendation nodes.

    The graph layer never imports a concrete profile store, memory store, or
    scoring engine.  Production wiring must inject hardened adapters explicitly;
    there is intentionally no default implementation or fallback engine.
    """

    load_profile: ProfileSnapshotLoader
    load_history: HistorySnapshotLoader
    plan_learning_path: LearnerPathEngine
    recommend_resources: ResourceRecommendationEngine

    def __post_init__(self) -> None:
        dependencies = {
            "load_profile": self.load_profile,
            "load_history": self.load_history,
            "plan_learning_path": self.plan_learning_path,
            "recommend_resources": self.recommend_resources,
        }
        for name, dependency in dependencies.items():
            if not callable(dependency):
                raise TypeError(f"{name} must be callable")


__all__ = [
    "HistorySnapshotLoader",
    "LearnerPathEngine",
    "LearningGuidanceContractError",
    "LearningGuidanceRuntime",
    "ProfileSnapshotLoader",
    "ResourceRecommendationEngine",
]
