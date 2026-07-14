"""Injected runtime boundary for strict learner-guidance graph nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from src.learning_guidance.contracts import (
    LEARNER_PATH_PROVIDER_MAX_CHARS,
    LEARNER_PATH_PROVIDER_MAX_STEPS,
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
    LearnerPathPlanV1,
    LearnerProfileSnapshotV1,
    ResourceRecommendationBatchV1,
    ResourceRecommendationEngineRequestV1,
    build_learner_path_provider_policy_fingerprint,
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

    runtime_fingerprint: str
    provider_projection_max_steps: int
    provider_projection_max_chars: int
    load_profile: ProfileSnapshotLoader
    load_history: HistorySnapshotLoader
    plan_learning_path: LearnerPathEngine
    recommend_resources: ResourceRecommendationEngine
    _provider_projection_policy_fingerprint: str = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.runtime_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.runtime_fingerprint
        ):
            raise ValueError(
                "runtime_fingerprint must be a lowercase SHA-256 hex digest"
            )
        if (
            isinstance(self.provider_projection_max_steps, bool)
            or not 1
            <= self.provider_projection_max_steps
            <= LEARNER_PATH_PROVIDER_MAX_STEPS
        ):
            raise ValueError(
                "provider_projection_max_steps must be within the provider contract"
            )
        if (
            isinstance(self.provider_projection_max_chars, bool)
            or not 1
            <= self.provider_projection_max_chars
            <= LEARNER_PATH_PROVIDER_MAX_CHARS
        ):
            raise ValueError(
                "provider_projection_max_chars must be within the provider contract"
            )
        object.__setattr__(
            self,
            "_provider_projection_policy_fingerprint",
            build_learner_path_provider_policy_fingerprint(
                max_steps=self.provider_projection_max_steps,
                max_chars=self.provider_projection_max_chars,
            ),
        )
        dependencies = {
            "load_profile": self.load_profile,
            "load_history": self.load_history,
            "plan_learning_path": self.plan_learning_path,
            "recommend_resources": self.recommend_resources,
        }
        for name, dependency in dependencies.items():
            if not callable(dependency):
                raise TypeError(f"{name} must be callable")

    @property
    def provider_projection_policy_fingerprint(self) -> str:
        return self._provider_projection_policy_fingerprint


__all__ = [
    "HistorySnapshotLoader",
    "LearnerPathEngine",
    "LearningGuidanceContractError",
    "LearningGuidanceRuntime",
    "ProfileSnapshotLoader",
    "ResourceRecommendationEngine",
]
