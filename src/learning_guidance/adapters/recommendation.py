"""Strict four-factor resource recommendation engine for KnowledgeGraphV1."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import hashlib
import json
from typing import Literal, TypeAlias

from src.config.learning_guidance_config import ResourceRecommendationPolicyV1
from src.learning_guidance.contracts import (
    LearnerGoalSignalV1,
    LearnerHistoryEventV1,
    LearnerPreferenceSignalV1,
    LearnerSkillSignalV1,
    RecommendationResourceContextV1,
    RecommendationScoreFactorsV1,
    RecommendationScoreWeightsV1,
    ResourceRecommendationBatchV1,
    ResourceRecommendationEngineRequestV1,
    ResourceRecommendationEngineResultV1,
    ResourceRecommendationItemV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1


RECOMMENDATION_ENGINE_VERSION = "learning_guidance_recommendation_engine_v1"
RecommendationEngineErrorCode: TypeAlias = Literal[
    "recommendation_subject_unknown",
    "recommendation_resource_binding_invalid",
    "recommendation_clock_invalid",
    "recommendation_history_timestamp_invalid",
]


class RecommendationEngineError(RuntimeError):
    """Content-safe typed failure from the strict recommendation algorithm."""

    def __init__(self, *, code: RecommendationEngineErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: resource recommendation engine failed")


def _stable_recommendation_id(
    *,
    request_id: str,
    resource_id: str,
    topic_id: str,
    mode: str,
) -> str:
    encoded = json.dumps(
        {
            "algorithm": RECOMMENDATION_ENGINE_VERSION,
            "request_id": request_id,
            "resource_id": resource_id,
            "topic_id": topic_id,
            "mode": mode,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"recommendation_{hashlib.sha256(encoded).hexdigest()}"


def _aware_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise RecommendationEngineError(code="recommendation_clock_invalid")
    return value


class ResourceRecommendationEngineV1:
    """Score only candidates with complete topic-local four-factor evidence."""

    version = RECOMMENDATION_ENGINE_VERSION

    def __init__(
        self,
        *,
        knowledge_graph: KnowledgeGraphV1,
        policy: ResourceRecommendationPolicyV1,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(knowledge_graph, KnowledgeGraphV1):
            raise TypeError("knowledge_graph must be KnowledgeGraphV1")
        if not isinstance(policy, ResourceRecommendationPolicyV1):
            raise TypeError("policy must be ResourceRecommendationPolicyV1")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._knowledge_graph = knowledge_graph
        self._policy = policy
        self._clock = clock

    def _explicit_candidates(
        self,
        *,
        subject: str,
    ) -> tuple[RecommendationResourceContextV1, ...]:
        subject_node = self._knowledge_graph.subject(subject)
        if subject_node is None:
            raise RecommendationEngineError(code="recommendation_subject_unknown")
        return tuple(
            RecommendationResourceContextV1(
                resource_id=resource.resource_id,
                resource_type=resource.resource_type,
                subject=subject,
                topic_id=topic.topic_id,
                title=resource.title,
            )
            for topic in subject_node.topics
            for resource in topic.resources
        )

    def _validate_candidate(
        self,
        candidate: RecommendationResourceContextV1,
        *,
        subject: str,
    ) -> None:
        subject_node = self._knowledge_graph.subject(subject)
        if subject_node is None:
            raise RecommendationEngineError(code="recommendation_subject_unknown")
        if candidate.subject != subject or not any(
            topic.topic_id == candidate.topic_id for topic in subject_node.topics
        ):
            raise RecommendationEngineError(
                code="recommendation_resource_binding_invalid"
            )

    async def recommend(
        self,
        request: ResourceRecommendationEngineRequestV1,
    ) -> ResourceRecommendationEngineResultV1:
        if not isinstance(request, ResourceRecommendationEngineRequestV1):
            raise TypeError("request must be ResourceRecommendationEngineRequestV1")
        now = _aware_now(self._clock)
        candidates = (
            request.generated_resources
            if request.mode == "automatic_after_generation"
            else self._explicit_candidates(subject=request.subject)
        )
        skills_by_topic: dict[str, LearnerSkillSignalV1] = {
            signal.topic_id: signal
            for signal in request.profile.skills
            if signal.subject == request.subject
        }
        goals_by_topic: dict[str, tuple[LearnerGoalSignalV1, ...]] = {
            topic_id: tuple(
                goal
                for goal in request.profile.goals
                if goal.subject == request.subject and goal.topic_id == topic_id
            )
            for topic_id in skills_by_topic
        }
        preferences_by_slot: dict[tuple[str, str], LearnerPreferenceSignalV1] = {
            (signal.topic_id, signal.dimension): signal
            for signal in request.profile.preferences
            if signal.subject == request.subject
        }
        history_by_topic: dict[str, tuple[LearnerHistoryEventV1, ...]] = {
            topic_id: tuple(
                event for event in request.history.events if event.topic_id == topic_id
            )
            for topic_id in skills_by_topic
        }
        weights = RecommendationScoreWeightsV1(
            weakness=self._policy.weights.weakness,
            forgetting=self._policy.weights.forgetting,
            preference=self._policy.weights.preference,
            goal=self._policy.weights.goal,
        )
        horizon_seconds = self._policy.forgetting_horizon_days * 86_400.0
        scored: list[
            tuple[
                float,
                RecommendationResourceContextV1,
                RecommendationScoreFactorsV1,
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []
        for candidate in candidates:
            self._validate_candidate(candidate, subject=request.subject)
            skill = skills_by_topic.get(candidate.topic_id)
            goals = goals_by_topic.get(candidate.topic_id, ())
            events = history_by_topic.get(candidate.topic_id, ())
            dimension = self._policy.preference_for(candidate.resource_type)
            preference = preferences_by_slot.get((candidate.topic_id, dimension))
            if skill is None or not goals or not events or preference is None:
                continue
            if any(event.observed_at > now for event in events):
                raise RecommendationEngineError(
                    code="recommendation_history_timestamp_invalid"
                )
            latest_observed_at = max(event.observed_at for event in events)
            elapsed_seconds = (now - latest_observed_at).total_seconds()
            weakness = 1.0 - skill.level
            forgetting = min(elapsed_seconds / horizon_seconds, 1.0)
            goal = max(signal.importance * (1.0 - signal.progress) for signal in goals)
            combined = (
                weakness * weights.weakness
                + forgetting * weights.forgetting
                + preference.strength * weights.preference
                + goal * weights.goal
            )
            if combined < self._policy.min_combined_score:
                continue
            factors = RecommendationScoreFactorsV1(
                weakness=weakness,
                forgetting=forgetting,
                preference=preference.strength,
                goal=goal,
                combined=combined,
                weights=weights,
            )
            scored.append(
                (
                    combined,
                    candidate,
                    factors,
                    (
                        skill.signal_id,
                        *(signal.signal_id for signal in goals),
                        preference.signal_id,
                    ),
                    tuple(event.history_id for event in events),
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1].resource_id, item[1].topic_id))
        selected = scored[: self._policy.top_n]
        if not selected:
            return ResourceRecommendationEngineResultV1(
                schema_version="resource_recommendation_engine_result_v1",
                request_id=request.request_id,
                mode=request.mode,
                user_id=request.user_id,
                subject=request.subject,
                status="unavailable",
                unavailable_reason="no_eligible_candidates",
                batch=None,
            )
        items = tuple(
            ResourceRecommendationItemV1(
                recommendation_id=_stable_recommendation_id(
                    request_id=request.request_id,
                    resource_id=candidate.resource_id,
                    topic_id=candidate.topic_id,
                    mode=request.mode,
                ),
                resource_id=candidate.resource_id,
                resource_type=candidate.resource_type,
                subject=candidate.subject,
                topic_id=candidate.topic_id,
                title=candidate.title,
                rank=rank,
                score_factors=factors,
                reason=(
                    "Ranked by exact topic-local weakness, forgetting, preference, "
                    "and goal evidence."
                ),
                profile_signal_ids=profile_signal_ids,
                history_ids=history_ids,
                source_resource_ids=(
                    (candidate.resource_id,)
                    if request.mode == "automatic_after_generation"
                    else ()
                ),
            )
            for rank, (
                _score,
                candidate,
                factors,
                profile_signal_ids,
                history_ids,
            ) in enumerate(selected, start=1)
        )
        batch = ResourceRecommendationBatchV1(
            schema_version="resource_recommendation_batch_v1",
            mode=request.mode,
            user_id=request.user_id,
            subject=request.subject,
            generated_at=now,
            items=items,
            summary=f"Generated {len(items)} strict evidence-bound recommendations.",
        )
        return ResourceRecommendationEngineResultV1(
            schema_version="resource_recommendation_engine_result_v1",
            request_id=request.request_id,
            mode=request.mode,
            user_id=request.user_id,
            subject=request.subject,
            status="available",
            unavailable_reason=None,
            batch=batch,
        )


__all__ = [
    "RECOMMENDATION_ENGINE_VERSION",
    "RecommendationEngineError",
    "ResourceRecommendationEngineV1",
]
