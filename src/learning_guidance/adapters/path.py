"""Deterministic, evidence-bound learner path engine for KnowledgeGraphV1."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import hashlib
import json
from typing import Literal, TypeAlias

from src.config.learning_guidance_config import LearnerPathPolicyV1
from src.learning_guidance.contracts import (
    LearnerHistoryEventV1,
    LearnerPathEngineRequestV1,
    LearnerPathPlanV1,
    LearnerPathStepV1,
    LearnerSkillSignalV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1, KnowledgeTopicV1


PATH_ENGINE_VERSION = "learning_guidance_path_engine_v1"
PathEngineErrorCode: TypeAlias = Literal[
    "path_subject_unknown",
    "path_topic_evidence_unavailable",
    "path_clock_invalid",
    "path_history_timestamp_invalid",
]


class PathEngineError(RuntimeError):
    """Content-safe typed failure from the strict path algorithm."""

    def __init__(self, *, code: PathEngineErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: learner path engine failed")


def _stable_step_id(*, request_id: str, topic_id: str) -> str:
    encoded = json.dumps(
        {
            "algorithm": PATH_ENGINE_VERSION,
            "request_id": request_id,
            "topic_id": topic_id,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"path_step_{hashlib.sha256(encoded).hexdigest()}"


def _aware_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise PathEngineError(code="path_clock_invalid")
    return value


class LearnerPathEngineV1:
    """Plan only topic steps backed by exact profile and history evidence."""

    version = PATH_ENGINE_VERSION

    def __init__(
        self,
        *,
        knowledge_graph: KnowledgeGraphV1,
        policy: LearnerPathPolicyV1,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(knowledge_graph, KnowledgeGraphV1):
            raise TypeError("knowledge_graph must be KnowledgeGraphV1")
        if not isinstance(policy, LearnerPathPolicyV1):
            raise TypeError("policy must be LearnerPathPolicyV1")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._knowledge_graph = knowledge_graph
        self._policy = policy
        self._clock = clock

    def _status(
        self,
        *,
        topic: KnowledgeTopicV1,
        skill: LearnerSkillSignalV1,
        events: tuple[LearnerHistoryEventV1, ...],
        skills_by_topic: dict[str, LearnerSkillSignalV1],
        now: datetime,
    ) -> tuple[
        Literal["ready", "blocked", "reinforce", "repeat", "skip"],
        str,
    ]:
        if any(event.observed_at > now for event in events):
            raise PathEngineError(code="path_history_timestamp_invalid")
        cutoff = now - timedelta(days=self._policy.recent_failure_window_days)
        has_recent_failure = any(
            event.outcome_score is not None
            and event.outcome_score < self._policy.repeat_outcome_threshold
            and event.observed_at >= cutoff
            for event in events
        )
        if has_recent_failure:
            return "repeat", "recent topic-local outcome is below the repeat threshold"
        if (
            skill.level >= self._policy.mastery_level
            and skill.confidence >= self._policy.mastery_confidence
        ):
            return "skip", "mastery level and confidence meet the skip thresholds"
        if skill.level <= self._policy.reinforce_level:
            return "reinforce", "skill level is at or below the reinforce threshold"
        unmet_prerequisites = tuple(
            prerequisite_id
            for prerequisite_id in topic.prerequisite_topic_ids
            if prerequisite_id not in skills_by_topic
            or skills_by_topic[prerequisite_id].level < self._policy.mastery_level
            or skills_by_topic[prerequisite_id].confidence
            < self._policy.mastery_confidence
        )
        if unmet_prerequisites:
            return "blocked", "one or more exact prerequisites are not mastered"
        return "ready", "topic evidence and prerequisite thresholds permit study"

    async def plan(self, request: LearnerPathEngineRequestV1) -> LearnerPathPlanV1:
        if not isinstance(request, LearnerPathEngineRequestV1):
            raise TypeError("request must be LearnerPathEngineRequestV1")
        subject = self._knowledge_graph.subject(request.subject)
        if subject is None:
            raise PathEngineError(code="path_subject_unknown")
        now = _aware_now(self._clock)
        skills_by_topic = {
            signal.topic_id: signal
            for signal in request.profile.skills
            if signal.subject == request.subject
        }
        goals_by_topic = {
            topic.topic_id: tuple(
                goal
                for goal in request.profile.goals
                if goal.subject == request.subject and goal.topic_id == topic.topic_id
            )
            for topic in subject.topics
        }
        history_by_topic = {
            topic.topic_id: tuple(
                event
                for event in request.history.events
                if event.topic_id == topic.topic_id
            )
            for topic in subject.topics
        }
        eligible_topics = tuple(
            topic
            for topic in subject.topics
            if topic.topic_id in skills_by_topic
            and goals_by_topic[topic.topic_id]
            and history_by_topic[topic.topic_id]
        )
        selected_topics = eligible_topics[: self._policy.max_steps]
        if not selected_topics:
            raise PathEngineError(code="path_topic_evidence_unavailable")

        def target_priority(topic: KnowledgeTopicV1) -> tuple[float, float, int]:
            goals = goals_by_topic[topic.topic_id]
            goal_priority = max(
                goal.importance * (1.0 - goal.progress) for goal in goals
            )
            weakness = 1.0 - skills_by_topic[topic.topic_id].level
            position = next(
                index
                for index, candidate in enumerate(selected_topics)
                if candidate.topic_id == topic.topic_id
            )
            return goal_priority, weakness, -position

        target_topic_id = max(selected_topics, key=target_priority).topic_id
        steps: list[LearnerPathStepV1] = []
        for position, topic in enumerate(selected_topics, start=1):
            skill = skills_by_topic[topic.topic_id]
            goals = goals_by_topic[topic.topic_id]
            events = history_by_topic[topic.topic_id]
            status, reason = self._status(
                topic=topic,
                skill=skill,
                events=events,
                skills_by_topic=skills_by_topic,
                now=now,
            )
            steps.append(
                LearnerPathStepV1(
                    step_id=_stable_step_id(
                        request_id=request.request_id,
                        topic_id=topic.topic_id,
                    ),
                    position=position,
                    topic_id=topic.topic_id,
                    subject=request.subject,
                    title=topic.title,
                    status=status,
                    estimated_hours=topic.estimated_hours,
                    reason=reason,
                    recommended_resource_types=(
                        request.requested_resource_types
                        if topic.topic_id == target_topic_id
                        else ()
                    ),
                    profile_signal_ids=(
                        skill.signal_id,
                        *(goal.signal_id for goal in goals),
                    ),
                    history_ids=tuple(event.history_id for event in events),
                )
            )
        return LearnerPathPlanV1(
            schema_version="learner_path_plan_v1",
            user_id=request.user_id,
            subject=request.subject,
            generated_at=now,
            steps=tuple(steps),
            summary=(
                f"Generated {len(steps)} evidence-bound topic steps; "
                "requested resources are bound to one ranked target topic."
            ),
        )


__all__ = [
    "LearnerPathEngineV1",
    "PATH_ENGINE_VERSION",
    "PathEngineError",
]
