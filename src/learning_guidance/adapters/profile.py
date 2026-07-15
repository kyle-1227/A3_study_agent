"""Strict projection from persisted UserProfile records into guidance signals."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from src.config.learning_guidance_config import PreferenceDimension
from src.learning_guidance.contracts import (
    LearnerGoalSignalV1,
    LearnerPreferenceSignalV1,
    LearnerProfileSnapshotV1,
    LearnerSkillSignalV1,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.profile.schema import Goal
from src.profile.storage import SQLiteProfileStore


PROFILE_ADAPTER_VERSION = "learning_guidance_profile_adapter_v1"
ProfileAdapterErrorCode: TypeAlias = Literal[
    "profile_binding_schema_invalid",
    "profile_binding_identity_invalid",
    "profile_binding_topic_invalid",
    "profile_skill_binding_invalid",
    "profile_goal_binding_invalid",
    "profile_preference_binding_invalid",
    "profile_snapshot_invalid",
]


class ProfileAdapterError(RuntimeError):
    """Content-safe typed failure for a declared V1 profile binding."""

    def __init__(self, *, code: ProfileAdapterErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: learning-guidance profile binding failed")


class _StrictBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def reject_unnormalized_text(cls, value: object) -> object:
        if isinstance(value, str) and (not value.strip() or value != value.strip()):
            raise ValueError("text fields must be normalized and non-blank")
        return value


class ProfileSkillBindingV1(_StrictBinding):
    signal_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)


class ProfileGoalBindingV1(_StrictBinding):
    signal_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    goal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProfilePreferenceBindingV1(_StrictBinding):
    signal_id: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    dimension: PreferenceDimension
    strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class LearningGuidanceProfileBindingV1(_StrictBinding):
    """JSON-native binding stored only under extra.learning_guidance_v1."""

    schema_version: Literal["learning_guidance_profile_v1"]
    skills: list[ProfileSkillBindingV1] = Field(min_length=1, max_length=200)
    goals: list[ProfileGoalBindingV1] = Field(min_length=1, max_length=50)
    preferences: list[ProfilePreferenceBindingV1] = Field(max_length=200)

    @model_validator(mode="after")
    def validate_binding_inventory(self) -> "LearningGuidanceProfileBindingV1":
        signal_ids = tuple(
            binding.signal_id
            for binding in (*self.skills, *self.goals, *self.preferences)
        )
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("profile binding signal ids must be unique")
        skill_slots = tuple(
            (binding.subject, binding.topic_id) for binding in self.skills
        )
        if len(skill_slots) != len(set(skill_slots)):
            raise ValueError("profile skill bindings must use unique topic slots")
        goal_fingerprints = tuple(binding.goal_fingerprint for binding in self.goals)
        if len(goal_fingerprints) != len(set(goal_fingerprints)):
            raise ValueError("profile goal bindings must use unique goal fingerprints")
        preference_slots = tuple(
            (binding.subject, binding.topic_id, binding.dimension)
            for binding in self.preferences
        )
        if len(preference_slots) != len(set(preference_slots)):
            raise ValueError("profile preference bindings must use unique slots")
        return self


def _topic_is_bound(
    knowledge_graph: KnowledgeGraphV1,
    *,
    subject: str,
    topic_id: str,
) -> bool:
    subject_node = knowledge_graph.subject(subject)
    return subject_node is not None and any(
        topic.topic_id == topic_id for topic in subject_node.topics
    )


def profile_goal_fingerprint(goal: Goal) -> str:
    """Fingerprint one exact persisted goal without relying on list position."""

    if not isinstance(goal, Goal):
        raise TypeError("goal must be Goal")
    encoded = json.dumps(
        {
            "algorithm": "learning_guidance_profile_goal_v1",
            "goal": goal.model_dump(mode="json"),
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProfileSnapshotAdapterV1:
    """Real strict adapter over SQLiteProfileStore.load_strict()."""

    version = PROFILE_ADAPTER_VERSION

    def __init__(
        self,
        *,
        store: SQLiteProfileStore,
        knowledge_graph: KnowledgeGraphV1,
    ) -> None:
        if not isinstance(store, SQLiteProfileStore):
            raise TypeError("store must be SQLiteProfileStore")
        if not isinstance(knowledge_graph, KnowledgeGraphV1):
            raise TypeError("knowledge_graph must be KnowledgeGraphV1")
        self._store = store
        self._knowledge_graph = knowledge_graph

    async def load(self, user_id: str) -> LearnerProfileSnapshotV1 | None:
        profile = await self._store.load_strict(user_id)
        if profile is None:
            return None
        if profile.user_id != user_id:
            raise ProfileAdapterError(code="profile_binding_identity_invalid")
        if "learning_guidance_v1" not in profile.extra:
            return None
        raw_binding = profile.extra["learning_guidance_v1"]
        try:
            binding = LearningGuidanceProfileBindingV1.model_validate(raw_binding)
        except ValidationError:
            raise ProfileAdapterError(code="profile_binding_schema_invalid") from None

        skills: list[LearnerSkillSignalV1] = []
        for item in binding.skills:
            if not _topic_is_bound(
                self._knowledge_graph,
                subject=item.subject,
                topic_id=item.topic_id,
            ):
                raise ProfileAdapterError(code="profile_binding_topic_invalid")
            stored = profile.skills.get(item.topic_id)
            if stored is None:
                raise ProfileAdapterError(code="profile_skill_binding_invalid")
            skills.append(
                LearnerSkillSignalV1(
                    signal_id=item.signal_id,
                    subject=item.subject,
                    topic_id=item.topic_id,
                    level=stored.level,
                    confidence=stored.confidence,
                )
            )

        goals_by_fingerprint: dict[str, Goal] = {}
        for stored_goal in profile.goals:
            fingerprint = profile_goal_fingerprint(stored_goal)
            if fingerprint in goals_by_fingerprint:
                raise ProfileAdapterError(code="profile_goal_binding_invalid")
            goals_by_fingerprint[fingerprint] = stored_goal

        goals: list[LearnerGoalSignalV1] = []
        for item in binding.goals:
            if not _topic_is_bound(
                self._knowledge_graph,
                subject=item.subject,
                topic_id=item.topic_id,
            ):
                raise ProfileAdapterError(code="profile_binding_topic_invalid")
            stored = goals_by_fingerprint.get(item.goal_fingerprint)
            if stored is None:
                raise ProfileAdapterError(code="profile_goal_binding_invalid")
            try:
                goals.append(
                    LearnerGoalSignalV1(
                        signal_id=item.signal_id,
                        subject=item.subject,
                        topic_id=item.topic_id,
                        goal=stored.goal,
                        importance=stored.importance,
                        progress=stored.progress,
                    )
                )
            except ValidationError:
                raise ProfileAdapterError(code="profile_goal_binding_invalid") from None

        preferences: list[LearnerPreferenceSignalV1] = []
        for item in binding.preferences:
            if not _topic_is_bound(
                self._knowledge_graph,
                subject=item.subject,
                topic_id=item.topic_id,
            ):
                raise ProfileAdapterError(code="profile_binding_topic_invalid")
            stored_strength = getattr(profile.learning_style, item.dimension)
            if stored_strength != item.strength:
                raise ProfileAdapterError(code="profile_preference_binding_invalid")
            preferences.append(
                LearnerPreferenceSignalV1(
                    signal_id=item.signal_id,
                    subject=item.subject,
                    topic_id=item.topic_id,
                    dimension=item.dimension,
                    strength=item.strength,
                )
            )

        try:
            return LearnerProfileSnapshotV1(
                schema_version="learner_profile_snapshot_v1",
                user_id=profile.user_id,
                skills=tuple(skills),
                goals=tuple(goals),
                preferences=tuple(preferences),
            )
        except ValidationError:
            raise ProfileAdapterError(code="profile_snapshot_invalid") from None


__all__ = [
    "LearningGuidanceProfileBindingV1",
    "PROFILE_ADAPTER_VERSION",
    "ProfileAdapterError",
    "ProfileGoalBindingV1",
    "ProfilePreferenceBindingV1",
    "ProfileSkillBindingV1",
    "ProfileSnapshotAdapterV1",
    "profile_goal_fingerprint",
]
