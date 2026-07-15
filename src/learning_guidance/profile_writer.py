"""Strict, idempotent profile binding writes for learning guidance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
from src.learning_guidance.adapters.profile import (
    LearningGuidanceProfileBindingV1,
    ProfileGoalBindingV1,
    ProfilePreferenceBindingV1,
    ProfileSkillBindingV1,
    profile_goal_fingerprint,
)
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.profile.schema import (
    ExtractedProfileInfo,
    Goal,
    ProfileUpdateResult,
    UserProfile,
)
from src.profile.storage import SQLiteProfileStore
from src.profile.updater import update_profile
from src.user_identity import UserIdentityError, validate_user_id


PROFILE_BINDING_EXTRA_KEY = "learning_guidance_v1"
PROFILE_WRITE_RECEIPT_EXTRA_KEY = "learning_guidance_profile_write_v1"
PROFILE_WRITE_HASH_PREFIX = "learning-guidance-profile-write:v1:"
PROFILE_WRITE_SOURCE_HASH_PREFIX = "learning-guidance-profile-source:v1:"
PROFILE_SIGNAL_ID_PREFIX = "learning-guidance-profile-signal:v1:"

ProfileWriteSourceKind: TypeAlias = Literal[
    "profile_write_request_v1",
    "onboard_v2",
]

ProfileWriterErrorCode: TypeAlias = Literal[
    "profile_write_identity_invalid",
    "profile_write_topic_invalid",
    "profile_write_profile_mismatch",
    "profile_write_request_conflict",
    "profile_write_reserved_metadata_invalid",
    "profile_write_binding_invalid",
    "profile_write_binding_missing",
    "profile_write_goal_identity_invalid",
]


class ProfileWriterError(RuntimeError):
    """Content-safe typed failure from the strict profile writer."""

    def __init__(self, *, code: ProfileWriterErrorCode) -> None:
        self.code = code
        super().__init__(f"{code}: learning-guidance profile write failed")


class _StrictWriteModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
    )

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def reject_unnormalized_text(cls, value: object) -> object:
        if isinstance(value, str) and (not value.strip() or value != value.strip()):
            raise ValueError("text fields must be normalized and non-blank")
        return value


class ProfileSkillWriteV1(_StrictWriteModel):
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    level: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ProfileGoalWriteV1(_StrictWriteModel):
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    goal: str = Field(min_length=1, max_length=500)
    importance: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    progress: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ProfilePreferenceWriteV1(_StrictWriteModel):
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    dimension: PreferenceDimension
    strength: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class LearningGuidanceProfileWriteRequestV1(_StrictWriteModel):
    """One complete, explicit request to create a profile binding once."""

    schema_version: Literal["learning_guidance_profile_write_request_v1"]
    request_id: str = Field(min_length=1, max_length=160)
    user_id: str = Field(min_length=1, max_length=160)
    skills: list[ProfileSkillWriteV1] = Field(min_length=1, max_length=200)
    goals: list[ProfileGoalWriteV1] = Field(min_length=1, max_length=50)
    preferences: list[ProfilePreferenceWriteV1] = Field(max_length=200)

    @field_validator("user_id")
    @classmethod
    def validate_durable_user_id(cls, value: str) -> str:
        try:
            return validate_user_id(value)
        except UserIdentityError as exc:
            raise ValueError(exc.code) from None

    @model_validator(mode="after")
    def validate_explicit_inventory(self) -> "LearningGuidanceProfileWriteRequestV1":
        skill_slots = tuple((item.subject, item.topic_id) for item in self.skills)
        if len(skill_slots) != len(set(skill_slots)):
            raise ValueError("profile write skills must use unique topic slots")
        goal_slots = tuple(
            (item.subject, item.topic_id, item.goal) for item in self.goals
        )
        if len(goal_slots) != len(set(goal_slots)):
            raise ValueError("profile write goals must use unique logical slots")
        goal_texts = tuple(item.goal for item in self.goals)
        if len(goal_texts) != len(set(goal_texts)):
            raise ValueError("profile write goal text must be globally unique")
        preference_slots = tuple(
            (item.subject, item.topic_id, item.dimension) for item in self.preferences
        )
        if len(preference_slots) != len(set(preference_slots)):
            raise ValueError("profile write preferences must use unique slots")
        skill_topic_slots = frozenset(
            (item.subject, item.topic_id) for item in self.skills
        )
        goal_topic_slots = frozenset(
            (item.subject, item.topic_id) for item in self.goals
        )
        if skill_topic_slots != goal_topic_slots:
            raise ValueError("every bound topic requires explicit skill and goal data")
        if any(
            (item.subject, item.topic_id) not in skill_topic_slots
            for item in self.preferences
        ):
            raise ValueError("preferences must reference a skill-and-goal topic")
        strengths_by_dimension: dict[PreferenceDimension, float] = {}
        for item in self.preferences:
            prior = strengths_by_dimension.setdefault(item.dimension, item.strength)
            if prior != item.strength:
                raise ValueError(
                    "one profile preference dimension must use one exact strength"
                )
        expected_preference_slots = frozenset(
            (subject, topic_id, dimension)
            for subject, topic_id in skill_topic_slots
            for dimension in strengths_by_dimension
        )
        if frozenset(preference_slots) != expected_preference_slots:
            raise ValueError(
                "each selected preference dimension must cover every bound topic"
            )
        return self


@dataclass(frozen=True, slots=True)
class CompiledLearningGuidanceProfileWriteV1:
    """Deeply immutable internal projection of one validated JSON request."""

    schema_version: Literal["learning_guidance_profile_write_request_v1"]
    request_id: str
    user_id: str
    skills: tuple[ProfileSkillWriteV1, ...]
    goals: tuple[ProfileGoalWriteV1, ...]
    preferences: tuple[ProfilePreferenceWriteV1, ...]


class LearningGuidanceProfileWriteSourceV1(_StrictWriteModel):
    """Canonical identity of the complete business command creating a profile."""

    schema_version: Literal["learning_guidance_profile_write_source_v1"]
    source_kind: ProfileWriteSourceKind
    source_hash: str = Field(
        pattern=rf"^{PROFILE_WRITE_SOURCE_HASH_PREFIX}[0-9a-f]{{64}}$"
    )


def _revalidate_profile_write_request(
    request: LearningGuidanceProfileWriteRequestV1,
) -> LearningGuidanceProfileWriteRequestV1:
    if not isinstance(request, LearningGuidanceProfileWriteRequestV1):
        raise TypeError("request must be LearningGuidanceProfileWriteRequestV1")
    return LearningGuidanceProfileWriteRequestV1.model_validate(
        dict(vars(request)),
        strict=True,
    )


def compile_profile_write_request_v1(
    request: LearningGuidanceProfileWriteRequestV1,
) -> CompiledLearningGuidanceProfileWriteV1:
    """Revalidate the public model and explicitly freeze its collections."""

    validated = _revalidate_profile_write_request(request)
    return CompiledLearningGuidanceProfileWriteV1(
        schema_version=validated.schema_version,
        request_id=validated.request_id,
        user_id=validated.user_id,
        skills=tuple(validated.skills),
        goals=tuple(validated.goals),
        preferences=tuple(validated.preferences),
    )


class LearningGuidanceProfileWriteReceiptV1(_StrictWriteModel):
    """Durable idempotency receipt stored beside the reader binding."""

    schema_version: Literal["learning_guidance_profile_write_receipt_v1"]
    request_id: str = Field(min_length=1, max_length=160)
    request_hash: str = Field(pattern=rf"^{PROFILE_WRITE_HASH_PREFIX}[0-9a-f]{{64}}$")
    source_kind: ProfileWriteSourceKind
    source_hash: str = Field(
        pattern=rf"^{PROFILE_WRITE_SOURCE_HASH_PREFIX}[0-9a-f]{{64}}$"
    )


class LearningGuidanceProfileWriteResultV1(_StrictWriteModel):
    schema_version: Literal["learning_guidance_profile_write_result_v1"]
    status: Literal["created", "replayed"]
    request_id: str = Field(min_length=1, max_length=160)
    request_hash: str = Field(pattern=rf"^{PROFILE_WRITE_HASH_PREFIX}[0-9a-f]{{64}}$")
    profile: UserProfile
    binding: LearningGuidanceProfileBindingV1


def _canonical_digest(*, algorithm: str, payload: object) -> str:
    encoded = json.dumps(
        {"algorithm": algorithm, "payload": payload},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_profile_write_request_hash(
    request: LearningGuidanceProfileWriteRequestV1,
) -> str:
    """Bind idempotency to the exact strict write request."""

    validated = _revalidate_profile_write_request(request)
    digest = _canonical_digest(
        algorithm="learning_guidance_profile_write_request_v1",
        payload=validated.model_dump(mode="json"),
    )
    return f"{PROFILE_WRITE_HASH_PREFIX}{digest}"


def build_profile_write_source_v1(
    *,
    source_kind: ProfileWriteSourceKind,
    payload: object,
) -> LearningGuidanceProfileWriteSourceV1:
    """Fingerprint the complete validated command without storing its content."""

    digest = _canonical_digest(
        algorithm=f"learning_guidance_profile_write_source_v1:{source_kind}",
        payload=payload,
    )
    return LearningGuidanceProfileWriteSourceV1(
        schema_version="learning_guidance_profile_write_source_v1",
        source_kind=source_kind,
        source_hash=f"{PROFILE_WRITE_SOURCE_HASH_PREFIX}{digest}",
    )


def profile_write_source_for_request_v1(
    request: LearningGuidanceProfileWriteRequestV1,
) -> LearningGuidanceProfileWriteSourceV1:
    """Build the explicit source identity for a standalone profile write."""

    validated = _revalidate_profile_write_request(request)
    return build_profile_write_source_v1(
        source_kind="profile_write_request_v1",
        payload=validated.model_dump(mode="json"),
    )


def _stable_signal_id(
    *,
    kind: Literal["skill", "goal", "preference"],
    user_id: str,
    subject: str,
    topic_id: str,
    discriminator: str,
) -> str:
    digest = _canonical_digest(
        algorithm="learning_guidance_profile_signal_v1",
        payload={
            "kind": kind,
            "user_id": user_id,
            "subject": subject,
            "topic_id": topic_id,
            "discriminator": discriminator,
        },
    )
    return f"{PROFILE_SIGNAL_ID_PREFIX}{kind}:{digest}"


def _validate_user_id(user_id: str) -> None:
    try:
        validate_user_id(user_id)
    except UserIdentityError:
        raise ProfileWriterError(code="profile_write_identity_invalid")


def _validate_profile_identity(profile: UserProfile, *, user_id: str) -> None:
    if not isinstance(profile, UserProfile) or profile.user_id != user_id:
        raise ProfileWriterError(code="profile_write_identity_invalid")
    _validate_user_id(profile.user_id)


def _apply_onboard_v2_command_fields(
    candidate: UserProfile,
    *,
    base_profile: UserProfile,
) -> None:
    """Apply direct onboarding facts without replacing existing profile history."""

    if not isinstance(candidate.extra, dict) or not isinstance(
        base_profile.extra, dict
    ):
        raise ProfileWriterError(code="profile_write_profile_mismatch")
    if "nickname" not in base_profile.extra or "grade" not in base_profile.extra:
        raise ProfileWriterError(code="profile_write_profile_mismatch")
    nickname = base_profile.extra["nickname"]
    grade = base_profile.extra["grade"]
    if not isinstance(nickname, str) or not isinstance(grade, str):
        raise ProfileWriterError(code="profile_write_profile_mismatch")
    if not isinstance(base_profile.dislikes, list) or any(
        not isinstance(value, str) for value in base_profile.dislikes
    ):
        raise ProfileWriterError(code="profile_write_profile_mismatch")

    candidate.dislikes = list(base_profile.dislikes)
    candidate.extra = dict(candidate.extra)
    candidate.extra["nickname"] = nickname
    candidate.extra["grade"] = grade


def _validate_topic(
    knowledge_graph: KnowledgeGraphV1,
    *,
    subject: str,
    topic_id: str,
) -> None:
    subject_node = knowledge_graph.subject(subject)
    if subject_node is None or all(
        topic.topic_id != topic_id for topic in subject_node.topics
    ):
        raise ProfileWriterError(code="profile_write_topic_invalid")


def _goals_by_text_and_fingerprint(
    profile: UserProfile,
) -> tuple[dict[str, Goal], dict[str, Goal]]:
    by_text: dict[str, Goal] = {}
    by_fingerprint: dict[str, Goal] = {}
    for goal in profile.goals:
        if goal.goal in by_text:
            raise ProfileWriterError(code="profile_write_goal_identity_invalid")
        fingerprint = profile_goal_fingerprint(goal)
        if fingerprint in by_fingerprint:
            raise ProfileWriterError(code="profile_write_goal_identity_invalid")
        by_text[goal.goal] = goal
        by_fingerprint[fingerprint] = goal
    return by_text, by_fingerprint


def _build_binding(
    *,
    profile: UserProfile,
    request: CompiledLearningGuidanceProfileWriteV1,
    knowledge_graph: KnowledgeGraphV1,
) -> LearningGuidanceProfileBindingV1:
    _validate_profile_identity(profile, user_id=request.user_id)
    goals_by_text, _goals_by_fingerprint = _goals_by_text_and_fingerprint(profile)

    skills: list[ProfileSkillBindingV1] = []
    for skill_item in request.skills:
        _validate_topic(
            knowledge_graph,
            subject=skill_item.subject,
            topic_id=skill_item.topic_id,
        )
        stored_skill = profile.skills.get(skill_item.topic_id)
        if (
            stored_skill is None
            or stored_skill.level != skill_item.level
            or stored_skill.confidence != skill_item.confidence
        ):
            raise ProfileWriterError(code="profile_write_profile_mismatch")
        skills.append(
            ProfileSkillBindingV1(
                signal_id=_stable_signal_id(
                    kind="skill",
                    user_id=request.user_id,
                    subject=skill_item.subject,
                    topic_id=skill_item.topic_id,
                    discriminator="skill",
                ),
                subject=skill_item.subject,
                topic_id=skill_item.topic_id,
            )
        )

    goals: list[ProfileGoalBindingV1] = []
    for goal_item in request.goals:
        _validate_topic(
            knowledge_graph,
            subject=goal_item.subject,
            topic_id=goal_item.topic_id,
        )
        stored_goal = goals_by_text.get(goal_item.goal)
        if (
            stored_goal is None
            or stored_goal.importance != goal_item.importance
            or stored_goal.progress != goal_item.progress
        ):
            raise ProfileWriterError(code="profile_write_profile_mismatch")
        goals.append(
            ProfileGoalBindingV1(
                signal_id=_stable_signal_id(
                    kind="goal",
                    user_id=request.user_id,
                    subject=goal_item.subject,
                    topic_id=goal_item.topic_id,
                    discriminator=goal_item.goal,
                ),
                subject=goal_item.subject,
                topic_id=goal_item.topic_id,
                goal_fingerprint=profile_goal_fingerprint(stored_goal),
            )
        )

    preferences: list[ProfilePreferenceBindingV1] = []
    for preference_item in request.preferences:
        _validate_topic(
            knowledge_graph,
            subject=preference_item.subject,
            topic_id=preference_item.topic_id,
        )
        if (
            getattr(profile.learning_style, preference_item.dimension)
            != preference_item.strength
        ):
            raise ProfileWriterError(code="profile_write_profile_mismatch")
        preferences.append(
            ProfilePreferenceBindingV1(
                signal_id=_stable_signal_id(
                    kind="preference",
                    user_id=request.user_id,
                    subject=preference_item.subject,
                    topic_id=preference_item.topic_id,
                    discriminator=preference_item.dimension,
                ),
                subject=preference_item.subject,
                topic_id=preference_item.topic_id,
                dimension=preference_item.dimension,
                strength=preference_item.strength,
            )
        )

    try:
        return LearningGuidanceProfileBindingV1(
            schema_version="learning_guidance_profile_v1",
            skills=skills,
            goals=goals,
            preferences=preferences,
        )
    except ValidationError:
        raise ProfileWriterError(code="profile_write_binding_invalid") from None


def _binding_structure(
    binding: LearningGuidanceProfileBindingV1,
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        [
            ("skill", item.signal_id, item.subject, item.topic_id)
            for item in binding.skills
        ]
        + [
            ("goal", item.signal_id, item.subject, item.topic_id)
            for item in binding.goals
        ]
        + [
            ("preference", item.signal_id, item.subject, item.topic_id)
            for item in binding.preferences
        ]
    )


def _request_binding_structure(
    request: CompiledLearningGuidanceProfileWriteV1,
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        [
            (
                "skill",
                _stable_signal_id(
                    kind="skill",
                    user_id=request.user_id,
                    subject=item.subject,
                    topic_id=item.topic_id,
                    discriminator="skill",
                ),
                item.subject,
                item.topic_id,
            )
            for item in request.skills
        ]
        + [
            (
                "goal",
                _stable_signal_id(
                    kind="goal",
                    user_id=request.user_id,
                    subject=item.subject,
                    topic_id=item.topic_id,
                    discriminator=item.goal,
                ),
                item.subject,
                item.topic_id,
            )
            for item in request.goals
        ]
        + [
            (
                "preference",
                _stable_signal_id(
                    kind="preference",
                    user_id=request.user_id,
                    subject=item.subject,
                    topic_id=item.topic_id,
                    discriminator=item.dimension,
                ),
                item.subject,
                item.topic_id,
            )
            for item in request.preferences
        ]
    )


def _validate_request_topics(
    request: CompiledLearningGuidanceProfileWriteV1,
    *,
    knowledge_graph: KnowledgeGraphV1,
) -> None:
    topic_slots = (
        tuple((item.subject, item.topic_id) for item in request.skills)
        + tuple((item.subject, item.topic_id) for item in request.goals)
        + tuple((item.subject, item.topic_id) for item in request.preferences)
    )
    for subject, topic_id in topic_slots:
        _validate_topic(
            knowledge_graph,
            subject=subject,
            topic_id=topic_id,
        )


def _parse_reserved_metadata(
    profile: UserProfile,
) -> (
    tuple[
        LearningGuidanceProfileWriteReceiptV1,
        LearningGuidanceProfileBindingV1,
    ]
    | None
):
    has_binding = PROFILE_BINDING_EXTRA_KEY in profile.extra
    has_receipt = PROFILE_WRITE_RECEIPT_EXTRA_KEY in profile.extra
    if not has_binding and not has_receipt:
        return None
    if has_binding != has_receipt:
        raise ProfileWriterError(code="profile_write_reserved_metadata_invalid")
    try:
        receipt = LearningGuidanceProfileWriteReceiptV1.model_validate(
            profile.extra[PROFILE_WRITE_RECEIPT_EXTRA_KEY]
        )
        binding = LearningGuidanceProfileBindingV1.model_validate(
            profile.extra[PROFILE_BINDING_EXTRA_KEY]
        )
    except ValidationError:
        raise ProfileWriterError(
            code="profile_write_reserved_metadata_invalid"
        ) from None
    return receipt, binding


def _refresh_binding(
    *,
    before_profile: UserProfile,
    updated_profile: UserProfile,
    binding: LearningGuidanceProfileBindingV1,
    knowledge_graph: KnowledgeGraphV1,
) -> LearningGuidanceProfileBindingV1:
    _validate_profile_identity(updated_profile, user_id=before_profile.user_id)
    _before_goals_by_text, before_goals_by_fingerprint = _goals_by_text_and_fingerprint(
        before_profile
    )
    updated_goals_by_text, _ = _goals_by_text_and_fingerprint(updated_profile)

    skills: list[ProfileSkillBindingV1] = []
    for skill_binding in binding.skills:
        _validate_topic(
            knowledge_graph,
            subject=skill_binding.subject,
            topic_id=skill_binding.topic_id,
        )
        if (
            before_profile.skills.get(skill_binding.topic_id) is None
            or updated_profile.skills.get(skill_binding.topic_id) is None
            or skill_binding.signal_id
            != _stable_signal_id(
                kind="skill",
                user_id=before_profile.user_id,
                subject=skill_binding.subject,
                topic_id=skill_binding.topic_id,
                discriminator="skill",
            )
        ):
            raise ProfileWriterError(code="profile_write_binding_invalid")
        skills.append(skill_binding)

    goals: list[ProfileGoalBindingV1] = []
    for goal_binding in binding.goals:
        _validate_topic(
            knowledge_graph,
            subject=goal_binding.subject,
            topic_id=goal_binding.topic_id,
        )
        before_goal = before_goals_by_fingerprint.get(goal_binding.goal_fingerprint)
        if before_goal is None:
            raise ProfileWriterError(code="profile_write_binding_invalid")
        updated_goal = updated_goals_by_text.get(before_goal.goal)
        if updated_goal is None or goal_binding.signal_id != _stable_signal_id(
            kind="goal",
            user_id=before_profile.user_id,
            subject=goal_binding.subject,
            topic_id=goal_binding.topic_id,
            discriminator=before_goal.goal,
        ):
            raise ProfileWriterError(code="profile_write_binding_invalid")
        goals.append(
            ProfileGoalBindingV1(
                signal_id=goal_binding.signal_id,
                subject=goal_binding.subject,
                topic_id=goal_binding.topic_id,
                goal_fingerprint=profile_goal_fingerprint(updated_goal),
            )
        )

    preferences: list[ProfilePreferenceBindingV1] = []
    for preference_binding in binding.preferences:
        _validate_topic(
            knowledge_graph,
            subject=preference_binding.subject,
            topic_id=preference_binding.topic_id,
        )
        if (
            getattr(before_profile.learning_style, preference_binding.dimension)
            != preference_binding.strength
            or preference_binding.signal_id
            != _stable_signal_id(
                kind="preference",
                user_id=before_profile.user_id,
                subject=preference_binding.subject,
                topic_id=preference_binding.topic_id,
                discriminator=preference_binding.dimension,
            )
        ):
            raise ProfileWriterError(code="profile_write_binding_invalid")
        preferences.append(
            ProfilePreferenceBindingV1(
                signal_id=preference_binding.signal_id,
                subject=preference_binding.subject,
                topic_id=preference_binding.topic_id,
                dimension=preference_binding.dimension,
                strength=getattr(
                    updated_profile.learning_style,
                    preference_binding.dimension,
                ),
            )
        )

    try:
        return LearningGuidanceProfileBindingV1(
            schema_version="learning_guidance_profile_v1",
            skills=skills,
            goals=goals,
            preferences=preferences,
        )
    except ValidationError:
        raise ProfileWriterError(code="profile_write_binding_invalid") from None


class LearningGuidanceProfileWriterV1:
    """Atomic create-once binding writer and existing-binding evolution path."""

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

    async def create_once(
        self,
        request: LearningGuidanceProfileWriteRequestV1,
        *,
        base_profile: UserProfile,
        source: LearningGuidanceProfileWriteSourceV1,
    ) -> LearningGuidanceProfileWriteResultV1:
        if not isinstance(request, LearningGuidanceProfileWriteRequestV1):
            raise TypeError("request must be LearningGuidanceProfileWriteRequestV1")
        if not isinstance(base_profile, UserProfile):
            raise TypeError("base_profile must be UserProfile")
        if not isinstance(source, LearningGuidanceProfileWriteSourceV1):
            raise TypeError("source must be LearningGuidanceProfileWriteSourceV1")
        validated_source = LearningGuidanceProfileWriteSourceV1.model_validate(
            dict(vars(source)),
            strict=True,
        )
        validated_request = _revalidate_profile_write_request(request)
        compiled_request = compile_profile_write_request_v1(validated_request)
        _validate_profile_identity(base_profile, user_id=compiled_request.user_id)
        _validate_request_topics(
            compiled_request,
            knowledge_graph=self._knowledge_graph,
        )
        expected_structure = _request_binding_structure(compiled_request)
        request_hash = stable_profile_write_request_hash(validated_request)
        receipt = LearningGuidanceProfileWriteReceiptV1(
            schema_version="learning_guidance_profile_write_receipt_v1",
            request_id=compiled_request.request_id,
            request_hash=request_hash,
            source_kind=validated_source.source_kind,
            source_hash=validated_source.source_hash,
        )
        status: Literal["created", "replayed"] = "created"

        def mutation(current: UserProfile | None) -> UserProfile:
            nonlocal status
            if current is None:
                candidate = base_profile.model_copy(deep=True)
                binding = _build_binding(
                    profile=candidate,
                    request=compiled_request,
                    knowledge_graph=self._knowledge_graph,
                )
            else:
                existing = _parse_reserved_metadata(current)
                if existing is not None:
                    stored_receipt, stored_binding = existing
                    if (
                        stored_receipt.request_id != compiled_request.request_id
                        or stored_receipt.request_hash != request_hash
                        or stored_receipt.source_kind != validated_source.source_kind
                        or stored_receipt.source_hash != validated_source.source_hash
                    ):
                        raise ProfileWriterError(code="profile_write_request_conflict")
                    validated_binding = _refresh_binding(
                        before_profile=current,
                        updated_profile=current,
                        binding=stored_binding,
                        knowledge_graph=self._knowledge_graph,
                    )
                    if _binding_structure(validated_binding) != expected_structure:
                        raise ProfileWriterError(code="profile_write_request_conflict")
                    status = "replayed"
                    return current
                candidate = current.model_copy(deep=True)
                binding = _build_binding(
                    profile=candidate,
                    request=compiled_request,
                    knowledge_graph=self._knowledge_graph,
                )
                if validated_source.source_kind == "onboard_v2":
                    _apply_onboard_v2_command_fields(
                        candidate,
                        base_profile=base_profile,
                    )

            candidate.extra = dict(candidate.extra)
            candidate.extra[PROFILE_BINDING_EXTRA_KEY] = binding.model_dump(mode="json")
            candidate.extra[PROFILE_WRITE_RECEIPT_EXTRA_KEY] = receipt.model_dump(
                mode="json"
            )
            return candidate

        persisted = await self._store.mutate_strict(
            compiled_request.user_id,
            mutation,
        )
        reserved = _parse_reserved_metadata(persisted)
        if reserved is None:
            raise ProfileWriterError(code="profile_write_binding_missing")
        _persisted_receipt, persisted_binding = reserved
        return LearningGuidanceProfileWriteResultV1(
            schema_version="learning_guidance_profile_write_result_v1",
            status=status,
            request_id=compiled_request.request_id,
            request_hash=request_hash,
            profile=persisted,
            binding=persisted_binding,
        )

    async def evolve_existing_binding(
        self,
        *,
        user_id: str,
        extracted: ExtractedProfileInfo,
    ) -> ProfileUpdateResult:
        """Update a profile and refresh only its already-declared binding slots."""

        _validate_user_id(user_id)
        if not isinstance(extracted, ExtractedProfileInfo):
            raise TypeError("extracted must be ExtractedProfileInfo")
        captured_result: ProfileUpdateResult | None = None

        def mutation(current: UserProfile | None) -> UserProfile:
            nonlocal captured_result
            if current is None:
                raise ProfileWriterError(code="profile_write_binding_missing")
            reserved = _parse_reserved_metadata(current)
            if reserved is None:
                raise ProfileWriterError(code="profile_write_binding_missing")
            receipt, binding = reserved
            before_profile = current.model_copy(deep=True)
            captured_result = update_profile(current, extracted)
            refreshed = _refresh_binding(
                before_profile=before_profile,
                updated_profile=captured_result.profile,
                binding=binding,
                knowledge_graph=self._knowledge_graph,
            )
            captured_result.profile.extra = dict(captured_result.profile.extra)
            captured_result.profile.extra[PROFILE_BINDING_EXTRA_KEY] = (
                refreshed.model_dump(mode="json")
            )
            captured_result.profile.extra[PROFILE_WRITE_RECEIPT_EXTRA_KEY] = (
                receipt.model_dump(mode="json")
            )
            return captured_result.profile

        persisted = await self._store.mutate_strict(user_id, mutation)
        if captured_result is None:
            raise AssertionError("strict profile evolution did not run its mutation")
        return ProfileUpdateResult(
            profile=persisted,
            changes=captured_result.changes,
            new_observations=captured_result.new_observations,
        )


__all__ = [
    "LearningGuidanceProfileWriteReceiptV1",
    "LearningGuidanceProfileWriteSourceV1",
    "CompiledLearningGuidanceProfileWriteV1",
    "LearningGuidanceProfileWriteRequestV1",
    "LearningGuidanceProfileWriteResultV1",
    "LearningGuidanceProfileWriterV1",
    "PROFILE_BINDING_EXTRA_KEY",
    "PROFILE_SIGNAL_ID_PREFIX",
    "PROFILE_WRITE_HASH_PREFIX",
    "PROFILE_WRITE_SOURCE_HASH_PREFIX",
    "PROFILE_WRITE_RECEIPT_EXTRA_KEY",
    "ProfileGoalWriteV1",
    "ProfilePreferenceWriteV1",
    "ProfileSkillWriteV1",
    "ProfileWriterError",
    "build_profile_write_source_v1",
    "compile_profile_write_request_v1",
    "profile_write_source_for_request_v1",
    "stable_profile_write_request_hash",
]
