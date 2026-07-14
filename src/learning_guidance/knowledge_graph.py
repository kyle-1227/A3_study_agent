"""Strict curated knowledge-graph contract for learning guidance.

The V1 artifact is intentionally independent from the legacy curriculum graph.
It accepts one explicit YAML path, performs no identity repair, and fails closed
on schema, inventory, or topology drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from src.resource_contracts import ResourceType


def _normalized_text(value: str) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError("text must be normalized and non-blank")
    return value


def _normalized_identifier(value: str) -> str:
    _normalized_text(value)
    if not value[0].isalnum() or not value[-1].isalnum():
        raise ValueError("identifier must start and end with an alphanumeric")
    if any(
        not (character.isascii() and (character.isalnum() or character in "._:-"))
        for character in value
    ):
        raise ValueError(
            "identifier may contain only ASCII alphanumerics, dot, underscore, "
            "colon, and hyphen"
        )
    if value != value.lower():
        raise ValueError("identifier must be lowercase")
    return value


def _freeze_sequence(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    raise TypeError("value must be represented by a YAML sequence")


NormalizedText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=500),
    AfterValidator(_normalized_text),
]
NormalizedIdentifier: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=200),
    AfterValidator(_normalized_identifier),
]
IdentifierTuple: TypeAlias = Annotated[
    tuple[NormalizedIdentifier, ...],
    BeforeValidator(_freeze_sequence),
]
KnowledgePointTuple: TypeAlias = Annotated[
    tuple[NormalizedText, ...],
    BeforeValidator(_freeze_sequence),
    Field(min_length=1, max_length=200),
]


class _StrictKnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CatalogResourceV1(_StrictKnowledgeModel):
    resource_id: NormalizedIdentifier
    resource_type: ResourceType
    title: NormalizedText


CatalogResourceTuple: TypeAlias = Annotated[
    tuple[CatalogResourceV1, ...],
    BeforeValidator(_freeze_sequence),
    Field(min_length=1, max_length=200),
]


class KnowledgeTopicV1(_StrictKnowledgeModel):
    topic_id: NormalizedIdentifier
    title: NormalizedText
    difficulty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    estimated_hours: float = Field(gt=0.0, allow_inf_nan=False)
    prerequisite_topic_ids: IdentifierTuple
    knowledge_points: KnowledgePointTuple
    resources: CatalogResourceTuple

    @field_validator("prerequisite_topic_ids")
    @classmethod
    def validate_prerequisite_inventory(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("prerequisite_topic_ids must be unique")
        return values

    @field_validator("knowledge_points")
    @classmethod
    def validate_knowledge_points(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("knowledge_points must be unique")
        return values


KnowledgeTopicTuple: TypeAlias = Annotated[
    tuple[KnowledgeTopicV1, ...],
    BeforeValidator(_freeze_sequence),
    Field(min_length=1, max_length=2_000),
]


class KnowledgeSubjectV1(_StrictKnowledgeModel):
    subject_id: NormalizedIdentifier
    title: NormalizedText
    topics: KnowledgeTopicTuple


KnowledgeSubjectTuple: TypeAlias = Annotated[
    tuple[KnowledgeSubjectV1, ...],
    BeforeValidator(_freeze_sequence),
    Field(min_length=1, max_length=200),
]


class KnowledgeGraphV1(_StrictKnowledgeModel):
    schema_version: Literal["knowledge_graph_v1"]
    data_version: NormalizedText
    subjects: KnowledgeSubjectTuple

    @model_validator(mode="after")
    def validate_graph_inventory_and_topology(self) -> "KnowledgeGraphV1":
        subject_ids = tuple(subject.subject_id for subject in self.subjects)
        if len(subject_ids) != len(set(subject_ids)):
            raise PydanticCustomError(
                "knowledge_graph_duplicate_subject_id",
                "subject_id values must be globally unique",
            )

        topic_subject: dict[str, str] = {}
        topics: dict[str, KnowledgeTopicV1] = {}
        resource_ids: set[str] = set()
        for subject in self.subjects:
            for topic in subject.topics:
                if topic.topic_id in topics:
                    raise PydanticCustomError(
                        "knowledge_graph_duplicate_topic_id",
                        "topic_id values must be globally unique",
                    )
                topics[topic.topic_id] = topic
                topic_subject[topic.topic_id] = subject.subject_id
                for resource in topic.resources:
                    if resource.resource_id in resource_ids:
                        raise PydanticCustomError(
                            "knowledge_graph_duplicate_resource_id",
                            "resource_id values must be globally unique",
                        )
                    resource_ids.add(resource.resource_id)

        for topic_id, topic in topics.items():
            for prerequisite_id in topic.prerequisite_topic_ids:
                if prerequisite_id == topic_id:
                    raise PydanticCustomError(
                        "knowledge_graph_self_prerequisite",
                        "a topic cannot require itself",
                    )
                prerequisite_subject = topic_subject.get(prerequisite_id)
                if prerequisite_subject is None:
                    raise PydanticCustomError(
                        "knowledge_graph_unknown_prerequisite",
                        "every prerequisite_topic_id must reference a known topic",
                    )
                if prerequisite_subject != topic_subject[topic_id]:
                    raise PydanticCustomError(
                        "knowledge_graph_cross_subject_prerequisite",
                        "KnowledgeGraphV1 forbids cross-subject prerequisites",
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(topic_id: str) -> None:
            if topic_id in visiting:
                raise PydanticCustomError(
                    "knowledge_graph_cycle",
                    "topic prerequisites must form a directed acyclic graph",
                )
            if topic_id in visited:
                return
            visiting.add(topic_id)
            for prerequisite_id in topics[topic_id].prerequisite_topic_ids:
                visit(prerequisite_id)
            visiting.remove(topic_id)
            visited.add(topic_id)

        for topic_id in topics:
            visit(topic_id)

        for subject in self.subjects:
            position_by_topic = {
                topic.topic_id: position
                for position, topic in enumerate(subject.topics)
            }
            for topic in subject.topics:
                if any(
                    position_by_topic[prerequisite_id]
                    >= position_by_topic[topic.topic_id]
                    for prerequisite_id in topic.prerequisite_topic_ids
                ):
                    raise PydanticCustomError(
                        "knowledge_graph_topological_order",
                        "topics must be stored in prerequisite-first order",
                    )
        return self

    @property
    def artifact_fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def subject(self, subject_id: str) -> KnowledgeSubjectV1 | None:
        return next(
            (subject for subject in self.subjects if subject.subject_id == subject_id),
            None,
        )

    def topic(self, topic_id: str) -> KnowledgeTopicV1 | None:
        for subject in self.subjects:
            for topic in subject.topics:
                if topic.topic_id == topic_id:
                    return topic
        return None


class KnowledgeGraphLoadError(RuntimeError):
    """Base typed error for one explicit curated artifact."""

    def __init__(self, *, artifact_path: Path, reason: str) -> None:
        self.artifact_path = artifact_path
        self.reason = reason
        super().__init__(f"knowledge graph error at {artifact_path}: {reason}")


class KnowledgeGraphPathError(KnowledgeGraphLoadError):
    """The explicit artifact path cannot be read."""


class KnowledgeGraphYamlError(KnowledgeGraphLoadError):
    """The artifact is not valid YAML."""


class KnowledgeGraphYamlRootError(KnowledgeGraphLoadError):
    """The YAML document root is not a mapping."""


class KnowledgeGraphValidationError(KnowledgeGraphLoadError):
    """The strict schema or business validator rejected the artifact."""

    def __init__(
        self,
        *,
        artifact_path: Path,
        validation_errors: tuple[tuple[str, str], ...],
    ) -> None:
        self.validation_errors = validation_errors
        summary = "; ".join(
            f"{location}: {error_type}" for location, error_type in validation_errors
        )
        super().__init__(artifact_path=artifact_path, reason=summary)


def load_knowledge_graph(artifact_path: Path) -> KnowledgeGraphV1:
    """Load and strictly validate one explicitly supplied curated artifact."""

    if not isinstance(artifact_path, Path):
        raise KnowledgeGraphPathError(
            artifact_path=Path("<invalid-path-type>"),
            reason="artifact_path must be a pathlib.Path instance",
        )
    try:
        text = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KnowledgeGraphPathError(
            artifact_path=artifact_path,
            reason=f"{type(exc).__name__} while reading the file",
        ) from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise KnowledgeGraphYamlError(
            artifact_path=artifact_path,
            reason="invalid YAML syntax",
        ) from exc
    if not isinstance(payload, dict):
        raise KnowledgeGraphYamlRootError(
            artifact_path=artifact_path,
            reason="YAML document root must be a mapping",
        )
    try:
        return KnowledgeGraphV1.model_validate(payload)
    except ValidationError as exc:
        details = tuple(
            (
                ".".join(str(part) for part in error["loc"]) or "root",
                str(error["type"]),
            )
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise KnowledgeGraphValidationError(
            artifact_path=artifact_path,
            validation_errors=details,
        ) from exc


__all__ = [
    "CatalogResourceV1",
    "KnowledgeGraphLoadError",
    "KnowledgeGraphPathError",
    "KnowledgeGraphV1",
    "KnowledgeGraphValidationError",
    "KnowledgeGraphYamlError",
    "KnowledgeGraphYamlRootError",
    "KnowledgeSubjectV1",
    "KnowledgeTopicV1",
    "load_knowledge_graph",
]
