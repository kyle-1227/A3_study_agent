"""Strict public terminal for explicit resource recommendations.

``RecommendationFinalV1`` is deliberately separate from Resource Final V3.
Recommendation-only success must not weaken the Resource Final invariant that a
successful resource terminal contains at least one generated resource.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from typing import Literal, TypeAlias, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.learning_guidance.contracts import ResourceRecommendationOutputV1
from src.learning_guidance.knowledge_graph import KnowledgeGraphV1
from src.resource_contracts import ResourceType


RECOMMENDATION_FINAL_V1_SCHEMA_VERSION: Literal["recommendation_final_v1"] = (
    "recommendation_final_v1"
)
RECOMMENDATION_FINAL_V1_PAYLOAD_HASH_PREFIX = "recommendation-final-payload:v1"
RECOMMENDATION_FINAL_V1_ID_PREFIX = "recommendation-final:v1"
RECOMMENDATION_CANDIDATE_INVENTORY_HASH_PREFIX = "recommendation-inventory:v1"
RECOMMENDATION_CANDIDATE_SNAPSHOT_ID_PREFIX = "recommendation-candidates:v1"

RecommendationFinalStatus: TypeAlias = Literal["available", "unavailable"]
ExplicitRecommendationUnavailableReason: TypeAlias = Literal[
    "missing_user_id",
    "missing_subject",
    "profile_unavailable",
    "history_unavailable",
    "no_eligible_candidates",
    "unsupported_subject_scope",
]

_UNAVAILABLE_SUMMARIES: dict[ExplicitRecommendationUnavailableReason, str] = {
    "missing_user_id": (
        "Personalized recommendations require an authenticated learner identity."
    ),
    "missing_subject": "Personalized recommendations require one explicit subject.",
    "profile_unavailable": (
        "Personalized recommendations are unavailable because the learner profile "
        "is unavailable."
    ),
    "history_unavailable": (
        "Personalized recommendations are unavailable because learner history "
        "is unavailable."
    ),
    "no_eligible_candidates": (
        "Personalized recommendations are unavailable because no catalog "
        "candidate met the strict evidence and score thresholds."
    ),
    "unsupported_subject_scope": (
        "Personalized recommendations do not support the requested subject scope."
    ),
}

_HASH_PATTERN = r"^recommendation-final-payload:v1:[0-9a-f]{64}$"
_FINAL_ID_PATTERN = r"^recommendation-final:v1:[0-9a-f]{64}$"
_INVENTORY_HASH_PATTERN = r"^recommendation-inventory:v1:[0-9a-f]{64}$"
_SNAPSHOT_ID_PATTERN = r"^recommendation-candidates:v1:[0-9a-f]{64}$"
_SOURCE_FINGERPRINT_PATTERN = r"^[0-9a-f]{64}$"
_NONEMPTY_ID_PATTERN = r"^.*\S.*$"


class RecommendationFinalContractError(RuntimeError):
    """Raised when a validated recommendation cannot become a public terminal."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class _StrictPublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @model_validator(mode="after")
    def reject_unnormalized_text(self) -> "_StrictPublicModel":
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, str) and (not value.strip() or value != value.strip()):
                raise ValueError("text fields must be normalized and non-blank")
        return self


class RecommendationFinalCandidateV1(_StrictPublicModel):
    """One public catalog target proven to exist in the curated graph snapshot."""

    resource_id: str = Field(min_length=1, max_length=200)
    resource_type: ResourceType
    subject: str = Field(min_length=1, max_length=120)
    topic_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)


class RecommendationCandidateSnapshotContentV1(_StrictPublicModel):
    """Content-addressed projection of the catalog inventory used by a terminal."""

    schema_version: Literal["recommendation_candidate_snapshot_v1"]
    source_schema_version: Literal["knowledge_graph_v1"]
    source_data_version: str = Field(min_length=1, max_length=500)
    source_fingerprint: str = Field(pattern=_SOURCE_FINGERPRINT_PATTERN)
    subject: str = Field(min_length=1, max_length=120)
    candidate_count: int = Field(ge=1, le=400_000)
    inventory_hash: str = Field(pattern=_INVENTORY_HASH_PATTERN)
    # JSON has no tuple type. This field-level exception permits only the wire
    # array-to-tuple container projection; nested models remain strict, and the
    # public Mapping validator rejects Python tuples before JSON encoding.
    targets: tuple[RecommendationFinalCandidateV1, ...] = Field(
        min_length=1,
        max_length=50,
        strict=False,
    )

    @model_validator(mode="after")
    def validate_snapshot_content(self) -> "RecommendationCandidateSnapshotContentV1":
        target_ids = tuple(item.resource_id for item in self.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError(
                "candidate snapshot target resource_id values must be unique"
            )
        if self.candidate_count < len(self.targets):
            raise ValueError("candidate_count cannot be smaller than target count")
        if any(item.subject != self.subject for item in self.targets):
            raise ValueError("candidate snapshot targets must match its subject")
        return self


class RecommendationCandidateSnapshotV1(RecommendationCandidateSnapshotContentV1):
    """Candidate snapshot with a derived content-consistency identity."""

    snapshot_id: str = Field(pattern=_SNAPSHOT_ID_PATTERN)

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> "RecommendationCandidateSnapshotV1":
        expected = stable_recommendation_candidate_snapshot_id(self)
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match candidate snapshot content")
        return self


class RecommendationFinalItemV1(_StrictPublicModel):
    """Bounded public recommendation without private learner evidence IDs."""

    recommendation_id: str = Field(min_length=1, max_length=160)
    resource_id: str = Field(min_length=1, max_length=200)
    resource_type: ResourceType
    topic_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    rank: int = Field(ge=1, le=50)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str = Field(min_length=1, max_length=1_000)


class RecommendationFinalV1Content(_StrictPublicModel):
    """Canonical public body before its derived payload hash and final ID."""

    schema_version: Literal["recommendation_final_v1"]
    type: Literal["recommendation_final"]
    thread_id: str = Field(pattern=_NONEMPTY_ID_PATTERN, max_length=160)
    request_id: str = Field(pattern=_NONEMPTY_ID_PATTERN, max_length=160)
    terminal_status: RecommendationFinalStatus
    mode: Literal["explicit_request"]
    user_id: str | None = Field(max_length=160)
    subject: str | None = Field(max_length=120)
    learning_guidance_runtime_fingerprint: str = Field(
        pattern=_SOURCE_FINGERPRINT_PATTERN
    )
    generated_at: str | None = Field(max_length=64)
    # See the targets field above for the strict wire/domain boundary.
    recommendations: tuple[RecommendationFinalItemV1, ...] = Field(
        max_length=50,
        strict=False,
    )
    candidate_snapshot: RecommendationCandidateSnapshotV1 | None
    unavailable_reason: ExplicitRecommendationUnavailableReason | None
    summary: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_terminal_truth(self) -> "RecommendationFinalV1Content":
        try:
            parsed_request_id = UUID(self.request_id)
        except ValueError as exc:
            raise ValueError("request_id must be a UUID") from exc
        if str(parsed_request_id) != self.request_id:
            raise ValueError("request_id must use canonical UUID text")

        if self.generated_at is not None:
            try:
                parsed_generated_at = datetime.fromisoformat(self.generated_at)
            except ValueError as exc:
                raise ValueError("generated_at must use canonical ISO 8601") from exc
            if (
                parsed_generated_at.tzinfo is None
                or parsed_generated_at.utcoffset() is None
            ):
                raise ValueError("generated_at must be timezone-aware")
            if parsed_generated_at.isoformat() != self.generated_at:
                raise ValueError("generated_at must use canonical ISO 8601")

        if self.terminal_status == "available":
            if (
                self.user_id is None
                or self.subject is None
                or self.generated_at is None
                or not self.recommendations
                or self.candidate_snapshot is None
                or self.unavailable_reason is not None
            ):
                raise ValueError(
                    "available recommendation final requires user, subject, "
                    "generated_at, recommendations, and candidate snapshot only"
                )
            if self.candidate_snapshot.subject != self.subject:
                raise ValueError("candidate snapshot subject must match the final")
        else:
            if (
                self.recommendations
                or self.candidate_snapshot is not None
                or self.generated_at is not None
                or self.unavailable_reason is None
            ):
                raise ValueError(
                    "unavailable recommendation final requires only an explicit reason"
                )
            if (
                self.unavailable_reason == "missing_subject"
                and self.subject is not None
            ):
                raise ValueError("missing_subject final cannot contain a subject")
            if (
                self.unavailable_reason == "missing_user_id"
                and self.user_id is not None
            ):
                raise ValueError("missing_user_id final cannot contain a user_id")
            if self.unavailable_reason != "missing_user_id" and self.user_id is None:
                raise ValueError(
                    "unavailable reasons other than missing_user_id require a user_id"
                )
            return self

        recommendation_ids = tuple(
            item.recommendation_id for item in self.recommendations
        )
        if len(recommendation_ids) != len(set(recommendation_ids)):
            raise ValueError("recommendation_id values must be unique")
        ranks = tuple(item.rank for item in self.recommendations)
        if ranks != tuple(range(1, len(self.recommendations) + 1)):
            raise ValueError("recommendation ranks must be contiguous and ordered")

        if self.candidate_snapshot is None:
            raise AssertionError("available final requires a candidate snapshot")
        targets = {
            candidate.resource_id: candidate
            for candidate in self.candidate_snapshot.targets
        }
        if set(targets) != {item.resource_id for item in self.recommendations}:
            raise ValueError(
                "candidate snapshot targets must exactly match recommendation targets"
            )
        for item in self.recommendations:
            target = targets[item.resource_id]
            if (
                target.resource_type != item.resource_type
                or target.topic_id != item.topic_id
                or target.title != item.title
            ):
                raise ValueError(
                    "recommendation target differs from candidate snapshot"
                )
        return self


class RecommendationFinalV1(RecommendationFinalV1Content):
    """Authoritative recommendation-only terminal with stable request identity."""

    recommendation_final_id: str = Field(pattern=_FINAL_ID_PATTERN)
    payload_hash: str = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_derived_identity(self) -> "RecommendationFinalV1":
        expected_hash = stable_recommendation_final_v1_hash(self)
        if self.payload_hash != expected_hash:
            raise ValueError("payload_hash does not match Recommendation Final V1")
        expected_id = stable_recommendation_final_v1_id(
            thread_id=self.thread_id,
            request_id=self.request_id,
            payload_hash=self.payload_hash,
        )
        if self.recommendation_final_id != expected_id:
            raise ValueError(
                "recommendation_final_id does not match request identity and payload"
            )
        return self


def build_recommendation_final_v1(
    *,
    thread_id: str,
    request_id: str,
    output: ResourceRecommendationOutputV1,
    knowledge_graph: KnowledgeGraphV1,
    expected_user_id: str | None,
    expected_runtime_fingerprint: str,
) -> RecommendationFinalV1:
    """Project one validated explicit result against its real curated catalog."""

    if not isinstance(output, ResourceRecommendationOutputV1):
        raise TypeError("output must be ResourceRecommendationOutputV1")
    if not isinstance(knowledge_graph, KnowledgeGraphV1):
        raise TypeError("knowledge_graph must be KnowledgeGraphV1")
    if output.mode != "explicit_request":
        raise RecommendationFinalContractError(
            code="recommendation_final_mode_mismatch",
            reason="recommendation-only terminal accepts explicit_request mode only",
        )
    if output.request_id != request_id:
        raise RecommendationFinalContractError(
            code="recommendation_final_request_mismatch",
            reason="recommendation output request_id differs from terminal request_id",
        )
    if output.user_id != expected_user_id:
        raise RecommendationFinalContractError(
            code="recommendation_final_user_mismatch",
            reason="recommendation output user_id differs from authenticated state",
        )
    if output.runtime_fingerprint != expected_runtime_fingerprint:
        raise RecommendationFinalContractError(
            code="recommendation_final_runtime_mismatch",
            reason="recommendation output runtime fingerprint is no longer current",
        )

    if output.status == "unavailable":
        if output.unavailable_reason is None:
            raise AssertionError("validated unavailable output requires a reason")
        if output.unavailable_reason not in _UNAVAILABLE_SUMMARIES:
            raise RecommendationFinalContractError(
                code="recommendation_final_unavailable_reason_mismatch",
                reason="unavailable reason is not valid for explicit recommendations",
            )
        unavailable_reason = cast(
            ExplicitRecommendationUnavailableReason,
            output.unavailable_reason,
        )
        content = RecommendationFinalV1Content(
            schema_version=RECOMMENDATION_FINAL_V1_SCHEMA_VERSION,
            type="recommendation_final",
            thread_id=thread_id,
            request_id=request_id,
            terminal_status="unavailable",
            mode="explicit_request",
            user_id=output.user_id,
            subject=output.subject,
            learning_guidance_runtime_fingerprint=output.runtime_fingerprint,
            generated_at=None,
            recommendations=(),
            candidate_snapshot=None,
            unavailable_reason=unavailable_reason,
            summary=_UNAVAILABLE_SUMMARIES[unavailable_reason],
        )
        return _sign_recommendation_final(content)

    if output.batch is None or output.subject is None:
        raise AssertionError("validated available output requires batch and subject")
    graph_subject = knowledge_graph.subject(output.subject)
    if graph_subject is None:
        raise RecommendationFinalContractError(
            code="recommendation_final_unknown_subject",
            reason="recommendation subject is absent from the curated knowledge graph",
        )

    inventory: list[dict[str, str]] = []
    candidate_by_id: dict[str, tuple[str, ResourceType, str, str]] = {}
    for topic in graph_subject.topics:
        for resource in topic.resources:
            inventory.append(
                {
                    "resource_id": resource.resource_id,
                    "resource_type": resource.resource_type,
                    "subject": graph_subject.subject_id,
                    "topic_id": topic.topic_id,
                    "title": resource.title,
                }
            )
            candidate_by_id[resource.resource_id] = (
                topic.topic_id,
                resource.resource_type,
                resource.title,
                graph_subject.subject_id,
            )

    targets: list[RecommendationFinalCandidateV1] = []
    recommendations: list[RecommendationFinalItemV1] = []
    for item in output.batch.items:
        candidate_record = candidate_by_id.get(item.resource_id)
        if candidate_record is None:
            raise RecommendationFinalContractError(
                code="recommendation_final_unknown_candidate",
                reason="recommendation target is absent from the curated catalog",
            )
        topic_id, resource_type, title, subject = candidate_record
        if (
            resource_type != item.resource_type
            or topic_id != item.topic_id
            or title != item.title
            or subject != item.subject
        ):
            raise RecommendationFinalContractError(
                code="recommendation_final_candidate_mismatch",
                reason="recommendation target differs from its curated catalog record",
            )
        private_reference_ids = (*item.profile_signal_ids, *item.history_ids)
        if any(
            reference_id in item.recommendation_id
            or reference_id in item.title
            or reference_id in item.reason
            for reference_id in private_reference_ids
        ):
            raise RecommendationFinalContractError(
                code="recommendation_final_private_evidence_exposure",
                reason="public recommendation text contains a private evidence ID",
            )
        targets.append(
            RecommendationFinalCandidateV1(
                resource_id=item.resource_id,
                resource_type=resource_type,
                subject=subject,
                topic_id=topic_id,
                title=title,
            )
        )
        recommendations.append(
            RecommendationFinalItemV1(
                recommendation_id=item.recommendation_id,
                resource_id=item.resource_id,
                resource_type=item.resource_type,
                topic_id=item.topic_id,
                title=item.title,
                rank=item.rank,
                score=item.score_factors.combined,
                reason=item.reason,
            )
        )

    inventory_hash = _stable_hash(
        RECOMMENDATION_CANDIDATE_INVENTORY_HASH_PREFIX,
        inventory,
    )
    snapshot_content = RecommendationCandidateSnapshotContentV1(
        schema_version="recommendation_candidate_snapshot_v1",
        source_schema_version="knowledge_graph_v1",
        source_data_version=knowledge_graph.data_version,
        source_fingerprint=knowledge_graph.artifact_fingerprint,
        subject=output.subject,
        candidate_count=len(inventory),
        inventory_hash=inventory_hash,
        targets=tuple(targets),
    )
    snapshot = RecommendationCandidateSnapshotV1(
        **snapshot_content.model_dump(),
        snapshot_id=stable_recommendation_candidate_snapshot_id(snapshot_content),
    )
    content = RecommendationFinalV1Content(
        schema_version=RECOMMENDATION_FINAL_V1_SCHEMA_VERSION,
        type="recommendation_final",
        thread_id=thread_id,
        request_id=request_id,
        terminal_status="available",
        mode="explicit_request",
        user_id=output.user_id,
        subject=output.subject,
        learning_guidance_runtime_fingerprint=output.runtime_fingerprint,
        generated_at=output.batch.generated_at.isoformat(),
        recommendations=tuple(recommendations),
        candidate_snapshot=snapshot,
        unavailable_reason=None,
        summary=f"Personalized recommendations available: {len(recommendations)}.",
    )
    return _sign_recommendation_final(content)


def validate_recommendation_final_v1(value: object) -> RecommendationFinalV1:
    """Validate one checkpoint or transport payload with strict JSON semantics."""

    if isinstance(value, RecommendationFinalV1):
        payload = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        _reject_non_json_native_sequences(value)
        payload = dict(value)
    else:
        raise TypeError("Recommendation Final V1 payload must be an object")
    return RecommendationFinalV1.model_validate_json(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        strict=True,
    )


def _reject_non_json_native_sequences(value: object, *, path: str = "root") -> None:
    """Reject Python-only containers before JSON encoding can normalize them."""

    if isinstance(value, tuple):
        raise TypeError(f"{path} must use a JSON array, not a Python tuple")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_json_native_sequences(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            _reject_non_json_native_sequences(item, path=f"{path}.{key}")


def stable_recommendation_candidate_snapshot_id(
    value: RecommendationCandidateSnapshotContentV1,
) -> str:
    if not isinstance(value, RecommendationCandidateSnapshotContentV1):
        raise TypeError("value must be RecommendationCandidateSnapshotContentV1")
    payload = value.model_dump(mode="json", exclude={"snapshot_id"})
    return _stable_hash(RECOMMENDATION_CANDIDATE_SNAPSHOT_ID_PREFIX, payload)


def stable_recommendation_final_v1_hash(
    value: RecommendationFinalV1Content,
) -> str:
    if not isinstance(value, RecommendationFinalV1Content):
        raise TypeError("value must be RecommendationFinalV1Content")
    payload = value.model_dump(
        mode="json",
        exclude={"recommendation_final_id", "payload_hash"},
    )
    return _stable_hash(RECOMMENDATION_FINAL_V1_PAYLOAD_HASH_PREFIX, payload)


def stable_recommendation_final_v1_id(
    *,
    thread_id: str,
    request_id: str,
    payload_hash: str,
) -> str:
    return _stable_hash(
        RECOMMENDATION_FINAL_V1_ID_PREFIX,
        {
            "thread_id": thread_id,
            "request_id": request_id,
            "payload_hash": payload_hash,
        },
    )


def _sign_recommendation_final(
    content: RecommendationFinalV1Content,
) -> RecommendationFinalV1:
    payload_hash = stable_recommendation_final_v1_hash(content)
    return RecommendationFinalV1(
        **content.model_dump(),
        recommendation_final_id=stable_recommendation_final_v1_id(
            thread_id=content.thread_id,
            request_id=content.request_id,
            payload_hash=payload_hash,
        ),
        payload_hash=payload_hash,
    )


def _stable_hash(prefix: str, payload: object) -> str:
    body = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


__all__ = [
    "RECOMMENDATION_FINAL_V1_SCHEMA_VERSION",
    "ExplicitRecommendationUnavailableReason",
    "RecommendationCandidateSnapshotContentV1",
    "RecommendationCandidateSnapshotV1",
    "RecommendationFinalCandidateV1",
    "RecommendationFinalContractError",
    "RecommendationFinalItemV1",
    "RecommendationFinalStatus",
    "RecommendationFinalV1",
    "RecommendationFinalV1Content",
    "build_recommendation_final_v1",
    "stable_recommendation_candidate_snapshot_id",
    "stable_recommendation_final_v1_hash",
    "stable_recommendation_final_v1_id",
    "validate_recommendation_final_v1",
]
