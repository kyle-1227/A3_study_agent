"""Strict manifests and completeness validation for parent-child generations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from src.rag.parent_child._storage_io import (
    atomic_write_bytes,
    canonical_json_bytes,
    model_json_bytes,
    resolve_under_root,
    sha256_bytes,
    validate_relative_path,
)
from src.rag.parent_child.bm25_artifact import digest_identifier_set
from src.rag.parent_child.models import ChildDocument, ParentRecord


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ManifestContractError(RuntimeError):
    """Raised when a sealed manifest violates a business invariant."""


class GenerationCompletenessError(ManifestContractError):
    """Raised when generation artifact ID sets or metadata do not agree."""

    def __init__(self, report: CompletenessReport) -> None:
        self.report = report
        super().__init__(
            "generation completeness validation failed: "
            + ", ".join(report.failure_codes)
        )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 value")


def compute_policy_id(output_affecting_payload: Mapping[str, JsonValue]) -> str:
    """Compute a policy ID from a JSON-only output-affecting payload."""

    return sha256_bytes(canonical_json_bytes(dict(output_affecting_payload)))


class PolicyManifest(BaseModel):
    """Canonical output-affecting parent-child chunk policy."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    canonicalization_version: str = Field(min_length=1)
    id_algorithm_version: str = Field(min_length=1)
    extraction: dict[str, JsonValue]
    page_assembly: dict[str, JsonValue]
    cleaning: dict[str, JsonValue]
    structure: dict[str, JsonValue]
    atomic_blocks: dict[str, JsonValue]
    parent_split: dict[str, JsonValue]
    child_split: dict[str, JsonValue]
    metadata_contract_version: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _policy_id_matches_payload(self) -> PolicyManifest:
        payload = self.model_dump(mode="json", exclude={"policy_id"})
        expected = compute_policy_id(payload)
        if self.policy_id != expected:
            raise ValueError("policy_id does not match output-affecting configuration")
        return self


class PolicyManifestSet(BaseModel):
    """Complete sorted policy inventory for one immutable generation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["policy_manifest_set_v1"]
    policies: tuple[PolicyManifest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_policy_inventory(self) -> PolicyManifestSet:
        policy_ids = tuple(policy.policy_id for policy in self.policies)
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy manifest set contains duplicate policy IDs")
        if policy_ids != tuple(sorted(policy_ids)):
            raise ValueError("policy manifest set must be sorted by policy_id")
        return self


class SubjectManifestEntry(BaseModel):
    """One subject's immutable source and policy inventory."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    subject_id: str = Field(min_length=1)
    directory_relpath: str = Field(min_length=1)
    source_file_count: int = Field(ge=0)
    source_manifest_sha256: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    exclusion_state: Literal["active", "excluded"]
    exclusion_reason: str

    @model_validator(mode="after")
    def _validate_subject_entry(self) -> SubjectManifestEntry:
        validate_relative_path(self.directory_relpath)
        _require_sha256(
            self.source_manifest_sha256,
            field_name="source_manifest_sha256",
        )
        if self.exclusion_state == "active" and self.exclusion_reason:
            raise ValueError("active subjects cannot have an exclusion reason")
        if self.exclusion_state == "excluded" and not self.exclusion_reason:
            raise ValueError("excluded subjects require an exclusion reason")
        if self.exclusion_state == "excluded" and (
            self.parent_count != 0 or self.child_count != 0
        ):
            raise ValueError("excluded subjects cannot contain indexed records")
        return self


class SubjectManifest(BaseModel):
    """The complete subject inventory sealed into a generation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    entries: tuple[SubjectManifestEntry, ...]

    @model_validator(mode="after")
    def _subject_ids_are_unique_and_sorted(self) -> SubjectManifest:
        subject_ids = [entry.subject_id for entry in self.entries]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject manifest contains duplicate subject IDs")
        if subject_ids != sorted(subject_ids):
            raise ValueError("subject manifest entries must be sorted by subject_id")
        return self


ArtifactType = Literal[
    "chroma_children",
    "parent_store",
    "bm25_corpus",
    "bm25_manifest",
    "policy_manifest",
    "subject_manifest",
    "build_report",
]


class ArtifactDescriptor(BaseModel):
    """Digest and schema identity for a generation artifact."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    artifact_type: ArtifactType
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_descriptor(self) -> ArtifactDescriptor:
        validate_relative_path(self.relative_path)
        _require_sha256(self.sha256, field_name="artifact sha256")
        return self


class EmbeddingManifestIdentity(BaseModel):
    """Non-secret identity of the child embedding artifact."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url_identity: str = Field(min_length=1)
    input_types: tuple[str, ...] = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    distance_metric: Literal["cosine", "l2", "ip"]

    @model_validator(mode="after")
    def _validate_embedding_identity(self) -> EmbeddingManifestIdentity:
        if any(not input_type for input_type in self.input_types):
            raise ValueError("embedding input types cannot be empty")
        if len(self.input_types) != len(set(self.input_types)):
            raise ValueError("embedding input types cannot contain duplicates")
        _require_sha256(self.fingerprint, field_name="embedding fingerprint")
        return self


class Bm25ManifestIdentity(BaseModel):
    """Tokenizer identity shared by every subject-local BM25 artifact."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    tokenizer_name: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    dictionary_sha256: str = Field(min_length=1)
    tokenizer_fingerprint: str = Field(min_length=1)
    artifact_format: Literal["jsonl"]

    @model_validator(mode="after")
    def _validate_bm25_identity(self) -> Bm25ManifestIdentity:
        _require_sha256(self.dictionary_sha256, field_name="BM25 dictionary digest")
        _require_sha256(
            self.tokenizer_fingerprint,
            field_name="BM25 tokenizer fingerprint",
        )
        return self


class GenerationCounts(BaseModel):
    """Sealed generation cardinalities."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_count: int = Field(ge=0)
    subject_count: int = Field(ge=0)
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    bm25_child_count: int = Field(ge=0)


class GenerationIntegrityCounts(BaseModel):
    """All integrity counters that must be zero before activation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    duplicate_parent_count: int = Field(ge=0)
    duplicate_child_count: int = Field(ge=0)
    orphan_child_count: int = Field(ge=0)
    unreferenced_parent_count: int = Field(ge=0)
    generation_mismatch_count: int = Field(ge=0)
    policy_mismatch_count: int = Field(ge=0)
    subject_mismatch_count: int = Field(ge=0)
    bm25_mismatch_count: int = Field(ge=0)
    chroma_mismatch_count: int = Field(ge=0)

    def all_zero(self) -> bool:
        """Return whether every completeness counter is zero."""

        return all(value == 0 for value in self.model_dump(mode="python").values())


class CompletenessReport(BaseModel):
    """Deterministic cross-artifact parent/child completeness result."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    counts: GenerationCounts
    integrity: GenerationIntegrityCounts
    parent_id_set_sha256: str = Field(min_length=1)
    child_id_set_sha256: str = Field(min_length=1)
    bm25_child_id_set_sha256: str = Field(min_length=1)
    chroma_child_id_set_sha256: str = Field(min_length=1)
    validation_passed: bool
    failure_codes: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_report(self) -> CompletenessReport:
        for field_name in (
            "parent_id_set_sha256",
            "child_id_set_sha256",
            "bm25_child_id_set_sha256",
            "chroma_child_id_set_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name=field_name)
        expected_passed = self.integrity.all_zero()
        if self.validation_passed != expected_passed:
            raise ValueError("validation_passed conflicts with integrity counters")
        if self.validation_passed and self.failure_codes:
            raise ValueError("successful completeness reports cannot have failures")
        if not self.validation_passed and not self.failure_codes:
            raise ValueError("failed completeness reports require failure codes")
        if tuple(sorted(set(self.failure_codes))) != self.failure_codes:
            raise ValueError("failure_codes must be unique and sorted")
        return self


class GenerationManifest(BaseModel):
    """Sealed immutable generation manifest eligible for registry activation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    build_state: Literal["ready"]
    code_revision: str = Field(min_length=1)
    build_time_utc: datetime
    collection_name: str = Field(min_length=1)
    artifacts: tuple[ArtifactDescriptor, ...] = Field(min_length=1)
    embedding: EmbeddingManifestIdentity
    bm25: Bm25ManifestIdentity
    subject_manifest_sha256: str = Field(min_length=1)
    policy_manifest_sha256: str = Field(min_length=1)
    subject_fingerprint: str = Field(min_length=1)
    policy_fingerprint: str = Field(min_length=1)
    source_fingerprint: str = Field(min_length=1)
    parent_id_set_sha256: str = Field(min_length=1)
    child_id_set_sha256: str = Field(min_length=1)
    counts: GenerationCounts
    integrity: GenerationIntegrityCounts
    validation_report_sha256: str = Field(min_length=1)
    validation_passed: Literal[True]

    @model_validator(mode="after")
    def _validate_sealed_manifest(self) -> GenerationManifest:
        digest_fields = (
            "subject_manifest_sha256",
            "policy_manifest_sha256",
            "subject_fingerprint",
            "policy_fingerprint",
            "source_fingerprint",
            "parent_id_set_sha256",
            "child_id_set_sha256",
            "validation_report_sha256",
        )
        for field_name in digest_fields:
            _require_sha256(getattr(self, field_name), field_name=field_name)
        if self.build_time_utc.tzinfo is None:
            raise ValueError("build_time_utc must be timezone-aware")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("generation artifact paths must be unique")
        required_types = {
            "chroma_children",
            "parent_store",
            "bm25_corpus",
            "bm25_manifest",
            "policy_manifest",
            "subject_manifest",
            "build_report",
        }
        actual_types = {artifact.artifact_type for artifact in self.artifacts}
        missing_types = required_types - actual_types
        if missing_types:
            raise ValueError(
                "generation manifest is missing artifact types: "
                + ", ".join(sorted(missing_types))
            )
        if not self.integrity.all_zero():
            raise ValueError("ready generation manifest has nonzero integrity counters")
        if self.counts.child_count != self.counts.bm25_child_count:
            raise ValueError("ready generation BM25 and child counts must match")
        return self


def _mismatch_size(left: Sequence[str], right: Sequence[str]) -> int:
    return (
        len(set(left).symmetric_difference(right))
        + (len(left) - len(set(left)))
        + (len(right) - len(set(right)))
    )


def build_completeness_report(
    parents: Sequence[ParentRecord],
    children: Sequence[ChildDocument],
    bm25_child_ids_by_subject: Mapping[str, Sequence[str]],
    chroma_child_ids: Sequence[str],
    expected_policy_by_subject: Mapping[str, str],
    *,
    report_schema_version: str,
    expected_generation_id: str,
    source_count: int,
) -> CompletenessReport:
    """Compare all parent/child/BM25/Chroma identities without hiding failures."""

    if not report_schema_version or not expected_generation_id:
        raise ValueError("report schema version and generation ID are required")
    if isinstance(source_count, bool) or source_count < 0:
        raise ValueError("source_count must be a non-negative integer")

    parent_ids = [parent.parent_id for parent in parents]
    child_ids = [child.metadata.child_id for child in children]
    bm25_ids = [
        child_id
        for subject in sorted(bm25_child_ids_by_subject)
        for child_id in bm25_child_ids_by_subject[subject]
    ]
    parent_by_id = {parent.parent_id: parent for parent in parents}
    referenced_parent_ids = {child.metadata.parent_id for child in children}
    subjects = {parent.subject for parent in parents} | {
        child.metadata.subject for child in children
    }

    duplicate_parent_count = len(parent_ids) - len(set(parent_ids))
    duplicate_child_count = len(child_ids) - len(set(child_ids))
    orphan_child_count = sum(
        child.metadata.parent_id not in parent_by_id for child in children
    )
    unreferenced_parent_count = sum(
        parent_id not in referenced_parent_ids for parent_id in set(parent_ids)
    )
    generation_mismatch_count = sum(
        parent.generation_id != expected_generation_id for parent in parents
    ) + sum(
        child.metadata.generation_id != expected_generation_id for child in children
    )

    policy_mismatch_count = len(
        subjects.symmetric_difference(expected_policy_by_subject)
    )
    policy_mismatch_count += sum(
        expected_policy_by_subject.get(parent.subject) != parent.policy_id
        for parent in parents
    )
    subject_mismatch_count = len(
        subjects.symmetric_difference(bm25_child_ids_by_subject)
    )
    for child in children:
        parent = parent_by_id.get(child.metadata.parent_id)
        if parent is None:
            continue
        policy_mismatch_count += child.metadata.policy_id != parent.policy_id
        policy_mismatch_count += (
            expected_policy_by_subject.get(child.metadata.subject)
            != child.metadata.policy_id
        )
        subject_mismatch_count += child.metadata.subject != parent.subject
        if child.metadata.child_id not in set(
            bm25_child_ids_by_subject.get(child.metadata.subject, ())
        ):
            subject_mismatch_count += 1

    bm25_mismatch_count = _mismatch_size(child_ids, bm25_ids)
    chroma_mismatch_count = _mismatch_size(child_ids, chroma_child_ids)
    integrity = GenerationIntegrityCounts(
        duplicate_parent_count=duplicate_parent_count,
        duplicate_child_count=duplicate_child_count,
        orphan_child_count=orphan_child_count,
        unreferenced_parent_count=unreferenced_parent_count,
        generation_mismatch_count=generation_mismatch_count,
        policy_mismatch_count=policy_mismatch_count,
        subject_mismatch_count=subject_mismatch_count,
        bm25_mismatch_count=bm25_mismatch_count,
        chroma_mismatch_count=chroma_mismatch_count,
    )
    counter_to_code = {
        "duplicate_parent_count": "duplicate_parent",
        "duplicate_child_count": "duplicate_child",
        "orphan_child_count": "orphan_child",
        "unreferenced_parent_count": "unreferenced_parent",
        "generation_mismatch_count": "generation_mismatch",
        "policy_mismatch_count": "policy_mismatch",
        "subject_mismatch_count": "subject_mismatch",
        "bm25_mismatch_count": "bm25_id_set_mismatch",
        "chroma_mismatch_count": "chroma_id_set_mismatch",
    }
    failure_codes = tuple(
        sorted(
            counter_to_code[field]
            for field, value in integrity.model_dump(mode="python").items()
            if value
        )
    )
    return CompletenessReport(
        schema_version=report_schema_version,
        generation_id=expected_generation_id,
        counts=GenerationCounts(
            source_count=source_count,
            subject_count=len(subjects),
            parent_count=len(parents),
            child_count=len(children),
            bm25_child_count=len(bm25_ids),
        ),
        integrity=integrity,
        parent_id_set_sha256=digest_identifier_set(parent_ids),
        child_id_set_sha256=digest_identifier_set(child_ids),
        bm25_child_id_set_sha256=digest_identifier_set(bm25_ids),
        chroma_child_id_set_sha256=digest_identifier_set(chroma_child_ids),
        validation_passed=integrity.all_zero(),
        failure_codes=failure_codes,
    )


def assert_generation_complete(
    parents: Sequence[ParentRecord],
    children: Sequence[ChildDocument],
    bm25_child_ids_by_subject: Mapping[str, Sequence[str]],
    chroma_child_ids: Sequence[str],
    expected_policy_by_subject: Mapping[str, str],
    *,
    report_schema_version: str,
    expected_generation_id: str,
    source_count: int,
) -> CompletenessReport:
    """Return a passing report or raise with the complete failed report attached."""

    report = build_completeness_report(
        parents,
        children,
        bm25_child_ids_by_subject,
        chroma_child_ids,
        expected_policy_by_subject,
        report_schema_version=report_schema_version,
        expected_generation_id=expected_generation_id,
        source_count=source_count,
    )
    if not report.validation_passed:
        raise GenerationCompletenessError(report)
    return report


def write_strict_model(
    root: str | Path,
    relative_path: str,
    model: BaseModel,
    *,
    overwrite: bool,
) -> Path:
    """Atomically write an already validated model as canonical JSON."""

    return atomic_write_bytes(
        root,
        relative_path,
        model_json_bytes(model),
        overwrite=overwrite,
    )


def read_strict_model(
    root: str | Path,
    relative_path: str,
    model_type: type[_ModelT],
) -> _ModelT:
    """Read JSON through the requested strict Pydantic model."""

    path = resolve_under_root(root, relative_path, must_exist=True)
    if not path.is_file() or path.is_symlink():
        raise ManifestContractError("manifest path must reference a regular file")
    return model_type.model_validate_json(path.read_bytes())
