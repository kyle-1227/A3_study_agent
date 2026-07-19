"""Strict mutable Parent--Child primary index control plane."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.rag_index_config import RagIndexConfig
from src.rag.parent_child._storage_io import (
    atomic_write_bytes,
    canonical_json_bytes,
    model_json_bytes,
    resolve_under_root,
    sha256_bytes,
    sha256_path,
    validate_generation_id,
    validate_relative_path,
)
from src.rag.parent_child.bm25_artifact import (
    compute_tokenizer_fingerprint,
    read_subject_bm25_artifact,
)
from src.rag.parent_child.chroma_runtime_snapshot import ChromaRuntimeSnapshot
from src.rag.parent_child.manifests import (
    PolicyManifestSet,
    SubjectManifest,
    read_strict_model,
)
from src.rag.parent_child.parent_store import ParentStore

PRIMARY_ROOT = "primary"
PRIMARY_STATE_RELATIVE_PATH = "primary/primary_state.json"
PRIMARY_METADATA_FILENAME = "primary_metadata.json"
PRIMARY_VALIDATION_FILENAME: Final[Literal["primary_validation.json"]] = (
    "primary_validation.json"
)


def compute_embedding_fingerprint(config: RagIndexConfig) -> str:
    """Fingerprint the configured embedding identity without a provider call."""

    embedding = config.embedding
    return sha256_bytes(
        canonical_json_bytes(
            {
                "base_url": embedding.base_url,
                "document_input_type": embedding.document_input_type,
                "endpoint_path": embedding.endpoint_path,
                "expected_dimension": embedding.expected_dimension,
                "input_type_field": embedding.input_type_field,
                "metric": embedding.distance_metric,
                "model": embedding.model,
                "normalization_contract": embedding.normalization_contract,
                "protocol": embedding.protocol,
                "provider": embedding.provider,
                "provider_routing": (
                    None
                    if embedding.provider_routing is None
                    else {
                        "allow_fallbacks": embedding.provider_routing.allow_fallbacks,
                        "order": embedding.provider_routing.order,
                    }
                ),
                "query_input_type": embedding.query_input_type,
            }
        )
    )


class PrimaryIndexError(RuntimeError):
    """Primary state or artifacts are unsafe to serve."""


class PrimaryIndexStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["primary_index_state_v1"]
    primary_revision: int = Field(ge=1)
    active_directory_relative_path: str
    metadata_relative_path: str
    validation_relative_path: str
    updated_at_utc: datetime
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_status: Literal["valid"]

    @field_validator(
        "active_directory_relative_path",
        "metadata_relative_path",
        "validation_relative_path",
    )
    @classmethod
    def relative(cls, value: str) -> str:
        validate_relative_path(value)
        return value

    @model_validator(mode="after")
    def contained(self) -> "PrimaryIndexStateV1":
        root = Path(self.active_directory_relative_path)
        if self.active_directory_relative_path != revision_path(self.primary_revision):
            raise ValueError("primary revision path is invalid")
        if (
            Path(self.metadata_relative_path) != root / PRIMARY_METADATA_FILENAME
            or Path(self.validation_relative_path) != root / PRIMARY_VALIDATION_FILENAME
        ):
            raise ValueError("primary control files must be inside the revision")
        if self.updated_at_utc.tzinfo is None:
            raise ValueError("primary update time must be timezone-aware")
        return self


class PrimaryIndexMetadataV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["primary_index_metadata_v1"]
    primary_revision: int = Field(ge=1)
    artifact_identity: str = Field(min_length=1, max_length=160)
    built_at_utc: datetime
    collection_name: str = Field(min_length=3)
    chroma_directory_relative_path: Literal["chroma_children"]
    parent_store_relative_path: Literal["parents.sqlite"]
    policy_manifest_relative_path: Literal["policy_manifest.json"]
    subject_manifest_relative_path: Literal["subject_manifest.json"]
    bm25_directory_relative_path: Literal["bm25"]
    validation_relative_path: Literal["primary_validation.json"]
    embedding_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_dimension: int = Field(gt=0)
    distance_metric: Literal["cosine", "l2", "ip"]
    bm25_tokenizer_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_policy_map: dict[str, str] = Field(min_length=1)
    available_subjects: tuple[str, ...] = Field(min_length=1)
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_status: Literal["valid"]

    @field_validator("artifact_identity")
    @classmethod
    def identity(cls, value: str) -> str:
        return validate_generation_id(value)

    @model_validator(mode="after")
    def valid(self) -> "PrimaryIndexMetadataV1":
        if self.built_at_utc.tzinfo is None:
            raise ValueError("primary build time must be timezone-aware")
        if self.available_subjects != tuple(sorted(set(self.available_subjects))):
            raise ValueError("primary subjects must be sorted and unique")
        if set(self.available_subjects) != set(self.subject_policy_map):
            raise ValueError("primary subjects and policies must match")
        if any(
            not subject or subject != subject.strip() or not policy
            for subject, policy in self.subject_policy_map.items()
        ):
            raise ValueError("primary subject policy map is invalid")
        return self


class PrimaryIndexValidationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["primary_index_validation_v1"]
    primary_revision: int = Field(ge=1)
    artifact_identity: str
    validated_at_utc: datetime
    validation_status: Literal["valid"]
    validated_subjects: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self) -> "PrimaryIndexValidationV1":
        if self.validated_at_utc.tzinfo is None:
            raise ValueError("primary validation time must be timezone-aware")
        if self.validated_subjects != tuple(sorted(set(self.validated_subjects))):
            raise ValueError("validated subjects must be sorted and unique")
        return self


class PrimaryPublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["primary_publish_result_v1"]
    primary_revision: int = Field(ge=1)
    active_directory_relative_path: str
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def revision_path(revision: int) -> str:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("primary revision must be positive")
    return f"primary/revisions/r{revision}"


def primary_revision_relative_path(revision: int) -> str:
    return revision_path(revision)


def primary_metadata_relative_path(revision: int) -> str:
    return f"{revision_path(revision)}/{PRIMARY_METADATA_FILENAME}"


def primary_validation_relative_path(revision: int) -> str:
    return f"{revision_path(revision)}/{PRIMARY_VALIDATION_FILENAME}"


def primary_staging_relative_path(build_id: str) -> str:
    return f"primary/.staging/{validate_generation_id(build_id)}"


def fingerprint(config: RagIndexConfig) -> str:
    token = compute_tokenizer_fingerprint(
        tokenizer_name=config.bm25.tokenizer,
        tokenizer_version=config.bm25.tokenizer_version,
        dictionary_sha256=config.bm25.dictionary_hash,
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "primary_index_config_fingerprint_v1",
                "collection_name": config.storage.collection_name,
                "embedding_fingerprint": compute_embedding_fingerprint(config),
                "embedding_dimension": config.embedding.expected_dimension,
                "distance_metric": config.embedding.distance_metric,
                "bm25_tokenizer_fingerprint": token,
                "subject_policy_map": dict(sorted(config.subject_policy_map.items())),
                "retrieval": config.retrieval.model_dump(mode="json"),
            }
        )
    )


compute_primary_config_fingerprint = fingerprint


def metadata_from_config(
    config: RagIndexConfig,
    *,
    primary_revision: int,
    artifact_identity: str,
    available_subjects: tuple[str, ...],
    built_at_utc: datetime,
) -> PrimaryIndexMetadataV1:
    return PrimaryIndexMetadataV1(
        schema_version="primary_index_metadata_v1",
        primary_revision=primary_revision,
        artifact_identity=artifact_identity,
        built_at_utc=built_at_utc,
        collection_name=config.storage.collection_name,
        chroma_directory_relative_path="chroma_children",
        parent_store_relative_path="parents.sqlite",
        policy_manifest_relative_path="policy_manifest.json",
        subject_manifest_relative_path="subject_manifest.json",
        bm25_directory_relative_path="bm25",
        validation_relative_path=PRIMARY_VALIDATION_FILENAME,
        embedding_fingerprint=compute_embedding_fingerprint(config),
        embedding_dimension=config.embedding.expected_dimension,
        distance_metric=config.embedding.distance_metric,
        bm25_tokenizer_fingerprint=compute_tokenizer_fingerprint(
            tokenizer_name=config.bm25.tokenizer,
            tokenizer_version=config.bm25.tokenizer_version,
            dictionary_sha256=config.bm25.dictionary_hash,
        ),
        subject_policy_map=dict(sorted(config.subject_policy_map.items())),
        available_subjects=available_subjects,
        config_fingerprint=fingerprint(config),
        validation_status="valid",
    )


primary_metadata_from_config = metadata_from_config


def _path(root: Path, relative: str, *, directory: bool) -> Path:
    path = resolve_under_root(root, relative, must_exist=True)
    if (
        path.is_symlink()
        or (directory and not path.is_dir())
        or (not directory and not path.is_file())
    ):
        raise PrimaryIndexError("primary artifact path is invalid")
    if path.is_dir() and any(item.is_symlink() for item in path.rglob("*")):
        raise PrimaryIndexError("primary artifact contains a symlink")
    return path


class _NoQueryEmbedding:
    def embed_query(self, text: str) -> list[float]:
        del text
        raise PrimaryIndexError("structural validation cannot call a provider")


def validate_primary_revision(
    *,
    config: RagIndexConfig,
    artifact_root: Path,
    metadata: PrimaryIndexMetadataV1,
    chroma_snapshot: ChromaRuntimeSnapshot | None = None,
    validated_at_utc: datetime | None = None,
) -> PrimaryIndexValidationV1:
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise PrimaryIndexError("primary revision root is invalid")
    root = artifact_root.resolve(strict=True)
    if (
        metadata.config_fingerprint != fingerprint(config)
        or metadata.collection_name != config.storage.collection_name
        or metadata.embedding_fingerprint != compute_embedding_fingerprint(config)
        or metadata.embedding_dimension != config.embedding.expected_dimension
        or metadata.distance_metric != config.embedding.distance_metric
        or metadata.subject_policy_map
        != dict(sorted(config.subject_policy_map.items()))
    ):
        raise PrimaryIndexError("primary metadata differs from strict config")
    expected_tokenizer = compute_tokenizer_fingerprint(
        tokenizer_name=config.bm25.tokenizer,
        tokenizer_version=config.bm25.tokenizer_version,
        dictionary_sha256=config.bm25.dictionary_hash,
    )
    if metadata.bm25_tokenizer_fingerprint != expected_tokenizer:
        raise PrimaryIndexError("primary BM25 identity differs from config")
    source_chroma = _path(
        root,
        metadata.chroma_directory_relative_path,
        directory=True,
    )
    chroma = source_chroma
    if chroma_snapshot is not None:
        if not isinstance(chroma_snapshot, ChromaRuntimeSnapshot):
            raise PrimaryIndexError("primary Chroma snapshot is invalid")
        if chroma_snapshot.index_root != config.storage.index_root.resolve(strict=True):
            raise PrimaryIndexError("primary Chroma snapshot index root differs")
        if chroma_snapshot.source_sha256 != sha256_path(source_chroma):
            raise PrimaryIndexError("primary Chroma snapshot differs from source")
        chroma = chroma_snapshot.persist_directory
        if (
            not chroma.is_absolute()
            or chroma.is_symlink()
            or not chroma.is_dir()
            or any(item.is_symlink() for item in chroma.rglob("*"))
        ):
            raise PrimaryIndexError("primary Chroma snapshot is unsafe")
    _path(root, metadata.parent_store_relative_path, directory=False)
    _path(root, metadata.bm25_directory_relative_path, directory=True)
    policies = read_strict_model(
        root, metadata.policy_manifest_relative_path, PolicyManifestSet
    )
    if {policy.policy_id for policy in policies.policies} != set(config.chunk_policies):
        raise PrimaryIndexError("primary policy inventory differs from config")
    subjects = read_strict_model(
        root, metadata.subject_manifest_relative_path, SubjectManifest
    )
    active = tuple(
        item for item in subjects.entries if item.exclusion_state == "active"
    )
    if (
        subjects.generation_id != metadata.artifact_identity
        or tuple(item.subject_id for item in active) != metadata.available_subjects
        or {item.subject_id: item.policy_id for item in active}
        != metadata.subject_policy_map
        or any(item.parent_count <= 0 or item.child_count <= 0 for item in active)
    ):
        raise PrimaryIndexError("primary subject inventory differs from metadata")
    from src.rag.parent_child.runtime_resources import ChromaChildSearchChannel

    vector = None
    parents = None
    try:
        vector = ChromaChildSearchChannel(
            persist_directory=chroma,
            collection_name=metadata.collection_name,
            generation_id=metadata.artifact_identity,
            expected_dimension=metadata.embedding_dimension,
            distance_metric=metadata.distance_metric,
            query_embedding_provider=_NoQueryEmbedding(),
            child_lookup_batch_size=config.embedding.batch_size,
        )
        parents = ParentStore.open_readonly(
            root,
            metadata.parent_store_relative_path,
            expected_schema_version=config.storage.parent_store_schema_version,
            expected_generation_id=metadata.artifact_identity,
            busy_timeout_seconds=config.storage.parent_store_busy_timeout_seconds,
        )
        parents.verify_integrity()
        for subject in metadata.available_subjects:
            _manifest, rows = read_subject_bm25_artifact(
                root,
                f"bm25/{subject}.manifest.json",
                expected_manifest_schema_version="bm25_manifest_v1",
                expected_generation_id=metadata.artifact_identity,
                expected_subject=subject,
                expected_tokenizer_fingerprint=metadata.bm25_tokenizer_fingerprint,
            )
            children = vector.get_children(tuple(row.child_id for row in rows))
            if len(rows) != len(children) or any(
                child.metadata.subject != subject
                or child.metadata.generation_id != metadata.artifact_identity
                or child.metadata.policy_id != metadata.subject_policy_map[subject]
                for child in children
            ):
                raise PrimaryIndexError("primary BM25 and Chroma identities differ")
    finally:
        if parents is not None:
            parents.close()
        if vector is not None:
            vector.close()
    now = validated_at_utc or datetime.now(UTC)
    return PrimaryIndexValidationV1(
        schema_version="primary_index_validation_v1",
        primary_revision=metadata.primary_revision,
        artifact_identity=metadata.artifact_identity,
        validated_at_utc=now,
        validation_status="valid",
        validated_subjects=metadata.available_subjects,
    )


def load_primary_state(index_root: Path) -> PrimaryIndexStateV1:
    return read_strict_model(
        index_root, PRIMARY_STATE_RELATIVE_PATH, PrimaryIndexStateV1
    )


def load_primary_metadata(
    index_root: Path, *, state: PrimaryIndexStateV1
) -> PrimaryIndexMetadataV1:
    metadata = read_strict_model(
        index_root, state.metadata_relative_path, PrimaryIndexMetadataV1
    )
    if (
        metadata.primary_revision != state.primary_revision
        or metadata.config_fingerprint != state.config_fingerprint
    ):
        raise PrimaryIndexError("primary state and metadata differ")
    return metadata


def load_primary_validation(
    index_root: Path,
    *,
    state: PrimaryIndexStateV1,
    metadata: PrimaryIndexMetadataV1,
) -> PrimaryIndexValidationV1:
    result = read_strict_model(
        index_root, state.validation_relative_path, PrimaryIndexValidationV1
    )
    if (
        result.primary_revision != state.primary_revision
        or result.artifact_identity != metadata.artifact_identity
        or result.validated_subjects != metadata.available_subjects
        or result.validation_status != "valid"
    ):
        raise PrimaryIndexError("primary validation result differs")
    return result


PrimaryStagingValidator = Callable[
    [Path, PrimaryIndexMetadataV1], PrimaryIndexValidationV1
]


class PrimaryIndexWorkspace:
    def __init__(self, *, index_root: Path, build_id: str) -> None:
        self.index_root = index_root
        self.build_id = validate_generation_id(build_id)
        self.staging_relative_path = primary_staging_relative_path(self.build_id)

    @classmethod
    def create(cls, *, index_root: Path, build_id: str) -> "PrimaryIndexWorkspace":
        root = index_root.resolve(strict=False)
        if root.is_symlink():
            raise PrimaryIndexError("primary index root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        workspace = cls(index_root=root, build_id=build_id)
        path = resolve_under_root(
            root, workspace.staging_relative_path, must_exist=False
        )
        if path.exists():
            raise FileExistsError("primary staging path already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
        return workspace

    @property
    def staging_path(self) -> Path:
        return resolve_under_root(
            self.index_root, self.staging_relative_path, must_exist=True
        )

    def next_revision(self) -> int:
        try:
            return load_primary_state(self.index_root).primary_revision + 1
        except FileNotFoundError:
            return 1

    def publish(
        self,
        *,
        metadata: PrimaryIndexMetadataV1,
        validate_staging: PrimaryStagingValidator,
        now: datetime | None = None,
    ) -> PrimaryPublishResult:
        if not callable(validate_staging):
            raise TypeError("validate_staging must be callable")
        if metadata.primary_revision != self.next_revision():
            raise PrimaryIndexError("primary revision is not next")
        staging = self.staging_path
        active_relative = revision_path(metadata.primary_revision)
        active = resolve_under_root(self.index_root, active_relative, must_exist=False)
        if active.exists():
            raise FileExistsError("primary revision already exists")
        atomic_write_bytes(
            staging,
            PRIMARY_METADATA_FILENAME,
            model_json_bytes(metadata),
            overwrite=False,
        )
        validation = validate_staging(staging, metadata)
        if (
            not isinstance(validation, PrimaryIndexValidationV1)
            or validation.primary_revision != metadata.primary_revision
            or validation.artifact_identity != metadata.artifact_identity
            or validation.validated_subjects != metadata.available_subjects
        ):
            raise PrimaryIndexError("primary staging validation is invalid")
        atomic_write_bytes(
            staging,
            PRIMARY_VALIDATION_FILENAME,
            model_json_bytes(validation),
            overwrite=False,
        )
        active.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, active)
        state = PrimaryIndexStateV1(
            schema_version="primary_index_state_v1",
            primary_revision=metadata.primary_revision,
            active_directory_relative_path=active_relative,
            metadata_relative_path=primary_metadata_relative_path(
                metadata.primary_revision
            ),
            validation_relative_path=primary_validation_relative_path(
                metadata.primary_revision
            ),
            updated_at_utc=now or datetime.now(UTC),
            config_fingerprint=metadata.config_fingerprint,
            validation_status="valid",
        )
        atomic_write_bytes(
            self.index_root,
            PRIMARY_STATE_RELATIVE_PATH,
            model_json_bytes(state),
            overwrite=True,
        )
        return PrimaryPublishResult(
            schema_version="primary_publish_result_v1",
            primary_revision=state.primary_revision,
            active_directory_relative_path=state.active_directory_relative_path,
            config_fingerprint=state.config_fingerprint,
        )


def primary_artifact_root(index_root: Path, *, state: PrimaryIndexStateV1) -> Path:
    return _path(index_root, state.active_directory_relative_path, directory=True)


def primary_revision_from_state_or_none(index_root: Path) -> int | None:
    try:
        return load_primary_state(index_root).primary_revision
    except FileNotFoundError:
        return None


__all__ = [
    "PRIMARY_METADATA_FILENAME",
    "PRIMARY_ROOT",
    "PRIMARY_STATE_RELATIVE_PATH",
    "PRIMARY_VALIDATION_FILENAME",
    "PrimaryIndexError",
    "PrimaryIndexMetadataV1",
    "PrimaryIndexStateV1",
    "PrimaryIndexValidationV1",
    "PrimaryIndexWorkspace",
    "PrimaryPublishResult",
    "PrimaryStagingValidator",
    "compute_primary_config_fingerprint",
    "load_primary_metadata",
    "load_primary_state",
    "load_primary_validation",
    "primary_artifact_root",
    "primary_metadata_from_config",
    "primary_metadata_relative_path",
    "primary_revision_from_state_or_none",
    "primary_revision_relative_path",
    "primary_staging_relative_path",
    "primary_validation_relative_path",
    "validate_primary_revision",
]
