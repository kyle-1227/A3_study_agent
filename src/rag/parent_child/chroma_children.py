"""Immutable child-vector artifact writer for staging generations."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Iterator, Literal, Protocol, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from pydantic import BaseModel, ConfigDict, Field

from src.rag.parent_child._storage_io import validate_generation_id
from src.rag.parent_child.bm25_artifact import digest_identifier_set
from src.rag.parent_child.exceptions import ParentChildError
from src.rag.parent_child.models import ChildDocument


DistanceMetric = Literal["cosine", "l2", "ip"]
_COLLECTION_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{1,61}[A-Za-z0-9])$"
)
_ALLOWED_SCALAR_TYPES = {str, int, float, bool}


class ChildChromaArtifactError(ParentChildError):
    """Base class for strict child Chroma artifact failures."""


class ChromaStagingPathError(ChildChromaArtifactError):
    """Raised when a requested artifact is outside its generation staging root."""


class ChromaInputContractError(ChildChromaArtifactError):
    """Raised when child records or writer parameters violate their contract."""


class ChromaEmbeddingContractError(ChildChromaArtifactError):
    """Raised when the injected embedding provider returns an invalid batch."""


class ChromaWriteError(ChildChromaArtifactError):
    """Raised when Chroma cannot create or persist the immutable collection."""


class ChromaVerificationError(ChildChromaArtifactError):
    """Raised when persisted Chroma data differs from the authoritative children."""


class DocumentEmbeddingProvider(Protocol):
    """Provider-neutral document embedding boundary used by the artifact writer."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one ordered batch without changing its cardinality."""


class ChromaChildrenArtifact(BaseModel):
    """Verified identity and counts for one persisted child collection."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["chroma_children_artifact_v1"]
    generation_id: str = Field(min_length=1)
    artifact_relative_path: str = Field(min_length=1)
    collection_name: str = Field(min_length=3, max_length=63)
    distance_metric: DistanceMetric
    embedding_dimension: int = Field(gt=0)
    child_count: int = Field(gt=0)
    child_id_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ExpectedChild:
    """Already validated values used for deterministic write and verification."""

    def __init__(self, child: ChildDocument) -> None:
        self.child_id = child.metadata.child_id
        self.document = child.content
        self.metadata = child.metadata.to_chroma_metadata()


def _validate_collection_name(collection_name: str) -> None:
    if _COLLECTION_NAME_PATTERN.fullmatch(collection_name) is None:
        raise ChromaInputContractError(
            "collection_name must be 3-63 characters, start and end with an "
            "alphanumeric character, and contain only alphanumerics, '.', '_', or '-'"
        )
    if ".." in collection_name:
        raise ChromaInputContractError("collection_name must not contain '..'")


def _validate_requested_generation_id(generation_id: str) -> str:
    try:
        return validate_generation_id(generation_id)
    except ValueError:
        raise ChromaInputContractError("generation_id is invalid") from None


def _validate_staging_paths(
    *,
    generation_staging_root: Path,
    persist_directory: Path,
    generation_id: str,
) -> tuple[Path, Path, str]:
    validated_generation_id = _validate_requested_generation_id(generation_id)
    if not generation_staging_root.is_absolute() or not persist_directory.is_absolute():
        raise ChromaStagingPathError(
            "generation_staging_root and persist_directory must be absolute paths"
        )
    if (
        generation_staging_root.is_symlink()
        or generation_staging_root.parent.is_symlink()
    ):
        raise ChromaStagingPathError("generation staging paths must not be symlinks")
    try:
        staging_root = generation_staging_root.resolve(strict=True)
    except OSError:
        raise ChromaStagingPathError(
            "generation_staging_root must be an existing directory"
        ) from None
    if not staging_root.is_dir():
        raise ChromaStagingPathError(
            "generation_staging_root must be an existing directory"
        )
    if (
        staging_root.name != validated_generation_id
        or staging_root.parent.name != ".staging"
    ):
        raise ChromaStagingPathError(
            "generation_staging_root must be .staging/<generation_id>"
        )

    persist_path = persist_directory.resolve(strict=False)
    if persist_path.parent != staging_root:
        raise ChromaStagingPathError(
            "persist_directory must be a direct child of generation_staging_root"
        )
    if persist_directory.is_symlink() or persist_path.exists():
        raise ChromaStagingPathError(
            "persist_directory must not already exist; Chroma artifacts are immutable"
        )
    return staging_root, persist_path, persist_path.name


def _validate_writer_inputs(
    *,
    collection_name: str,
    distance_metric: str,
    expected_dimension: int,
    batch_size: int,
) -> None:
    _validate_collection_name(collection_name)
    if distance_metric not in {"cosine", "l2", "ip"}:
        raise ChromaInputContractError("unsupported Chroma distance metric")
    if type(expected_dimension) is not int or expected_dimension <= 0:
        raise ChromaInputContractError("expected_dimension must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ChromaInputContractError("batch_size must be a positive integer")


def _prepare_children(
    children: Sequence[ChildDocument],
    *,
    expected_generation_id: str,
) -> tuple[_ExpectedChild, ...]:
    if not children:
        raise ChromaInputContractError("at least one child is required")
    prepared: list[_ExpectedChild] = []
    identifiers: list[str] = []
    for child in children:
        if not isinstance(child, ChildDocument):
            raise ChromaInputContractError(
                "children must contain validated ChildDocument instances"
            )
        if child.metadata.generation_id != expected_generation_id:
            raise ChromaInputContractError(
                "child generation_id differs from the requested generation"
            )
        expected = _ExpectedChild(child)
        if expected.metadata.get("child_id") != expected.child_id:
            raise ChromaInputContractError(
                "Chroma document ID must equal metadata.child_id"
            )
        for key, value in expected.metadata.items():
            if type(value) not in _ALLOWED_SCALAR_TYPES:
                raise ChromaInputContractError(
                    f"Chroma metadata must be scalar-only: field={key}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ChromaInputContractError(
                    f"Chroma metadata floats must be finite: field={key}"
                )
        identifiers.append(expected.child_id)
        prepared.append(expected)
    if len(set(identifiers)) != len(identifiers):
        raise ChromaInputContractError("duplicate child IDs are forbidden")
    return tuple(sorted(prepared, key=lambda item: item.child_id))


def _embed_batch(
    embedding_provider: DocumentEmbeddingProvider,
    batch: Sequence[_ExpectedChild],
    *,
    expected_dimension: int,
    batch_start: int,
) -> list[list[float]]:
    try:
        vectors = embedding_provider.embed_documents(
            [expected.document for expected in batch]
        )
    except Exception:
        raise ChromaEmbeddingContractError(
            "document embedding failed for "
            f"batch_start={batch_start}, batch_size={len(batch)}"
        ) from None
    if not isinstance(vectors, list) or len(vectors) != len(batch):
        raise ChromaEmbeddingContractError(
            "embedding result cardinality must equal input cardinality"
        )

    validated: list[list[float]] = []
    for row_index, vector in enumerate(vectors):
        if not isinstance(vector, list) or len(vector) != expected_dimension:
            raise ChromaEmbeddingContractError(
                f"embedding vector dimension mismatch at batch_offset={row_index}"
            )
        row: list[float] = []
        for coordinate in vector:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ChromaEmbeddingContractError(
                    "embedding coordinates must be finite numbers"
                )
            normalized = float(coordinate)
            if not math.isfinite(normalized):
                raise ChromaEmbeddingContractError(
                    "embedding coordinates must be finite numbers"
                )
            row.append(normalized)
        validated.append(row)
    return validated


def _metadata_matches(
    actual: dict[str, object],
    expected: dict[str, str | int | float | bool],
) -> bool:
    if actual.keys() != expected.keys():
        return False
    return all(
        type(actual[key]) is type(expected[key]) and actual[key] == expected[key]
        for key in expected
    )


def _verify_collection(
    collection: Collection,
    expected_children: Sequence[_ExpectedChild],
    *,
    collection_metadata: dict[str, str | int],
    expected_dimension: int,
    batch_size: int,
) -> None:
    if collection.count() != len(expected_children):
        raise ChromaVerificationError("persisted child count mismatch")
    if collection.metadata != collection_metadata:
        raise ChromaVerificationError("persisted collection metadata mismatch")
    configuration = collection.configuration
    if not isinstance(configuration, dict):
        raise ChromaVerificationError("Chroma collection configuration is invalid")
    hnsw = configuration.get("hnsw")
    if (
        not isinstance(hnsw, dict)
        or hnsw.get("space") != collection_metadata["hnsw:space"]
    ):
        raise ChromaVerificationError("persisted collection distance metric mismatch")

    actual_ids: list[str] = []
    for start in range(0, len(expected_children), batch_size):
        batch = expected_children[start : start + batch_size]
        expected_by_id = {item.child_id: item for item in batch}
        try:
            result = collection.get(
                ids=list(expected_by_id),
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            raise ChromaVerificationError(
                f"failed to read persisted child batch at batch_start={start}"
            ) from None
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        embeddings = result.get("embeddings")
        if (
            not isinstance(ids, list)
            or documents is None
            or metadatas is None
            or embeddings is None
            or len(ids) != len(batch)
            or len(documents) != len(batch)
            or len(metadatas) != len(batch)
            or len(embeddings) != len(batch)
        ):
            raise ChromaVerificationError(
                f"persisted child batch cardinality mismatch at batch_start={start}"
            )
        for child_id, document, metadata, vector in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            expected = expected_by_id.get(child_id)
            if expected is None:
                raise ChromaVerificationError(
                    "persisted collection contains unknown ID"
                )
            if document != expected.document:
                raise ChromaVerificationError(
                    f"persisted child document mismatch: child_id={child_id}"
                )
            if not isinstance(metadata, dict) or not _metadata_matches(
                metadata, expected.metadata
            ):
                raise ChromaVerificationError(
                    f"persisted child metadata mismatch: child_id={child_id}"
                )
            if len(vector) != expected_dimension or any(
                not math.isfinite(float(coordinate)) for coordinate in vector
            ):
                raise ChromaVerificationError(
                    f"persisted embedding contract mismatch: child_id={child_id}"
                )
            actual_ids.append(child_id)

    expected_ids = [child.child_id for child in expected_children]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ChromaVerificationError("persisted child ID set mismatch")


def iter_child_chroma_metadata_pages(
    collection: Collection,
    *,
    expected_count: int,
    page_size: int,
) -> Iterator[tuple[tuple[str, ...], tuple[dict[str, object], ...]]]:
    """Yield every child ID/metadata row through an explicit bounded read.

    The sealed child collection can be larger than SQLite's SQL-variable limit,
    so callers must not use one unbounded ``Collection.get`` for full-artifact
    validation.  A page must be complete, disjoint, and contain scalar metadata
    mappings; callers remain responsible for their domain-level ID digest and
    ``ChildMetadata`` validation.
    """

    if expected_count <= 0:
        raise ChromaVerificationError("child Chroma expected count must be positive")
    if page_size <= 0:
        raise ChromaVerificationError(
            "child Chroma metadata page size must be positive"
        )

    seen_ids: set[str] = set()
    for offset in range(0, expected_count, page_size):
        limit = min(page_size, expected_count - offset)
        try:
            result = collection.get(
                limit=limit,
                offset=offset,
                include=["metadatas"],
            )
        except Exception as exc:
            raise ChromaVerificationError("failed to read child metadata page") from exc
        identifiers = result.get("ids")
        raw_metadatas = result.get("metadatas")
        if not (
            isinstance(identifiers, list)
            and isinstance(raw_metadatas, list)
            and len(identifiers) == len(raw_metadatas) == limit
        ):
            raise ChromaVerificationError("child metadata page cardinality mismatch")
        page_ids: list[str] = []
        page_metadatas: list[dict[str, object]] = []
        for identifier, raw_metadata in zip(identifiers, raw_metadatas, strict=True):
            if not isinstance(identifier, str) or not identifier:
                raise ChromaVerificationError(
                    "child metadata page identifier is invalid"
                )
            if identifier in seen_ids:
                raise ChromaVerificationError(
                    "child metadata page contains duplicate ID"
                )
            if not isinstance(raw_metadata, dict):
                raise ChromaVerificationError("child metadata page entry is invalid")
            page_ids.append(identifier)
            page_metadatas.append(raw_metadata)
        seen_ids.update(page_ids)
        yield tuple(page_ids), tuple(page_metadatas)

    if len(seen_ids) != expected_count:
        raise ChromaVerificationError("child metadata pages did not cover expected IDs")


def write_child_chroma_artifact(
    children: Sequence[ChildDocument],
    *,
    generation_staging_root: Path,
    persist_directory: Path,
    generation_id: str,
    collection_name: str,
    distance_metric: DistanceMetric,
    expected_dimension: int,
    batch_size: int,
    embedding_provider: DocumentEmbeddingProvider,
) -> ChromaChildrenArtifact:
    """Create and fully verify one immutable child collection in staging."""

    validated_generation_id = _validate_requested_generation_id(generation_id)
    _validate_writer_inputs(
        collection_name=collection_name,
        distance_metric=distance_metric,
        expected_dimension=expected_dimension,
        batch_size=batch_size,
    )
    staging_root, persist_path, relative_path = _validate_staging_paths(
        generation_staging_root=generation_staging_root,
        persist_directory=persist_directory,
        generation_id=validated_generation_id,
    )
    prepared = _prepare_children(
        children,
        expected_generation_id=validated_generation_id,
    )
    collection_metadata: dict[str, str | int] = {
        "schema_version": "chroma_children_v1",
        "generation_id": validated_generation_id,
        "expected_dimension": expected_dimension,
        "hnsw:space": distance_metric,
    }

    settings = Settings(anonymized_telemetry=False)
    try:
        with chromadb.PersistentClient(
            path=str(persist_path), settings=settings
        ) as client:
            if client.list_collections():
                raise ChromaVerificationError(
                    "new Chroma artifact unexpectedly contains collections"
                )
            collection = client.create_collection(
                name=collection_name,
                metadata=collection_metadata,
                embedding_function=None,
                get_or_create=False,
            )
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                vectors = _embed_batch(
                    embedding_provider,
                    batch,
                    expected_dimension=expected_dimension,
                    batch_start=start,
                )
                collection.add(
                    ids=[item.child_id for item in batch],
                    documents=[item.document for item in batch],
                    metadatas=[item.metadata for item in batch],
                    embeddings=vectors,
                )
            collection_names = {item.name for item in client.list_collections()}
            if collection_names != {collection_name}:
                raise ChromaVerificationError(
                    "immutable Chroma artifact must contain exactly one collection"
                )
            _verify_collection(
                collection,
                prepared,
                collection_metadata=collection_metadata,
                expected_dimension=expected_dimension,
                batch_size=batch_size,
            )
    except ChildChromaArtifactError:
        raise
    except Exception:
        raise ChromaWriteError("failed to write child Chroma artifact") from None

    if not persist_path.is_dir() or persist_path.is_symlink():
        raise ChromaVerificationError("persisted Chroma artifact directory is invalid")
    if persist_path.parent != staging_root:
        raise ChromaVerificationError("persisted Chroma artifact escaped staging")
    child_ids = [item.child_id for item in prepared]
    return ChromaChildrenArtifact(
        schema_version="chroma_children_artifact_v1",
        generation_id=validated_generation_id,
        artifact_relative_path=relative_path,
        collection_name=collection_name,
        distance_metric=distance_metric,
        embedding_dimension=expected_dimension,
        child_count=len(prepared),
        child_id_set_sha256=digest_identifier_set(child_ids),
    )
