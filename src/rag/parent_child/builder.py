"""End-to-end immutable generation builder with strict cross-artifact sealing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.rag_index_config import RagIndexConfig
from src.rag.parent_child._storage_io import (
    canonical_json_bytes,
    model_json_bytes,
    sha256_bytes,
    sha256_path,
)
from src.rag.parent_child.bm25_artifact import (
    Bm25CorpusRow,
    compute_tokenizer_fingerprint,
    write_subject_bm25_artifact,
)
from src.rag.parent_child.chroma_children import (
    DocumentEmbeddingProvider,
    write_child_chroma_artifact,
)
from src.rag.parent_child.config_adapter import resolve_subject_chunk_policy
from src.rag.parent_child.generation import GenerationWorkspace, SealedGeneration
from src.rag.parent_child.loader import load_cleaned_source
from src.rag.parent_child.manifests import (
    ArtifactDescriptor,
    ArtifactType,
    Bm25ManifestIdentity,
    EmbeddingManifestIdentity,
    GenerationManifest,
    PolicyManifest,
    PolicyManifestSet,
    SubjectManifest,
    SubjectManifestEntry,
    assert_generation_complete,
    write_strict_model,
)
from src.rag.parent_child.models import (
    ChildDocument,
    ParentRecord,
    SourceEntry,
)
from src.rag.parent_child.parent_store import create_parent_store
from src.rag.parent_child.registry import GenerationRegistry
from src.rag.parent_child.splitter import build_parent_child_bundle
from src.rag.subject_catalog import SubjectCatalog, SubjectCatalogEntry


class GenerationBuildError(RuntimeError):
    """An immutable generation failed before reaching READY."""


class Bm25Tokenizer(Protocol):
    def __call__(self, text: str) -> Sequence[str]: ...


class GenerationBuildRequest(BaseModel):
    """Explicit non-secret identity for one requested generation build."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["generation_build_request_v1"]
    generation_id: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)

    @field_validator("generation_id", "code_revision")
    @classmethod
    def _stripped(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("build identity fields must already be stripped")
        return value


class GenerationStageTimings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    catalog_and_chunk_ms: float = Field(ge=0.0)
    parent_store_ms: float = Field(ge=0.0)
    bm25_ms: float = Field(ge=0.0)
    chroma_ms: float = Field(ge=0.0)
    validation_and_seal_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)

    @field_validator("*")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("generation stage timings must be finite")
        return value


class GenerationBuildReport(BaseModel):
    """Content-free, non-secret build diagnostics sealed with the generation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["generation_build_report_v1"]
    generation_id: str
    code_revision: str
    source_count: int = Field(ge=0)
    subject_count: int = Field(ge=0)
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    policy_count: int = Field(ge=0)
    timings: GenerationStageTimings


class GenerationBuildResult(BaseModel):
    """READY, sealed result; activation is deliberately a separate operation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["generation_build_result_v1"]
    generation_id: str
    registry_state: Literal["READY"]
    activated: Literal[False]
    sealed: SealedGeneration
    manifest: GenerationManifest


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _artifact_size(path: Path) -> int:
    if path.is_symlink():
        raise GenerationBuildError("generation artifacts must not be symlinks")
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise GenerationBuildError("generation artifact path does not exist")
    size = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise GenerationBuildError("generation artifacts must not contain symlinks")
        if child.is_file():
            size += child.stat().st_size
    return size


def _descriptor(
    root: Path,
    *,
    artifact_type: ArtifactType,
    relative_path: str,
    schema_version: str,
) -> ArtifactDescriptor:
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise GenerationBuildError("artifact path escapes generation staging root")
    return ArtifactDescriptor(
        artifact_type=artifact_type,
        relative_path=relative_path,
        sha256=sha256_path(path),
        schema_version=schema_version,
        size_bytes=_artifact_size(path),
    )


def _strict_tokens(tokenizer: Bm25Tokenizer, text: str) -> tuple[str, ...]:
    try:
        raw = tokenizer(text)
    except Exception as exc:
        raise GenerationBuildError("BM25 document tokenization failed") from exc
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise GenerationBuildError("BM25 tokenizer must return a token sequence")
    tokens = tuple(raw)
    if not tokens or any(
        not isinstance(token, str) or not token.strip() for token in tokens
    ):
        raise GenerationBuildError(
            "every child must produce non-empty, nonblank BM25 tokens"
        )
    return tokens


def _source_manifest_digest(
    sources: Sequence[tuple[str, str]],
) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(sources)))


def compute_embedding_fingerprint(config: RagIndexConfig) -> str:
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


def _doc_type(extension: str) -> str:
    mapping = {".pdf": "pdf", ".md": "markdown", ".txt": "text"}
    try:
        return mapping[extension]
    except KeyError as exc:
        raise GenerationBuildError(
            "catalog returned unsupported source extension"
        ) from exc


class GenerationBuilder:
    """Build, validate, seal, and register READY without activating."""

    def __init__(
        self,
        *,
        config: RagIndexConfig,
        registry: GenerationRegistry,
        embedding_provider: DocumentEmbeddingProvider,
        bm25_tokenizer: Bm25Tokenizer,
        build_clock: Callable[[], datetime],
    ) -> None:
        self._config = config
        self._registry = registry
        self._embedding_provider = embedding_provider
        self._bm25_tokenizer = bm25_tokenizer
        self._build_clock = build_clock
        if not config.catalog.data_root.is_absolute():
            raise ValueError("catalog.data_root must be resolved absolute before build")
        if not config.storage.index_root.is_absolute():
            raise ValueError(
                "storage.index_root must be resolved absolute before build"
            )

    def _catalog_and_chunk(
        self,
        request: GenerationBuildRequest,
    ) -> tuple[
        tuple[SubjectCatalogEntry, ...],
        tuple[PolicyManifest, ...],
        tuple[ParentRecord, ...],
        tuple[ChildDocument, ...],
    ]:
        snapshot = SubjectCatalog(
            config=self._config.catalog,
            subject_policy_map=self._config.subject_policy_map,
        ).discover()
        policies: dict[str, PolicyManifest] = {}
        parents: list[ParentRecord] = []
        children: list[ChildDocument] = []
        for subject in snapshot.subjects:
            resolved_policy = resolve_subject_chunk_policy(
                self._config, subject.subject_id
            )
            policies[subject.policy_id] = resolved_policy.policy_manifest
            for source in subject.sources:
                cleaned = load_cleaned_source(
                    SourceEntry(
                        schema_version="source_entry_v1",
                        source_path=source.source_path,
                        data_root=snapshot.data_root,
                        subject=subject.subject_id,
                        doc_type=_doc_type(source.extension),
                    ),
                    resolved_policy.loader_config,
                )
                bundle = build_parent_child_bundle(
                    cleaned,
                    resolved_policy.parent_child_policy,
                    request.generation_id,
                )
                parents.extend(bundle.parents)
                children.extend(bundle.children)
        return (
            snapshot.subjects,
            tuple(policies[key] for key in sorted(policies)),
            tuple(parents),
            tuple(children),
        )

    def build(self, request: GenerationBuildRequest) -> GenerationBuildResult:
        total_start = perf_counter_ns()
        stage_code = "register_building"
        registered = False
        try:
            self._registry.register_building(request.generation_id)
            registered = True
            workspace = GenerationWorkspace.create(
                self._config.storage.index_root,
                request.generation_id,
                marker_schema_version=(
                    self._config.storage.owner_marker_schema_version
                ),
            )

            stage_code = "catalog_and_chunk"
            stage_start = perf_counter_ns()
            subjects, policies, parents, children = self._catalog_and_chunk(request)
            catalog_and_chunk_ms = _elapsed_ms(stage_start)
            if not parents or not children:
                raise GenerationBuildError("generation contains no parents or children")

            stage_code = "parent_store"
            stage_start = perf_counter_ns()
            create_parent_store(
                workspace.staging_path,
                "parents.sqlite",
                parents,
                store_schema_version=(self._config.storage.parent_store_schema_version),
                expected_generation_id=request.generation_id,
                busy_timeout_seconds=(
                    self._config.storage.parent_store_busy_timeout_seconds
                ),
            )
            parent_store_ms = _elapsed_ms(stage_start)

            stage_code = "bm25"
            stage_start = perf_counter_ns()
            bm25_ids: dict[str, tuple[str, ...]] = {}
            bm25_paths: list[tuple[str, str]] = []
            tokenizer_fingerprint = compute_tokenizer_fingerprint(
                tokenizer_name=self._config.bm25.tokenizer,
                tokenizer_version=self._config.bm25.tokenizer_version,
                dictionary_sha256=self._config.bm25.dictionary_hash,
            )
            children_by_subject: dict[str, list[ChildDocument]] = {
                subject.subject_id: [] for subject in subjects
            }
            for child in children:
                children_by_subject[child.metadata.subject].append(child)
            for subject_id in sorted(children_by_subject):
                subject_children = sorted(
                    children_by_subject[subject_id],
                    key=lambda item: item.metadata.child_id,
                )
                rows = tuple(
                    Bm25CorpusRow(
                        schema_version="bm25_row_v1",
                        generation_id=request.generation_id,
                        subject=subject_id,
                        child_id=child.metadata.child_id,
                        tokens=_strict_tokens(self._bm25_tokenizer, child.content),
                    )
                    for child in subject_children
                )
                corpus_path = f"bm25/{subject_id}.jsonl"
                manifest_path = f"bm25/{subject_id}.manifest.json"
                write_subject_bm25_artifact(
                    workspace.staging_path,
                    corpus_path,
                    manifest_path,
                    rows,
                    manifest_schema_version="bm25_manifest_v1",
                    expected_generation_id=request.generation_id,
                    expected_subject=subject_id,
                    tokenizer_name=self._config.bm25.tokenizer,
                    tokenizer_version=self._config.bm25.tokenizer_version,
                    dictionary_sha256=self._config.bm25.dictionary_hash,
                )
                bm25_ids[subject_id] = tuple(row.child_id for row in rows)
                bm25_paths.append((corpus_path, manifest_path))
            bm25_ms = _elapsed_ms(stage_start)

            stage_code = "chroma"
            stage_start = perf_counter_ns()
            chroma_artifact = write_child_chroma_artifact(
                children,
                generation_staging_root=workspace.staging_path,
                persist_directory=workspace.staging_path / "chroma_children",
                generation_id=request.generation_id,
                collection_name=self._config.storage.collection_name,
                distance_metric=self._config.embedding.distance_metric,
                expected_dimension=self._config.embedding.expected_dimension,
                batch_size=self._config.embedding.batch_size,
                embedding_provider=self._embedding_provider,
            )
            chroma_ms = _elapsed_ms(stage_start)

            stage_code = "manifests_and_validation"
            stage_start = perf_counter_ns()
            policy_set = PolicyManifestSet(
                schema_version="policy_manifest_set_v1",
                policies=policies,
            )
            write_strict_model(
                workspace.staging_path,
                "policy_manifest.json",
                policy_set,
                overwrite=False,
            )
            parent_counts: dict[str, int] = {
                subject.subject_id: 0 for subject in subjects
            }
            child_counts: dict[str, int] = {
                subject.subject_id: 0 for subject in subjects
            }
            source_hashes: dict[str, list[tuple[str, str]]] = {
                subject.subject_id: [] for subject in subjects
            }
            parent_by_doc = {parent.doc_id: parent for parent in parents}
            for parent in parents:
                parent_counts[parent.subject] += 1
            for child in children:
                child_counts[child.metadata.subject] += 1
            for subject in subjects:
                for source in subject.sources:
                    matching = next(
                        (
                            parent
                            for parent in parent_by_doc.values()
                            if parent.source_relpath == source.source_relpath
                        ),
                        None,
                    )
                    if matching is None:
                        raise GenerationBuildError(
                            "source has no parent record after chunking"
                        )
                    source_hashes[subject.subject_id].append(
                        (source.source_relpath, matching.source_file_sha1)
                    )
            subject_manifest = SubjectManifest(
                schema_version="subject_manifest_v1",
                generation_id=request.generation_id,
                entries=tuple(
                    SubjectManifestEntry(
                        subject_id=subject.subject_id,
                        directory_relpath=subject.directory_path.relative_to(
                            self._config.catalog.data_root
                        ).as_posix(),
                        source_file_count=len(subject.sources),
                        source_manifest_sha256=_source_manifest_digest(
                            source_hashes[subject.subject_id]
                        ),
                        policy_id=subject.policy_id,
                        parent_count=parent_counts[subject.subject_id],
                        child_count=child_counts[subject.subject_id],
                        exclusion_state="active",
                        exclusion_reason="",
                    )
                    for subject in subjects
                ),
            )
            write_strict_model(
                workspace.staging_path,
                "subject_manifest.json",
                subject_manifest,
                overwrite=False,
            )
            self._registry.transition(request.generation_id, "VALIDATING")
            report = assert_generation_complete(
                parents,
                children,
                bm25_ids,
                tuple(child.metadata.child_id for child in children),
                self._config.subject_policy_map,
                report_schema_version="generation_validation_v1",
                expected_generation_id=request.generation_id,
                source_count=sum(len(subject.sources) for subject in subjects),
            )
            pre_report_elapsed = _elapsed_ms(stage_start)
            timings = GenerationStageTimings(
                catalog_and_chunk_ms=catalog_and_chunk_ms,
                parent_store_ms=parent_store_ms,
                bm25_ms=bm25_ms,
                chroma_ms=chroma_ms,
                validation_and_seal_ms=pre_report_elapsed,
                total_ms=_elapsed_ms(total_start),
            )
            build_report = GenerationBuildReport(
                schema_version="generation_build_report_v1",
                generation_id=request.generation_id,
                code_revision=request.code_revision,
                source_count=report.counts.source_count,
                subject_count=report.counts.subject_count,
                parent_count=report.counts.parent_count,
                child_count=report.counts.child_count,
                policy_count=len(policies),
                timings=timings,
            )
            write_strict_model(
                workspace.staging_path,
                "build_report.json",
                build_report,
                overwrite=False,
            )
            descriptors: list[ArtifactDescriptor] = [
                _descriptor(
                    workspace.staging_path,
                    artifact_type="chroma_children",
                    relative_path=chroma_artifact.artifact_relative_path,
                    schema_version=chroma_artifact.schema_version,
                ),
                _descriptor(
                    workspace.staging_path,
                    artifact_type="parent_store",
                    relative_path="parents.sqlite",
                    schema_version=self._config.storage.parent_store_schema_version,
                ),
                _descriptor(
                    workspace.staging_path,
                    artifact_type="policy_manifest",
                    relative_path="policy_manifest.json",
                    schema_version=policy_set.schema_version,
                ),
                _descriptor(
                    workspace.staging_path,
                    artifact_type="subject_manifest",
                    relative_path="subject_manifest.json",
                    schema_version=subject_manifest.schema_version,
                ),
                _descriptor(
                    workspace.staging_path,
                    artifact_type="build_report",
                    relative_path="build_report.json",
                    schema_version=build_report.schema_version,
                ),
            ]
            for corpus_path, manifest_path in bm25_paths:
                descriptors.extend(
                    (
                        _descriptor(
                            workspace.staging_path,
                            artifact_type="bm25_corpus",
                            relative_path=corpus_path,
                            schema_version="bm25_row_v1",
                        ),
                        _descriptor(
                            workspace.staging_path,
                            artifact_type="bm25_manifest",
                            relative_path=manifest_path,
                            schema_version="bm25_manifest_v1",
                        ),
                    )
                )
            descriptors.sort(key=lambda item: item.relative_path)
            policy_sha256 = sha256_path(workspace.staging_path / "policy_manifest.json")
            subject_sha256 = sha256_path(
                workspace.staging_path / "subject_manifest.json"
            )
            build_time = self._build_clock()
            if build_time.tzinfo is None:
                raise GenerationBuildError(
                    "build_clock must return timezone-aware time"
                )
            manifest = GenerationManifest(
                schema_version="generation_manifest_v1",
                generation_id=request.generation_id,
                build_state="ready",
                code_revision=request.code_revision,
                build_time_utc=build_time,
                collection_name=self._config.storage.collection_name,
                artifacts=tuple(descriptors),
                embedding=EmbeddingManifestIdentity(
                    provider=self._config.embedding.provider,
                    model=self._config.embedding.model,
                    base_url_identity=(
                        self._config.embedding.base_url.rstrip("/")
                        + self._config.embedding.endpoint_path
                    ),
                    input_types=(
                        self._config.embedding.document_input_type,
                        self._config.embedding.query_input_type,
                    ),
                    fingerprint=compute_embedding_fingerprint(self._config),
                    dimension=self._config.embedding.expected_dimension,
                    distance_metric=self._config.embedding.distance_metric,
                ),
                bm25=Bm25ManifestIdentity(
                    tokenizer_name=self._config.bm25.tokenizer,
                    tokenizer_version=self._config.bm25.tokenizer_version,
                    dictionary_sha256=self._config.bm25.dictionary_hash,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    artifact_format="jsonl",
                ),
                subject_manifest_sha256=subject_sha256,
                policy_manifest_sha256=policy_sha256,
                subject_fingerprint=sha256_bytes(model_json_bytes(subject_manifest)),
                policy_fingerprint=sha256_bytes(model_json_bytes(policy_set)),
                source_fingerprint=_source_manifest_digest(
                    tuple(
                        source
                        for subject_sources in source_hashes.values()
                        for source in subject_sources
                    )
                ),
                parent_id_set_sha256=report.parent_id_set_sha256,
                child_id_set_sha256=report.child_id_set_sha256,
                counts=report.counts,
                integrity=report.integrity,
                validation_report_sha256=sha256_bytes(model_json_bytes(report)),
                validation_passed=True,
            )
            sealed = workspace.seal(manifest, report)
            self._registry.mark_ready(
                request.generation_id,
                manifest_sha256=sealed.manifest_sha256,
            )
            return GenerationBuildResult(
                schema_version="generation_build_result_v1",
                generation_id=request.generation_id,
                registry_state="READY",
                activated=False,
                sealed=sealed,
                manifest=manifest,
            )
        except Exception as exc:
            if registered:
                try:
                    self._registry.mark_failed(
                        request.generation_id,
                        failure_code=stage_code[:128],
                        failure_type=type(exc).__name__[:128],
                    )
                except Exception as registry_exc:
                    raise GenerationBuildError(
                        "generation failed and registry failure state could not be recorded"
                    ) from registry_exc
            raise


__all__ = [
    "Bm25Tokenizer",
    "GenerationBuildError",
    "GenerationBuildReport",
    "GenerationBuildRequest",
    "GenerationBuildResult",
    "GenerationBuilder",
    "GenerationStageTimings",
    "compute_embedding_fingerprint",
]
